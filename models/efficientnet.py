import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import models.layers as nl
import pdb
import math

__all__ = [
    'efficientnetb0',
    'efficientnetb1',
    'efficientnetb2',
    'efficientnetb3',
    'efficientnetb4',
    'efficientnetb5',
    'efficientnetb6',
    'efficientnetb7',
]

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nl.SharableConv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nl.SharableConv2d(in_planes, reduced_dim, 1),
            Swish(),
            nl.SharableConv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nl.SharableConv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))

###########################


class Sequential_Debug(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class EfficientNet(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, network_width_multiplier=1.0, shared_layer_info={}, init_weights=True, progressive_init=False, seed=1):
        super(EfficientNet, self).__init__()
        self.features = features
        self.network_width_multiplier = network_width_multiplier
        self.shared_layer_info = shared_layer_info
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights(seed=seed)

        if progressive_init:
            self._initialize_weights_2()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Dropout(0.2))
            self.classifiers.append(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier'] * 1280), num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes

            self.classifiers.append(nn.Dropout(0.2))
            self.classifiers.append(nn.Linear(int(1280*self.network_width_multiplier), num_classes))

            print(self.classifiers)
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)*2-1].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)*2-1].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)*2-1]

def make_layers_cifar100(cfg, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, network_width_multiplier=1.0, batch_norm=False, groups=1):

    settings = [
        # t,  c, n, s, k
        [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
        [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
        [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
        [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
        [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
        [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
        [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
    ]
    out_channels = _round_filters(int(32*network_width_multiplier), width_mult)
    features = [ConvBNReLU(3, out_channels, 3, stride=2)]

    in_channels = out_channels
    for t, c, n, s, k in settings:
        out_channels = _round_filters(c, width_mult)
        repeats = _round_repeats(n, depth_mult)
        for i in range(repeats):
            stride = s if i == 0 else 1
            features += [MBConvBlock(in_channels, int(out_channels*network_width_multiplier), expand_ratio=t, stride=stride, kernel_size=k)]
            in_channels = int(out_channels*network_width_multiplier)

    last_channels = _round_filters(int(1280*network_width_multiplier), width_mult)
    print(last_channels)
    features += [ConvBNReLU(in_channels, last_channels, 1)]

    return Sequential_Debug(*features)


def efficientnetb0(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, seed=1, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs, seed=seed)

def efficientnetb1(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.0, depth_mult=1.1, dropout_rate=0.2, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

def efficientnetb2(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.1, depth_mult=1.2, dropout_rate=0.3, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

def efficientnetb3(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.2, depth_mult=1.4, dropout_rate=0.3, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

def efficientnetb4(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.4, depth_mult=1.8, dropout_rate=0.4, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

def efficientnetb5(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.6, depth_mult=2.2, dropout_rate=0.4, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

def efficientnetb6(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.8, depth_mult=2.6, dropout_rate=0.5, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

def efficientnetb7(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=2.0, depth_mult=3.1, dropout_rate=0.5, network_width_multiplier=network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)

