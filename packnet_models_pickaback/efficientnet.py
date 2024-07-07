
import torch
import torch.nn as nn
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
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
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
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
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
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
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
    def __init__(self, features, dataset_history, dataset2num_classes, init_weights=True):
        super(EfficientNet, self).__init__()
        self.features = features
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
            self.classifiers.append(nn.Linear(1280, num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes

            self.classifiers.append(nn.Dropout(0.2))
            self.classifiers.append(nn.Linear(1280, num_classes))

            print(self.classifiers)
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)*2-1].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)*2-1].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

def make_layers_cifar100(cfg, batch_norm=False, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):

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
    out_channels = _round_filters(32, width_mult)
    features = [ConvBNReLU(3, out_channels, 3, stride=2)]

    in_channels = out_channels
    for t, c, n, s, k in settings:
        out_channels = _round_filters(c, width_mult)
        repeats = _round_repeats(n, depth_mult)
        for i in range(repeats):
            stride = s if i == 0 else 1
            features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
            in_channels = out_channels

    last_channels = _round_filters(1280, width_mult)
    print(last_channels)
    features += [ConvBNReLU(in_channels, last_channels, 1)]

    return Sequential_Debug(*features)


def efficientnetb0(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.0, depth_mult=1.0, dropout_rate=0.2), dataset_history, dataset2num_classes, **kwargs)

def efficientnetb1(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.0, depth_mult=1.1, dropout_rate=0.2), dataset_history, dataset2num_classes, **kwargs)

def efficientnetb2(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.1, depth_mult=1.2, dropout_rate=0.3), dataset_history, dataset2num_classes, **kwargs)

def efficientnetb3(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.2, depth_mult=1.4, dropout_rate=0.3), dataset_history, dataset2num_classes, **kwargs)

def efficientnetb4(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.4, depth_mult=1.8, dropout_rate=0.4), dataset_history, dataset2num_classes, **kwargs)

def efficientnetb5(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.6, depth_mult=2.2, dropout_rate=0.4), dataset_history, dataset2num_classes, **kwargs)

def efficientnetb6(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=1.8, depth_mult=2.6, dropout_rate=0.5), dataset_history, dataset2num_classes, **kwargs)

def efficientnetb7(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return EfficientNet(make_layers_cifar100([], width_mult=2.0, depth_mult=3.1, dropout_rate=0.5), dataset_history, dataset2num_classes, **kwargs)

