import torch.nn as nn
import models.layers as nl
import pdb
import math

### mobilenetv2 not yet ###

__all__ = [
    'mobilenetv1',
    'mobilenetv2'
]
    
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

#############################################

class VGG(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, network_width_multiplier=1.0, shared_layer_info={}, init_weights=True, progressive_init=False):
        super(VGG, self).__init__()
        self.features = features
        self.network_width_multiplier = network_width_multiplier
        self.shared_layer_info = shared_layer_info
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

        if progressive_init:
            self._initialize_weights_2()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nl.SharableLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier'] * 1024), num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(int(1024*self.network_width_multiplier), num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nl.SharableConv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
        )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # dw
        nl.SharableConv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        # pw
        nl.SharableConv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        )
        
def make_layers_cifar100_v1(cfg, network_width_multiplier, batch_norm=False, groups=1):
    layers = []
    in_channels = 3
    layers += [conv_bn(in_channels, int(32 * network_width_multiplier), 2),
               conv_dw(int(32 * network_width_multiplier), int(64 * network_width_multiplier), 1),
               conv_dw(int(64 * network_width_multiplier), int(128 * network_width_multiplier), 2),
               conv_dw(int(128 * network_width_multiplier), int(128 * network_width_multiplier), 1),
               conv_dw(int(128 * network_width_multiplier), int(256 * network_width_multiplier), 2),
               conv_dw(int(256 * network_width_multiplier), int(256 * network_width_multiplier), 1),
               conv_dw(int(256 * network_width_multiplier), int(512 * network_width_multiplier), 2),
               conv_dw(int(512 * network_width_multiplier), int(512 * network_width_multiplier), 1),
               conv_dw(int(512 * network_width_multiplier), int(512 * network_width_multiplier), 1),
               conv_dw(int(512 * network_width_multiplier), int(512 * network_width_multiplier), 1),
               conv_dw(int(512 * network_width_multiplier), int(512 * network_width_multiplier), 1),
               conv_dw(int(512 * network_width_multiplier), int(512 * network_width_multiplier), 1),
               conv_dw(int(512 * network_width_multiplier), int(1024 * network_width_multiplier), 2),
               conv_dw(int(1024 * network_width_multiplier), int(1024 * network_width_multiplier), 1),
               nn.AdaptiveAvgPool2d(1)]

    layers += [
        View(-1, int(1024 * network_width_multiplier))
    ]

    return Sequential_Debug(*layers)

def mobilenetv1(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return VGG(make_layers_cifar100_v1(custom_cfg, network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)
     
#############################################
def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nl.SharableConv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nl.SharableConv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nl.SharableConv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class VGG2(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, network_width_multiplier=1.0, shared_layer_info={}, init_weights=True, progressive_init=False):
        super(VGG2, self).__init__()
        self.features = features
        self.network_width_multiplier = network_width_multiplier
        self.shared_layer_info = shared_layer_info
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

        if progressive_init:
            self._initialize_weights_2()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nl.SharableLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier'] * 1280), num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(int(1280*self.network_width_multiplier), num_classes)) ## updated ##
            self.classifiers.append(nn.AdaptiveAvgPool2d(1)) ## updated ##
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        
def make_layers_cifar100_v2(cfg, network_width_multiplier, batch_norm=False, groups=1):
    configs=[
        # t, c, n, s
        [1, int(16*network_width_multiplier), 1, 1],
        [6, int(24*network_width_multiplier), 2, 2],
        [6, int(32*network_width_multiplier), 3, 2],
        [6, int(64*network_width_multiplier), 4, 2],
        [6, int(96*network_width_multiplier), 3, 1],
        [6, int(160*network_width_multiplier), 3, 2],
        [6, int(320*network_width_multiplier), 1, 1]
    ]
            
    layers = []
    in_channels = 3
    layers += [conv3x3(in_channels, int(32*network_width_multiplier), stride=2)]
    in_channels = int(32*network_width_multiplier)
    for t, c, n, s in configs:
        for i in range(n):
            stride = s if i == 0 else 1
            layers += [InvertedBlock(ch_in=in_channels, ch_out=c, expand_ratio=t, stride=stride)]
            in_channels = c
    layers += [conv1x1(in_channels, int(1280*network_width_multiplier)),
               nn.Dropout2d(0.2)]

    layers += [
        View(-1, int(1280*network_width_multiplier))
    ]

    return Sequential_Debug(*layers)
                
def mobilenetv2(custom_cfg=[], dataset_history=[], dataset2num_classes={}, network_width_multiplier=1.0, groups=1, shared_layer_info={}, **kwargs):
    return VGG2(make_layers_cifar100_v2(custom_cfg, network_width_multiplier, batch_norm=True, groups=groups), dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, **kwargs)
