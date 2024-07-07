import torch.nn as nn
import pdb
import math
import torch.nn.functional as F

__all__ = [
    'mobilenetv1',
    'mobilenetv2',
    'mobilenetv2_1',
    'mobilenetv2_2',
    'mobilenetv2_3',
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

class VGG(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(1024, num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(1024, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

# MobileNet v1 ############################################
        
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
        )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        # pw
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        )
        
def make_layers_cifar100_v1(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    layers += [conv_bn(in_channels, 32, 2),
               conv_dw(32, 64, 1),
               conv_dw(64, 128, 2),
               conv_dw(128, 128, 1),
               conv_dw(128, 256, 2),
               conv_dw(256, 256, 1),
               conv_dw(256, 512, 2),
               conv_dw(512, 512, 1),
               conv_dw(512, 512, 1),
               conv_dw(512, 512, 1),
               conv_dw(512, 512, 1),
               conv_dw(512, 512, 1),
               conv_dw(512, 1024, 2),
               conv_dw(1024, 1024, 1),
               nn.AdaptiveAvgPool2d(1)]

    layers += [
        View(-1, 1024)
    ]

    return Sequential_Debug(*layers)

def mobilenetv1(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return VGG(make_layers_cifar100_v1(custom_cfg), dataset_history, dataset2num_classes, **kwargs)
     
# MobileNet v2 ############################################


class InvertedBlock_mobilenetv2(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock_mobilenetv2, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        # layers = []
        # if expand_ratio != 1:
        #     layers.append(conv1x1(ch_in, hidden_dim))
        # layers.extend([
        #     #dw
        #     dwise_conv(hidden_dim, stride=stride),
        #     #pw
        #     conv1x1(hidden_dim, ch_out)
        # ])
        layers = []
        if expand_ratio == 1:
            layers.extend([
                #dw
                # dwise_conv(hidden_dim, stride=stride),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pw
                # conv1x1(hidden_dim, ch_out)
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            ])
        else:
            layers.extend([
                # layers.append(conv1x1(ch_in, hidden_dim))
                nn.Conv2d(ch_in, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #dw
                # dwise_conv(hidden_dim, stride=stride),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pw
                # conv1x1(hidden_dim, ch_out)
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, init_weights=True):
        super(MobileNetV2, self).__init__()
        self.features = features
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x).view(-1, 1280)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Dropout2d(0.2))
            self.classifiers.append(nn.Linear(1280, num_classes)) ## updated ##

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Dropout2d(0.2))
            self.classifiers.append(nn.Linear(1280, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)*2-1].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)*2-1].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

def make_layers_cifar100_mobilenetv2(cfg, batch_norm=False):
    configs=[
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
            
    layers = []
    in_channels = 3
    layers += [conv3x3(in_channels, 32, stride=2)]
    in_channels = 32
    for t, c, n, s in configs:
        for i in range(n):
            stride = s if i == 0 else 1
            layers += [InvertedBlock_mobilenetv2(ch_in=in_channels, ch_out=c, expand_ratio=t, stride=stride)]
            in_channels = c
    layers += [conv1x1(in_channels, 1280),
                nn.AdaptiveAvgPool2d(1)]

    return Sequential_Debug(*layers)

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
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
        # layers = []
        # if expand_ratio != 1:
        #     # layers.append(conv1x1(ch_in, hidden_dim))
        #     nn.Conv2d(ch_in, hidden_dim, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU6(inplace=True),
        # layers.extend([
        #     #dw
        #     # dwise_conv(hidden_dim, stride=stride),
        #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU6(inplace=True),
        #     #pw
        #     # conv1x1(hidden_dim, ch_out)
        #     nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(ch_out),
        # ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class VGG2(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, init_weights=True):
        super(VGG2, self).__init__()
        self.features = features
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(1280, num_classes)) ## updated ##
            self.classifiers.append(nn.AdaptiveAvgPool2d(1)) ## updated ##

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(1280, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        
def make_layers_cifar100_v2(cfg, batch_norm=False):
    configs=[
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
            
    layers = []
    in_channels = 3
    layers += [conv3x3(in_channels, 32, stride=2)]
    in_channels = 32
    for t, c, n, s in configs:
        for i in range(n):
            stride = s if i == 0 else 1
            layers += [InvertedBlock(ch_in=in_channels, ch_out=c, expand_ratio=t, stride=stride)]
            in_channels = c
    layers += [conv1x1(in_channels, 1280),
               nn.Dropout2d(0.2)]

    layers += [
        View(-1, 1280)
    ]

    return Sequential_Debug(*layers)



def make_layers_cifar100_v2_2(cfg, batch_norm=False):
    configs=[
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
            
    layers = []
    in_channels = 3
    layers += [conv3x3(in_channels, 32, stride=2)]
    in_channels = 32
    for t, c, n, s in configs:
        for i in range(n):
            stride = s if i == 0 else 1
            layers += [InvertedBlock(ch_in=in_channels, ch_out=c, expand_ratio=t, stride=stride)]
            in_channels = c
    layers += [conv1x1(in_channels, 1280),
                nn.AdaptiveAvgPool2d(1)]

    layers += [nn.Dropout2d(0.2),
        View(-1, 1280)
    ]

    return Sequential_Debug(*layers)

'''
WRONG
###################
# https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV2.py
###################

class InvertedBlock2(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock2, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        # layers = []
        # if expand_ratio != 1:
        #     layers.append(conv1x1(ch_in, hidden_dim))
        # layers.extend([
        #     #dw
        #     dwise_conv(hidden_dim, stride=stride),
        #     #pw
        #     conv1x1(hidden_dim, ch_out)
        # ])
        layers = []
        if expand_ratio == 1:
            layers.extend([
                #dw
                # dwise_conv(hidden_dim, stride=stride),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pw
                # conv1x1(hidden_dim, ch_out)
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            ])
        else:
            layers.extend([
                # layers.append(conv1x1(ch_in, hidden_dim))
                nn.Conv2d(ch_in, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #dw
                # dwise_conv(hidden_dim, stride=stride),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pw
                # conv1x1(hidden_dim, ch_out)
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, dataset_history, dataset2num_classes, init_weights=True):
        super(MobileNetV2, self).__init__()
        self.datasets = dataset_history 
        self.dataset2num_classes = dataset2num_classes

        configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
                
        in_channels = 3
        conv = [conv3x3(in_channels, 32, stride=2)]

        in_channels = 32
        for t, c, n, s in configs:
            for i in range(n):
                stride = s if i == 0 else 1
                conv += [InvertedBlock2(ch_in=in_channels, ch_out=c, expand_ratio=t, stride=stride)]
                in_channels = c
        self.features = nn.Sequential(*conv)
        self.last_conv = conv1x1(in_channels, 1280)
        self.avg_pool = [nn.AdaptiveAvgPool2d(1), nn.Dropout2d(0.2)]
        self.classifiers = nn.ModuleList()

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(1280, num_classes)) ## updated ##

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(1280, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

def make_layers_cifar100_v2_3(cfg, batch_norm=False):
    configs=[
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
            
    conv = []
    in_channels = 3
    conv += [conv3x3(in_channels, 32, stride=2)]
    in_channels = 32
    for t, c, n, s in configs:
        for i in range(n):
            stride = s if i == 0 else 1
            conv += [InvertedBlock2(ch_in=in_channels, ch_out=c, expand_ratio=t, stride=stride)]
            in_channels = c
    conv += [conv1x1(in_channels, 1280),
                nn.AdaptiveAvgPool2d(1)]

    conv += [
        View(-1, 1280),
        nn.Dropout2d(0.2)
    ]

    return Sequential_Debug(*conv)
''         
'''



class InvertedBlock2(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock2, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        # layers = []
        # if expand_ratio != 1:
        #     layers.append(conv1x1(ch_in, hidden_dim))
        # layers.extend([
        #     #dw
        #     dwise_conv(hidden_dim, stride=stride),
        #     #pw
        #     conv1x1(hidden_dim, ch_out)
        # ])
        layers = []
        if expand_ratio != 1:
            # layers.append(conv1x1(ch_in, hidden_dim))
            nn.Conv2d(ch_in, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        layers.extend([
            #dw
            # dwise_conv(hidden_dim, stride=stride),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            #pw
            # conv1x1(hidden_dim, ch_out)
            nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ch_out),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class VGG3(nn.Module):
    def __init__(self, features, dataset_history, dataset2num_classes, init_weights=True):
        super(VGG3, self).__init__()
        self.features = features
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes

        if self.datasets:
            self._reconstruct_classifiers()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(1280, num_classes)) ## updated ##
            self.classifiers.append(nn.AdaptiveAvgPool2d(1)) ## updated ##

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(1280, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        
def make_layers_cifar100_v2_3(cfg, batch_norm=False):
    configs=[
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
            
    layers = []
    in_channels = 3
    layers += [conv3x3(in_channels, 32, stride=2)]
    in_channels = 32
    for t, c, n, s in configs:
        for i in range(n):
            stride = s if i == 0 else 1
            layers += [InvertedBlock(ch_in=in_channels, ch_out=c, expand_ratio=t, stride=stride)]
            in_channels = c
    layers += [conv1x1(in_channels, 1280)]

    layers += [
        View(-1, 1280),
       nn.Dropout2d(0.2)
    ]

    return Sequential_Debug(*layers)

def mobilenetv2(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return MobileNetV2(make_layers_cifar100_mobilenetv2(custom_cfg), dataset_history, dataset2num_classes, **kwargs)

def mobilenetv2_1(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return VGG2(make_layers_cifar100_v2(custom_cfg), dataset_history, dataset2num_classes, **kwargs)

def mobilenetv2_2(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return VGG2(make_layers_cifar100_v2_2(custom_cfg), dataset_history, dataset2num_classes, **kwargs)

def mobilenetv2_3(custom_cfg=[], dataset_history=[], dataset2num_classes={}, **kwargs):
    return VGG3(make_layers_cifar100_v2_3(custom_cfg), dataset_history, dataset2num_classes, **kwargs)

