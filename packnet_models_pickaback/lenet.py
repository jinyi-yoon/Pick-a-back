import torch.nn as nn
import pdb

__all__ = [
    'lenet5'
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
            self.classifiers.append(nn.Linear(84, num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            print('!!!! RUN !!!!')
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(84, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
            # print(self.classifiers)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

def make_layers_cifar100(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=5, bias=False) #conv2d = nn.Conv2d(in_channels, v, kernel_size=5, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    layers += [
        View(-1, 400),
        nn.Linear(400, 120),
        nn.ReLU(True),
        nn.Linear(120, 84),
        nn.ReLU(True)
    ]

    return Sequential_Debug(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def lenet5(custom_cfg, dataset_history=[], dataset2num_classes={}, **kwargs):
    return VGG(make_layers_cifar100(custom_cfg), dataset_history, dataset2num_classes, **kwargs)
