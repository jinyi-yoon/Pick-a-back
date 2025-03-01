import os
import yaml
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


config_path = os.path.join(os.path.dirname(__file__), 'dataset_config.yaml')
with open(config_path, 'r') as f:
    dataset_config = yaml.safe_load(f)


def get_transforms(cfg, dataset_name=None, is_train=True):
    transform_list = []

    if is_train:
        if 'crop' in cfg:
            crop_size = tuple(cfg['crop'].get('size', [32, 32]))
            padding = cfg['crop'].get('padding', 4)
            transform_list.append(transforms.RandomCrop(crop_size, padding=padding))
        elif 'resize' in cfg:
            resize_size = tuple(cfg['resize'])
            transform_list.append(transforms.Resize(resize_size))
        
        if cfg.get('random_horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
    else:
        if 'resize' in cfg:
            resize_size = tuple(cfg['resize'])
            transform_list.append(transforms.Resize(resize_size))
    
    transform_list.append(transforms.ToTensor())

    if 'mean' in cfg and 'std' in cfg:
        mean = None
        std = None

        if isinstance(cfg['mean'], dict):
            if dataset_name and dataset_name in cfg['mean']:
                mean = cfg['mean'][dataset_name]
                std = cfg['std'][dataset_name]
            else:
                if 'cifar100' in cfg['mean']:
                    mean = cfg['mean']['cifar100']
                    std = cfg['std']['cifar100']
                else:
                    raise ValueError
        else:
            mean = cfg['mean']
            std = cfg['std']

        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
    return transforms.Compose(transform_list)


def train_loader(config_name, batch_size, dataset_name=None, sub_dataset=None, num_workers=4, pin_memory=True):
    cfg = dataset_config[config_name]

    if sub_dataset is not None and cfg.get("subfolder", False):
        train_path = os.path.join(cfg['train_path'], sub_dataset)
    else:
        train_path = cfg['train_path']
                    
    train_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=True)
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def val_loader(config_name, batch_size, dataset_name=None, sub_dataset=None, num_workers=4, pin_memory=True):
    cfg = dataset_config[config_name]

    if sub_dataset is not None and cfg.get("subfolder", False):
        test_path = os.path.join(cfg['test_path'], sub_dataset)
    else:
        test_path = cfg['test_path']
    
    val_transform = get_transforms(cfg, dataset_name=dataset_name, is_train=False)
    val_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
