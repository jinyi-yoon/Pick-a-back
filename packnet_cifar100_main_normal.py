"""Main entry point for doing all stuff."""
import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

import logging
import os
import pdb
import math
from tqdm import tqdm
import sys
import numpy as np

import utils
from utils import Optimizers
from utils.packnet_manager import Manager
import utils.cifar100_dataset as dataset
import packnet_models

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnetb0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnetb1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnetb2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnetb3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnetb4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnetb5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnetb6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnetb7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='vgg16_bn_cifar100',
                   help='Architectures')
parser.add_argument('--num_classes', type=int, default=-1,
                   help='Num outputs for dataset')

# Optimization options.
parser.add_argument('--lr', type=float, default=0.1,
                   help='Learning rate for parameters, used for baselines')

parser.add_argument('--batch_size', type=int, default=32,
                   help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=100,
                   help='input batch size for validation')
parser.add_argument('--workers', type=int, default=24, help='')
parser.add_argument('--weight_decay', type=float, default=4e-5,
                   help='Weight decay')

# Paths.
parser.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
parser.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
parser.add_argument('--val_path', type=str, default='',
                   help='Location of test data')

# Other.
parser.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--checkpoint_format', type=str,
                    default='./{save_folder}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--epochs', type=int, default=160,
                    help='number of epochs to train')
parser.add_argument('--restore_epoch', type=int, default=0, help='')
parser.add_argument('--save_folder', type=str,
                    help='folder name inside one_check folder')
parser.add_argument('--load_folder', default='', help='')
parser.add_argument('--one_shot_prune_perc', type=float, default=0.5,
                   help='% of neurons to prune per layer')
parser.add_argument('--mode',
                   choices=['finetune', 'prune', 'inference'],
                   help='Run mode')
parser.add_argument('--logfile', type=str, help='file to save baseline accuracy')
parser.add_argument('--initial_from_task', type=str, help="")
parser.add_argument('--use_imagenet_pretrained', action='store_true', default=False,
                    help='')
parser.add_argument('--jsonfile', type=str, help='file to restore baseline validation accuracy')


def main():
    """Do stuff."""
    args = parser.parse_args()
    if args.save_folder and not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        args.cuda = False

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    #cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    resume_folder = args.load_folder
    for try_epoch in range(200, 0, -1):
        if os.path.exists(args.checkpoint_format.format(
            save_folder=resume_folder, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    if args.restore_epoch:
        resume_from_epoch = args.restore_epoch

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    if resume_from_epoch:
        filepath = args.checkpoint_format.format(save_folder=resume_folder, epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        checkpoint_keys = checkpoint.keys()
        dataset_history = checkpoint['dataset_history']
        dataset2num_classes = checkpoint['dataset2num_classes']
        masks = checkpoint['masks']
        if 'shared_layer_info' in checkpoint_keys:
            shared_layer_info = checkpoint['shared_layer_info']
        else:
            shared_layer_info = {}

        if 'num_for_construct' in checkpoint_keys:
            num_for_construct = checkpoint['num_for_construct']
    else:
        dataset_history = []
        dataset2num_classes = {}
        masks = {}
        shared_layer_info = {}

    if args.arch == 'vgg16_bn_cifar100':
        model = packnet_models.__dict__[args.arch](pretrained=False, dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    elif args.arch == 'lenet5':
        custom_cfg = [6, 'A', 16, 'A']
        model = packnet_models.__dict__[args.arch](custom_cfg, dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    elif args.arch == 'mobilenetv1':
        model = packnet_models.__dict__[args.arch]([], dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    elif 'mobilenetv2' in args.arch:
        model = packnet_models.__dict__[args.arch]([], dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    elif 'efficientnet' in args.arch:
        model = packnet_models.__dict__[args.arch]([], dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    elif args.arch == 'resnet50':
        model = packnet_models.__dict__[args.arch](dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    else:
        print('Error!')
        sys.exit(0)

    # Add and set the model dataset
    model.add_dataset(args.dataset, args.num_classes)
    model.set_dataset(args.dataset)
    model = model.cuda()
    
    # For datasets whose image_size is 224 and also the first task
    if args.use_imagenet_pretrained and model.datasets.index(args.dataset) == 0: ##### Jinee ##### model.module.datasets.index(args.dataset) == 0:
        curr_model_state_dict = model.state_dict()
        if args.arch == 'vgg16_bn':
            state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
            curr_model_state_dict = model.state_dict()
            for name, param in state_dict.items():
                if 'classifier' not in name:
                    curr_model_state_dict[name].copy_(param)
            curr_model_state_dict['features.45.weight'].copy_(state_dict['classifier.0.weight'])
            curr_model_state_dict['features.45.bias'].copy_(state_dict['classifier.0.bias'])
            curr_model_state_dict['features.48.weight'].copy_(state_dict['classifier.3.weight'])
            curr_model_state_dict['features.48.bias'].copy_(state_dict['classifier.3.bias'])
            if args.dataset == 'imagenet':
                curr_model_state_dict['classifiers.0.weight'].copy_(state_dict['classifier.6.weight'])
                curr_model_state_dict['classifiers.0.bias'].copy_(state_dict['classifier.6.bias'])
        elif args.arch == 'vgg16_bn_cifar100': ##### Jinee #####
            state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
            for name, param in state_dict.items():
                if 'classifier' not in name and 'bias' not in name:
                    curr_model_state_dict[name].copy_(param)
            if args.dataset == 'imagenet':
                curr_model_state_dict['classifiers.0.weight'].copy_(state_dict['classifier.6.weight'])
                curr_model_state_dict['classifiers.0.bias'].copy_(state_dict['classifier.6.bias'])
        elif args.arch == 'resnet50':
            state_dict = model_zoo.load_url(model_urls['resnet50'])
            for ([name, param], [name2, param2]) in zip(state_dict.items(), curr_model_state_dict.items()):
                if 'fc' not in name:
                    curr_model_state_dict[name].copy_(param) ##### Jinee ##### ['module.' + name].copy_(param)
            if args.dataset == 'imagenet':
                curr_model_state_dict['module.classifiers.0.weight'].copy_(state_dict['fc.weight'])
                curr_model_state_dict['module.classifiers.0.bias'].copy_(state_dict['fc.bias'])
        elif 'mobilenetv2' in args.arch:
            state_dict = model_zoo.load_url(model_urls['mobilenetv2'])
            for ([name, param], [name2, param2]) in zip(state_dict.items(), curr_model_state_dict.items()):                
                if 'classifier' not in name and 'bias' not in name:               
                    curr_model_state_dict[name2].copy_(param)
        elif 'efficientnet' in args.arch:
            state_dict = model_zoo.load_url(model_urls[args.arch])
            for ([name, param], [name2, param2]) in zip(state_dict.items(), curr_model_state_dict.items()):                
                if 'classifier' not in name and 'bias' not in name:               
                    curr_model_state_dict[name2].copy_(param)
        else:
            print("Currently, we didn't define the mapping of {} between imagenet pretrained weight and our model".format(args.arch))
            sys.exit(5)

    if not masks:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if 'classifiers' in name:
                    continue
                mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                masks[name] = mask

    if args.dataset not in shared_layer_info:
        shared_layer_info[args.dataset] = {
            'conv_bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'fc_bias': {}
        }

    train_loader = dataset.train_loader(args.dataset, args.batch_size)
    val_loader = dataset.val_loader(args.dataset, args.val_batch_size)

    # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
    if args.save_folder != args.load_folder:
        start_epoch = 0
    else:
        start_epoch = resume_from_epoch

    manager = Manager(args, model, shared_layer_info, masks, train_loader, val_loader)

    if args.mode == 'inference':
        manager.load_checkpoint_for_inference(resume_from_epoch, resume_folder)
        manager.validate(resume_from_epoch-1)
        return

    lr = args.lr
    # update all layers
    named_params = dict(model.named_parameters())
    params_to_optimize_via_SGD = []
    named_params_to_optimize_via_SGD = []
    masks_to_optimize_via_SGD = []
    named_masks_to_optimize_via_SGD = []

    for tuple_ in named_params.items():
        if 'classifiers' in tuple_[0]:
            if '.{}.'.format(model.datasets.index(args.dataset)) in tuple_[0]: ##### Jinee ##### if '.{}.'.format(model.module.datasets.index(args.dataset)) in tuple_[0]:
                params_to_optimize_via_SGD.append(tuple_[1])
                named_params_to_optimize_via_SGD.append(tuple_)
            continue
        else:
            params_to_optimize_via_SGD.append(tuple_[1])
            named_params_to_optimize_via_SGD.append(tuple_)

    # here we must set weight decay to 0.0,
    # because the weight decay strategy in build-in step() function will change every weight elem in the tensor,
    # which will hurt previous tasks' accuracy. (Instead, we do weight decay ourself in the `prune.py`)
    optimizer_network = optim.SGD(params_to_optimize_via_SGD, lr=lr,
                          weight_decay=0.0, momentum=0.9, nesterov=True)

    optimizers = Optimizers()
    optimizers.add(optimizer_network, lr)

    manager.load_checkpoint(optimizers, resume_from_epoch, resume_folder)

    """Performs training."""
    curr_lrs = []
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            curr_lrs.append(param_group['lr'])
            break

    if args.mode == 'prune':
        print()
        print('Sparsity ratio: {}'.format(args.one_shot_prune_perc))
        print('Before pruning: ')
        with open(args.jsonfile, 'r') as jsonfile:
            json_data = json.load(jsonfile)
            baseline_acc = float(json_data[args.dataset])
        print('Execute one shot pruning ...')
        manager.one_shot_prune(args.one_shot_prune_perc)
    elif args.mode == 'finetune':
        manager.pruner.make_finetuning_mask()

    for epoch_idx in range(start_epoch, args.epochs):
        avg_train_acc = manager.train(optimizers, epoch_idx, curr_lrs)
        avg_val_acc = manager.validate(epoch_idx)

        if args.mode == 'finetune':
            if epoch_idx + 1 == 50 or epoch_idx + 1 == 80:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] *= 0.1
                curr_lrs[0] = param_group['lr']

        if args.mode == 'prune':
            if epoch_idx + 1 == 25:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] *= 0.1
                curr_lrs[0] = param_group['lr']


    if args.save_folder is not None:
        pass
    else:
        print('Something is wrong! Block the program with pdb')
        pdb.set_trace()

    if args.mode == 'finetune':
        manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)
        if args.logfile:
            json_data = {}
            if os.path.isfile(args.logfile):
                with open(args.logfile) as json_file:
                    json_data = json.load(json_file)
            json_data[args.dataset] = '{:.4f}'.format(avg_val_acc)
            with open(args.logfile, 'w') as json_file:
                json.dump(json_data, json_file)

        if avg_train_acc < 0.97:
            print('Cannot prune any more!')

    elif args.mode == 'prune':
        #if avg_train_acc > 0.97 and (avg_val_acc - baseline_acc) >= -0.01:
        if avg_train_acc > 0.97:
            manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)
        else:
            manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)
            print('Pruning too much!')

    print('-' * 16)

if __name__ == '__main__':
    main()
