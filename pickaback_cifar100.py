"""Main entry point for doing all stuff."""
# import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter
import logging
import os
import pdb
import math
from tqdm import tqdm
import sys
import numpy as np

import utils_pickaback as utils
from utils_pickaback import Optimizers
from utils_pickaback.packnet_manager import Manager
import utils_pickaback.cifar100_dataset as dataset
import packnet_models_pickaback as packnet_models

import copy
from scipy import spatial
import csv

################################
# default
################################
arch = 'lenet5'
num_classes = -1
lr = 0.1
batch_size = 32
val_batch_size = 100
workers = 24
weight_decay = 4e-5
dataset_name = ''
train_path = ''
val_path = ''
cuda = True
seed = 1
checkpoint_format = './{save_folder}/checkpoint-{epoch}.pth.tar'
epochs = 160
restore_epoch = 0
save_folder = ''
load_folder = ''
one_shot_prune_perc = 0.5
mode = ''
logfile = ''
initial_from_task = ''
################################
DATASETS = [
    'None',  # dummy
    'aquatic_mammals',
    'fish',
    'flowers',
    'food_containers',
    'fruit_and_vegetables',
    'household_electrical_devices',
    'household_furniture',
    'insects',
    'large_carnivores',
    'large_man-made_outdoor_things',
    'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores',
    'medium_mammals',
    'non-insect_invertebrates',
    'people',
    'reptiles',
    'small_mammals',
    'trees',
    'vehicles_1',
    'vehicles_2'
]
################################
val_batch_size = 50
epsilon = 0.1
max_iterations = 100
################################

target_id = 14

ddvcc_list = []
ddvec_list = []
for task_id in range(1, 21):
    arch = 'lenet5'        
    dataset_name = DATASETS[task_id]
    dataset_name_target = DATASETS[target_id]
    dataset_name_test = DATASETS[task_id]
    dataset_name_test_target = DATASETS[target_id]
    num_classes = 5
    lr = 1e-2
    weight_decay = 4e-5
    load_folder = 'checkpoints_'+arch+'/baseline_scratch/' + arch + '/' + dataset_name
    load_folder2 = 'checkpoints_'+arch+'/baseline_scratch/' + arch + '/' + dataset_name_target
    epochs = 100
    mode = 'inference'
    logfile = 'logs'+arch+'/baseline_cifar100_acc_temp.txt'
    ################################
    
    if save_folder and not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        cuda = False
    
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    resume_folder = load_folder
    for try_epoch in range(200, 0, -1):
        if os.path.exists(checkpoint_format.format(
                save_folder=resume_folder, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break
        
    resume_from_epoch2 = 0
    resume_folder2 = load_folder2
    for try_epoch2 in range(200, 0, -1):
        if os.path.exists(checkpoint_format.format(
                save_folder=resume_folder2, epoch=try_epoch2)):
            resume_from_epoch2 = try_epoch2
            break
    
    if restore_epoch:
        resume_from_epoch = restore_epoch
    if restore_epoch:
        resume_from_epoch2 = restore_epoch
    
    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(train_path, val_path, dataset_name)
    
    if resume_from_epoch:
        filepath = checkpoint_format.format(save_folder=resume_folder, epoch=resume_from_epoch)
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
        
    if resume_from_epoch2:
        filepath2 = checkpoint_format.format(save_folder=resume_folder2, epoch=resume_from_epoch2)
        checkpoint2 = torch.load(filepath2)
        checkpoint_keys2 = checkpoint2.keys()
        dataset_history2 = checkpoint2['dataset_history']
        dataset2num_classes2 = checkpoint2['dataset2num_classes']
        masks2 = checkpoint2['masks']
        if 'shared_layer_info' in checkpoint_keys2:
            shared_layer_info2 = checkpoint2['shared_layer_info']
        else:
            shared_layer_info2 = {}
    
        if 'num_for_construct' in checkpoint_keys2:
            num_for_construct2 = checkpoint2['num_for_construct']
    else:
        dataset_history2 = []
        dataset2num_classes2 = {}
        masks2 = {}
        shared_layer_info2 = {}
    
    if arch == 'vgg16_bn_cifar100':
        model = packnet_models.__dict__[arch](pretrained=False, dataset_history=dataset_history,
                                              dataset2num_classes=dataset2num_classes)    
        model2 = packnet_models.__dict__[arch](pretrained=False, dataset_history=dataset_history2,
                                              dataset2num_classes=dataset2num_classes2)            
    elif arch == 'lenet5':
        custom_cfg = [6, 'A', 16, 'A']
        model = packnet_models.__dict__[arch](custom_cfg, dataset_history=dataset_history,
                                              dataset2num_classes=dataset2num_classes)    
        model2 = packnet_models.__dict__[arch](custom_cfg, dataset_history=dataset_history2,
                                              dataset2num_classes=dataset2num_classes2)
    elif arch == 'mobilenetv1':
        model = packnet_models.__dict__[arch]([], dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch]([], dataset_history=dataset_history2, dataset2num_classes=dataset2num_classes2)
    elif 'mobilenetv2' in arch:
        model = packnet_models.__dict__[arch]([], dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch]([], dataset_history=dataset_history2, dataset2num_classes=dataset2num_classes2)
    elif 'efficientnet' in arch:
        model = packnet_models.__dict__[arch]([], dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch]([], dataset_history=dataset_history2, dataset2num_classes=dataset2num_classes2)
    elif arch == 'resnet50':
        model = packnet_models.__dict__[arch](dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
        model2 = packnet_models.__dict__[arch](dataset_history=dataset_history2, dataset2num_classes=dataset2num_classes2)
    else:
        print('Error!')
        sys.exit(0)
    
    # Add and set the model dataset
    model.add_dataset(dataset_name, num_classes)
    model.set_dataset(dataset_name)
    model2.add_dataset(dataset_name_target, num_classes)
    model2.set_dataset(dataset_name_target)

    if not masks:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if 'classifiers' in name:
                    continue
                mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                masks[name] = mask

    if not masks2:
        for name, module in model2.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if 'classifiers' in name:
                    continue
                mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                masks2[name] = mask
    
    if dataset_name not in shared_layer_info:
        shared_layer_info[dataset_name] = {
            'conv_bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'fc_bias': {}
        }
    
    if dataset_name_target not in shared_layer_info2:
        shared_layer_info2[dataset_name_target] = {
            'conv_bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'fc_bias': {}
        }
    # print('... dataset setting')
    
    model = model.cuda()
    model2 = model2.cuda()
    
    train_loader = dataset.cifar100_train_loader(dataset_name, batch_size)
    val_loader = dataset.cifar100_val_loader(dataset_name_test, val_batch_size)
    train_loader2 = dataset.cifar100_train_loader(dataset_name_target, batch_size)
    val_loader2 = dataset.cifar100_val_loader(dataset_name_test_target, val_batch_size)
    
    # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
    if save_folder != load_folder:
        start_epoch = 0
    else:
        start_epoch = resume_from_epoch
    # print('... initial process')
    
    manager = Manager(dataset_name, checkpoint_format, weight_decay, cuda,
                      model, shared_layer_info, masks, train_loader, val_loader)
    
    manager2 = Manager(dataset_name_target, checkpoint_format, weight_decay, cuda,
                      model2, shared_layer_info2, masks2, train_loader2, val_loader2)
    
    # print('... inference mode')
    manager.load_checkpoint_for_inference(resume_from_epoch, resume_folder)
    manager2.load_checkpoint_for_inference(resume_from_epoch2, resume_folder2)

    print('======================')

    manager.pruner.apply_mask()
    manager.model.eval()
    manager2.pruner.apply_mask()
    manager2.model.eval()
    
    ##### gen_profiling_inputs_search #####
    with torch.no_grad():
        data1, target1 = next(iter(manager.val_loader))
        data2, target2 = next(iter(manager2.val_loader))
        
        if manager.cuda:
            data1, target1 = data1.cuda(), target1.cuda()
            data2, target2 = data2.cuda(), target2.cuda()
            inputs = np.concatenate([data1.cpu(), data2.cpu()])
            
            outputs1 = manager.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            outputs2 = manager2.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
    
    #################################
    
    input_shape = inputs[0].shape
    n_inputs = inputs.shape[0]
    ndims = np.prod(input_shape)
    
    initial_outputs1 = copy.deepcopy(outputs1)
    initial_outputs2 = copy.deepcopy(outputs2)
    
    def input_metrics(inputs):
        with torch.no_grad():
            outputs1 = manager.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            outputs2 = manager2.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
        output_dists1 = spatial.distance.cdist(outputs1, outputs1)
        output_dists2 = spatial.distance.cdist(outputs2, outputs2)
        diversity1 = np.mean(output_dists1)
        diversity2 = np.mean(output_dists2)
        return diversity1, diversity2
    
    def evaluate_inputs(inputs):
        with torch.no_grad():
            outputs1 = manager.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            outputs2 = manager2.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            
        metrics1, metrics2 = input_metrics(inputs)
        
        output_dist1 = np.mean(spatial.distance.cdist(outputs1, initial_outputs1).diagonal())
        output_dist2 = np.mean(spatial.distance.cdist(outputs2, initial_outputs2).diagonal())
        
        return output_dist1 * output_dist2 * metrics1 * metrics2
    
    score = evaluate_inputs(inputs)
    for iteration in range(max_iterations):
        mutation_pos = np.random.randint(0, ndims)
        mutation = np.zeros(ndims).astype(np.float32)
        mutation[mutation_pos] = epsilon
        mutation = np.reshape(mutation, input_shape)
        
        mutation_batch = np.zeros(shape=inputs.shape).astype(np.float32)
        mutation_idx = np.random.randint(0, n_inputs)
        mutation_batch[mutation_idx] = mutation
        
        mutate_right_inputs = inputs+mutation_batch
        mutate_right_score = evaluate_inputs(mutate_right_inputs)
        mutate_left_inputs = inputs - mutation_batch
        mutate_left_score = evaluate_inputs(mutate_left_inputs)
        
        if mutate_right_score <= score and mutate_left_score <= score:
            continue
        if mutate_right_score > mutate_left_score:
            inputs = mutate_right_inputs
            score = mutate_right_score
        else:
            inputs = mutate_left_inputs
            score = mutate_left_score
    
    profiling_inputs = inputs
    
    ##### computing metrics #####
    input_metrics_1, input_metrics_2 = input_metrics(profiling_inputs)
    
    ##### compute_ddv #####
    def compute_ddv_cos(inputs):
        global outputs
        global outputs2
        
        with torch.no_grad():
            dists = []
            outputs = manager.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            n_pairs = int(len(list(inputs)) / 2)
            for i in range(n_pairs):
                ya = outputs[i]
                yb = outputs[i + n_pairs]
                dist = spatial.distance.cosine(ya, yb)
                dists.append(dist)
                
            dists2 = []
            outputs2 = manager2.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            n_pairs2 = int(len(list(inputs)) / 2)
            for i in range(n_pairs2):
                ya = outputs2[i]
                yb = outputs2[i + n_pairs]
                dist = spatial.distance.cosine(ya, yb)
                dists2.append(dist)
        return np.array(dists), np.array(dists2)
    
    def compute_ddv_euc(inputs):
        global outputs
        global outputs2
        
        with torch.no_grad():
            dists = []
            outputs = manager.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            n_pairs = int(len(list(inputs)) / 2)
            for i in range(n_pairs):
                ya = outputs[i]
                yb = outputs[i + n_pairs]
                dist = spatial.distance.euclidean(ya, yb) # dist = spatial.distance.cosine(ya, yb)
                dists.append(dist)
                
            dists2 = []
            outputs2 = manager2.model(torch.Tensor(inputs).cuda()).to('cpu').tolist()
            n_pairs2 = int(len(list(inputs)) / 2)
            for i in range(n_pairs2):
                ya = outputs2[i]
                yb = outputs2[i + n_pairs]
                dist = spatial.distance.euclidean(ya, yb) # dist = spatial.distance.cosine(ya, yb)
                dists2.append(dist)
        return np.array(dists), np.array(dists2)
    
    ##### compute_similarity #####
    def compute_sim_cos(ddv1, ddv2):
        return spatial.distance.cosine(ddv1, ddv2)
    
    # DDV-CC
    ddv1, ddv2 = compute_ddv_cos(profiling_inputs)
    ddv_distance = compute_sim_cos(ddv1, ddv2)
    print('DDV cos-cos [%d => %d] %.5f'%(task_id, target_id, ddv_distance))
    ddvcc_list.append(ddv_distance)

    # DDV-EC
    ddv1, ddv2 = compute_ddv_euc(profiling_inputs)
    ddv_distance = compute_sim_cos(ddv1, ddv2)
    print('DDV euc-cos [%d => %d] %.5f'%(task_id, target_id, ddv_distance))
    ddvec_list.append(ddv_distance)

print('Selected backbone for target '+str(target_id)+' = (euc) '+str(ddvec_list.index(max(ddvec_list))+1))
# print('Selected backbone for target '+str(target_id)+' = (cos) '+str(ddvcc_list.index(max(ddvcc_list))+1))
