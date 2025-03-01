#!/bin/bash

# 사용법: ./find_backbone.sh dataset_name
DATASET_NAME=$1  

echo "Running with dataset config: $DATASET_NAME"  
GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 pickaback_cifar100.py --dataset_config $DATASET_NAME
