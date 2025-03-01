#!/bin/bash

# 사용법: ./w_backbone.sh dataset_config
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_CONFIG=$1
TARGET_TASK_ID=1
DATASET=$(python3 get_dataset_name.py $DATASET_CONFIG $TARGET_TASK_ID)

NUM_CLASSES=-1
INIT_LR=1e-2
PRUNING_LR=1e-3

GPU_ID=0
<<<<<<< HEAD
arch='perceiver'
finetune_epochs=100
network_width_multiplier=1.0
max_network_width_multiplier=2.0
pruning_ratio_interval=0.1
lr_mask=1e-4
total_num_tasks=5
=======
ARCH='perceiver'
FINETUNE_EPOCHS=100
NETWORK_WIDTH_MULTIPLIER=1.0
MAX_NETWORK_WIDTH_MULTIPLIER=2.0
PRUNING_RATIO_INTERVAL=0.1
LR_MASK=1e-4
TOTAL_NUM_TASKS=5
>>>>>>> n24news

task_id=12
target_id=14
seed=2

version_name='CPG_fromsingle_scratch_woexp_target'
single_version_name='CPG_single_scratch_woexp'
baseline_file="logs_${ARCH}/baseline_${DATASET_CONFIG}_acc.txt"
checkpoints_name="checkpoints_${ARCH}"

####################
##### Training #####
####################
state=2
while [ $state -eq 2 ]; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 CPG_cifar100_main_normal.py \
       --arch $ARCH \
       --dataset $DATASET --num_classes $NUM_CLASSES \
       --lr $INIT_LR \
       --lr_mask $LR_MASK \
       --weight_decay 4e-5 \
       --save_folder $checkpoints_name/$version_name/$ARCH/${DATASET[$task_id]}/${DATASET[$target_id]}/scratch \
       --load_folder $checkpoints_name/$single_version_name/$ARCH/${DATASET[$task_id]}/gradual_prune \
       --epochs $FINETUNE_EPOCHS \
       --mode finetune \
       --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
       --max_allowed_network_width_multiplier $MAX_NETWORK_WIDTH_MULTIPLIER \
       --pruning_ratio_to_acc_record_file $checkpoints_name/$version_name/$ARCH/${DATASET[$task_id]}/${DATASET[$target_id]}/gradual_prune/record.txt \
       --jsonfile $baseline_file \
       --log_path $checkpoints_name/$version_name/$ARCH/${DATASET[$task_id]}/${DATASET[$target_id]}/train.log \
       --total_num_tasks $TOTAL_NUM_TASKS \
       --seed $seed

    state=$?
    if [ $state -eq 2 ]; then
        if [[ "$NETWORK_WIDTH_MULTIPLIER" == "$MAX_NETWORK_WIDTH_MULTIPLIER" ]]; then
            break
        fi
        
        NETWORK_WIDTH_MULTIPLIER=$(bc <<< $NETWORK_WIDTH_MULTIPLIER+0.5)
        echo "New network_width_multiplier: $NETWORK_WIDTH_MULTIPLIER"
        continue
    elif [ $state -eq 3 ]; then
        echo "You should provide the baseline_${DATASET_CONFIG}_acc.txt as criterion to decide whether the capacity of network is enough for new task"
        exit 0
    fi
done
