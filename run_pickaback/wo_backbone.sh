#!/bin/bash

# 사용법: ./wo_backbone.sh dataset_config
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 dataset_config"
    exit 1
fi

DATASET_CONFIG=$1
TASK_ID=4

DATASET=$(python3 get_dataset_name.py $DATASET_CONFIG $TASK_ID)

GPU_ID=0
ARCH='perceiver'
FINETUNE_EPOCHS=100
NUM_CLASSES=-1
INIT_LR=1e-2
PRUNING_LR=1e-3
LR_MASK=1e-4
NETWORK_WIDTH_MULTIPLIER=1.0
MAX_NETWORK_WIDTH_MULTIPLIER=1.0
PRUNING_RATIO_INTERVAL=0.1
TOTAL_NUM_TASKS=5
seed=2

VERSION_NAME='CPG_single_scratch_woexp'
CHECKPOINTS_NAME="checkpoints_${ARCH}"
BASELINE_FILE="logs_${ARCH}/baseline_${DATASET_CONFIG}_acc.txt"

state=2
while [ $state -eq 2 ]; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 CPG_cifar100_main_normal.py \
       --arch $ARCH \
       --dataset_config $DATASET_CONFIG \
       --dataset $DATASET \
       --num_classes $NUM_CLASSES \
       --lr $INIT_LR \
       --lr_mask $LR_MASK \
       --weight_decay 4e-5 \
       --save_folder ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/scratch \
       --epochs $FINETUNE_EPOCHS \
       --mode finetune \
       --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
       --max_allowed_network_width_multiplier $MAX_NETWORK_WIDTH_MULTIPLIER \
       --pruning_ratio_to_acc_record_file ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/gradual_prune/record.txt \
       --jsonfile $BASELINE_FILE \
       --log_path ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/train.log \
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

NR_EPOCH_FOR_EACH_PRUNE=20
PRUNING_FREQUENCY=10
START_SPARSITY=0.0
END_SPARSITY=0.1
NROF_EPOCH=$NR_EPOCH_FOR_EACH_PRUNE

if [ $state -ne 5 ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 CPG_cifar100_main_normal.py \
        --arch $ARCH \
        --dataset_config $DATASET_CONFIG \
        --dataset $DATASET \
        --num_classes $NUM_CLASSES  \
        --lr $PRUNING_LR \
        --lr_mask 0.0 \
        --weight_decay 4e-5 \
        --save_folder ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/gradual_prune \
        --load_folder ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/scratch \
        --epochs $NROF_EPOCH \
        --mode prune \
        --initial_sparsity=$START_SPARSITY \
        --target_sparsity=$END_SPARSITY \
        --pruning_frequency=$PRUNING_FREQUENCY \
        --pruning_interval=4 \
        --jsonfile $BASELINE_FILE \
        --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
        --max_allowed_network_width_multiplier $MAX_NETWORK_WIDTH_MULTIPLIER \
        --pruning_ratio_to_acc_record_file ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/gradual_prune/record.txt \
        --log_path ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/train.log \
        --total_num_tasks $TOTAL_NUM_TASKS \
        --seed $seed

    for RUN_ID in $(seq 1 9); do
        NROF_EPOCH=$NR_EPOCH_FOR_EACH_PRUNE
        START_SPARSITY=$END_SPARSITY
        if [ $RUN_ID -lt 9 ]; then
            END_SPARSITY=$(bc <<< $END_SPARSITY+$PRUNING_RATIO_INTERVAL)
        else
            END_SPARSITY=$(bc <<< $END_SPARSITY+0.05)
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID python3 CPG_cifar100_main_normal.py \
            --arch $ARCH \
            --dataset_config $DATASET_CONFIG \
            --dataset $DATASET \
            --num_classes $NUM_CLASSES \
            --lr $PRUNING_LR \
            --lr_mask 0.0 \
            --weight_decay 4e-5 \
            --save_folder ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/gradual_prune \
            --load_folder ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/gradual_prune \
            --epochs $NROF_EPOCH \
            --mode prune \
            --initial_sparsity=$START_SPARSITY \
            --target_sparsity=$END_SPARSITY \
            --pruning_frequency=$PRUNING_FREQUENCY \
            --pruning_interval=4 \
            --jsonfile $BASELINE_FILE \
            --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
            --max_allowed_network_width_multiplier $MAX_NETWORK_WIDTH_MULTIPLIER \
            --pruning_ratio_to_acc_record_file ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/gradual_prune/record.txt \
            --log_path ${CHECKPOINTS_NAME}/${VERSION_NAME}/$ARCH/${DATASET}/train.log \
            --total_num_tasks $TOTAL_NUM_TASKS \
            --seed $seed
    done
fi
