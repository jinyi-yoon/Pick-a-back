#!/bin/bash


TARGET_TASK_ID=1

dataset=(
    'None'     # dummy
    'aquatic_mammals' #1
    'fish' #2
    'flowers' #3
    'food_containers' #4
    'fruit_and_vegetables' #5
    'household_electrical_devices' #6
    'household_furniture' #7
    'insects' #8
    'large_carnivores' #9
    'large_man-made_outdoor_things' #10
    'large_natural_outdoor_scenes' #11
    'large_omnivores_and_herbivores' #12
    'medium_mammals' #13
    'non-insect_invertebrates' #14
    'people' #15
    'reptiles' #16
    'small_mammals' #17
    'trees' #18
    'vehicles_1' #19
    'vehicles_2' #20
)

num_classes=(
    5
)

init_lr=(
    1e-2
)

pruning_lr=(
    1e-3
)

GPU_ID=0
arch='lenet5'
finetune_epochs=100
network_width_multiplier=1.0
max_network_width_multiplier=1.0
pruning_ratio_interval=0.1
lr_mask=1e-4

task_id=4

version_name='CPG_single_scratch_woexp' ##### CHANGE #####
baseline_file='logs_'$arch'/baseline_cifar100_acc_scratch.txt' #### CHANGE #####

####################################
##### Select the pruning ratio #####
####################################
python tools/choose_appropriate_pruning_ratio_for_next_task_notrmfiles.py \
    --pruning_ratio_to_acc_record_file checkpoints_${arch}/$version_name/$arch/${dataset[task_id]}/gradual_prune/record.txt \
    --baseline_acc_file $baseline_file \
    --allow_acc_loss 0.001 \
    --dataset ${dataset[task_id]} \
    --network_width_multiplier $network_width_multiplier \
    --log_path checkpoints_${arch}/$version_name/$arch/${dataset[task_id]}/train.log
