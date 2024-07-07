# Pick-a-back
This is an implementation for 'Pick-a-back: Selective Device-to-Device Knowledge Transfer in Federated Continual Learning' (_under review_).

First of all, we would like to express our sincere gratitude to the open source for their valuable contributions to this research:
* [ModelDiff: testing-based DNN similarity comparison for model reuse detection](https://github.com/MadryLab/modeldiff)
* [Compacting, Picking and Growing for Unforgetting Continual Learning](https://github.com/ivclab/CPG)
* [Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights](https://github.com/arunmallya/piggyback)

---

### Ready for data
Download the CIFAR-100 dataset in the directory 'data/' and unzip the file. The structure of the dataset is as follows:
```
└─cifar100_org
    ├─test
    │  ├─aquatic_mammals
    │  │  ├─beaver
    │  │  ├─dolphin
    │  │  ├─otter
    │  │  ├─seal
    │  │  └─whale
    │  ├─ ...
    │  └─vehicles_2
    │      ├─lawn_mower
    │      ├─rocket
    │      ├─streetcar
    │      ├─tank
    │      └─tractor
    └─train
        ├─aquatic_mammals
        │  ├─beaver
        │  ├─dolphin
        │  ├─otter
        │  ├─seal
        │  └─whale
        ├─ ...
        └─vehicles_2
            ├─lawn_mower
            ├─rocket
            ├─streetcar
            ├─tank
            └─tractor
```
All the subclasses should be located under the superclass. Please check the hierarchy of the dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## STEP 1) Baseline accuracy
Use the following command to train the scratch model for each task. This accuracy is used for the threshold of model expansion decision and pruning ratio selection.
```console
bash run_pickaback/baseline.sh
```
It generates 'logs_lenet5/baseline_cifar100_acc_scratch.txt' file. The checkpoints are available at 'checkpoints_lenet/baseline_scratch.'

## STEP 2-1) Train the backbone task
The following command is to train the first task. The first task is used as a backbone model for the next task. It processes the training and pruning steps.
```console
bash run_pickaback/wo_backbone.sh
```
The checkpoints are generated at 'checkpoints_lenet5/CPG_single_scratch_woexp.'


## STEP 2-2) Select the pruning ratio
We select the proper pruning ratio for the backbone model. Please refer to the details from [gradual pruning](https://arxiv.org/abs/1710.01878).
```console
bash run_pickaback/select_pruning_ratio_of_backbone.sh
```
The selected pruning ratio is provided as:
> 2023-05-24 17:13:43,124 - root - INFO - We choose pruning ratio 0.8


## STEP 3-1) Find the similar model
Based on [ModelDiff](https://dl.acm.org/doi/abs/10.1145/3460319.3464816), we select the model with similar decision pattern for the target task.
```console
bash run_pickaback/find_backbone.sh
```
It find the similar model for the target task as:
> Selected backbone for target 14 = (euc) 4


## STEP 3-2) Train the target task
Run the following command to train the continual model for the target task.
```console
bash run_pickaback/w_backbone.sh
```
The checkpoints are available at 'checkpoints_lenet5/CPG_fromsingle_scratch_woexp_target.' You can get the final accuracy as follows:
> 2023-04-11 21:40:14,129 - root - INFO - In validate()-> Val Ep. #100 loss: 0.742, accuracy: 75.60, sparsity: 0.000, task2 ratio: 1.794, zero ratio: 0.000, mpl: 1.4142135623730951, shared_ratio: 0.778
