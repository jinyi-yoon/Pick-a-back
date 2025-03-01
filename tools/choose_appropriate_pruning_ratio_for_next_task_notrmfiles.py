import shutil
import argparse
import json
import pdb
import os
import logging
import sys
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--pruning_ratio_to_acc_record_file', type=str, help='Path to the record file')
parser.add_argument('--allow_acc_loss', type=float, default=0.0, help='Allowed accuracy loss')
parser.add_argument('--baseline_acc_file', type=str, help='Path to the baseline accuracy JSON file')
parser.add_argument('--dataset', type=str, default='', help='Dataset name (if empty, will be selected from YAML)')
parser.add_argument('--dataset_config', type=str, default='', help='Dataset configuration key defined in dataset_config.yaml (e.g., cifar100, n24news)')
parser.add_argument('--task_id', type=int, default=-1, help='Task id (index) to select dataset from YAML')
parser.add_argument('--network_width_multiplier', type=float, help='Network width multiplier')
parser.add_argument('--max_allowed_network_width_multiplier', type=float, help='Max allowed network width multiplier')
parser.add_argument('--log_path', type=str, help='Path to the log file')

def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def main():
    args = parser.parse_args()
    if args.log_path:
        set_logger(args.log_path)
    
    if args.dataset == "" and args.dataset_config != "" and args.task_id >= 0:
        config_path = os.path.join(os.path.dirname(__file__), 'dataset_config.yaml')
        with open(config_path, 'r') as f:
            config_yaml = yaml.safe_load(f)
        datasets_array = config_yaml[args.dataset_config]['DATASETS']
        if args.task_id < len(datasets_array):
            args.dataset = datasets_array[args.task_id]
        else:
            print("Error: task_id {} is out of range for {} dataset".format(args.task_id, args.dataset_config))
            sys.exit(1)

    save_folder = args.pruning_ratio_to_acc_record_file.rsplit('/', 1)[0]
    with open(args.baseline_acc_file, 'r') as jf:
        json_data = json.load(jf)
        criterion_acc = float(json_data[args.dataset])

    with open(args.pruning_ratio_to_acc_record_file, 'r') as jf:
        json_data = json.load(jf)
        acc_before_prune = json_data['0.0']
        json_data.pop('0.0')
        available_pruning_ratios = sorted(json_data.keys(), key=float, reverse=True)
        flag_match = False
        chosen_pruning_ratio = 0.0

        for pruning_ratio in available_pruning_ratios:
            acc = json_data[pruning_ratio]
            if (acc + args.allow_acc_loss >= criterion_acc) or (
                (args.network_width_multiplier == args.max_allowed_network_width_multiplier) and (acc_before_prune < criterion_acc)):
                chosen_pruning_ratio = pruning_ratio
                checkpoint_folder = os.path.join(save_folder, str(pruning_ratio))
                for filename in os.listdir(checkpoint_folder):
                    shutil.copyfile(os.path.join(checkpoint_folder, filename),
                                    os.path.join(save_folder, filename))
                flag_match = True
                break

        if not flag_match:
            logging.info('We select scratch')
            folder_before = os.path.join(save_folder.rsplit('/', 1)[0], 'scratch')
            for filename in os.listdir(folder_before):
                shutil.copyfile(os.path.join(folder_before, filename),
                                os.path.join(save_folder, filename))

        logging.info('We choose pruning_ratio {}'.format(chosen_pruning_ratio))

if __name__ == '__main__':
    main()
