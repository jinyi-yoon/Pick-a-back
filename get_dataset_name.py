import sys, yaml, os

config_path = os.path.join(os.path.dirname(__file__), '/home/youlee/MM-Pick-a-back/utils/dataset_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset_config = sys.argv[1]  
task_id = int(sys.argv[2])
datasets = config[dataset_config]['DATASETS']

print(datasets[task_id])
