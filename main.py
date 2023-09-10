import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time
import copy
import random
import numpy as np

from utils import log_end_epoch, get_item_propensity, get_common_path
    

def main(hyper_params, gpu_id = None):
    if gpu_id is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from jax.config import config
    if 'float64' in hyper_params and hyper_params['float64'] == True: config.update('jax_enable_x64', True)

    from data import Dataset

    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])

    os.makedirs("/content/drive/MyDrive/lib/results/logs/", exist_ok=True)
    hyper_params['log_file'] = "/content/drive/MyDrive/lib/results/logs/" + get_common_path(hyper_params) + ".txt"
    
    data = Dataset(hyper_params)
    hyper_params = copy.deepcopy(data.hyper_params) # Updated w/ data-stats

if __name__ == "__main__":
    from hyper_params import hyper_params
    main(hyper_params)
