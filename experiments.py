import time
import copy
import random
import numpy as np

from utils import log_end_epoch, get_item_propensity, get_common_path
from data import Dataset
from hyper_params import hyper_params

def train_KRR(neural_kernel, hyper_params, data):
    from KRR_model import make_kernelized_rr_forward
    from eval import evaluate

    # This just instantiates the function
    kernelized_rr_forward, kernel_fn = make_kernelized_rr_forward(neural_kernel, hyper_params)
    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)

    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@10"
    best_metric, best_lamda = None, None

    # Validate on the validation-set
    for lamda in hyper_params['grid_search_lamda'] if hyper_params['grid_search'] else [hyper_params['lamda']]:
        hyper_params['lamda'] = lamda
        val_metrics = evaluate(hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix)
        if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric): best_metric, best_lamda = val_metrics[VAL_METRIC], lamda

    # Return metrics with the best lamda on the test-set
    hyper_params['lamda'] = best_lamda
    final_time = time.time()
    test_metrics = evaluate(hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix, test_set_eval = True)
    print("Training of the final model time: ", time.time() - final_time)

    log_end_epoch(hyper_params, test_metrics, 0, time.time() - start_time)
    start_time = time.time()

    return test_metrics


def train_MLP(hyper_params, data):
    from MLP_model_keras import make_MLP_model
    from eval import evaluate

    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)

    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@10"
    best_metric = None
    best_params = {'adam_lr': None,
                   'epochs': None,
                   'batch_size': None,
                   'reg': None}
    metrics = []

    X_train = data.data['train_matrix'].todense()
    y_train = data.data['train_matrix'].todense()
    i = 1

    # Validate on the validation-set
    if hyper_params['grid_search']:
        for adam_lr in hyper_params['grid_search_adam_lr']:
          hyper_params['adam_lr'] = adam_lr
          for epochs in hyper_params['grid_search_epochs']:
            hyper_params['epochs'] = epochs
            for batch_size in hyper_params['grid_search_batch_size']:
              hyper_params['batch_size'] = batch_size
              #print(i)
              i += 1
              for reg in hyper_params['grid_search_reg']:
                hyper_params['reg'] = reg
                predict_MLP = make_MLP_model(X_train, y_train, hyper_params)
                method = lambda x, y, reg: predict_MLP(y)
                val_metrics = evaluate(hyper_params, method, data,
                                       item_propensity, sampled_matrix)
                metrics.append(val_metrics[VAL_METRIC])
                if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric):
                  best_metric = val_metrics[VAL_METRIC]
                  best_params = {'adam_lr': adam_lr,
                                 'epochs': epochs,
                                 'batch_size': batch_size,
                                 'reg': reg}

        # Return metrics with the best lamda on the test-set
        hyper_params['adam_lr'] = best_params['adam_lr']
        hyper_params['epochs'] = best_params['epochs']
        hyper_params['batch_size'] = best_params['batch_size']
        hyper_params['reg'] = best_params['reg']

    X_train_val = X_train + data.data['val_matrix'].todense()
    y_train_val = y_train + data.data['val_matrix'].todense()
    final_time = time.time()
    predict_MLP = make_MLP_model(X_train_val, y_train_val, hyper_params)
    method = lambda x, y, reg: predict_MLP(y)
    test_metrics = evaluate(hyper_params, method, data, item_propensity, sampled_matrix, test_set_eval = True)
    print("Training of the final model time: ", time.time() - final_time)
    log_end_epoch(hyper_params, test_metrics, 0, time.time() - start_time)
    start_time = time.time()

    return test_metrics

def train_MLP_last_layer(hyper_params, data):
    from MLP_model_keras import make_MLP_model_last_layer
    from eval import evaluate

    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)

    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@10"
    best_metric = None
    best_params = {'adam_lr': None,
                   'epochs': None,
                   'batch_size': None,
                   'reg': None}
    metrics = []

    X_train = data.data['train_matrix'].todense()
    y_train = data.data['train_matrix'].todense()
    i = 1

    # Validate on the validation-set
    if hyper_params['grid_search']:
        for adam_lr in hyper_params['grid_search_adam_lr']:
          hyper_params['adam_lr'] = adam_lr
          for epochs in hyper_params['grid_search_epochs']:
            hyper_params['epochs'] = epochs
            for batch_size in hyper_params['grid_search_batch_size']:
              hyper_params['batch_size'] = batch_size
              #print(i)
              i += 1
              for reg in hyper_params['grid_search_reg']:
                hyper_params['reg'] = reg
                predict_MLP = make_MLP_model_last_layer(X_train, y_train, hyper_params)
                method = lambda x, y, reg: predict_MLP(y)
                val_metrics = evaluate(hyper_params, method, data,
                                       item_propensity, sampled_matrix)
                metrics.append(val_metrics[VAL_METRIC])
                if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric):
                  best_metric = val_metrics[VAL_METRIC]
                  best_params = {'adam_lr': adam_lr,
                                 'epochs': epochs,
                                 'batch_size': batch_size,
                                 'reg': reg}

        # Return metrics with the best lamda on the test-set
        hyper_params['adam_lr'] = best_params['adam_lr']
        hyper_params['epochs'] = best_params['epochs']
        hyper_params['batch_size'] = best_params['batch_size']
        hyper_params['reg'] = best_params['reg']

    X_train_val = X_train + data.data['val_matrix'].todense()
    y_train_val = y_train + data.data['val_matrix'].todense()
    final_time = time.time()
    predict_MLP = make_MLP_model_last_layer(X_train_val, y_train_val, hyper_params)
    method = lambda x, y, reg: predict_MLP(y)
    test_metrics = evaluate(hyper_params, method, data, item_propensity, sampled_matrix, test_set_eval = True)
    print("Training of the final model time: ", time.time() - final_time)
    log_end_epoch(hyper_params, test_metrics, 0, time.time() - start_time)
    start_time = time.time()

    return test_metrics


def train_KRR_basic(kernel, hyper_params, data):
    from KRR_model import make_kernelized_rr_forward
    from eval import evaluate

    # This just instantiates the function
    kernelized_rr_forward, kernel_fn = make_kernelized_rr_forward(kernel, hyper_params)
    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)

    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@10"
    best_metric, best_lamda, best_gamma = None, None, None

    # Validate on the validation-set
    if hyper_params['grid_search']:
          for lamda in hyper_params['grid_search_lamda']:
              hyper_params['lamda'] = lamda
              for gamma in hyper_params['grid_search_gamma']:
                  hyper_params['gamma'] = gamma
                  val_metrics = evaluate(hyper_params, kernelized_rr_forward,
                                         data, item_propensity, sampled_matrix)
                  if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric):
                      best_metric, best_lamda, best_gamma = val_metrics[VAL_METRIC], lamda, gamma

          # Return metrics with the best lamda on the test-set
          hyper_params['lamda'] = best_lamda
          hyper_params['gamma'] = best_gamma
          print(best_lamda)
          print(best_gamma)

    final_time = time.time()
    test_metrics = evaluate(hyper_params, kernelized_rr_forward, data, item_propensity,
                            sampled_matrix, test_set_eval = True)
    print("Training of the final model time: ", time.time() - final_time)

    log_end_epoch(hyper_params, test_metrics, 0, time.time() - start_time)
    start_time = time.time()

    return test_metrics


def check_MLP_variability(data, hyper_params):
    X_train_val = data.data['train_matrix'].todense() + data.data['val_matrix'].todense()
    y_train_val = data.data['train_matrix'].todense() + data.data['val_matrix'].todense()
    hyper_params['grid_search'] = False

    from MLP_model_keras import make_MLP_model
    from eval import evaluate
    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)

    times = []
    hrs = []
    ndcgs = []

    for i in range(10):
        final_time = time.time()
        predict_MLP = make_MLP_model(X_train_val, y_train_val, hyper_params)
        method = lambda x, y, reg: predict_MLP(y)
        test_metrics = evaluate(hyper_params, method, data, item_propensity, sampled_matrix, test_set_eval = True)
        times.append(time.time() - final_time)
        hrs.append(test_metrics['HR@10'])
        ndcgs.append(test_metrics['NDCG@10'])

    results = {
        "Mean time": np.array(times).mean(),
        "Variance time": np.array(times).std(),
        "Mean HR": np.array(hrs).mean(),
        "Variance HR": np.array(hrs).std(),
        "Mean NDCG": np.array(ndcgs).mean(),
        "Variance NDCG": np.array(ndcgs).std()
    }

    return results

def check_KRR_variability(neural_kernel, data, hyper_params):
    hyper_params['grid_search'] = False

    from KRR_model import make_kernelized_rr_forward
    from eval import evaluate

    # This just instantiates the function
    kernelized_rr_forward, kernel_fn = make_kernelized_rr_forward(neural_kernel, hyper_params)
    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)

    times = []
    hrs = []
    ndcgs = []

    for i in range(10):
        final_time = time.time()
        test_metrics = evaluate(hyper_params, kernelized_rr_forward, data, item_propensity, sampled_matrix, test_set_eval = True)
        times.append(time.time() - final_time)
        hrs.append(test_metrics['HR@10'])
        ndcgs.append(test_metrics['NDCG@10'])

    results = {
        "Mean time": np.array(times).mean(),
        "Variance time": np.array(times).std(),
        "Mean HR": np.array(hrs).mean(),
        "Variance HR": np.array(hrs).std(),
        "Mean NDCG": np.array(ndcgs).mean(),
        "Variance NDCG": np.array(ndcgs).std()
    }

    return results


def check_MLP_last_layer_variability(data, hyper_params):
    X_train_val = data.data['train_matrix'].todense() + data.data['val_matrix'].todense()
    y_train_val = data.data['train_matrix'].todense() + data.data['val_matrix'].todense()
    hyper_params['grid_search'] = False
    hyper_params['adam_lr'] = 0.001
    hyper_params['epochs'] = 100
    hyper_params['batch_size'] = 128
    hyper_params['reg'] = 10.0
    hyper_params['depth'] = 3

    from MLP_model_keras import make_MLP_model_last_layer
    from eval import evaluate
    sampled_matrix = data.sample_users(hyper_params['user_support']) # Random user sample

    # Used for computing the PSP-metric
    item_propensity = get_item_propensity(hyper_params, data)

    times = []
    hrs = []
    ndcgs = []

    for i in range(10):
        final_time = time.time()
        predict_MLP = make_MLP_model_last_layer(X_train_val, y_train_val, hyper_params)
        method = lambda x, y, reg: predict_MLP(y)
        test_metrics = evaluate(hyper_params, method, data, item_propensity, sampled_matrix, test_set_eval = True)
        times.append(time.time() - final_time)
        hrs.append(test_metrics['HR@10'])
        ndcgs.append(test_metrics['NDCG@10'])

    results = {
        "Mean time": np.array(times).mean(),
        "Variance time": np.array(times).std(),
        "Mean HR": np.array(hrs).mean(),
        "Variance HR": np.array(hrs).std(),
        "Mean NDCG": np.array(ndcgs).mean(),
        "Variance NDCG": np.array(ndcgs).std()
    }

    return results

