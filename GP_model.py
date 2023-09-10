import gpflow
import tensorflow as tf
import numpy as np
import time

# S - NNGP
# H - NTK

def NNGP_tfkernel(X, X2, d_max):
    if X2 is None:
        X_new = X
    else:
        n1 = tf.shape(X)[0]
        X_new = tf.concat([X, X2], 0)

    S0 = tf.matmul(X_new, X_new, transpose_b=True)
    i = 0

    loop_cond = lambda S, i: tf.less(i, d_max)

    def loop_body(S, i):
        L = tf.linalg.diag_part(S)
        P = tf.clip_by_value(tf.sqrt(tf.tensordot(L, L, axes=0)), clip_value_min=1e-9, clip_value_max=1e16)
        Sn = tf.clip_by_value(S / P, clip_value_min=-1.0, clip_value_max=1.0)
        S_new = (Sn * (np.pi - tf.acos(Sn)) + tf.sqrt(1.0 - Sn * Sn)) * P / 2.0 / np.pi

        return S_new, i+1

    S, _ = tf.while_loop(loop_cond, loop_body, [S0, i])

    if X2 is None:
      S_new = S
    else:
      S_new = S[:n1, n1:]

    return S_new


def NTK_tfkernel(X, X2, d_max):
    if X2 is None:
        X_new = X
    else:
        n1 = tf.shape(X)[0]
        X_new = tf.concat([X, X2], 0)

    S0 = tf.matmul(X_new, X_new, transpose_b=True)
    H0 = S0
    i = 0

    loop_cond = lambda S, H, i: tf.less(i, d_max)

    def loop_body(S, H, i):
        L = tf.linalg.diag_part(S)
        P = tf.clip_by_value(tf.sqrt(tf.tensordot(L, L, axes=0)), clip_value_min=1e-9, clip_value_max=1e100)
        Sn = tf.clip_by_value(S / P, clip_value_min=-1.0, clip_value_max=1.0)
        S_new = (Sn * (np.pi - tf.acos(Sn)) + tf.sqrt(1.0 - Sn * Sn)) * P / 2.0 / np.pi
        Hn = H * (np.pi - tf.acos(Sn)) / 2.0 / np.pi
        H_new = Hn + S_new

        return S_new, H_new, i+1

    _, H, _ = tf.while_loop(loop_cond, loop_body, [S0, H0, i])

    if X2 is None:
      H_new = H
    else:
      H_new = H[:n1, n1:]

    return H_new


class NeuralKernel(gpflow.kernels.Kernel):
    def __init__(self, neural_kernel, d_max):
        super().__init__(active_dims = None)
        self.neural_kernel = neural_kernel
        self.d_max = d_max

    def K(self, X, X2=None):
        if self.neural_kernel == 'ntk':
            return NTK_tfkernel(X, X2, self.d_max)
        elif self.neural_kernel == 'nngp':
            return NNGP_tfkernel(X, X2, self.d_max)

    def K_diag(self, X):
        if self.neural_kernel == 'ntk':
            K = NTK_tfkernel(X, X2=None, d_max=self.d_max)
            return tf.linalg.diag_part(K)
        elif self.neural_kernel == 'nngp':
            K = NNGP_tfkernel(X, X2=None, d_max=self.d_max)
            return tf.linalg.diag_part(K)

def invlink(f):
    return gpflow.likelihoods.Bernoulli().invlink(f).numpy()

def eval_GP(data, hyper_params, max_iter, test_set_eval=False):
    train_matrix = np.array(data.data['train_matrix'].todense())

    m = gpflow.models.VGP(
        (train_matrix, train_matrix),
        likelihood=gpflow.likelihoods.Bernoulli(),
        kernel=NeuralKernel('nngp', hyper_params['depth'])
    )

    start_time = time.time()
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss,
                 variables=m.trainable_variables,
                 options=dict(maxiter=max_iter),
                 method="L-BFGS-B")
    print(f"Training time for max_iter={max_iter}: ", time.time() - start_time)

    from eval import evaluate_batch
    INF = float(1e6)
    topk = 10
    metrics = {'HR@10': 0., 'NDCG@10': 0.}

    # Train positive set -- these items will be set to -infinity while prediction on the val/test set
    train_positive_list = list(map(list, data.data['train_positive_set']))
    if test_set_eval:
        for u in range(len(train_positive_list)): train_positive_list[u] += list(data.data['val_positive_set'][u])

    # Train positive interactions (in matrix form) as context for prediction on val/test set
    eval_context = data.data['train_matrix']
    if test_set_eval: eval_context += data.data['val_matrix']

    # What needs to be predicted
    to_predict = data.data['val_positive_set']
    if test_set_eval: to_predict = data.data['test_positive_set']

    logits, _ = invlink(m.predict_f(np.array(eval_context.todense())))

    # Marking train-set consumed items as negative INF
    for b in range(len(logits)): logits[b][train_positive_list[b]] = -INF

    indices = (-logits).argsort()[:, :topk].tolist()


    for b in range(len(logits)):
        num_pos = float(len(to_predict[b]))

        metrics['HR@{}'.format(topk)] += float(len(set(indices[b][:topk]) & to_predict[b])) / float(min(num_pos, topk))

        dcg, idcg = 0.0, 0.0
        for at, pred in enumerate(indices[b][:topk]):
            if pred in to_predict[b]: 
                dcg += 1.0 / np.log2(at + 2)
            if at < num_pos: 
                idcg += 1.0 / np.log2(at + 2)

        metrics['NDCG@{}'.format(topk)] += dcg / idcg

    return metrics


def grid_search_GP(data, hyper_params):
    # Evaluation
    start_time = time.time()

    VAL_METRIC = "HR@10"
    best_metric, best_max_iter = None, None
    metrics = []
    topk = 10

    # Validate on the validation-set
    for max_iter in hyper_params['grid_search_max_iter']:
        hyper_params['max_iter'] = max_iter
        val_metrics = eval_GP(data, hyper_params, max_iter)
        metrics.append(val_metrics['HR@10'])
        if (best_metric is None) or (val_metrics[VAL_METRIC] > best_metric):
            best_metric, best_max_iter = val_metrics[VAL_METRIC], max_iter

    # Return metrics with the best lamda on the test-set
    hyper_params['max_iter'] = best_max_iter
    final_time = time.time()
    test_metrics = eval_GP(data, hyper_params, best_max_iter, test_set_eval=True)
    for kind in ['HR', 'NDCG']:
        test_metrics['{}@{}'.format(kind, topk)] = round(
            float(100.0 * test_metrics['{}@{}'.format(kind, topk)]) / hyper_params['num_users'], 4
        )
    print("Training of the final model time: ", time.time() - final_time)
        
