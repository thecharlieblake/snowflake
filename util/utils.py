import tensorflow as tf
import numpy as np
import scipy.signal
import pickle
from itertools import zip_longest


def gauss_selfKL_firstfixed(mu, logstd):
    """
        @brief:
            KL divergence with itself, holding first argument fixed
            Use stop gradient to cut the gradient flows
    """
    mu1, logstd1 = list(map(tf.stop_gradient, [mu, logstd]))
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)


# probability to take action x, given paramaterized guassian distribution
def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2 * logstd)
    gp = (
        -tf.square(x - mu) / (2 * var)
        - 0.5 * tf.math.log(tf.constant(2 * np.pi))
        - logstd
    )
    return tf.reduce_sum(gp, [1])


# KL divergence between two paramaterized guassian distributions
def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)

    kl = tf.reduce_sum(
        logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5
    )
    return kl


# Shannon entropy for a paramaterized guassian distributions
def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))
    return h


# hmm, interesting... they are using this to get the reward
def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cat_sample(prob_nk, seed=1234):

    npr = np.random.RandomState(seed)
    assert prob_nk.ndim == 2
    # prob_nk: batchsize x actions
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype="i")
    for (n, csprob_k, r) in zip(range(N), csprob_nk, npr.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(
        isinstance(a, int) for a in out
    ), "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(
        [tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0
    )


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # in numpy
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = 0.1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)  # the surrogate loss
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.compat.v1.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.compat.v1.assign(v, tf.reshape(theta[start : start + size], shape))
            )
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.op.eval(session=self.session)


class GetWeightsForPrefix(object):
    """
        @brief:
            call this function to get the weights in the policy network
    """

    def __init__(self, session, var_list, prefix="policy"):
        self.session = session
        all_ops = [var for var in var_list if prefix in var.name]
        self._logstd_ops = [var for var in all_ops if "logstd" in var.name]
        self._embedding_ops = [var for var in all_ops if "embedding" in var.name]
        self._all_others_ops = [
            var
            for var in all_ops
            if "logstd" not in var.name and "embedding" not in var.name
        ]

    def __call__(self, skip_logstd=False, skip_embedding=False):
        ret_ops = []
        if not skip_embedding:
            ret_ops.extend(self._embedding_ops)
        ret_ops.extend(self._all_others_ops)
        if not skip_logstd:
            ret_ops.extend(self._logstd_ops)
        return self.session.run(ret_ops)


class SetWeightsForPrefix(object):
    """
        @brief:
            call this function to set the weights in the policy network
            One thing interesting, we are using the placeholders to set
            the variables
    """

    def __init__(self, session, var_list, prefix="policy"):
        self.session = session
        self.policy_vars = [var for var in var_list if prefix in var.name]

        self.placeholders = {}

        self.logstd_assigns = []
        self.embedding_assigns = []
        self.all_other_assigns = []

        with tf.compat.v1.get_default_graph().as_default():
            for var in self.policy_vars:
                self.placeholders[var.name] = tf.compat.v1.placeholder(
                    tf.float32, var.get_shape()
                )
                curr_assign = tf.compat.v1.assign(var, self.placeholders[var.name])
                if "embedding" in var.name:
                    self.embedding_assigns.append(curr_assign)
                elif "logstd" in var.name:
                    self.logstd_assigns.append(curr_assign)
                else:
                    self.all_other_assigns.append(curr_assign)

    def __call__(self, weights, skip_logstd=False, skip_embedding=False, dump_weights=False):
        feed_dict = {}
        v_idx = 0
        for var in self.policy_vars:
            if skip_logstd and "logstd" in var.name:
                continue
            if skip_embedding and "embedding" in var.name:
                continue
            feed_dict[self.placeholders[var.name]] = weights[v_idx]
            v_idx += 1

        if dump_weights:
            dump = {var.name: weights[i] for i, var in enumerate(self.policy_vars)}
            with open(f"/opt/project/{dump_weights}", 'wb') as f:
                pickle.dump(dump, f)

        assign_ops = self.all_other_assigns
        if not skip_embedding:
            assign_ops.append(self.embedding_assigns)
            if not skip_logstd:
                assign_ops.append(self.logstd_assigns)
        self.session.run(assign_ops, feed_dict)


def xavier_initializer(self, shape):
    dim_sum = np.sum(shape)
    if len(shape) == 1:
        dim_sum += 1
    bound = np.sqrt(6.0 / dim_sum)
    return tf.random_uniform(shape, minval=-bound, maxval=bound)


def fully_connected(
    input_layer, input_size, output_size, weight_init, bias_init, scope, trainable
):
    with tf.compat.v1.variable_scope(scope):
        w = tf.compat.v1.get_variable(
            "w", [input_size, output_size], initializer=weight_init, trainable=trainable
        )
        b = tf.compat.v1.get_variable(
            "b", [output_size], initializer=bias_init, trainable=trainable
        )
    return tf.matmul(input_layer, w) + b


def apply_grads_for_task(
    tvars,
    blueprint_task_name,
    task_for_grads_name,
    optimizer,
    grads,
    average_logstd_grads=False,
    average_embedding_grads=True,
):

    common_params_grads = []
    logstd_grads = []
    embedding_grads = []

    # this will apply policy-related vars for the blueprint task using the grads from task_for_grads_name
    # this will still update context/logstd for the respected policies if they are not skipped

    vars_to_apply_grads = []
    for i in range(len(tvars[blueprint_task_name])):
        if not average_logstd_grads and "logstd" in tvars[blueprint_task_name][i].name:
            logstd_grads.append(
                (grads[task_for_grads_name][i], tvars[task_for_grads_name][i])
            )
        elif (
            not average_embedding_grads
            and "embedding" in tvars[blueprint_task_name][i].name
        ):
            embedding_grads.append(
                (grads[task_for_grads_name][i], tvars[task_for_grads_name][i])
            )
        else:
            common_params_grads.append(grads[task_for_grads_name][i])
            vars_to_apply_grads.append(tvars[blueprint_task_name][i])
    update_op = optimizer.apply_gradients(
        list(zip(common_params_grads, vars_to_apply_grads))
        + logstd_grads
        + embedding_grads
    )
    return update_op


def average_and_apply_grads_per_task(
    tvars,
    blueprint_task_name,
    optimizer,
    grads,
    training_tasks,
    mixing_ratio,
    average_logstd_grads=False,
    average_embedding_grads=True,
):

    averaged_grads = {"main": [], "node": [], "edge": [], "encoder": [], "decoder": []}
    logstd_grads = {"main": [], "node": [], "edge": [], "encoder": [], "decoder": []}
    embedding_grads = {"main": [], "node": [], "edge": [], "encoder": [], "decoder": []}

    graph_cat = {
        "node": "GRU_node",
        "edge": "prop_edge",
        "encoder": "MLP_embedding",
        "decoder": "MLP_out",
    }

    vars_to_apply_grads = {"main": [], "node": [], "edge": [], "encoder": [], "decoder": []}
    for i in range(len(tvars[blueprint_task_name])):
        var_category = "main"
        for section, keyword in graph_cat.items():
            if (keyword in tvars[blueprint_task_name][i].name) and (section in optimizer):
                var_category = section
        if not average_logstd_grads and "logstd" in tvars[blueprint_task_name][i].name:
            for tname in training_tasks:
                logstd_grads[var_category].append((grads[tname][i], tvars[tname][i]))
        elif (
            not average_embedding_grads
            and "embedding" in tvars[blueprint_task_name][i].name
        ):
            for tname in training_tasks:
                embedding_grads[var_category].append((grads[tname][i], tvars[tname][i]))
        else:
            cg = [grads[t][i] for t in training_tasks]
            if cg[0] is None:
                continue
            # rescale the grads along 0th dimension using the mixing-ratio weights
            # we need reshaping to do proper broadcasting for multidimensional arrays
            mrr = tf.reshape(mixing_ratio, (-1, *([1] * len(cg[0].shape))))
            averaged_grads[var_category].append(tf.reduce_mean(mrr * tf.stack(cg), axis=0))
            vars_to_apply_grads[var_category].append(tvars[blueprint_task_name][i])

    update_op = {}
    for var_category, opt in optimizer.items():
        update_op[var_category] = opt.apply_gradients(
            list(zip(averaged_grads[var_category], vars_to_apply_grads[var_category])) + logstd_grads[var_category] + embedding_grads[var_category]
        )

    return update_op, [t.name for t in vars_to_apply_grads['main'] + vars_to_apply_grads['node'] + vars_to_apply_grads['edge'] + vars_to_apply_grads['encoder'] + vars_to_apply_grads['decoder']]


def count_params(var_list):
    res = 0
    for variable in var_list:
        shape = variable.get_shape()
        res += np.prod(shape.as_list())
    return res


def means_diff_len(major_lists):
    return [
        np.mean([
            v
            for v
            in minor_elems
            if v is not None
        ])
        for minor_elems
        in zip_longest(*major_lists)
    ]


def nan_to_zero(A):
    return tf.where(tf.math.is_nan(A), tf.zeros_like(A), A)


# see https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
def pairwise_dist(A):
    r = tf.reduce_sum(A * A, 1)
    r = tf.reshape(r, [-1, 1])
    return r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)


def row_diff(A):
    return tf.reduce_mean(nan_to_zero(tf.sqrt(pairwise_dist(A))))


def col_diff(A):
    A_norm = A / tf.norm(A, ord=1, axis=0)
    return row_diff(tf.transpose(A_norm))