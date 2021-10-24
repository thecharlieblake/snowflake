from util.utils import fully_connected
from network.abstractions import Network, Baseline, build_mlp_output
import tensorflow as tf
import numpy as np


def normc_initializer(npr, std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = npr.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class MlpNetwork(Network):
    """
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    """

    def __init__(
        self,
        session,
        name_scope,
        input_size,
        output_size,
        ob_placeholder=None,
        weight_init_methods="normc",
        trainable=True,
        build_network_now=True,
        args=None,
    ):
        """
            @input:
                @ob_placeholder:
                    if this placeholder is not given, we will make one in this
                    class.

                @trainable:
                    If it is set to true, then the policy weights will be
                    trained. It is useful when the class is a subnet which
                    is not trainable

        """
        super().__init__(
            session,
            name_scope,
            input_size,
            output_size,
            ob_placeholder=ob_placeholder,
            weight_init_methods=weight_init_methods,
            trainable=trainable,
            args=args,
        )

        if self._init_method != "normc":
            raise NotImplementedError(
                "Only normc is implemented for MLP nets right now"
            )
        self._weight_init = normc_initializer(self._npr, 0.01)

        if build_network_now:
            with tf.compat.v1.get_default_graph().as_default():
                tf.compat.v1.set_random_seed(args.seed)
                self._build_model()

    def _build_model(self):

        self._iteration = tf.Variable(0, trainable=False, name="step")

        # two initializer
        weight_init = normc_initializer(self._npr)
        bias_init = tf.constant_initializer(0)

        with tf.compat.v1.variable_scope(self._name_scope):
            self._layer = self._input
            self._layer_input_size = self._input_size
            for i_layer in range(len(self._network_shape)):
                self._layer = fully_connected(
                    self._layer,
                    self._layer_input_size,
                    self._network_shape[i_layer],
                    weight_init,
                    bias_init,
                    "policy_" + str(i_layer),
                    trainable=self._trainable,
                )
                self._layer = tf.nn.tanh(self._layer)
                self._layer_input_size = self._network_shape[i_layer]

        self._set_var_list()

    def get_input_placeholder(self):
        return self._input


class MlpBaseline(MlpNetwork, Baseline):
    def __init__(
        self,
        session,
        name_scope,
        input_size,
        task_name,
        ob_placeholder=None,
        trainable=True,
        build_network_now=True,
        args=None,
    ):

        self._use_ppo = True
        self._ppo_clip = args.ppo_clip
        self._output_size = 1

        __class__.__mro__[1].__init__(
            self,
            session=session,
            name_scope=name_scope,
            input_size=input_size,
            output_size=self._output_size,
            ob_placeholder=ob_placeholder,
            trainable=trainable,
            build_network_now=build_network_now,
            args=args,
        )
        self._network_output = build_mlp_output(self)
        self._set_var_list()
        self._weight_init = None
        self._build_train_placeholders()
        self._build_loss()

    def predict(self, path):
        # prepare the obs into shape [Batch_size, ob_size] float32 variables
        obs = path["obs"].astype("float32")
        obs = obs.reshape(obs.shape[0], -1)
        return self._session.run(self._vpred, feed_dict={self._input: obs})

    def get_ob_placeholder(self):
        return self._input
