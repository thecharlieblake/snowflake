from util import init_path
from util import model_saver
from util import logger
import numpy as np
import tensorflow as tf


def build_gnn_output(pi_net, node_info):
    action_mu_output = []

    pi_net._vib_decoder_kl = []
    for output_type, out_idx in node_info["output_type_dict"].items():
        hid_out = tf.reshape(
            tf.gather(
                pi_net._final_node_hidden,
                pi_net._output_type_idx[output_type],
                name="output_type_{}".format(output_type),
            ),
            (-1, *pi_net._final_node_hidden.shape[1:]),
        )

        if pi_net.args.output_block_conditioning is not None:
            context_feature_size = pi_net._hidden_dim
            if pi_net.args.output_block_conditioning == "hidden":
                context = tf.reshape(
                    hid_out, [pi_net._batch_size_int, -1, pi_net._hidden_dim],
                )
                context = tf.reduce_mean(context, axis=1)
                # tiling with this shape multiplexes the same hidden context n_nodes of times
                # this appends the same graph context to each of the nodes before outputting a policy
            elif pi_net.args.output_block_conditioning == "root":
                # use this to get the state of the root:
                context = tf.gather(
                    pi_net._final_node_hidden, pi_net._node_type_idx["root"],
                )
                context = tf.reshape(
                    context, [pi_net._batch_size_int, pi_net._hidden_dim],
                )
            elif pi_net.args.output_block_conditioning == "hidden&root":
                hid_context = tf.reshape(
                    hid_out, [pi_net._batch_size_int, -1, pi_net._hidden_dim],
                )
                # hid_context = tf.reduce_sum(hid_context, axis=1)
                hid_context = tf.reduce_mean(hid_context, axis=1)
                root_context = tf.gather(
                    pi_net._final_node_hidden, pi_net._node_type_idx["root"],
                )
                root_context = tf.reshape(
                    root_context, [pi_net._batch_size_int, pi_net._hidden_dim],
                )

                context = tf.concat([hid_context, root_context], axis=1)
                context_feature_size = pi_net._hidden_dim * 2  # concatenated -> *2
            else:
                raise NotImplementedError(
                    f"Unknown output block conditioning type: {pi_net.args.output_block_conditioning}"
                )
            context = tf.tile(context, (1, len(out_idx)))
            context = tf.reshape(context, [-1, context_feature_size])
            if pi_net.args.always_feed_node_features:
                enc_out = tf.reshape(
                    tf.gather(
                        pi_net._node_hidden[-1],
                        pi_net._output_type_idx[output_type],
                        name="output_type_{}".format(output_type),
                    ),
                    (-1, *pi_net._node_hidden[-1].shape[1:]),
                )
                context = tf.concat([context, enc_out], axis=1)

            output_mode_input = tf.concat([hid_out, context], axis=1)
        else:
            output_mode_input = hid_out

        if pi_net.args.vib_updater:
            action_mu_output.append(pi_net._MLP_Out[output_type](output_mode_input))
            pi_net._vib_decoder_kl.append(pi_net._MLP_Out[output_type].get_kl())
        else:
            action_mu_output.append(pi_net._MLP_Out[output_type](output_mode_input)[-1])
    action_mu_output = tf.concat(action_mu_output, 0)
    action_mu_output = tf.gather(
        action_mu_output, pi_net._inverse_output_type_idx, name="output_inverse",
    )

    return tf.reshape(action_mu_output, [pi_net._batch_size_int, -1])


from util.utils import fully_connected


def build_mlp_output(pi_net):
    bias_init = tf.constant_initializer(0)
    return fully_connected(
        pi_net._layer,
        pi_net._layer_input_size,
        pi_net._output_size,
        pi_net._weight_init,
        bias_init,
        pi_net._name_scope + "/policy_output",
        trainable=pi_net._trainable,
    )


class Network:
    def __init__(
        self,
        session,
        name_scope,
        input_size,
        output_size,
        weight_init_methods="orthogonal",
        ob_placeholder=None,
        trainable=True,
        placeholder_list=None,
        args=None,
    ):
        self._session = session
        self._name_scope = name_scope
        self._input_size = input_size
        self._output_size = output_size
        self._base_dir = init_path.get_abs_base_dir()
        self._init_method = weight_init_methods
        self._input = ob_placeholder
        self._trainable = trainable
        self._network_shape = args.network_shape
        self._seed = args.seed
        self._npr = np.random.RandomState(args.seed)
        self._placeholder_list = placeholder_list
        self._input_obs = ob_placeholder
        self.args = args

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = []
        self._freeze_var_list = []

        for var in tf.compat.v1.trainable_variables():
            if self._name_scope not in var.name:
                continue
            freeze = False
            for freeze_var in self.args.freeze_vars:
                if freeze_var in var.name:
                    freeze = True
                    break
            if freeze:
                self._freeze_var_list.append(var)
            else:
                self._trainable_var_list.append(var)

        all_var_list = [
            var
            for var in tf.compat.v1.global_variables()
            if self._name_scope in var.name or "batch_norm" in var.name
        ]
        all_var_list_order = [v for v in all_var_list if "embedding" in v.name] +\
                             [v for v in all_var_list if "embedding" not in v.name]

        self._all_var_list = all_var_list_order

    def get_var_list(self):
        return self._trainable_var_list, self._all_var_list

    def get_iteration_var(self):
        return self._iteration

    def load_checkpoint(
        self,
        ckpt_path,
        transfer_env="Nothing2Nothing",
        logstd_option="load",
        gnn_option_list=None,
        mlp_raw_transfer=False,
    ):
        # in this function, we check if the checkpoint has been loaded
        model_saver.load_tf_model(
            self._session,
            ckpt_path,
            tf_var_list=self._all_var_list,
            transfer_env=transfer_env,
            logstd_option=logstd_option,
            gnn_option_list=gnn_option_list,
            mlp_raw_transfer=mlp_raw_transfer,
        )

    def load_checkpoint_from_dict(
        self,
        checkpoint_dict,
        transfer_env="Nothing2Nothing",
        logstd_option="load",
        gnn_option_list=None,
        mlp_raw_transfer=False,
    ):
        model_saver.load_tf_model_from_dict(
            self._session,
            checkpoint_dict,
            tf_var_list=self._all_var_list,
            remove_var_str="trpo_agent_(policy|baseline)_.",
            transfer_env=transfer_env,
            logstd_option=logstd_option,
            gnn_option_list=gnn_option_list,
            mlp_raw_transfer=mlp_raw_transfer,
        )

    def save_checkpoint(self, ckpt_path):
        model_saver.save_tf_model(self._session, ckpt_path, self._all_var_list)

    def checkpoint_dict(self):
        return model_saver.tf_model_to_dict(self._session, self._all_var_list)


class Baseline:
    def get_vf_loss(self):
        return self._loss

    def get_target_return_placeholder(self):
        return self._target_returns

    def get_vpred_placeholder(self):
        return self._vpred

    def _build_loss(self):
        """
            @brief: note that the value clip idea is also used here!
        """
        if self.args.vf_loss_type == "huber":
            self._loss = tf.compat.v1.losses.huber_loss(
                self._target_returns, self._vpred
            )
        else:
            self._loss = tf.reduce_mean(tf.square(self._vpred - self._target_returns))

    def _build_train_placeholders(self):
        # the names are defined in policy network, reshape to [None]

        self._vpred = tf.reshape(self._network_output, [-1])

        self._target_returns = tf.compat.v1.placeholder(
            tf.float32, shape=[None], name="target_returns"
        )
