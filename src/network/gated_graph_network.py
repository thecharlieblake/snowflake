import tensorflow as tf
import numpy as np
from util import logger
from util import utils
from network import nn_cells as nn
from network.abstractions import Network, Baseline, build_gnn_output


class GatedGraphNetwork(Network):
    """
        @brief:
            Gated Graph Sequence Neural Networks.
            Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015.
            arXiv preprint arXiv:1511.05493.
    """

    def __init__(
        self,
        session,
        name_scope,
        input_size,
        output_size,
        task_name,
        weight_init_methods="orthogonal",
        ob_placeholder=None,
        trainable=True,
        build_network_now=True,
        placeholder_list=None,
        node_info=None,
        inference_mode=False,
        args=None,
    ):
        """
            @input: the same as the ones defined in "policy_network"
        """
        self._node_update_method = args.node_update_method
        self._task_name = task_name
        self._node_info = node_info

        super().__init__(
            session,
            name_scope,
            input_size,
            output_size,
            ob_placeholder=ob_placeholder,
            trainable=trainable,
            placeholder_list=placeholder_list,
            weight_init_methods=args.init_method,
            args=args,
        )
        self._root_connection_option = args.root_connection_option
        self._num_prop_steps = args.gnn_num_prop_steps
        self._gnn_node_option = args.gnn_node_option
        self._gnn_output_option = args.gnn_output_option
        self._gnn_embedding_option = args.gnn_embedding_option
        self._hidden_dim = args.gnn_node_hidden_dim
        self._input_feat_dim = args.gnn_input_feat_dim
        self.inference_mode = inference_mode

        assert self._input_feat_dim == self._hidden_dim
        logger.debug("Network shape is {}".format(self._network_shape))

        if self.args.record_oversmoothing:
            self.oversmoothing_stats = {
                layer: {
                    node_type: {"mean_r": [], "mean_z": [], "row-diff": [], "col-diff": []}
                    for node_type in self._node_info["node_type_dict"] if node_type != "root"
                } for layer in range(self._num_prop_steps)
            }

        if build_network_now:
            self._build_model()

    def get_input_obs_placeholder(self):
        return self._input_obs

    def get_gnn_idx_placeholder(self):
        """
            @brief: return the placeholders to the agent to construct feed dict
        """
        return (
            self._receive_idx,
            self._send_idx,
            self._node_type_idx,
            self._inverse_node_type_idx,
            self._output_type_idx,
            self._inverse_output_type_idx,
            self._batch_size_int,
        )

    def _build_model(self):
        """
            @brief: everything about the network goes here
        """
        with tf.compat.v1.get_default_graph().as_default():
            tf.compat.v1.set_random_seed(self._seed)

            # record the iteration count
            self._iteration = tf.Variable(0, trainable=False, name="step")

            # self._io_size_check()

            # prepare the network's input and output
            self._prepare()

            # define the network here
            self._build_network_weights()
            self._build_network_graph()
            self._build_output_graph()

            self._set_var_list() # all *unique* norm moving variables

    def get_input_parameters_placeholder(self):
        return self._input_parameters

    def _build_baseline_loss(self):
        self._baseline_loss = tf.reduce_mean(
            tf.square(self._vpred - self._target_returns)
        )

    def _build_baseline_train_placeholders(self):
        self._target_returns = tf.compat.v1.placeholder(
            tf.float32, shape=[None], name="target_returns"
        )

    def _build_network_weights(self):
        """
            @brief: build the network
        """

        # step 1: build the weight parameters (mlp, gru)
        with tf.compat.v1.variable_scope(self._name_scope):
            # step 1_1: build the embedding matrix (mlp)
            # tensor shape (None, para_size) --> (None, input_dim - ob_size)
            assert self._input_feat_dim % 2 == 0
            if "noninput" not in self._gnn_embedding_option:
                self._MLP_embedding = {
                    node_type: nn.VIB(self._input_feat_dim // 2, f'{self._name_scope}/MLP_embedding_node_type_{node_type}') if self.args.vib_updater
                    else nn.MLP(
                        [
                            self._input_feat_dim // 2,
                            self._node_info["para_size_dict"][node_type],
                        ],
                        init_method=self._init_method,
                        act_func=["tanh"] * 1,  # one layer at most
                        add_bias=True,
                        scope="MLP_embedding_node_type_{}".format(node_type),
                        layernorm=self.args.layernorm,
                        batchnorm=self.args.batchnorm,
                        renorm=self.args.renorm,
                        norm_position=self.args.norm_position,
                        record_activation=self.args.record_mlp_activation,
                        inference_mode=self.inference_mode,
                    )
                    for node_type in self._node_info["node_type_dict"]
                    if self._node_info["ob_size_dict"][node_type] > 0
                }
                self._MLP_embedding.update(
                    {
                        node_type: nn.VIB(self._input_feat_dim - int(self._input_feat_dim * self.args.observation_share) // 2,
                                          f'{self._name_scope}/MLP_embedding_node_type_{node_type}') if self.args.vib_updater
                        else nn.MLP(
                            [
                                self._input_feat_dim
                                - int(
                                    self._input_feat_dim * self.args.observation_share
                                ),
                                self._node_info["para_size_dict"][node_type],
                            ],
                            init_method=self._init_method,
                            act_func=["tanh"] * 1,  # one layer at most
                            add_bias=True,
                            scope="MLP_embedding_node_type_{}".format(node_type),
                            layernorm=self.args.layernorm,
                            batchnorm=self.args.batchnorm,
                            renorm=self.args.renorm,
                            norm_position=self.args.norm_position,
                            record_activation=self.args.record_mlp_activation,
                            inference_mode=self.inference_mode,
                        )
                        for node_type in self._node_info["node_type_dict"]
                        if self._node_info["ob_size_dict"][node_type] == 0
                    }
                )
            else:
                embedding_vec_size = (
                    max(
                        np.reshape(
                            [
                                max(self._node_info["node_parameters"][i_key])
                                for i_key in self._node_info["node_parameters"]
                            ],
                            [-1],
                        )
                    )
                    + 1
                )
                embedding_vec_size = int(embedding_vec_size)
                self._embedding_variable = {}
                # out = np.eye(embedding_vec_size).astype(np.float32)
                out = self._npr.randn(
                    embedding_vec_size,
                    self._input_feat_dim
                    - int(self._input_feat_dim * self.args.observation_share),
                ).astype(np.float32)
                out *= 1.0 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))

                self._embedding_variable[False] = tf.Variable(
                    out, name="embedding_HALF", trainable=self.args.learnable_context
                )

                if np.any(
                    [
                        node_size == 0
                        for _, node_size in self._node_info["ob_size_dict"].items()
                    ]
                ):
                    out = self._npr.randn(
                        embedding_vec_size, self._input_feat_dim
                    ).astype(np.float32)
                    out *= 1.0 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                    self._embedding_variable[True] = tf.Variable(
                        out,
                        name="embedding_FULL",
                        trainable=self.args.learnable_context,
                    )
            # step 1_2: build the ob mapping matrix (mlp)
            # tensor shape (None, para_size) --> (None, input_dim - ob_size)
            self._MLP_ob_mapping = {
                node_type: nn.VIB(int(self._input_feat_dim * self.args.observation_share), f'{self._name_scope}/MLP_embedding_node_type_{node_type}') if self.args.vib_updater
                    else nn.MLP(
                    [
                        int(self._input_feat_dim * self.args.observation_share),
                        # self._input_feat_dim - embedding_vec_size,
                        self._node_info["ob_size_dict"][node_type],
                    ],
                    init_method=self._init_method,
                    act_func=["tanh"] * 1,  # one layer at most
                    add_bias=True,
                    scope="MLP_embedding_node_type_{}".format(node_type),
                    norm_position=self.args.norm_position,
                    record_activation=self.args.record_mlp_activation,
                    inference_mode=self.inference_mode,
                )
                for node_type in self._node_info["node_type_dict"]
                if self._node_info["ob_size_dict"][node_type] > 0
            }

        with tf.compat.v1.variable_scope(self._name_scope):
            # step 1_4: build the mlp for the propogation between nodes

            MLP_prop_shape = (
                self._network_shape + [self._hidden_dim] + [self._hidden_dim]
            )
            if self.args.mind_the_receiver:
                # concate the receiver features when calculating the message
                # fortunately, at this stage, all the features are in hidden dim, and we do not need to pad one or another since the size is the same for all
                MLP_prop_shape[-1] *= 2

            #if self.args.always_feed_node_features:
            #    MLP_prop_shape[-1] *= 2

            self._MLP_prop, self._Node_update = {}, {}
            if self.args.no_sharing_between_layers:
                for tt in range(self._num_prop_steps):
                    if self.args.identity_edge_updater == 0:
                        self._MLP_prop[tt] = self.gen_edge_updater(MLP_prop_shape, self.args.vib_updater)
                    self._Node_update[tt] = self.gen_node_updater()
            else:
                if self.args.identity_edge_updater == 0:
                    edge_updater = self.gen_edge_updater(MLP_prop_shape, self.args.vib_updater)
                node_updater = self.gen_node_updater()
                for tt in range(self._num_prop_steps):
                    if self.args.identity_edge_updater == 0:
                        self._MLP_prop[tt] = edge_updater
                    self._Node_update[tt] = node_updater

            logger.debug(
                "building prop mlp for edge type {}".format(
                    self._node_info["edge_type_list"]
                )
            )

            logger.debug(
                "building node update function for node type {}".format(
                    self._node_info["node_type_dict"]
                )
            )

            # step 1_6: the mlp for the mu of the actions
            # 2 for std
            nshape = (
                self._network_shape
                if not self.args.reduced_decoder
                else self._network_shape[1:]
            )
            MLP_out_shape = (
                nshape + [1] + [self._hidden_dim]
            )  # (l_1, l_2, ..., l_o, l_i)
            MLP_out_act_func = ["tanh"] * (len(MLP_out_shape) - 1)
            MLP_out_act_func[-1] = None
            if self.args.no_decoder:
                MLP_out_shape = MLP_out_shape[-2:]
                MLP_out_act_func = MLP_out_act_func[-2:]
            self._build_output_weights(MLP_out_shape, MLP_out_act_func)

    def gen_node_updater(self):
        # step 1_5: build the node update function for each node type
        if self._node_update_method == "GRU":
            in_size = self._hidden_dim
            if self.args.mind_the_receiver and self.args.identity_edge_updater:
                in_size *= 2
            if self.args.global_context is not None:
                if self.args.global_context == "hidden&root":
                    in_size += 2 * self._hidden_dim
                else:
                    in_size += self._hidden_dim
            if self.args.always_feed_node_features:
                in_size += self._hidden_dim
            if self.args.one_updater_for_all:
                node_update = nn.GRU(
                    in_size,
                    self._hidden_dim,
                    init_method=self._init_method,
                    scope="GRU_node_{}".format("common"),
                    stop_hidden_grad=self.args.stop_hidden_grad,
                )
                return {
                    i_node_type: node_update
                    for i_node_type in self._node_info["node_type_dict"]
                }
            else:
                return {
                    i_node_type: nn.GRU(
                        in_size,
                        self._hidden_dim,
                        init_method=self._init_method,
                        scope="GRU_node_{}".format(i_node_type),
                        stop_hidden_grad=self.args.stop_hidden_grad,
                    )
                    for i_node_type in self._node_info["node_type_dict"]
                }
        else:
            assert self._node_update_method == "MLP"
            hidden_MLP_update_shape = (
                self._network_shape[1:]
                if self.args.reduced_core
                else self._network_shape
            )
            if self.args.one_updater_for_all:
                node_update = nn.MLPU(
                    message_dim=self._hidden_dim
                    if not self.args.always_feed_node_features
                    else self._hidden_dim * 2,
                    embedding_dim=self._hidden_dim,
                    hidden_shape=hidden_MLP_update_shape,
                    init_method=self._init_method,
                    act_func_type="tanh",
                    add_bias=True,
                    scope="MLPU_node_{}".format("common"),
                )
                return {
                    i_node_type: node_update
                    for i_node_type in self._node_info["node_type_dict"]
                }
            else:
                return {
                    i_node_type: nn.MLPU(
                        message_dim=self._hidden_dim
                        if not self.args.always_feed_node_features
                        else self._hidden_dim * 2,
                        embedding_dim=self._hidden_dim,
                        hidden_shape=hidden_MLP_update_shape,
                        init_method=self._init_method,
                        act_func_type="tanh",
                        add_bias=True,
                        scope="MLPU_node_{}".format(i_node_type),
                    )
                    for i_node_type in self._node_info["node_type_dict"]
                }

    def gen_edge_updater(self, MLP_prop_shape, vib=False):
        return {
            i_edge: nn.VIB(MLP_prop_shape[-2], f'{self._name_scope}/MLP_prop_edge_{i_edge}') if vib
            else nn.MLP(
                MLP_prop_shape,
                init_method=self._init_method,
                act_func=[self.args.edge_act_fn] * (len(MLP_prop_shape) - 1),
                add_bias=True,
                scope=f"MLP_prop_edge_{i_edge}",
                layernorm=self.args.layernorm,
                batchnorm=self.args.batchnorm,
                renorm=self.args.renorm,
                norm_position=self.args.norm_position,
                record_activation=self.args.record_mlp_activation,
                inference_mode=self.inference_mode,
            )
            for i_edge in self._node_info["edge_type_list"]
        }

    def cosine_distance(self, a, b):
        na = tf.nn.l2_normalize(a, 2)
        nb = tf.nn.l2_normalize(b, 2)
        return 1 - tf.matmul(na, nb, transpose_b=True)

    def hid2mad(self, hid_states):
        hid_reshaped = tf.reshape(
            hid_states, (self._batch_size_int, -1, self._hidden_dim)
        )  # now batch is the first dim
        # compute cosine distance of each with each inside a batch
        mads = self.cosine_distance(hid_reshaped, hid_reshaped)

        # compute average cosine distance in a row not counting zeros
        nonzero_elems_in_rows = tf.compat.v1.count_nonzero(
            mads, axis=2, dtype=tf.float32
        )
        mads = tf.reduce_sum(mads, axis=2) / nonzero_elems_in_rows

        # compute average of the step above also removing zeros if needed
        nonzero_elems_in_cols = tf.compat.v1.count_nonzero(
            mads, axis=1, dtype=tf.float32
        )
        mads = tf.reduce_sum(mads, axis=1) / nonzero_elems_in_cols

        # return average MAD across the batch
        return tf.reduce_mean(mads)

    def compute_MAD(self):
        self.mad = [self.hid2mad(self._input_node_hidden)]
        for tt in range(self._num_prop_steps):
            self.mad.append(self.hid2mad(self._node_hidden[tt]))

    def _build_network_graph(self):
        # step 2: gather the input_feature from obs and node parameters
        if "noninput" not in self._gnn_embedding_option:
            self._input_embedding = {
                node_type: self._MLP_embedding[node_type](
                    self._input_parameters[node_type]
                )[-1]
                for node_type in self._node_info["node_type_dict"]
            }
        else:
            self._input_embedding = {
                # _embedding_variable is a (#embedding_type, dim/2) matrix. This gets the relevant slice of that
                # matrix for the corresponding embedding, returning a (batch_zs, dim/2) root matrix
                # and a (batch_sz * #joints, dim/2) joint matrix
                node_type: tf.gather(
                    self._embedding_variable[  # for 6-centipede: (6, 32)
                        self._node_info["ob_size_dict"][node_type] == 0
                    ],
                    tf.reshape(self._input_parameters[node_type], [-1]),  # batch_sz (node_type=root), batch_sz * #joints (joint)
                )
                for node_type in self._node_info["node_type_dict"]
            }

        if self.args.vib_updater:
            self._ob_feat = {
                node_type: self._MLP_ob_mapping[node_type](self._input_obs[node_type])  # No [-1]
                for node_type in self._node_info["node_type_dict"]
                if self._node_info["ob_size_dict"][node_type] > 0
            }
            self._vib_encoder_kl = [
                self._MLP_ob_mapping[node_type].get_kl()
                for node_type in self._node_info["node_type_dict"]
                if self._node_info["ob_size_dict"][node_type] > 0
            ]
        else:
            # self._input_obs[node_type] is joint: (batch_sz * #joints, obs_for_joint)
            # root *bypasses* encoder!
            # output: root:(batch_sz, dim/2), joint: (batch_sz * #joints, dim/2)
            self._ob_feat = {
                node_type: self._MLP_ob_mapping[node_type](self._input_obs[node_type])[-1]
                for node_type in self._node_info["node_type_dict"]
                if self._node_info["ob_size_dict"][node_type] > 0
            }

        self._ob_feat.update(
            {
                node_type: self._input_obs[node_type]
                for node_type in self._node_info["node_type_dict"]
                if self._node_info["ob_size_dict"][node_type] == 0
            }
        )

        # root:(batch_sz, dim), joint: (batch_sz * #joints, dim)
        self._input_feat = {  # shape: [node_num, embedding_size + ob_size]
            node_type: tf.concat(
                [self._input_embedding[node_type], self._ob_feat[node_type]], axis=1
            )
            for node_type in self._node_info["node_type_dict"]
        }
        self._input_node_hidden = self._input_feat
        # (batch_sz * (#joints + 1), dim)
        self._input_node_hidden = tf.concat(
            [
                self._input_node_hidden[node_type]
                for node_type in self._node_info["node_type_dict"]
            ],
            axis=0,
        )

        # Reshuffles to get all roots first, followed by all joints in batch sequence
        pre_shape = self._input_node_hidden.shape
        self._input_node_hidden = tf.gather(  # get node order into graph order
            self._input_node_hidden,
            self._inverse_node_type_idx,
            name="get_order_back_gather_init",
        )
        self._input_node_hidden = tf.reshape(
            self._input_node_hidden, (-1, *pre_shape[1:])
        )

        # step 3: unroll the propogation
        self._node_hidden = [None] * (self._num_prop_steps + 1)
        self._node_hidden[-1] = self._input_node_hidden  # trick to use [-1]
        self._prop_msg = [None] * self._node_info["num_edge_type"]
        self._vib_edge_kl = [[0 for _ in range(self._node_info["num_edge_type"])] for _ in range(self._num_prop_steps)]
        self._vib_edge_stats = {
            stat: [[0 for _ in range(self._node_info["num_edge_type"])] for _ in range(self._num_prop_steps)]
            for stat in ["mean_mu", "var_mu", "mean_rho", "var_rho"]
        }

        # self._node_info['relation_matrix']
        with tf.compat.v1.variable_scope(self._name_scope):
            for tt in range(self._num_prop_steps):
                ee = 0
                for i_edge_type in self._node_info["edge_type_list"]:
                    if self.args.mind_the_receiver:
                        # use the receiver features when sending a message
                        node_activate = tf.concat(
                            [
                                tf.gather(
                                    self._node_hidden[tt - 1], self._send_idx[i_edge_type],
                                ),
                                tf.gather(self._node_hidden[tt - 1], self._receive_idx,),
                            ],
                            1,
                            name="edge_id_{}_prop_steps_{}".format(i_edge_type, tt),
                        )
                        if self.args.always_feed_node_features:
                            raise NotImplementedError(
                                "always_feed_node_features is not implemented together with mind the receiver yet, but this is easy to do"
                            )
                    else:
                        pre_shape = self._node_hidden[tt - 1].shape
                        node_activate = tf.gather(
                            self._node_hidden[tt - 1],
                            self._send_idx[i_edge_type],
                            name="edge_id_{}_prop_steps_{}".format(i_edge_type, tt),
                        )
                        node_activate = tf.reshape(node_activate, (-1, *pre_shape[1:]))

                    if self.args.identity_edge_updater == 0:
                        if self.args.vib_updater:
                            self._prop_msg[ee] = self._MLP_prop[tt][i_edge_type](node_activate)
                            self._vib_edge_kl[tt][ee] = self._MLP_prop[tt][i_edge_type].get_kl()
                            self._vib_edge_stats["mean_mu"][tt][ee] = self._MLP_prop[tt][i_edge_type].mean_mu
                            self._vib_edge_stats["var_mu"][tt][ee] = self._MLP_prop[tt][i_edge_type].var_mu
                            self._vib_edge_stats["mean_rho"][tt][ee] = self._MLP_prop[tt][i_edge_type].mean_rho
                            self._vib_edge_stats["var_rho"][tt][ee] = self._MLP_prop[tt][i_edge_type].var_rho
                        else:
                            self._prop_msg[ee] = self._MLP_prop[tt][i_edge_type](node_activate)[-1]
                    else:
                        # Use this if want an identity edge updater, having a several layer MLP seems to be an overkill here
                        self._prop_msg[ee] = node_activate
                    ee += 1

                # aggregate messages
                concat_msg = tf.concat(self._prop_msg, 0)
                self.concat_msg = concat_msg

                # this is average aggregation sum/counters
                if self.args.msg_aggregation == "mean":
                    message = tf.math.unsorted_segment_mean(
                        concat_msg,
                        self._receive_idx,
                        self._node_info["num_nodes"] * self._batch_size_int,
                    )
                elif self.args.msg_aggregation == "sum":
                    message = tf.math.unsorted_segment_sum(
                        concat_msg,
                        self._receive_idx,
                        self._node_info["num_nodes"] * self._batch_size_int,
                    )
                elif self.args.msg_aggregation == "max":
                    message = tf.math.unsorted_segment_max(
                        concat_msg,
                        self._receive_idx,
                        self._node_info["num_nodes"] * self._batch_size_int,
                    )
                else:
                    raise NotImplementedError("Unknown message aggregation function")

                message = tf.reshape(message, (-1, *concat_msg.shape[1:]))

                # update the hidden states via GRU
                new_state = []
                for i_node_type, ntv in self._node_info["node_type_dict"].items():
                    msg_for_type = tf.gather(
                        message,
                        self._node_type_idx[i_node_type],
                        name="GRU_message_node_type_{}_prop_step_{}".format(
                            i_node_type, tt
                        ),
                    )
                    msg_for_type = tf.reshape(msg_for_type, (-1, *message.shape[1:]))
                    hid_for_type = tf.gather(
                        self._node_hidden[tt - 1],
                        self._node_type_idx[i_node_type],
                        name="GRU_feat_node_type_{}_prop_steps_{}".format(i_node_type, tt),
                    )
                    hid_for_type = tf.reshape(
                        hid_for_type, (-1, *self._node_hidden[tt - 1].shape[1:])
                    )
                    input_feat_for_type = tf.gather(
                        self._node_hidden[-1],
                        self._node_type_idx[i_node_type],
                        name="GRU_feat_node_type_{}_prop_steps_{}".format(i_node_type, tt),
                    )
                    if self.args.global_context is not None:
                        context_feature_size = self._hidden_dim
                        if self.args.global_context == "hidden":
                            hid_state_for_context = tf.reshape(
                                hid_for_type, [self._batch_size_int, len(ntv), -1],
                            )
                            context = tf.reduce_mean(hid_state_for_context, axis=1)
                        elif self.args.global_context == "input":
                            # -1 stores the initial features for the nodes
                            glob_inpt = tf.gather(
                                self._node_hidden[-1],
                                self._node_type_idx[i_node_type],
                                name="GRU_feat_node_type_{}_prop_steps_{}".format(
                                    i_node_type, tt
                                ),
                            )
                            hid_state_for_context = tf.reshape(
                                glob_inpt, [self._batch_size_int, len(ntv), -1],
                            )
                            context = tf.reduce_mean(hid_state_for_context, axis=1)
                        elif self.args.global_context == "root":
                            hid_state_for_context = tf.gather(
                                self._node_hidden[tt - 1],
                                self._node_type_idx["root"],
                                name="GRU_feat_node_type_{}_prop_steps_{}".format(
                                    i_node_type, tt
                                ),
                            )
                            # 1 is here because there is only one root node in the graph
                            context = tf.reshape(
                                hid_state_for_context, [self._batch_size_int, -1]
                            )
                            context_feature_size = self._hidden_dim
                        elif self.args.global_context == "hidden&root":
                            hid_context = tf.reshape(
                                hid_for_type,
                                [self._batch_size_int, len(ntv), self._hidden_dim],
                            )
                            hid_context = tf.reduce_mean(hid_context, axis=1)

                            root_context = tf.gather(
                                self._node_hidden[tt - 1],
                                self._node_type_idx["root"],
                                name="GRU_feat_node_type_{}_prop_steps_{}".format(
                                    i_node_type, tt
                                ),
                            )
                            # 1 is here because there is only one root node in the graph
                            root_context = tf.reshape(
                                root_context, [self._batch_size_int, self._hidden_dim]
                            )
                            context = tf.concat([hid_context, root_context], axis=1)
                            context_feature_size = self._hidden_dim * 2
                        else:
                            raise NotImplementedError(
                                f"{self.args.global_context} global type has not been implemented"
                            )

                        context = tf.tile(context, (1, len(ntv)))
                        context = tf.reshape(
                            context, [self._batch_size_int * len(ntv), context_feature_size]
                        )


                        if self.args.always_feed_node_features:
                            node_upd_in = tf.concat(
                                [msg_for_type, context, input_feat_for_type], -1
                            )
                        else:
                            node_upd_in = tf.concat([msg_for_type, context], -1)
                    else:
                        if self.args.always_feed_node_features:
                            node_upd_in = tf.concat([msg_for_type, input_feat_for_type], -1)
                        else:
                            node_upd_in = msg_for_type

                    ns = self._Node_update[tt][i_node_type](node_upd_in, hid_for_type)
                    new_state.append(ns) # 1,16,64 is batch, #nodes, hid_dim

                    if self.args.record_oversmoothing and i_node_type != "root":
                        self.oversmoothing_stats[tt][i_node_type]["mean_r"] = self._Node_update[tt][i_node_type].mean_r
                        self.oversmoothing_stats[tt][i_node_type]["mean_z"] = self._Node_update[tt][i_node_type].mean_z
                        ns_per_batch = tf.reshape(ns, (self._batch_size_int, -1, self._hidden_dim))
                        self.oversmoothing_stats[tt][i_node_type]["row-diff"] = tf.reduce_mean(tf.map_fn(utils.row_diff, ns_per_batch))
                        self.oversmoothing_stats[tt][i_node_type]["col-diff"] = tf.reduce_mean(tf.map_fn(utils.col_diff, ns_per_batch))

                if tt == self._num_prop_steps - 1:
                    self.output_hidden_state = {
                        node_type: new_state[i_id]
                        for i_id, node_type in enumerate(self._node_info["node_type_dict"])
                    }
                new_state = tf.concat(new_state, 0)
                self._node_hidden[tt] = tf.gather(
                    new_state,
                    self._inverse_node_type_idx,
                    name="get_order_back_gather_prop_steps_{}".format(tt),
                )
                self._node_hidden[tt] = tf.reshape(
                    self._node_hidden[tt], (-1, *new_state.shape[1:])
                )

        # for each layer for each node type
        if self.args.record_mlp_activation:
            self.mlp_stats = {
                i_edge_type: {
                    tt: {
                        lyr: {
                            stat_name: tf.math.reduce_mean(data)
                            for stat_name, data in stats.items()
                        }
                        for lyr, stats in self._MLP_prop[tt][i_edge_type].layer_stats.items()
                    }
                    for tt in range(self._num_prop_steps)
                }
                for i_edge_type in self._node_info["edge_type_list"]
            }
        if self.args.batchnorm:
            self.norm_update_ops = []
            self.norm_moving_averages = {}
            for i_edge_type in self._node_info["edge_type_list"]:
                self.norm_moving_averages[i_edge_type] = {}
                for tt in range(self._num_prop_steps):
                    self.norm_moving_averages[i_edge_type][tt] = {}
                    for norm, norm_name in zip(self._MLP_prop[tt][i_edge_type].norm, ["pre",0,1,2,3,4,5,6,7,8,9]):
                        if len(norm.updates) > 0:
                            self.norm_update_ops += norm.updates
                            self.norm_moving_averages[i_edge_type][tt][norm_name] = {
                                "mean": tf.math.reduce_mean(norm.moving_mean),
                                "var": tf.math.reduce_mean(norm.moving_variance),
                            }
                    if not self.args.no_sharing_between_layers:
                        break

        self._final_node_hidden = self._node_hidden[-2]

    def get_num_nodes(self):
        return self._node_info["num_nodes"]

    def _io_error_message(self):
        logger.error(
            "The output and input size is not matched!"
            + " ({}, {}) vs. ({}, {})".format(
                self._input_size,
                self._output_size,
                self._node_info["debug_info"]["ob_size"],
                self._node_info["debug_info"]["action_size"],
            )
        )


class GatedGraphBaseline(GatedGraphNetwork, Baseline):
    def __init__(
        self,
        session,
        name_scope,
        input_size,
        task_name,
        placeholder_list,
        weight_init_methods="orthogonal",
        ob_placeholder=None,
        trainable=True,
        build_network_now=True,
        args=None,
        node_info=None,
    ):

        root_connection_option = args.root_connection_option
        root_connection_option = root_connection_option.replace("Rn", "Ra")
        root_connection_option = root_connection_option.replace("Rb", "Ra")
        assert (
            "Rb" in root_connection_option or "Ra" in root_connection_option
        ), logger.error(
            "Root connection option {} invalid for baseline".format(
                root_connection_option
            )
        )

        self._ppo_clip = args.ppo_clip

        # 1 is the gated graph net, 2 is the baseline
        __class__.__mro__[1].__init__(
            self,
            session,
            name_scope,
            input_size,
            task_name=task_name,
            output_size=1,
            placeholder_list=placeholder_list,
            weight_init_methods=weight_init_methods,
            ob_placeholder=ob_placeholder,
            trainable=trainable,
            build_network_now=build_network_now,
            args=args,
            node_info=node_info,
        )
        self._build_train_placeholders()
        self._build_loss()

    def _prepare(self):
        (
            self._receive_idx,
            self._send_idx,
            self._node_type_idx,
            self._inverse_node_type_idx,
            self._output_type_idx,
            self._inverse_output_type_idx,
            self._batch_size_int,
            self._input_obs,
            self._input_parameters,
        ) = self._placeholder_list

    def predict(self, feed_dict):
        """
            @brief:
                generate the baseline function.
        """
        baseline = self._session.run(self._vpred, feed_dict=feed_dict)
        baseline = baseline.reshape([-1])
        return baseline

    def _io_size_check(self):
        """
            @brief:
                check if the environment's input size and output size is matched
                with the one parsed from the mujoco xml file
        """
        is_io_matched = (
            self._input_size == self._node_info["debug_info"]["ob_size"]
            and self._output_size == 1
        )

        assert is_io_matched, self._io_error_message()

    def _build_output_weights(self, MLP_out_shape, MLP_out_act_func):
        MLP_out_shape[-2] = 1  # policy uses 2, with the second dimension for logstd
        self._MLP_Out = nn.VIB(MLP_out_shape[-2], f'{self._name_scope}/MLP_out') if self.args.vib_updater else nn.MLP(
            MLP_out_shape,
            init_method=self._init_method,
            act_func=MLP_out_act_func,
            add_bias=True,
            scope="MLP_out",
            layernorm=False,
            batchnorm=False,
            renorm=self.args.renorm,
            norm_position=self.args.norm_position,
            record_activation=self.args.record_mlp_activation,
            inference_mode=self.inference_mode,
        )

    def _build_output_graph(self):
        self._final_node_hidden = tf.reshape(
            self._final_node_hidden,
            [self._batch_size_int, self._node_info["num_nodes"], -1],
        )
        # having the critic which looks only at the root makes the problem severely partially oservable
        # having sum will probably destroy the transfer behaviour
        # average??? what else can be it?
        if self.args.baseline_type == "root":
            self.final_root_hidden = tf.reshape(
                self._final_node_hidden[:, 0, :], [self._batch_size_int, -1]
            )
            self._network_output = self._MLP_Out(self.final_root_hidden)[-1]
        elif self.args.baseline_type == "sum":
            self.final_root_hidden = tf.reshape(
                tf.reduce_sum(self._final_node_hidden, 1), [self._batch_size_int, -1]
            )
            self._network_output = self._MLP_Out(self.final_root_hidden)[-1]
        elif self.args.baseline_type == "avg":
            self.final_root_hidden = tf.reshape(
                tf.reduce_mean(self._final_node_hidden, 1), [self._batch_size_int, -1]
            )
            self._network_output = self._MLP_Out(self.final_root_hidden)[-1]
        elif self.args.baseline_type == "nonlinsum":
            self._network_output = tf.reduce_sum(
                self._MLP_Out(self._final_node_hidden)[-1], 1
            )


class GatedGraphPolicy(GatedGraphNetwork):
    def _prepare(self):
        """
            @brief:
                get the input placeholders ready. The _input placeholder has
                different size from the input we use for the general network.
        """
        # step 1: build the input_obs and input_parameters
        if self._input_obs is None:
            self._input_obs = {
                node_type: tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self._node_info["ob_size_dict"][node_type]],
                    name="input_ob_placeholder_ggnn",
                )
                for node_type in self._node_info["node_type_dict"]
            }
        else:
            assert False, logger.error("Input mustnt be given to the ggnn")

        input_parameter_dtype = (
            tf.int32 if "noninput" in self._gnn_embedding_option else tf.float32
        )
        self._input_parameters = {
            node_type: tf.compat.v1.placeholder(
                input_parameter_dtype,
                [None, self._node_info["para_size_dict"][node_type]],
                name="input_para_placeholder_ggnn",
            )
            for node_type in self._node_info["node_type_dict"]
        }

        # step 2: the receive and send index
        self._receive_idx = tf.compat.v1.placeholder(
            tf.int32, shape=(None), name="receive_idx"
        )
        self._send_idx = {
            edge_type: tf.compat.v1.placeholder(
                tf.int32, shape=(None), name="send_idx_{}".format(edge_type)
            )
            for edge_type in self._node_info["edge_type_list"]
        }

        # step 3: the node type index and inverse node type index
        self._node_type_idx = {
            node_type: tf.compat.v1.placeholder(
                tf.int32, shape=(None), name="node_type_idx_{}".format(node_type)
            )
            for node_type in self._node_info["node_type_dict"]
        }
        self._inverse_node_type_idx = tf.compat.v1.placeholder(
            tf.int32, shape=(None), name="inverse_node_type_idx"
        )

        # step 4: the output node index
        self._output_type_idx = {
            output_type: tf.compat.v1.placeholder(
                tf.int32, shape=(None), name="output_type_idx_{}".format(output_type),
            )
            for output_type in self._node_info["output_type_dict"]
        }

        self._inverse_output_type_idx = tf.compat.v1.placeholder(
            tf.int32, shape=(None), name="inverse_output_type_idx"
        )

        # step 5: batch_size
        self._batch_size_int = tf.compat.v1.placeholder(
            tf.int32, shape=(), name="batch_size_int"
        )

    def _build_output_weights(self, MLP_out_shape, MLP_out_act_func):

        if self.args.output_block_conditioning is not None:
            if self.args.output_block_conditioning in ["hidden", "root"]:
                MLP_out_shape[-1] += self._hidden_dim
            elif self.args.output_block_conditioning in ["hidden&root"]:
                mult = 2
                if self.args.always_feed_node_features:
                    mult = 3
                MLP_out_shape[-1] += mult * self._hidden_dim
            else:
                raise NotImplementedError(
                    f"Unknown output_block_conditioning type {self.args.output_block_conditioning}"
                )

        self._MLP_Out = {
            output_type: nn.VIB(MLP_out_shape[-2], f'{self._name_scope}/MLP_out') if self.args.vib_updater else nn.MLP(
                MLP_out_shape,
                init_method=self._init_method,
                act_func=MLP_out_act_func,
                add_bias=True,
                scope="MLP_out",
                layernorm=False,
                batchnorm=False,
                renorm=self.args.renorm,
                record_activation=self.args.record_mlp_activation,
                inference_mode=self.inference_mode,
            )
            for output_type in self._node_info["output_type_dict"]
        }

    def _build_output_graph(self):
        pass

    # def _io_size_check(self):
    #     """
    #         @brief:
    #             check if the environment's input size and output size is matched
    #             with the one parsed from the mujoco xml file
    #     """
    #     is_io_matched = (
    #         self._input_size == self._node_info["debug_info"]["ob_size"]
    #         and self._output_size == self._node_info["debug_info"]["action_size"]
    #     )
    #     assert is_io_matched, self._io_error_message()


class NerveNetPlusPolicy(GatedGraphPolicy):
    def _prepare(self):
        self._input_hidden_state = {
            node_type: tf.compat.v1.placeholder(
                tf.float32,
                [None, self._hidden_dim],
                name="input_hidden_dim_" + node_type,
            )
            for node_type in self._node_info["node_type_dict"]
        }
        super()._prepare()

    def get_input_hidden_state_placeholder(self):
        """
        self._input_hidden_state = {
            node_type: tf.compat.v1.placeholder(
                tf.float32,
                [None, self._hidden_dim],
                name='input_hidden_dim'
            )
            for node_type in self._node_info['node_type_dict']
        }
        """
        return self._input_hidden_state

    def _build_network_graph(self):
        super()._build_network_graph()

    def get_output_hidden_state_list(self):
        return [
            self.output_hidden_state[key] for key in self._node_info["node_type_dict"]
        ]
