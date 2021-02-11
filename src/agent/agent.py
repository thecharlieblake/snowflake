import multiprocessing
import numpy as np
from graph_util import graph_data_util, gnn_util
from network import NetworkTypes
from util import init_path


class base_agent(multiprocessing.Process):
    def __init__(
        self, args, task_name, task_q, result_q, thread_name, name_scope="trpo_agent",
    ):

        multiprocessing.Process.__init__(self, name=thread_name)
        self.task_q = task_q
        self.result_q = result_q
        self.args = args
        self.name_scope = name_scope
        self.task_name = task_name
        self.policy_var_list = None
        self.tf_var_list = None
        self.iteration = None

        # the gnn parameters
        if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
            self.gnn_parameter_initialization()

        self.base_path = init_path.get_base_dir()

    def fetch_policy_info(self):

        # input placeholders to the policy networks
        if (
            self.args.policy_network_type == NetworkTypes.nervenet.name
            or self.args.policy_network_type == NetworkTypes.nervenetpp.name
        ):
            # index placeholders
            (
                self.receive_idx_placeholder,
                self.send_idx_placeholder,
                self.node_type_idx_placeholder,
                self.inverse_node_type_idx_placeholder,
                self.output_type_idx_placeholder,
                self.inverse_output_type_idx_placeholder,
                self.batch_size_int_placeholder,
            ) = self.policy_network.get_gnn_idx_placeholder()

            # the graph_obs placeholders and graph_parameters_placeholders
            self.graph_obs_placeholder = self.policy_network.get_input_obs_placeholder()
            self.graph_parameters_placeholder = (
                self.policy_network.get_input_parameters_placeholder()
            )

            self.gnn_placeholder_list = [
                self.receive_idx_placeholder,
                self.send_idx_placeholder,
                self.node_type_idx_placeholder,
                self.inverse_node_type_idx_placeholder,
                self.output_type_idx_placeholder,
                self.inverse_output_type_idx_placeholder,
                self.batch_size_int_placeholder,
                self.graph_obs_placeholder,
                self.graph_parameters_placeholder,
            ]

            if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                (
                    self.step_receive_idx_placeholder,
                    self.step_send_idx_placeholder,
                    self.step_node_type_idx_placeholder,
                    self.step_inverse_node_type_idx_placeholder,
                    self.step_output_type_idx_placeholder,
                    self.step_inverse_output_type_idx_placeholder,
                    self.step_batch_size_int_placeholder,
                ) = self.step_policy_network.get_gnn_idx_placeholder()

                # the graph_obs placeholders and graph_parameters_placeholders
                self.step_graph_obs_placeholder = (
                    self.step_policy_network.get_input_obs_placeholder()
                )
                self.step_graph_parameters_placeholder = (
                    self.step_policy_network.get_input_parameters_placeholder()
                )
        else:
            self.obs_placeholder = self.policy_network.get_input_placeholder()

        self.raw_obs_placeholder = None
        # with tf.compat.v1.get_default_graph().as_default():

        (
            self.policy_var_list,
            self.all_policy_var_list,
        ) = self.policy_network.get_var_list()
        self.iteration = self.policy_network.get_iteration_var()
        self.iteration_add_op = self.iteration.assign_add(1)

    def gnn_parameter_initialization(self):
        """
            @brief:
                the parameters for the gnn, see the gated_graph_network_policy
                file for details what these variables mean.
        """
        self.receive_idx = None
        self.send_idx = None
        self.node_type_idx = None
        self.inverse_node_type_idx = None
        self.output_type_idx = None
        self.inverse_output_type_idx = None
        self.last_batch_size = -1

    def prepared_policy_network_feeddict(
        self, obs_n, node_info, rollout_data=None, step_model=False
    ):
        """
            @brief: prepare the feed dict for the policy network part
        """
        nervenetplus_batch_pos = None
        if rollout_data is not None:
            obs_n = np.concatenate(obs_n)

        if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:

            if (
                not self.args.policy_network_type == NetworkTypes.nervenetpp.name
                or obs_n.shape[0] == 1
            ):
                (
                    graph_obs,
                    graph_parameters,
                    self.receive_idx,
                    self.send_idx,
                    self.node_type_idx,
                    self.inverse_node_type_idx,
                    self.output_type_idx,
                    self.inverse_output_type_idx,
                    self.last_batch_size,
                ) = graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    obs_n,
                    self.receive_idx,
                    self.send_idx,
                    self.node_type_idx,
                    self.inverse_node_type_idx,
                    self.output_type_idx,
                    self.inverse_output_type_idx,
                    self.last_batch_size,
                    request_data=["ob", "idx"],
                )
            else:
                assert rollout_data is not None

                # preprocess the episodic information
                (
                    graph_obs,
                    graph_parameters,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = graph_data_util.construct_graph_input_feeddict(
                    node_info, obs_n, -1, -1, -1, -1, -1, -1, -1, request_data=["ob"],
                )
                (
                    nervenetplus_batch_pos,
                    total_size,
                ) = gnn_util.nervenetplus_step_assign(
                    rollout_data, self.args.gnn_num_prop_steps
                )
                (
                    _,
                    _,
                    self.receive_idx,
                    self.send_idx,
                    self.node_type_idx,
                    self.inverse_node_type_idx,
                    self.output_type_idx,
                    self.inverse_output_type_idx,
                    self.last_batch_size,
                ) = graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    np.empty([int(total_size / self.args.gnn_num_prop_steps)]),
                    self.receive_idx,
                    self.send_idx,
                    self.node_type_idx,
                    self.inverse_node_type_idx,
                    self.output_type_idx,
                    self.inverse_output_type_idx,
                    self.last_batch_size,
                    request_data=["idx"],
                )

            if step_model:
                feed_dict = {
                    self.step_batch_size_int_placeholder: int(self.last_batch_size),
                    self.step_receive_idx_placeholder: self.receive_idx,
                    self.step_inverse_node_type_idx_placeholder: self.inverse_node_type_idx,
                    self.step_inverse_output_type_idx_placeholder: self.inverse_output_type_idx,
                }

                # append the input obs and parameters
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[self.step_graph_obs_placeholder[i_node_type]] = graph_obs[
                        i_node_type
                    ]
                    feed_dict[
                        self.step_graph_parameters_placeholder[i_node_type]
                    ] = graph_parameters[i_node_type]

                # append the send idx
                for i_edge in node_info["edge_type_list"]:
                    feed_dict[self.step_send_idx_placeholder[i_edge]] = self.send_idx[
                        i_edge
                    ]

                # append the node type idx
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[
                        self.step_node_type_idx_placeholder[i_node_type]
                    ] = self.node_type_idx[i_node_type]

                # append the output type idx
                for i_output_type in node_info["output_type_dict"]:
                    feed_dict[
                        self.step_output_type_idx_placeholder[i_output_type]
                    ] = self.output_type_idx[i_output_type]

                # if the raw_obs is needed for the baseline
                if self.raw_obs_placeholder is not None:
                    feed_dict[self.raw_obs_placeholder] = obs_n
            else:

                # construct the graph input feed dict
                # in this case, we need to get the receive_idx, send_idx,
                # node_idx, inverse_node_idx ready. These index will be helpful
                # to telling the network how to pass and update the information
                (
                    graph_obs,
                    graph_parameters,
                    self.receive_idx,
                    self.send_idx,
                    self.node_type_idx,
                    self.inverse_node_type_idx,
                    self.output_type_idx,
                    self.inverse_output_type_idx,
                    self.last_batch_size,
                ) = graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    obs_n,
                    self.receive_idx,
                    self.send_idx,
                    self.node_type_idx,
                    self.inverse_node_type_idx,
                    self.output_type_idx,
                    self.inverse_output_type_idx,
                    self.last_batch_size,
                )

                feed_dict = {
                    self.batch_size_int_placeholder: int(self.last_batch_size),
                    self.receive_idx_placeholder: self.receive_idx,
                    self.inverse_node_type_idx_placeholder: self.inverse_node_type_idx,
                    self.inverse_output_type_idx_placeholder: self.inverse_output_type_idx,
                }

                # append the input obs and parameters
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[self.graph_obs_placeholder[i_node_type]] = graph_obs[
                        i_node_type
                    ]
                    feed_dict[
                        self.graph_parameters_placeholder[i_node_type]
                    ] = graph_parameters[i_node_type]

                # append the send idx
                for i_edge in node_info["edge_type_list"]:
                    feed_dict[self.send_idx_placeholder[i_edge]] = self.send_idx[i_edge]

                # append the node type idx
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[
                        self.node_type_idx_placeholder[i_node_type]
                    ] = self.node_type_idx[i_node_type]

                # append the output type idx
                for i_output_type in node_info["output_type_dict"]:
                    feed_dict[
                        self.output_type_idx_placeholder[i_output_type]
                    ] = self.output_type_idx[i_output_type]

                # if the raw_obs is needed for the baseline
                if self.raw_obs_placeholder is not None:
                    feed_dict[self.raw_obs_placeholder] = obs_n
        else:
            # it is the most easy case, nice and easy
            feed_dict = {self.obs_placeholder: obs_n}

        self.nervenetplus_batch_pos = nervenetplus_batch_pos
        return feed_dict, nervenetplus_batch_pos

    def get_sess(self):
        return self.session

    def get_iteration_count(self):
        return self.session.run(self.iteration)

    def get_experiment_name(self):
        """
            @brief:
                this is the unique id of the experiments. it might be useful if
                we are running several tasks on the server
        """
        return "_".join(self.args.task) + "_" + self.args.time_id
