import multiprocessing
import os
import time
import collections

import gym
import numpy as np
import tensorflow as tf
import pandas as pd

import network
from agent import rollout_master_agent
from agent.agent import base_agent
from graph_util import graph_data_util, gnn_util
from graph_util.mujoco_parser import parse_mujoco_template
from network import NetworkTypes
from network.abstractions import build_gnn_output, build_mlp_output
from network.net_maker import (
    build_session,
    build_baseline_network,
    build_policy_network,
    build_nervenetpp_policy,
)
from util import init_path
from util import logger
from util import model_saver
from util import parallel_util
from util import summary_handler
from util import utils
from collections import defaultdict
import sklearn.gaussian_process as gp
import sklearn.linear_model as linear_model
import itertools
import pickle
from collections import deque
from util.utils import count_params
from util.utils import means_diff_len
import re


def build_update_op_preprocess(action_size, task_name):
    """
        @brief: The preprocess that is shared by trpo, ppo and vpg updates
    """
    # the input placeholders for the input
    action_placeholder = tf.compat.v1.placeholder(
        tf.float32, [None, action_size], name=f"action_sampled_in_rollout_{task_name}",
    )
    advantage_placeholder = tf.compat.v1.placeholder(
        tf.float32, [None], name=f"advantage_value_{task_name}"
    )
    oldaction_dist_mu_placeholder = tf.compat.v1.placeholder(
        tf.float32, [None, action_size], name=f"old_act_dist_mu_{task_name}"
    )
    oldaction_dist_logstd_placeholder = tf.compat.v1.placeholder(
        tf.float32, [None, action_size], name=f"old_act_dist_logstd_{task_name}"
    )
    batch_size_float_placeholder = tf.compat.v1.placeholder(
        tf.float32, [], name=f"batch_size_float_{task_name}"
    )

    return {
        "action_placeholder": action_placeholder,
        "advantage_placeholder": advantage_placeholder,
        "oldaction_dist_mu_placeholder": oldaction_dist_mu_placeholder,
        "oldaction_dist_logstd_placeholder": oldaction_dist_logstd_placeholder,
        "batch_size_float_placeholder": batch_size_float_placeholder,
    }




class Bandit:
    def __init__(
        self, n_tasks, value_range=(0.5, 1.0), options_per_task=5, fifo_size=10
    ):
        # epsilon greedy with a window bandit
        # replace with UCB/thompson sampling later if the idea works
        self.sampling_options = list(
            itertools.product(
                *[np.linspace(*value_range, options_per_task) for _ in range(n_tasks)]
            )
        )
        self.arms = [deque(maxlen=fifo_size) for _ in self.sampling_options]
        self.means = [1000 for _ in self.arms]
        self.last_sampled = None
        self.n_arms = len(self.arms)
        self.total_pulls = 0
        self.last_ret = 0
        self.max_avg_return = -np.infty
        self.min_avg_return = np.infty

    def push_return(self, ret):
        if ret > self.max_avg_return:
            self.max_avg_return = ret
        if ret < self.min_avg_return:
            self.min_avg_return = ret

        # for the first time, there is no update, only sampling, we do not need to push it
        if self.total_pulls > 0:
            ret_diff = ret - self.last_ret
            self.arms[self.last_sampled].append(ret_diff)
            self.means[self.last_sampled] = np.mean(self.arms[self.last_sampled])
            self.last_ret = ret

    def pull_process(self, chosen_arm):
        self.total_pulls += 1
        self.last_sampled = chosen_arm
        # plt.figure()
        # plt.bar(
        #     range(self.n_arms),
        #     self.means,
        #     tick_label=[str(el) for el in self.sampling_options],
        # )
        # plt.savefig("bandits.png")
        # plt.close()


class EpsGreedyBandit(Bandit):
    def __init__(
        self, n_tasks, value_range=(0.5, 1.0), options_per_task=4, fifo_size=10, eps=0.9
    ):
        super().__init__(
            n_tasks,
            value_range=value_range,
            options_per_task=options_per_task,
            fifo_size=fifo_size,
        )
        self.eps = eps

    def pull(self):
        if np.random.randn() < self.eps:
            idx = np.random.randint(0, self.n_arms)
        else:
            idx = np.argmax(self.means)
        super().pull_process(idx)
        return self.sampling_options[idx]


class UCB1Bandit(Bandit):
    # check windowed ucb version
    def pull(self):
        counts = [len(el) for el in self.arms]
        if 0 in counts:
            idx = np.argmin(counts)
        else:
            idx = np.argmax(
                [
                    self.means[i]
                    + np.sqrt(2 * np.log(self.total_pulls) / len(self.arms[i]))
                    for i in range(self.n_arms)
                ]
            )
        super().pull_process(idx)
        return self.sampling_options[idx]


class WindowedUCB1Bandit(Bandit):
    def __init__(
        self,
        n_tasks,
        value_range=(0.5, 1.0),
        options_per_task=3,
        fifo_size=20,
        eps=0.3,
    ):
        super().__init__(
            n_tasks,
            value_range=value_range,
            options_per_task=options_per_task,
            fifo_size=fifo_size,
        )
        self.eps = eps
        self.counts = [deque(maxlen=fifo_size) for _ in range(self.n_arms)]
        self.window_size = fifo_size

    def push_return(self, ret):
        # for the first time, there is no update, only sampling, we do not need to push it
        if self.total_pulls > 0:
            ret_diff = ret - self.last_ret
            for arm_idx in range(self.n_arms):
                if arm_idx == self.last_sampled:
                    self.arms[arm_idx].append(ret_diff)
                else:
                    self.arms[arm_idx].append(0)
            self.means[self.last_sampled] = (
                sum(self.arms[self.last_sampled])
                / sum(self.counts[self.last_sampled])
                / sum(self.sampling_options[self.last_sampled])
            )
            self.last_ret = ret

    def pull(self):
        if self.total_pulls < self.n_arms:
            idx = self.total_pulls
        elif np.random.randn() < self.eps:
            idx = np.random.randint(0, self.n_arms)
        else:
            idx = np.argmax(
                [
                    self.means[i]
                    + np.sqrt(
                        2
                        * np.log(min(self.total_pulls, self.window_size))
                        / sum(self.counts[i])
                    )
                    for i in range(self.n_arms)
                ]
            )
        super().pull_process(idx)
        for arm_idx in range(self.n_arms):
            if arm_idx == self.last_sampled:
                self.counts[idx].append(1)
            else:
                self.counts[idx].append(0)
        return self.sampling_options[idx]


class ThompsonBandit(Bandit):

    # https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
    def __init__(
        self, n_tasks, value_range=(0.5, 1.0), options_per_task=5, fifo_size=10
    ):

        super().__init__(
            n_tasks,
            value_range=value_range,
            options_per_task=options_per_task,
            fifo_size=fifo_size,
        )
        self._alphas = [1 for _ in range(self.n_arms)]
        self._betas = [1 for _ in range(self.n_arms)]

    def push_return(self, ret):
        super().push_return(ret)
        if self.total_pulls > 0:
            norm_ret = (ret - self.min_avg_return) / (
                self.max_avg_return - self.min_avg_return + 1e-8
            )
            self._alphas[self.last_sampled] += norm_ret
            self._betas[self.last_sampled] += 1 - norm_ret

    def pull(self):
        idx = np.argmax(
            [
                np.random.beta(self._alphas[i], self._betas[i])
                for i in range(self.n_arms)
            ]
        )
        super().pull_process(idx)
        return self.sampling_options[idx]


class optimization_agent(base_agent):
    def __init__(
        self, args, name_scope="trpo_agent",
    ):

        self.learner_tasks = multiprocessing.JoinableQueue()
        self.learner_results = multiprocessing.Queue()
        self.training_tasks = args.task
        self.test_transfer_tasks = args.test_transfer_tasks
        self.tasks = self.training_tasks + self.test_transfer_tasks
        self.timesteps_to_sample_next = {
            tname: args.timesteps_per_batch for tname in self.tasks
        }
        self.observation_size = {}
        self.action_size = {}
        self.action_dict = {}
        self.body_dict = {}
        for tname in self.tasks:
            dummy_env = gym.make(tname)
            self.observation_size[tname] = dummy_env.observation_space.shape[0]
            self.action_size[tname] = dummy_env.action_space.shape[0]
            self.action_dict[tname] = dummy_env.env.model._actuator_id2name
            self.body_dict[tname] = dummy_env.env.model._body_id2name
        self.obs_description = self.get_obs_description()

        self.node_info = (
            None
            if args.policy_network_type == NetworkTypes.fixedsizenet.name
            else {
                tname: parse_mujoco_template(
                    tname,
                    args.gnn_input_feat_dim,
                    args.gnn_node_option,
                    args.root_connection_option,
                    args.gnn_output_option,
                    args.gnn_embedding_option,
                    args.no_sharing_within_layers,
                )
                for tname in self.tasks
            }
        )

        self.sample_sizes = np.zeros(
            (10000, len(self.tasks)), dtype=np.float32
        )  # mixing ratio
        self.sample_sizes_targets = np.zeros(
            (10000), dtype=np.float32
        )  # storing t+1 returns after t mix ratio

        if args.sample_budgeting == "epsgreedy_bandit":
            self.bandit = EpsGreedyBandit(len(self.tasks))
        elif args.sample_budgeting == "thompson_bandit":
            self.bandit = ThompsonBandit(len(self.tasks))
        elif args.sample_budgeting == "ucb_bandit":
            self.bandit = UCB1Bandit(len(self.tasks))
        elif args.sample_budgeting == "windowed_ucb_bandit":
            self.bandit = WindowedUCB1Bandit(
                len(self.tasks), fifo_size=args.bandit_window_size
            )

        self.mixing_ratio_ph = tf.compat.v1.placeholder(
            tf.float32, [len(self.training_tasks)], name="mixing_ratio"
        )
        self.max_avg_return = -np.infty
        self.min_avg_return = np.infty
        #########

        super(optimization_agent, self).__init__(
            args=args,
            task_name=self.tasks[0],
            task_q=self.learner_tasks,
            result_q=self.learner_results,
            thread_name='optimization_agent',
            name_scope=name_scope
        )

        # the variables and networks to be used, init them before use them
        self.env_info = None

        # used to save the checkpoint files
        self.best_reward = -np.inf
        self.last_save_iteration = 0
        self.timesteps_so_far = 0
        self._npr = np.random.RandomState(args.seed)

        self.start()  # Starts its own parallelism (weird huh), but the new thread blocks until the start signal in 343
        # Uses the rollout master (same thread) to start the agents (new threads)
        self.rollout_agent = rollout_master_agent.parallel_rollout_master_agent(
            args, self.node_info, self.tasks, self.obs_description, self.action_dict)
        # Note: of these 3 signal-steps, this signal is sent to self, not rollout agents. Optimisation (separate) thread
        # goes and does some weight setup stuff and dumps the result in learner_results
        self.learner_tasks.put(parallel_util.START_SIGNAL)
        self.learner_tasks.join()  # Just waiting for self.run() to issue task_done
        # Waits for self to finish the weight setup stuff, then sends AGENT_SET_POLICY_WEIGHTS to the agents.
        self.rollout_agent.set_policy_weights(self.learner_results.get())
        self.start_time = time.time()

    def run(self):
        """
            @brief:
                this is the standard function to be called by the
                "multiprocessing.Process"
        """
        self.build_models()
        if self.args.ckpt_name is not None:
            self.restore_all()

        # the main training process
        while True:
            next_task = self.task_q.get()

            # Kill the learner
            if next_task is None or next_task == parallel_util.END_SIGNAL:
                self.task_q.task_done()
                break

            # Get the policy network weights
            elif next_task == parallel_util.START_SIGNAL:
                # sync non-blueprint policies/baselines at the beginning
                for sp in self.set_policy.values():
                    sp(
                        self.get_policy[self.task_name](
                            skip_logstd=self.args.multitask_logstd_joint == 0,
                            skip_embedding="separate" in self.args.gnn_embedding_option,
                        ),
                        skip_logstd=self.args.multitask_logstd_joint == 0,
                        skip_embedding="separate" in self.args.gnn_embedding_option,
                    )
                if self.args.multitask_critic_joint == 1:
                    for sb in self.sync_baselines:
                        sb(self.get_blueprint_baseline())
                if self.test_transfer_tasks:
                    self.sync_test_transfer_tasks()
                res = {k: v() for k, v in self.get_policy.items()}
                self.task_q.task_done()
                self.result_q.put(res)

            # Updating the network
            else:
                if self.args.test:
                    paths, running_means, timesteps_to_sample = next_task
                    paths = paths[self.task_name]
                    episoderewards = np.array([path["rewards"].sum() for path in paths])
                    self.task_q.task_done()
                    # testing on one task only
                    stats = {self.args.task[0]: {"avg_reward": episoderewards.mean()}}
                    logger.info(stats)
                    return_data = {
                        "policy_weights": {k: v() for k, v in self.get_policy.items()},
                        "stats": stats,
                        "totalsteps": self.args.max_timesteps + 100,
                        "iteration": self.get_iteration_count(),
                        "std_reward": episoderewards.std(),
                        "avg_reward": episoderewards.mean(),
                        "max_reward": np.amax(episoderewards),
                        "min_reward": np.amin(episoderewards),
                        "median_reward": np.median(episoderewards),
                        "raw_rewards": episoderewards,
                        "timesteps_to_sample_next": self.get_sampling_sizes(),
                    }
                    self.result_q.put(return_data)
                # the actual training step
                else:
                    paths, running_means, sampled_timesteps = next_task
                    stats = self.update_parameters(
                        paths, running_means, sampled_timesteps
                    )
                    self.task_q.task_done()

                    if self.test_transfer_tasks:
                        self.sync_test_transfer_tasks()

                    return_data = {
                        "policy_weights": {k: v() for k, v in self.get_policy.items()},
                        "stats": stats,
                        "totalsteps": self.timesteps_so_far,
                        "iteration": self.get_iteration_count(),
                        "timesteps_to_sample_next": self.get_sampling_sizes(),
                    }
                    self.result_q.put(return_data)

    def build_baseline_network(self):

        ob_placeholder_dict = {
            tname: (
                self.obs_placeholder[tname]
                if self.args.policy_network_type == NetworkTypes.fixedsizenet.name
                else tf.compat.v1.placeholder(
                    tf.float32, [None, self.observation_size[tname]], name="ob_input"
                )
            )
            for tname in self.tasks
        }

        self.baseline_network = {
            tname: build_baseline_network(
                self.args,
                self.session,
                self.name_scope + "_baseline_" + str(tid),
                self.observation_size[tname],
                tname,
                ob_placeholder=ob_placeholder_dict[tname],
                placeholder_list=self.gnn_placeholder_list_dict[tname]
                if self.args.value_network_type == NetworkTypes.nervenet.name
                else None,
                node_info=self.node_info[tname] if self.node_info else None,
            )
            for tid, tname in enumerate(self.tasks)
        }
        if (
            not self.args.policy_network_type == NetworkTypes.fixedsizenet.name
            and not self.args.value_network_type == NetworkTypes.nervenet.name
        ):
            # in this case the raw obs and ob is different
            self.raw_obs_placeholder = {
                tname: self.baseline_network[tname].get_ob_placeholder()
                for tname in self.tasks
            }

    def build_models(self):
        """
            @brief:
                this is the function where the rollout agents and optimization
                agent build their networks, set up the placeholders, and gather
                the variable list.
        """
        # make sure that the agent has a session
        self.session = build_session(self.args.use_gpu)
        self.build_policy_network()
        self.build_baseline_network()

        # step 2: get the placeholders for the network
        self.target_return_placeholder = {
            tname: self.baseline_network[tname].get_target_return_placeholder()
            for tname in self.tasks
        }

        # the training op and graphs
        self.build_ppo_update_op()
        self.update_parameters = self.update_ppo_parameters

        # init the network parameters (xavier initializer)
        self.session.run(tf.compat.v1.global_variables_initializer())

        # the set weight policy ops
        self.get_policy = {
            tname: utils.GetWeightsForPrefix(
                self.session, self.all_policy_var_list_dict[tname]
            )
            for tname in self.tasks
        }

        self.set_policy = {
            tname: utils.SetWeightsForPrefix(
                self.session, self.all_policy_var_list_dict[tname]
            )
            for tname in self.training_tasks
            if tname != self.task_name and tname not in self.test_transfer_tasks
        }

        blueprint_baseline_var_list, _ = self.baseline_network[
            self.task_name
        ].get_var_list()
        self.get_blueprint_baseline = utils.GetWeightsForPrefix(
            self.session, blueprint_baseline_var_list, prefix="baseline"
        )
        self.sync_baselines = [
            utils.SetWeightsForPrefix(
                self.session,
                self.baseline_network[tname].get_var_list()[0],
                prefix="baseline",
            )
            for tname in self.training_tasks
            if tname != self.task_name and tname not in self.test_transfer_tasks
        ]

        # prepare the feed_dict info for the ppo minibatches
        self.batch_feed_dict_key = {}
        self.graph_batch_feed_dict_key = {}
        self.static_feed_dict = {}
        self.dynamical_feed_dict_key = {}
        self.graph_index_feed_dict = {}
        self.graph_index_feed_dict = {}
        for tname in self.training_tasks:
            self.prepare_feed_dict_map(tname)

        # set the summary writer
        self.summary_writer = summary_handler.gym_summary_handler(
            self.session,
            self.get_experiment_name(),
            enable=self.args.write_summary,
            summary_dir=self.args.output_dir,
        )

    def sync_test_transfer_tasks(self):
        sync_start = time.time()
        # First main task to numpy
        policy_checkpoint = self.policy_network[self.task_name].checkpoint_dict()
        baseline_checkpoint = self.baseline_network[self.task_name].checkpoint_dict()
        # for each test transfer task load
        for trans_task in self.test_transfer_tasks:
            if trans_task in self.args.max_act_test_transfer_tasks:
                load_policy = self.override_param_value(policy_checkpoint, 'logstd', -np.inf)
            else:
                load_policy = policy_checkpoint
            self.policy_network[trans_task].load_checkpoint_from_dict(load_policy,
            transfer_env=f"{self.task_name}2{trans_task}".replace("-v1", ""),
            logstd_option=self.args.logstd_option,
            gnn_option_list=[
                self.args.gnn_node_option,
                self.args.root_connection_option,
                self.args.gnn_output_option,
                self.args.gnn_embedding_option,
            ],
            mlp_raw_transfer=self.args.mlp_raw_transfer,)
            self.baseline_network[trans_task].load_checkpoint_from_dict(baseline_checkpoint)

        sync_time = (time.time() - sync_start) / 60.0
        logger.info("sync test transfer spent : %.2f mins" % (sync_time))

    def override_param_value(self, param_dict, param_name, new_val):
        return {
            param: np.full(val.shape, new_val) if param_name in param else val
            for param, val in param_dict.items()
        }

    def build_ppo_update_op(self):
        self.task_specific_placeholders = {}
        self.log_p_n = {}
        self.log_oldp_n = {}
        self.kl = {}
        self.ratio = {}
        self.ratio_clip = {}
        self.ent = {}
        self.surr = {}
        self.vf_loss = {}
        self.loss = {}
        self.vib_encoder_loss = {}
        self.vib_edge_loss = {}
        self.vib_decoder_loss = {}
        grads = {}
        self.grad_norms = {}
        self.clipped_grads = {}
        self.tvars = {}
        vf_tvars = {}
        vf_grads = {}
        self.per_task_update_op = {}
        self.edge_norms = {}
        self.spectral_radius = {}
        self.clip_stats = {}

        self.lr_placeholder = tf.compat.v1.placeholder(
            tf.float32, [], name="learning_rate"
        )
        self.separate_node_lr_placeholder = tf.compat.v1.placeholder(tf.float32, [], name="separate_node_lr")
        self.separate_edge_lr_placeholder = tf.compat.v1.placeholder(tf.float32, [], name="separate_edge_lr")
        self.separate_encoder_lr_placeholder = tf.compat.v1.placeholder(tf.float32, [], name="separate_encoder_lr")
        self.separate_decoder_lr_placeholder = tf.compat.v1.placeholder(tf.float32, [], name="separate_decoder_lr")
        self.current_lr = self.args.lr
        if self.args.pi_opt == "adam":
            self.optimizer = {"main": tf.compat.v1.train.AdamOptimizer(self.lr_placeholder)}
        elif self.args.pi_opt == "sgd":
            self.optimizer = {"main": tf.compat.v1.train.GradientDescentOptimizer(self.lr_placeholder)}
        else:
            raise NotImplementedError("Unknown optimizer")
        if self.args.separate_node_lr > -0.5:
            self.current_node_lr = self.args.separate_node_lr
            if   self.args.pi_opt == "adam": self.optimizer['node'] = tf.compat.v1.train.AdamOptimizer           (self.separate_node_lr_placeholder)
            elif self.args.pi_opt == "sgd":  self.optimizer['node'] = tf.compat.v1.train.GradientDescentOptimizer(self.separate_node_lr_placeholder)
        if self.args.separate_edge_lr > -0.5:
            self.current_edge_lr = self.args.separate_edge_lr
            if   self.args.pi_opt == "adam": self.optimizer['edge'] = tf.compat.v1.train.AdamOptimizer           (self.separate_edge_lr_placeholder)
            elif self.args.pi_opt == "sgd":  self.optimizer['edge'] = tf.compat.v1.train.GradientDescentOptimizer(self.separate_edge_lr_placeholder)
        if self.args.separate_encoder_lr > -0.5:
            self.current_encoder_lr = self.args.separate_encoder_lr
            if   self.args.pi_opt == "adam": self.optimizer['encoder'] = tf.compat.v1.train.AdamOptimizer           (self.separate_encoder_lr_placeholder)
            elif self.args.pi_opt == "sgd":  self.optimizer['encoder'] = tf.compat.v1.train.GradientDescentOptimizer(self.separate_encoder_lr_placeholder)
        if self.args.separate_decoder_lr > -0.5:
            self.current_decoder_lr = self.args.separate_decoder_lr
            if   self.args.pi_opt == "adam": self.optimizer['decoder'] = tf.compat.v1.train.AdamOptimizer           (self.separate_decoder_lr_placeholder)
            elif self.args.pi_opt == "sgd":  self.optimizer['decoder'] = tf.compat.v1.train.GradientDescentOptimizer(self.separate_decoder_lr_placeholder)

        for tname in self.training_tasks:



            self.task_specific_placeholders[tname] = build_update_op_preprocess(
                self.action_size[tname], tname
            )

            self.log_p_n[tname] = utils.gauss_log_prob(
                self.action_dist_mu[tname],
                self.action_dist_logstd[tname],
                self.task_specific_placeholders[tname]["action_placeholder"],
            )

            # what are the probabilities of taking self.action, given new and old distributions
            self.log_oldp_n[tname] = utils.gauss_log_prob(
                self.task_specific_placeholders[tname]["oldaction_dist_mu_placeholder"],
                self.task_specific_placeholders[tname][
                    "oldaction_dist_logstd_placeholder"
                ],
                self.task_specific_placeholders[tname]["action_placeholder"],
            )

            self.ratio[tname] = tf.exp(self.log_p_n[tname] - self.log_oldp_n[tname])

            self.kl[tname] = (
                utils.gauss_KL(
                    self.task_specific_placeholders[tname][
                        "oldaction_dist_mu_placeholder"
                    ],
                    self.task_specific_placeholders[tname][
                        "oldaction_dist_logstd_placeholder"
                    ],
                    self.action_dist_mu[tname],
                    self.action_dist_logstd[tname],
                )
                / self.task_specific_placeholders[tname]["batch_size_float_placeholder"]
            )
            # the entropy
            self.ent[tname] = (
                utils.gauss_ent(
                    self.action_dist_mu[tname], self.action_dist_logstd[tname]
                )
                / self.task_specific_placeholders[tname]["batch_size_float_placeholder"]
            )

            # step 1: the surrogate loss
            self.ratio_clip[tname] = tf.clip_by_value(
                self.ratio[tname], 1.0 - self.args.ppo_clip, 1.0 + self.args.ppo_clip
            )

            self.surr[tname] = tf.minimum(
                self.ratio_clip[tname]
                * self.task_specific_placeholders[tname]["advantage_placeholder"],
                self.ratio[tname]
                * self.task_specific_placeholders[tname]["advantage_placeholder"],
            )
            #self.surr[tname] = -tf.reduce_mean(self.surr[tname])

            self.surr[tname] += self.args.ent_coef * self.ent[tname]

            # ratio | adv | LHS | RHS | min | clip?
            #
            # >>    | + | r>>A+ | (1+e)A+ | (1+e)A+ | T
            # >>    | - | r>>A- | (1+e)A- | r>>A    | F
            # else  | . |       |         | =       | F
            # <<    | + | r<<A+ | (1-e)A+ | r<<A+   | F
            # <<    | - | r<<A- | (1-e)A- | (1-e)A  | T

            # Therefore rule = clip if:
            # - ratio > 1+e and A +ve
            # - ratio < 1-e and A -ve

            if self.args.record_ppo_clip:
                pos_adv = tf.greater(self.task_specific_placeholders[tname]["advantage_placeholder"], 0)
                neg_adv = tf.logical_not(pos_adv)
                upper_clip_ratio = tf.greater(self.ratio[tname], 1 + self.args.ppo_clip)
                lower_clip_ratio = tf.less(self.ratio[tname], 1 - self.args.ppo_clip)
                no_clip_ratio = tf.logical_not(tf.logical_or(upper_clip_ratio, lower_clip_ratio))

                self.clip_stats[tname] = {
                    'pos_a_upper_clip'      : tf.reduce_sum(tf.cast(tf.math.logical_and(pos_adv, upper_clip_ratio), tf.float32)),
                    'pos_a_no_clip_ratio'   : tf.reduce_sum(tf.cast(tf.math.logical_and(pos_adv, no_clip_ratio   ), tf.float32)),
                    'pos_a_lower_clip_ratio': tf.reduce_sum(tf.cast(tf.math.logical_and(pos_adv, lower_clip_ratio), tf.float32)),
                    'neg_a_upper_clip_ratio': tf.reduce_sum(tf.cast(tf.math.logical_and(neg_adv, upper_clip_ratio), tf.float32)),
                    'neg_a_no_clip_ratio'   : tf.reduce_sum(tf.cast(tf.math.logical_and(neg_adv, no_clip_ratio   ), tf.float32)),
                    'neg_a_lower_clip_ratio': tf.reduce_sum(tf.cast(tf.math.logical_and(neg_adv, lower_clip_ratio), tf.float32)),
                }

            # step 2: the value function loss. if not tf, the loss will be fit in
            # update_parameters_postprocess
            self.vf_loss[tname] = self.baseline_network[tname].get_vf_loss()
            self.loss[tname] = self.surr[tname]

            # step 4: weight decay
            self.weight_decay_loss = tf.zeros([1])
            self.encoder_weight_decay_loss = tf.zeros([1])
            self.edge_weight_decay_loss = tf.zeros([1])
            self.decoder_weight_decay_loss = tf.zeros([1])
            for var in tf.compat.v1.trainable_variables():
                self.weight_decay_loss += tf.nn.l2_loss(var)
                if "MLP_embedding_node_type" in var.name:
                    self.encoder_weight_decay_loss += tf.nn.l2_loss(var)
                if "MLP_prop_edge" in var.name:
                    self.edge_weight_decay_loss += tf.nn.l2_loss(var)
                if "MLP_out" in var.name:
                    self.decoder_weight_decay_loss += tf.nn.l2_loss(var)

            if self.args.use_weight_decay:
                self.loss[tname] -= (
                    self.weight_decay_loss * self.args.weight_decay_coeff
                )
            if self.args.use_edge_weight_decay:
                self.loss[tname] -= (
                        self.edge_weight_decay_loss * self.args.edge_weight_decay_coeff
                )
            if self.args.vib_updater:
                self.vib_encoder_loss[tname] = tf.reduce_mean(self.policy_network[tname]._vib_encoder_kl)
                self.vib_edge_loss[tname] = tf.reduce_mean(self.policy_network[tname]._vib_edge_kl)
                self.vib_decoder_loss[tname] = tf.reduce_mean(self.policy_network[tname]._vib_decoder_kl)
                self.loss[tname] += self.args.vib_beta * (self.vib_encoder_loss[tname] + self.vib_edge_loss[tname] + self.vib_decoder_loss[tname]) / 3
                self.mean_vib_edge_mu, self.var_vib_edge_mu = tf.reduce_mean(self.policy_network[tname]._vib_edge_stats["mean_mu"]), tf.reduce_mean(self.policy_network[tname]._vib_edge_stats["var_mu"])
                self.mean_vib_edge_rho, self.var_vib_edge_rho = tf.reduce_mean(self.policy_network[tname]._vib_edge_stats["mean_rho"]), tf.reduce_mean(self.policy_network[tname]._vib_edge_stats["var_rho"])

            if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
                # we need to clip the gradient for the ggnn
                # self.tvars = tf.compat.v1.trainable_variables()
                self.tvars[tname] = self.policy_network[tname]._trainable_var_list
                grads[tname] = tf.gradients(-tf.reduce_mean(self.loss[tname]), self.tvars[tname])

                if self.args.record_grads:
                    grad_calc = lambda item: tf.gradients(-item, self.tvars[tname])
                    self.per_example_grad = tf.map_fn(grad_calc, self.loss[tname], fn_output_signature=[tf.TensorSpec(x.shape, x.dtype) for x in grads[tname]])

                    self.grad_mean = [tf.math.reduce_mean(g, axis=0) for g in self.per_example_grad if g is not None]
                    self.grad_std = [tf.math.reduce_std(g, axis=0) for g in self.per_example_grad if g is not None]
                    self.snr = [tf.abs(m) / s for m, s in zip(self.grad_mean, self.grad_std)]

                    self.grad_var = [tf.math.reduce_variance(g, axis=0) for g in self.per_example_grad if g is not None]
                    #self.grad_var_sum = [tf.math.reduce_sum(gv) for gv in self.grad_var]
                    #self.grad_norm_square = [tf.square(tf.norm(g)) for g in grads[tname] if g is not None]
                    #self.noise = [a / b for a, b in zip(self.grad_var_sum, self.grad_norm_square)]
                    self.record_grad_stats = {"grad_mean": self.grad_mean, "grad_std": self.grad_std, "grad_var": self.grad_var, "signal_to_noise": self.snr}#, "grad_var_sum": self.grad_var_sum, "grad_norm_squared": self.grad_norm_square, "noise": self.noise}

                (
                    self.clipped_grads[tname],
                    self.grad_norms[tname],
                ) = tf.clip_by_global_norm(
                    grads[tname], self.args.grad_clip_value, name="clipping_gradient",
                )
            else:
                self.tvars[tname] = self.policy_network[tname]._trainable_var_list
                self.clipped_grads[tname] = tf.gradients(
                    self.loss[tname], self.tvars[tname]
                )
            vf_tvars[tname] = self.baseline_network[tname]._trainable_var_list
            vf_grads[tname] = tf.gradients(self.vf_loss[tname], vf_tvars[tname])

            if self.args.sequential_task_sampling:
                self.per_task_update_op[tname] = utils.apply_grads_for_task(
                    self.tvars,
                    self.task_name,
                    tname,
                    self.optimizer,
                    self.clipped_grads,
                    average_logstd_grads=self.args.multitask_logstd_joint == 1,
                    average_embedding_grads="separate"
                    not in self.args.gnn_embedding_option,
                )

        self.grad_cosine = {}
        if len(self.training_tasks) > 1:
            for i in range(len(self.clipped_grads[self.training_tasks[0]])):
                g1 = tf.reshape(self.clipped_grads[self.training_tasks[0]][i], [-1])
                g2 = tf.reshape(self.clipped_grads[self.training_tasks[1]][i], [-1])
                if g1.shape == g2.shape:
                    g1 = tf.nn.l2_normalize(g1)
                    g2 = tf.nn.l2_normalize(g2)
                    self.grad_cosine[i] = tf.reduce_sum(g1 * g2)

        self.update_op, self.updated_var_names = utils.average_and_apply_grads_per_task(
            self.tvars,
            self.task_name,
            self.optimizer,
            self.clipped_grads,
            self.training_tasks,
            self.mixing_ratio_ph,
            average_logstd_grads=self.args.multitask_logstd_joint == 1,
            average_embedding_grads="separate" not in self.args.gnn_embedding_option,
        )

        self.edge_norms[tname] = {}
        if self.args.record_edge_norm:
            for var in self.policy_network[tname]._all_var_list:
                if "MLP_prop_edge" in var.name and "/w:" in var.name:
                    self.edge_norms[tname][var.name] = tf.linalg.svd(var, full_matrices=False, compute_uv=False)[0]

        self.spectral_radius[tname] = {}
        if self.args.record_spectral_radius:
            for var in self.policy_network[tname]._all_var_list:
                if ("/w:" in var.name or "/w_" in var.name) and "embedding" not in var.name and "MLP_out/layer_2" not in var.name:
                    a, _ = tf.linalg.eig(var)
                    b = tf.math.abs(a)
                    c = b[0]
                    self.spectral_radius[tname][var.name] = c  # tf.math.abs(tf.linalg.eig(var)[0])[0]

        self.per_task_update_vf_op = {}
        self.update_vf_op = {}
        if self.args.multitask_critic_joint == 1:
            self.update_vf_op = utils.average_and_apply_grads_per_task(
                vf_tvars,
                self.task_name,
                tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.args.value_lr, epsilon=1e-5
                ),
                vf_grads,
                self.training_tasks,
                self.mixing_ratio_ph,
                average_embedding_grads="separate"
                not in self.args.gnn_embedding_option,
            )
        else:
            if self.args.sequential_task_sampling:
                for tname in self.training_tasks:
                    self.per_task_update_vf_op[
                        tname
                    ] = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=self.args.value_lr, epsilon=1e-5
                    ).minimize(
                        tf.reduce_mean(
                            [self.vf_loss[tname] for tname in self.training_tasks]
                        )
                    )
            else:
                for tname in self.training_tasks:
                    self.vf_optim = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=self.args.value_lr, epsilon=1e-5
                    )
                    clipped_grads = tf.gradients(self.vf_loss[tname], vf_tvars[tname])
                    # clipped_grads, _ = tf.clip_by_global_norm(
                    #     vf_grads, self.args.grad_clip_value
                    # )
                    self.update_vf_op[tname] = self.vf_optim.apply_gradients(
                        list(zip(clipped_grads, vf_tvars[tname]))
                    )
                # self.update_vf_op = tf.compat.v1.train.AdamOptimizer(learning_rate = self.args.value_lr, epsilon = 1e-5).minimize(
                #     tf.reduce_mean([self.vf_loss[tname] for tname in self.training_tasks]))

    def get_mixing_ratio(self, prev_avg_env_reward, timesteps_to_sample):
        ctr = self.get_iteration_count() % self.sample_sizes_targets.shape[0]
        if prev_avg_env_reward > self.max_avg_return:
            self.max_avg_return = prev_avg_env_reward
        if prev_avg_env_reward < self.min_avg_return:
            self.min_avg_return = prev_avg_env_reward

        # standartize the returns, eps is needed for the initial case, when min/max are the same number
        # I keep the raw returns because otherwise we will have only ones since the returns will grow
        # with policies improving and all the targets will be ones
        # (prev_avg_env_reward - self.min_avg_return)/(self.max_avg_return - self.min_avg_return + 1e-8)
        # mixing ratio stuff############

        self.sample_sizes_targets[ctr - 1] = prev_avg_env_reward
        self.sample_sizes[ctr] = [
            timesteps_to_sample[tname] for tname in self.training_tasks
        ]
        mix_ratio = np.ones(len(self.training_tasks)) / len(self.training_tasks)
        logger.info(f"MIX RATIO: {mix_ratio}")
        return mix_ratio

    def update_ppo_parameters(self, paths, ob_normalizer_info, timesteps_to_sample):
        """
            @brief: update the ppo
        """
        avg_env_reward = np.array(
            [
                el["rewards"].sum()
                for tname in self.training_tasks
                for el in paths[tname]
            ]
        ).mean()

        # step 1: get the data dict

        feed_dicts_dict = {}  # one k,v per one task
        for tname in self.training_tasks:
            cpath = paths[tname]
            feed_dict, nervenetplus_batch_pos = self.prepared_network_feeddict(
                cpath, tname
            )
            feed_dicts_dict[tname] = feed_dict
        # step 2: train the network
        logger.info(
            "| %11s | %11s | %11s | %11s| %11s"
            % ("surr", "kl", "ent", "vf_loss", "weight_l2")
        )
        self.timesteps_so_far += sum([ts for ts in timesteps_to_sample.values()])
        grad_coses = []
        grad_norms = []

        curr_iter_mixing_ratio = self.get_mixing_ratio(
            avg_env_reward, timesteps_to_sample
        )
        if "bandit" in self.args.sample_budgeting:
            self.bandit.push_return(avg_env_reward)

        if self.args.sequential_task_sampling:
            sequential_sampling_curr_task_name = np.random.choice(self.training_tasks)
            print(f"Using task {sequential_sampling_curr_task_name} for an update")

        for i_epochs in range(self.args.optim_epochs):
            if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                # minibatch_id_candidate = list(range(len(nervenetplus_batch_pos)))
                # self._npr.shuffle(minibatch_id_candidate)
                # # make sure that only timesteps per batch is used
                # minibatch_id_candidate = minibatch_id_candidate[
                #     : int(self.args.timesteps_per_batch / self.args.gnn_num_prop_steps)
                # ]
                pass
            else:
                minibatch_id_candidate = {
                    tname: list(
                        range(
                            feed_dicts_dict[tname][
                                self.task_specific_placeholders[tname][
                                    "action_placeholder"
                                ]
                            ].shape[0]
                        )
                    )
                    for tname in self.training_tasks
                }
                for tname in self.training_tasks:
                    self._npr.shuffle(minibatch_id_candidate[tname])
                    # make sure that only timesteps per batch is used
                    minibatch_id_candidate[tname] = minibatch_id_candidate[tname][
                        : timesteps_to_sample[tname]
                    ]
            current_id = {tname: 0 for tname in self.training_tasks}
            (
                surrogate_epoch,
                kl_epoch,
                entropy_epoch,
                vf_epoch,
                weight_epoch,
                curr_grad_cos,
            ) = ([], [], [], [], [], [])
            # when at least one task is out of batches, stop iterating
            tasks_done = {tname: False for tname in self.training_tasks}

            while not np.all(list(tasks_done.values())):
                # fetch the minidata batch
                sub_feed_dict = {}
                for tname in self.training_tasks:
                    (
                        sub_feed_dict[tname],
                        current_id[tname],
                        minibatch_id_candidate[tname],
                    ) = self.construct_minibatchFeeddict_from_feeddict(
                        feed_dicts_dict[tname],
                        minibatch_id_candidate[tname],
                        current_id[tname],
                        self.args.optim_batch_size,
                        tname,
                        is_all_feed=self.args.minibatch_all_feed,
                        nervenetplus_batch_pos=nervenetplus_batch_pos,
                    )

                    if not tasks_done[tname]:
                        tasks_done[tname] = not (
                            current_id[tname] + self.args.optim_batch_size
                            <= len(minibatch_id_candidate[tname])
                            and current_id[tname] >= 0
                        )

                # test mixing the grads on the uniform distribution
                # there is no learning yet and the performance should
                # be similar to the naive implementation with just tf.mean
                mega_sub_feed_dict = {self.mixing_ratio_ph: curr_iter_mixing_ratio}
                for v in sub_feed_dict.values():
                    mega_sub_feed_dict.update(v)
                # train for one iteration in this epoch

                if self.args.sequential_task_sampling:
                    (
                        _,
                        i_surrogate_mini,
                        i_kl_mini,
                        i_entropy_mini,
                        # i_weight_mini,
                    ) = self.session.run(
                        [
                            self.per_task_update_op[sequential_sampling_curr_task_name],
                            self.surr[sequential_sampling_curr_task_name],
                            self.kl[sequential_sampling_curr_task_name],
                            self.ent[sequential_sampling_curr_task_name],
                            # self.weight_decay_loss,
                        ],
                        feed_dict=mega_sub_feed_dict,
                    )
                    i_weight_mini = 1.0
                    # train the value network with fixed network coeff
                    _, i_vf_mini = self.session.run(
                        [
                            self.per_task_update_vf_op[
                                sequential_sampling_curr_task_name
                            ],
                            self.vf_loss[sequential_sampling_curr_task_name],
                        ],
                        feed_dict=mega_sub_feed_dict,
                    )
                    i_surrogate_mini = {
                        sequential_sampling_curr_task_name: i_surrogate_mini
                    }
                    i_kl_mini = {sequential_sampling_curr_task_name: i_kl_mini}
                    i_entropy_mini = {
                        sequential_sampling_curr_task_name: i_entropy_mini
                    }
                    i_vf_mini = {sequential_sampling_curr_task_name: i_vf_mini}
                else:
                    to_run = [
                        self.update_op,
                        self.policy_network[self.task_name].norm_update_ops if self.args.batchnorm else {},
                        self.surr,
                        self.kl,
                        self.ent,
                        self.weight_decay_loss,
                        self.grad_cosine,
                    ]
                    if len(self.grad_norms) > 0:
                        to_run.append(self.grad_norms)
                        (
                            _,
                            _,
                            i_surrogate_mini,
                            i_kl_mini,
                            i_entropy_mini,
                            i_weight_mini,
                            grad_cos,
                            i_grad_norm,
                        ) = self.session.run(to_run, feed_dict=mega_sub_feed_dict)
                        grad_norms.append(i_grad_norm)
                    else:
                        (
                            _,
                            _,
                            i_surrogate_mini,
                            i_kl_mini,
                            i_entropy_mini,
                            i_weight_mini,
                            grad_cos,
                        ) = self.session.run(to_run, feed_dict=mega_sub_feed_dict)

                    curr_grad_cos.append(list(grad_cos.values()))
                    # train the value network with fixed network coeff
                    _, i_vf_mini = self.session.run(
                        [self.update_vf_op, self.vf_loss], feed_dict=mega_sub_feed_dict
                    )

                # I wonder if this is the correct place to do that?
                # Should we do that as a placeholder feeding the weights?
                # sync non-blueprint policies/baselines at the beginning
                for sp in self.set_policy.values():
                    sp(
                        self.get_policy[self.task_name](
                            skip_logstd=self.args.multitask_logstd_joint == 0,
                            skip_embedding="separate" in self.args.gnn_embedding_option,
                        ),
                        skip_logstd=self.args.multitask_logstd_joint == 0,
                        skip_embedding="separate" in self.args.gnn_embedding_option,
                    )
                if self.args.multitask_critic_joint == 1:
                    for sb in self.sync_baselines:
                        sb(self.get_blueprint_baseline())
                surrogate_epoch.append(sum(i_surrogate_mini.values()))
                kl_epoch.append(sum(i_kl_mini.values()))
                entropy_epoch.append(sum(i_entropy_mini.values()))
                vf_epoch.append(sum(i_vf_mini.values()))
                weight_epoch.append(i_weight_mini)

            surrogate_epoch = np.mean(surrogate_epoch)
            kl_epoch = np.mean(kl_epoch)
            entropy_epoch = np.mean(entropy_epoch)
            vf_epoch = np.mean(vf_epoch)
            weight_epoch = np.mean(weight_epoch)
            grad_coses.extend(curr_grad_cos)

            logger.info(
                "| %10.8f | %10.8f | %10.4f | %10.4f | %10.4f"
                % (surrogate_epoch, kl_epoch, entropy_epoch, vf_epoch, weight_epoch)
            )
            if len(self.training_tasks) > 1:
                logger.info(f"Mean grad cosine similarity: {np.mean(curr_grad_cos)}")
        if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
            i_surrogate_total = surrogate_epoch
            i_kl_total = kl_epoch
            i_entropy_total = entropy_epoch
            i_vf_total = vf_epoch
            i_weight_total = weight_epoch
        else:
            (
                record_grad_stats,
                i_surrogate_total,
                i_kl_total,
                i_entropy_total,
                i_vf_total,
                i_weight_total,
                i_encoder_weight_total,
                i_edge_weight_total,
                i_decoder_weight_total,
                ad_logstd,
                edge_norms,
                spectral_radius,
                clip_stats,
                loss,
                mlp_stats,
                batchnorm_stats,
                vib_encoder_loss,
                vib_edge_loss,
                vib_decoder_loss,
                mean_vib_mu, var_vib_mu, mean_vib_rho, var_vib_rho,
                oversmoothing_stats,
            ) = self.session.run(
                [
                    self.record_grad_stats if self.args.record_grads else {},
                    self.surr,
                    self.kl,
                    self.ent,
                    self.vf_loss,
                    self.weight_decay_loss,
                    self.encoder_weight_decay_loss,
                    self.edge_weight_decay_loss,
                    self.decoder_weight_decay_loss,
                    {task: v for task, v in self.action_dist_logstd.items() if task in self.training_tasks},
                    self.edge_norms if self.args.record_edge_norm else {},
                    self.spectral_radius if self.args.record_spectral_radius else {},
                    self.clip_stats if self.args.record_ppo_clip else {},
                    self.loss,
                    self.policy_network[self.task_name].mlp_stats if self.args.record_mlp_activation else {},
                    self.policy_network[self.task_name].norm_moving_averages if self.args.batchnorm else {},
                    self.vib_encoder_loss if self.args.vib_updater else {},
                    self.vib_edge_loss if self.args.vib_updater else {},
                    self.vib_decoder_loss if self.args.vib_updater else {},
                    self.mean_vib_edge_mu if self.args.vib_updater else {},
                    self.var_vib_edge_mu if self.args.vib_updater else {},
                    self.mean_vib_edge_rho if self.args.vib_updater else {},
                    self.var_vib_edge_rho if self.args.vib_updater else {},
                    self.policy_network[self.task_name].oversmoothing_stats if self.args.record_oversmoothing else {},
                ],
                feed_dict=mega_sub_feed_dict,
            )

        self.update_adaptive_hyperparams(i_kl_total)
        stats = defaultdict(dict)
        for tname in self.training_tasks:
            stats[tname]["entropy"] = i_entropy_total[tname]
            stats[tname]["kl"] = i_kl_total[tname]
            stats[tname]["surr_loss"] = -i_surrogate_total[tname].mean()
            stats[tname]["vf_loss"] = i_vf_total[tname]
            stats[tname]["avg_reward"] = np.array(
                [
                    el["rewards"].sum()
                    for v in paths.values()
                    for el in v
                    if el["task"] == tname
                ]
            ).mean()
            #stats[tname]["sample_size"] = timesteps_to_sample[tname]
            if len(grad_norms) > 0:
                stats[tname]["grad_norm"] = np.mean([el[tname] for el in grad_norms])
            #for i, action in self.action_dict[tname].items():
            #    logstd = ad_logstd[tname].mean(0)[i]
            #    std = np.exp(logstd)
            #    stats[tname][f"logstd_{action}"]=logstd
            #    stats[tname][f"std_{action}"]=std
            #for r_type in ["forward", "ctrl", "contact", "survive"]:
            #     stats[tname][f"avg_reward_{r_type}"] = np.mean(
            #         [np.sum(path["reward_breakdown"][f"reward_{r_type}"]) for path in paths[tname]]
            #     )
            #act_mags, joint_r2s = [], []
            #for rollout in paths[tname]:
            #    if rollout["task"] == tname:
            #        act_mag, joint_r2 = self.get_action_metrics(rollout)
            #        act_mags.append(act_mag)
            #        joint_r2s.append(joint_r2)
            #stats[tname]["mean_action_magnitude"] = np.mean(act_mags)
            #stats[tname]["mean_joint_r2"] = np.mean(joint_r2s)
            if self.args.record_grads:
                param_groups = [
                    #'MLP_embedding_node_type_root.*/w', 'MLP_embedding_node_type_joint.*/w',
                    'MLP_prop_edge.*/layer_0/w', 'MLP_prop_edge.*/layer_1/w',
                    'GRU_node_root.*/w_hi', 'GRU_node_root.*/w_hr', 'GRU_node_root.*/w_hu',
                    'GRU_node_root.*/w_xi', 'GRU_node_root.*/w_xr', 'GRU_node_root.*/w_xu',
                    'GRU_node_joint.*/w_hi', 'GRU_node_joint.*/w_hr', 'GRU_node_joint.*/w_hu',
                    'GRU_node_joint.*/w_xi', 'GRU_node_joint.*/w_xr', 'GRU_node_joint.*/w_xu',
                    #'MLP_out.*/layer_0', 'MLP_out.*/layer_1',
                ]
                for stat_name, stat_data in record_grad_stats.items():
                    for group in param_groups:
                        values = []
                        for data_name, data in zip(self.updated_var_names, stat_data):
                            if re.search(group, data_name):
                                values.append(data)
                        stats[tname][f"record_{stat_name}_{group}"] = np.mean(values)
            if self.args.record_oversmoothing:
                for layer, data in oversmoothing_stats.items():
                    for node_type, data_ in data.items():
                        for stat, v in data_.items():
                            stats[tname][f"GRU_stats_{node_type}_{stat}_{layer}"] = v
            if self.args.record_edge_norm:
                for edge_weight_name, norm_value in edge_norms[tname].items():
                    stats[tname][f"edge_norm--{edge_weight_name}"] = norm_value
            if self.args.record_spectral_radius:
                for weight_name, sr in spectral_radius[tname].items():
                    stats[tname][f"spectral_radius--{weight_name}"] = sr
            if self.args.record_ppo_clip:
                for clip_stat, v in clip_stats[tname].items():
                    stats[tname][f"ppo_clip_stats--{clip_stat}"] = v
            stats[tname][f"loss"] = -loss[tname].mean()
            if self.args.record_mlp_activation:
                one_edge_type = len(mlp_stats.values()) == 1
                for i_edge_type, _mlp_stats in mlp_stats.items():
                    for tt, __mlp_stats in _mlp_stats.items():
                        layers_shared = not self.args.no_sharing_between_layers
                        for layer, ___mlp_stats in __mlp_stats.items():
                            extra_name = ""
                            if not one_edge_type:
                                extra_name = f"{extra_name}{i_edge_type}-"
                            if not layers_shared:
                                extra_name = f"{extra_name}{tt}-"
                            extra_name = f"{extra_name}{layer}-"
                            for stat_name, data in ___mlp_stats.items():
                                stats[tname][f"mlp-stats-{extra_name}{stat_name}"] = data
                        if self.args.batchnorm:
                            for norm_name, norm_data in batchnorm_stats[i_edge_type][tt].items():
                                extra_name = ""
                                if not one_edge_type:
                                    extra_name = f"{extra_name}{i_edge_type}-"
                                if not layers_shared:
                                    extra_name = f"{extra_name}{tt}-"
                                extra_name = f"{extra_name}{norm_name}-"
                                for stat_name, stat_data in norm_data.items():
                                    stats[tname][f"batchnorm-{extra_name}moving-{stat_name}"] = stat_data
                        if not self.args.no_sharing_between_layers:
                            break
            if self.args.vib_updater:
                stats[tname][f"vib_encoder_loss"] = vib_encoder_loss[tname]
                stats[tname][f"vib_edge_loss"] = vib_edge_loss[tname]
                stats[tname][f"vib_decoder_loss"] = vib_decoder_loss[tname]
                stats[tname][f"vib_mean_mu"] = mean_vib_mu
                stats[tname][f"vib_var_mu"] = var_vib_mu
                stats[tname][f"vib_mean_rho"] = mean_vib_rho
                stats[tname][f"vib_var_rho"] = var_vib_rho

        stats["all_envs"] = {}
        stats["all_envs"]["weight_l2_loss"] = i_weight_total
        # if self.args.use_edge_weight_decay:
        #     stats["all_envs"]["encoder_weight_l2_loss"] = i_encoder_weight_total
        #     stats["all_envs"]["edge_weight_l2_loss"] = i_edge_weight_total
        #     stats["all_envs"]["decoder_weight_l2_loss"] = i_decoder_weight_total
        stats["all_envs"]["avg_reward"] = avg_env_reward

        #stats["all_envs"]["entropy"] = np.mean(list(i_entropy_total.values()))
        #stats["all_envs"]["kl"] = np.mean(list(i_kl_total.values()))
        #stats["all_envs"]["surr_loss"] = np.mean(list(i_surrogate_total.values()))
        #stats["all_envs"]["vf_loss"] = np.mean(list(i_vf_total.values()))
        stats["all_envs"]["total_steps"] = self.timesteps_so_far
        cos_mean = np.mean(grad_coses)
        #stats["all_envs"]["grad_cosine"] = (
        #    cos_mean if not np.isnan(cos_mean) else 1.0
        #)  # 1.0 if we have one task only

        for tname in self.test_transfer_tasks:
            stats['test_transfer']['{}_reward'.format(tname)] = np.array(
                [
                    el["rewards"].sum()
                    for v in paths.values()
                    for el in v
                    if el["task"] == tname
                ]
            ).mean()

        train_ob_normalizer_info = {t: info for t, info in ob_normalizer_info.items() if t in self.training_tasks}
        self.record_summary_and_ckpt(paths, stats, train_ob_normalizer_info)

        return stats

    def get_action_metrics(self, rollout_data):
        act_means, act_corr = [], []
        for i, part in enumerate(["hip", "body_", "ankle", "bodyupdown"]):
            data = {}
            for j, joint_type in self.action_dict[self.task_name].items():
                if part in joint_type:
                    part_data = rollout_data["action_dists_mu"][:, j]
                    if part is "hip" and int(joint_type.split("_")[-1]) % 2 == 0:
                        part_data *= -1
                    data[joint_type] = part_data
            df = pd.DataFrame(data)
            corr = df.corr()
            np.fill_diagonal(corr.values, 0)
            act_means.append(df.abs().mean().mean())
            act_corr.append(np.square(corr).mean().mean())

        return np.mean(act_means), np.mean(act_corr)

    def update_adaptive_hyperparams(self, i_kl_total):
        if self.args.lr_schedule == "adaptive":
            mean_kl = np.mean(list(i_kl_total.values()))
            if mean_kl > self.args.target_kl_high * self.args.target_kl:
                self.current_lr /= self.args.lr_alpha
            if mean_kl < self.args.target_kl_low * self.args.target_kl:
                self.current_lr *= self.args.kl_alpha

            self.current_lr = max(self.current_lr, 3e-10)
            self.current_lr = min(self.current_lr, 1e-2)
        elif self.args.lr_schedule == "linear":
            self.current_lr = self.args.lr * max(
                1.0 - float(self.timesteps_so_far) / self.args.max_timesteps, 0.0
            )

    def record_summary_and_ckpt(self, paths, stats, ob_normalizer_info):
        # logger the information and write summary
        for k, v in stats.items():
            logger.info(k + ": " + " " * (40 - len(k)) + str(v))

        current_iteration = self.get_iteration_count()

        if current_iteration % self.args.min_ckpt_iteration_diff == 0:
            logger.info("------------- Printing hyper-parameters -----------")
            for key, val in self.args.__dict__.items():
                logger.info("{}: {}".format(key, val))
            logger.info("experiment name: ".format(self.get_experiment_name()))

        # add the summary
        if current_iteration % self.args.summary_freq == 0:
            for key, val in stats.items():
                for tk, tv in val.items():
                    if type(tv) == list:
                        for i in range(len(tv)):
                            self.summary_writer.manually_add_scalar_summary(
                                f"{key}/{tk}_{i}", tv[i], current_iteration
                            )
                    else:
                        self.summary_writer.manually_add_scalar_summary(
                            f"{key}/{tk}", tv, current_iteration
                        )

        # save the model if needed
        current_reward = stats["all_envs"]["avg_reward"]
        if self.best_reward < current_reward:
            self.best_reward = current_reward
            if (
                    (current_iteration > self.args.checkpoint_start_iteration
                     and current_iteration - self.last_save_iteration
                     > self.args.min_ckpt_iteration_diff)
                    or (current_iteration == self.args.checkpoint_start_iteration)
            ):
                self.save_all(current_iteration, ob_normalizer_info)
                self.last_save_iteration = current_iteration

        if self.best_reward > -np.inf:  # it means that we updated it
            logger.info("Current max reward: {}".format(self.best_reward))

        self.session.run(self.iteration_add_op)

    def save_all(self, current_iteration, ob_normalizer_info={}):
        """
            @brief: save the model into several npy file.
        """
        model_name = self.get_output_path(save=True)
        # save multiple models:
        model_name += "_" + str(current_iteration)

        # save the parameters of the policy network, baseline network
        # and the ob_normalizer
        self.policy_network[self.task_name].save_checkpoint(model_name + "_policy.npy")
        self.baseline_network[self.task_name].save_checkpoint(model_name + "_baseline.npy")
        model_saver.save_numpy_model(model_name + "_normalizer.npy", ob_normalizer_info)
        model_saver.save_tf_model(self.session, model_name + "_policy_opt.npy", self.optimizer['main'].variables())
        if self.args.separate_node_lr > -0.5: model_saver.save_tf_model(self.session, model_name + "_policy_opt_node.npy", self.optimizer['node'].variables())
        if self.args.separate_edge_lr > -0.5: model_saver.save_tf_model(self.session, model_name + "_policy_opt_edge.npy", self.optimizer['edge'].variables())
        if self.args.separate_encoder_lr > -0.5: model_saver.save_tf_model(self.session, model_name + "_policy_opt_encoder.npy", self.optimizer['encoder'].variables())
        if self.args.separate_decoder_lr > -0.5: model_saver.save_tf_model(self.session, model_name + "_policy_opt_decoder.npy", self.optimizer['decoder'].variables())
        model_saver.save_tf_model(self.session, model_name + "_vf_opt.npy", self.vf_optim.variables())

        logger.info("[SAVE_CKPT] saving files under the name of {}".format(model_name))

    def restore_all(self):
        """
            @brief: restore the parameters
        """
        model_name = self.get_output_path(save=False)

        # load the parameters one by one
        self.policy_network[self.task_name].load_checkpoint(
            model_name + "_policy.npy",
            transfer_env=self.args.transfer_env,
            logstd_option=self.args.logstd_option,
            gnn_option_list=[
                self.args.gnn_node_option,
                self.args.root_connection_option,
                self.args.gnn_output_option,
                self.args.gnn_embedding_option,
            ],
            mlp_raw_transfer=self.args.mlp_raw_transfer,
        )
        self.baseline_network[self.task_name].load_checkpoint(
            model_name + "_baseline.npy"
        )
        self.last_save_iteration = self.get_iteration_count()

        if (not self.args.test) and self.args.optim_load:
            model_saver.load_tf_model(self.session, model_name + "_policy_opt.npy", tf_var_list=self.optimizer["main"].variables())
            if self.args.separate_node_lr > -0.5: model_saver.load_tf_model(self.session, model_name + "_policy_opt_node.npy", tf_var_list=self.optimizer["node"].variables())
            if self.args.separate_edge_lr > -0.5: model_saver.load_tf_model(self.session, model_name + "_policy_opt_edge.npy", tf_var_list=self.optimizer["edge"].variables())
            if self.args.separate_encoder_lr > -0.5: model_saver.load_tf_model(self.session, model_name + "_policy_opt_encoder.npy", tf_var_list=self.optimizer["encoder"].variables())
            if self.args.separate_decoder_lr > -0.5: model_saver.load_tf_model(self.session, model_name + "_policy_opt_decoder.npy", tf_var_list=self.optimizer["decoder"].variables())
            model_saver.load_tf_model(self.session, model_name + "_vf_opt.npy", tf_var_list=self.vf_optim.variables())

        logger.info("[LOAD_CKPT] saving files under the name of {}".format(model_name))
        logger.info("[LOAD_CKPT] The normalizer is loaded in the rollout agents")

    def get_output_path(self, save=True):
        if save:
            if self.args.output_dir is None:
                path = init_path.get_base_dir()
                path = os.path.abspath(path)
            else:
                path = os.path.abspath(self.args.output_dir)
            base_path = os.path.join(path, "checkpoint")
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            model_name = os.path.join(base_path, self.get_experiment_name())
        else:
            path = self.args.ckpt_name
            model_name = path
        return model_name

    def generate_advantage(self, data_dict, feed_dict, task):
        """
            @brief: calculate the parameters for the advantage function
        """
        # get the baseline function
        if self.args.value_network_type == NetworkTypes.nervenet.name:
            baseline_data = self.baseline_network[task].predict(feed_dict)
            current_id = 0
            for path in data_dict:
                path["baseline"] = baseline_data[
                    current_id : current_id + len(path["rewards"])
                ]
                current_id += len(path["rewards"])

            assert current_id == len(baseline_data), logger.error(
                "Extra baseline predicted? ({} vs {})".format(
                    current_id, len(baseline_data)
                )
            )
        else:
            for path in data_dict:
                # the predicted value function (baseline function)
                path["baseline"] = self.baseline_network[task].predict(path)

        # esitmate the advantages
        if self.args.advantage_method == "raw":
            for path in data_dict:
                # the gamma discounted rollout value function
                path["returns"] = utils.discount(path["rewards"], self.args.gamma)
                path["advantage"] = path["returns"] - path["baseline"]
                path["target_return"] = path["returns"]
        else:
            for path in data_dict:
                # the gamma discounted rollout value function
                path["returns"] = utils.discount(path["rewards"], self.args.gamma)

                # init the advantage
                path["advantage"] = np.zeros(path["returns"].shape)

                num_steps = len(path["returns"])

                # generate the GAE advantage
                for i_step in reversed(list(range(num_steps))):
                    if i_step < num_steps - 1:
                        delta = (
                            path["rewards"][i_step]
                            + self.args.gamma * path["baseline"][i_step + 1]
                            - path["baseline"][i_step]
                        )
                        path["advantage"][i_step] = (
                            delta
                            + self.args.gamma
                            * self.args.gae_lam
                            * path["advantage"][i_step + 1]
                        )
                    else:
                        delta = path["rewards"][i_step] - path["baseline"][i_step]
                        path["advantage"][i_step] = delta

                path["target_return"] = path["advantage"] + path["baseline"]

        # standardized advantage function
        advant_n = np.concatenate([path["advantage"] for path in data_dict])

        if self.args.advantage_normalization:
            advant_n -= advant_n.mean()
            advant_n /= advant_n.std() + 1e-8  # standardize to mean 0 stddev 1
        return advant_n

    def prepared_network_feeddict(self, data_dict, task):
        """
            @brief:
                For the graph neural network, we will need to change the size
                from [batch_size, ob_dim] into [batch_size * node_size, ob_dim].
            @return:
                @1. feed_dict for trpo or ppo update
                    @self.action_placeholder
                    @self.advantage_placeholder
                    @self.oldaction_dist_mu_placeholder
                    @self.oldaction_dist_logstd_placeholder
                    @self.batch_size_float_placeholder

                    # baseline function
                    @self.target_return_placeholder (for ppo only)

                @2. feed_dict for generating the policy action
                    @self.obs_placeholder

                    # for ggnn only
                    @self.graph_obs_placeholder
                    @self.graph_parameters_placeholder
                    @self.batch_size_int_placeholder
                    @self.receive_idx_placeholder
                    @self.send_idx_placeholder
                    @self.inverse_node_type_idx_placeholder
                    @self.output_idx_placeholder

                @3. feed_dict for baseline if baseline is a fc-policy and policy is
                    a ggnn policy
                    @self.raw_obs_placeholder

                NOTE: for minibatch training, check @prepare_feed_dict_map and
                    @construct_minibatchFeeddict_from_feeddict
        """
        # step 1: For ggnn and fc policy, we have different obs in feed_dict
        feed_dict, nervenetplus_batch_pos = self.prepared_policy_network_feeddict(
            [path["obs"] for path in data_dict],
            self.node_info[task] if self.node_info else None,
            data_dict,
            task=task,
        )
        # step 2: prepare the advantage function, old action / obs needed for
        # trpo / ppo updates
        advant_n = self.generate_advantage(data_dict, feed_dict, task)
        action_dist_mu = np.concatenate([path["action_dists_mu"] for path in data_dict])
        action_dist_logstd = np.concatenate(
            [path["action_dists_logstd"] for path in data_dict]
        )
        action_n = np.concatenate([path["actions"] for path in data_dict])

        feed_dict.update(
            {
                self.task_specific_placeholders[task]["action_placeholder"]: action_n,
                self.task_specific_placeholders[task][
                    "advantage_placeholder"
                ]: advant_n,
                self.task_specific_placeholders[task][
                    "oldaction_dist_mu_placeholder"
                ]: action_dist_mu,
                self.task_specific_placeholders[task][
                    "oldaction_dist_logstd_placeholder"
                ]: action_dist_logstd,
                self.task_specific_placeholders[task][
                    "batch_size_float_placeholder"
                ]: np.array(float(len(action_n))),
            }
        )

        # step 3: feed_dict to update value function
        target_return = np.concatenate([path["target_return"] for path in data_dict])
        feed_dict.update({self.target_return_placeholder[task]: target_return})
        feed_dict.update({self.lr_placeholder: self.current_lr})
        if self.args.separate_node_lr > -0.5: feed_dict.update({self.separate_node_lr_placeholder: self.current_node_lr})
        if self.args.separate_edge_lr > -0.5: feed_dict.update({self.separate_edge_lr_placeholder: self.current_edge_lr})
        if self.args.separate_encoder_lr > -0.5: feed_dict.update({self.separate_encoder_lr_placeholder: self.current_encoder_lr})
        if self.args.separate_decoder_lr > -0.5: feed_dict.update({self.separate_decoder_lr_placeholder: self.current_decoder_lr})
        if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
            self._input_hidden_state = (
                self.policy_network.get_input_hidden_state_placeholder()
            )
            for node_type in self.node_info[self.task_name]["node_type_dict"]:
                feed_dict[self._input_hidden_state[node_type]] = np.concatenate(
                    [path["hidden_state"][node_type] for path in data_dict]
                )

        return feed_dict, nervenetplus_batch_pos

    def construct_minibatchFeeddict_from_feeddict(
        self,
        feed_dict,
        minibatch_id_candidate,
        current_id,
        batch_size,
        task_name,
        is_all_feed=False,
        nervenetplus_batch_pos=None,
    ):

        if (
            is_all_feed
            and self.args.policy_network_type == NetworkTypes.nervenetpp.name
        ):
            raise NotImplementedError

        sub_feed_dict = {}
        if (
            not is_all_feed
            or self.args.policy_network_type == NetworkTypes.nervenetpp.name
        ):

            if (
                not is_all_feed
                and not self.args.policy_network_type == NetworkTypes.nervenetpp.name
            ):
                assert batch_size <= len(minibatch_id_candidate), logger.error(
                    "Please increse the rollout size!"
                )
                if current_id + batch_size > len(minibatch_id_candidate):
                    logger.warning(
                        "Shuffling the ids. This is okay in MT regime with uneven sampling"
                    )
                    current_id = 0
                    self._npr.shuffle(minibatch_id_candidate)
                candidate_id = minibatch_id_candidate[
                    current_id : current_id + batch_size
                ]
            elif (
                not is_all_feed
                and self.args.policy_network_type == NetworkTypes.nervenetpp.name
            ):
                pass
                # assert batch_size <= len(minibatch_id_candidate), logger.error(
                #     "Please increase the rollout size!"
                # )
                #
                # if current_id + batch_size > len(minibatch_id_candidate):
                #     logger.warning("shuffling the ids")
                #     current_id = 0
                #     self._npr.shuffle(minibatch_id_candidate)
                #
                # nervenetplus_candidate_id = [
                #     nervenetplus_batch_pos[i_id]
                #     for i_id in minibatch_id_candidate[
                #         current_id : current_id
                #         + int(batch_size / self.args.gnn_num_prop_steps)
                #     ]
                # ]
                # candidate_id = []
                #
                # for chosen_pos in nervenetplus_candidate_id:
                #     candidate_id.extend(
                #         list(
                #             range(chosen_pos, chosen_pos + self.args.gnn_num_prop_steps)
                #         )
                #     )

            # step 0: nervenet util
            if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                pass
                # for node_type in self.node_info[task_name]["node_type_dict"]:
                #     node_num = len(
                #         self.node_info[task_name]["node_type_dict"][node_type]
                #     )
                #     # node_type_candidate_id = [i_id * node_num for i_id in nervenetplus_candidate_id]
                #     node_type_candidate_id = []
                #     for i_id in nervenetplus_candidate_id:
                #         node_type_candidate_id.extend(
                #             list(range(i_id * node_num, (i_id + 1) * node_num))
                #         )
                #     sub_feed_dict[self._input_hidden_state[node_type]] = feed_dict[
                #         self._input_hidden_state[node_type]
                #     ][node_type_candidate_id]

            # step 0: id validation and id picking
            # assert batch_size <= len(minibatch_id_candidate), logger.error(
            #     "Please increse the rollout size!"
            # )
            #
            # if current_id + batch_size > len(minibatch_id_candidate):
            #     logger.warning("shuffling the ids")
            #     current_id = 0
            #     self._npr.shuffle(minibatch_id_candidate)
            # candidate_id = minibatch_id_candidate[current_id : current_id + batch_size]

            # step 1: gather sub feed dict by sampling from the feed_dict
            for key in self.batch_feed_dict_key[task_name]:
                sub_feed_dict[key] = feed_dict[key][candidate_id]
            # step 2: update the graph_batch_feed_dict
            if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
                for node_type in self.node_info[task_name]["node_type_dict"]:
                    num_nodes = len(
                        self.node_info[task_name]["node_type_dict"][node_type]
                    )
                    graph_candidate_id = [
                        list(range(i_id * num_nodes, (i_id + 1) * num_nodes))
                        for i_id in candidate_id
                    ]

                    # flatten the ids
                    graph_candidate_id = sum(graph_candidate_id, [])
                    for key in self.graph_batch_feed_dict_key[task_name]:
                        sub_feed_dict[key[node_type]] = feed_dict[key[node_type]][
                            graph_candidate_id
                        ]

            # step 3: static elements which are invariant to batch_size
            sub_feed_dict.update(self.static_feed_dict[task_name])

            # step 4: update dynamical elements
            for key in self.dynamical_feed_dict_key[task_name]:
                sub_feed_dict[key] = feed_dict[key]

            # step 6: get the graph index feed_dict
            if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
                sub_feed_dict.update(self.graph_index_feed_dict[task_name])

            # step 5: update current id
            if not self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                current_id = current_id + batch_size  # update id
            else:
                current_id = current_id + int(batch_size / self.args.gnn_num_prop_steps)
        else:
            # feed the whole feed_dictionary into the network
            sub_feed_dict = feed_dict
            current_id = -1  # -1 means invalid

        return sub_feed_dict, current_id, minibatch_id_candidate

    def prepare_feed_dict_map(self, task):
        """
            @brief:
                When trying to get the sub diction in
                @construct_minibatchFeeddict_from_feeddict, some key are just
                directly transferable. While others might need some other work

                @1. feed_dict for trpo or ppo update

                    # baseline function

                @2. feed_dict for generating the policy action

                    # for ggnn only
                @3. feed_dict for baseline if baseline is a fc-policy and policy
                    is a ggnn policy

            @return:
                @self.batch_feed_dict_key:
                    Shared between the fc policy network and ggnn network.
                    Most of them are only used for the update.

                        @self.action_placeholder
                        @self.advantage_placeholder
                        @self.oldaction_dist_mu_placeholder
                        @self.oldaction_dist_logstd_placeholder

                        (if use fc policy)
                        @self.obs_placeholder

                        (if use_ggn and baseline not gnn)
                        @self.raw_obs_placeholder

                        (if use tf baseline)
                        @self.target_return_placeholder (for ppo only)

                @self.graph_batch_feed_dict_key
                    Used by the ggnn. This feed_dict key list is a little bit
                    different from @self.batch_feed_dict_key if we want to do
                    minibatch

                        @self.graph_obs_placeholder
                        @self.graph_parameters_placeholder

                @self.static_feed_dict:
                    static elements that are set by optim parameters. These
                    parameters are set differently between minibatch_all_feed
                    equals 0 / equals 1

                        @self.batch_size_float_placeholder
                        @self.batch_size_int_placeholder

                @self.dynamical_feed_dict_key:
                    elements that could be changing from time to time

                        @self.lr_placeholder

                @self.graph_index_feed_dict:
                    static index for the ggnn.

                        @self.receive_idx_placeholder
                        @self.inverse_node_type_idx_placeholder
                        @self.output_idx_placeholder
                        @self.send_idx_placeholder[i_edge]
                        @self.node_type_idx_placeholder[i_node_type]

        """
        # step 1: gather the key for batch_feed_dict
        self.batch_feed_dict_key[task] = [
            self.task_specific_placeholders[task]["action_placeholder"],
            self.task_specific_placeholders[task]["advantage_placeholder"],
            self.task_specific_placeholders[task]["oldaction_dist_mu_placeholder"],
            self.task_specific_placeholders[task]["oldaction_dist_logstd_placeholder"],
        ]

        if self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
            self.batch_feed_dict_key[task].append(self.obs_placeholder[task])

        if (
            self.args.policy_network_type != NetworkTypes.fixedsizenet.name
            and self.args.value_network_type == NetworkTypes.fixedsizenet.name
        ):
            self.batch_feed_dict_key[task].append(self.raw_obs_placeholder[task])

        self.batch_feed_dict_key[task].append(self.target_return_placeholder[task])

        # step 2: gather the graph batch feed_dict
        self.graph_batch_feed_dict_key[task] = []
        if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
            self.graph_batch_feed_dict_key[task].extend(
                [
                    self.graph_obs_placeholder[task],
                    self.graph_parameters_placeholder[task],
                ]
            )

        # step 2: gather the static feed_dictionary
        self.static_feed_dict[task] = {
            self.task_specific_placeholders[task][
                "batch_size_float_placeholder"
            ]: np.array(float(self.args.optim_batch_size))
        }
        if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
            self.static_feed_dict[task].update(
                {self.batch_size_int_placeholder[task]: self.args.optim_batch_size}
            )

        # step 3: gather the dynamical feed_dictionary
        self.dynamical_feed_dict_key[task] = []
        self.dynamical_feed_dict_key[task].append(self.lr_placeholder)
        if self.args.separate_node_lr > -0.5: self.dynamical_feed_dict_key[task].append(self.separate_node_lr_placeholder)
        if self.args.separate_edge_lr > -0.5: self.dynamical_feed_dict_key[task].append(self.separate_edge_lr_placeholder)
        if self.args.separate_encoder_lr > -0.5: self.dynamical_feed_dict_key[task].append(self.separate_encoder_lr_placeholder)
        if self.args.separate_decoder_lr > -0.5: self.dynamical_feed_dict_key[task].append(self.separate_decoder_lr_placeholder)

        # step 4: gather the graph_index feed_dict
        if not self.args.policy_network_type == NetworkTypes.fixedsizenet.name:
            if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                assert self.args.optim_batch_size % self.args.gnn_num_prop_steps == 0
                dummy_obs = np.zeros(
                    [int(self.args.optim_batch_size / self.args.gnn_num_prop_steps), 10]
                )
            else:
                # construct a dummy obs to pass the batch size info
                dummy_obs = np.zeros([self.args.optim_batch_size, 10])

            node_info = self.node_info[task]

            # get the index for minibatches
            (
                _,
                _,
                receive_idx,
                send_idx,
                node_type_idx,
                inverse_node_type_idx,
                output_type_idx,
                inverse_output_type_idx,
                _,
            ) = graph_data_util.construct_graph_input_feeddict(
                node_info, dummy_obs, -1, -1, -1, -1, -1, -1, -1, request_data=["idx"]
            )
            self.graph_index_feed_dict[task] = {
                self.receive_idx_placeholder[task]: receive_idx,
                self.inverse_node_type_idx_placeholder[task]: inverse_node_type_idx,
                self.inverse_output_type_idx_placeholder[task]: inverse_output_type_idx,
            }

            # append the send idx
            for i_edge in node_info["edge_type_list"]:
                self.graph_index_feed_dict[task][
                    self.send_idx_placeholder[task][i_edge]
                ] = send_idx[i_edge]

            # append the node type idx
            for i_node_type in node_info["node_type_dict"]:
                self.graph_index_feed_dict[task][
                    self.node_type_idx_placeholder[task][i_node_type]
                ] = node_type_idx[i_node_type]

            # append the node type idx
            for i_output_type in node_info["output_type_dict"]:
                self.graph_index_feed_dict[task][
                    self.output_type_idx_placeholder[task][i_output_type]
                ] = output_type_idx[i_output_type]

    def get_sampling_sizes(self):
        if self.args.sample_budgeting == "naive":
            sampling_mults = [1.0 for _ in self.tasks]
        elif self.args.sample_budgeting == "regression":
            # do a GP regression here
            REGR_MIN_SAMPLES = 10
            SAMPLING_LOWER_BOUND = 0.5
            OPT_PTS = 100
            ctr = (
                0
                if self.session is None
                else self.get_iteration_count() % self.sample_sizes_targets.shape[0]
            )
            if ctr < REGR_MIN_SAMPLES + 1:
                sampling_mults = np.random.uniform(
                    SAMPLING_LOWER_BOUND, 1, len(self.tasks)
                )
            else:
                st_pt = ctr - min(OPT_PTS, ctr)
                X = self.sample_sizes[st_pt : ctr - 1] / self.args.timesteps_per_batch
                y = (
                    np.concatenate(
                        [
                            self.sample_sizes_targets[st_pt - 1 :][:1],
                            self.sample_sizes_targets[st_pt : ctr - 1],
                        ]
                    )
                    - self.min_avg_return
                ) / (self.max_avg_return - self.min_avg_return + 1e-8)
                y = np.diff(y)
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.scatter(X, y)
                # plt.savefig("test.png")
                # plt.close()
                # kernel = gp.kernels.Matern()
                # regr = gp.GaussianProcessRegressor(
                #     kernel=kernel, alpha=1e-4, n_restarts_optimizer=10, normalize_y=True
                # )
                # regr.fit(X, y)
                # sampling_mults = rnd[regr.sample_y(rnd).argmax(axis=0)][0]
                regr = linear_model.LinearRegression().fit(X, y)
                # now, sample a lot of pts, get regr values, pick the best
                rnd = np.random.uniform(
                    SAMPLING_LOWER_BOUND, 1, (1000, len(self.tasks))
                )
                sampling_mults = rnd[regr.predict(rnd).argmax(axis=0)]

        elif "bandit" in self.args.sample_budgeting:
            sampling_mults = self.bandit.pull()
            print(sampling_mults)
        else:
            raise NotImplementedError(
                f"Unknown sample_budgeting type: {self.args.sample_budgeting}"
            )
        return {
            tname: int(self.args.timesteps_per_batch * sm)
            for tname, sm in zip(self.tasks, sampling_mults)
        }

    def update_step(self):
        # runs a bunch of async processes that collect rollouts
        rollout_start = time.time()
        # master agent modifies the dict, we will need it later for the opt step to count stats
        print(self.timesteps_to_sample_next)
        paths, running_means, task_times = self.rollout_agent.rollout(
            self.timesteps_to_sample_next.copy()
        )

        if self.args.log_rollouts:
            to_log = {
                "action_map": self.action_dict[self.task_name],
                "action_dists_mu": [path["action_dists_mu"] for path in paths[self.task_name]],
                "rewards": means_diff_len([path["rewards"] for path in paths[self.task_name]]),
            }

            filename = f"{self.args.label}.pickle"
            dpath = (
                init_path.get_base_dir()
                if self.args.output_dir is None
                else self.args.output_dir
            )
            dpath = os.path.abspath(dpath)
            with open(os.path.join(dpath, "../analysis/paper_graph_gen/rollout/data", filename), "wb") as f:
                pickle.dump(to_log, f)

        rollout_time = (time.time() - rollout_start) / 60.0
        learn_start = time.time()
        self.learner_tasks.put((paths, running_means, self.timesteps_to_sample_next))
        self.learner_tasks.join()
        results = self.learner_results.get()

        self.timesteps_to_sample_next = results["timesteps_to_sample_next"]

        learn_time = (time.time() - learn_start) / 60.0

        self.rollout_agent.set_policy_weights(results["policy_weights"])
        logger.info("------------- Iteration %d --------------" % results["iteration"])
        logger.info("total time: %.2f mins" % ((time.time() - self.start_time) / 60.0))
        logger.info("optimization agent spent : %.2f mins" % (learn_time))
        logger.info("rollout agent spent : %.2f mins" % (rollout_time))
        for task, tm in task_times.items():
            logger.info(f"rollout {task} time : {tm / 60.0:.2f} mins")

        return results

    def end(self):
        self.rollout_agent.end()
        self.learner_tasks.put(parallel_util.END_SIGNAL)

    def build_policy_network(self):

        self.policy_network = {
            tname: build_policy_network(
                self.args,
                self.session,
                self.name_scope + "_policy_" + str(tid),
                self.observation_size[tname],
                tname,
                self.action_size[tname],
                ob_placeholder=tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.observation_size[tname]],
                    name=f"ob_input_{tname}",
                ),
                node_info=self.node_info[tname] if self.node_info else None,
            )
            for tid, tname in enumerate(self.tasks)
        }

        self.build_network_output()
        for tname, pnet in self.policy_network.items():
            logger.info(
                f"Policy for {tname} has {count_params(pnet._trainable_var_list)} params"
            )
            for el in pnet._trainable_var_list:
                print(el)
            if self.args.freeze_vars:
                logger.info("Freeze vars:")
                for el in pnet._freeze_var_list:
                    print(el)

        if self.args.policy_network_type == NetworkTypes.nervenet.name:
            self.raw_obs_placeholder = None
        elif self.args.policy_network_type == NetworkTypes.nervenetpp.name:
            self.step_policy_network = {
                tname: build_nervenetpp_policy(
                    self.args,
                    self.session,
                    self.name_scope,
                    self.observation_size[tname],
                    tname,
                    self.action_size[tname],
                    is_step=True,
                    node_info=self.node_info[tname],
                )
                for tid, tname in enumerate(self.tasks)
            }
            (step_policy_var_list, _,) = self.step_policy_network[
                self.task_name
            ].get_var_list()
            self.set_step_policy = utils.SetWeightsForPrefix(
                self.session, step_policy_var_list
            )

        self.fetch_policy_info()

    def build_network_output(self):
        if isinstance(
            self.policy_network[self.task_name],
            network.gated_graph_network.GatedGraphNetwork,
        ):
            self.action_dist_mu = {
                tname: build_gnn_output(
                    self.policy_network[tname], self.node_info[tname]
                )
                for tname in self.tasks
            }

            self.action_dist_logstd = {}
            self.action_dist_logstd_param = {}

            for tname in self.tasks:
                with tf.compat.v1.variable_scope(
                    self.policy_network[tname]._name_scope + "_1/"
                ):
                    with tf.compat.v1.variable_scope(
                        self.policy_network[tname]._name_scope
                    ):
                        # this is not a joke
                        # there was a bug(?) in the original code which created a context inside of context
                        # and if we want to load the old model, we need to imitate that
                        # or I should probably either rewrite the model loader or resave it with proper names
                        if self.args.logstd_out_type == "separate":
                            # size: [1, num_action]
                            self.action_dist_logstd[tname] = tf.Variable(
                                (
                                    0.0 * self._npr.randn(1, self.action_size[tname])
                                ).astype(np.float32),
                                name=f"policy_logstd_{tname}",
                                trainable=self.policy_network[tname]._trainable,
                            )
                            # step 1_8: build the log std for the actions
                            self.action_dist_logstd_param[tname] = tf.tile(
                                self.action_dist_logstd[tname],
                                tf.stack((tf.shape(self.action_dist_mu[tname])[0], 1)),
                            )
                        elif self.args.logstd_out_type == "scalar":
                            # size: [1, num_action]
                            self.action_dist_logstd[tname] = tf.Variable(
                                0.0 * self._npr.randn(1, 1).astype(np.float32),
                                name=f"policy_logstd_{tname}",
                                trainable=self.policy_network[tname]._trainable,
                            )

                            self.action_dist_logstd_param[tname] = tf.tile(
                                self.action_dist_logstd[tname]
                                * tf.compat.v1.ones(
                                    self.action_size[tname], dtype=tf.float32
                                ),
                                tf.stack((tf.shape(self.action_dist_mu[tname])[0], 1,)),
                            )

                self.policy_network[tname]._set_var_list()
            self.action_dist_logstd = self.action_dist_logstd_param

        else:
            tname = self.task_name
            self.action_dist_mu = {tname: build_mlp_output(self.policy_network[tname])}
            with tf.compat.v1.variable_scope(
                self.policy_network[tname]._name_scope + "/"
            ):
                # / here is to reuse entering the same scope from the policy and not append _1
                # so that we could load an old model
                # https://stackoverflow.com/a/38908142/1768248
                # size: [1, num_action]
                self.action_dist_logstd = {
                    tname: tf.Variable(
                        (0 * self._npr.randn(1, self.action_size[tname])).astype(
                            np.float32
                        ),
                        name="policy_logstd",
                        trainable=self.policy_network[tname]._trainable,
                    )
                }
                # size: [batch, num_action]
                self.action_dist_logstd_param = {
                    tname: tf.tile(
                        self.action_dist_logstd[tname],
                        tf.stack((tf.shape(self.action_dist_mu[tname])[0], 1)),
                    )
                }
            self.policy_network[tname]._set_var_list()

    def fetch_policy_info(self):
        self.receive_idx_placeholder = {}
        self.send_idx_placeholder = {}
        self.node_type_idx_placeholder = {}
        self.inverse_node_type_idx_placeholder = {}
        self.output_type_idx_placeholder = {}
        self.inverse_output_type_idx_placeholder = {}
        self.batch_size_int_placeholder = {}
        self.graph_obs_placeholder = {}
        self.graph_parameters_placeholder = {}
        self.gnn_placeholder_list_dict = {}
        self.trainable_policy_var_list_dict = {}
        self.all_policy_var_list_dict = {}

        if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
            self.step_receive_idx_placeholder = {}
            self.step_send_idx_placeholder = {}
            self.step_node_type_idx_placeholder = {}
            self.step_inverse_node_type_idx_placeholder = {}
            self.step_output_type_idx_placeholder = {}
            self.step_inverse_output_type_idx_placeholder = {}
            self.step_batch_size_int_placeholder = {}
            self.step_graph_obs_placeholder = {}
            self.step_graph_parameters_placeholder = {}
        self.obs_placeholder = {}
        # input placeholders to the policy networks
        for tname in self.tasks:
            if (
                self.args.policy_network_type == NetworkTypes.nervenet.name
                or self.args.policy_network_type == NetworkTypes.nervenetpp.name
            ):
                # index placeholders
                (
                    self.receive_idx_placeholder[tname],
                    self.send_idx_placeholder[tname],
                    self.node_type_idx_placeholder[tname],
                    self.inverse_node_type_idx_placeholder[tname],
                    self.output_type_idx_placeholder[tname],
                    self.inverse_output_type_idx_placeholder[tname],
                    self.batch_size_int_placeholder[tname],
                ) = self.policy_network[tname].get_gnn_idx_placeholder()

                # the graph_obs placeholders and graph_parameters_placeholders
                self.graph_obs_placeholder[tname] = self.policy_network[
                    tname
                ].get_input_obs_placeholder()
                self.graph_parameters_placeholder[tname] = self.policy_network[
                    tname
                ].get_input_parameters_placeholder()

                self.gnn_placeholder_list_dict[tname] = [
                    self.receive_idx_placeholder[tname],
                    self.send_idx_placeholder[tname],
                    self.node_type_idx_placeholder[tname],
                    self.inverse_node_type_idx_placeholder[tname],
                    self.output_type_idx_placeholder[tname],
                    self.inverse_output_type_idx_placeholder[tname],
                    self.batch_size_int_placeholder[tname],
                    self.graph_obs_placeholder[tname],
                    self.graph_parameters_placeholder[tname],
                ]

                if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                    (
                        self.step_receive_idx_placeholder[tname],
                        self.step_send_idx_placeholder[tname],
                        self.step_node_type_idx_placeholder[tname],
                        self.step_inverse_node_type_idx_placeholder[tname],
                        self.step_output_type_idx_placeholder[tname],
                        self.step_inverse_output_type_idx_placeholder[tname],
                        self.step_batch_size_int_placeholder[tname],
                    ) = self.step_policy_network[tname].get_gnn_idx_placeholder()

                    # the graph_obs placeholders and graph_parameters_placeholders
                    self.step_graph_obs_placeholder[tname] = self.step_policy_network[
                        tname
                    ].get_input_obs_placeholder()
                    self.step_graph_parameters_placeholder[
                        tname
                    ] = self.step_policy_network[
                        tname
                    ].get_input_parameters_placeholder()
            else:
                self.obs_placeholder[tname] = self.policy_network[
                    tname
                ].get_input_placeholder()
            (
                self.trainable_policy_var_list_dict[tname],
                self.all_policy_var_list_dict[tname],
            ) = self.policy_network[tname].get_var_list()

        self.iteration = self.policy_network[self.task_name].get_iteration_var()
        self.iteration_add_op = self.iteration.assign_add(1)
        self.raw_obs_placeholder = None

        # with tf.compat.v1.get_default_graph().as_default():
        # if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
        #     self.step_action_dist_mu = self.step_policy_network.get_action_dist_mu()
        #
        #     # log std parameters of actions (all the same)
        #     self.step_action_dist_logstd_param = (
        #         self.step_policy_network.get_action_dist_logstd_param()
        #     )
        #     self.step_action_dist_logstd = self.step_action_dist_logstd_param
        # "policy_var_list": to be passed to the rollout agents
        # "all_policy_var_list": to be saved into the checkpoint

    def prepared_policy_network_feeddict(
        self, obs_n, node_info, rollout_data=None, step_model=False, task=None
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
                    self.receive_idx[task],
                    self.send_idx[task],
                    self.node_type_idx[task],
                    self.inverse_node_type_idx[task],
                    self.output_type_idx[task],
                    self.inverse_output_type_idx[task],
                    self.last_batch_size[task],
                ) = graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    obs_n,
                    self.receive_idx[task],
                    self.send_idx[task],
                    self.node_type_idx[task],
                    self.inverse_node_type_idx[task],
                    self.output_type_idx[task],
                    self.inverse_output_type_idx[task],
                    self.last_batch_size[task],
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
                    self.receive_idx[task],
                    self.send_idx[task],
                    self.node_type_idx[task],
                    self.inverse_node_type_idx[task],
                    self.output_type_idx[task],
                    self.inverse_output_type_idx[task],
                    self.last_batch_size[task],
                ) = graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    np.empty([int(total_size / self.args.gnn_num_prop_steps)]),
                    self.receive_idx[task],
                    self.send_idx[task],
                    self.node_type_idx[task],
                    self.inverse_node_type_idx[task],
                    self.output_type_idx[task],
                    self.inverse_output_type_idx[task],
                    self.last_batch_size[task],
                    request_data=["idx"],
                )

            if step_model:
                feed_dict = {
                    self.step_batch_size_int_placeholder[task]: int(
                        self.last_batch_size[task]
                    ),
                    self.step_receive_idx_placeholder[task]: self.receive_idx[task],
                    self.step_inverse_node_type_idx_placeholder[
                        task
                    ]: self.inverse_node_type_idx,
                    self.step_inverse_output_type_idx_placeholder[
                        task
                    ]: self.inverse_output_type_idx[task],
                }

                # append the input obs and parameters
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[
                        self.step_graph_obs_placeholder[task][i_node_type]
                    ] = graph_obs[i_node_type]
                    feed_dict[
                        self.step_graph_parameters_placeholder[task][i_node_type]
                    ] = graph_parameters[i_node_type]

                # append the send idx
                for i_edge in node_info["edge_type_list"]:
                    feed_dict[
                        self.step_send_idx_placeholder[task][i_edge]
                    ] = self.send_idx[task][i_edge]

                # append the node type idx
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[
                        self.step_node_type_idx_placeholder[task][i_node_type]
                    ] = self.node_type_idx[task][i_node_type]

                # append the output type idx
                for i_output_type in node_info["output_type_dict"]:
                    feed_dict[
                        self.step_output_type_idx_placeholder[task][i_output_type]
                    ] = self.output_type_idx[task][i_output_type]

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
                    self.receive_idx[task],
                    self.send_idx[task],
                    self.node_type_idx[task],
                    self.inverse_node_type_idx[task],
                    self.output_type_idx[task],
                    self.inverse_output_type_idx[task],
                    self.last_batch_size[task],
                ) = graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    obs_n,
                    self.receive_idx[task],
                    self.send_idx[task],
                    self.node_type_idx[task],
                    self.inverse_node_type_idx[task],
                    self.output_type_idx[task],
                    self.inverse_output_type_idx[task],
                    self.last_batch_size[task],
                )

                feed_dict = {
                    self.batch_size_int_placeholder[task]: int(
                        self.last_batch_size[task]
                    ),
                    self.receive_idx_placeholder[task]: self.receive_idx[task],
                    self.inverse_node_type_idx_placeholder[
                        task
                    ]: self.inverse_node_type_idx[task],
                    self.inverse_output_type_idx_placeholder[
                        task
                    ]: self.inverse_output_type_idx[task],
                }

                # append the input obs and parameters
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[
                        self.graph_obs_placeholder[task][i_node_type]
                    ] = graph_obs[i_node_type]
                    feed_dict[
                        self.graph_parameters_placeholder[task][i_node_type]
                    ] = graph_parameters[i_node_type]

                # append the send idx
                for i_edge in node_info["edge_type_list"]:
                    feed_dict[self.send_idx_placeholder[task][i_edge]] = self.send_idx[
                        task
                    ][i_edge]

                # append the node type idx
                for i_node_type in node_info["node_type_dict"]:
                    feed_dict[
                        self.node_type_idx_placeholder[task][i_node_type]
                    ] = self.node_type_idx[task][i_node_type]

                # append the output type idx
                for i_output_type in node_info["output_type_dict"]:
                    feed_dict[
                        self.output_type_idx_placeholder[task][i_output_type]
                    ] = self.output_type_idx[task][i_output_type]

                # if the raw_obs is needed for the baseline

                if self.raw_obs_placeholder is not None:
                    feed_dict[self.raw_obs_placeholder[task]] = obs_n
        else:
            # it is the most easy case, nice and easy
            feed_dict = {self.obs_placeholder[task]: obs_n}

        self.nervenetplus_batch_pos = nervenetplus_batch_pos
        return feed_dict, nervenetplus_batch_pos

    def gnn_parameter_initialization(self):
        """
            @brief:
                the parameters for the gnn, see the gated_graph_network_policy
                file for details what these variables mean.
        """
        self.receive_idx = {tname: None for tname in self.tasks}
        self.send_idx = {tname: None for tname in self.tasks}
        self.node_type_idx = {tname: None for tname in self.tasks}
        self.inverse_node_type_idx = {tname: None for tname in self.tasks}
        self.output_type_idx = {tname: None for tname in self.tasks}
        self.inverse_output_type_idx = {tname: None for tname in self.tasks}
        self.last_batch_size = {tname: -1 for tname in self.tasks}

    def get_obs_description(self):
        joint_names = list(self.action_dict[self.tasks[0]].values())
        body_names = list(self.body_dict[self.tasks[0]].values())[1:]

        obs_description = [
            ("root", "root", "qpos", "position", "z"),
            ("root", "root", "qpos", "orientation", "x"),
            ("root", "root", "qpos", "orientation", "y"),
            ("root", "root", "qpos", "orientation", "z"),
            ("root", "root", "qpos", "orientation", "a"),
        ]

        for jn in joint_names:
            obs_description.append(("joint", jn, "qpos", "position", "x"))

        obs_description += [
            ("root", "root", "qvel", "velocity", "x"),
            ("root", "root", "qvel", "velocity", "y"),
            ("root", "root", "qvel", "velocity", "z"),
            ("root", "root", "qvel", "angular velocity", "x"),
            ("root", "root", "qvel", "angular velocity", "y"),
            ("root", "root", "qvel", "angular velocity", "z"),
        ]

        for jn in joint_names:
            obs_description.append(("joint", jn, "qvel", "velocity", "x"))

        obs_description += [
            ("root", "root", "cfrc", "force", "x"),
            ("root", "root", "cfrc", "force", "y"),
            ("root", "root", "cfrc", "force", "z"),
            ("root", "root", "cfrc", "torque", "x"),
            ("root", "root", "cfrc", "torque", "y"),
            ("root", "root", "cfrc", "torque", "z"),
        ]

        for bn in body_names:
            obs_description.append(("body", bn, "cfrc", "force", "x"))
            obs_description.append(("body", bn, "cfrc", "force", "y"))
            obs_description.append(("body", bn, "cfrc", "force", "z"))
            obs_description.append(("body", bn, "cfrc", "torque", "x"))
            obs_description.append(("body", bn, "cfrc", "torque", "y"))
            obs_description.append(("body", bn, "cfrc", "torque", "z"))

        return obs_description
