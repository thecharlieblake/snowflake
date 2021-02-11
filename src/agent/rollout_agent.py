import numpy as np
import tensorflow as tf
import gym
import time
import network
from network import NetworkTypes
import os
from util import utils
from util import ob_normalizer
from util import logger
from util import parallel_util
from util import init_path
from agent.agent import base_agent
from network.net_maker import build_session
from network.abstractions import build_gnn_output, build_mlp_output
from network.net_maker import build_policy_network, build_nervenetpp_policy
import pickle
import datetime

N_PERTURBATIONS = 10
EPS_LIMIT = 0.2


class rollout_agent(base_agent):
    def __init__(
        self,
        args,
        task_name,
        task_q,
        result_q,
        actor_id,
        obs_description,
        action_dict,
        monitor,
        name_scope="policy-actor",
        init_filter_parameters=None,
        node_info=None,
    ):
        dummy_env = gym.make(task_name.split("_lk_")[-1])
        self.observation_size = dummy_env.observation_space.shape[0]
        self.action_size = dummy_env.action_space.shape[0]
        self.node_info = node_info
        self.obs_description = obs_description
        self.action_dict = action_dict
        # the base agent
        super(rollout_agent, self).__init__(
            args=args,
            task_name=task_name,
            task_q=task_q,
            result_q=result_q,
            thread_name='{}_{}'.format(name_scope, actor_id),
            name_scope=name_scope
        )
        self.allow_monitor = monitor
        self.actor_id = actor_id
        self._npr = np.random.RandomState(args.seed + actor_id)

        if init_filter_parameters is not None:
            self.ob_normalizer = ob_normalizer.normalizer(
                mean=init_filter_parameters["mean"],
                variance=init_filter_parameters["variance"],
                num_steps=init_filter_parameters["step"],
            )
        else:
            self.ob_normalizer = ob_normalizer.normalizer()

        logger.info("The sampler {} is online".format(self.actor_id))

    def run(self):
        self.build_model()

        while True:
            next_task = self.task_q.get(block=True)

            # Essentially 2 points at which signals are sent to worker threads.
            # 1) Setup: Optimisation agent init calls all 3 (in following order) NOTE: a & b only if burn=True!!!
            #   a) START: Optimisation agent sets up rollout master, which calls init actors, which sends signal
            #             (this is done for running means burn-in)
            #   b) SYNCH: Optimisation agent sets up rollout master, which calls init actors, which sends signal
            #   c) WEIGHTS: Optimisation agent calls set policy weights in rollout master, which then sends signal
            # 2) Update step: main calls optimisation agent update step
            #   a) SYNCH: update step calls master agent rollout, which sends signal
            #   b) WEIGHTS: at the end of master agent rollout, calls set policy weights which sends signal

            # Collect rollouts
            # rollout master 125 (ask for rollouts) ultimately rollout 68 (from optimisation agent update step 1698)
            # plus master agent init actors ultimately Optimisation agent 338 (init)
            # [note Optimisation agent 341 (init) is for *optimisation agent* not rollout agents]
            if next_task[0] == parallel_util.START_SIGNAL:
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(path)

            # Set normalizer
            # Rollout master agent 262 (init actors) ultimately Optimisation agent 338 (init),
            # 76 (rollout) ultimately Optimisation agent 1698 (update step)
            elif next_task[0] == parallel_util.AGENT_SYNCHRONIZE_FILTER:
                self.ob_normalizer.set_parameters(
                    next_task[1][self.task_name]["mean"],
                    next_task[1][self.task_name]["variance"],
                    next_task[1][self.task_name]["step"],
                )
                time.sleep(0.001)  # yield the process
                self.task_q.task_done()

            # Set parameters of the actor policy
            # Rollout master agent 86 (set policy weights), ultimately called by Optimisation agent 343 (init)
            # and by Optimisation agent 1722 (update step)
            elif next_task[0] == parallel_util.AGENT_SET_POLICY_WEIGHTS:
                self.set_policy(next_task[1])
                time.sleep(0.001)  # yield the process
                self.task_q.task_done()

            # Kill all the thread
            elif (
                next_task[0] == parallel_util.END_ROLLOUT_SIGNAL
                or next_task[0] == parallel_util.END_SIGNAL
            ):
                logger.info("kill message for sampler {}".format(self.actor_id))
                self.task_q.task_done()
                break
            else:
                logger.error("Invalid task type {} for agents".format(next_task[0]))
        return

    def wrap_env_monitor(self):
        if self.allow_monitor:

            def video_callback(episode):
                return episode % self.args.video_freq < 6

            if self.args.output_dir is None:
                base_path = init_path.get_base_dir()
            else:
                base_path = self.args.output_dir

            if self.args.video_path:
                path = self.args.video_path
            else:
                path = os.path.join(
                    base_path, "video", self.task_name + "_" + self.args.time_id
                )
            path = os.path.abspath(path)
            if not os.path.exists(path):
                os.makedirs(path)
            self.env = gym.wrappers.Monitor(
                self.env, path, video_callable=video_callback
            )

    def build_env(self):
        self.env = gym.make(self.task_name.split("_lk_")[-1])
        self.sub_task_list = self.task_name
        self.env.seed(self._npr.randint(0, 999999))
        self.wrap_env_monitor()  # wrap the environment with monitor if needed

    def build_model(self):
        self.build_env()
        self.session = build_session(self.args.use_gpu)
        self.build_policy_network()
        self.session.run(tf.compat.v1.global_variables_initializer())

        self.set_policy = utils.SetWeightsForPrefix(self.session, self.all_policy_var_list)
        self.get_policy = utils.GetWeightsForPrefix(self.session, self.all_policy_var_list)

    def do_perturbations(self, ob, mu):
        pert = np.zeros((N_PERTURBATIONS + 1, ob.shape[0], mu.shape[1]))

        # batch this
        for po in range(ob.shape[0]):
            pert[0][po] = mu.flatten()
            ceps = np.random.uniform(-EPS_LIMIT, EPS_LIMIT, N_PERTURBATIONS)
            newob = np.zeros((N_PERTURBATIONS, ob.shape[0]))
            newob[:, po] = ceps
            _, pmu, _ = self.act(ob + newob, batched=True)
            pert[1:, po] = pmu
        return pert

    def rollout(self):
        start_time = time.time()

        if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
            self.current_state = {
                node_type: np.zeros(
                    [
                        len(self.node_info["node_type_dict"][node_type]),
                        self.args.gnn_node_hidden_dim,
                    ]
                )
                for node_type in self.node_info["node_type_dict"]
            }
            hidden_state = {
                node_type: [] for node_type in self.node_info["node_type_dict"]
            }

        (obs, actions, rewards, action_dists_mu, action_dists_logstd, raw_obs) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        reward_breakdown = {f"reward_{r}": [] for r in ["forward", "ctrl", "contact", "survive"]}
        path = dict()

        # start the env (reset the environment)
        raw_ob = self.env.reset()
        ob = self.ob_normalizer.filter(raw_ob)

        perturbation_data = []
        # run the game
        while True:
            # generate the policy
            if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                for node_type in self.node_info["node_type_dict"]:
                    hidden_state[node_type].append(self.current_state[node_type])

            action, action_dist_mu, action_dist_logstd = self.act(ob)
            if self.args.test and self.args.test and self.args.perturbation_analysis:
                perturbation_data.append(self.do_perturbations(ob, action_dist_mu))

            # record the stats
            obs.append(ob)
            raw_obs.append(raw_ob)
            actions.append(action)
            action_dists_mu.append(action_dist_mu)
            action_dists_logstd.append(action_dist_logstd)

            # take the action
            res = self.env.step(action)
            raw_ob = res[0]
            ob = self.ob_normalizer.filter(raw_ob)

            rewards.append((res[1]))
            #for rt in reward_breakdown:
            #    reward_breakdown[rt].append(res[3][rt])

            if res[2]:  # terminated
                path = {
                    "task": self.task_name,
                    "actor_id": self.actor_id,
                    "obs": np.array(obs),
                    "raw_obs": np.array(raw_obs),
                    "action_dists_mu": np.concatenate(action_dists_mu),
                    "action_dists_logstd": np.concatenate(action_dists_logstd),
                    "rewards": np.array(rewards),
                    "actions": np.array(actions),
                    "time": time.time() - start_time,
                    "reward_breakdown": reward_breakdown,
                }
                if self.args.test:
                    path["perturbations"] = np.array(perturbation_data)
                if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                    path["hidden_state"] = hidden_state
                    for node_type in self.node_info["node_type_dict"]:
                        path["hidden_state"][node_type] = np.concatenate(
                            hidden_state[node_type]
                        )
                break

        return path

    def act(self, obs, batched=False):
        if not batched:
            obs = np.expand_dims(obs, 0)
        feed_dict, _ = self.prepared_policy_network_feeddict(obs, self.node_info)

        if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
            self._input_hidden_state = (
                self.policy_network.get_input_hidden_state_placeholder()
            )
            for node_type in self.node_info["node_type_dict"]:
                feed_dict[self._input_hidden_state[node_type]] = self.current_state[
                    node_type
                ]

            results = self.session.run(
                [self.action_dist_mu, self.action_dist_logstd]
                + self.policy_network.get_output_hidden_state_list(),
                feed_dict=feed_dict,
            )
            action_dist_mu = results[0]
            action_dist_logstd = results[1]
            self.current_state = {
                node_info: results[2 + iid]
                for iid, node_info in enumerate(self.node_info["node_type_dict"])
            }

        else:
            action_dist_mu, action_dist_logstd, nma = self.session.run(
                [self.action_dist_mu, self.action_dist_logstd, self.policy_network.norm_moving_averages if self.args.batchnorm else {},], feed_dict=feed_dict
            )
        # samples the guassian distribution
        act = action_dist_mu + np.exp(action_dist_logstd) * self._npr.randn(
            *action_dist_logstd.shape
        )
        act = act.ravel()
        return act, action_dist_mu, action_dist_logstd

    def build_policy_network(self):
        self.pi_net_input = tf.compat.v1.placeholder(
            tf.float32, [None, self.observation_size], name="ob_input"
        )
        self.policy_network = build_policy_network(
            self.args,
            self.session,
            self.name_scope + "_policy",
            self.observation_size,
            self.task_name,
            self.action_size,
            ob_placeholder=self.pi_net_input,
            node_info=self.node_info,
            inference_mode=True,
        )

        if self.args.policy_network_type == NetworkTypes.nervenet.name:
            self.raw_obs_placeholder = None
        elif self.args.policy_network_type == NetworkTypes.nervenetpp.name:
            self.step_policy_network = build_nervenetpp_policy(
                self.args,
                self.session,
                self.name_scope,
                self.observation_size,
                self.task_name,
                self.action_size,
                is_step=True,
                node_info=self.node_info,
            )
            (step_policy_var_list, _,) = self.step_policy_network.get_var_list()
            self.set_step_policy = utils.SetWeightsForPrefix(
                self.session, step_policy_var_list
            )
        self.build_network_output()

        self.fetch_policy_info()

    def build_network_output(self):
        if isinstance(
            self.policy_network, network.gated_graph_network.GatedGraphNetwork
        ):

            self.action_dist_mu = build_gnn_output(self.policy_network, self.node_info)

            with tf.compat.v1.variable_scope(self.policy_network._name_scope + "_1/"):
                with tf.compat.v1.variable_scope(self.policy_network._name_scope):
                    # this is not a joke
                    # there was a bug(?) in the original code which created a context inside of context
                    # and if we want to load the old model, we need to imitate that
                    # or I should probably either rewrite the model loader or resave it with proper names
                    if self.args.logstd_out_type == "separate":
                        # size: [1, num_action]
                        self.action_dist_logstd = tf.Variable(
                            (0.0 * self._npr.randn(1, self.action_size)).astype(
                                np.float32
                            ),
                            name="policy_logstd",
                            trainable=self.policy_network._trainable,
                        )
                        # step 1_8: build the log std for the actions
                        self.action_dist_logstd_param = tf.tile(
                            self.action_dist_logstd,
                            tf.stack((tf.shape(self.action_dist_mu)[0], 1)),
                        )
                    elif self.args.logstd_out_type == "scalar":
                        # size: [1, num_action]
                        self._action_dist_logstd = tf.Variable(
                            0.0 * self._npr.randn(1, 1).astype(np.float32),
                            name="policy_logstd",
                            trainable=self.policy_network._trainable,
                        )
                        self.action_dist_logstd_param = tf.tile(
                            self._action_dist_logstd
                            * tf.compat.v1.ones(self.action_size, dtype=tf.float32),
                            tf.stack((tf.shape(self.action_dist_mu)[0], 1)),
                        )
                self.action_dist_logstd = self.action_dist_logstd_param
                if self.args.policy_network_type == NetworkTypes.nervenetpp.name:
                    self.step_action_dist_mu = build_gnn_output(
                        self.step_policy_network, self.node_info
                    )
                    if self.args.logstd_out_type == "separate":
                        # size: [1, num_action]
                        self.step_action_dist_logstd = tf.Variable(
                            (0.0 * self._npr.randn(1, self.action_size)).astype(
                                np.float32
                            ),
                            name="policy_logstd",
                            trainable=self.policy_network._trainable,
                        )
                        # step 1_8: build the log std for the actions
                        self.step_action_dist_logstd_param = tf.tile(
                            self.action_dist_logstd,
                            tf.stack((tf.shape(self.step_action_dist_mu)[0], 1)),
                        )
                    elif self.args.logstd_out_type == "scalar":
                        # size: [1, num_action]
                        self.step_action_dist_logstd = tf.Variable(
                            0.0 * self._npr.randn(1, 1).astype(np.float32),
                            name="policy_logstd",
                            trainable=self.policy_network._trainable,
                        )
                        self.step_action_dist_logstd_param = tf.tile(
                            self.step_action_dist_logstd
                            * tf.compat.v1.ones(self.action_size, dtype=tf.float32),
                            tf.stack((tf.shape(self.action_dist_mu)[0], 1)),
                        )

                    self.step_action_dist_logstd = self.step_action_dist_logstd_param

        else:
            self.action_dist_mu = build_mlp_output(self.policy_network)
            with tf.compat.v1.variable_scope(self.policy_network._name_scope + "/"):
                # / here is to reuse entering the same scope from the policy and not append _1
                # so that we could load an old model
                # https://stackoverflow.com/a/38908142/1768248
                # size: [1, num_action]
                self.action_dist_logstd = tf.Variable(
                    (0 * self._npr.randn(1, self.action_size)).astype(np.float32),
                    name="policy_logstd",
                    trainable=self.policy_network._trainable,
                )

                # size: [batch, num_action]
                self.action_dist_logstd_param = tf.tile(
                    self.action_dist_logstd,
                    tf.stack((tf.shape(self.action_dist_mu)[0], 1)),
                )
        self.policy_network._set_var_list()
