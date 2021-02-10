import multiprocessing
import os
from collections import defaultdict

import gym
import numpy as np
import time

from agent.rollout_agent import rollout_agent
from graph_util import structure_mapper
from util import init_path
from util import logger
from util import model_saver
from util import parallel_util


class parallel_rollout_master_agent:
    def __init__(self, args, node_info, task_names, obs_description, action_dict):
        """
            @brief:
                the master agent has several actors (or samplers) to do the
                sampling for it.
        """
        self.args = args
        self.task_names = task_names
        self.obs_description = obs_description
        self.action_dict = action_dict
        self.observation_size = {}
        self.actor_envs_by_id = args.thread_agents

        logger.info(f"Actor envs by id: {self.actor_envs_by_id}")

        # init the running means
        self.running_mean_info = {
            tname: {
                "mean": 0.0,
                "variance": 1,
                "step": 0.01,
                "square_sum": 0.01,
                "sum": 0.0,
            }
            for tname in set(self.actor_envs_by_id)
        }
        self.node_info = node_info
        self.load_running_means()

        # init the multiprocess actors
        self.tasks = {
            tname: multiprocessing.JoinableQueue() for tname in self.task_names
        }
        self.results = {tname: multiprocessing.Queue() for tname in self.task_names}

        # we will start by running 20,000 / 1000 = 20 episodes for the first
        # iteration
        self.average_timesteps_in_episode = {
            tname: gym.spec(tname.split("_lk_")[-1]).max_episode_steps for tname in self.task_names
        }
        self.init_actors()

    def rollout(self, timesteps_needed):
        # step 1: ask the samplers to generate rollouts
        rollout_data, task_times = self.ask_for_rollouts(timesteps_needed)

        # step 2: update the running means
        running_mean_start_time = time.time()
        self.update_running_means(rollout_data)
        task_times['update running means'] = time.time() - running_mean_start_time

        # step 3: synchronize the filter statistics for all samplers
        for tname in self.actor_envs_by_id:
            self.tasks[tname].put(
                (parallel_util.AGENT_SYNCHRONIZE_FILTER, self.running_mean_info)
            )
        self.join_task_queues()

        # tell the trpo agent about the information of normalizer
        return rollout_data, self.running_mean_info, task_times

    def set_policy_weights(self, parameters):
        for t in self.actor_envs_by_id:
            self.tasks[t].put(
                (parallel_util.AGENT_SET_POLICY_WEIGHTS, parameters[t])
            )
        self.join_task_queues()

    def join_task_queues(self):
        for task_name, task_queue in self.tasks.items():
            task_queue.join()

    def end(self):
        for tname in self.actor_envs_by_id:
            self.tasks[tname].put((parallel_util.END_ROLLOUT_SIGNAL, None))

    def ask_for_rollouts(self, timesteps_needed):
        """
            @brief:
                Run the experiments until a total of @timesteps_per_batch
                timesteps are collected.
        """
        num_timesteps_received = {tname: 0 for tname in self.task_names}
        task_times = {tname: 0 for tname in self.task_names}
        rollout_data = defaultdict(list)

        while True:

            if self.args.test:
                num_rollouts = {self.task_names[0]: self.args.test}
            else:
                # calculate the expected number of episodes needed
                num_rollouts = {
                    tname: int(
                        np.ceil(
                            timesteps_needed[tname]
                            / self.average_timesteps_in_episode[tname]
                        )
                    )
                    for tname in self.task_names
                }
            # request episodes
            for tname, task_rolls in num_rollouts.items():
                for _ in range(task_rolls):
                    self.tasks[tname].put((parallel_util.START_SIGNAL, None))
            self.join_task_queues()
            # collect episodes
            for tname, task_rolls in num_rollouts.items():
                for _ in range(task_rolls):
                    traj_episode = self.results[tname].get()
                    num_timesteps_received[traj_episode["task"]] += len(
                        traj_episode["rewards"]
                    )
                    task_times[tname] += traj_episode['time']
                    rollout_data[tname].append(traj_episode)

                # update average timesteps per episode
                self.average_timesteps_in_episode[tname] = float(
                    num_timesteps_received[tname]
                ) / len(rollout_data[tname])
                timesteps_needed[tname] -= num_timesteps_received[tname]

            if np.all(np.array(list(timesteps_needed.values())) <= 0) or self.args.test:
                break

        logger.info("Rollouts generating ...")
        for tname, tval in num_timesteps_received.items():
            logger.info(
                "{}: {} time steps from {} episodes collected".format(
                    tname, tval, len(rollout_data[tname])
                )
            )

        return rollout_data, task_times

    def update_running_means(self, paths):
        # collect the info

        new_stats = {
            tname: {"new_sum": 0.0, "new_step_sum": 0.0, "new_sq_sum": 0.0}
            for tname in set(self.actor_envs_by_id)
        }
        for tname, tpaths in paths.items():
            if tname not in self.args.test_transfer_tasks:
                for path in tpaths:
                    raw_obs = path["raw_obs"]
                    tname = path["task"]
                    new_stats[tname]["new_sum"] += raw_obs.sum(axis=0)
                    new_stats[tname]["new_step_sum"] += raw_obs.shape[0]
                    new_stats[tname]["new_sq_sum"] += (np.square(raw_obs)).sum(axis=0)

        # update the parameters
        for tname, tvalue in new_stats.items():
            if tname not in self.args.test_transfer_tasks:
                self.running_mean_info[tname]["sum"] += tvalue["new_sum"]
                self.running_mean_info[tname]["square_sum"] += tvalue["new_sq_sum"]
                self.running_mean_info[tname]["step"] += tvalue["new_step_sum"]
                self.running_mean_info[tname]["mean"] = (
                    self.running_mean_info[tname]["sum"]
                    / self.running_mean_info[tname]["step"]
                )

                self.running_mean_info[tname]["variance"] = np.maximum(
                    self.running_mean_info[tname]["square_sum"]
                    / self.running_mean_info[tname]["step"]
                    - np.square(self.running_mean_info[tname]["mean"]),
                    1e-2,
                )

        for tname in self.args.test_transfer_tasks:
            self.running_mean_info[tname] = structure_mapper.map_transfer_env_running_mean(
                    self.args.task[0],
                    tname,
                    self.running_mean_info[self.args.task[0]],
                    self.observation_size[tname],
                    self.args.gnn_node_option,
                    self.args.root_connection_option,
                    self.args.gnn_output_option,
                    self.args.gnn_embedding_option,
                )

    def load_running_means(self):
        # load the observation running mean
        if self.args.ckpt_name is not None:
            base_path = os.path.join(init_path.get_base_dir(), "checkpoint")
            logger.debug("[LOAD_CKPT] loading observation normalizer info")

            loaded_running_means = model_saver.load_numpy_model(
                os.path.join(self.args.ckpt_name + "_normalizer.npy"),
                numpy_var_list=self.running_mean_info,
            )
        else:
            loaded_running_means = {}

        if not self.args.transfer_env == "Nothing2Nothing":
            if self.args.mlp_raw_transfer == 0:
                assert "shared" in self.args.gnn_embedding_option

            if self.args.task[0] not in loaded_running_means:
                ienv, oenv = [env + "-v1" for env in self.args.transfer_env.split("2")]
                dummy_env = gym.make(oenv)
                self.observation_size[oenv] = dummy_env.observation_space.shape[0]
                if oenv not in loaded_running_means:
                    loaded_running_means[
                        oenv
                    ] = structure_mapper.map_transfer_env_running_mean(
                        ienv,
                        oenv,
                        loaded_running_means[ienv],
                        self.observation_size[oenv],
                        self.args.gnn_node_option,
                        self.args.root_connection_option,
                        self.args.gnn_output_option,
                        self.args.gnn_embedding_option,
                    )

        for t in self.args.test_transfer_tasks:
            dummy_env = gym.make(t)
            self.observation_size[t] = dummy_env.observation_space.shape[0]
            if self.args.ckpt_name is not None:
                if t not in loaded_running_means:
                    loaded_running_means[
                        t
                    ] = structure_mapper.map_transfer_env_running_mean(
                        self.args.task[0],
                        t,
                        loaded_running_means[self.args.task[0]],
                        self.observation_size[t],
                        self.args.gnn_node_option,
                        self.args.root_connection_option,
                        self.args.gnn_output_option,
                        self.args.gnn_embedding_option,
                    )
        self.running_mean_info = {**self.running_mean_info, **loaded_running_means}


    def init_actors(self):
        """
            @brief: init the actors and start the multiprocessing
        """

        self.actors = []

        for i in range(len(self.actor_envs_by_id)):
            self.actors.append(
                rollout_agent(
                    self.args,
                    self.actor_envs_by_id[i],
                    self.tasks[self.actor_envs_by_id[i]],
                    self.results[self.actor_envs_by_id[i]],
                    i,
                    self.obs_description,
                    self.action_dict,
                    self.args.monitor if i == 0 else False,
                    init_filter_parameters=self.running_mean_info[
                        self.actor_envs_by_id[i]
                    ],
                    node_info=self.node_info[self.actor_envs_by_id[i]]
                    if self.node_info
                    else None,
                )
            )

        for i_actor in self.actors:
            i_actor.start()

        # burn-in running means before training
        if self.args.burn_in_running_means:
            self.update_running_means(
                self.ask_for_rollouts(  # Sends START_SIGNAL
                    {
                        tname: self.args.timesteps_per_batch * 2
                        for tname in self.args.task
                    }
                )
            )
            for tname in self.actor_envs_by_id:
                self.tasks[tname].put(
                    (parallel_util.AGENT_SYNCHRONIZE_FILTER, self.running_mean_info)
                )
            self.join_task_queues()
