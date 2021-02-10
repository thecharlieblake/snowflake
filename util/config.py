import argparse
from util import init_path
import random
import numpy as np
from network import NetworkTypes


def get_config():
    # get the parameters
    parser = argparse.ArgumentParser(
        description="graph_rl.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default="CentipedeSix-v1",
        help="the mujoco environment to test",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor for value function",
    )
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--seed", type=int, default=random.randint(0, 10000000))

    # training configuration
    parser.add_argument(
        "--timesteps_per_batch",
        type=int,
        default=2048,
        help="number of steps in the rollout",
    )
    parser.add_argument("--max_timesteps", type=int, default=1e7)
    parser.add_argument(
        "--advantage_method", type=str, default="gae", choices=["gae", "raw"]
    )
    parser.add_argument("--gae_lam", type=float, default=0.95)
    parser.add_argument("--use_gpu", type=int, default=0, help="1 for yes, 0 for no")

    # ppo configuration
    # parser.add_argument("--num_threads", type=int, default=5)
    parser.add_argument("--ppo_clip", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip_value", type=float, default=5.0)
    parser.add_argument("--optim_epochs", type=int, default=10)
    parser.add_argument("--optim_batch_size", type=int, default=256)
    parser.add_argument(
        "--minibatch_all_feed",
        type=int,
        default=0,
        help="if set 1, batch_size = dataset_size",
    )
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--target_kl_high", type=float, default=2)
    parser.add_argument("--target_kl_low", type=float, default=0.5)
    parser.add_argument("--use_weight_decay", type=int, default=0)
    parser.add_argument("--weight_decay_coeff", type=float, default=1e-5)
    parser.add_argument("--use_edge_weight_decay", type=int, default=0)
    parser.add_argument("--edge_weight_decay_coeff", type=float, default=1e-5)
    parser.add_argument("--use_gru_weight_decay", type=int, default=0)
    parser.add_argument("--gru_weight_decay_coeff", type=float, default=1e-5)

    # the checkpoint and summary setting
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--checkpoint_start_iteration", "-c", type=int, default=0)
    parser.add_argument("--min_ckpt_iteration_diff", "-m", type=int, default=70)
    parser.add_argument("--summary_freq", type=int, default=10)
    parser.add_argument("--db_log_freq", type=int, default=1)
    parser.add_argument("--video_freq", type=int, default=2500)
    parser.add_argument(
        "--time_id",
        type=str,
        default=init_path.get_time(),
        help="the special id of the experiment",
    )

    # network settings
    parser.add_argument(
        "--baseline_type",
        type=str,
        default="root",
        choices=["root", "sum", "avg", "nonlinsum"],
    )
    parser.add_argument(
        "--vf_loss_type", type=str, default="squared", choices=["squared", "huber"]
    )
    parser.add_argument(
        "--network_shape",
        type=str,
        default="64,64",
        help="For the general policy network",
    )

    # adaptive kl (we are not using it in the coming experiments)
    parser.add_argument("--kl_alpha", type=float, default=1.5)
    parser.add_argument("--kl_eta", type=float, default=50)

    # adaptive lr (maybe only necessary for the other_agents)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="linear",
        help='["linear", "constant", "adaptive"]',
    )
    parser.add_argument("--lr_alpha", type=int, default=2)

    # debug setting
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--burn_in_running_means", type=int, default=0)
    parser.add_argument("--write_log", type=int, default=1)
    parser.add_argument("--write_summary", type=int, default=1)
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument(
        "--test", type=int, default=0, help="if not 0, test for this number of episodes"
    )
    parser.add_argument(
        "--no_decoder",
        type=int,
        default=0,
        help="If 1, readout is just the identity function",
    )
    parser.add_argument(
        "--local",
        type=int,
        default=0,
        help="If 1, running locally",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--optim_load",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--freeze_vars",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--log_rollouts",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--record_grads",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--record_edge_norm",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--record_spectral_radius",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--record_ppo_clip",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--separate_node_lr",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--separate_edge_lr",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--separate_encoder_lr",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--separate_decoder_lr",
        type=float,
        default=-1.0,
    )

    # the settings for the ggnn
    get_ggnn_config(parser)
    args = parser.parse_args()
    args = post_process(args)

    return args


def get_ggnn_config(parser):
    parser.add_argument(
        "--edge_act_fn",
        type=str,
        default="tanh",
    )
    parser.add_argument(
        "--no_sharing_between_layers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--no_sharing_within_layers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--policy_network_type",
        type=str,
        default=NetworkTypes.nervenet.name,
        choices=[e.name for e in NetworkTypes],
    )
    parser.add_argument(
        "--value_network_type",
        type=str,
        default=NetworkTypes.fixedsizenet.name,
        choices=[e.name for e in NetworkTypes],
    )

    parser.add_argument("--gnn_input_feat_dim", type=int, default=64)
    parser.add_argument("--gnn_node_hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_num_prop_steps", type=int, default=4)
    parser.add_argument("--gnn_init_method", type=str, default="orthogonal")
    parser.add_argument(
        "--gnn_node_option",
        type=str,
        default="nG,nB",
        help="""
                            @brief:
                                @'nG' / 'yG':
                                    whether we allow 'geom' node in the graph
                                @'nB' / 'yB':
                                    whether we allow 'body' node in the graph
                        """,
    )
    parser.add_argument(
        "--root_connection_option",
        type=str,
        default="nN,Rn,uE",
        help="""
                        Change this parameter to change the graph.

                        @Options:
                            ['nN,Rn', 'nN,Rb', 'nN,Ra',
                             'yN,Rn', 'yN,Rb', 'yN,Ra'] + ['uE', 'sE']
                        @brief:
                            @'nN' / 'yN':
                                Whether we allow neighbour connection in the
                                model.
                            @'Rn' / 'Rb' / 'Ra':
                                No additional connection from the root.
                                Add connections from the root to 'body' node.
                                Add connections from the root to all nodes.
                            @'uE' / 'sE':
                                'uE': only one edge type will be used
                                'sE': different edge types will be used
                        """,
    )
    parser.add_argument(
        "--gnn_output_option",
        type=str,
        default="unified",
        help="""
                        @Options:
                            ["unified", "separate", "shared"]

                        @unified:
                            Only one output MLP is used for all output joint
                        @separate:
                            For every joint, a unique MLP is assigned to
                            generate the output torque.
                        @shared:
                            For separate type of nodes, we have different MLP.
                            For example, the left thigh joint and right thigh
                            joint share the same output mlp
                        """,
    )
    parser.add_argument(
        "--gnn_embedding_option",
        type=str,
        default="noninput_shared",
        help="""
                        @Options:
                            ["parameter", "shared", "noninput_separate",
                             "noninput_shared"]

                        @parameter:
                            Embedding input is the node parameter vector
                        @shared:
                            Embedding input is the one-hot encoding. For nodes
                            with the same structure position, e.g. left thigh
                            and right thigh, we provide shared encoding.
                        @separate:
                            Embedding input is the one-hot encoding. For each
                            node, we provide separate encoding.
                        @noninput_separate:
                            Embedding input is just a gather index to select
                            the variable, every input is different
                        @noninput_shared:
                            Embedding input is just a gather index to select
                            the variable, shared embedding is used
                        """,
    )
    parser.add_argument("--baseline_loss_coeff", type=float, default=1.0)
    parser.add_argument(
        "--node_update_method",
        type=str,
        default="GRU",
        help="could be either GRU or MLP update",
    )
    parser.add_argument(
        "--transfer_env",
        type=str,
        default="Nothing2Nothing",
        help="""
                        the pretrained env and the new env name, for example, we
                        can use "SnakeFour2SnakeThree"
                        """,
    )
    parser.add_argument(
        "--test_transfer_tasks",
        type=str,
        nargs="+",
        help="""
                        runs a test thread on a transfer environment and reports
                        results"
                        """,
    )
    parser.add_argument(
        "--max_act_test_transfer_tasks",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--thread_agents",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--logstd_option",
        type=str,
        default="load",
        help="""
                            @"load":
                                load the logstd as what it should be
                            @"load_float":
                                load logstd, but add a constant (bigger
                                exploration)
                            @"fix_float":
                                do not load logstd, load a constant (bigger
                                exploration)
                        """,
    )
    parser.add_argument(
        "--mlp_raw_transfer",
        type=int,
        default=0,
        help="""
                        if set to 1, we just load the raw mlp transfer
                        So we have mlpt, mlpr, and ggnn for transfer learning
                        if set to 2, we skip the unmatched weights
                        """,
    )

    parser.add_argument(
        "--mind_the_receiver",
        type=int,
        default=0,
        help="If set to 1, it will feed the receiver feature to the edge updater as an input similarly to Battaglia et al.",
    )
    parser.add_argument(
        "--record_mlp_activation",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--record_oversmoothing",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--layernorm",
        type=int,
        default=0,
        help="If 1, do layernorm after the MLP activations as in deepminds lib.",
    )
    parser.add_argument(
        "--batchnorm",
        type=int,
        default=0,
        help="If 1, do layernorm after the MLP activations as in deepminds lib.",
    )
    parser.add_argument(
        "--renorm",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--norm_position",
        type=str,
        default="all-post",
        choices=["before", "after-pre", "all-pre", "after-post", "all-post"],
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="orthogonal",
        choices=["orthogonal", "xavier"],
    )
    parser.add_argument(
        "--global_context",
        type=str,
        choices=["hidden", "input", "root", "hidden&root"],
        help="If set to 1, it will append the sum of the messages to the node updater for the core",
    )

    parser.add_argument(
        "--output_block_conditioning",
        type=str,
        choices=["hidden", "root", "hidden&root"],
        help="If set to 1, it will append the sum of the messages to the node updater for the output block",
    )

    parser.add_argument(
        "--msg_aggregation",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="If set to 1, it will append the sum of the messages to the node updater",
    )

    parser.add_argument(
        "--stop_hidden_grad",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--vib_updater",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--vib_beta",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--identity_edge_updater",
        type=int,
        default=0,
        help="If set to 1, there will be no edge updater for the core part, full features of the node will be propagated",
    )
    parser.add_argument(
        "--always_feed_node_features",
        type=int,
        default=0,
        help="If set to 1, node observations will be fed at every step together with the hidden_state and the incoming message similarly as in Battaglia et al.",
    )
    parser.add_argument(
        "--logstd_out_type",
        type=str,
        default="separate",
        choices=["separate", "scalar"],
        help="Separate will create a state-independend logstd vector as in traditional PPO implementation. ",
    )

    parser.add_argument(
        "--std_scalar_val",
        type=float,
        default=0.3,  # as in learnt model from NerveNet
        help="If logstd_out_type is scalar, this number will be used for all joints",
    )

    parser.add_argument(
        "--multitask_critic_joint",
        type=int,
        default=0,
        help="If set to 1, it will learn one critic for all tasks. Not applicable for an MLP critic.",
    )

    parser.add_argument(
        "--multitask_logstd_joint",
        type=int,
        default=1,
        help="If set to 0, it will learn separate logstds per task.",
    )

    parser.add_argument(
        "--label", type=str, default="", help="label of a sacred experiment"
    )

    # mix the grads in the mt regime in this proportion
    # order is the same as order in th
    parser.add_argument(
        "--mt_grad_mixing_ratio",
        type=float,
        nargs="+",
        help="the grads will be mixed in this proportion when multitask. Order is the same as in args.task",
    )

    parser.add_argument(
        "--grad_mixing_targets",
        type=str,
        default="return_deltas",
        choices=["return_deltas", "returns"],
        help="What to use as a target in a GP regression",
    )

    parser.add_argument(
        "--sample_budgeting",
        type=str,
        default="naive",
        choices=[
            "naive",
            "regression",
            "epsgreedy_bandit",
            "thompson_bandit",
            "ucb_bandit",
            "windowed_ucb_bandit",
        ],
        help="What to use as a target in a GP regression",
    )
    parser.add_argument(
        "--bandit_window_size",
        type=int,
        default=150,
        help="Sliding window size for bandits if --sample_budgeting uses the windowed version",
    )

    parser.add_argument(
        "--sequential_task_sampling",
        type=int,
        default=0,
        help="If set to 1, in MT regime, this will use one task per update rather than average all grads from all tasks",
    )

    parser.add_argument(
        "--one_updater_for_all",
        type=int,
        default=0,
        help="If set to 1, one vertex updater is used for all the nodes. NerveNet had separate ones per node type.",
    )

    parser.add_argument(
        "--ent_coef", type=float, default=0.0, help="Entropy coeff for loss",
    )

    parser.add_argument(
        "--reduced_decoder",
        type=int,
        default=0,
        help="If set to 1, decoder shape truncates the first layer from the network_shape",
    )
    parser.add_argument(
        "--reduced_core",
        type=int,
        default=0,
        help="If set to 1, decoder shape truncates the first layer from the network_shape",
    )

    parser.add_argument(
        "--learnable_context",
        type=int,
        default=1,
        help="If set to 1, context modulation is learnt as in the original implementation. If 0, the initial values are preserved.",
    )
    parser.add_argument(
        "--advantage_normalization",
        type=int,
        default=1,
        help="If set to 1, advantage is normalized across the batch.",
    )
    parser.add_argument(
        "--perturbation_analysis",
        type=int,
        default=0,
        help="If set to 1, advantage is normalized across the batch.",
    )
    parser.add_argument(
        "--observation_share",
        type=float,
        default=0.5,
        help="input_feat_dim*observation_share will be the output of the observation feature extractor, the rest will be the embedding of the node",
    )

    parser.add_argument("--pi_opt", type=str, default="adam", choices=["adam", "sgd"])


def post_process(args):
    if args.debug:
        args.write_log = 0
        args.write_summary = 0
        args.monitor = 0

    # parse the network shape
    if args.network_shape == "0":
        args.network_shape = []
    else:
        args.network_shape = [
            int(hidden_size) for hidden_size in args.network_shape.split(",")
        ]
    args.logstd_scalar_val = np.log(args.std_scalar_val)

    if isinstance(args.task, str):
        args.task = [args.task]
    if len(args.task) != len(set(args.task)):
        raise ValueError("You should not input same tasks in the multitask regime")
    if args.mt_grad_mixing_ratio is not None:
        if len(args.task) != len(args.mt_grad_mixing_ratio):
            raise ValueError(
                "mt_grad_mixing_ratio should have the same number of arguments as args.task"
            )
        if not np.equal(1.0, sum(args.mt_grad_mixing_ratio)):
            raise ValueError("mt_grad_mixing_ratio should sum up to 1.0")

    if (
        args.multitask_critic_joint == 1
        and args.value_network_type == NetworkTypes.fixedsizenet.name
        and len(args.task) > 1
    ):
        raise ValueError(
            "We can't train a joint critic if it is an MLP. Use GNN instead or use separate critics."
        )

    if not args.test_transfer_tasks:
        args.test_transfer_tasks = []

    if not args.max_act_test_transfer_tasks:
        args.max_act_test_transfer_tasks = []

    if not args.freeze_vars:
        args.freeze_vars = []

    if (args.ckpt_name is not None) \
            and (args.task[0] not in args.ckpt_name)\
            and (args.transfer_env is 'Nothing2Nothing'):
        raise ValueError(
            "Must set transfer_env if checkpoint and task do not match"
        )
    if not args.thread_agents:
        args.thread_agents = args.task
    
    return args
