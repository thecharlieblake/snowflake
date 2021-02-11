import tensorflow as tf
from network.mlp_network import MlpBaseline
from network.gated_graph_network import GatedGraphBaseline
from network.gated_graph_network import GatedGraphPolicy, NerveNetPlusPolicy
from network import mlp_network
from network import NetworkTypes


def build_session(use_gpu):
    if use_gpu:
        config = tf.compat.v1.ConfigProto(device_count={"GPU": 1})
    else:
        config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    config.gpu_options.allow_growth = True  # don't take full gpu memory
    return tf.compat.v1.Session(config=config)


def build_baseline_network(
    args,
    session,
    name_scope,
    observation_size,
    task_name,
    ob_placeholder=None,
    placeholder_list=None,
    node_info=None,
):
    """
        @brief:
            Build the baseline network, and fetch the baseline variable list
    """
    # step 1: build the baseline network
    if args.value_network_type == NetworkTypes.nervenet.name:
        baseline_network = GatedGraphBaseline(
            session=session,
            name_scope=name_scope,
            input_size=observation_size,
            task_name=task_name,
            placeholder_list=placeholder_list,
            ob_placeholder=None,
            trainable=True,
            build_network_now=True,
            args=args,
            node_info=node_info,
        )
    else:
        baseline_network = MlpBaseline(
            session=session,
            name_scope=name_scope,
            input_size=observation_size,
            task_name=task_name,
            ob_placeholder=ob_placeholder,
            build_network_now=True,
            args=args,
        )

    return baseline_network


def build_policy_network(
    args,
    session,
    name_scope,
    observation_size,
    task_name,
    action_size,
    ob_placeholder=None,
    node_info=None,
    inference_mode=False,
):
    if args.policy_network_type == NetworkTypes.nervenet.name:
        policy_network = GatedGraphPolicy(
            session=session,
            name_scope=name_scope,
            input_size=observation_size,
            task_name=task_name,
            output_size=action_size,
            ob_placeholder=None,
            trainable=True,
            build_network_now=True,
            placeholder_list=None,
            args=args,
            node_info=node_info,
            inference_mode=inference_mode,
        )
    elif args.policy_network_type == NetworkTypes.nervenetpp.name:
        policy_network = build_nervenetpp_policy(
            args,
            session,
            name_scope,
            observation_size,
            task_name,
            action_size,
            is_step=False,
            node_info=node_info,
            inference_mode=inference_mode,
        )
    else:
        policy_network = mlp_network.MlpNetwork(
            session=session,
            name_scope=name_scope + "_policy",
            input_size=observation_size,
            output_size=action_size,
            ob_placeholder=ob_placeholder,
            trainable=True,
            build_network_now=True,
            args=args,
        )
    return policy_network


def build_nervenetpp_policy(
    args,
    session,
    name_scope,
    observation_size,
    task_name,
    action_size,
    is_step,
    node_info=None,
):
    return NerveNetPlusPolicy(
        session=session,
        name_scope=name_scope + "_policy" if not is_step else "step_policy",
        input_size=observation_size,
        output_size=action_size,
        task_name=task_name,
        ob_placeholder=None,
        trainable=True,
        build_network_now=True,
        placeholder_list=None,
        args=args,
        node_info=node_info,
    )
