from util import init_path
from util import logger
from . import mujoco_parser
import numpy as np


_BASE_PATH = init_path.get_abs_base_dir()


def construct_ob_size_dict(node_info, input_feat_dim):
    """
        @brief: for each node type, we collect the ob size for this type
    """
    node_info["ob_size_dict"] = {}
    for node_type in node_info["node_type_dict"]:
        node_ids = node_info["node_type_dict"][node_type]

        # record the ob_size for each type of node
        if node_ids[0] in node_info["input_dict"]:
            node_info["ob_size_dict"][node_type] = len(
                node_info["input_dict"][node_ids[0]]
            )
        else:
            node_info["ob_size_dict"][node_type] = 0

        node_ob_size = [
            len(node_info["input_dict"][node_id])
            for node_id in node_ids
            if node_id in node_info["input_dict"]
        ]

        if len(node_ob_size) == 0:
            continue

        assert node_ob_size.count(node_ob_size[0]) == len(node_ob_size), logger.error(
            "Nodes (type {}) have wrong ob size: {}!".format(node_type, node_ob_size)
        )

    return node_info


def get_inverse_type_offset(node_info, mode):
    assert mode in ["output", "node"], logger.error("Invalid mode: {}".format(mode))
    node_info["inverse_" + mode + "_extype_offset"] = []
    node_info["inverse_" + mode + "_intype_offset"] = []
    node_info["inverse_" + mode + "_self_offset"] = []
    node_info["inverse_" + mode + "_original_id"] = []
    current_offset = 0
    for mode_type in node_info[mode + "_type_dict"]:
        i_length = len(node_info[mode + "_type_dict"][mode_type])
        # the original id
        node_info["inverse_" + mode + "_original_id"].extend(
            node_info[mode + "_type_dict"][mode_type]
        )

        # In one batch, how many element is listed before this type?
        # e.g.: [A, A, C, B, C, A], with order [A, B, C] --> [0, 0, 4, 3, 4, 0]
        node_info["inverse_" + mode + "_extype_offset"].extend(
            [current_offset] * i_length
        )

        # In current type, what is the position of this node?
        # e.g.: [A, A, C, B, C, A] --> [0, 1, 0, 0, 1, 2]
        node_info["inverse_" + mode + "_intype_offset"].extend(list(range(i_length)))

        # how many nodes are in this type?
        # e.g.: [A, A, C, B, C, A] --> [3, 3, 2, 1, 2, 3]
        node_info["inverse_" + mode + "_self_offset"].extend([i_length] * i_length)
        current_offset += i_length

    sorted_id = np.array(node_info["inverse_" + mode + "_original_id"])
    sorted_id.sort()
    node_info["inverse_" + mode + "_original_id"] = [
        node_info["inverse_" + mode + "_original_id"].index(i_node)
        for i_node in sorted_id
    ]

    node_info["inverse_" + mode + "_extype_offset"] = np.array(
        [
            node_info["inverse_" + mode + "_extype_offset"][i_node]
            for i_node in node_info["inverse_" + mode + "_original_id"]
        ]
    )
    node_info["inverse_" + mode + "_intype_offset"] = np.array(
        [
            node_info["inverse_" + mode + "_intype_offset"][i_node]
            for i_node in node_info["inverse_" + mode + "_original_id"]
        ]
    )
    node_info["inverse_" + mode + "_self_offset"] = np.array(
        [
            node_info["inverse_" + mode + "_self_offset"][i_node]
            for i_node in node_info["inverse_" + mode + "_original_id"]
        ]
    )

    return node_info


def get_receive_send_idx(node_info):
    # register the edges that shows up, get the number of edge type
    edge_dict = mujoco_parser.EDGE_TYPE
    edge_type_list = []  # if one type of edge exist, register

    for edge_id in range(1000):
        if edge_id == 0:
            continue  # the self loop is not considered here
        if (node_info["relation_matrix"] == edge_id).any():
            edge_type_list.append(edge_id)

    node_info["edge_type_list"] = edge_type_list
    node_info["num_edge_type"] = len(edge_type_list)

    receive_idx_raw = {}
    receive_idx = []
    send_idx = {}
    for edge_type in node_info["edge_type_list"]:
        receive_idx_raw[edge_type] = []
        send_idx[edge_type] = []
        i_id = np.where(node_info["relation_matrix"] == edge_type)
        for i_edge in range(len(i_id[0])):
            send_idx[edge_type].append(i_id[0][i_edge])
            receive_idx_raw[edge_type].append(i_id[1][i_edge])
            receive_idx.append(i_id[1][i_edge])

    node_info["receive_idx"] = receive_idx
    node_info["receive_idx_raw"] = receive_idx_raw
    node_info["send_idx"] = send_idx
    node_info["num_edges"] = len(receive_idx)

    return node_info


def nervenetplus_step_assign(rollout_data, nstep):
    data_batch_id = []  # the start of nstep subsamples
    current_pos = 0
    for i_episode in rollout_data:
        episode_length = len(i_episode["rewards"])
        start_pos = np.random.randint(nstep) + current_pos

        # length = 6, start_pos = 1, nstep = 5 --> [1,2,3,4,5]: num_pos = 1
        # print((episode_length + current_pos - start_pos) / nstep)
        num_pos = int(np.floor((episode_length + current_pos - start_pos) / nstep))

        if num_pos == 0:
            start_ids = [0]
        elif num_pos < 0:
            start_ids = []
        else:
            start_ids = [i_pos * nstep + start_pos for i_pos in range(num_pos)]
        data_batch_id.extend(start_ids)
        current_pos += episode_length

    return data_batch_id, len(data_batch_id)
