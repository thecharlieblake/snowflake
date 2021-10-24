from util import init_path
import numpy as np
from util import logger
import os
from graph_util import structure_mapper
import re

init_path.bypass_frost_warning()


def save_tf_model(sess, model_path, tf_var_list=[]):
    """
        @brief: save the tensorflow variables into a numpy npy file
    """
    is_file_valid(model_path, save_file=True)
    output_save_list = tf_model_to_dict(sess, tf_var_list)

    # save the model
    np.save(model_path, output_save_list)


def tf_model_to_dict(sess, tf_var_list=[]):
    logger.info("\tSAVING tensorflow variables")

    # get the tf weights one by one
    output_save_list = dict()
    weights_list = sess.run(tf_var_list)
    output_save_list = {var.name: weights for var, weights in zip(tf_var_list, weights_list)}
    logger.info("\t\t[Checkpoint] saving tf parameters {}".format([var.name for var in tf_var_list]))

    return output_save_list


def save_numpy_model(model_path, numpy_var_list=[]):
    """
        @brief: save the numpy variables into a numpy npy file
    """
    is_file_valid(model_path, save_file=True)

    logger.info("\tSAVING numpy variables")

    # get the numpy weights one by one
    output_save_list = dict()
    for key, var in list(numpy_var_list.items()):
        output_save_list[key] = var
        logger.info("\t\t[Checkpoint] saving numpy parameter {}".format(key))
    # save the model
    np.save(model_path, output_save_list)


def load_tf_model_from_dict(
    sess,
    output_save_list,
    tf_var_list=[],
    remove_var_str="",
    transfer_env="Nothing2Nothing",
    logstd_option="load",
    gnn_option_list=None,
    mlp_raw_transfer=False,
):
    assign_tuples = []
    tf_name_list_stripped = [re.sub(remove_var_str, "", var.name) for var in tf_var_list]

    # get the weights one by one
    for name, val in list(output_save_list.items()):
        name_stripped = re.sub(remove_var_str, "", name)
        if name_stripped in tf_name_list_stripped or "logstd" in name_stripped:
            logger.info("\t\tloading TF pretrained parameters {}".format(name))
            if "logstd" in name_stripped:
                name_to_remove = [el for el in tf_name_list_stripped if "logstd" in name_stripped][0]
                tf_name_list_stripped.remove(name_to_remove)
                var = [var for var in tf_var_list if re.sub(remove_var_str, "", var.name) == name_to_remove][0]
            else:
                tf_name_list_stripped.remove(name_stripped)  # just for sanity check
                # pick up the variable that has the name
                var = [var for var in tf_var_list if re.sub(remove_var_str, "", var.name) == name_stripped][0]

            # NOTE: this is important part
            if np.prod(var.get_shape().as_list()) != np.prod(val.shape):
                # if there is an unmatch, then we must be doing a transfer
                # learning
                if transfer_env == "Nothing2Nothing" or mlp_raw_transfer == 2:
                    # for the baseline, no one is preloading!
                    logger.warning("Skipping variable: {}".format(var.name))
                else:
                    # load the policy logstd, happens for the gnn, mlpr and mlpt
                    if "policy_logstd" in var.name:
                        if "fix" in logstd_option:
                            added_constant = float(logstd_option.split("_")[1])
                        else:
                            added_constant = 0.0

                        new_value = structure_mapper.map_output(
                            transfer_env, val, added_constant, gnn_option_list
                        )
                        assign_tuples.append((var, new_value))
                    elif "policy_0/w:0" in var.name:
                        new_value = structure_mapper.map_input(
                            transfer_env, val, 0.0, gnn_option_list
                        )
                        assign_tuples.append((var, new_value))
                    elif "policy_output" in var.name:
                        new_value = structure_mapper.map_output(
                            transfer_env, val, 0.0, gnn_option_list
                        )
                        assign_tuples.append((var, new_value))
                    else:
                        logger.warning("Skipping variable: {}".format(var.name))

            else:
                if "policy_logstd" in var.name and "fix" in logstd_option:
                    single_value = float(logstd_option.split("_")[1])
                    assign_tuples.append((var, np.ones(var.get_shape().as_list()) * single_value))
                else:
                    assign_tuples.append((var, val))
        else:
            logger.warning("\t\t**** Parameters Not Exist **** {}".format(name))

    assign_ops = []
    for var, val in assign_tuples:
        assign_ops.append(var.assign(val))
    sess.run(assign_ops)

    if len(tf_name_list_stripped) > 0:
        logger.warning(
            "Some parameters are not load from the checkpoint: {}".format(tf_name_list_stripped)
        )


def load_tf_model(
    sess,
    model_path,
    tf_var_list=[],
    ignore_prefix="INVALID",
    transfer_env="Nothing2Nothing",
    logstd_option="load",
    gnn_option_list=None,
    mlp_raw_transfer=False,
):
    """
        @brief: load the tensorflow variables from a numpy npy files
    """
    is_file_valid(model_path)
    logger.info("\tLOADING tensorflow variables")
    output_save_list = np.load(model_path, encoding="latin1", allow_pickle=True).item()
    load_tf_model_from_dict(sess, output_save_list, tf_var_list, ignore_prefix, transfer_env, logstd_option,
                            gnn_option_list, mlp_raw_transfer)


def load_numpy_model(model_path, numpy_var_list={}):
    """
        @brief: load numpy variables from npy files. The variables could be
            from baseline or from ob_normalizer
        @output:
            It is worth mentioning that this function only returns the value,
            but won't load the value (while the tf variables will be loaded at
            the same time)
    """
    is_file_valid(model_path)
    logger.info("LOADING numpy variables")

    output_save_list = np.load(model_path, encoding="latin1", allow_pickle=True).item()
    return output_save_list
    numpy_name_list = [key for key, val in list(numpy_var_list.items())]

    # get the weights one by one
    for name, val in list(output_save_list.items()):
        if name in numpy_name_list:
            logger.info("\t\tloading numpy pretrained parameters {}".format(name))
            numpy_name_list.remove(name)  # just for sanity check
            numpy_var_list[name] = val
        else:
            logger.warning("\t\t**** Parameters Not Exist **** {}".format(name))

    if len(numpy_name_list) > 0:
        logger.warning(
            "Some parameters are not load from the checkpoint: {}".format(
                numpy_name_list
            )
        )
    return numpy_var_list


"""
    @brief:
        The following variables might be a little bit of out-dated
"""


def model_save_from_list(sess, model_path, tf_var_list=[], numpy_var_list={}):
    """
        @brief:
            if the var list is given, we just save them
    """
    if not model_path.endswith(".npy"):
        model_path = model_path + ".npy"

    logger.info("saving checkpoint to {}".format(model_path))
    output_save_list = dict()

    # get the tf weights one by one
    for var in tf_var_list:
        weights = sess.run(var)
        output_save_list[var.name] = weights
        logger.info("[checkpoint] saving tf parameter {}".format(var.name))

    # get the numpy weights one by one
    for key, var in list(numpy_var_list.items()):
        output_save_list[key] = var
        logger.info("[checkpoint] saving numpy parameter {}".format(key))

    # save the model
    np.save(model_path, output_save_list)

    return


def model_load_from_list(
    sess,
    model_path,
    tf_var_list=[],
    numpy_var_list={},
    target_scope_switch="trpo_agent_policy",
):
    """
        @brief:
            if the var list is given, we just save them
        @input:
            @target_scope_switch:
    """
    if not model_path.endswith(".npy"):
        model_path = model_path + ".npy"
        logger.warning('[checkpoint] adding the ".npy" to the path name')
    logger.info("[checkpoint] loading checkpoint from {}".format(model_path))

    output_save_list = np.load(model_path, encoding="latin1", allow_pickle=True).item()
    tf_name_list = [var.name for var in tf_var_list]
    numpy_name_list = [key for key, val in list(numpy_var_list.items())]

    # get the weights one by one
    for name, val in list(output_save_list.items()):
        name = name.replace("trpo_agent_policy", target_scope_switch)
        if name not in tf_name_list and name not in numpy_var_list:
            logger.info("**** Parameters Not Exist **** {}".format(name))
            continue
        elif name in tf_name_list:
            logger.info("loading TF pretrained parameters {}".format(name))
            tf_name_list.remove(name)  # just for sanity check

            # pick up the variable that has the name
            var = [var for var in tf_var_list if var.name == name][0]
            assign_op = var.assign(val)
            sess.run(assign_op)  # or `assign_op.op.run()`
        else:
            logger.info("loading numpy pretrained parameters {}".format(name))
            numpy_name_list.remove(name)  # just for sanity check

            # pick up the variable that has the name
            numpy_var_list[name] = val

    if len(tf_name_list) or len(numpy_name_list) > 0:
        logger.warning(
            "Some parameters are not load from the checkpoint: {}\n {}".format(
                tf_name_list, numpy_name_list
            )
        )
    return numpy_var_list


def is_file_valid(model_path, save_file=False):
    assert model_path.endswith(".npy"), logger.error(
        "Invalid file provided {}".format(model_path)
    )
    if not save_file:
        assert os.path.exists(model_path), logger.error(
            "file not found: {}".format(model_path)
        )
    logger.info("[LOAD/SAVE] checkpoint path is {}".format(model_path))
