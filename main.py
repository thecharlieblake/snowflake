import tensorflow as tf
from agent import optimization_agent
from util.config import get_config
from util import logger
import numpy as np

tf.compat.v1.disable_eager_execution()

def main(args):
    if args.write_log:
        logger.set_file_handler(
            path=args.output_dir,
            prefix="mujoco_" + "_".join(args.task),
            time_str=args.time_id,
        )

    learner_agent = optimization_agent.optimization_agent(args)
    i = 0
    while True:
        results = learner_agent.update_step()
        totalsteps = results["totalsteps"]
        logger.info("%d total steps have happened" % totalsteps)

        if totalsteps > args.max_timesteps:
            break
        i += 1
    learner_agent.end()

    if args.test:
        logger.info(
            "Test performance ({} rollouts): {} (std: {})".format(
                args.test, results["avg_reward"], results["std_reward"]
            )
        )

        logger.info(
            "max: {}, min: {}, median: {}".format(
                results["max_reward"], results["min_reward"], results["median_reward"]
            )
        )

        logger.info(
            "raw_rewards: {}".format(
                np.array2string(results["raw_rewards"], separator=",")
            )
        )
        return np.array2string(results["raw_rewards"], separator=",")


if __name__ == "__main__":
    args = get_config()
    main(args)
