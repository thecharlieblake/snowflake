# Snowflake

This repo contains software for our paper:
*Snowflake: Scaling GNNs to High-Dimensional Continuous Control via Parameter Freezing*.

Our code builds the [NerveNet codebase](https://github.com/WilsonWangTHU/NerveNet) written for *NerveNet: Learning Structured Policy with Graph Neural Networks*
by Wang et al., 2018.

## Build

To set up Snowflake, we recommend the use of the docker scripts contained here.

### Requirements

One pre-requisite is the possession of MuJoCo license,
in the form of a file named `mjkey.txt`.
Our `Dockerfile` expects this to be placed in the root of this repo
(i.e. `snowflake/mjkey.txt`).
This is not supplied and **must be provided by the user**.

### Docker Build

To install the necessary dependencies, run:
```
./docker_build.sh
```
This will run our `Dockerfile` to create an image named `snowflake`.
If you wish to install without the use of Docker then consult our `Dockerfile` for the required
dependencies.

(Note: Owing to our use of MuJoCo there are quite a few dependencies required.)

This `Dockerfile` is set up to run Snowflake on CPUs. In most cases MuJoCo is CPU intensive
enough that a GPU is of little benefit.
If one does wish to use a GPU then modifications to the `Dockerfile` will be required.

## Usage

### Docker Run

The next step is to run:
```
./docker_run.sh
```
This creates and enters a container, mounting this Snowflake repo as the working directory.
As this is a mounted volume, changes made to files in this repo within the container will
persist on the main system.

From within this container, to use Snowflake, run:
```
python main.py [--configs...]
```

### Configuration

To see what configuration options are available,
either run `python main.py --help`, or consult `util/config.py`.

For example,
```
python main.py
  --task CentipedeSix-v1
  --thread_agents CentipedeSix-v1 CentipedeSix-v1
  --network_shape 64,64
  --lr 0.0003
  --timesteps_per_batch 2048
  --optim_batch_size 256
  --label example_run
```
will run `Centipede-6` across two threads, with a 64x64 message function,
a learning rate of 0.003, a batch size of 2048 and minibatch size of 256.

Each of these configurations are set to have a default which typically matches
the standard value used in our experiments (although this may vary across tasks).
Simply running `python main.py` with no configurations will train a `Centipede-6`
using the default settings.

### Output

Results and training statistics will be printed to standard output, and also recorded in the
`log/` directory.

They are also recorded in the `summary/` directory, in a format that can be used with
Tensorboard, e.g. `tensorboard --logdir summary/`.

Checkpoints of the model are also saved periodically in the `checkpoint/` directory.
The model can be loaded into the agent using the command
`python main.py --ckpt_name checkpoint/etc...`.
This can either be used to resume stopped training runs, or load a model trained on one
task into the agent of another task.
