import os
import minedojo
#from minedojo.sim import InventoryItem
import ray


from ray.tune.result import DEFAULT_RESULTS_DIR
from minedojo.sim import *

import logging
from ray.tune.result import DEFAULT_RESULTS_DIR
import ray
from datetime import datetime
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

LOCAL_TESTING = True



def _env_creator(env_config):
    from minedojo.sim.rllib.rllib import MineDojoMultiAgent
    print("env_config: ")
    print(env_config)
    return MineDojoMultiAgent.from_config()

def my_config():
    ### Resume chekpoint_path ###
    resume_checkpoint_path = None

    ### Model params ###

    # Whether dense reward should come from potential function or not
    use_phi = True

    # whether to use recurrence in ppo model
    use_lstm = False

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3


    # whether to use D2RL https://arxiv.org/pdf/2010.09163.pdf (concatenation the result of last conv layer to each hidden layer); works only when use_lstm is False
    D2RL = False
    ### Training Params ###

    num_workers = 30 if not LOCAL_TESTING else 2

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 12000 if not LOCAL_TESTING else 800

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    sgd_minibatch_size = 2000 if not LOCAL_TESTING else 800

    # Rollout length
    rollout_fragment_length = 400

    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 420 if not LOCAL_TESTING else 2

    # Stepsize of SGD.
    lr = 5e-5

    # Learning rate schedule.
    lr_schedule = None
    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]
    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1

    # Discount factor
    gamma = 0.99

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.98

    # Whether the value function shares layers with the policy model
    vf_share_layers = True

    # How much the loss of the value network is weighted in overall loss
    vf_loss_coeff = 1e-4

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 0.2
    entropy_coeff_end = 0.1
    entropy_coeff_horizon = 3e5

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2

    # PPO clipping factor
    clip_param = 0.05

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    num_sgd_iter = 8 if not LOCAL_TESTING else 1

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 25

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 1

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400

    # Number of games to simulation each evaluation
    evaluation_num_games = 1

    # Whether to display rollouts in evaluation
    evaluation_display = False

    # Where to log the ray dashboard stats
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp")

    # Where to store model checkpoints and training stats
    results_dir = DEFAULT_RESULTS_DIR

    # Whether tensorflow should execute eagerly or not
    eager = False

    # Whether to log training progress and debugging info
    verbose = True

    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp")

    ### Environment Params ### TODO
    # Which mission to start
    mission = "harvest_milk"

    #agent info (Expand)
    initial_inventory1 = None
    initial_inventory2 = None


    # Name of directory to store training results in (stored in ~/ray_results/<experiment_name>)

    params_str = str(use_phi) + "_nw=%d_vf=%f_es=%f_en=%f_kl=%f" % (
        num_workers,
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff,
    )

    experiment_name = "{0}_{1}_{2}".format("PPO", "MineDojo", params_str)

    # Max episode length
    horizon = 400

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0


    # to be passed into the rllib.PPOTrainer class
    training_params = {
        "num_workers": num_workers,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "num_sgd_iter": num_sgd_iter,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "grad_clip": grad_clip,
        "gamma": gamma,
        "lambda": lmbda,
        "vf_share_layers": vf_share_layers,
        "vf_loss_coeff": vf_loss_coeff,
        "kl_coeff": kl_coeff,
        "clip_param": clip_param,
        "num_gpus": num_gpus,
        "seed": seed,
        "evaluation_interval": evaluation_interval,
        "entropy_coeff_schedule": [
            (0, entropy_coeff_start),
            (entropy_coeff_horizon, entropy_coeff_end),
        ],
        "eager_tracing": eager,
        "log_level": "WARN" if verbose else "ERROR",
    }

    # To be passed into AgentEvaluator constructor and _evaluate function
    evaluation_params = {
        "ep_length": evaluation_ep_length,
        "num_games": evaluation_num_games,
        "display": evaluation_display,
    }

    ray_params = {
        "temp_dir" : temp_dir,
        "env_creator": _env_creator,
    }

    params = {
        "training_params": training_params,
        "shared_policy": shared_policy,
        "num_training_iters": num_training_iters,
        "evaluation_params": evaluation_params,
        "experiment_name": experiment_name,
        "save_every": save_freq,
        "seeds": seeds,
        "results_dir": results_dir,
        "ray_params": ray_params,
        "resume_checkpoint_path": resume_checkpoint_path,
        "verbose": verbose,
    }
    return params

def run(params):
    run_name = params["experiment_name"]
    trainer = gen_trainer_from_params(params)
    os.system("pkill -9 -f java")
    for i in range(params["num_training_iters"]):
        if params["verbose"]:
            print("Starting training iteration", i)
        #result = trainer.train()
    return result

def gen_trainer_from_params(params):
    print("hello")
    print(params["ray_params"]["temp_dir"])
    if not ray.is_initialized():
        init_params = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "_temp_dir": params["ray_params"]["temp_dir"],
            "log_to_driver": params["verbose"],
            "logging_level": logging.INFO
            if params["verbose"]
            else logging.CRITICAL,
        }
        ray.init(**init_params)
        register_env("MineDojo_Env", params["ray_params"]["env_creator"])

        training_params = params["training_params"]
        #env = minedojo.make(task_id="harvest_milk", image_size=(288,512))
        print(training_params["num_gpus"])
        logdir_prefix = "{0}_{1}_{2}".format(
        params["experiment_name"], params["training_params"]["seed"], timestr
        )

        def select_policy():
            return "ppo"

        config = PPOConfig()
        config = config.resources(num_gpus=1, num_learner_workers = 0)
        config = config.rollouts(num_rollout_workers=1)
        config = config.training(model={'vf_share_layers' : training_params["vf_share_layers"]})
        config = config.training(lr_schedule=training_params["lr_schedule"],
                                 use_gae=True,lambda_=training_params["lambda"],
                                 use_kl_loss=True, kl_coeff=training_params["kl_coeff"],
                                 sgd_minibatch_size=training_params["sgd_minibatch_size"],
                                 num_sgd_iter=training_params["num_sgd_iter"],
                                 vf_loss_coeff=training_params["vf_loss_coeff"],
                                 clip_param=training_params["clip_param"],
                                 grad_clip=training_params["grad_clip"],
                                 entropy_coeff=training_params["entropy_coeff_schedule"],
                                )
        algo = config.build(env="MineDojo_Env")
        return algo

if __name__ == "__main__":
    params = my_config()
    seeds = params["seeds"]
    results = []
    for seed in seeds:
        # Override the seed
        params["training_params"]["seed"] = seed

        # Do the thing
        result = run(params)
        results.append(result)
    print(results)