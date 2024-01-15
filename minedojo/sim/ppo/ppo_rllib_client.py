
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
from minedojo.sim.ppo.ppo_rllib import RllibPPOModel
from ray.rllib.models import ModelCatalog
from minedojo.sim.rllib.rllib import gen_trainer_from_params, save_trainer, load_trainer
LOCAL_TESTING = True


def _env_creator(env_config):
    from minedojo.sim.rllib.rllib import MineDojoMultiAgent

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

    #num_workers = 10 if not LOCAL_TESTING else 2

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0 if not LOCAL_TESTING else 0

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 3000 if not LOCAL_TESTING else 200
    
    # size of minibatches we divide up each batch into before
    # performing gradient steps
    sgd_minibatch_size = 120 if not LOCAL_TESTING else 8

    # Rollout length
    rollout_fragment_length = 500 if not LOCAL_TESTING else 100

    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 5 if not LOCAL_TESTING else 5

    # Stepsize of SGD.
    lr = 5e-5
    
    lr_start = 2.5e-4
    lr_end = 5e-5
    lr_time = 50 * 1000000

    # Learning rate schedule.
    lr_schedule = [
            [0, lr_start],
            [lr_time, lr_end],
        ]
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
    num_sgd_iter = 2 if not LOCAL_TESTING else 2

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 25

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 1

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 100

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
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff,
    )

    experiment_name = "{0}_{1}_{2}".format("PPO", "MineDojo", params_str)

    # Max episode length
    horizon = 100

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0


    # to be passed into the rllib.PPOTrainer class
    training_params = {
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
        "custom_model_id": "MyPPOModel",
        "custom_model_cls": RllibPPOModel,
        "temp_dir" : temp_dir,
        "env_creator": _env_creator,
    }

    model_params = {
        "use_lstm": use_lstm,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "CELL_SIZE": 256,
        "D2RL": D2RL,
    }

    params = {
        "model_params": model_params,
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
    saved_path = params["resume_checkpoint_path"]
    if saved_path:
        trainer = load_trainer(save_path=saved_path, true_num_workers=False)
    else:
        trainer = gen_trainer_from_params(params)
    #os.system("pkill -9 -f java")
    result = []
    for i in range(params["num_training_iters"]):
        if params["verbose"]:
            print("Starting training iteration", i)
        result = trainer.train()
        save_path = save_trainer(trainer, params)
        if params["verbose"]:
            print("save path:", save_path)
            print(result)
    return result


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
  
