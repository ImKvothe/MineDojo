import minedojo
import logging
from ray.tune.result import DEFAULT_RESULTS_DIR
import ray
from datetime import datetime
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig


class MineDojoMultiAgent(MultiAgentEnv):
    def __init__(self, base_env):
        self.base_env = base_env
        self._episode_id = None
        self._agent_ids = self.reset()

    def step(self, action_dict):
        action = [
            action_dict[self.curr_agents[0]],
            action_dict[self.curr_agents[1]],
        ]
        print(action)
        obs, reward, done, info = self.base_env.step(action)
        obs2 = {self.curr_agents[0]: obs[0], self.curr_agents[1]: obs[1]}
        rewards = {self.curr_agents[0]: reward[0], self.curr_agents[1]: reward[1]}
        dones = {self.curr_agents[0]: done[0], self.curr_agents[1]: done[1]}
        infos = {self.curr_agents[0]: info[0], self.curr_agents[1]: info[1]}
        return obs2, rewards, dones, infos

    def reset(self):
        obs = self.base_env.reset()
        obs0 = obs[0]
        obs1 = obs[1]
        self.curr_agents = self._populate_agents()
        return {self.curr_agents[0]: obs0, self.curr_agents[1]: obs1 }

    def _populate_agents(self):
        agents = ["ppo_0", "ppo_1"]
        self._setup_action_space()
        self._setup_observation_space()
        return agents


    def _setup_action_space(self):
        self.action_space = self.base_env.action_space
        

    def _setup_observation_space(self):
        self.observation_space = self.base_env.observation_space
        print(self.observation_space)

def gen_trainer_from_params(params):
    print(params)
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
        env = minedojo.make(task_id="harvest_milk", image_size=(288,512))
        print(training_params["num_gpus"])
        logdir_prefix = "{0}_{1}_{2}".format(
        params["experiment_name"], params["training_params"]["seed"], timestr
        )
        
        def select_policy():
            return "ppo"
        
        config = PPOConfig()
        config = config.resources(num_gpus=training_params["num_gpus"])
        config = config.rollouts(num_rollout_workers=training_params["num_workers"])
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
         
        