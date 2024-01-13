
import minedojo
import logging
from ray.tune.result import DEFAULT_RESULTS_DIR
import ray
import os
import dill
import random
from datetime import datetime
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from collections import OrderedDict

import copy
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from minedojo.sim import ALL_CRAFT_SMELT_ITEMS, ALL_ITEMS
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
LOCAL_TESTING = True
act_space = None
obs_space = None
iteration = 0

N_ALL_ITEMS = len(ALL_ITEMS)
ACTION_MAP = {
    0: np.array([0, 0, 0, 12, 12, 0, 0, 0]),  # no-op
    1: np.array([1, 0, 0, 12, 12, 0, 0, 0]),  # forward
    2: np.array([2, 0, 0, 12, 12, 0, 0, 0]),  # back
    3: np.array([0, 1, 0, 12, 12, 0, 0, 0]),  # left
    4: np.array([0, 2, 0, 12, 12, 0, 0, 0]),  # right
    5: np.array([1, 0, 1, 12, 12, 0, 0, 0]),  # jump + forward
    6: np.array([1, 0, 2, 12, 12, 0, 0, 0]),  # sneak + forward
    7: np.array([1, 0, 3, 12, 12, 0, 0, 0]),  # sprint + forward
    8: np.array([0, 0, 0, 11, 12, 0, 0, 0]),  # pitch down (-15)
    9: np.array([0, 0, 0, 13, 12, 0, 0, 0]),  # pitch up (+15)
    10: np.array([0, 0, 0, 12, 11, 0, 0, 0]),  # yaw down (-15)
    11: np.array([0, 0, 0, 12, 13, 0, 0, 0]),  # yaw up (+15)
    12: np.array([0, 0, 0, 12, 12, 3, 0, 0]),  # attack
    13: np.array([0, 0, 0, 12, 12, 1, 0, 0]),  # use
    14: np.array([0, 0, 0, 12, 12, 2, 0, 0]),  # drop
    15: np.array([0, 0, 0, 12, 12, 4, 0, 0]),  # craft
    16: np.array([0, 0, 0, 12, 12, 5, 0, 0]),  # equip
    17: np.array([0, 0, 0, 12, 12, 6, 0, 0]),  # place
    18: np.array([0, 0, 0, 12, 12, 7, 0, 0]),  # destroy
}
ITEM_ID_TO_NAME = dict(enumerate(ALL_ITEMS))
ITEM_NAME_TO_ID = dict(zip(ALL_ITEMS, range(N_ALL_ITEMS)))


class MineDojoMultiAgent(MultiAgentEnv):
    def __init__(self,
        base_env,
        height: int = 64,
        width: int = 64,
        sticky_attack: Optional[int] = 30,
        pitch_limits: Tuple[int, int] = (-60, 60),
        seed: Optional[int] = None,
        **kwargs: Optional[Dict[Any, Any]],):

        self._height = height
        self._width = width
        self._pitch_limits = pitch_limits
        self._sticky_attack = sticky_attack
        self._sticky_attack_counter1 = 0
        self._sticky_attack_counter2 = 0

        self._pos1 = kwargs.get("start_position1", None)
        self._pos2 = kwargs.get("start_position2", None)
        self._start_pos1 = copy.deepcopy(self._pos1)
        self._start_pos2 = copy.deepcopy(self._pos2)

        self.base_env = base_env
        self._episode_id = None
        super().__init__()

        self._agent_ids = set(self.reset()[0].keys())

    def reset(self, *, seed=None, options=None):
        global iteration
        iteration = 0
        print("reset")
        obs = self.base_env.reset()
        self._pos1 = {
            "x": float(obs[0]["location_stats"]["pos"][0]),
            "y": float(obs[0]["location_stats"]["pos"][1]),
            "z": float(obs[0]["location_stats"]["pos"][2]),
            "pitch": float(obs[0]["location_stats"]["pitch"].item()),
            "yaw": float(obs[0]["location_stats"]["yaw"].item()),
        }
        self._pos2 = {
            "x": float(obs[1]["location_stats"]["pos"][0]),
            "y": float(obs[1]["location_stats"]["pos"][1]),
            "z": float(obs[1]["location_stats"]["pos"][2]),
            "pitch": float(obs[1]["location_stats"]["pitch"].item()),
            "yaw": float(obs[1]["location_stats"]["yaw"].item()),
        }

        self._inventory_max = np.zeros(N_ALL_ITEMS)
        self.curr_agents = self._populate_agents()

        info = {
            "life_stats1": {
                "life": float(obs[0]["life_stats"]["life"].item()),
                "oxygen": float(obs[0]["life_stats"]["oxygen"].item()),
                "food": float(obs[0]["life_stats"]["food"].item()),
            },
            "location_stats1": copy.deepcopy(self._pos1),
            "biomeid1": float(obs[0]["location_stats"]["biome_id"].item()),
            "life_stats2": {
                "life": float(obs[1]["life_stats"]["life"].item()),
                "oxygen": float(obs[1]["life_stats"]["oxygen"].item()),
                "food": float(obs[1]["life_stats"]["food"].item()),
            },
            "location_stats2": copy.deepcopy(self._pos2),
            "biomeid2": float(obs[1]["location_stats"]["biome_id"].item()),
        }

        obs0 = OrderedDict(sorted(self._convert_obs(obs[0]).items()))
        obs1 = OrderedDict(sorted(self._convert_obs(obs[1]).items()))
        obs = {self.curr_agents[0]: obs0, self.curr_agents[1]: obs1}
        return obs, info


    def step(self, action_dict):
        global iteration
        iteration = iteration + 1
        if "ppo_0" not in action_dict: #Debugging purposes
            print("action_dict")
            print(action_dict)
            raise Exception
            self.base_env.reset()
            action1 = self.base_env.action_space.no_op()
            action2 = self.base_env.action_space.no_op()
            action1[0] = 1
            action2[0] = 1
        else: 
            action1 = self._convert_action(action_dict[self.curr_agents[0]], 1)
            action2 = self._convert_action(action_dict[self.curr_agents[1]], 2)
            next_pitch1 = self._pos1["pitch"] + (action1[3] - 12) * 15
            next_pitch2 = self._pos2["pitch"] + (action2[3] - 12) * 15

            if not (self._pitch_limits[0] <= next_pitch1 <= self._pitch_limits[1]):
                action1[3] = 12
    
            if not (self._pitch_limits[0] <= next_pitch2 <= self._pitch_limits[1]):
                action2[3] = 12

        actions = [
            action1,
            action2,
         ]
        obs, reward, done, info = self.base_env.step(actions)

        self._pos1 = {
            "x": float(obs[0]["location_stats"]["pos"][0]),
            "y": float(obs[0]["location_stats"]["pos"][1]),
            "z": float(obs[0]["location_stats"]["pos"][2]),
            "pitch": float(obs[0]["location_stats"]["pitch"].item()),
            "yaw": float(obs[0]["location_stats"]["yaw"].item()),
        }

        self._pos2 = {
            "x": float(obs[1]["location_stats"]["pos"][0]),
            "y": float(obs[1]["location_stats"]["pos"][1]),
            "z": float(obs[1]["location_stats"]["pos"][2]),
            "pitch": float(obs[1]["location_stats"]["pitch"].item()),
            "yaw": float(obs[1]["location_stats"]["yaw"].item()),
        }

        obs0 = OrderedDict(sorted(self._convert_obs(obs[0]).items()))
        obs1 = OrderedDict(sorted(self._convert_obs(obs[1]).items()))
        obs = {self.curr_agents[0]: obs0, self.curr_agents[1]: obs1}
        #if (iteration == 0):
        #  done = True
        rewards = {self.curr_agents[0]: reward[0], self.curr_agents[1]: reward[1]}
        dones = {self.curr_agents[0]: done, self.curr_agents[1]: done, "__all__": done}  ## Cambiar done para cada agente
        terminated = {self.curr_agents[0]: False, self.curr_agents[1]: False}
        terminated["__all__"] = False
        if (done == True):
            print("done in multiagent")
            print(iteration)
        infos = {self.curr_agents[0]: info[0], self.curr_agents[1]: info[1]}

        return obs, rewards, dones, terminated, infos

    def _populate_agents(self):
        agents = ["ppo_0", "ppo_1"]
        self._inventory = {}
        self._inventory_names = None
        self._inventory_max = np.zeros(N_ALL_ITEMS)
        self.act_space = gym.spaces.MultiDiscrete(
            np.array([len(ACTION_MAP.keys()), len(ALL_CRAFT_SMELT_ITEMS), N_ALL_ITEMS])
        )
        global act_space
        act_space = self.act_space
        self.observation = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, self.base_env.observation_space["agent_0"]["rgb"].shape, np.uint8),
                "inventory": gym.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float64),
                "inventory_max": gym.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float64),
                "inventory_delta": gym.spaces.Box(-np.inf, np.inf, (N_ALL_ITEMS,), np.float64),
                "equipment": gym.spaces.Box(0.0, 1.0, (N_ALL_ITEMS,), np.int32),
                "life_stats": gym.spaces.Box(-20.0, np.array([20.0, 20.0, 300.0]), (3,), np.float32),
                "damage_received": gym.spaces.Box(0, 40.0, (1,), np.float32),
                "mask_action_type": gym.spaces.Box(0, 1, (len(ACTION_MAP),), bool),
                "mask_equip_place": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), bool),
                "mask_destroy": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), bool),
                "mask_craft_smelt": gym.spaces.Box(0, 1, (len(ALL_CRAFT_SMELT_ITEMS),), bool),
            }
        )
        global obs_space
        obs_space = self.observation

        ob_space = {}
        action_space = {}
        for agent in agents:
            ob_space[agent] = self.observation
            action_space[agent] = self.act_space

        self.observation_space = gym.spaces.Dict(ob_space)
        self.action_space = gym.spaces.Dict(action_space)
        return agents



    @classmethod
    def from_config(cls):
      #base_env = minedojo.make(task_id="harvest_milk", image_size=(288,512), training = True)
      #base_env = minedojo.make(task_id="combat_spider_plains_leather_armors_diamond_sword_shield", image_size=(288,512), training = True)
      base_env = minedojo.make(task_id="harvest_wool_with_shears_and_sheep", image_size=(288,512), training = True)
      #base_env = minedojo.make(task_id="harvest_log_with_diamond_axe", image_size=(288,512), training = True)
      return cls(base_env)

    def _convert_inventory(self, inventory: Dict[str, Any]) -> np.ndarray:
      converted_inventory = np.zeros(N_ALL_ITEMS)
      self._inventory = {}  # map for each item the position in the inventory
      self._inventory_names = np.array(
          ["_".join(item.split(" ")) for item in inventory["name"].copy().tolist()]
      )  # names of the objects in the inventory
      for i, (item, quantity) in enumerate(zip(inventory["name"], inventory["quantity"])):
          item = "_".join(item.split(" "))
          # save all the position of the items in the inventory
          if item not in self._inventory:
              self._inventory[item] = [i]
          else:
              self._inventory[item].append(i)
          # count the items in the inventory
          if item == "air":
              converted_inventory[ITEM_NAME_TO_ID[item]] += 1
          else:
              converted_inventory[ITEM_NAME_TO_ID[item]] += quantity
      self._inventory_max = np.maximum(converted_inventory, self._inventory_max)
      return converted_inventory

    def _convert_inventory_delta(self, inventory_delta: Dict[str, Any]) -> np.ndarray:
        # the inventory counts, as a vector with one entry for each Minecraft item
        converted_inventory_delta = np.zeros(N_ALL_ITEMS)
        for item, quantity in zip(inventory_delta["inc_name_by_craft"], inventory_delta["inc_quantity_by_craft"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] += quantity
        for item, quantity in zip(inventory_delta["dec_name_by_craft"], inventory_delta["dec_quantity_by_craft"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] -= quantity
        for item, quantity in zip(inventory_delta["inc_name_by_other"], inventory_delta["inc_quantity_by_other"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] += quantity
        for item, quantity in zip(inventory_delta["dec_name_by_other"], inventory_delta["dec_quantity_by_other"]):
            item = "_".join(item.split(" "))
            converted_inventory_delta[ITEM_NAME_TO_ID[item]] -= quantity
        return converted_inventory_delta

    def _convert_equipment(self, equipment: Dict[str, Any]) -> np.ndarray:
        equip = np.zeros(N_ALL_ITEMS, dtype=np.int32)
        equip[ITEM_NAME_TO_ID["_".join(equipment["name"][0].split(" "))]] = 1
        return equip

    def _convert_masks(self, masks: Dict[str, Any]) -> Dict[str, np.ndarray]:
        equip_mask = np.array([False] * N_ALL_ITEMS)
        destroy_mask = np.array([False] * N_ALL_ITEMS)
        for item, eqp_mask, dst_mask in zip(self._inventory_names, masks["equip"], masks["destroy"]):
            idx = ITEM_NAME_TO_ID[item]
            equip_mask[idx] = eqp_mask
            destroy_mask[idx] = dst_mask
        masks["action_type"][5:7] *= np.any(equip_mask).item()
        masks["action_type"][7] *= np.any(destroy_mask).item()
        return {
            "mask_action_type": np.concatenate((np.array([True] * 12), masks["action_type"][1:])),
            "mask_equip_place": equip_mask,
            "mask_destroy": destroy_mask,
            "mask_craft_smelt": masks["craft_smelt"],
        }

    def _convert_action(self, action: np.ndarray, agent_num) -> np.ndarray:
        action_int = int(action[0])
        if action_int > 12:
            action_int = random.randint(0,12)
        converted_action = ACTION_MAP[action_int].copy()
        if self._sticky_attack:
            if agent_num == 1:
                if converted_action[5] == 3:
                    self._sticky_attack_counter1 = self._sticky_attack - 1
                if self._sticky_attack_counter1 > 0:
                    converted_action[5] = 3
                    self._sticky_attack_counter1 -= 1
                #elif converted_action[5] != 3:
                #    self._sticky_attack_counter1 = 0 
            else:
                if converted_action[5] == 3:
                    self._sticky_attack_counter2 = self._sticky_attack - 1
                if self._sticky_attack_counter2 > 0:
                    converted_action[5] = 3
                    self._sticky_attack_counter2 -= 1
                #else:
                #    self._sticky_attack_counter2 = 0 
        # if the agent selects the craft action (value 4 in index 5 of the converted actions),
        # then it also selects the element to craft
        converted_action[6] = int(action[1]) if converted_action[5] == 4 else 0
        # if the agent selects the equip/place/destroy action (value 5 or 6 or 7 in index 5 of the converted actions),
        # then it also selects the element to equip/place/destroy
        if converted_action[5] in {5, 6, 7}:
            if ITEM_ID_TO_NAME[int(action[2])] in self._inventory:
              converted_action[7] = self._inventory[ITEM_ID_TO_NAME[int(action[2])]][0]
            else:
              converted_action[7] = 0
        else:
            converted_action[7] = 0
        return converted_action

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {
            "rgb": obs["rgb"].copy(),
            "inventory": self._convert_inventory(obs["inventory"]),
            "inventory_max": self._inventory_max,
            "inventory_delta": self._convert_inventory_delta(obs["delta_inv"]),
            "equipment": self._convert_equipment(obs["equipment"]),
            "life_stats": np.concatenate(
                (obs["life_stats"]["life"], obs["life_stats"]["food"], obs["life_stats"]["oxygen"])
            ),
            "damage_received": obs["damage_source"]["damage_amount"].copy(),
            **self._convert_masks(obs["masks"]),
        }

def gen_trainer_from_params(params):
    if not ray.is_initialized():
        _system_config = {"local_fs_capacity_threshold": 0.99}
        init_params = {
            "ignore_reinit_error": True,
            "num_gpus": 1,
            "_temp_dir": params["ray_params"]["temp_dir"],
            "log_to_driver": params["verbose"],
            "logging_level": logging.INFO
            if params["verbose"]
            else logging.CRITICAL,
            "_system_config" : _system_config
        }
        context = ray.init(**init_params)
        register_env("MineDojo_Env", params["ray_params"]["env_creator"])
        ModelCatalog.register_custom_model(
            params["ray_params"]["custom_model_id"],
            params["ray_params"]["custom_model_cls"],
        )



        training_params = params["training_params"]
        model_params = params["model_params"]
        logdir_prefix = "{0}_{1}_{2}".format(
        params["experiment_name"], params["training_params"]["seed"], timestr
        )

        global obs_space
        global act_space

        def gen_policy():
        # supported policy types thus far
            config = {
                "model": {
                    "custom_model_config": model_params,
                    "custom_model": "MyPPOModel",
                }
            }
            return (
                None,
                obs_space,
                act_space,
                config,
            )

        def select_policy(agent_id, episode, worker, **kwargs):
          return "ppo"


        multi_agent_config = {}
        all_policies = ["ppo"]

        multi_agent_config["policies"] = {
            policy: gen_policy() for policy in all_policies
        }

        multi_agent_config["policy_mapping_fn"] = select_policy
        multi_agent_config["policies_to_train"] = {"ppo"}

        config = PPOConfig()
        config = config.resources(num_gpus=0)
        config = config.rollouts(num_rollout_workers=1, rollout_fragment_length = training_params["rollout_fragment_length"])
        config = config.framework(framework = "tf")
        config = config.training(model = {"vf_share_layers" : training_params["vf_share_layers"]})
        config = config.training(lr_schedule=training_params["lr_schedule"],
                                 use_gae=True,
                                 lambda_=training_params["lambda"],
                                 use_kl_loss=True,
                                 kl_coeff=training_params["kl_coeff"],
                                 num_sgd_iter=training_params["num_sgd_iter"],
                                 vf_loss_coeff=training_params["vf_loss_coeff"],
                                 clip_param=training_params["clip_param"],
                                 grad_clip=training_params["grad_clip"],
                                 train_batch_size = training_params["train_batch_size"],
                                 sgd_minibatch_size = 12,
                                 entropy_coeff = 0.2,
                                 vf_share_layers = True
                                )
        config = config.multi_agent(policies = multi_agent_config["policies"], policy_mapping_fn = multi_agent_config["policy_mapping_fn"], policies_to_train =  multi_agent_config["policies_to_train"] )
        algo = config.build(env="MineDojo_Env")
        return algo

def save_trainer(trainer, params, path=None):
    """
    Saves a serialized trainer checkpoint at `path`. If none provided, the default path is
    ~/ray_results/<experiment_results_dir>/checkpoint_<i>
    """
    # Save trainer
    save_result = trainer.save(path)
    path_to_checkpoint = save_result.checkpoint.path
    print(path_to_checkpoint)

    # Save params used to create trainer in /path/to/checkpoint_dir/config.pkl
    config = copy.deepcopy(params)
    config_path = os.path.join(os.path.dirname(path_to_checkpoint), "config.pkl")

    # Note that we use dill (not pickle) here because it supports function serialization
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return path_to_checkpoint 

def load_trainer(save_path, true_num_workers=False):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint directory
    Additionally we decide if we want to use the same number of remote workers (see ray library Training APIs)
    as we store in the previous configuration, by default = False, we use only the local worker
    (see ray library API)
    """
    # Read in params used to create trainer
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    if not true_num_workers:
        # Override this param to lower overhead in trainer creation
        config["training_params"]["num_workers"] = 0

    if config["training_params"]["num_gpus"] == 1:
        # all other configs for the server can be kept for local testing
        config["training_params"]["num_gpus"] = 0

    # Get un-trained trainer object with proper config
    trainer = gen_trainer_from_params(config)
    # Load weights into dummy object
    trainer.restore(save_path)
    return trainer
