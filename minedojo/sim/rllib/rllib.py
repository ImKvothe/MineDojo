import minedojo
import logging
from ray.tune.result import DEFAULT_RESULTS_DIR
import ray
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
from minedojo.sim.wrappers.ar_nn import ARNNWrapper

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
    12: np.array([0, 0, 0, 12, 12, 1, 0, 0]),  # use
    13: np.array([0, 0, 0, 12, 12, 2, 0, 0]),  # drop
    14: np.array([0, 0, 0, 12, 12, 3, 0, 0]),  # attack
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
        pitch_limits: Tuple[int, int] = (-60, 60),
        seed: Optional[int] = None,
        **kwargs: Optional[Dict[Any, Any]],):

        self._height = height
        self._width = width
        self._pitch_limits = pitch_limits
        self._pos1 = kwargs.get("start_position1", None)
        self._pos2 = kwargs.get("start_position2", None)

        self.base_env = base_env
        self._episode_id = None
        super().__init__()
        print("hello2")
        self._agent_ids = set(self.reset()[0].keys())

    def step(self, action_dict):
        print(self._inventory)
        print(action_dict)
        print(self.curr_agents)
        a1 = action_dict[self.curr_agents[0]]
        a2 = action_dict[self.curr_agents[1]]
        action1 = self._convert_action(action_dict[self.curr_agents[0]])
        action2 = self._convert_action(action_dict[self.curr_agents[1]])
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
        print(actions)

        rewards = {self.curr_agents[0]: reward[0], self.curr_agents[1]: reward[1]}
        dones = {self.curr_agents[0]: done, self.curr_agents[1]: done}  ## Cambiar done para cada agente
        terminated = dones
        infos = {self.curr_agents[0]: info[0], self.curr_agents[1]: info[1]}

        return obs, rewards, dones, terminated, infos

    def reset(self, *, seed=None, options=None):
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

    def _populate_agents(self):
        agents = ["ppo_0", "ppo_1"]
        self._inventory = {}
        self._inventory_names = None
        self._inventory_max = np.zeros(N_ALL_ITEMS)
        act_space = gym.spaces.MultiDiscrete(
            np.array([len(ACTION_MAP.keys()), len(ALL_CRAFT_SMELT_ITEMS), N_ALL_ITEMS])
        )
        observation = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, self.base_env.observation_space["agent_0"]["rgb"].shape, np.uint8),
                "inventory": gym.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float64),
                "inventory_max": gym.spaces.Box(0.0, np.inf, (N_ALL_ITEMS,), np.float64),
                "inventory_delta": gym.spaces.Box(-np.inf, np.inf, (N_ALL_ITEMS,), np.float64),
                "equipment": gym.spaces.Box(0.0, 1.0, (N_ALL_ITEMS,), np.int32),
                "life_stats": gym.spaces.Box(0.0, np.array([20.0, 20.0, 300.0]), (3,), np.float32),
                "mask_action_type": gym.spaces.Box(0, 1, (len(ACTION_MAP),), bool),
                "mask_equip_place": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), bool),
                "mask_destroy": gym.spaces.Box(0, 1, (N_ALL_ITEMS,), bool),
                "mask_craft_smelt": gym.spaces.Box(0, 1, (len(ALL_CRAFT_SMELT_ITEMS),), bool),
            }
        )
        ob_space = {}
        action_space = {}
        for agent in agents:
            ob_space[agent] = observation
            action_space[agent] = act_space

        self.observation_space = gym.spaces.Dict(ob_space)
        self.action_space = gym.spaces.Dict(action_space)
        return agents



    @classmethod
    def from_config(cls):
      base_env = minedojo.make(task_id="harvest_milk", image_size=(288,512))
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

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        converted_action = ACTION_MAP[int(action[0])].copy()
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
            **self._convert_masks(obs["masks"]),
        }
