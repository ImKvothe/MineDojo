

import math
from typing import Union, Sequence

import gym
import numpy as np

from ...sim import MineDojoSim
from ....sim import spaces as spaces
from ....sim.mc_meta import mc as MC
from ....sim.inventory import InventoryItem


class NNActionSpaceWrapper(gym.Wrapper):
    """
    Action wrapper to transform native action space to a new space friendly to train NNs
    """

    def __init__(
        self,
        env: Union[MineDojoSim, gym.Wrapper],
        discretized_camera_interval: Union[int, float] = 15,
        strict_check: bool = True,
    ):
        agents_act = env.action_space
        #print(agents_act)
        for agent in agents_act:
            assert (
                "equip" in env.action_space[agent].keys()
                and "place" in env.action_space[agent].keys()
                and "swap_slot" not in env.action_space[agent].keys()
            ), "please use this wrapper with event_level_control = True"
            assert (
                "inventory" in env.observation_space[agent].keys()
            ), f"missing inventory from obs space"
        super().__init__(env=env)

        n_pitch_bins = math.ceil(360 / discretized_camera_interval) + 1
        n_yaw_bins = math.ceil(360 / discretized_camera_interval) + 1

        self.action_space = spaces.MultiDiscrete(
            [
                3,  # forward and back, 0: noop, 1: forward, 2: back
                3,  # 0: noop, 1: left, 2: right
                4,  # 0: noop, 1: jump, 2: sneak, 3: sprint
                n_pitch_bins,  # camera pitch, 0: -180, n_pitch_bins - 1: +180
                n_yaw_bins,  # camera yaw, 0: -180, n_yaw_bins - 1: +180,
                8,  # functional actions, 0: no_op, 1: use, 2: drop, 3: attack 4: craft 5: equip 6: place 7: destroy
                len(MC.ALL_CRAFT_SMELT_ITEMS),  # arg for "craft"
                MC.N_INV_SLOTS,  # arg for "equip", "place", and "destroy"
            ],
            noop_vec=[
                0,
                0,
                0,
                (n_pitch_bins - 1) // 2,
                (n_yaw_bins - 1) // 2,
                0,
                0,
                0,
            ],
        )
        self._cam_interval = discretized_camera_interval
        self._inventory_names = None
        self._strict_check = strict_check

    def action(self, actions):
        """
        NN action to Malmo action
        """
        assert self.action_space.contains(actions[0]) #CHECK
        destroy_item = (False, None)
        noop = self.env.action_space.no_op()
        items = list(noop.items())
        skiped_action = [False, False]
        i = 0
        while i < len(items):
            # ------ parse main actions ------
            # parse forward and back
            if actions[i][0] == 1:
                list(noop.items())[i][1]["forward"] = 1
            elif actions[i][0] == 2:
                list(noop.items())[i][1]["back"] = 1
            # parse left and right
            if actions[i][1] == 1:
                list(noop.items())[i][1]["left"] = 1
            elif actions[i][1] == 2:
                list(noop.items())[i][1]["right"] = 1
            # parse jump sneak and sprint
            if actions[i][2] == 1:
                list(noop.items())[i][1]["jump"] = 1
            elif actions[i][2] == 2:
                list(noop.items())[i][1]["sneak"] = 1
            elif actions[i][2] == 3:
                list(noop.items())[i][1]["sprint"] = 1
            # parse camera pitch
            list(noop.items())[i][1]["camera"][0] = float(actions[i][3]) * self._cam_interval + (-180)
            # parse camera yaw
            list(noop.items())[i][1]["camera"][1] = float(actions[i][4]) * self._cam_interval + (-180)

            # ------ parse functional actions ------
            fn_action = actions[i][5]
            # note that 0 is no_op
            if fn_action == 0:
                pass
            elif fn_action == 1:
                list(noop.items())[i][1]["use"] = 1
            elif fn_action == 2:
                list(noop.items())[i][1]["drop"] = 1
            elif fn_action == 3:
                list(noop.items())[i][1]["attack"] = 1
            elif fn_action == 4:
                item_to_craft = MC.ALL_CRAFT_SMELT_ITEMS[actions[i][6]]
                if item_to_craft in MC.ALL_HAND_CRAFT_ITEMS_NN_ACTIONS:
                    list(noop.items())[i][1]["craft"] = item_to_craft
                elif item_to_craft in MC.ALL_TABLE_CRAFT_ONLY_ITEMS_NN_ACTIONS:
                    list(noop.items())[i][1]["craft_with_table"] = item_to_craft
                elif item_to_craft in MC.ALL_SMELT_ITEMS_NN_ACTIONS:
                    list(noop.items())[i][1]["smelt"] = item_to_craft
                elif self._strict_check:
                    skiped_action[i] = True
                    pass
                    #raise ValueError(f"Unknown item {item_to_craft} to craft/smelt!")
            elif fn_action == 5:
                assert actions[i][7] in list(range(MC.N_INV_SLOTS))
                item_id = self._inventory_names[i][actions[i][7]].replace(" ", "_")
                if item_id == "air":
                    if self._strict_check:
                        skiped_action[i] = True
                        pass
                        #raise ValueError(
                        #    "Trying to equip air, raise error with strict check."
                        #    "You shouldn't execute this action, maybe something wrong with the mask!"
                        #)
                else:
                    list(noop.items())[i][1]["equip"] = item_id
            elif fn_action == 6:
                assert actions[i][7] in list(range(MC.N_INV_SLOTS))
                item_id = self._inventory_names[i][actions[i][7]].replace(" ", "_")
                if item_id == "air":
                    if self._strict_check:
                        skiped_action[i] = True
                        pass
                        #raise ValueError(
                        #    "Trying to equip air, raise error with strict check."
                        #    "You shouldn't execute this action, maybe something wrong with the mask!"
                        #)
                else:
                    list(noop.items())[i][1]["place"] = item_id
            elif fn_action == 7:
                assert actions[i][7] in list(range(MC.N_INV_SLOTS))
                item_id = self._inventory_names[i][actions[i][7]].replace(" ", "_")
                if item_id == "air":
                    if self._strict_check:
                        skiped_action[i] = True
                        pass
                        #raise ValueError(
                        #    "Trying to destroy air, raise error with strict check."
                        #    "You shouldn't execute this action, maybe something wrong with the mask!"
                        #)
                else:
                    destroy_item = (True, actions[i][7])
            else:
                return noop
                #raise ValueError(f"Unknown value {fn_action} for function action")
            i = i + 1
        if skiped_action[0] == True:
            list(noop.items())[0][1]["attack"] = 1
        if skiped_action[1] == True:
            list(noop.items())[1][1]["attack"] = 1   
        return noop, destroy_item

    def reverse_action(self, actions):
        """
        Malmo action to NN actions[i] || Ivan: Not used?
        """
        # first convert camera actions to [-pi, +pi]
        i = 0
        while i < len(actions):
            actions[i]["camera"] = (
                np.arctan2(
                    np.sin(actions[i]["camera"] * np.pi / 180),
                    np.cos(actions[i]["camera"] * np.pi / 180),
                )
                * 180
                / np.pi
            )
            assert self.env.action_space.contains(actions[0])

            noop = self.action_space.no_op()
            # ------ parse main actions ------
            # parse forward and back
            if actions[i]["forward"] == 1 and actions[i]["back"] == 1:
                # cancel each other, noop
                pass
            elif actions[i]["forward"] == 1:
                noop[0] = 1
            elif actions[i]["back"] == 1:
                noop[0] = 2
            # parse left and right
            if actions[i]["left"] == 1 and actions[i]["right"] == 1:
                # cancel each other, noop
                pass
            elif actions[i]["left"] == 1:
                noop[1] = 1
            elif actions[i]["right"] == 1:
                noop[1] = 2
            # parse jump, sneak, sprint
            # prioritize jump
            if actions[i]["jump"] == 1:
                noop[2] = 1
            else:
                if actions[i]["sneak"] == 1 and actions[i]["sprint"] == 1:
                    # cancel each other, noop
                    pass
                elif actions[i]["sneak"] == 1:
                    noop[2] = 2
                elif actions[i]["sprint"] == 1:
                    noop[2] = 3
            # parse camera pitch
            noop[3] = math.ceil((actions[i]["camera"][0] - (-180)) / self._cam_interval)
            # parse camera yaw
            noop[4] = math.ceil((actions[i]["camera"][1] - (-180)) / self._cam_interval)

            # ------ parse functional actions ------
            # order: attack > use > craft > equip > place > drop > destroy
            if actions[i]["attack"] == 1:
                noop[5] = 3
            elif actions[i]["use"] == 1:
                noop[5] = 1
            elif actions[i]["craft"] != "none" and actions[i]["craft"] != 0:
                craft = actions[i]["craft"]
                if isinstance(craft, int):
                    craft = MC.ALL_PERSONAL_CRAFTING_ITEMS[craft - 1]
                noop[5] = 4
                noop[6] = MC.ALL_CRAFT_SMELT_ITEMS.index(craft)
            elif actions[i]["craft_with_table"] != "none" and actions[i]["craft_with_table"] != 0:
                craft = actions[i]["craft_with_table"]
                if isinstance(craft, int):
                    craft = MC.ALL_CRAFTING_TABLE_ITEMS[craft - 1]
                noop[5] = 4
                noop[6] = MC.ALL_CRAFT_SMELT_ITEMS.index(craft)
            elif actions[i]["smelt"] != "none" and actions[i]["smelt"] != 0:
                smelt = actions[i]["smelt"]
                if isinstance(smelt, int):
                    smelt = MC.ALL_SMELTING_ITEMS[smelt - 1]
                noop[5] = 4
                noop[6] = MC.ALL_CRAFT_SMELT_ITEMS.index(smelt)
            elif actions[i]["equip"] != "none" and actions[i]["equip"] != 0:
                equip = actions[i]["equip"]
                if isinstance(equip, int):
                    equip = MC.ALL_ITEMS[equip - 1]
                equip = equip.replace("_", " ")
                if equip not in self._inventory_names[i]:
                    if self._strict_check:
                        raise ValueError(
                            f"try to equip {equip}, but it is not in the inventory {self._inventory_names[i]}"
                        )
                else:
                    slot_idx = np.where(self._inventory_names[i] == equip)[0][0]
                    noop[5] = 5
                    noop[7] = slot_idx
            elif actions[i]["place"] != "none" and actions[i]["place"] != 0:
                place = actions[i]["place"]
                if isinstance(place, int):
                    place = MC.ALL_ITEMS[place - 1]
                place = place.replace("_", " ")
                if place not in self._inventory_names[i]:
                    if self._strict_check:
                        raise ValueError(
                            f"try to place {place}, but it is not in the inventory {self._inventory_names[i]}"
                        )
                else:
                    slot_idx = np.where(self._inventory_names[i] == place)[0][0]
                    noop[5] = 6
                    noop[7] = slot_idx
            elif actions[i]["drop"] == 1:
                noop[5] = 2
        return noop

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        inventory1 = obs[0]["inventory"]["name"].copy()
        inventory2 = obs[1]["inventory"]["name"].copy()
        self._inventory_names = [inventory1, inventory2]
        return obs

    def step(self, actions):
        #print("entro al wrapper nnaction")
        #print(actions)
        malmo_action, destroy_item = self.action(actions)
        destroy_item, destroy_slot = destroy_item
        if destroy_item:
            obs, reward, done, info = self.env.set_inventory(
                inventory_list=[
                    InventoryItem(name="air", slot=destroy_slot, quantity=1, variant=0)
                ],
                action=malmo_action,
            )
        else:
            obs, reward, done, info = self.env.step(malmo_action)
        #print("vuelvo al wrapper nnaction")

        # handle malmo's lags for 2 agents
        if actions[0][5] in {2, 4, 5, 6, 7} or actions[1][5] in {2, 4, 5, 6, 7}:
            for _ in range(2):
                obs, reward, done, info = self.env.step(self.env.action_space.no_op())
        self._inventory_names[0] = obs[0]["inventory"]["name"].copy()
        self._inventory_names[1] = obs[1]["inventory"]["name"].copy()
        return obs, reward, done, info
