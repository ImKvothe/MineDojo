from functools import partial
from mypy_extensions import Arg
from typing import Union, Callable, Dict


__all__ = [
    "reward_fn_base",
    "simple_inventory_based_reward",
    "simple_stat_kill_entity_based_reward",
    "possess_item_reward",
    "survive_per_day_reward",
    "survive_n_days_reward",
    "use_any_item_reward",
]


# takes an initial info dict (t = 0), a pre info dict (t - 1), a current info dict (t), and elapsed time-steps,
# return a scalar reward value
reward_fn_base = Callable[
    [
        Arg(list, "ini_info_dict"),
        Arg(list, "pre_info_dict"),
        Arg(list, "cur_info_dict"),
        Arg(int, "elapsed_timesteps"),
    ],
    float,
]


def _simple_stat_kill_entity_based_reward(
    name: str,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["stat"]["kill_entity"][{name}]`.
    """
    rew1 = weight * (
        cur_info_dict[0]["stat"]["kill_entity"][name]
        - pre_info_dict[0]["stat"]["kill_entity"][name]
    )
    rew2 = weight * (
        cur_info_dict[1]["stat"]["kill_entity"][name]
        - pre_info_dict[1]["stat"]["kill_entity"][name]
    )
    return [rew1, rew2]


def simple_stat_kill_entity_based_reward(
    name: str, weight: Union[int, float], **kwargs
) -> reward_fn_base:
    return partial(_simple_stat_kill_entity_based_reward, name=name, weight=weight)


def _simple_inventory_based_reward(
    name: str,
    weight: Union[int, float],
    ini_info_dict,
    pre_info_dict,
    cur_info_dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["inventory"]`
    """
    #print(type(cur_info_dict))
    rew1 = (
        sum(
            [
                inv_item["quantity"]
                for inv_item in cur_info_dict[0]["inventory"]
                if inv_item["name"] == name
            ]
        )
        - sum(
            [
                inv_item["quantity"]
                for inv_item in pre_info_dict[0]["inventory"]
                if inv_item["name"] == name
            ]
        )
    ) * weight
    
    rew2 = (
        sum(
            [
                inv_item["quantity"]
                for inv_item in cur_info_dict[1]["inventory"]
                if inv_item["name"] == name
            ]
        )
        - sum(
            [
                inv_item["quantity"]
                for inv_item in pre_info_dict[1]["inventory"]
                if inv_item["name"] == name
            ]
        )
    ) * weight
    return [rew1, rew2]


def simple_inventory_based_reward(
    name: str, weight: Union[int, float], **kwargs
) -> reward_fn_base:
    return partial(_simple_inventory_based_reward, name=name, weight=weight)


def _possess_item_reward(
    name: str,
    weight: Union[int, float],
    quantity: int,
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    rew1 = (
        float(
            (
                sum(
                    [
                        inv_item["quantity"]
                        for inv_item in cur_info_dict[0]["inventory"]
                        if inv_item["name"] == name
                    ]
                )
                - sum(
                    [
                        inv_item["quantity"]
                        for inv_item in ini_info_dict[0]["inventory"]
                        if inv_item["name"] == name
                    ]
                )
            )
            >= quantity
        )
        * weight
    )

    rew2 = (
        float(
            (
                sum(
                    [
                        inv_item["quantity"]
                        for inv_item in cur_info_dict[1]["inventory"]
                        if inv_item["name"] == name
                    ]
                )
                - sum(
                    [
                        inv_item["quantity"]
                        for inv_item in ini_info_dict[1]["inventory"]
                        if inv_item["name"] == name
                    ]
                )
            )
            >= quantity
        )
        * weight
    )
    return [rew1, rew2]


def possess_item_reward(
    name: str, quantity: int, weight: Union[int, float]
) -> reward_fn_base:
    return partial(_possess_item_reward, name=name, quantity=quantity, weight=weight)


def _survive_per_day_reward(
    mc_ticks_per_day: int,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    time_since_death_pre1 = pre_info_dict[0]["stat"]["time_since_death"]
    time_since_death_pre2 = pre_info_dict[1]["stat"]["time_since_death"]
    time_since_death_cur1 = cur_info_dict[0]["stat"]["time_since_death"]
    time_since_death_cur2 = cur_info_dict[1]["stat"]["time_since_death"]
    survived_days_pre1 = time_since_death_pre1 // mc_ticks_per_day
    survived_days_pre2 = time_since_death_pre2 // mc_ticks_per_day
    survived_days_cur1 = time_since_death_cur1 // mc_ticks_per_day
    survived_days_cur2 = time_since_death_cur2 // mc_ticks_per_day
    rew1 = (survived_days_cur1 - survived_days_pre1) * weight
    rew2 = (survived_days_cur2 - survived_days_pre2) * weight 
    return [rew1, rew2]


def survive_per_day_reward(
    mc_ticks_per_day: int, weight: Union[int, float]
) -> reward_fn_base:
    return partial(
        _survive_per_day_reward, mc_ticks_per_day=mc_ticks_per_day, weight=weight
    )


def _survive_n_days_reward(
    mc_ticks_per_day: int,
    target_days: int,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    rew1 = weight * float(
        cur_info_dict[0]["stat"]["time_since_death"] >= mc_ticks_per_day * target_days
    )

    rew2 = weight * float(
        cur_info_dict[1]["stat"]["time_since_death"] >= mc_ticks_per_day * target_days
    )

      
    return [rew1, rew2]


def survive_n_days_reward(
    mc_ticks_per_day: int, target_days: int, weight: Union[int, float]
) -> reward_fn_base:
    return partial(
        _survive_n_days_reward,
        mc_ticks_per_day=mc_ticks_per_day,
        target_days=target_days,
        weight=weight,
    )


def _use_any_item_reward(
    items_and_weights: Dict[str, Union[int, float]],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    reward any usage of item in items_and_weights.keys()
    """
    rew1 = sum(
        [
            (
                cur_info_dict[0]["stat"]["use_item"]["minecraft"][item]
                - pre_info_dict[0]["stat"]["use_item"]["minecraft"][item]
            )
            * weight
            for item, weight in items_and_weights.items()
        ]
    )

    rew2 = sum(
        [
            (
                cur_info_dict[1]["stat"]["use_item"]["minecraft"][item]
                - pre_info_dict[1]["stat"]["use_item"]["minecraft"][item]
            )
            * weight
            for item, weight in items_and_weights.items()
        ]
    )


    return [rew1, rew2]


def use_any_item_reward(
    items_and_weights: Dict[str, Union[int, float]]
) -> reward_fn_base:
    return partial(_use_any_item_reward, items_and_weights=items_and_weights)
