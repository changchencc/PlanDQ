# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
import pdb
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


class Data_Sampler(object):
    def __init__(
        self,
        data,
        device,
        reward_tune="no",
        K=4,
        p=0.1,
        normalize=False,
    ):
        self.state = torch.from_numpy(data["observations"][:-1]).float().to(device)
        self.next_state = torch.from_numpy(data["observations"][1:]).float().to(device)
        self.action = torch.from_numpy(data["actions"][:-1]).float().to(device)
        reward = torch.from_numpy(data["rewards"][:-1]).view(-1, 1).float().to(device)
        done = data["terminals"][:-1] | data["timeouts"][:-1]
        self.not_done = 1.0 - torch.from_numpy(done).view(-1, 1).float().to(device)

        if normalize:
            min_ret, max_ret = return_range(data, 1000)
            reward /= max_ret - min_ret
            reward *= 1000
            # state_mean, state_std = compute_mean_std(self.state, eps=1e-3)
        self.K = K
        self.p = p
        not_done_seq = torch.stack(
            [
                self.not_done[i : i + self.K + 1]
                for i in range(self.not_done.shape[0] - self.K - 1)
            ],
            dim=0,
        ).squeeze(-1)
        self.not_done_seq = torch.where(
            torch.cumprod(not_done_seq, dim=-1)[:, -2] == 1
        )[0]
        self.reward_tune = reward_tune

        self.size = self.not_done_seq.shape[0]
        self.state_dim = self.state.shape[1]
        self.state_dim = self.state_dim * 2
        self.action_dim = self.action.shape[1]

        self.device = device

        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        self.reward = reward.to(device)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        ind = self.not_done_seq[ind]

        min_window_size = 0
        max_window_size = np.random.default_rng().geometric(p=self.p) * self.K
        goal_idx = torch.randint(
            min_window_size, max_window_size, size=(batch_size,), device=self.device
        )
        goal_idx = torch.minimum(
            goal_idx.new_ones(goal_idx.shape) * self.size, ind + goal_idx
        )

        goal = self.state[goal_idx]
        feat = self.state[ind + 1]

        gnorm = torch.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
        fnorm = torch.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
        norm = torch.maximum(gnorm, fnorm)
        r_g = torch.einsum("...i,...i->...", goal / norm, feat / norm).unsqueeze(-1)

        reward = self.reward[ind] + r_g

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            reward,
            self.not_done[ind],
            goal,
        )


def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(
        torch.tensor(trajs_rt)
    )
    reward /= rt_max - rt_min
    reward *= 1000.0
    return reward


class Normalizer:
    """
    parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class LimitsNormalizer(Normalizer):
    """
    maps [ xmin, xmax ] to [ -1, 1 ]
    """

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        """
        x : [ -1, 1 ]
        """
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.0

        return x * (self.maxs - self.mins) + self.mins
