# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
import pdb
from pathlib import Path
import json
import re
import os
import cv2

from torchvision.transforms.functional import affine
from torchvision.transforms import ColorJitter
from torch.utils.data import IterableDataset


class Data_Sampler(object):
    def __init__(
        self,
        data,
        device,
        reward_tune="no",
        K=4,
        p=0.25,
    ):
        self.state = torch.from_numpy(data["observations"][:-1]).float().to(device)
        self.action = torch.from_numpy(data["actions"]).float().to(device)
        self.next_state = torch.from_numpy(data["observations"][1:]).float().to(device)
        reward = torch.from_numpy(data["rewards"]).view(-1, 1).float().to(device)
        if "timeouts" not in data:
            self.not_done = 1.0 - torch.from_numpy(data["terminals"]).view(
                -1, 1
            ).float().to(device)
        else:
            self.not_done = 1.0 - torch.from_numpy(data["timeouts"]).view(
                -1, 1
            ).float().to(device)

        self.K = K
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
        self.p = p

        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (reward - 0.25) * 2.0
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

        s_feat = self.state[ind + 1][:, :2]
        g_feat = goal[:, :2]
        r_g = (-torch.linalg.norm(g_feat - s_feat, dim=-1, keepdims=True)).exp()
        not_done = (r_g != 1.0).float() * self.not_done[ind]

        reward = self.reward[ind] + r_g

        actions = self.action[ind]
        return (
            self.state[ind],
            actions,
            self.next_state[ind],
            reward,
            not_done,
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


class Transform:
    def __init__(self, max_translate=6, brightness=0.1, contrast=0.1, hue=0.02):
        self.color_jitter = ColorJitter(
            brightness=brightness, contrast=contrast, hue=hue
        )
        self.max_translate = max_translate

    def __call__(self, image, eval=False):
        if not eval:
            translate = list(
                np.random.randint(-self.max_translate, self.max_translate + 1, size=2)
            )
            image = affine(image, angle=0, translate=translate, scale=1, shear=[0])
            image = self.color_jitter(image)
        image = 2.0 * image / 255.0 - 1.0
        return image


def arctanh_actions(actions, *args, **kwargs):
    epsilon = 1e-4
    assert (
        actions.min() >= -1 and actions.max() <= 1
    ), f"applying arctanh to actions in range [{actions.min()}, {actions.max()}]"
    actions = torch.clamp(actions, -1 + epsilon, 1 - epsilon)
    actions = torch.arctanh(actions)
    return actions
