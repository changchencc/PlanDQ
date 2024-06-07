# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
import math

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA

import pdb


"""
Ensemble critic code is borrowed from:
https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/edac.py
"""


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorizedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_critics: int,
        hidden_dim=256,
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values.permute(1, 0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount,
        tau,
        num_critics=64,
        max_q_backup=False,
        eta=1.0,
        beta_schedule="linear",
        n_timesteps=100,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        lr_decay=False,
        lr_maxt=1000,
        grad_norm=1.0,
        goal_dim=0,
        lcb_coef=4.0,
    ):

        self.goal_dim = goal_dim
        if goal_dim:
            state_dim = state_dim // 2 + goal_dim
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.model,
            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
            predict_epsilon=False,
            clip_denoised=False,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = VectorizedCritic(state_dim, action_dim, num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.0
            )

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup
        self.lcb_coef = lcb_coef
        self.goal_dim = goal_dim

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {
            "bc_loss": [],
            "ql_loss": [],
            "ql_std": [],
            "ql_random_std": [],
            "actor_loss": [],
            "critic_loss": [],
        }
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done, goal = replay_buffer.sample(
                batch_size
            )
            if self.goal_dim:
                goal = goal[:, : self.goal_dim]
            state = torch.cat([state, goal], dim=-1)
            next_state = torch.cat([next_state, goal], dim=-1)

            """ Q Training """
            current_q = self.critic(state, action)
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q = self.critic_target(next_state_rpt, next_action_rpt)
                target_q = target_q.view(batch_size, 10, -1).max(dim=1)[0]
            else:
                next_action = self.ema_model(next_state)
                target_q = self.critic_target(next_state, next_action)
            target_q = (reward + not_done * self.discount * target_q).detach()
            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            with torch.no_grad():
                next_action = self.ema_model(state)
            q_values_new_action_ensembles = self.critic(state, next_action)
            mu = q_values_new_action_ensembles.mean(dim=1, keepdim=True)
            std = q_values_new_action_ensembles.std(dim=1, keepdim=True)
            q_values_new_action = mu - self.lcb_coef * std
            q_loss = (
                -q_values_new_action.mean()
                / q_values_new_action_ensembles.abs().mean().detach()
            )

            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            with torch.no_grad():
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                max_action = self.max_action
                random_actions = -max_action + 2 * max_action * torch.rand_like(action)

                q_random_std = self.critic(state, random_actions).std(0).mean().item()

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar(
                        "Actor Grad Norm", actor_grad_norms.max().item(), self.step
                    )
                    log_writer.add_scalar(
                        "Critic Grad Norm", critic_grad_norms.max().item(), self.step
                    )
                log_writer.add_scalar("BC Loss", bc_loss.item(), self.step)
                log_writer.add_scalar("QL Loss", q_loss.item(), self.step)
                log_writer.add_scalar("Critic Loss", critic_loss.item(), self.step)
                # log_writer.add_scalar(
                #     "Diversity Loss", diversity_loss.item(), self.step
                # )
                log_writer.add_scalar(
                    "Target_Q Mean", target_q.mean().item(), self.step
                )

            metric["actor_loss"].append(actor_loss.item())
            metric["bc_loss"].append(bc_loss.item())
            metric["ql_loss"].append(q_loss.item())
            metric["critic_loss"].append(critic_loss.item())
            # metric["diversity_loss"].append(diversity_loss.item())
            metric["ql_std"].append(std.mean().item())
            metric["ql_random_std"].append(q_random_std)

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            actor_lr = self.actor_lr_scheduler.get_last_lr()[0]
            critic_lr = self.critic_lr_scheduler.get_last_lr()[0]
            metric.update(
                {
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                }
            )

        return metric

    def sample_action(self, state, goal, num_sample=50):
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(self.device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = torch.cat([state, goal], dim=-1)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():

            action = self.actor.sample(state_rpt)
            q_value = self.critic_target(state_rpt, action)
            q_mean = q_value.mean(dim=1, keepdim=True).flatten()
            q_std = q_value.std(dim=1, keepdim=True).flatten()
            q_value = q_mean - self.lcb_coef * q_std
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{id}.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic_{id}.pth")
            torch.save(self.critic_target.state_dict(), f"{dir}/critic_target_{id}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic.pth")
            torch.save(self.critic_target.state_dict(), f"{dir}/critic_target.pth")

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{id}.pth"))
            self.critic.load_state_dict(torch.load(f"{dir}/critic_{id}.pth"))
            self.critic_target.load_state_dict(
                torch.load(f"{dir}/critic_target_{id}.pth")
            )
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pth"))
            self.critic_target.load_state_dict(torch.load(f"{dir}/critic_target.pth"))
