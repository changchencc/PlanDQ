# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gym
import numpy as np
import os
import torch
import json

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter

import diffuser.utils as hl_utils
from diffuser.guides.policies import Policy
from matplotlib import pyplot as plt
import pdb

hyperparameters = {
    "antmaze-medium-diverse-v2": {
        "lr": 3e-4,
        "eta": 1.0,
        "max_q_backup": False,
        "reward_tune": "cql_antmaze",
        "eval_freq": 100,
        "num_epochs": 2000,
        "gn": 7.0,
        "top_k": 1,
    },
    "antmaze-large-diverse-v2": {
        "lr": 3e-4,
        "eta": 1.0,
        "max_q_backup": False,
        "reward_tune": "cql_antmaze",
        "eval_freq": 100,
        "num_epochs": 2000,
        "gn": 7.0,
        "top_k": 1,
    },
    "antmaze-ultra-diverse-v0": {
        "lr": 3e-4,
        "eta": 1.0,
        "max_q_backup": False,
        "reward_tune": "ultra_antmaze",
        "eval_freq": 100,
        "num_epochs": 2000,
        "gn": 7.0,
        "top_k": 1,
    },
}


def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    # load hl_planner
    class HLParser(hl_utils.Parser):
        dataset: str = args.env_name
        config: str = "config.antmaze_hl"

    hlargs = HLParser().parse_args("plan")
    hl_planner = load_hl_planner(hlargs)

    # Load buffer
    dataset = env.get_dataset()
    data_sampler = Data_Sampler(
        dataset,
        device,
        args.reward_tune,
        K=hlargs.jump,
        p=args.p,
    )
    utils.print_banner("Loaded buffer")

    state_dim = data_sampler.state_dim
    action_dim = data_sampler.action_dim

    from agents.ql_diffusion import Diffusion_QL as Agent

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=args.discount,
        tau=args.tau,
        max_q_backup=args.max_q_backup,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.T,
        eta=args.eta,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_maxt=args.num_epochs,
        grad_norm=args.gn,
        goal_dim=args.goal_dim,
        lcb_coef=args.lcb_coef,
    )

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.0)
    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.0
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    i = 0
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(
            data_sampler,
            iterations=iterations,
            batch_size=args.batch_size,
            log_writer=writer,
        )
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular("Trained Epochs", curr_epoch)
        logger.record_tabular("BC Loss", np.mean(loss_metric["bc_loss"]))
        logger.record_tabular("QL Loss", np.mean(loss_metric["ql_loss"]))
        logger.record_tabular("Actor Loss", np.mean(loss_metric["actor_loss"]))
        logger.record_tabular("Critic Loss", np.mean(loss_metric["critic_loss"]))

        if "actor_lr" in loss_metric:
            logger.record_tabular("Actor LR", loss_metric["actor_lr"])
        if "critic_lr" in loss_metric:
            logger.record_tabular("Critic LR", loss_metric["critic_lr"])
        logger.dump_tabular()

        # Evaluation
        with torch.no_grad():
            (
                eval_res,
                eval_res_std,
                eval_norm_res,
                eval_norm_res_std,
            ) = eval_policy(
                args,
                hl_planner,
                agent,
                args.env_name,
                args.seed,
                eval_episodes=100,
                output_dir=output_dir,
                replan=args.replan,
            )
        evaluations.append(
            [
                eval_res,
                eval_res_std,
                eval_norm_res,
                eval_norm_res_std,
                np.mean(loss_metric["bc_loss"]),
                np.mean(loss_metric["ql_loss"]),
                np.mean(loss_metric["actor_loss"]),
                np.mean(loss_metric["critic_loss"]),
                curr_epoch,
            ]
        )
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular("Average Episodic Reward", eval_res)
        logger.record_tabular("Average Episodic N-Reward", eval_norm_res)
        logger.dump_tabular()

        bc_loss = np.mean(loss_metric["bc_loss"])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(
    args,
    hl_planner,
    policy,
    env_name,
    seed,
    eval_episodes=10,
    output_dir=None,
    replan=1,
):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    replan = replan

    scores = []
    if "ultra" in args.env_name:
        dist_thr = 13
    elif "large" in args.env_name:
        dist_thr = 6
    elif "medium" in args.env_name:
        dist_thr = 4

    for _ in range(eval_episodes):
        traj_return = 0.0
        state, done = eval_env.reset(), False
        target = eval_env.target_goal
        target = np.array(list(target) + [0.0] * 27)

        t = 0

        while not done:
            conditions = {0: state, hl_planner.diffusion_model.horizon - 1: target}
            tgt_dist = np.linalg.norm(target[:2] - state[:2])
            if tgt_dist > dist_thr:
                _, samples = hl_planner(conditions, batch_size=1, verbose=False)
                goal = samples.observations[0, 1]
                dist = np.linalg.norm(goal[:2] - state[:2])
                tgt_goal_dist = np.linalg.norm(target[:2] - goal[:2])
                n_try = 1
                while (n_try <= 5) and (
                    (dist > 7.0) or (dist <= 0.5) or (tgt_goal_dist > tgt_dist)
                ):
                    _, samples = hl_planner(conditions, batch_size=1, verbose=False)
                    goal = samples.observations[0, 1]
                    dist = np.linalg.norm(goal[:2] - state[:2])
                    tgt_goal_dist = np.linalg.norm(target[:2] - goal[:2])
                    n_try += 1
            else:
                goal[:2] = target[:2]
                dist = np.linalg.norm(goal[:2] - state[:2])

            # goal_idx = 1
            # while (dist < 0.5) and (goal_idx < samples.observations.shape[1] - 1):
            #     goal_idx += 1
            #     goal = samples.observations[0, goal_idx]
            #     dist = np.linalg.norm(goal[:2] - state[:2])

            j = 0
            while ((dist > 0.5) or (n_try > 5)) and (j < replan) and (not done):

                if args.goal_dim:
                    goal = goal[: args.goal_dim]
                action = policy.sample_action(state, goal)
                next_state, reward, done, _ = eval_env.step(action)
                state = next_state
                traj_return += reward
                dist = np.linalg.norm(goal[:2] - state[:2])
                j += 1
                t += 1
                pos = np.array2string(state[:2], precision=2, floatmode="fixed")
                subgoal_pos = np.array2string(goal[:2], precision=2, floatmode="fixed")
                print(
                    f"step t-{t}/j-{j}/s-{sum(scores)}:\t pos: {pos}, subgoal_pos: {subgoal_pos}"
                )
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}"
    )
    return avg_reward, std_reward, avg_norm_score, std_norm_score


def load_hl_planner(args):
    diffusion_experiment = hl_utils.load_diffusion(
        args.logbase,
        args.dataset,
        args.diffusion_loadpath,
        epoch="latest",
    )

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset

    policy = Policy(
        diffusion,
        dataset.normalizer,
    )

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default="exp", type=str)  # Experiment ID
    parser.add_argument(
        "--device", default=0, type=int
    )  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument(
        "--env_name", default="antmaze-ultra-diverse-v0", type=str
    )  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)  # Logging directory
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--save_best_model", action="store_true")

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default="vp", type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument(
        "--ms", default="offline", type=str, help="['online', 'offline']"
    )
    parser.add_argument("--goal_dim", default=15, type=int)
    parser.add_argument("--replan", default=15, type=int)
    parser.add_argument("--p", default=0.2, type=float)
    parser.add_argument("--lcb_coef", default=4.0, type=float)

    args = parser.parse_args()
    args.dir = "results"
    args.lr_decay = True

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f"{args.dir}"

    args.num_epochs = hyperparameters[args.env_name]["num_epochs"]
    args.eval_freq = hyperparameters[args.env_name]["eval_freq"]
    args.eval_episodes = 15 if "v2" in args.env_name else 100

    args.lr = hyperparameters[args.env_name]["lr"]
    args.eta = hyperparameters[args.env_name]["eta"]
    args.max_q_backup = hyperparameters[args.env_name]["max_q_backup"]
    args.reward_tune = hyperparameters[args.env_name]["reward_tune"]
    args.gn = hyperparameters[args.env_name]["gn"]
    args.top_k = hyperparameters[args.env_name]["top_k"]

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay:
        file_name += "|lr_decay"
    file_name += f"|ms-{args.ms}"

    if args.ms == "offline":
        file_name += f"|k-{args.top_k}"
    file_name += f"|{args.seed}"
    file_name += f"|{args.goal_dim}"
    file_name += f"|{args.eta}"
    file_name += f"|{args.max_q_backup}"
    file_name += f"|{args.reward_tune}"
    file_name += f"|{args.p}"
    file_name += f"|{args.lcb_coef}"

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    variant = vars(args)

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(
        f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}"
    )

    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args)
