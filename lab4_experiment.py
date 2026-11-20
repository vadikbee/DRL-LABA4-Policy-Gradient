from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


torch.set_num_threads(1)
DEVICE = torch.device("cpu")
ENV_ID = "CartPole-v1"
SEEDS = (12, 34, 56, 78, 90)
TOTAL_STEPS = 30000
EVAL_EVERY = 2048


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


@dataclass
class Episode:
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]


def collect_episode(
    env: gym.Env,
    policy: PolicyNet,
    max_episode_steps: int = 500,
) -> Episode:
    state, _ = env.reset()
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for _ in range(max_episode_steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        distribution = Categorical(logits=policy(state_tensor))
        action_tensor = distribution.sample()
        next_state, reward, terminated, truncated, _ = env.step(int(action_tensor.item()))
        states.append(np.asarray(state, dtype=np.float32))
        actions.append(int(action_tensor.item()))
        rewards.append(float(reward))
        log_probs.append(distribution.log_prob(action_tensor))
        entropies.append(distribution.entropy())
        state = next_state
        if terminated or truncated:
            break

    return Episode(states, actions, rewards, log_probs, entropies)


def evaluate_policy(policy: PolicyNet, seed: int, episodes: int = 10) -> tuple[float, float, float]:
    env = gym.make(ENV_ID)
    scores: list[float] = []
    for episode_idx in range(episodes):
        state, _ = env.reset(seed=seed + 10000 + episode_idx)
        score = 0.0
        for _ in range(500):
            with torch.no_grad():
                action = int(torch.argmax(policy(torch.as_tensor(state, dtype=torch.float32))).item())
            state, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            if terminated or truncated:
                break
        scores.append(score)
    env.close()
    scores_array = np.asarray(scores, dtype=np.float32)
    return float(scores_array.mean()), float(scores_array.std()), float(np.mean(scores_array >= 475.0))


def add_checkpoint(
    history: dict[str, list[float]],
    policy: PolicyNet,
    seed: int,
    total_steps: int,
    next_checkpoint: int,
) -> int:
    if total_steps >= next_checkpoint:
        score, score_std, success_rate = evaluate_policy(policy, seed)
        history["eval_steps"].append(float(total_steps))
        history["eval_mean"].append(score)
        history["eval_std"].append(score_std)
        history["success_rate"].append(success_rate)
        return next_checkpoint + EVAL_EVERY
    return next_checkpoint


def train_reinforce(seed: int, total_steps: int = TOTAL_STEPS) -> dict:
    set_seed(seed)
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=8e-4)
    history = {"train_steps": [], "train_score": [], "loss": [], "entropy": [], "eval_steps": [], "eval_mean": [], "eval_std": [], "success_rate": []}
    steps = 0
    next_checkpoint = EVAL_EVERY
    while steps < total_steps:
        episode = collect_episode(env, policy)
        returns = torch.as_tensor(discounted_returns(episode.rewards, 0.99), dtype=torch.float32)
        log_probs = torch.stack(episode.log_probs)
        entropy = torch.stack(episode.entropies).mean()
        loss = -(log_probs * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        steps += len(episode.rewards)
        history["train_steps"].append(float(steps))
        history["train_score"].append(float(sum(episode.rewards)))
        history["loss"].append(float(loss.detach()))
        history["entropy"].append(float(entropy.detach()))
        next_checkpoint = add_checkpoint(history, policy, seed, steps, next_checkpoint)
    if not history["eval_steps"] or history["eval_steps"][-1] < steps:
        score, score_std, success_rate = evaluate_policy(policy, seed)
        history["eval_steps"].append(float(steps))
        history["eval_mean"].append(score)
        history["eval_std"].append(score_std)
        history["success_rate"].append(success_rate)
    env.close()
    history["final_score"], history["final_std"], history["final_success"] = evaluate_policy(policy, seed, episodes=30)
    return history


def train_reinforce_baseline(seed: int, total_steps: int = TOTAL_STEPS) -> dict:
    set_seed(seed)
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=8e-4)
    history = {"train_steps": [], "train_score": [], "loss": [], "entropy": [], "baseline": [], "eval_steps": [], "eval_mean": [], "eval_std": [], "success_rate": []}
    steps = 0
    next_checkpoint = EVAL_EVERY
    while steps < total_steps:
        episode = collect_episode(env, policy)
        returns_np = discounted_returns(episode.rewards, 0.99)
        returns = torch.as_tensor(returns_np, dtype=torch.float32)
        baseline = returns.mean().detach()
        advantages = returns - baseline
        log_probs = torch.stack(episode.log_probs)
        entropy = torch.stack(episode.entropies).mean()
        loss = -(log_probs * advantages).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        steps += len(episode.rewards)
        history["train_steps"].append(float(steps))
        history["train_score"].append(float(sum(episode.rewards)))
        history["loss"].append(float(loss.detach()))
        history["entropy"].append(float(entropy.detach()))
        history["baseline"].append(float(baseline))
        next_checkpoint = add_checkpoint(history, policy, seed, steps, next_checkpoint)
    if not history["eval_steps"] or history["eval_steps"][-1] < steps:
        score, score_std, success_rate = evaluate_policy(policy, seed)
        history["eval_steps"].append(float(steps))
        history["eval_mean"].append(score)
        history["eval_std"].append(score_std)
        history["success_rate"].append(success_rate)
    env.close()
    history["final_score"], history["final_std"], history["final_success"] = evaluate_policy(policy, seed, episodes=30)
    return history


def train_vpg(seed: int, total_steps: int = TOTAL_STEPS) -> dict:
    set_seed(seed)
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    value = ValueNet(env.observation_space.shape[0]).to(DEVICE)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=6e-4)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    history = {"train_steps": [], "train_score": [], "policy_loss": [], "value_loss": [], "entropy": [], "eval_steps": [], "eval_mean": [], "eval_std": [], "success_rate": []}
    steps = 0
    next_checkpoint = EVAL_EVERY
    while steps < total_steps:
        episode = collect_episode(env, policy)
        states = torch.as_tensor(np.asarray(episode.states), dtype=torch.float32)
        returns = torch.as_tensor(discounted_returns(episode.rewards, 0.99), dtype=torch.float32)
        values = value(states)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actions = torch.as_tensor(episode.actions, dtype=torch.long)
        distribution = Categorical(logits=policy(states))
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        policy_loss = -(log_probs * advantages).mean() - 0.01 * entropy
        value_loss = F.mse_loss(values, returns)
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_optimizer.step()
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        value_optimizer.step()
        steps += len(episode.rewards)
        history["train_steps"].append(float(steps))
        history["train_score"].append(float(sum(episode.rewards)))
        history["policy_loss"].append(float(policy_loss.detach()))
        history["value_loss"].append(float(value_loss.detach()))
        history["entropy"].append(float(entropy.detach()))
        next_checkpoint = add_checkpoint(history, policy, seed, steps, next_checkpoint)
    if not history["eval_steps"] or history["eval_steps"][-1] < steps:
        score, score_std, success_rate = evaluate_policy(policy, seed)
        history["eval_steps"].append(float(steps))
        history["eval_mean"].append(score)
        history["eval_std"].append(score_std)
        history["success_rate"].append(success_rate)
    env.close()
    history["final_score"], history["final_std"], history["final_success"] = evaluate_policy(policy, seed, episodes=30)
    return history


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_grad(grads: tuple[torch.Tensor | None, ...], model: nn.Module) -> torch.Tensor:
    pieces = []
    for param, grad in zip(model.parameters(), grads):
        pieces.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(pieces)


def set_flat_params(model: nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        size = param.numel()
        param.data.copy_(vector[offset:offset + size].view_as(param))
        offset += size


def conjugate_gradients(fvp, b: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iterations):
        z = fvp(p)
        alpha = rdotr / (torch.dot(p, z) + 1e-8)
        x = x + alpha * p
        r = r - alpha * z
        new_rdotr = torch.dot(r, r)
        if float(new_rdotr) < 1e-10:
            break
        p = r + (new_rdotr / (rdotr + 1e-8)) * p
        rdotr = new_rdotr
    return x


def train_trpo(seed: int, total_steps: int = TOTAL_STEPS, batch_steps: int = 2048) -> dict:
    set_seed(seed)
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    value = ValueNet(env.observation_space.shape[0]).to(DEVICE)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    history = {"train_steps": [], "train_score": [], "policy_loss": [], "value_loss": [], "kl": [], "accepted_fraction": [], "eval_steps": [], "eval_mean": [], "eval_std": [], "success_rate": []}
    steps = 0
    while steps < total_steps:
        states_list: list[np.ndarray] = []
        actions_list: list[int] = []
        returns_list: list[float] = []
        episode_scores: list[float] = []
        collected_steps = 0
        while collected_steps < batch_steps and steps + collected_steps < total_steps:
            episode = collect_episode(env, policy)
            episode_returns = discounted_returns(episode.rewards, 0.99)
            states_list.extend(episode.states)
            actions_list.extend(episode.actions)
            returns_list.extend(episode_returns.tolist())
            episode_scores.append(float(sum(episode.rewards)))
            collected_steps += len(episode.rewards)
        steps += collected_steps
        states = torch.as_tensor(np.asarray(states_list), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions_list), dtype=torch.long)
        returns = torch.as_tensor(np.asarray(returns_list), dtype=torch.float32)
        with torch.no_grad():
            old_logits = policy(states)
            old_distribution = Categorical(logits=old_logits)
            old_log_probs = old_distribution.log_prob(actions)
            old_probs = old_distribution.probs.detach()
            old_values = value(states)
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def surrogate() -> torch.Tensor:
            distribution = Categorical(logits=policy(states))
            ratio = torch.exp(distribution.log_prob(actions) - old_log_probs)
            return (ratio * advantages).mean()

        def mean_kl() -> torch.Tensor:
            distribution = Categorical(logits=policy(states))
            return (old_probs * (torch.log(old_probs + 1e-8) - torch.log(distribution.probs + 1e-8))).sum(dim=1).mean()

        objective = surrogate()
        policy_grads = torch.autograd.grad(objective, policy.parameters())
        gradient = flat_grad(policy_grads, policy).detach()
        kl_grads = torch.autograd.grad(mean_kl(), policy.parameters(), create_graph=True)
        flat_kl_grad = flat_grad(kl_grads, policy)

        def fvp(vector: torch.Tensor) -> torch.Tensor:
            kl_vector = (flat_kl_grad * vector).sum()
            second_grads = torch.autograd.grad(kl_vector, policy.parameters(), retain_graph=True)
            return flat_grad(second_grads, policy).detach() + 0.1 * vector

        direction = conjugate_gradients(fvp, gradient)
        curvature = torch.dot(direction, fvp(direction))
        scale = torch.sqrt(torch.tensor(0.01) / (0.5 * curvature + 1e-8))
        full_step = direction * scale
        old_vector = flat_params(policy)
        old_objective = float(objective.detach())
        accepted_fraction = 0.0
        accepted_kl = 0.0
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0):
            set_flat_params(policy, old_vector + fraction * full_step)
            new_objective = float(surrogate().detach())
            kl_value = float(mean_kl().detach())
            if fraction == 0.0 or (math.isfinite(new_objective) and new_objective > old_objective and kl_value <= 0.01):
                accepted_fraction = fraction
                accepted_kl = kl_value
                break
        if accepted_fraction == 0.0:
            set_flat_params(policy, old_vector)
            accepted_kl = 0.0

        for _ in range(5):
            value_loss = F.mse_loss(value(states), returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            value_optimizer.step()

        history["train_steps"].append(float(steps))
        history["train_score"].append(float(np.mean(episode_scores)))
        history["policy_loss"].append(float(-surrogate().detach()))
        history["value_loss"].append(float(value_loss.detach()))
        history["kl"].append(float(accepted_kl))
        history["accepted_fraction"].append(float(accepted_fraction))
        score, score_std, success_rate = evaluate_policy(policy, seed)
        history["eval_steps"].append(float(steps))
        history["eval_mean"].append(score)
        history["eval_std"].append(score_std)
        history["success_rate"].append(success_rate)
    env.close()
    history["final_score"], history["final_std"], history["final_success"] = evaluate_policy(policy, seed, episodes=30)
    return history


def run_all(seeds: tuple[int, ...] = SEEDS, total_steps: int = TOTAL_STEPS) -> dict:
    runners = {
        "REINFORCE": train_reinforce,
        "REINFORCE + baseline": train_reinforce_baseline,
        "VPG": train_vpg,
        "TRPO": train_trpo,
    }
    result: dict[str, dict[str, dict]] = {}
    for name, runner in runners.items():
        result[name] = {}
        for seed in seeds:
            started = time.perf_counter()
            result[name][str(seed)] = runner(seed, total_steps)
            result[name][str(seed)]["wall_time"] = time.perf_counter() - started
            print(f"{name:24s} seed={seed}: score={result[name][str(seed)]['final_score']:.1f} time={result[name][str(seed)]['wall_time']:.1f}s")
    return result


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    results = run_all()
    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(to_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved results/metrics.json")
