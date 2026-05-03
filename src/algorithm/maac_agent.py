from src.algorithm.prioritized_replay_buffer import PrioritizedReplayBuffer
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.algorithm.actor import Actor
from src.algorithm.critic import Critic

class MAACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        actor_lr: float = 2e-6,
        critic_lr: float = 1e-5,
        gamma: float = 0.99,
        tau: float = 0.001,
        grad_clip: float = 1.0,
        noise_init: float = 0.1,   # 初始噪声
        noise_final: float = 0.01,  # 最低噪声（σ_min）
        noise_decay: float = 1e-6,  # 衰减系数（λ）
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip

        # ---------------- 噪声衰减参数 ----------------
        self.noise_init = noise_init       # 初始噪声 σ₀
        self.noise_std = noise_init        # 当前噪声 σ_t
        self.noise_final = noise_final     # 最低噪声 σ_min
        self.noise_decay = noise_decay     # 指数衰减系数 λ，较慢的衰减速度
        self.total_steps = 0               # 全局步数计数，用于噪声衰减
        # OU噪声参数初始化
        self.ou_theta = 0.15
        self.ou_noise = [np.zeros(action_dim, dtype=np.float32) for _ in range(num_agents)]

        # ---------------- 网络初始化 ----------------
        self.actors: List[Actor] = [Actor(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.actor_targets: List[Actor] = [Actor(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        for tgt, src in zip(self.actor_targets, self.actors):
            tgt.load_state_dict(src.state_dict())

        # 每个 Actor 的优化器 + 余弦学习率调度
        self.actor_optimisers = [optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
        self.actor_schedulers = [CosineAnnealingLR(opt, T_max=150000, eta_min=1e-6) for opt in self.actor_optimisers]

        total_state = state_dim * num_agents
        total_action = action_dim * num_agents
        # === 双 Critic ===
        self.critic1 = Critic(total_state, total_action, num_agents).to(self.device)
        self.critic2 = Critic(total_state, total_action, num_agents).to(self.device)
        self.critic1_target = Critic(total_state, total_action, num_agents).to(self.device)
        self.critic2_target = Critic(total_state, total_action, num_agents).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=1e-5)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=1e-5)

        self.critic1_scheduler = CosineAnnealingLR(self.critic1_opt, T_max=150000, eta_min=1e-6)
        self.critic2_scheduler = CosineAnnealingLR(self.critic2_opt, T_max=150000, eta_min=1e-6)

        # ---------------- 奖励归一化统计 ----------------
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.num_rewards = 0

        # 缓存最新一次的损失，方便主循环记录
        self.last_actor_loss: float | None = None
        self.last_critic_loss: float | None = None

    def select_action(self, states: np.ndarray, noise_bool=True) -> np.ndarray:
        """根据策略网络选择动作，并根据需要添加噪声。"""
        self.total_steps += 1
        self._update_noise()

        states_t = torch.FloatTensor(states).to(self.device)
        actions: List[np.ndarray] = []

        for i, actor in enumerate(self.actors):
            # 获取策略网络输出的动作均值
            mu = actor(states_t[i].unsqueeze(0)).detach().cpu().numpy()[0]

            # 可以控制是否添加探索噪声（使用 OU 噪声）
            if noise_bool:
                # OU噪声更新: θ*(0 - x) + σ * N(0,1)
                self.ou_noise[i] += self.ou_theta * (-self.ou_noise[i]) + self.noise_std * np.random.randn(*mu.shape)
                noise = self.ou_noise[i]
            else:
                noise = np.zeros_like(mu)  # 如果没有噪声，则不添加

            # 计算最终的动作，并确保其在[-1, 1]范围内
            actions.append(np.clip(mu + noise, -1.0, 1.0))

        return np.asarray(actions, dtype=np.float32)

    def update(self, replay_buffer: PrioritizedReplayBuffer, batch_size: int = 512, policy_delay: int = 2) -> None:
        if replay_buffer.tree.n_entries < batch_size:
            return

        # 采样并获取重要性权重
        batch, idxs, is_weights = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        is_weights_t = torch.FloatTensor(is_weights).to(self.device).unsqueeze(-1)

        # 转换为张量
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        rewards_norm = self._normalize_rewards(rewards)
        rewards_t = torch.FloatTensor(rewards_norm).to(self.device).sum(dim=1, keepdim=True)
        # ---------- 目标动作平滑 ----------
        with torch.no_grad():
            next_actions = torch.stack([tgt(next_states_t[:, i, :])for i, tgt in enumerate(self.actor_targets)], dim=1)
            noise = torch.clamp(0.2 * torch.randn_like(next_actions), -0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            target_q1 = self.critic1_target(next_states_t, next_actions)
            target_q2 = self.critic2_target(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2)
            y = rewards_t + self.gamma * (1 - dones_t) * target_q

        current_q1 = self.critic1(states_t, actions_t)
        current_q2 = self.critic2(states_t, actions_t)
        td_errors = (y - current_q1).detach().cpu().numpy().flatten()  # 用 critic1 计算
        huber = nn.functional.smooth_l1_loss
        critic_loss = (is_weights_t *(huber(current_q1, y, reduction='none') +
                                      huber(current_q2, y, reduction='none'))/2).mean()

        # 把最新 critic_loss 缓存下来
        self.last_critic_loss = float(critic_loss.item())

        replay_buffer.update_priorities(idxs, np.abs(td_errors))

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip)
        self.critic1_opt.step()
        self.critic2_opt.step()

        # -------- 延迟更新 Actor --------
        self.update_step = getattr(self, "update_step", 0) + 1
        if self.update_step % policy_delay == 0:
            for p in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                p.requires_grad_(False)
            new_actions = torch.stack([actor(states_t[:, i, :]) for i, actor in enumerate(self.actors)], dim=1)
            actor_loss = -self.critic1(states_t, new_actions).mean()

            # 缓存最新 actor_loss
            self.last_actor_loss = float(actor_loss.item())

            for opt in self.actor_optimisers:
                opt.zero_grad()
            actor_loss.backward()
            for actor in self.actors:
                nn.utils.clip_grad_norm_(actor.parameters(), self.grad_clip)

            # Actor 学习率调度
            for opt, sched in zip(self.actor_optimisers, self.actor_schedulers):
                opt.step()  # 更新权重
                sched.step()  # 再推进学习率调度器

            for p in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                p.requires_grad_(True)
            # 软更新
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
            for tgt, src in zip(self.actor_targets, self.actors):
                self._soft_update(tgt, src)

        # Critic 学习率调度
        self.critic1_scheduler.step()
        self.critic2_scheduler.step()

    def _soft_update(self, tgt: nn.Module, src: nn.Module) -> None:
        """Polyak 软更新：tgt ← (1-τ)·tgt + τ·src"""
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.mul_(1 - self.tau).add_(sp.data, alpha=self.tau)

    def _update_noise(self) -> None:
        """按全局步数执行噪声的慢速衰减。"""
        # 指数衰减更新噪声标准差
        self.noise_std = max(
            self.noise_final,
            self.noise_init * np.exp(-self.noise_decay * self.total_steps)
        )

    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """使用滑动平均更新奖励的均值和方差，以平滑奖励标准化。"""
        batch_mean = rewards.mean()
        batch_var = rewards.var() + 1e-8
        # 利用滑动平均更新全局均值和方差（α=0.99）
        alpha = 0.99
        self.reward_mean = self.reward_mean * alpha + batch_mean * (1 - alpha)
        self.reward_var = self.reward_var * alpha + batch_var * (1 - alpha)
        return (rewards - self.reward_mean) / np.sqrt(self.reward_var)

    def reset_noise(self) -> None:
        """重置 OU 噪声的内部状态。"""
        self.ou_noise = [np.zeros_like(n) for n in self.ou_noise]
