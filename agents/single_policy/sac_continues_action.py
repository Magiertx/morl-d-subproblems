import os
import copy
import torch
import numpy as np
import gymnasium as gym
import torch.nn.functional as F
from torch import optim
from typing import Union

from misc.network import polyak_update
from agents.utils.sac_net import Actor, SoftQNetwork


class SACContinues:
    def __init__(self,
                 id: int,
                 env: gym.Env,
                 obs_shape: tuple,
                 action_shape: tuple,
                 action_space: any,
                 reward_dim: int,
                 actor_net_arch: list,
                 critic_net_arch: list,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 batch_size: int = 256,
                 learning_starts: int = 1_000,
                 policy_freq: int = 1,
                 target_net_freq: int = 1,
                 clip_grad_norm: bool = True,
                 actor_clip_norm: float = 2.0,
                 critic_clip_norm: float = 5.0,
                 device: str = 'auto',
                 seed: int = 42,
                 name: str = 'sac',
                 ):
        self.id = id
        self.env = env
        self.device = torch.device(device)
        self.seed = seed
        self.obs_dim = obs_shape[0]
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.reward_dim = reward_dim
        self.action_space = action_space
        self.actor_net_arch = actor_net_arch
        self.critic_net_arch = critic_net_arch
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq
        self.clip_grad_norm = clip_grad_norm
        self.actor_clip_norm = actor_clip_norm
        self.critic_clip_norm = critic_clip_norm

        self.name = name
        self.weights = None  # will be set later by set_weights(w)

        self.num_episodes = 0

        self.actor = Actor(self.obs_dim,
                           self.action_dim,
                           self.env.action_space,
                           net_arch=self.actor_net_arch).to(self.device)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.actor_lr)

        self.qf1 = SoftQNetwork(
            self.obs_dim,
            self.action_dim,
            output_dim=self.reward_dim,
            net_arch=critic_net_arch,
        ).to(self.device)
        self.qf2 = SoftQNetwork(
            self.obs_dim,
            self.action_dim,
            output_dim=self.reward_dim,
            net_arch=critic_net_arch,
        ).to(self.device)
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.critic_lr)

        self.global_step = 0
        self.obs, _ = self.env.reset(seed=self.seed)

    def load(self, path: str, file_name: str, load_replay_buffer: bool = True) -> None:
        full_path = os.path.join(path, f'{file_name}.pt')
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f'No checkpoint file found at {full_path}.')
        pass

    def eval(self, obs: Union[np.ndarray, torch.Tensor], w: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action = self.actor.get_action(obs)
        return action.detach().cpu().numpy()[0]

    def set_weights(self, weights: Union[np.ndarray, torch.Tensor]) -> None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        self.weights = weights.float()

    def update(self, batch: tuple):
        """
        Multi-Objective Soft Actor-Critic update with decomposition-based scalarization.

        Key design principle:
        - Critic: Learns Q^π(s,a) ∈ ℝ^m (expected returns per objective under policy π)
          Uses vector Bellman updates to maintain accurate, unbiased Q-value estimates.
          The Bellman equation Q_i(s,a) = r_i + γ·Q_i(s',a') holds per objective.

        - Actor: Optimizes policy according to user preferences via scalarization g(Q).
          Gradients flow through the (possibly non-linear) scalarization function.

        The critic provides "what are the expected returns?" (objective facts about π).
        The actor decides "which actions are preferred?" (subjective preferences).
        This separation is valid because Bellman describes return accumulation,
        independent of how the policy selects actions.
        """
        if self.global_step < self.learning_starts:
            return

        s_obs, s_actions, s_rewards, s_next_obs, s_dones = batch
        not_done = 1.0 - s_dones.view(-1, 1).float()

        # ==================== CRITIC UPDATE ====================
        # Goal: Learn Q^π(s,a) ∈ ℝ^m — the expected return per objective under policy π.
        #
        # Vector Bellman equation (holds per objective i, independent of policy type):
        #   Q_i^π(s,a) = r_i + γ · 𝔼_{a'~π}[Q_i^π(s',a')]
        #
        # The scalarization function is only used for discrete Q-network selection
        # (which of the two Q-networks to trust for bootstrapping), not for gradients.

        with torch.no_grad():
            # Sample next action from current policy
            next_a, next_logp, _ = self.actor.sample(s_next_obs)

            # Get Q-value vectors from both target networks
            q1_next_vec = self.qf1_target(s_next_obs, next_a)  # ∈ ℝ^{batch × m}
            q2_next_vec = self.qf2_target(s_next_obs, next_a)  # ∈ ℝ^{batch × m}

            # Discrete selection: use scalarization to decide which Q-network to bootstrap from.
            # This is analogous to "clipped double Q-learning" but using preference-based selection.
            # No gradients flow through this selection — it's a discrete choice.
            # Select the q-network that is "better" according to user preferences.
            s1 = (self.weights * q1_next_vec).sum(dim=-1)
            s2 = (self.weights * q2_next_vec).sum(dim=-1)
            use_q1 = (s1 <= s2).unsqueeze(-1)
            q_next_chosen = torch.where(use_q1, q1_next_vec, q2_next_vec)

            # Vector Bellman target: r + γ·Q(s',a') applied per objective
            target_vec = s_rewards + not_done * self.gamma * q_next_chosen

        # Predict current Q-values
        q1_cur_vec = self.qf1(s_obs, s_actions)
        q2_cur_vec = self.qf2(s_obs, s_actions)

        # Loss: Mean squared error per objective, summed across objectives.
        # This is equivalent to learning each objective independently,
        # ensuring unbiased Q-value estimates that satisfy the Bellman equation.
        qf_loss = F.mse_loss(q1_cur_vec, target_vec) + F.mse_loss(q2_cur_vec, target_vec)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                list(self.qf1.parameters()) + list(self.qf2.parameters()),
                self.critic_clip_norm
            )
        self.q_optimizer.step()

        # ==================== ACTOR UPDATE ====================
        # Goal: Find policy π that maximizes the scalarized Q-value g(Q^π(s,π(s))).
        #
        # The actor loss is: 𝔼_s[α·log π(a|s) - g(Q(s,a))]
        #
        # Gradients flow through the scalarization function g(·), allowing the policy
        # to optimize for non-linear preferences (e.g., Tchebycheff, AASF).
        # This is valid because the actor's job is to optimize preferences,
        # not to satisfy the Bellman equation.

        if self.global_step % self.policy_freq == 0:
            # Sample action from current policy
            pi, log_pi, _ = self.actor.sample(s_obs)

            # Get Q-value vectors for the sampled actions
            q1_pi_vec = self.qf1(s_obs, pi)
            q2_pi_vec = self.qf2(s_obs, pi)

            # Apply scalarization function — gradients DO flow through this.
            # The scalarization encodes user preferences over objectives.
            q1_pi_s = (self.weights * q1_pi_vec).sum(dim=-1)
            q2_pi_s = (self.weights * q2_pi_vec).sum(dim=-1)

            # Conservative estimate: take minimum of the two scalarized values
            q_min_pi_s = torch.minimum(q1_pi_s, q2_pi_s)

            # Actor loss: maximize scalarized Q-value while maintaining entropy
            actor_loss = (self.alpha * log_pi.squeeze(-1) - q_min_pi_s).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_norm)
            self.actor_optimizer.step()

        # ==================== TARGET NETWORK UPDATE ====================
        # Soft update of target networks for training stability
        if self.global_step % self.target_net_freq == 0:
            polyak_update(self.qf1.parameters(), self.qf1_target.parameters(), self.tau)
            polyak_update(self.qf2.parameters(), self.qf2_target.parameters(), self.tau)

    def collect_sample(self, replay_buffer):
        if self.global_step < self.learning_starts:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self.obs, dtype=torch.float, device=self.device).unsqueeze(0)
                action, _, _ = self.actor.sample(obs_tensor)
                action = action.detach().cpu().numpy()[0]
        next_obs, vector_reward, terminated, truncated, info = self.env.step(action)
        replay_buffer.add(self.obs, action, vector_reward, next_obs, terminated)
        self.obs = next_obs
        self.global_step += 1
        if terminated or truncated:
            self.num_episodes += 1
            self.obs, _ = self.env.reset(seed=self.seed + self.num_episodes)
