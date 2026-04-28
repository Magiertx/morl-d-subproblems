import os
import copy
import json

import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym

from typing import Optional
from typing_extensions import override

from agents.utils.agent import Agent
from misc.pareto import ParetoArchive
from misc.utils import ParetoFrontStore, setup_directories
from misc.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from misc.evaluation import log_metrics, evaluate_single_weight
from agents.single_policy.sac_continues_action import SACContinues
from misc.weights import generate_das_dennis_weights, generate_layer_energy_weights, generate_dirichlet_weights


class MOSAC(Agent):
    def __init__(self,
                 env_id: str,
                 env: gym.Env,
                 num_subproblems: int = 6,
                 init_w_sampling: str = 'uniform',
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 archive_size: Optional[int] = None,
                 buffer_size: int = 1_000_000,
                 actor_net_arch: list[int] = [256, 256],
                 critic_net_arch: list[int] = [256, 256],
                 batch_size: int = 256,
                 learning_starts: int = 1_000,
                 gradient_updates: int = 1,
                 policy_freq: int = 1,
                 target_net_freq: int = 1,
                 clip_grad_norm: bool = True,
                 actor_clip_norm: float = 2.0,
                 critic_clip_norm: float = 5.0,
                 max_episode_steps: int = 500,
                 update_felten: bool = False,  # Critic update after https://jair.org/index.php/jair/article/view/15702
                 log: bool = True,
                 seed: int = 42,
                 device: str = 'auto',
                 name: str = 'mo_sac'):
        """
        Multi-Objective Soft Actor-Critic (MO-SAC) agent using decomposition-based
        multi-policy reinforcement learning. Trains one independent SAC agent per
        weight vector, all sharing a single replay buffer. Each agent optimizes a
        scalarized reward signal defined by its assigned weight vector, and policy
        snapshots are stored in a Pareto archive to approximate the Pareto front.

        Args:
            env_id (str):
                Gymnasium environment ID used to instantiate per-agent environments.
            env (gym.Env):
                A temporary environment instance used to infer observation and action
                space properties. Closed after initialization.
            num_subproblems (int):
                Number of scalarized subproblems (weight vectors) and corresponding
                independent SAC agents. Controls the decomposition size k.
            init_w_sampling (str):
                Weight vector initialization strategy. Options:
                - 'uniform': Das-Dennis lattice for m=2, layered Riesz-s-energy for m=3.
                - 'dirichlet' or 'random': Dirichlet-sampled weight vectors.
            actor_lr (float):
                Learning rate for the actor network optimizer.
            critic_lr (float):
                Learning rate for the critic network optimizer.
            gamma (float):
                Discount factor for future rewards.
            tau (float):
                Polyak averaging coefficient for soft target network updates.
            alpha (float):
                Entropy regularization coefficient controlling the exploration-
                exploitation trade-off in SAC.
            archive_size (Optional[int]):
                Maximum number of non-dominated solutions retained in the Pareto
                archive. If None, the archive is unconstrained.
            buffer_size (int):
                Maximum capacity of the shared replay buffer.
            actor_net_arch (list[int]):
                Hidden layer sizes for the actor network.
            critic_net_arch (list[int]):
                Hidden layer sizes for the critic network.
            batch_size (int):
                Number of transitions sampled per gradient update step.
            learning_starts (int):
                Number of environment steps collected before gradient updates begin.
            gradient_updates (int):
                Number of gradient update steps performed per environment step.
            policy_freq (int):
                Frequency of actor network updates relative to critic updates.
            target_net_freq (int):
                Frequency of target network updates relative to critic updates.
            clip_grad_norm (bool):
                Whether to apply gradient norm clipping during optimization.
            actor_clip_norm (float):
                Maximum gradient norm for actor network updates.
            critic_clip_norm (float):
                Maximum gradient norm for critic network updates.
            max_episode_steps (int):
                Maximum number of steps per episode before truncation.
            update_felten (bool):
                If True, uses the scalar critic update rule from Felten et al.
                (https://jair.org/index.php/jair/article/view/15702) that applies
                scalarization within the Bellman backup. If False, uses the
                vector-valued critic update that preserves per-objective TD errors.
            log (bool):
                Whether to log training metrics to TensorBoard.
            seed (int):
                Random seed for reproducibility across environments and agents.
            device (str):
                Device for tensor computations. Use 'auto' to select GPU if
                available, otherwise CPU.
            name (str):
                Identifier string used for logging and saving.
        """
        super().__init__(env, device=device, seed=seed, name=name)
        self.env_id = env_id
        self.num_subproblems = num_subproblems
        self.init_w_sampling = init_w_sampling
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.actor_net_arch = actor_net_arch
        self.critic_net_arch = critic_net_arch
        self.archive_size = archive_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gradient_updates = gradient_updates
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq
        self.clip_grad_norm = clip_grad_norm
        self.actor_clip_norm = actor_clip_norm
        self.critic_clip_norm = critic_clip_norm
        self.max_episode_steps = max_episode_steps
        self.update_felten = update_felten
        self.log = log

        # Pareto archive to store evaluated policies
        self.archive = ParetoArchive(max_size=self.archive_size)
        self.replay_buffer = ReplayBuffer(self.obs_shape,
                                          self.action_dim,
                                          rew_dim=self.reward_dim,
                                          max_size=self.buffer_size)

        initial_weights = self._sample_weights()
        self.env_seed = self.seed
        self.agents = []
        for i in range(self.num_subproblems):
            agent = self._create_new_agent(initial_weights[i])
            self.agents.append(agent)

    def _create_new_agent(self, w: np.ndarray):
        env = mo_gym.make(self.env_id, max_episode_steps=self.max_episode_steps)
        env.observation_space.seed(self.env_seed)
        env.action_space.seed(self.env_seed)
        agent = SACContinues(
            id=self.env_seed,
            env=env,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_space=self.action_space,
            reward_dim=self.reward_dim,
            actor_net_arch=self.actor_net_arch,
            critic_net_arch=self.critic_net_arch,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            policy_freq=self.policy_freq,
            target_net_freq=self.target_net_freq,
            tau=self.tau,
            gamma=self.gamma,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_starts=self.learning_starts,
            clip_grad_norm=self.clip_grad_norm,
            actor_clip_norm=self.actor_clip_norm,
            critic_clip_norm=self.critic_clip_norm,
            device=self.device,
            seed=self.env_seed,
            name=f'sac_{self.env_seed}'
        )
        agent.set_weights(w)
        self.env_seed += 1
        return agent

    def _sample_weights(self) -> list[float]:
        if self.init_w_sampling == 'uniform':
            if self.reward_dim == 2:
                weights = generate_das_dennis_weights(num_weights=self.num_subproblems,
                                                      reward_dim=self.reward_dim)
                # initial_weights = generate_energy_weights(num_weights=self.num_subproblems, reward_dim=self.reward_dim)
            elif self.reward_dim == 3:
                layers = {3: [1],
                          4: [1, 0],
                          6: [2],
                          7: [2, 0],
                          9: [3],
                          10: [3, 0],
                          12: [4],
                          13: [4, 0],
                          15: [4, 1],
                          16: [4, 0, 1],
                          18: [5, 1],
                          19: [5, 0, 1],
                          21: [5, 2]}
                if self.num_subproblems not in layers.keys(): raise ValueError(
                    'Invalid number of subproblems specified in reward_dim')
                weights = generate_layer_energy_weights(layers=layers[self.num_subproblems],
                                                        reward_dim=self.reward_dim)
            else:
                raise NotImplementedError
        elif self.init_w_sampling == 'dirichlet' or self.init_w_sampling == 'random':
            weights = generate_dirichlet_weights(num_weights=self.num_subproblems, reward_dim=self.reward_dim,
                                                 alpha=1.0)
        else:
            raise NotImplementedError

        return weights

    @override
    def save(self, path: str, file_name: str, save_replay_buffer: bool = True) -> None:
        os.makedirs(path, exist_ok=True)
        for i, (agent, evaluation) in enumerate(zip(self.archive.individuals, self.archive.evaluations)):
            agent.save(path, f'{file_name}_{i}', save_replay_buffer)
            np.savetxt(os.path.join(path, f'{file_name}_{i}_eval.txt'), evaluation, delimiter=',')
            np.savetxt(os.path.join(path, f'{file_name}_{i}_weights.txt'), agent.weights.cpu().numpy(), delimiter=',')
        print(f'[INFO] All SAC agents saved to {path}.')

    @override
    def load(self, path: str, file_name: str, load_replay_buffer: bool = True) -> None:
        self.agents = []
        self.archive = ParetoArchive(max_size=self.archive_size)

        i = 0
        while True:
            weight_path = os.path.join(path, f'{file_name}_{i}_weights.txt')
            eval_path = os.path.join(path, f'{file_name}_{i}_eval.txt')
            if not os.path.exists(weight_path):
                break

            weights = np.loadtxt(weight_path, delimiter=',')
            evaluation = np.loadtxt(eval_path, delimiter=',')

            agent = self._create_new_agent(weights)
            agent.load(path, f'{file_name}_{i}', load_replay_buffer)

            self.archive.add(copy.deepcopy(agent), evaluation)
            self.agents.append(agent)
            i += 1

        print(f'[INFO] Loaded {len(self.agents)} SAC agents from {path}.')

    @override
    def save_config(self, path: str, file_name: str) -> None:
        config = {
            'env_id': self.env_id,
            'num_subproblems': self.num_subproblems,
            'init_w_sampling': self.init_w_sampling,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'archive_size': self.archive_size,
            'buffer_size': self.buffer_size,
            'actor_net_arch': self.actor_net_arch,
            'critic_net_arch': self.critic_net_arch,
            'batch_size': self.batch_size,
            'learning_starts': self.learning_starts,
            'gradient_updates': self.gradient_updates,
            'policy_freq': self.policy_freq,
            'target_net_freq': self.target_net_freq,
            'clip_grad_norm': self.clip_grad_norm,
            'actor_clip_norm': self.actor_clip_norm,
            'critic_clip_norm': self.critic_clip_norm,
            'max_episode_steps': self.max_episode_steps,
            'update_felten': self.update_felten,
            'seed': self.seed,
        }
        with open(os.path.join(path, f'{file_name}.json'), 'w') as f:
            json.dump(config, f)
        print(f'[INFO] MO-SAC configuration saved.')

    def _eval_all_agents(self,
                         agents,
                         eval_env: gym.Env,
                         ref_point: np.ndarray,
                         num_sample_weights: int,
                         eval_rep: int,
                         eval_seed: int,
                         eval_gamma: float,
                         writer: SummaryWriter,
                         save_front: bool,
                         pf_store: ParetoFrontStore | None,
                         known_pareto_front: Optional[list[np.ndarray]] = None,
                         log_verbose: int = 0,
                         ):
        for agent in agents:
            scalarized_return, scalarized_discounted_return, vec_return, disc_vec_return = (
                evaluate_single_weight(agent, env=eval_env, w=agent.weights.cpu().numpy(), rep=eval_rep,
                                       seed=eval_seed, eval_gamma=eval_gamma))
            if log_verbose > 0:
                writer.add_scalar(f'return/scalarized_return/agent_{agent.id}', scalarized_return, self.global_step)
                writer.add_scalar(f'return/scalarized_discounted_return/agent_{agent.id}', scalarized_discounted_return,
                                  self.global_step)
                for j, (r_val, d_val) in enumerate(zip(vec_return, disc_vec_return)):
                    writer.add_scalar(f'return/vec_return_{j}/agent_{agent.id}', r_val, self.global_step)
                    writer.add_scalar(f'return/vec_discounted_return_{j}/agent_{agent.id}', d_val, self.global_step)
            self.archive.add(copy.deepcopy(agent), disc_vec_return)

        front = self.archive.evaluations
        hv = log_metrics(front, ref_point=ref_point, known_pareto_front=known_pareto_front,
                         reward_dim=self.reward_dim, num_sample_weights=num_sample_weights,
                         global_step=self.global_step, writer=writer, log=self.log,
                         save_fronts=save_front, pf_store=pf_store)
        print(f'Hypervolume @ step {self.global_step}: {round(hv, 2)}')

    def _train_all_agents(self,
                          timesteps: int):
        remaining = timesteps
        while remaining > 0:
            step_count = min(len(self.agents), remaining)
            for idx, agent in enumerate(list(self.agents)[:step_count]):
                agent.collect_sample(self.replay_buffer)
                self.global_step += 1
                remaining -= 1

            if self.replay_buffer.size < self.batch_size:
                continue

            for _ in range(self.gradient_updates):
                batch = self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
                for ag in self.agents:
                    if self.update_felten:
                        ag.update_felten(batch)
                    else:
                        ag.update(batch)

    def _train_single_agent(self, agent_idx: int, timesteps: int):
        """
        Train a single specific agent for a given number of timesteps.
        
        Args:
            agent_idx (int): Index of the agent to train.
            timesteps (int): Number of environment steps to collect and train on.
        """
        remaining = timesteps
        agent = self.agents[agent_idx]
        while remaining > 0:
            agent.collect_sample(self.replay_buffer)
            self.global_step += 1
            remaining -= 1

            if self.replay_buffer.size < self.batch_size:
                continue

            for _ in range(self.gradient_updates):
                batch = self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
                if self.update_felten:
                    agent.update_felten(batch)
                else:
                    agent.update(batch)

    def _evaluate_agent(self, agent, env: gym.Env, weight: np.ndarray) -> float:
        """
        Evaluate a single agent's performance using its current policy.
        
        Args:
            agent (Agent): The agent instance to evaluate.
            env (gym.Env): The evaluation environment.
            weight (np.ndarray): The preference weight vector assigned to the agent.
            
        Returns:
            float: The scalarized performance score.
        """
        import torch
        all_rewards = []
        for _ in range(3):
            obs, _ = env.reset(); term = trunc = False; ep_r = np.zeros(self.reward_dim)
            while not (term or trunc):
                with torch.no_grad():
                    action = agent.eval(obs, weight)
                obs, r, term, trunc, _ = env.step(action)
                ep_r += r
            all_rewards.append(ep_r)
        w_np = weight.cpu().numpy() if hasattr(weight, 'cpu') else weight
        return float(np.dot(np.mean(all_rewards, axis=0), w_np))

    def _compute_signals(self, active_tasks: list) -> dict:
        """
        Compute performance signals used by budget allocation heuristics.
        
        Args:
            active_tasks (list): List of currently active task dictionaries.
            
        Returns:
            dict: Dictionary containing historical and current performance metrics.
        """
        scalars = [t['scalar_history'][-1] if t['scalar_history'] else -200 for t in active_tasks]
        gaps = [max(scalars + [-200]) - s for s in scalars]
        return {
            'scalar_rewards': scalars,
            'performance_gaps': gaps,
            'improvement_rates': [(t['scalar_history'][-1] - t['scalar_history'][-2]) if len(t['scalar_history']) >= 2 else 0.0 for t in active_tasks],
            'stagnation_counts': [t['stagnation_count'] for t in active_tasks],
            'scalar_histories': [list(t['scalar_history']) for t in active_tasks]
        }

    def train(
            self,
            total_timesteps: int,
            eval_timesteps: int,
            eval_env: gym.Env,
            ref_point: np.ndarray,
            heuristic=None,
            known_pareto_front: Optional[list[np.ndarray]] = None,
            num_eval_weights: int = 100,
            eval_rep: int = 5,
            eval_seed: int = 43,
            eval_gamma: float = 0.99,
            save_fronts: bool = False,
            save_models: bool = False,
            log_verbose: int = 0,
            log_dir: str = 'mo_sac',
            file_name: str = 'mo_sac',
    ):
        """
        Train all MO-SAC agents for a fixed total interaction budget using a
        shared replay buffer. All agents collect environment transitions in
        parallel and update their networks from a shared minibatch sampled
        at each step. Policy snapshots are periodically evaluated and added
        to the Pareto archive to approximate the Pareto front.

        Args:
            total_timesteps (int):
                Total number of environment interaction steps across all agents.
            eval_timesteps (int):
                Number of environment steps between consecutive evaluations.
                Controls the archiving frequency and therefore Cardinality.
            eval_env (gym.Env):
                Separate environment instance used exclusively for policy
                evaluation to avoid interference with training rollouts.
            ref_point (np.ndarray):
                Reference point for Hypervolume computation. Should be
                dominated by all Pareto-optimal solutions.
            heuristic:
                Optional budget allocation heuristic instance that implements
                methods for deciding which agent to train at each interval and
                when to deactivate underperforming agents. If None, a simple
                round-robin allocation strategy is used where all agents are trained
                for equal amounts of time without deactivation.
            known_pareto_front (Optional[list[np.ndarray]]):
                Ground-truth Pareto front for logging additional metrics such
                as IGD, if available. If None, only archive-based metrics
                are computed.
            num_eval_weights (int):
                Number of weight vectors sampled uniformly from the preference
                simplex for Expected Utility computation during evaluation.
            eval_rep (int):
                Number of evaluation episodes per weight vector. Results are
                averaged to reduce variance.
            eval_seed (int):
                Random seed for the evaluation environment to ensure
                reproducible evaluation rollouts.
            eval_gamma (float):
                Discount factor used when computing discounted returns during
                evaluation. May differ from the training discount factor gamma.
            save_fronts (bool):
                If True, Pareto front snapshots are saved to disk at each
                evaluation checkpoint.
            save_models (bool):
                If True, all agent models in the Pareto archive are saved
                to disk upon training completion.
            log_verbose (int):
                Verbosity level for per-agent TensorBoard logging. If 0,
                only aggregate metrics are logged. Higher values log
                per-agent returns and objective values.
            log_dir (str):
                Directory path for TensorBoard logs, saved configurations,
                and optionally model checkpoints and Pareto front snapshots.
            file_name (str):
                Base file name used for all saved outputs including logs,
                configurations, models, and Pareto front files.
        """
        runs_path, model_path, config_path, pf_store = setup_directories(log_dir, file_name, save_fronts, save_models)

        self.save_config(config_path, file_name)
        writer = SummaryWriter(log_dir=runs_path, filename_suffix=f'{self.seed:04d}')
        eval_env.action_space.seed(eval_seed)
        eval_env.observation_space.seed(eval_seed)

        self._eval_all_agents(
            self.agents,
            eval_env=eval_env,
            ref_point=ref_point,
            num_sample_weights=num_eval_weights,
            eval_rep=eval_rep,
            eval_seed=eval_seed,
            eval_gamma=eval_gamma,
            writer=writer,
            save_front=save_fronts,
            pf_store=pf_store,
            known_pareto_front=known_pareto_front,
            log_verbose=log_verbose
        )

        active_tasks = [{'id': idx, 'agent': agent, 'scalar_history': [], 'stagnation_count': 0, 'active': True, 'weight': agent.weights} for idx, agent in enumerate(self.agents)]
        
        for t in active_tasks:
            s = self._evaluate_agent(t['agent'], eval_env, t['weight'])
            t['scalar_history'].append(s)

        while self.global_step < total_timesteps:
            steps_this_interval = min(eval_timesteps, total_timesteps - self.global_step)

            if heuristic is not None and heuristic.__class__.__name__ != 'RoundRobinHeuristic':
                active_list = [t for t in active_tasks if t['active']]
                if not active_list: break

                signals = self._compute_signals(active_tasks)
                for i, t in enumerate(active_list):
                    if heuristic.should_deactivate(t, i, signals):
                        t['active'] = False
                        print(f"--- Task {t['id']} ELIMINATED at {self.global_step} steps ---")

                active_list = [t for t in active_tasks if t['active']]
                if not active_list: break

                signals = self._compute_signals(active_tasks)
                chosen_idx = heuristic.select_next_task(active_list, signals)
                chosen = active_list[chosen_idx]
                
                print(f"Allocating {steps_this_interval} steps to Task {chosen['id']}")
                self._train_single_agent(chosen['id'], steps_this_interval)
                
                s = self._evaluate_agent(chosen['agent'], eval_env, chosen['weight'])
                prev_s = chosen['scalar_history'][-1]
                chosen['stagnation_count'] = 0 if s > prev_s + 0.5 else chosen['stagnation_count'] + 1
                chosen['scalar_history'].append(s)
            else:
                self._train_all_agents(timesteps=steps_this_interval)

            self._eval_all_agents(
                self.agents,
                eval_env=eval_env,
                ref_point=ref_point,
                num_sample_weights=num_eval_weights,
                eval_rep=eval_rep,
                eval_seed=eval_seed,
                eval_gamma=eval_gamma,
                writer=writer,
                save_front=save_fronts,
                pf_store=pf_store,
                known_pareto_front=known_pareto_front,
                log_verbose=log_verbose
            )

        self.env.close()
        eval_env.close()

        if save_models:
            self.save(path=model_path, file_name=file_name)

        if writer:
            writer.close()
