import os
import json
import glob
import torch
import pickle

import numpy as np
import gymnasium as gym

from copy import deepcopy
from typing import Optional
from typing_extensions import override
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process, Queue, Event

from agents.utils.agent import Agent
from misc.evaluation import log_metrics
from misc.utils import setup_directories
from misc.weights import generate_das_dennis_weights, generate_layer_energy_weights, generate_dirichlet_weights

from agents.single_policy.ppo.ppo import PPO
from agents.single_policy.ppo.sample import Sample
from agents.single_policy.ppo.ppo_worker import ppo_worker
from agents.single_policy.ppo.a2c_ppo.model import Policy
from agents.single_policy.ppo.a2c_ppo.envs import make_vec_envs
from agents.single_policy.ppo.external_pareto import ExternalPareto


class MOPPO(Agent):
    def __init__(
            self,
            env_id: str,
            tmp_env: gym.Env,
            num_subproblems: int = 6,
            num_processes: int = 8,
            archive_size: Optional[int] = None,
            num_steps: int = 512,
            gamma: float = 0.995,
            obj_rms: bool = True,
            ob_rms: bool = True,
            use_proper_time_limits: bool = True,
            net_arch: list[int] = [64, 64],
            layernorm: bool = False,
            learning_rate: float = 3e-4,
            eps: float = 1e-4,
            clip_param: float = 0.2,
            ppo_epoch: int = 10,
            num_mini_batches: int = 32,
            entropy_coef: float = 0.0,
            value_loss_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_clipped_value_loss: bool = True,
            use_linear_lr_decay: bool = False,
            lr_decay_ratio: float = 1.0,
            use_gae: bool = True,
            gae_lambda: float = 0.95,
            init_w_sampling: str = 'uniform',
            log: bool = True,
            seed: int = 42,
            max_episode_steps: int = 500,
            device: str = 'cpu',
            name: str = 'mo_ppo',
    ):
        """
        Multi-Objective Proximal Policy Optimization (MO-PPO) agent using
        decomposition-based multi-policy reinforcement learning. Trains one
        independent PPO agent per weight vector, each optimizing a scalarized
        reward signal defined by its assigned preference. Agents collect
        on-policy rollouts in parallel via separate worker processes and do
        not share experience. Policy snapshots are maintained in an external
        Pareto archive to approximate the Pareto front over training.

        Unlike MO-SAC, MO-PPO does not use a shared replay buffer; each agent
        relies exclusively on its own on-policy rollouts, making the
        per-subproblem data budget the primary constraint on learning quality.

        Args:
            env_id (str):
                Gymnasium environment ID used to instantiate worker environments
                for parallel rollout collection.
            tmp_env (gym.Env):
                A temporary environment instance used to infer observation and
                action space properties. Closed after initialization.
            num_subproblems (int):
                Number of scalarized subproblems (weight vectors) and
                corresponding independent PPO agents. Controls the decomposition
                size k.
            num_processes (int):
                Number of parallel environment processes used per PPO worker
                for rollout collection.
            archive_size (int):
                Maximum number of non-dominated policies retained in the
                external Pareto archive. Acts as a bounded archive capacity.
            num_steps (int):
                Number of environment steps collected per rollout before a PPO
                update is performed. Determines the on-policy batch size per
                agent as num_steps * num_processes.
            gamma (float):
                Discount factor for future rewards.
            obj_rms (bool):
                If True, applies running mean-std normalization to the
                per-objective rewards during rollout collection.
            ob_rms (bool):
                If True, applies running mean-std normalization to
                observations during rollout collection.
            use_proper_time_limits (bool):
                If True, handles episode time limits correctly by masking
                terminal states that are due to truncation rather than
                true episode termination.
            net_arch (list[int]):
                Hidden layer sizes for both actor and critic networks.
            layernorm (bool):
                If True, applies layer normalization after each hidden layer
                in the actor and critic networks.
            learning_rate (float):
                Learning rate for the PPO optimizer.
            eps (float):
                Epsilon term added to the optimizer denominator for numerical
                stability.
            clip_param (float):
                PPO clipping parameter controlling the maximum policy update
                step size via the surrogate objective.
            ppo_epoch (int):
                Number of optimization epochs performed on each collected
                rollout batch before discarding the data.
            num_mini_batches (int):
                Number of mini-batches the rollout batch is split into for
                each PPO optimization epoch.
            entropy_coef (float):
                Coefficient for the entropy bonus in the PPO loss, encouraging
                exploration. Set to 0.0 to disable entropy regularization.
            value_loss_coef (float):
                Coefficient for the value function loss term in the total
                PPO loss.
            max_grad_norm (float):
                Maximum gradient norm for gradient clipping during PPO updates.
            use_clipped_value_loss (bool):
                If True, applies clipping to the value function loss analogous
                to the policy surrogate clipping.
            use_linear_lr_decay (bool):
                If True, linearly decays the learning rate from its initial
                value to zero over the course of training.
            lr_decay_ratio (float):
                Final learning rate as a fraction of the initial learning rate
                when linear decay is enabled.
            use_gae (bool):
                If True, uses Generalized Advantage Estimation (GAE) to compute
                advantages. If False, uses standard discounted returns.
            gae_lambda (float):
                Lambda parameter for GAE controlling the bias-variance trade-off
                in advantage estimation.
            init_w_sampling (str):
                Weight vector initialization strategy. Options:
                - 'uniform': Das-Dennis lattice for m=2, layered Riesz-s-energy
                  for m=3.
                - 'dirichlet' or 'random': Dirichlet-sampled weight vectors.
            log (bool):
                Whether to log training metrics to TensorBoard.
            seed (int):
                Random seed for reproducibility across environments and agents.
            max_episode_steps (int):
                Maximum number of steps per episode before truncation.
            device (str):
                Device for tensor computations. MO-PPO typically runs on CPU
                due to its parallel multiprocessing architecture.
            name (str):
                Identifier string used for logging and saving.
        """
        super().__init__(tmp_env, device=device, seed=seed, name=name)
        torch.set_num_threads(1)

        # --- Environment & seeding ---
        self.env_id = env_id
        self.tmp_env = tmp_env
        self.seed = seed
        self.device = device

        # --- Population / Evolution parameters ---
        self.num_subproblems = num_subproblems
        self.num_processes = num_processes
        self.archive_size = archive_size

        # --- Rollout parameters ---
        self.num_steps = num_steps
        self.gamma = gamma
        self.obj_rms = obj_rms
        self.ob_rms = ob_rms
        self.use_proper_time_limits = use_proper_time_limits

        # --- PPO network / optimizer ---
        self.net_arch = net_arch
        self.layernorm = layernorm
        self.learning_rate = learning_rate
        self.eps = eps
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batches = num_mini_batches
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_episode_steps = max_episode_steps

        # --- Learning‐rate schedule & GAE ---
        self.use_linear_lr_decay = use_linear_lr_decay
        self.lr_decay_ratio = lr_decay_ratio
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.init_w_sampling = init_w_sampling
        self.log = log

        self.ep = ExternalPareto(self.archive_size)

        self.initial_weights = self._sample_weights()

        # Build initial sample batch
        self.initial_samples = []
        for idx, w in enumerate(self.initial_weights):
            # Build policy + agent
            policy = Policy(
                self.obs_shape,
                self.action_space,
                net_arch=net_arch,
                reward_dim=self.reward_dim,
                layernorm=self.layernorm
            ).to(self.device)

            agent = PPO(
                actor_critic=policy,
                clip_param=self.clip_param,
                ppo_epoch=self.ppo_epoch,
                num_mini_batches=self.num_mini_batches,
                value_loss_coef=self.value_loss_coef,
                entropy_coef=self.entropy_coef,
                lr=self.learning_rate,
                eps=self.eps,
                max_grad_norm=self.max_grad_norm,
                use_clipped_value_loss=self.use_clipped_value_loss,
            )
            # Gather initial normalization stats by stepping a dummy VecEnv
            venv = make_vec_envs(
                env_name=self.env_id,
                seed=self.seed,
                num_processes=self.num_processes,
                gamma=self.gamma,
                log_dir=None,
                device=self.device,
                allow_early_resets=False,
                obj_rms=self.obj_rms,
                ob_rms=self.ob_rms,
                max_episode_steps=max_episode_steps,
                multiprocessing_envs=False
            )
            env_params = {
                'ob_rms': deepcopy(venv.ob_rms) if venv.ob_rms is not None else None,
                'ret_rms': deepcopy(venv.ret_rms) if venv.ret_rms is not None else None,
                'obj_rms': deepcopy(venv.obj_rms) if venv.obj_rms is not None else None
            }
            venv.close()

            # Create sample and evaluate its initial objective vector
            sample = Sample(env_params, policy, agent, weights=torch.Tensor(w), learning_rate=self.learning_rate,
                            eps=self.eps)
            sample.objs = -np.inf
            self.initial_samples.append(sample)

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
        for i, sample in enumerate(self.ep.sample_batch):
            sample.save(path, f'{file_name}_{i}')
            # Save weights and objectives alongside each sample
            np.savetxt(
                os.path.join(path, f'{file_name}_{i}_weights.txt'),
                sample.weights.cpu().numpy(),
                delimiter=','
            )
            if sample.objs is not None and not np.isscalar(sample.objs):
                np.savetxt(
                    os.path.join(path, f'{file_name}_{i}_objs.txt'),
                    sample.objs,
                    delimiter=','
                )
        print(f'[INFO] All MO-PPO agents saved to {path}.')

    @override
    def load(
            self,
            path: str,
            file_name: str,
            load_replay_buffer: bool = True,
            load_archive: bool = False
    ) -> None:
        # Find all saved actor_critic files and sort by index
        pattern = os.path.join(path, f'{file_name}_*_actor_critic.pt')
        files = sorted(
            glob.glob(pattern),
            key=lambda f: int(os.path.basename(f)
                              .replace(f'{file_name}_', '')
                              .replace('_actor_critic.pt', ''))
        )

        loaded_samples = []
        for actor_path in files:
            base = os.path.basename(actor_path)
            idx = int(base.replace(f'{file_name}_', '').replace('_actor_critic.pt', ''))

            # Load weights from saved file rather than initial_weights
            weight_path = os.path.join(path, f'{file_name}_{idx}_weights.txt')
            if not os.path.exists(weight_path):
                print(f'[WARNING] Weight file not found for agent {idx}, skipping.')
                continue
            weights = np.loadtxt(weight_path, delimiter=',')

            policy = Policy(
                obs_shape=self.obs_shape,
                action_space=self.action_space,
                net_arch=self.net_arch,
                layernorm=self.layernorm,
                reward_dim=self.reward_dim
            ).to(self.device)
            state = torch.load(actor_path, map_location=self.device)
            policy.load_state_dict(state)

            agent = PPO(
                actor_critic=policy,
                clip_param=self.clip_param,
                ppo_epoch=self.ppo_epoch,
                num_mini_batches=self.num_mini_batches,
                value_loss_coef=self.value_loss_coef,
                entropy_coef=self.entropy_coef,
                lr=self.learning_rate,
                eps=self.eps,
                max_grad_norm=self.max_grad_norm,
                use_clipped_value_loss=self.use_clipped_value_loss,
            )

            opt_path = os.path.join(path, f'{file_name}_{idx}_optimizer.pt')
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location=self.device)
                agent.optimizer.load_state_dict(opt_state)

            env_path = os.path.join(path, f'{file_name}_{idx}_env_params.pkl')
            env_params = None
            if os.path.exists(env_path):
                with open(env_path, 'rb') as f:
                    env_params = pickle.load(f)

            objs_path = os.path.join(path, f'{file_name}_{idx}_objs.txt')
            objs = None
            if os.path.exists(objs_path):
                objs = np.loadtxt(objs_path, delimiter=',', ndmin=1)

            sample = Sample(
                env_params, policy, agent,
                weights=torch.Tensor(weights),
                learning_rate=self.learning_rate,
                eps=self.eps
            )
            sample.objs = objs
            loaded_samples.append(sample)

        self.ep.sample_batch = np.array(loaded_samples)
        if loaded_samples:
            self.ep.obj_batch = np.vstack([s.objs for s in loaded_samples])
        else:
            self.ep.obj_batch = np.empty((0,))

        print(f'[INFO] Loaded {len(loaded_samples)} MO-PPO agents from {path}.')

    @override
    def save_config(self, path: str, file_name: str):
        config = {
            'num_subproblems': self.num_subproblems,
            'num_processes': self.num_processes,
            'archive_size': self.archive_size,
            'num_steps': self.num_steps,
            'gamma': self.gamma,
            'obj_rms': self.obj_rms,
            'ob_rms': self.ob_rms,
            'use_proper_time_limits': self.use_proper_time_limits,
            'net_arch': self.net_arch,
            'layernorm': self.layernorm,
            'learning_rate': self.learning_rate,
            'eps': self.eps,
            'clip_param': self.clip_param,
            'ppo_epoch': self.ppo_epoch,
            'num_mini_batches': self.num_mini_batches,
            'entropy_coef': self.entropy_coef,
            'value_loss_coef': self.value_loss_coef,
            'max_grad_norm': self.max_grad_norm,
            'use_clipped_value_loss': self.use_clipped_value_loss,
            'use_linear_lr_decay': self.use_linear_lr_decay,
            'lr_decay_ratio': self.lr_decay_ratio,
            'use_gae': self.use_gae,
            'gae_lambda': self.gae_lambda,
            'max_episode_steps': self.max_episode_steps,
            'init_w_sampling': self.init_w_sampling,
            'seed': self.seed,
        }
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(os.path.join(path, f'{file_name}.json'), 'w') as f:
            json.dump(config, f, indent=4)
        print(f'[INFO] MO-PPO configuration saved')

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
            ref_point: np.ndarray,
            heuristic=None,
            eval_timesteps: int = 10_000,
            known_pareto_front: Optional[list[np.ndarray]] = None,
            num_eval_weights: int = 100,
            eval_rep: int = 5,
            eval_seed: int = 43,
            eval_gamma: float = 0.99,
            save_fronts: bool = False,
            save_models: bool = False,
            log_dir: str = 'mo_ppo',
            file_name: str = 'mo_ppo',
    ):
        """
        Train all MO-PPO agents for a fixed total interaction budget using
        parallel on-policy rollout collection. Each agent runs in a separate
        worker process, collecting rollouts independently without sharing
        experience. After each interval of eval_timesteps environment steps,
        policy snapshots are evaluated and added to the external Pareto archive.

        Args:
            total_timesteps (int):
                Total number of environment interaction steps aggregated across
                all agents. Each agent receives total_timesteps / num_subproblems
                steps under a fixed budget.
            ref_point (np.ndarray):
                Reference point for Hypervolume computation. Should be dominated
                by all Pareto-optimal solutions in the archive.
            heuristic (Optional[DynamicHeuristic]):
                An instantiated heuristic object used for dynamic budget allocation 
                (e.g., RoundRobinHeuristic, EliminationHeuristic). If None, defaults 
                to a standard static round-robin approach.
            eval_timesteps (int):
                Number of aggregated environment steps between consecutive
                evaluations and archive updates. Controls archiving frequency
                and therefore Cardinality of the resulting Pareto front.
            known_pareto_front (Optional[list[np.ndarray]]):
                Ground-truth Pareto front for logging additional metrics such
                as IGD, if available. If None, only archive-based metrics
                are computed.
            num_eval_weights (int):
                Number of weight vectors sampled uniformly from the preference
                simplex for Expected Utility computation during evaluation.
            eval_rep (int):
                Number of evaluation episodes per weight vector. Results are
                averaged to reduce variance in the objective estimates.
            eval_seed (int):
                Random seed for evaluation rollouts to ensure reproducible
                performance estimates across checkpoints.
            eval_gamma (float):
                Discount factor used when computing discounted returns during
                evaluation. May differ from the training discount factor gamma.
            save_fronts (bool):
                If True, Pareto front snapshots are saved to disk at each
                evaluation checkpoint.
            save_models (bool):
                If True, all agent models in the Pareto archive are saved
                to disk upon training completion.
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

        # Compute iteration interval from timestep interval
        timesteps_per_iteration = self.num_subproblems * self.num_steps

        total_iterations = total_timesteps // timesteps_per_iteration

        # Track actual timesteps for evaluation
        current_iteration = 0
        current_timestep = 0

        # Initial evaluation at step 0
        if self.log:
            hv = log_metrics(
                self.ep.obj_batch, ref_point, known_pareto_front,
                reward_dim=self.reward_dim,
                num_sample_weights=num_eval_weights,
                global_step=0,
                writer=writer,
                log=self.log,
                save_fronts=save_fronts,
                pf_store=pf_store
            )
            print(f'Hypervolume @ step 0: {round(hv, 2)}')
        next_eval_timestep = eval_timesteps

        # Track active tasks with a dictionary similar to mo_sac.py
        active_tasks = [{'id': idx, 'scalar_history': [], 'stagnation_count': 0, 'active': True} for idx in range(len(self.initial_samples))]
        
        # Initial quick evaluation to seed history
        for t in active_tasks:
            # Seed history with a low initial value to allow first comparison
            t['scalar_history'].append(-1000.0)

        all_samples = self.initial_samples.copy()

        while current_timestep < total_timesteps:
            # Compute how many iterations until next eval (or end)
            timesteps_to_next_eval = next_eval_timestep - current_timestep
            iters_to_next_eval = max(1, timesteps_to_next_eval // timesteps_per_iteration)

            # Don't exceed total timesteps
            max_iters_remaining = (total_timesteps - current_timestep) // timesteps_per_iteration
            this_update = min(iters_to_next_eval, max_iters_remaining)

            if this_update < 1:
                break

            end_iteration = current_iteration + this_update

            if heuristic is not None and heuristic.__class__.__name__ != 'RoundRobinHeuristic':
                active_list = [t for t in active_tasks if t['active']]
                if not active_list: break

                signals = self._compute_signals(active_tasks)
                for i, t in enumerate(active_list):
                    if heuristic.should_deactivate(t, i, signals):
                        t['active'] = False
                        print(f"--- Task {t['id']} ELIMINATED at {current_timestep} steps ---")

                active_list = [t for t in active_tasks if t['active']]
                if not active_list: break

                signals = self._compute_signals(active_tasks)
                chosen_idx = heuristic.select_next_task(active_list, signals)
                chosen_real_idx = active_list[chosen_idx]['id']
                
                print(f"Allocating {this_update * timesteps_per_iteration} steps to Task {chosen_real_idx}")
                selected_ids = [chosen_real_idx]
                selected_samples = [all_samples[chosen_real_idx]]
            else:
                selected_ids = [i for i, t in enumerate(active_tasks) if t['active']]
                selected_samples = [all_samples[i] for i in selected_ids]

            if not selected_samples: break

            # Launch PPO workers
            processes = []
            results_queue = Queue()
            done_event = Event()
            for sample_id, sample in zip(selected_ids, selected_samples):
                p = Process(target=ppo_worker,
                            args=(sample_id, sample, self.device, current_iteration, end_iteration, total_iterations,
                                  self.env_id, self.seed, self.num_processes, self.num_steps, self.gamma,
                                  self.obj_rms, self.ob_rms, self.reward_dim,
                                  self.use_linear_lr_decay, self.lr_decay_ratio, self.learning_rate,
                                  self.use_gae, self.gae_lambda, self.use_proper_time_limits, self.max_episode_steps,
                                  eval_rep, eval_seed, eval_gamma, results_queue, done_event))
                p.start()
                processes.append(p)

            # Collect results
            all_sample_batch = []
            for _ in processes:
                rl_results = results_queue.get()
                returned_samples = [Sample.copy_from(s) for s in rl_results['offspring_batch']]
                all_sample_batch += returned_samples
                
                # Update the specific sample in our all_samples list
                real_id = rl_results['task_id']
                all_samples[real_id] = returned_samples[-1]
                
                # Compute scalar return from the worker's evaluation and update history
                latest_sample = returned_samples[-1]
                scalar_val = np.dot(latest_sample.objs, latest_sample.weights)
                
                task = active_tasks[real_id]
                prev_s = task['scalar_history'][-1]
                task['stagnation_count'] = 0 if scalar_val > prev_s + 0.5 else task['stagnation_count'] + 1
                task['scalar_history'].append(scalar_val)

            # Signal completion
            done_event.set()
            for p in processes:
                p.join()

            # Update archive
            for sample in all_sample_batch:
                self.ep.update([sample])

            # Update counters
            current_iteration += this_update
            current_timestep = current_iteration * timesteps_per_iteration

            # Evaluate if we've crossed the threshold
            if current_timestep >= next_eval_timestep and self.log:
                self.global_step = current_timestep
                hv = log_metrics(
                    self.ep.obj_batch, ref_point, known_pareto_front,
                    reward_dim=self.reward_dim,
                    num_sample_weights=num_eval_weights,
                    global_step=self.global_step,
                    writer=writer,
                    log=self.log,
                    save_fronts=save_fronts,
                    pf_store=pf_store
                )
                print(f'Hypervolume @ step {self.global_step}: {round(hv, 2)}')
                next_eval_timestep += eval_timesteps

        # Final evaluation
        self.global_step = current_timestep
        if self.log:
            hv = log_metrics(
                self.ep.obj_batch, ref_point, known_pareto_front,
                reward_dim=self.reward_dim,
                num_sample_weights=num_eval_weights,
                global_step=self.global_step,
                writer=writer,
                log=self.log,
                save_fronts=save_fronts,
                pf_store=pf_store
            )
            print(f'Hypervolume @ step {self.global_step}: {round(hv, 2)}')

        self.env.close()

        if save_models:
            self.save(model_path, file_name)

        if writer:
            writer.close()
