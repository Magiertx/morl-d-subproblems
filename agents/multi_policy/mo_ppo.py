import os
import json
import glob
import torch
import pickle

import numpy as np
import gymnasium as gym

from copy import deepcopy
from typing import Union, Optional
from typing_extensions import override
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool

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

# Thesis signal extensions (G1-G3) live in the root repo (/heuristics/signals.py)
# and are only importable when training is launched through the entry scripts
# (train_mo_ppo.py adds the root to sys.path). Keep the framework usable
# standalone by degrading gracefully to the base signal set.
try:
    from heuristics.signals import (norm_rolling_improvement, probability_of_improvement,
                                    dominance_rank, dominance_improvement)
    _EXT_SIGNALS = True
except ImportError:
    _EXT_SIGNALS = False

def _available_cpus():
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _locked_history_append(history_file: str, header: str, rows: list) -> None:
    """Append rows to the shared history.csv under an exclusive lock file.

    All heuristic runs of one (env, algo, k, seed) share a single history.csv;
    on the SLURM cluster several of them may run concurrently, so both the
    header existence check and the append must be serialized (otherwise:
    duplicated headers mid-file / interleaved partial rows — review 2026-07-03).
    Lock: O_CREAT|O_EXCL lock file (portable across Windows, Linux and NFS);
    stale locks older than 60s are stolen, after 120s we write unlocked as a
    last resort (losing lock safety beats losing the run's results).
    """
    import time as _time
    lock_path = history_file + '.lock'
    deadline = _time.time() + 120.0
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if _time.time() - os.path.getmtime(lock_path) > 60.0:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            if _time.time() > deadline:
                break
            _time.sleep(0.05)
    try:
        write_header = not os.path.isfile(history_file) or os.path.getsize(history_file) == 0
        with open(history_file, 'a', encoding='utf-8') as f:
            if write_header:
                f.write(header)
            for row in rows:
                f.write(row)
    finally:
        if fd is not None:
            os.close(fd)
            try:
                os.remove(lock_path)
            except OSError:
                pass


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
            device: Union[torch.device, str] = 'cpu',
            name: str = 'mo_ppo',
    ):
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

    def _sample_weights(self):
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
    def eval(self, obs: Union[np.ndarray, torch.Tensor], w: Union[np.ndarray, torch.Tensor]) -> Union[
        np.ndarray, torch.Tensor]:
        pass

    @override
    def load(
            self,
            path: str,
            file_name: str,
            load_replay_buffer: bool = True,
            load_archive: bool = False
    ) -> None:
        pattern = os.path.join(path, f'{file_name}_*_actor_critic.pt')
        files = sorted(glob.glob(pattern))

        loaded_samples = []
        for i, actor_path in enumerate(files):
            base = os.path.basename(actor_path)
            idx = int(base[len(file_name) + 1: base.find('_actor_critic.pt')])

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
                arr = np.loadtxt(objs_path, delimiter=',', ndmin=1)
                objs = arr

            sample = Sample(env_params, policy, agent, weights=torch.Tensor(self.initial_weights[i]),
                            learning_rate=self.learning_rate, eps=self.eps)
            sample.objs = objs
            loaded_samples.append(sample)

        self.ep.sample_batch = loaded_samples
        if loaded_samples:
            self.ep.obj_batch = np.vstack([s.objs for s in loaded_samples])
        else:
            self.ep.obj_batch = np.empty((0,))

        print(f'[INFO] Loaded {len(loaded_samples)} MO-PPO agents from {path}')

    @override
    def save(self, path: str, file_name: str, save_replay_buffer: bool = True) -> None:
        for i, sample in enumerate(self.ep.sample_batch):
            sample.save(path, f'{file_name}_{i}')
        print(f'[INFO] All MO-PPO agents saved to {path}.')

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
        """Performance signals consumed by dynamic budget-allocation heuristics."""
        scalars = [t['scalar_history'][-1] if t['scalar_history'] else -200 for t in active_tasks]
        gaps = [max(scalars + [-200]) - s for s in scalars]
        signals = {
            'scalar_rewards': scalars,
            'performance_gaps': gaps,
            'improvement_rates': [
                (t['scalar_history'][-1] - t['scalar_history'][-2]) if len(t['scalar_history']) >= 2 else 0.0
                for t in active_tasks
            ],
            'stagnation_counts': [t['stagnation_count'] for t in active_tasks],
            'scalar_histories': [list(t['scalar_history']) for t in active_tasks],
        }
        if _EXT_SIGNALS:
            signals['norm_improvement_rates'] = [
                norm_rolling_improvement(t['scalar_history']) for t in active_tasks]
            signals['prob_improvements'] = [
                probability_of_improvement(t['eval_returns_history'][-2], t['eval_returns_history'][-1])
                if len(t.get('eval_returns_history', [])) >= 2 else 0.5
                for t in active_tasks]
            signals['dominance_ranks'] = [
                t['dominance_history'][-1] if t.get('dominance_history') else 0
                for t in active_tasks]
            signals['dominance_improvements'] = [
                dominance_improvement(t.get('dominance_history', [])) for t in active_tasks]
        return signals

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
        runs_path, model_path, config_path, pf_store = setup_directories(
            log_dir, file_name, save_fronts, save_models)
        self.save_config(config_path, file_name)
        writer = SummaryWriter(log_dir=runs_path, filename_suffix=f'{self.seed:04d}')

        # ── Iteration bookkeeping ────────────────────────────────────────────
        # One PPO worker iteration consumes (num_processes * num_steps) env steps.
        steps_per_task_iteration = self.num_processes * self.num_steps
        total_iterations = total_timesteps // (self.num_subproblems * self.num_steps)
        current_iteration = 0
        current_timestep = 0
        next_eval_timestep = eval_timesteps

        # Treat RoundRobin (or no heuristic) as "all active tasks in parallel".
        # Any other dynamic heuristic picks a single task per round.
        is_round_robin = (heuristic is None
                          or heuristic.__class__.__name__ == 'RoundRobinHeuristic')

        # Pool size: as many workers as fit on the CPU allocation, capped at k.
        # Each worker uses DummyVecEnv (single-threaded), so 1 CPU per worker.
        cpus_per_task = _available_cpus()
        pool_size = min(cpus_per_task, self.num_subproblems)
        print(f'[INFO] Using persistent worker pool of size {pool_size} '
              f'(cpus={cpus_per_task}, num_processes={self.num_processes}, '
              f'k={self.num_subproblems}, heuristic={heuristic.__class__.__name__ if heuristic else "None"}).')

        # ── Initial evaluation at step 0 ─────────────────────────────────────
        hv = log_metrics(
            self.ep.obj_batch, ref_point, known_pareto_front,
            reward_dim=self.reward_dim,
            num_sample_weights=num_eval_weights,
            global_step=0,
            writer=writer, log=self.log,
            save_fronts=save_fronts, pf_store=pf_store)
        print(f'Hypervolume @ step 0: {round(hv)}')

        all_samples = list(self.initial_samples)

        # Per-subproblem state used by the heuristic. Indexed by sample_id.
        # scalar_history startet LEER (Review 2026-07-11 Punkt 1b): der alte
        # -1000.0-Platzhalter machte die erste echte Rate zu einem +1000-Artefakt
        # und vergiftete jedes darauf aufbauende Signal. SAC seedet mit dem
        # echten Initial-Skalar (hat eine Initial-Evaluation); PPO hat keine,
        # also bleiben Raten undefiniert (=0), bis zwei echte Punkte vorliegen.
        active_tasks = [
            {'id': idx, 'scalar_history': [], 'stagnation_count': 0, 'active': True, 'timesteps_trained': 0,
             'eval_returns_history': [], 'dominance_history': []}
            for idx in range(len(self.initial_samples))
        ]

        import time as time_mod
        start_time = time_mod.perf_counter()

        def _log_history(spent_budget):
            import os
            elapsed = time_mod.perf_counter() - start_time
            history_file = os.path.join(os.path.dirname(os.path.dirname(pf_store.path)), "history.csv") if pf_store else None
            if not history_file:
                return
            h_name = getattr(heuristic, 'label', None) or (heuristic.__class__.__name__.replace("Heuristic", "") if heuristic else "RoundRobin")
            # Run parameters mirrored into every row so runs with different
            # settings stay distinguishable in the pooled results.
            import platform
            run_params = (f"{self.env_id},{self.reward_dim},{self.num_subproblems},"
                          f"{total_timesteps},{eval_timesteps},"
                          f"{int(bool(getattr(heuristic, 'warmup', False)))},"
                          f"{int(getattr(heuristic, 'warmup_steps', 0) or 0)},"
                          f"{platform.node()}")
            rows = []
            for i, task in enumerate(active_tasks):
                sample = all_samples[i]
                if hasattr(sample, 'objs') and sample.objs is not None and not np.array_equal(sample.objs, -np.inf):
                    objs = sample.objs
                else:
                    objs = np.zeros(self.reward_dim)

                r_time = -200.0
                r_ener_f = -200.0
                r_ener_b = -200.0
                if self.reward_dim == 2:
                    r_ener_f = objs[0]
                    r_ener_b = objs[1]
                elif self.reward_dim == 3:
                    r_time = objs[0]
                    r_ener_f = objs[1]
                    r_ener_b = objs[2]

                scalar = float(np.dot(objs, sample.weights.cpu().numpy())) if hasattr(sample, 'weights') and sample.weights is not None else -200.0
                # Latest individual eval-episode returns (';'-joined, CSV-safe) — G2 logging.
                eval_scalars = ";".join(f"{v:.4f}" for v in task['eval_returns_history'][-1]) if task.get('eval_returns_history') else ""
                rows.append(f"{self.seed},{h_name},ON,{spent_budget},{i},{scalar},{r_time},{r_ener_f},{r_ener_b},{task.get('timesteps_trained', 0)},{elapsed:.2f},{eval_scalars},{run_params}\n")
            _locked_history_append(
                history_file,
                "seed,heuristic,algo,spent_budget,task_id,scalar,r_time,r_ener_f,r_ener_b,timesteps_trained,training_time,eval_scalars,env_id,num_obj,k,total_timesteps,eval_timesteps,warmup,warmup_steps,host\n",
                rows)
            if writer:
                writer.add_scalar("eval/training_time", elapsed, global_step=spent_budget)

        _log_history(0)

        # ── Persistent pool for the entire training run ──────────────────────
        with Pool(processes=pool_size) as pool:

            while current_timestep < total_timesteps:
                # Per-round cost depends on whether we run all active tasks or just one.
                if is_round_robin:
                    num_active = sum(1 for t in active_tasks if t['active'])
                    if num_active == 0:
                        break
                    timesteps_per_round = num_active * steps_per_task_iteration
                else:
                    timesteps_per_round = steps_per_task_iteration

                timesteps_to_next_eval = next_eval_timestep - current_timestep
                iters_to_next_eval = max(1, timesteps_to_next_eval // timesteps_per_round)
                max_iters_remaining = (total_timesteps - current_timestep) // timesteps_per_round
                this_update = min(iters_to_next_eval, max_iters_remaining)
                if this_update < 1:
                    break

                end_iteration = current_iteration + this_update

                # ── Heuristic-driven task selection ──────────────────────────
                if is_round_robin:
                    selected_ids = [i for i, t in enumerate(active_tasks) if t['active']]
                else:
                    active_list = [t for t in active_tasks if t['active']]
                    if not active_list:
                        break

                    # Signals MUST be computed over exactly the list handed to the
                    # heuristic: all heuristics index them positionally, so a
                    # full-k array vs. filtered active_list shifts every index
                    # after the first elimination (IndexError / wrong task).
                    signals = self._compute_signals(active_list)
                    # Never let a single deactivation sweep empty the pool: the
                    # in-heuristic "last task" guards check the round-START
                    # length, so simultaneous eliminations could kill every
                    # task and silently end the run early (review 2026-07-03).
                    remaining = len(active_list)
                    for i, t in enumerate(active_list):
                        if remaining <= 1:
                            break
                        if heuristic.should_deactivate(t, i, signals):
                            t['active'] = False
                            remaining -= 1
                            print(f"--- Task {t['id']} ELIMINATED at {current_timestep} steps ---")

                    active_list = [t for t in active_tasks if t['active']]
                    if not active_list:
                        break

                    signals = self._compute_signals(active_list)
                    chosen_idx = heuristic.select_next_task(active_list, signals)
                    chosen_real_id = active_list[chosen_idx]['id']
                    selected_ids = [chosen_real_id]
                    print(f"Allocating {this_update * steps_per_task_iteration} steps to Task {chosen_real_id}")

                selected_samples = [all_samples[i] for i in selected_ids]
                if not selected_samples:
                    break

                # Build task args — one tuple per selected subproblem.
                task_args = [
                    (sample_id, sample, self.device,
                     current_iteration, end_iteration, total_iterations,
                     self.env_id, self.seed, self.num_processes,
                     self.num_steps, self.gamma,
                     self.obj_rms, self.ob_rms, self.reward_dim,
                     self.use_linear_lr_decay, self.lr_decay_ratio, self.learning_rate,
                     self.use_gae, self.gae_lambda,
                     self.use_proper_time_limits, self.max_episode_steps,
                     eval_rep, eval_seed, eval_gamma)
                    for sample_id, sample in zip(selected_ids, selected_samples)
                ]

                # Pool dispatches all tasks across pool_size workers; blocks until done.
                results = pool.starmap(ppo_worker, task_args)

                # Collect offspring and update per-task heuristic state.
                all_sample_batch = []
                for r in results:
                    returned = [Sample.copy_from(s) for s in r['offspring_batch']]
                    all_sample_batch += returned

                    real_id = r['task_id']
                    latest = returned[-1]
                    all_samples[real_id] = latest

                    scalar_val = float(np.dot(latest.objs, latest.weights))
                    task = active_tasks[real_id]
                    task['timesteps_trained'] += this_update * steps_per_task_iteration
                    prev_s = task['scalar_history'][-1] if task['scalar_history'] else None
                    task['stagnation_count'] = 0 if (prev_s is None or scalar_val > prev_s + 0.5) else task['stagnation_count'] + 1
                    task['scalar_history'].append(scalar_val)

                    # Individual eval-episode returns, scalarized on the task
                    # weight — feeds probability of improvement (G2).
                    if r.get('eval_episode_objs') is not None:
                        w_np = latest.weights.cpu().numpy() if hasattr(latest.weights, 'cpu') else np.asarray(latest.weights)
                        episode_scalars = [float(np.dot(ep, w_np)) for ep in r['eval_episode_objs']]
                        task['eval_returns_history'].append(episode_scalars)

                # Update archive
                for sample in all_sample_batch:
                    self.ep.update([sample])

                # Dominance rank of each trained task vs. the updated archive (G3).
                if _EXT_SIGNALS:
                    for r in results:
                        task = active_tasks[r['task_id']]
                        task['dominance_history'].append(
                            dominance_rank(all_samples[r['task_id']].objs, self.ep.obj_batch))

                # Advance counters
                current_iteration += this_update
                current_timestep += this_update * timesteps_per_round

                # Eval checkpoint
                if current_timestep >= next_eval_timestep:
                    self.global_step = current_timestep
                    hv = log_metrics(
                        self.ep.obj_batch, ref_point, known_pareto_front,
                        reward_dim=self.reward_dim,
                        num_sample_weights=num_eval_weights,
                        global_step=self.global_step,
                        writer=writer, log=self.log,
                        save_fronts=save_fronts, pf_store=pf_store)
                    _log_history(self.global_step)
                    print(f'Hypervolume @ step {self.global_step}: {round(hv)}')
                    elapsed_min = (time_mod.perf_counter() - start_time) / 60.0
                    pct = 100.0 * current_timestep / total_timesteps
                    eta_min = elapsed_min * (total_timesteps - current_timestep) / max(1, current_timestep)
                    print(f'[PROGRESS] {current_timestep:,}/{total_timesteps:,} steps '
                          f'({pct:.1f}%) | {elapsed_min:.1f} min | ETA ~{eta_min:.1f} min', flush=True)
                    next_eval_timestep += eval_timesteps

            # ── Final evaluation ──────────────────────────────────────────────
            self.global_step = current_timestep
            hv = log_metrics(
                self.ep.obj_batch, ref_point, known_pareto_front,
                reward_dim=self.reward_dim,
                num_sample_weights=num_eval_weights,
                global_step=self.global_step,
                writer=writer, log=self.log,
                save_fronts=save_fronts, pf_store=pf_store)
            _log_history(self.global_step)
            print(f'Hypervolume @ step {self.global_step}: {round(hv)}')

        # Pool is now closed (context manager exited)

        if save_models:
            self.save(model_path, file_name)
        self.env.close()
        if writer:
            writer.close()