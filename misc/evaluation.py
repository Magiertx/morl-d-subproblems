import torch
import numpy as np
import gymnasium as gym

from typing import Optional, Any

from numpy import floating
from torch.utils.tensorboard import SummaryWriter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


from agents.utils.agent import Agent
from misc.utils import ParetoFrontStore
from misc.metric import hypervolume, sparsity, expected_utility, cardinality, igd, maximum_utility_loss


def _evaluate_episode(agent: Agent, env: gym.Env, w: np.ndarray, scalarization=np.dot, eval_gamma: float = 0.99,
                      seed: int = 42, obs_rms: any = None) -> tuple[float, float, np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    done = False
    vec_return, disc_vec_return = np.zeros_like(w), np.zeros_like(w)
    gamma = 1.0
    while not done:
        if obs_rms is not None:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
        action = agent.eval(obs, w)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        vec_return += r
        disc_vec_return += gamma * r
        gamma *= eval_gamma

    if w is None:
        scalarized_return = vec_return
        scalarized_discounted_return = disc_vec_return
    else:
        scalarized_return = scalarization(w, vec_return)
        scalarized_discounted_return = scalarization(w, disc_vec_return)
    return scalarized_return, scalarized_discounted_return, vec_return, disc_vec_return


def evaluate_single_weight(agent: Agent, env: gym.Env, w: np.ndarray, scalarization=np.dot, rep: int = 5,
                           eval_gamma: float = 0.99, seed: int = 42, obs_rms: any = None,
                           return_episodes: bool = False):
    evals = [
        _evaluate_episode(agent=agent, env=env, w=w, scalarization=scalarization, eval_gamma=eval_gamma, seed=seed + i,
                          obs_rms=obs_rms)
        for i in range(rep)]
    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)
    if return_episodes:
        # Per-episode tuples (scalarized, scalarized_discounted, vec, disc_vec) —
        # needed for the probability-of-improvement signal (TODO G2).
        return avg_scalarized_return, avg_scalarized_discounted_return, avg_vec_return, avg_disc_vec_return, evals
    return avg_scalarized_return, avg_scalarized_discounted_return, avg_vec_return, avg_disc_vec_return


def evaluate_multiple_weights(agent: Agent, env: gym.Env, weights: np.ndarray, scalarization=np.dot, rep: int = 5,
                              eval_gamma: float = 0.99, seed: int = 42) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    avg_scalarized_return, avg_scalarized_discounted_return, avg_vec_return, avg_disc_vec_return = [], [], [], []
    for w in weights:
        scalarized_return, scalarized_discounted_return, vec_return, disc_vec_return = evaluate_single_weight(
            agent=agent, env=env, w=w, scalarization=scalarization, rep=rep, eval_gamma=eval_gamma, seed=seed)
        avg_scalarized_return.append(scalarized_return)
        avg_scalarized_discounted_return.append(scalarized_discounted_return)
        avg_vec_return.append(vec_return)
        avg_disc_vec_return.append(disc_vec_return)
    return np.array(avg_scalarized_return), np.array(avg_scalarized_discounted_return), np.array(
        avg_vec_return), np.array(avg_disc_vec_return)


def compute_metrics(points: list,
                    hv_ref_point: np.ndarray,
                    reward_dim: int,
                    num_sample_weights: int,
                    ref_front: Optional[list[np.ndarray]]):
    P = np.asarray(points, dtype=np.float64)
    if P.size == 0:
        empty = 0.0
        return empty, empty, empty, 0, P, None, None

    nd_idx = NonDominatedSorting().do(-P, only_non_dominated_front=True)
    pareto_front = P[nd_idx]
    weights_set = get_reference_directions('energy', reward_dim, num_sample_weights)

    hv = hypervolume(hv_ref_point, pareto_front)
    sp = sparsity(pareto_front)
    eu = expected_utility(pareto_front, weights_set=weights_set)


    card = cardinality(pareto_front)

    if ref_front is None:
        return hv, sp, eu, card, pareto_front, None, None

    ref_front_arr = np.asarray(ref_front, dtype=np.float64)
    generational_distance = igd(known_front=ref_front_arr, current_estimate=pareto_front)
    mul = maximum_utility_loss(front=pareto_front, reference_set=ref_front_arr, weights_set=weights_set)

    return hv, sp, eu, card, pareto_front, generational_distance, mul


def log_metrics(front: list,
                ref_point: np.ndarray,
                known_pareto_front: Optional[list[np.ndarray]],
                reward_dim: int,
                num_sample_weights: int,
                global_step: int,
                writer: SummaryWriter,
                log: bool,
                save_fronts: bool = False,
                pf_store: ParetoFrontStore = None,
                tag_prefix: str = 'eval',
                weights: Optional[np.ndarray] = None,   # ← new
                ) -> float:
    if len(front) == 0:
        return 0.0
    hv, sp, eu, card, filtered_front, generational_distance, mul = compute_metrics(
        points=front,
        hv_ref_point=ref_point,
        reward_dim=reward_dim,
        num_sample_weights=num_sample_weights,
        ref_front=known_pareto_front)

    if log and writer:
        writer.add_scalar(f'{tag_prefix}/hypervolume', hv, global_step=global_step)
        writer.add_scalar(f'{tag_prefix}/sparsity', sp, global_step=global_step)
        writer.add_scalar(f'{tag_prefix}/eu', eu, global_step=global_step)
        writer.add_scalar(f'{tag_prefix}/cardinality', card, global_step=global_step)
        if known_pareto_front is not None:
            writer.add_scalar(f'{tag_prefix}/generational_distance', generational_distance, global_step=global_step)
            writer.add_scalar(f'{tag_prefix}/mul', mul, global_step=global_step)
            # print(f'@step={global_step} IGD={round(generational_distance,2)}, MU={round(mul,2)}')
        writer.flush()
        if save_fronts and pf_store is not None:
            if tag_prefix == 'policy':
                # Save raw (unfiltered) points with aligned weight vectors
                # so each row can be traced back to its agent
                pf_store.add(np.array(front, dtype=np.float32),
                             global_step=global_step,
                             weights=weights)
            else:
                # Archive: save Pareto-filtered front, no per-point agent mapping
                pf_store.add(filtered_front, global_step=global_step)
            pf_store.flush()
    return hv
