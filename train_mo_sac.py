import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heuristics.dynamic_heuristics import RoundRobinHeuristic, RankBasedHeuristic, BanditHeuristic, EliminationHeuristic

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import numpy as np
import mo_gymnasium as mo_gym

from agents.multi_policy.mo_sac import MOSAC
from misc.utils import read_env_config, read_algo_config


def main():
    parser = argparse.ArgumentParser(description='Run MO-SAC')
    parser.add_argument('--env', type=str, default='halfcheetah',
                        choices=['ant', 'halfcheetah', 'hopper', 'humanoid', 'swimmer', 'walker2d'],
                        help='Environment ID key used in configs/environment_configs.json.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for environment, agent, and weight sampling.')
    parser.add_argument('--max_episode_steps', type=int, default=500,
                        help='Maximum number of steps per episode (passed to mo-gym env).')
    parser.add_argument('--total_timesteps', type=int, default=10_500_000,
                        help='Total number of environment interaction steps for training.')
    parser.add_argument('--num_subproblems', type=int, default=6,
                        help='Number of MO-SAC subproblems / weight vectors (policies) to optimize in parallel.')
    parser.add_argument('--init_w_sampling', type=str, default='uniform',
                        help='Initial weight sampling strategy.')
    parser.add_argument('--heuristic', type=str, default='round-robin',
                        choices=['round-robin', 'rank-based', 'bandit', 'elimination'],
                        help='Budget allocation heuristic to use during training.')

    args = parser.parse_args()

    heuristic_map = {
        'round-robin': (RoundRobinHeuristic, {}),
        'rank-based': (RankBasedHeuristic, {}),
        'bandit': (BanditHeuristic, {'exploration_constant': 1.0}),
        'elimination': (EliminationHeuristic, {'window_size': 5}),
    }
    h_cls, h_kwargs = heuristic_map[args.heuristic]
    heuristic_obj = h_cls(**h_kwargs)
    env_config = read_env_config(f'configs/environment_configs.json')
    env_id = env_config[args.env]['env_id']
    ref_point = env_config[args.env]['ref_point']

    env = mo_gym.make(env_id, max_episode_steps=args.max_episode_steps)
    eval_env = mo_gym.make(env_id, max_episode_steps=args.max_episode_steps)
    ref_point = np.array(ref_point)

    config = read_algo_config(f'configs/multi_policy/mo_sac.json')

    agent = MOSAC(
        env_id=env_id,
        env=env,
        num_subproblems=args.num_subproblems,
        init_w_sampling=args.init_w_sampling,
        archive_size=None,
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        alpha=config['alpha'],
        buffer_size=config['buffer_size'],
        actor_net_arch=config['actor_net_arch'],
        critic_net_arch=config['critic_net_arch'],
        batch_size=config['batch_size'],
        learning_starts=config['learning_starts'],
        gradient_updates=config['gradient_updates'],
        policy_freq=config['policy_freq'],
        target_net_freq=config['target_net_freq'],
        clip_grad_norm=config['clip_grad_norm'],
        actor_clip_norm=config['actor_clip_norm'],
        critic_clip_norm=config['critic_clip_norm'],
        max_episode_steps=args.max_episode_steps,
        update_felten=False,
        log=True,
        seed=args.seed,
        device='auto',
        name='mo_sac'
    )

    log_dir = f'{agent.name}/{args.env}/{args.init_w_sampling}/k_{args.num_subproblems:04d}/ws/s_{args.seed:04d}'

    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        heuristic=heuristic_obj,
        eval_timesteps=10_000,
        known_pareto_front=None,
        num_eval_weights=100,
        eval_rep=5,
        eval_seed=0,
        eval_gamma=0.99,
        save_fronts=True,
        save_models=False,
        log_dir=log_dir,
        log_verbose=0
    )


if __name__ == "__main__":
    main()
