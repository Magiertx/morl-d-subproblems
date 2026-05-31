import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heuristics.dynamic_heuristics import (
    RoundRobinHeuristic,
    RankBasedHeuristic,
    BanditHeuristic,
    EliminationHeuristic,
    StagnationBasedHeuristic,
)

import torch
import argparse
import numpy as np
import mo_gymnasium as mo_gym
from agents.multi_policy.mo_ppo import MOPPO
from misc.utils import read_env_config, read_algo_config

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main():
    torch.set_default_dtype(torch.float64)

    parser = argparse.ArgumentParser(description='Run MO-PPO')
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
                        choices=['round-robin', 'rank-based', 'bandit', 'elimination', 'stagnation-based'],
                        help='Budget allocation heuristic to use during training.')

    args = parser.parse_args()

    heuristic_map = {
        'round-robin': (RoundRobinHeuristic, {}),
        'rank-based': (RankBasedHeuristic, {}),
        'bandit': (BanditHeuristic, {'exploration_constant': 1.0}),
        'elimination': (EliminationHeuristic, {'window_size': 5}),
        'stagnation-based': (StagnationBasedHeuristic, {}),
    }
    h_cls, h_kwargs = heuristic_map[args.heuristic]
    heuristic_obj = h_cls(**h_kwargs)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_config = read_env_config(os.path.join(base_dir, 'configs', 'environment_configs.json'))
    env_id = env_config[args.env]['env_id']
    ref_point = env_config[args.env]['ref_point']

    env = mo_gym.make(env_id, max_episode_steps=args.max_episode_steps)
    ref_point = np.array(ref_point)

    config = read_algo_config(os.path.join(base_dir, 'configs', 'multi_policy', 'mo_ppo.json'))

    agent = MOPPO(
        env_id=env_id,
        tmp_env=env,
        num_subproblems=args.num_subproblems,
        init_w_sampling=args.init_w_sampling,
        num_processes=config['num_processes'],
        archive_size=None,
        num_steps=config['num_steps'],
        gamma=config['gamma'],
        obj_rms=config['obj_rms'],
        ob_rms=config['ob_rms'],
        use_proper_time_limits=config['use_proper_time_limits'],
        net_arch=config['net_arch'],
        layernorm=config['layernorm'],
        learning_rate=config['learning_rate'],
        eps=1e-5,
        clip_param=config['clip_param'],
        ppo_epoch=config['ppo_epoch'],
        num_mini_batches=config['num_mini_batches'],
        entropy_coef=config['entropy_coef'],
        value_loss_coef=config['value_loss_coef'],
        max_grad_norm=config['max_grad_norm'],
        use_clipped_value_loss=config['use_clipped_value_loss'],
        use_linear_lr_decay=config['use_linear_lr_decay'],
        lr_decay_ratio=config['lr_decay_ratio'],
        use_gae=config['use_gae'],
        gae_lambda=config['gae_lambda'],
        log=True,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        device='cpu',
        name='mo_ppo',
    )


    log_dir = (
            f'{agent.name}/{args.env}/{args.init_w_sampling}/k_{args.num_subproblems:04d}/ws/s_{args.seed:04d}')

    agent.train(
        total_timesteps=args.total_timesteps,
        eval_timesteps=10_000,
        ref_point=ref_point,
        heuristic=heuristic_obj,
        known_pareto_front=None,
        num_eval_weights=100,
        eval_rep=5,
        eval_seed=0,
        eval_gamma=0.99,
        save_fronts=True,
        save_models=False,
        log_dir=log_dir,
    )


if __name__ == '__main__':
    main()
