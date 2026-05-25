import torch
import numpy as np
import mo_gymnasium as mo_gym
from copy import deepcopy

from agents.single_policy.ppo.sample import (Sample)
from agents.single_policy.ppo.a2c_ppo.envs import make_vec_envs
from agents.single_policy.ppo.a2c_ppo.storage import RolloutStorage
from agents.single_policy.ppo.a2c_ppo.utils import update_linear_schedule


def evaluation(
        sample: Sample,
        env_id: str,
        reward_dim: int,
        eval_num: int,
        eval_seed: int,
        eval_gamma: float,
        max_episode_steps: int
) -> np.ndarray:
    """
    Run a deterministic evaluation of `sample.actor_critic` in a single-copy gym environment.
    """
    env = mo_gym.make(env_id, max_episode_steps=max_episode_steps)
    actor_critic = sample.actor_critic
    actor_critic.training = False
    ob_rms = sample.env_params.get('ob_rms', None)

    total_obj = np.zeros(reward_dim, dtype=float)

    with torch.no_grad():
        for i in range(eval_num):
            seed_i = eval_seed + i
            env.seed = seed_i
            obs, _ = env.reset(seed=seed_i)
            done = False
            discounted_gamma = 1.0

            while not done:
                # Normalize observation if observation-normalization is enabled
                if ob_rms is not None:
                    obs = np.clip(
                        (obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8),
                        -10.0,
                        10.0
                    )
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                _, action_tensor, _ = actor_critic.act(obs_tensor, deterministic=True)
                action = action_tensor.cpu().numpy().squeeze()

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_obj += discounted_gamma * reward
                discounted_gamma *= eval_gamma
                obs = next_obs
    actor_critic.training = True
    env.close()
    return total_obj / eval_num

def ppo_worker(sample_id: int,
               sample,
               device,
               start_iteration: int,
               end_iteration: int,
               total_iteration: int,
               env_id: str,
               seed: int,
               num_processes: int,
               num_steps: int,
               gamma: float,
               obj_rms: bool,
               ob_rms: bool,
               reward_dim: int,
               use_linear_lr_decay: bool,
               lr_decay_ratio: float,
               lr: float,
               use_gae: bool,
               gae_lambda: float,
               use_proper_time_limits: bool,
               max_episode_steps: int,
               eval_rep: int,
               eval_seed: int,
               eval_gamma: float):
    """
    Pool-friendly worker. Returns the result dict directly instead of
    pushing to a queue. Long-lived: called many times across the same
    Pool process, so the Python interpreter and torch state are reused.
    """
    torch.set_default_dtype(torch.float64)
    env_params, actor_critic, agent, weights = (
        sample.env_params, sample.actor_critic, sample.agent, sample.weights)

    envs = make_vec_envs(
        env_name=env_id, seed=seed, num_processes=num_processes,
        gamma=gamma, log_dir=None, device=device,
        allow_early_resets=False, obj_rms=obj_rms, ob_rms=ob_rms,
        multiprocessing_envs=False)

    if env_params['ob_rms'] is not None:
        envs.venv.ob_rms = deepcopy(env_params['ob_rms'])
    if env_params['ret_rms'] is not None:
        envs.venv.ret_rms = deepcopy(env_params['ret_rms'])
    if env_params['obj_rms'] is not None:
        envs.venv.obj_rms = deepcopy(env_params['obj_rms'])

    rollouts = RolloutStorage(
        num_steps=num_steps, num_processes=num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=1, reward_dim=reward_dim)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    offspring_list = []

    for j in range(start_iteration, end_iteration):
        torch.manual_seed(start_iteration + j + sample_id)
        if use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j * lr_decay_ratio,
                                   total_iteration, lr)

        for step in range(num_steps):
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step])
            obs, reward, done, infos = envs.step(action)
            obj_tensor = torch.zeros([num_processes, reward_dim])

            for idx, (d, info) in enumerate(zip(done, infos)):
                step_obj = np.asarray(info['obj'], dtype=np.float64)
                obj_tensor[idx] = torch.from_numpy(step_obj)

            masks = torch.FloatTensor(
                [[0.0] if d_ else [1.0] for d_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info else [1.0]
                 for info in infos])
            rollouts.insert(obs, 1, action, action_log_prob, value,
                            obj_tensor, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        rollouts.compute_returns(next_value, use_gae, gamma,
                                 gae_lambda, use_proper_time_limits)

        obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None

        agent.update(rollouts, weights, obj_rms_var)

        rollouts.after_update()
        env_params = {
            'ob_rms':  deepcopy(envs.ob_rms)  if envs.ob_rms  is not None else None,
            'ret_rms': deepcopy(envs.ret_rms) if envs.ret_rms is not None else None,
            'obj_rms': deepcopy(envs.obj_rms) if envs.obj_rms is not None else None,
        }

    # ── Evaluation after the chunk ──────────────────────────────────
    sample_out = Sample(env_params, deepcopy(actor_critic), deepcopy(agent),
                        deepcopy(weights), sample.learning_rate, sample.eps)
    disc_obj = evaluation(sample_out, env_id, reward_dim,
                          eval_rep, eval_seed, eval_gamma, max_episode_steps)
    sample_out.objs = disc_obj
    offspring_list.append(sample_out)

    envs.close()

    return {
        'task_id':         sample_id,
        'offspring_batch': np.array(offspring_list),
    }