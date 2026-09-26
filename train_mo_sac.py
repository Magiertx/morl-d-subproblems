import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heuristics.dynamic_heuristics import (
    RoundRobinHeuristic,
    RandomAllocationHeuristic,
    RankBasedHeuristic,
    BanditHeuristic,
    RottingBanditHeuristic,
    RoundRobinEarlyStoppingHeuristic,
    MLFQHeuristic,
    MarginalValueHeuristic,
    ProportionalShareHeuristic,
    CMuRuleHeuristic,
    SuccessiveHalvingHeuristic,
    FixedOrderHeuristic,
    FixedShareHeuristic,
    SELECTABLE_SIGNAL_KEYS,
)

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import numpy as np
import mo_gymnasium as mo_gym

from agents.multi_policy.mo_sac import MOSAC
from misc.utils import read_env_config, read_algo_config


def _kwargs_suffix(kwargs: dict) -> str:
    """Kompaktes, dateinamen-sicheres Suffix aus den Heuristik-Argumenten.

    Ohne Suffix wuerden Laeufe desselben (env, k, seed, Heuristik) mit
    unterschiedlichen Konstanten dieselbe history-/fronts-Datei
    ueberschreiben. Erlaubt sind nur [A-Za-z0-9._-]; das Trennzeichen '@'
    kollidiert nicht mit dem Signal-Trenner ':'.
    """
    if not kwargs:
        return ''
    import re
    parts = []
    for key in sorted(kwargs):
        val = kwargs[key]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            # '+' des Exponenten zu 'p': sonst bilden 1e+06 und 1e-06 nach der
            # Zeichenbereinigung dasselbe Suffix und damit denselben Dateinamen.
            # Vorzeichen explizit kodieren: '.strip("-")' unten wuerde ein
            # fuehrendes Minus entfernen, -0.05 und 0.05 ergaeben dasselbe
            # Label und damit denselben Dateinamen.
            text = f'{val:g}'.replace('+', 'p')
            text = ('m' + text[1:]) if text.startswith('-') else text
        elif isinstance(val, (list, tuple)):
            # Listenelemente durch denselben Pfad schicken, sonst fehlt
            # ihnen die Vorzeichen-/Exponentenbehandlung.
            def _one(v):
                t = f'{v:g}'.replace('+', 'p') if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
                return ('m' + t[1:]) if t.startswith('-') else t
            text = ','.join(_one(v) for v in val)
        else:
            # Symmetrie zu _one(): auch ein String mit fuehrendem Minus darf
            # nicht auf dieselbe Form wie sein positives Gegenstueck fallen.
            text = str(val)
            text = ('m' + text[1:]) if text.startswith('-') else text
        # Nicht erlaubte Zeichen ERSETZEN statt loeschen: bei Listenwerten
        # (z. B. shares='0.5,0.2') wuerden geloeschte Kommas verschiedene
        # Vektoren auf dasselbe Label abbilden. Key und Wert werden getrennt,
        # sonst liest sich 'orde'+'random' als 'orderandom'.
        text = re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('-')
        parts.append(f'{key[:4]}-{text}')
    return '@' + '_'.join(parts)


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
    parser.add_argument('--eval_timesteps', type=int, default=10_000,
                        help='Chunk size in env steps between heuristic decisions / evaluations '
                             '(SAC is step-based, any value works; 4096 matches one MO-PPO rollout).')
    parser.add_argument('--heuristic', type=str, default='round-robin',
                        help='Budget allocation heuristic, optionally with a signal variant '
                             '"<name>:<signal_key>", e.g. "bandit:prob_improvements". '
                             'Names: round-robin, random, rank-based, bandit, rotting-bandit, '
                             'rr-early-stopping, mlfq, marginal-value, '
                             'proportional-share, cmu-rule. '
                             f'Signal keys: {", ".join(SELECTABLE_SIGNAL_KEYS)}.')

    parser.add_argument('--heuristic_kwargs', type=str, default='',
                        help='JSON-Dict mit Konstruktor-Argumenten der Heuristik, z. B. '
                             '\'{"exploration_constant": 5.45e-4}\' oder \'{"order": "block"}\'. '
                             'Die Werte landen im Heuristik-Label (und damit in Dateinamen '
                             'und der history-Spalte), damit Varianten unterscheidbar bleiben.')

    args = parser.parse_args()

    heuristic_map = {
        'round-robin': (RoundRobinHeuristic, {}),
        'random': (RandomAllocationHeuristic, {}),
        'rank-based': (RankBasedHeuristic, {}),
        'bandit': (BanditHeuristic, {'exploration_constant': 1.0}),
        'rotting-bandit': (RottingBanditHeuristic, {}),
        'rr-early-stopping': (RoundRobinEarlyStoppingHeuristic, {}),
        'mlfq': (MLFQHeuristic, {}),
        'marginal-value': (MarginalValueHeuristic, {}),
        'proportional-share': (ProportionalShareHeuristic, {}),
        'cmu-rule': (CMuRuleHeuristic, {}),
        # Heuristiken der Nebenexperimente - NICHT Teil der 18
        # Konfigurationen der Hauptmatrix, werden getrennt ausgewertet.
        'successive-halving': (SuccessiveHalvingHeuristic, {}),
        'fixed-order': (FixedOrderHeuristic, {}),
        'fixed-share': (FixedShareHeuristic, {}),
    }
    # Heuristics whose identity IS their signal — no ":<signal_key>" variant.
    # cmu-rule laeuft fest mit c=dominance_ranks, mu=prob_improvements
    # (einzige Multi-Signal-Heuristik).
    fixed_signal = ('round-robin', 'random', 'mlfq', 'cmu-rule',
                    'successive-halving', 'fixed-order', 'fixed-share')

    h_name, _, h_signal = args.heuristic.partition(':')
    if h_name not in heuristic_map:
        parser.error(f"Unknown heuristic '{h_name}'. Valid: {', '.join(heuristic_map)}")
    h_cls, h_kwargs = heuristic_map[h_name]
    if h_signal:
        if h_name in fixed_signal:
            parser.error(f"Heuristic '{h_name}' does not support a signal variant.")
        h_kwargs = {**h_kwargs, 'signal_key': h_signal}
    # Freie Konstruktor-Argumente (Konstanten-Sweeps, Nebenexperimente).
    extra_kwargs = {}
    if args.heuristic_kwargs:
        import json as _json
        try:
            extra_kwargs = _json.loads(args.heuristic_kwargs)
        except ValueError as exc:
            parser.error(f'--heuristic_kwargs ist kein gueltiges JSON: {exc}')
        if not isinstance(extra_kwargs, dict):
            parser.error('--heuristic_kwargs muss ein JSON-Objekt sein.')
        h_kwargs = {**h_kwargs, **extra_kwargs}

    try:
        heuristic_obj = h_cls(**h_kwargs)
    except (TypeError, ValueError) as exc:
        # Frueh und sauber scheitern: sonst stirbt der Lauf erst auf dem
        # zugeteilten Rechenknoten mit einem Traceback.
        parser.error(f"Heuristik '{h_name}' mit diesen Argumenten nicht "
                     f"konstruierbar: {exc}")
    # Label lands in the history.csv 'heuristic' column so signal variants
    # stay distinguishable in the analysis (read by the orchestrator).
    heuristic_obj.label = (h_cls.__name__.replace('Heuristic', '')
                           + (f':{h_signal}' if h_signal else '')
                           + _kwargs_suffix(extra_kwargs))
    # Successive Halving braucht das Gesamtbudget fuer seinen Rung-Plan; ohne
    # diese Angabe faellt es auf reines Round-Robin zurueck. Ein explizit per
    # --heuristic_kwargs gesetzter Wert hat Vorrang, sonst wuerde das Label
    # einen Wert ausweisen, der nie gewirkt hat.
    if 'total_budget' not in extra_kwargs:
        heuristic_obj.total_budget = args.total_timesteps
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_config = read_env_config(os.path.join(base_dir, 'configs', 'environment_configs.json'))
    env_id = env_config[args.env]['env_id']
    ref_point = env_config[args.env]['ref_point']

    env = mo_gym.make(env_id, max_episode_steps=args.max_episode_steps)
    eval_env = mo_gym.make(env_id, max_episode_steps=args.max_episode_steps)
    ref_point = np.array(ref_point)

    config = read_algo_config(os.path.join(base_dir, 'configs', 'multi_policy', 'mo_sac.json'))

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
        log=True,
        seed=args.seed,
        device='auto',
        name='mo_sac'
    )

    log_dir = f'{agent.name}/{args.env}/{args.init_w_sampling}/k_{args.num_subproblems:04d}/ws/s_{args.seed:04d}'

    # Heuristik-Label im Dateinamen, damit Laeufe verschiedener Heuristiken mit
    # gleichem (env, k, seed) ihre fronts/*.npz + config nicht ueberschreiben
    # (Windows-sicher: ':' ist in Dateinamen verboten -> '-').
    file_name = f"{agent.name}__{heuristic_obj.label.replace(':', '-')}"

    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        heuristic=heuristic_obj,
        eval_timesteps=args.eval_timesteps,
        known_pareto_front=None,
        num_eval_weights=100,
        eval_rep=5,
        eval_seed=0,
        eval_gamma=0.99,
        save_fronts=True,
        save_models=False,
        log_dir=log_dir,
        file_name=file_name,
        log_verbose=0
    )


if __name__ == "__main__":
    main()
