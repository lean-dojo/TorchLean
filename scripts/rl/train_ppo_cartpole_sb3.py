#!/usr/bin/env python3
"""Stable-Baselines3 PPO baseline for Gymnasium CartPole.

This is a pragmatic baseline for TorchLean:
  - Fast, well-tested training loop (Python).
  - Useful to sanity-check environment + target performance ("solved" CartPole) while TorchLean’s
    Lean-side PPO/GAE definitions and boundary contracts evolve.

Run:
  python3 -m pip install --user 'gymnasium>=1.0' stable-baselines3
  python3 scripts/rl/train_ppo_cartpole_sb3.py

References:
  - Gymnasium API semantics: https://gymnasium.farama.org/
  - Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
  - Stable-Baselines3 PPO docs: https://stable-baselines3.readthedocs.io/
"""

from __future__ import annotations

from argparse import ArgumentParser


def main() -> int:
    """Train a small SB3 PPO CartPole policy for comparison runs."""
    ap = ArgumentParser(description="Train an SB3 PPO CartPole baseline for comparison.")
    ap.add_argument("--env-id", default="CartPole-v1")
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8, help="Vectorized environments for speed")
    args = ap.parse_args()

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    # Vectorized env: faster collection, standard SB3 practice.
    env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed)
    # Separate evaluation env, wrapped with Monitor by SB3's helper.
    eval_env = gym.make(args.env_id)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    print(f"mean_reward={mean_reward:.1f} std_reward={std_reward:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
