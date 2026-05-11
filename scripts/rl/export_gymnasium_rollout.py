#!/usr/bin/env python3
"""Collect a Gymnasium rollout and write TorchLean-compatible JSON.

The output format is accepted by:
  `NN.Runtime.RL.Boundary.loadRollout`

This script is a small exporter rather than a trainer. It moves data across the trust boundary:

  Python Gymnasium env -> JSON -> Lean contract check -> TorchLean RL update code

Schema (top-level):
{
  "meta": { "env_id": "...", "seed": 0, "steps": 128 },
  "transitions": [
    { "obs": ..., "action": 0, "reward": 0.0, "terminated": false, "truncated": false,
      "next_obs": ... },
    ...
  ]
}

Notes:
- Only supports discrete action spaces (`gymnasium.spaces.Discrete`).
- Observation must be JSON-serializable as nested numeric arrays (e.g. Box observations).
- ALE environments are registered automatically for `ALE/...` ids when `ale-py` is installed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_jsonable_obs(obs: Any) -> Any:
    """Convert observations into JSON arrays/scalars for the Lean rollout loader."""
    # Numpy arrays expose `.tolist()`. Python lists/tuples are already jsonable.
    if hasattr(obs, "tolist"):
        return obs.tolist()
    if isinstance(obs, (list, tuple)):
        return [_to_jsonable_obs(x) for x in obs]
    if isinstance(obs, (int, float, bool)) or obs is None:
        return obs
    raise TypeError(f"Unsupported observation type for JSON export: {type(obs)}")


@dataclass(frozen=True)
class Transition:
    """One transition in the schema consumed by `NN.Runtime.RL.Boundary.loadRollout`."""

    obs: Any
    action: int
    reward: float
    terminated: bool
    truncated: bool
    next_obs: Any


def main() -> None:
    """Collect random-policy Gymnasium transitions and write the rollout JSON file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="CartPole-v1", help="Gymnasium environment id")
    ap.add_argument(
        "--make-kwargs",
        default=None,
        help="JSON object of kwargs passed to gym.make(env_id, **kwargs), e.g. '{\"obs_type\":\"ram\"}'",
    )
    ap.add_argument("--steps", type=int, default=256, help="Number of environment steps to record")
    ap.add_argument("--seed", type=int, default=0, help="Reset seed")
    ap.add_argument("--out", default="gym_rollout.json", help="Output JSON path")
    args = ap.parse_args()

    import gymnasium as gym

    if args.env_id.startswith("ALE/"):
        try:
            import ale_py  # type: ignore

            gym.register_envs(ale_py)
        except Exception as e:
            raise RuntimeError(
                "Requested an `ALE/...` environment id, but `ale-py` could not be imported/registered. "
                "Try: python3 -m pip install --user ale-py 'gymnasium>=1.0'"
            ) from e

    make_kwargs: dict[str, Any] = {}
    if args.make_kwargs is not None:
        parsed = json.loads(args.make_kwargs)
        if not isinstance(parsed, dict):
            raise TypeError("--make-kwargs must be a JSON object")
        make_kwargs = parsed

    env = gym.make(args.env_id, **make_kwargs)
    try:
        if not hasattr(env, "action_space"):
            raise RuntimeError("Env has no action_space")

        # This bridge supports discrete action spaces.
        if env.action_space.__class__.__name__ != "Discrete":
            raise RuntimeError(f"Only Discrete action spaces are supported, got {env.action_space}")

        obs, _info = env.reset(seed=args.seed)
        transitions: list[Transition] = []

        for _t in range(args.steps):
            # This exporter samples actions randomly. If an episode ends, it
            # resets the environment but keeps collecting until the requested
            # step count is reached, giving Lean a fixed-size rollout fixture.
            action = int(env.action_space.sample())
            next_obs, reward, terminated, truncated, _info = env.step(action)
            transitions.append(
                Transition(
                    obs=_to_jsonable_obs(obs),
                    action=action,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    next_obs=_to_jsonable_obs(next_obs),
                )
            )
            obs = next_obs
            if terminated or truncated:
                obs, _info = env.reset()

        payload = {
            "meta": {"env_id": args.env_id, "seed": args.seed, "steps": args.steps},
            "transitions": [
                {
                    "obs": tr.obs,
                    "action": tr.action,
                    "reward": tr.reward,
                    "terminated": tr.terminated,
                    "truncated": tr.truncated,
                    "next_obs": tr.next_obs,
                }
                for tr in transitions
            ],
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {len(transitions)} transitions to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
