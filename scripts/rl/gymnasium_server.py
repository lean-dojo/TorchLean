#!/usr/bin/env python3
"""Gymnasium JSON-lines server used by TorchLean RL examples.

This starts one Gymnasium environment and then speaks a compact JSON-lines protocol over
stdin/stdout so a Lean process can:
  - reset the env, and
  - step the env with discrete actions.

Why a line protocol?
  - easy to debug (you can run this file directly and type JSON by hand)
  - easy to wrap with a Lean-side contract checker at the trust boundary

Protocol (one JSON object per line):
  {"cmd": "describe"}
    -> {"ok": true, "n_actions": <int>, "obs_shape": <list[int]>, "obs_dtype": <string>}

  {"cmd": "reset", "seed": 0}
    -> {"ok": true, "obs": <jsonable>}

  {"cmd": "step", "action": 0}
    -> {"ok": true, "obs": <jsonable>, "reward": 1.0,
        "terminated": false, "truncated": false}

  {"cmd": "close"}
    -> {"ok": true}

Notes:
  - Discrete action spaces (`gymnasium.spaces.Discrete`) in this bridge.
  - Observations are converted via `.tolist()` when available (NumPy arrays), else recursively
    converted for lists/tuples.

References:
  - Gymnasium API semantics (`reset`, `step`, `terminated` vs `truncated`):
    https://gymnasium.farama.org/
  - ALE registration for Atari envs:
    https://ale.farama.org/
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from typing import Any, Dict, NoReturn


JsonObject = Dict[str, Any]


def _to_jsonable_obs(obs: Any) -> Any:
    """Convert Gymnasium observations into values accepted by `json.dumps`."""
    # Numpy arrays expose `.tolist()`. Python lists/tuples are already jsonable.
    if hasattr(obs, "tolist"):
        return obs.tolist()
    if isinstance(obs, (list, tuple)):
        return [_to_jsonable_obs(x) for x in obs]
    if isinstance(obs, (int, float, bool)) or obs is None:
        return obs
    raise TypeError(f"Unsupported observation type for JSON transport: {type(obs)}")


def _write(obj: JsonObject) -> None:
    """Write one compact JSON response and flush so Lean can read it immediately."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _fail(message: str) -> NoReturn:
    """Raise a protocol error that will be reported as an `{ok:false}` response."""
    raise RuntimeError(message)


def _register_ale_if_needed(env_id: str) -> None:
    """Lazily register Atari/ALE environments only when an `ALE/...` id is requested."""
    if not env_id.startswith("ALE/"):
        return
    try:
        import ale_py  # type: ignore
        import gymnasium as gym

        gym.register_envs(ale_py)
    except Exception as e:
        raise RuntimeError(
            "Requested an `ALE/...` environment id, but `ale-py` could not be imported/registered. "
            "Try: python3 -m pip install --user ale-py 'gymnasium>=1.0'"
        ) from e


def main() -> int:
    """Run the Gymnasium environment server until stdin closes or `close` arrives."""
    ap = ArgumentParser(description="Run one Gymnasium env behind TorchLean's JSON-lines protocol.")
    ap.add_argument("--env-id", default="CartPole-v1", help="Gymnasium environment id")
    ap.add_argument(
        "--make-kwargs",
        default=None,
        help="JSON object of kwargs passed to gym.make(env_id, **kwargs), e.g. '{\"obs_type\":\"ram\"}'",
    )
    args = ap.parse_args()

    import gymnasium as gym

    # Atari/ALE environments live in `ale-py`; register them only when requested so CartPole users
    # do not need the dependency.
    _register_ale_if_needed(args.env_id)

    make_kwargs: Dict[str, Any] = {}
    if args.make_kwargs is not None:
        parsed = json.loads(args.make_kwargs)
        if not isinstance(parsed, dict):
            _fail("--make-kwargs must be a JSON object")
        make_kwargs = parsed

    env = gym.make(args.env_id, **make_kwargs)
    try:
        if env.action_space.__class__.__name__ != "Discrete":
            _fail(f"Only Discrete action spaces are supported, got {env.action_space}")

        n_actions = int(env.action_space.n)
        obs_shape = (
            list(env.observation_space.shape)
            if getattr(env.observation_space, "shape", None) is not None
            else []
        )
        obs_dtype = str(getattr(env.observation_space, "dtype", "unknown"))

        while True:
            line = sys.stdin.readline()
            if line == "":
                break  # EOF
            line = line.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise TypeError("request must be a JSON object")
                cmd = req.get("cmd")
                if cmd == "describe":
                    _write(
                        {
                            "ok": True,
                            "n_actions": n_actions,
                            "obs_shape": obs_shape,
                            "obs_dtype": obs_dtype,
                        }
                    )
                elif cmd == "reset":
                    seed = req.get("seed", None)
                    obs, _info = env.reset(seed=seed)
                    _write({"ok": True, "obs": _to_jsonable_obs(obs)})
                elif cmd == "step":
                    action = int(req["action"])
                    obs, reward, terminated, truncated, _info = env.step(action)
                    _write(
                        {
                            "ok": True,
                            "obs": _to_jsonable_obs(obs),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                elif cmd == "close":
                    _write({"ok": True})
                    break
                else:
                    raise KeyError("unknown cmd")
            except Exception as e:
                _write({"ok": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
