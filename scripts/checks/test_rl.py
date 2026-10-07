"""Check external RL bounds and the bridge protocol without optional Gym environments."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "rl/gymnasium_server.py"
SPEC = importlib.util.spec_from_file_location("gymnasium_server", SOURCE)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)

BOUNDS_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "NN/MLTheory/CROWN/Tactics/crown_verifier.py"
)
BOUNDS_SPEC = importlib.util.spec_from_file_location("crown_verifier", BOUNDS_SOURCE)
VERIFIER = importlib.util.module_from_spec(BOUNDS_SPEC)
BOUNDS_SPEC.loader.exec_module(VERIFIER)


class Discrete:
    n = 2
    start = 5

    def seed(self, value):
        pass

    def sample(self):
        return 6


class Environment:
    action_space = Discrete()
    observation_space = types.SimpleNamespace(shape=(1,), dtype="float32")

    def __init__(self):
        self.actions = []
        self.closed = False

    def reset(self, seed=None):
        return [0.0], {}

    def step(self, action):
        self.actions.append(action)
        return [1.0], 1.0, False, False, {}

    def close(self):
        self.closed = True


class GymProtocolTests(unittest.TestCase):
    def test_action_contract_and_space_offset(self):
        env = Environment()
        gym = types.SimpleNamespace(spaces=types.SimpleNamespace(Discrete=Discrete),
                                    make=lambda *a, **kw: env)
        requests = [{"cmd": "step", "action": value} for value in [0.5, True, "0", -1, 2, 0, 1]]
        requests += [{"cmd": "close"}]
        stdout = io.StringIO()
        with patch.dict("sys.modules", {"gymnasium": gym}), patch("sys.argv", ["bridge"]), \
             patch("sys.stdin", io.StringIO("\n".join(map(json.dumps, requests)))), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(BRIDGE.main(), 0)
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([r["ok"] for r in responses], [False] * 5 + [True] * 3)
        self.assertEqual(env.actions, [5, 6])
        self.assertTrue(env.closed)

    def test_rollout_uses_zero_based_action(self):
        env = Environment()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "rollout.json"
            BRIDGE.export_rollout(env, "mock", 1, 0, out)
            self.assertEqual(json.loads(out.read_text())["transitions"][0]["action"], 1)
        self.assertEqual(env.actions, [6])

    def test_nonfinite_response_not_published(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(ValueError):
            BRIDGE._write({"ok": True, "obs": [float("nan")]})
        self.assertEqual(stdout.getvalue(), "")


class PendulumBoundsTests(unittest.TestCase):
    def check_samples(self, lower, upper, samples):
        lo, hi = VERIFIER.pendulum_ibp([lower, -0.5], [upper, 0.75])
        for theta in samples:
            for omega in [-0.5, 0.0, 0.75]:
                actual = -9.81 * math.sin(theta) - 0.1 * omega
                self.assertLessEqual(lo[1], actual)
                self.assertLessEqual(actual, hi[1])
                self.assertLessEqual(lo[0], omega)
                self.assertLessEqual(omega, hi[0])

    def test_interior_extrema(self):
        self.check_samples(1.0, 2.0, [1.0, math.pi / 2, 2.0])
        self.check_samples(-2.0, -1.0, [-2.0, -math.pi / 2, -1.0])
        self.check_samples(-4.0, 4.0, [-4.0, -math.pi / 2, 0.0, math.pi / 2, 4.0])

    def test_central_branch_keeps_endpoint_precision(self):
        lo, hi = VERIFIER.pendulum_ibp([-0.25, 0.0], [0.5, 0.0])
        self.assertEqual(lo[1], -9.81 * math.sin(0.5))
        self.assertEqual(hi[1], -9.81 * math.sin(-0.25))
        self.check_samples(-0.25, 0.5, [-0.25, 0.0, 0.25, 0.5])

    def test_large_angles_use_global_range(self):
        self.check_samples(1.0e100, 2.0e100, [1.0e100, 1.5e100, 2.0e100])


if __name__ == "__main__":
    unittest.main()
