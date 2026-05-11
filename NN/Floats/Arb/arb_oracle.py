#!/usr/bin/env python3
"""
Arb (ball arithmetic) oracle for TorchLean.

This script is called from Lean via `IO.Process.output` (see `NN/Floats/Arb/Oracle.lean`).

What Arb gives you (mathematically)
----------------------------------
Arb represents a real number as a *ball* (midpoint ± radius) and guarantees the true value lies
inside the ball (validated numerics / directed rounding done internally). When you evaluate a
function using Arb operations, the result is a new enclosing ball.

This supports:
  - tight, *rigorous* enclosures of nonlinear functions,
  - cross-checking other numeric backends,
  - building external certificates that Lean can later check.

Trust boundary
--------------
This is an *external oracle* unless you also check its outputs in Lean:
  Lean → (Python) → python-flint → (C) Arb/FLINT.

Do NOT treat the returned bounds as trusted facts in proofs without a Lean-side checker.

Scope (what this script does today)
-----------------------------------
The oracle supports three "kinds" of queries:

  1) `unary` (default): a single unary function on a real interval `[lo, hi]`.
     This is the fast path used by the current Lean demo wrappers.

  2) `expr`: evaluate a small, safe JSON expression language over *balls*.
     This is a stepping stone toward "arbitrary functions" without using `eval`.

  3) `mlp`: evaluate a simple feedforward MLP described by JSON (weights/biases/activations),
     using Arb ball arithmetic for every scalar operation.

Important limitation: Arb's `arb` is a *scalar ball*. For multi-dimensional boxes, we treat each
dimension as an independent ball and propagate through the network; this is sound but can be loose
due to dependency/correlation effects (just like naive interval arithmetic).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable


def _require_python_flint():
    try:
        from flint import arb, ctx  # type: ignore
    except Exception as e:  # pragma: no cover
        msg = (
            "arb_oracle.py: missing dependency: python-flint (FLINT/Arb).\n"
            "Install with:\n"
            "  python3 -m pip install -U python-flint\n"
            "\n"
            f"Import error: {e}\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    return arb, ctx


@dataclass(frozen=True)
class MidRad10Exp:
    mid: str
    rad: str
    exp: str


def _mid_rad_10exp_str(x, digits: int) -> MidRad10Exp:
    mid, rad, exp = x.mid_rad_10exp(digits)
    return MidRad10Exp(mid=str(mid), rad=str(rad), exp=str(exp))

def _unary_ops() -> dict[str, Callable[[Any], Any]]:
    """
    Operation table for supported unary functions.

    We build this as a table (rather than a big if/else chain) so it is easy to extend while
    keeping the set of allowed operations explicit and safe.
    """

    def sigmoid(x):
        # 1 / (1 + exp(-x)) implemented using Arb primitives.
        one = (x - x) + 1
        return one / (one + (-x).exp())

    return {
        "tanh": lambda x: x.tanh(),
        "exp": lambda x: x.exp(),
        "log": lambda x: x.log(),
        "sqrt": lambda x: x.sqrt(),
        "sin": lambda x: x.sin(),
        "cos": lambda x: x.cos(),
        "sigmoid": sigmoid,
    }


def _binary_ops() -> dict[str, Callable[[Any, Any], Any]]:
    """
    Safe table of binary operations for `expr` evaluation.

    Note: `min/max` are not provided because ordering between balls can be undecidable when
    intervals overlap. If you need piecewise ops, encode them as separate certificates and check
    the case split in Lean.
    """

    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
    }


def _ball_interval(lo, hi):
    """Return the symmetric Arb ball equal to the convex hull of the endpoints."""
    if hi < lo:
        lo, hi = hi, lo
    return lo.union(hi)


def _unary_interval_enclosure(func: str, lo, hi, *, strategy: str) -> Any:
    """
    Enclose f([lo,hi]) as an Arb ball.

    Strategies:
      - `endpoints` (preferred for monotone functions): evaluate endpoints and take the hull.
      - `ball`: build the hull ball X=conv(lo,hi) and evaluate f(X) directly.

    Endpoint evaluation is tighter for monotone functions but not correct for non-monotone ones.
    """
    ops = _unary_ops()
    if func not in ops:
        raise ValueError(f"unsupported func: {func}")

    if hi < lo:
        lo, hi = hi, lo

    if strategy == "endpoints":
        ylo = ops[func](lo)
        yhi = ops[func](hi)
        return ylo.union(yhi)

    if strategy == "ball":
        x = lo.union(hi)
        return ops[func](x)

    raise ValueError(f"unsupported strategy: {strategy}")


def _eval_expr(node: Any, env: dict[str, Any]) -> Any:
    """
    Evaluate a JSON expression node into an Arb ball, using only a small whitelisted language.

    Grammar (informal):

      node :=
        {"var": "<name>"}
      | {"const": "<decimal string>"}
      | {"op": "<unary-op>", "args": [node]}
      | {"op": "<binary-op>", "args": [node, node]}
      | {"op": "neg", "args": [node]}

    All evaluation is done over Arb balls (not exact intervals); this is sound but can be looser
    than interval/affine methods because correlations are lost.
    """
    if not isinstance(node, dict):
        raise ValueError("expr node must be an object")

    if "var" in node:
        name = node["var"]
        if name not in env:
            raise ValueError(f"unknown variable: {name}")
        return env[name]

    if "const" in node:
        # Constants are passed in as exact decimal strings and converted to exact balls by Arb.
        return env["__arb__"](str(node["const"]))

    op = node.get("op")
    args = node.get("args", [])
    if not isinstance(args, list):
        raise ValueError("expr args must be a list")

    if op == "neg":
        if len(args) != 1:
            raise ValueError("neg expects 1 arg")
        return -_eval_expr(args[0], env)

    unary = _unary_ops()
    if op in unary:
        if len(args) != 1:
            raise ValueError(f"{op} expects 1 arg")
        return unary[op](_eval_expr(args[0], env))

    binary = _binary_ops()
    if op in binary:
        if len(args) != 2:
            raise ValueError(f"{op} expects 2 args")
        return binary[op](_eval_expr(args[0], env), _eval_expr(args[1], env))

    raise ValueError(f"unsupported op: {op}")


def _eval_mlp(request: dict[str, Any], arb_ctor) -> list[Any]:
    """
    Evaluate an MLP with interval (box) input using Arb *ball arithmetic*.

    Input JSON shape:
      {
        "input": {"lo": ["-0.1", ...], "hi": ["0.2", ...]},
        "layers": [
          {"W": [[...], ...], "b": [...], "act": "tanh"|"relu"|"identity"|...},
          ...
        ]
      }

    Semantics:
      - Each input coordinate i is represented as a ball conv(lo_i, hi_i).
      - Each neuron computes a ball for its pre-activation and post-activation.

    Note: this is *sound* but can be loose (dependency problem). For tighter NN bounds, use
    IBP/affine/CROWN in Lean and optionally call Arb only for hard nonlinearities.
    """
    inp = request.get("input", {})
    lo_s = inp.get("lo")
    hi_s = inp.get("hi")
    if not isinstance(lo_s, list) or not isinstance(hi_s, list) or len(lo_s) != len(hi_s):
        raise ValueError("mlp.input.lo/hi must be same-length lists")

    x = [_ball_interval(arb_ctor(str(a)), arb_ctor(str(b))) for a, b in zip(lo_s, hi_s)]
    layers = request.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("mlp.layers must be a list")

    unary = _unary_ops()

    def act_fn(name: str):
        if name == "identity":
            return lambda z: z
        if name == "relu":
            # ReLU isn't a built-in Arb primitive; define via max(0,z) approximately using balls.
            # This is a *very* coarse enclosure; for ReLU networks use your native CROWN backend.
            zero = (x[0] - x[0])  # 0 as an arb
            return lambda z: z.union(zero)  # hull of {z,0} is a safe enclosure
        if name in unary:
            return unary[name]
        raise ValueError(f"unsupported activation: {name}")

    for layer in layers:
        W = layer.get("W")
        b = layer.get("b")
        act = layer.get("act", "identity")
        if not isinstance(W, list) or not isinstance(b, list):
            raise ValueError("layer.W and layer.b must be lists")
        if len(W) != len(b):
            raise ValueError("layer.W rows must match bias length")
        if any((not isinstance(row, list) or len(row) != len(x)) for row in W):
            raise ValueError("each row of W must have length equal to input dimension")

        y: list[Any] = []
        for row, bi in zip(W, b):
            s = arb_ctor(str(bi))
            for wij, xj in zip(row, x):
                s = s + arb_ctor(str(wij)) * xj
            y.append(s)

        f = act_fn(str(act))
        x = [f(z) for z in y]

    return x


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Arb oracle (python-flint) for rigorous enclosures.")

    # Compatibility flags: the existing Lean wrappers call this `unary` mode.
    p.add_argument("--func", help="Unary function: tanh|exp|log|sqrt|sin|cos|sigmoid")
    p.add_argument("--lo", help="Lower endpoint (decimal string)")
    p.add_argument("--hi", help="Upper endpoint (decimal string)")

    # General request format used by the JSON integration path.
    p.add_argument(
        "--request",
        help="Path to JSON request (kind=unary|expr|mlp). If provided, overrides --func/--lo/--hi.",
    )

    p.add_argument("--prec-bits", type=int, default=200, help="Working precision (bits)")
    p.add_argument("--digits", type=int, default=50, help="Digits for mid_rad_10exp output")
    args = p.parse_args(argv)

    arb, ctx = _require_python_flint()

    with ctx.workprec(int(args.prec_bits)):
        digits = int(args.digits)

        # -------------------------------------------
        # Parse request
        # -------------------------------------------
        if args.request:
            with open(args.request, "r", encoding="utf-8") as f:
                req = json.load(f)
            kind = str(req.get("kind", "unary"))
        else:
            req = {"kind": "unary", "func": args.func, "lo": args.lo, "hi": args.hi}
            kind = "unary"

        if kind == "unary":
            func = req.get("func") if args.request else args.func
            lo_s = req.get("lo") if args.request else args.lo
            hi_s = req.get("hi") if args.request else args.hi
            if func is None or lo_s is None or hi_s is None:
                raise SystemExit("unary mode requires --func/--lo/--hi (or JSON request with func/lo/hi)")

            lo = arb(str(lo_s))
            hi = arb(str(hi_s))

            # Monotone endpoint evaluation is the default here because it matches the needs of NN
            # activation enclosures. For non-monotone functions, switch strategy to "ball".
            monotone = {"tanh", "exp", "log", "sqrt", "sigmoid"}
            strategy = "endpoints" if str(func) in monotone else "ball"
            y = _unary_interval_enclosure(str(func), lo, hi, strategy=strategy)

            x = _ball_interval(lo, hi)
            x_mre = _mid_rad_10exp_str(x, digits)
            y_mre = _mid_rad_10exp_str(y, digits)

            # Keep this exact output shape stable: Lean wrappers parse these keys.
            out = {
                "status": "ok",
                "tool": "arb_oracle",
                "func": str(func),
                "ctx": {"prec_bits": int(args.prec_bits), "digits": digits},
                "input": {"lo": str(lo_s), "hi": str(hi_s), "ball": x_mre.__dict__},
                "output": {
                    "ball": y_mre.__dict__,
                    "lo": y.lower().str(digits, radius=False),
                    "hi": y.upper().str(digits, radius=False),
                },
            }
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0

        if kind == "expr":
            vars_obj = req.get("vars", {})
            expr = req.get("expr")
            if not isinstance(vars_obj, dict) or expr is None:
                raise SystemExit("expr request requires {vars: {...}, expr: {...}}")

            env: dict[str, Any] = {"__arb__": arb}
            for name, iv in vars_obj.items():
                if not isinstance(iv, dict) or "lo" not in iv or "hi" not in iv:
                    raise SystemExit(f"bad var entry for {name}: expected {{lo,hi}}")
                env[str(name)] = _ball_interval(arb(str(iv["lo"])), arb(str(iv["hi"])))

            y = _eval_expr(expr, env)
            y_mre = _mid_rad_10exp_str(y, digits)
            out = {
                "status": "ok",
                "tool": "arb_oracle",
                "kind": "expr",
                "ctx": {"prec_bits": int(args.prec_bits), "digits": digits},
                "output": {
                    "ball": y_mre.__dict__,
                    "lo": y.lower().str(digits, radius=False),
                    "hi": y.upper().str(digits, radius=False),
                },
            }
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0

        if kind == "mlp":
            outs = _eval_mlp(req, arb)
            out_vec = []
            for y in outs:
                y_mre = _mid_rad_10exp_str(y, digits)
                out_vec.append(
                    {
                        "ball": y_mre.__dict__,
                        "lo": y.lower().str(digits, radius=False),
                        "hi": y.upper().str(digits, radius=False),
                    }
                )
            out = {
                "status": "ok",
                "tool": "arb_oracle",
                "kind": "mlp",
                "ctx": {"prec_bits": int(args.prec_bits), "digits": digits},
                "output": {"vector": out_vec},
            }
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0

        raise SystemExit(f"unsupported request kind: {kind}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
