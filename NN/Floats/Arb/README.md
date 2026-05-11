# Arb oracle backend (`NN/Floats/Arb`)

This folder integrates Arb ball arithmetic, through python-flint and FLINT/Arb, as an external oracle
callable from Lean.

## What you get

- Rigorous enclosures: Arb tracks a real value as a ball `m ± r` that contains the true value
  according to Arb's interval arithmetic.
- Configurable precision: you control the working precision in bits. Higher precision often
  tightens bounds, at higher compute cost.

## Trust model

This backend is not kernel reducible and should be treated as untrusted by default:

- Lean calls a Python process.
- The Python process calls Arb/FLINT.
- The result is a JSON payload that can be checked or inspected by Lean code.

If you want semantics defined inside Lean, use TorchLean's native float backends instead:

- `IEEE32Exec`: executable bit level float32 kernel (`NN/Floats/IEEEExec/`).
- `FP32` / `NF`: proof oriented rounding over `ℝ` (`NN/Floats/FP32/`, `NN/Floats/NeuralFloat/`).

## Installation

Install `python-flint` into the Python you will run from Lean:

```bash
python3 -m pip install -U python-flint
```

(Wheels exist for common CPython versions on Linux; if your default `python3` lacks wheels, use a
different Python and point Lean at it with `TORCHLEAN_ARB_PY`.)

## Running

Lean wrappers call `NN/Floats/Arb/arb_oracle.py` through `IO.Process`. You can select the Python
executable with `TORCHLEAN_ARB_PY`; otherwise the wrapper uses `python3`.

## Supported functions

The oracle script supports unary functions (in `unary` mode):

- `tanh`, `exp`, `log`, `sqrt`, `sin`, `cos`, `sigmoid`

For monotone functions (`tanh/exp/log/sqrt/sigmoid`) we use endpoint evaluation plus union for a
tight enclosure on `[lo, hi]`. For others we fall back to evaluating on the convex-hull ball.

## Beyond unary: expression / MLP requests

To handle more than single unary calls without using `eval`, the oracle also supports a small JSON
request format via `--request <file.json>`:

- `kind = "expr"`: evaluate a safe, whitelisted expression AST over Arb balls.
- `kind = "mlp"`: evaluate a simple MLP given weights/biases/activations using ball arithmetic.

These modes are intended for exploratory checks and certificate generation. They can produce useful
enclosures, but they may be looser than dedicated neural-network bound methods because repeated
variables introduce dependency effects.

See the docstring at the top of `NN/Floats/Arb/arb_oracle.py` for the exact JSON shapes.

## Integration strategy (how to use this in verification)

There are two main ways to integrate Arb into verification pipelines:

1. Certificate oracle (recommended): compute bounds externally (Python/Arb), emit a structured
   certificate, and check it in Lean (small trusted checker, large untrusted solver).
2. Hybrid primitive oracle: during IBP/CROWN propagation, call Arb only for expensive
   nonlinearities (e.g. `tanh/exp/log`) while keeping linear algebra in Lean. Per-node process
   calls can be slow unless queries are batched.

This API provides the oracle call and JSON parsing. A full NN backend also requires a certificate
shape and a declared set of supported graph ops.
