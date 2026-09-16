# Arb Oracle Backend (`NN/Floats/Arb`)

This folder integrates Arb ball arithmetic, through python-flint and FLINT/Arb, as an external oracle
callable from Lean.

## Ball Arithmetic

- Rigorous enclosures: Arb tracks a real value as a ball `m ± r` that contains the true value
  according to Arb's interval arithmetic.
- Configurable precision: you control the working precision in bits. Higher precision often
  tightens bounds, at higher compute cost.

## Files

- `Oracle.lean`: Lean side process wrapper, request/response types, JSON parsing, and helper
  functions for invoking the oracle.
- `arb_oracle.py`: Python implementation backed by `python-flint`/FLINT/Arb.

## Trust Model

This backend runs outside Lean's kernel:

- Lean calls a Python process.
- The Python process calls Arb/FLINT.
- The result is a JSON payload that can be checked or inspected by Lean code.

An Arb response can be used in either of two ways:

- check a small certificate derived from the Arb result, or
- state explicitly that the claim depends on the Arb/python-flint oracle.

For semantics defined entirely in Lean, use:

- FloatLib's configured `ExecFloat.Binary` formats: executable encoded arithmetic, including
  binary32 (`ExecFloat.Binary 8 23`).
- `FP32` / `NF`: proof oriented rounding over `ℝ` (`NN/Floats/FP32/` and
  `FloatLib.Floats.Formats.Flocq.NF`).

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

The deep-dive comparison command is:

```bash
lake exe torchlean floats_arb_ieee_compare
```

## Supported functions

The oracle script supports unary functions (in `unary` mode):

- `tanh`, `exp`, `log`, `sqrt`, `sin`, `cos`, `sigmoid`

For monotone functions (`tanh/exp/log/sqrt/sigmoid`) we use endpoint evaluation plus union for a
tight enclosure on `[lo, hi]`. For others we fall back to evaluating on the convex-hull ball.

## Expression And MLP Requests

To handle more than single unary calls without using `eval`, the oracle also supports a small JSON
request format via `--request <file.json>`:

- `kind = "expr"`: evaluate a safe, whitelisted expression AST over Arb balls.
- `kind = "mlp"`: evaluate a simple MLP given weights/biases/activations using ball arithmetic.

These modes support exploratory checks and certificate generation. Repeated variables introduce
dependency effects, so the resulting enclosures may be looser than those from dedicated
neural-network bound methods.

See the docstring at the top of `NN/Floats/Arb/arb_oracle.py` for the exact JSON shapes.

## Integration Strategy

There are two main ways to integrate Arb into verification pipelines:

1. Certificate producer: compute bounds externally with Python/Arb, emit a structured
   certificate, and check it in Lean (small trusted checker, large untrusted solver).
2. Hybrid primitive oracle: during IBP/CROWN propagation, call Arb only for expensive
   nonlinearities (e.g. `tanh/exp/log`) while keeping linear algebra in Lean. Per-node process
   calls can be slow unless queries are batched.

This API provides the oracle call and JSON parsing. A full NN backend also requires a certificate
shape and a declared set of supported graph ops.
