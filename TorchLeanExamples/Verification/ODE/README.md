# ODE corridor certificates

This directory verifies corridor certificates for a scalar ODE `u' = f(t, u)`, following Tanaka and
Yatabe, *Learn and Verify* (arXiv:2601.19818). A corridor gives lower and upper functions around a
proposed solution. Our checker reads both corridor networks from PyTorch JSON weights and bounds
them and their tangents along `dt = 1`, read as time derivatives, with IBP on time boxes. It
evaluates the right-hand side on those boxes and subdivides time until every box passes or a depth
or width limit stops it. Over `ℝ`, the theorems state what the corridor inequalities give: a
solution of the clamped ODE stays in the corridor and so solves the original ODE.

- `AST.lean`: right-hand-side expressions with interval endpoint pairs over `Context α`.
- `Parse.lean`: the right-hand-side language, with `t`, `u`, `pi`, division, `sin`, `cos`, `exp`
  and `log`.
- `ArithParse.lean`: the arithmetic parser of `Parse.lean` and the PINN residual parser, for any
  expression type that gives its constructors and identifiers.
- `TextCursor.lean`: `TextCursor`, its bounded scans, `parseNat` and `parseFloatBounds`, which
  `ArithParse.lean` reads tokens with.
- `Verify.lean`: the executable corridor checker behind `verify -- ode`.
- `Enclosure.lean`: `local_solution_of_clamped` on `[0, T]` and its constant
  extension after `T`, `extended_solution_of_clamped`.
- `EnclosureBackends.lean`: `decoded_eq_of_continuousOn`, which proves that a decoded binary
  trajectory continuous on an interval is constant there.

These theorems assume a continuous solution with the stated derivative and don't construct one or
prove existence or uniqueness. Sampled trajectories enter through real-valued interpolants.
Right-hand-side evaluation rounds outward: constants through their binary64 bounds, sums and
differences through `BoundOps`, products through `Graph.intervalMul`, and the other functions
through `NonlinearBoundOps`. No theorem states that the evaluator encloses the exact-real
expression. No theorem connects the checker to the corridor theorems: that needs interval
soundness, agreement with the real corridor functions, and a theorem relating a tangent graph to a
derivative. Training and the other producers of the corridor networks are outside the proofs.

## Certificates

- `sample_ode_cert.json`: the constant zero solution of `u′ = 0`, through a one-layer MLP. It
  passes with its pinned IEEE arithmetic. Start here.
- `sin_cert.json`: the same zero function through a sine-activation network. It passes with
  `--arithmetic ieee`, while native outward rounding can't close the zero-slack initial bound.
- `logistic_trivial_cert.json`: the coarse corridor `0 ≤ u ≤ 1` for `u′ = u(1-u)`, starting at
  `0.1`. It passes with `--model=direct --arithmetic ieee`, while the native graph path can't
  close its zero-slack differential inequality.

The verifier checks each segment as its own initial-value problem, from its `init` interval at
`t0`, and nothing requires a segment to start where the previous one ends.

`zero_mlp.json`, `one_mlp.json` and `zero_siren.json` encode the constant functions that the three
certificates use. Run the certificates from the repository root:

```bash
scripts/lake.sh exe verify -- ode --cert=TorchLeanExamples/Verification/ODE/sample_ode_cert.json
scripts/lake.sh exe verify -- ode --arithmetic ieee \
  --cert=TorchLeanExamples/Verification/ODE/sin_cert.json
scripts/lake.sh exe verify -- ode --model=direct --arithmetic ieee \
  --cert=TorchLeanExamples/Verification/ODE/logistic_trivial_cert.json
```

When a certificate declares `settings.arithmetic`, the verifier uses that setting unless the command
supplies `--arithmetic`, and `--model` overrides `settings.modelBackend` the same way. The `ieee`
arithmetic runs the direct backend in binary32, with the weights cast from binary64, and
`--model=torchlean` needs `--arithmetic native`. These commands keep zero slack. Switching
arithmetic or backend changes the computation that judges the candidate, so cite that choice when
reporting the result. A rejection means these bounds didn't establish the required inequality, and
by itself it isn't a counterexample to the ODE claim.

Without `--cert`, the command takes one segment inline: `--rhs`, `--t0`, `--t1`, `--init`, and
`--lower` and `--upper` for the two weight files, with `--maxDepth` (18), `--minWidth` (1e-3),
`--slack` (0) and `--verbose`.

Each input box of the time partitioner is seeded from its endpoints, which avoids reconstructing
them from a rounded center and radius. A failed interval is split recursively until every piece
passes, the depth or width limit is reached, or no representable split point lies strictly inside
it, and the final diagnostic names the condition that stopped it. An initial-time failure concerns a
single point, so subdividing later time intervals can't repair it.

To use a theorem, import its module, for example the corridor enclosure:

```lean
import TorchLeanExamples.Verification.ODE.Enclosure
```
