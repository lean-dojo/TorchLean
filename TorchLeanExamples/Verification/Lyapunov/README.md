# All-in-Lean Lyapunov Pipeline

This directory and `TorchLeanExamples/Verification/Lyapunov.lean` hold pipeline (iii) of the
TorchLean paper (arXiv:2602.22631, Figure 7): a 2D closed-loop system with a learned controller and
a learned Lyapunov candidate. Both training stages and the final IBP/CROWN bounds on the loss run in
Lean under `Binary32`. Stage 1 trains on random states. Stage 2 searches for states of positive loss
by projected gradient ascent and trains on the states it finds. The certificate type `Lyapunov.Cert`
and its sign consequences live here too.

## Files

- `Certificate.lean`: `Lyapunov.Cert` (a box `region` and the bounds `vLower`, `vUpper`,
  `derivativeLower` and `derivativeUpper`), `Lyapunov.Candidate` (the functions `value` and
  `orbitalDerivative`) and `Cert.ValidFor`, the proposition that both functions stay within
  those bounds at every point of the region. Parsing a certificate or checking the signs of its
  endpoints can't construct `ValidFor`.
- `SignConditions.lean`: `RealCert`, a certificate with real endpoints, and `RealCert.toCert`. In
  `TorchLean.Examples.Lyapunov.Cert`, given `ValidFor` over `ℝ`, `value_pos` turns `0 < vLower`
  into `V > 0` and `orbitalDerivative_neg` turns `derivativeUpper < 0` into `V̇ < 0` at every
  point of the region, and `sign_conditions` gives both. These are sign facts, not a stability
  theorem: `Lyapunov.Candidate` doesn't tie `V̇` to any dynamics.
- `TorchLeanExamples/Verification/Lyapunov.lean`: the shapes (a 2D state and one control input), the
  parameter shapes and `lossProgram`. The controller is `u = tanh(Wc x + bc)` and the candidate is
  `V = s(x)²` for a one-hidden-layer tanh network `s`. The dynamics are controlled van der Pol,
  `f = (x₂, -x₁ + (1 - x₁²) x₂ + u)`. With `∇V` in closed form, the loss is
  `relu(|x|²/10 - V) + relu(∇V·f + V/10)`. The same program is trained, differentiated for the state
  search and lowered to an IR graph.
- `Sampling.lean`: `sampleStateTensor`, which draws a state on a 24-bit grid in `[-rad, rad]²`
  from a `UInt64` key without using `Float`, `clampStateTensor`, and the defaults: learning rate
  1/20, PGD step size 1/20, sampling radius 2 and check-box half-width 1/10.
- `LossAnalysis.lean`: `projectedGradientAscentStep`, which moves the state along the loss gradient
  with the parameters fixed and clamps it to `[-radius, radius]²`, and `reportLossBox`, which runs
  IBP and CROWN on the loss over a box, prints the IBP and CROWN bounds and decides nothing.
- `AllInLean.lean`: the run, with hidden width 100 and Xavier-uniform weights and zero
  biases from the fixed seeds 0 to 5, cast from binary64 to binary32.

## Running the pipeline

```bash
scripts/lake.sh exe verify -- twostage-torchlean-cegis-van
```

The command is not part of `verify -- all`. Its flags set the length of each stage:

| Flags | Stage-1 steps | Stage-2 rounds | PGD steps per round |
| --- | --- | --- | --- |
| none | 1 | 1 | 1 |
| `--two` | 2 | 1 | 1 |
| `--paper` | 10 | 10 | 10 |
| `--long` | 20 | 10 | 20 |

`--long` takes precedence over `--paper`, and `--paper` over `--two`. Any other argument is
ignored. The fixed seeds and a sampler key that starts at 1 make a run reproducible from the source.

A run prints a banner and `width=100 stage1Steps=<n> stage2Rounds=<n> pgdSteps=<n>`. Stage 1 prints
`[stage1] step <i>: loss=<v>` at steps 0, 5, 10 and 15 when they occur, where `<v>` is the loss
before that step's SGD update. Stage 2 prints `[stage2] round <r>: loss=<v>` for each round, where
`<v>` is the loss at the state PGD found, before training on it. The run ends with `reportLossBox`
on the box of half-width 1/10 around the origin. Its output is a header line, the number of lowered
IR nodes, `[IBP] loss lo = <v>`, `[IBP] loss hi = <v>`, and either `[CROWN] loss lo = <v>` and
`[CROWN] loss hi = <v>` or the CROWN error message.

## Relation to `Lyapunov.Cert`

The pipeline does not construct `Cert.ValidFor`, and no theorem connects the bounds it
prints to the certificate type. None of `TorchLeanExamples/Verification/Lyapunov.lean`,
`Sampling.lean`, `LossAnalysis.lean` and `AllInLean.lean` imports `Certificate.lean` or
`SignConditions.lean`. The IBP and CROWN intervals are computed in `Binary32` arithmetic for the
final parameters on one box, and the run compares neither with zero.
