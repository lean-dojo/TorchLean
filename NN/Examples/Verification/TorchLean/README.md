# Native TorchLean Verification

These workflows lower a TorchLean model to the verifier IR and run Lean-side bound propagation:

```text
TorchLean model
  -> parameter payload
  -> verifier IR graph
  -> IBP/CROWN-style bound propagation
  -> checked margin, bound, or diagnostic report
```

Reusable workflow code belongs under `NN/Verification/Builtin`. Reusable CROWN/LiRPA data
structures, transfer rules, and proof files belong under `NN/MLTheory/CROWN`.

Run the maintained entry points through the unified verifier:

```bash
lake exe verify -- torchlean-ibp
lake exe verify -- torchlean-crown-ops
lake exe verify -- torchlean-transformer-ibp
lake exe verify -- torchlean-mlp-workflow
```

The first three commands use bundled, fixed weights. The MLP workflow trains its own small model.
In the printed reports, `lo` and `hi` are lower and upper output bounds over an input box, rather
than predictions for one input. A positive lower bound on a class margin establishes that ordering
within the checked region; a nonpositive bound leaves it unresolved. Loss bounds describe a range
of possible losses and do not establish convergence or accuracy on a dataset.

For example, the MSE case in `torchlean-crown-ops` varies each coordinate of `(0.3, -0.4)` by at most
`0.05`. Its native run reports a loss enclosure of approximately `[0.234125, 0.345425]`.
The transformer command runs IBP by default; `--with-crown` enables the slower experimental CROWN
paths. Check each algorithm's report: an unsupported-path diagnostic is not a successful bound.
Its LayerNorm interval follows the rounded normalization operations and can be very wide because
the interval calculation loses relationships between repeated values. A large upper bound means
this analysis is loose; it does not mean the model actually attains that loss.

Implementation map:

- `torchlean-ibp`: `NN.Verification.Builtin.IBPWorkflow`
- `torchlean-crown-ops`: `NN.Verification.Builtin.CrownOpsWorkflow`
- `torchlean-transformer-ibp`: `NN.Verification.Builtin.TransformerIBPWorkflow`
- `torchlean-mlp-workflow`: trains a classifier, then calls
  `trained.verify center (radius := 0.10) (norm := .inf) (property := .topLabel 0)`
  with the default `.alphaBetaCrown` method, which is fixed-relaxation alpha-CROWN with ReLU phases
  taken from IBP (printed as `Alpha-CROWN (IBP phases)`)

The `Proved/` subtree contains theorem-backed lowering and evaluator fragments. Runtime reports and
checker results remain separate from those theorems.
