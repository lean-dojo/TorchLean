# Deep Dives

The deep dives are for readers who know the quickstart and want a closer look at TorchLean's
internal boundaries. They cover advanced autograd, float semantics, rounding failures of standard
and library formulas, sequential models, IR shape behavior, widgets, PyTorch export, graph-based
verification, and the certificate of one Muon optimizer step.

Most deep dives are single files at the top level. Three subdirectories hold the rest:

- [`Floats/`](Floats/README.md): the `Floats/` files listed below, plus `LibraryBugs.lean`,
  `LossBugs/` and `ProbBugs/`.
- `Sequential/`: `MLP.lean`, the model that `Sequential.lean` runs.
- [`Stability/`](Stability/README.md): `RidgeRegression1D.lean`.

The `TorchLeanExamples` library builds all of them together with the other examples:

```bash
scripts/lake.sh build TorchLeanExamples
```

## Files

| File | Command or action | What it shows |
| --- | --- | --- |
| `AutogradTransforms.lean` | `scripts/lake.sh exe torchlean autograd_transforms` | Jacobians, Hessians, parameter and input directions, mixed and third derivatives, gradient stopping, and mathlib correctness proofs. |
| `TensorOperations.lean` | `scripts/lake.sh build TorchLeanExamples.DeepDives.TensorOperations` | Indexing, reshaping, `einsum` (the tensor-pattern syntax needs `open TorchLean.Tensor`), elementwise operations, and tensor widgets. |
| `Floats/Float32Semantics.lean` | `scripts/lake.sh exe torchlean float32_semantics` | Native binary32 execution compared with the bit-level IEEE reference. |
| `Floats/EffectiveRounding.lean` | `scripts/lake.sh build TorchLeanExamples.DeepDives.Floats.EffectiveRounding` | Kernel-checked links between finite bit-level results and rounded-real mantissa/exponent calculations, with no CLI. |
| `Floats/GraphNumericalCertificate.lean` | `scripts/lake.sh build TorchLeanExamples.DeepDives.Floats.GraphNumericalCertificate` | Range certificates and bit-level IEEE replay for a small graph and a two-layer MLP. The exact-real enclosure needs a separate proof. |
| `Floats/RoundingBugs.lean` | `scripts/lake.sh build TorchLeanExamples.DeepDives.Floats.RoundingBugs`, and `scripts/lake.sh test` runs its Softmax Collapse checks | Softmax Collapse, the sign of the cross-entropy gradient, cancellation in the cosine loss and the Gini impurity, the probability clamp at `1 - epsilon` in cross-entropy and binary cross-entropy, and the cosine loss at a nonpositive epsilon: each standard formula, its kernel-checked counterexamples in binary32, bfloat16 or binary64, and the theorem about the stable formula. |
| `Stability/RidgeRegression1D.lean` | open in an editor | 1D ridge stability, real and binary32. |
| `Sequential.lean` | `scripts/lake.sh exe torchlean sequential` | The MLP as a sequential model. |
| `IRAxisOps.lean` | `scripts/lake.sh exe torchlean ir_axis_ops` | Middle-axis softmax, LayerNorm, and concatenation as checked graphs of registered operators (`IR`), evaluated on CPU in the scalar type that `--arithmetic` selects. |
| `TorchIRPyTorch.lean` | `scripts/lake.sh exe torchlean torch_ir_pytorch --arch mlp > exported_model.py` | Emitting readable PyTorch code from the operator graph of a lowered model. |
| `GraphSemantics.lean` | `scripts/lake.sh exe torchlean graph_semantics` | One IR graph evaluated in executable binary32, its interval bounds, and the checker's soundness theorem. |
| `WidgetGallery.lean` | open in the Lean infoview | Widget commands on small examples. |
| `Muon.lean` | `scripts/lake.sh build TorchLeanExamples.DeepDives.Muon` | A certified Muon step. |

## Topics

Each file isolates one boundary:

- `AutogradTransforms.lean` keeps higher-order and directional differentiation out of the first
  autograd tutorial.
- `TensorOperations.lean` picks up where the tensor quickstart ends and does not repeat its basics.
- `Floats/` distinguishes Lean-defined executable IEEE behavior from rounded-real proof models, and
  checks with the kernel where standard and library formulas lose their value to rounding.
- `Stability/RidgeRegression1D.lean` defines the 1D ridge closed form over any scalar, proves its
  replace-one uniform stability over `ℝ`, and runs it in binary32. In any configured IEEE format,
  when every intermediate result is finite and the divisor is nonzero, its real value equals the
  round-after-each-operation semantics. `Ridge.abs_toReal_ridgeFit_sub_ridgeFit_le` bounds the gap
  between the real and floating-point fits by the forward error bound `IEEE754.errorBound`, when
  every divisor also exceeds its error bound and the count `n + 1` converts without rounding.
- `Sequential.lean` evaluates the MLP `Linear(2, 3) → ReLU → Linear(3, 1)` as an `nn.Sequential`
  under the pure interpreter on seeded parameters, then trains it for three SGD steps.
  `Sequential/MLP.lean` proves at every scalar that the model evaluates to `linear ∘ ReLU ∘ linear`.
- `IRAxisOps.lean` builds and evaluates graphs of registered operators: softmax and cat take an
  axis bounded by the rank, and LayerNorm normalizes its last axis after a reshape.
- `TorchIRPyTorch.lean` writes the operator graph of a lowered model, the graph the verifiers
  read, as PyTorch.
- `GraphSemantics.lean` ties one graph to several semantic views and separates the executable
  checks from the Lean statements.
- `WidgetGallery.lean` renders tensors, IR graphs, Float32 rounding, CROWN bounds, autograd tapes,
  training logs and gridworld policies in the infoview for human inspection, without making the
  widget output a proof.

Reusable algorithms should not live here. If a deep dive grows into an API, move the reusable part
under `TorchLean/Tensor`, `TorchLean/NN`, `TorchLean/NN/IR` or `TorchLean/Verify`, then keep the
example as a small tour of that API.

## Muon step certificates

`Muon.lean` is a proof tutorial about one optimizer update, with no CLI or training run. A **Muon
step certificate** is an ordinary Lean proof value. It records that the chosen matrix direction has
orthonormal columns, that the backend returned that direction, and that the updated state and
parameters follow the update rule. It is neither an exported certificate file nor a claim that
training will succeed.

### A one-column certificate

The `Concrete` namespace at the start of `Muon.lean` works through one step with actual numbers:

```text
                  [3/5]                       [2]
direction Q =     [4/5]       parameters P =   [3]

QᵀQ = [(3/5)² + (4/5)²] = [1]

                       [2 - (1/10)(3/5)]   [97/50]
P' = P - (1/10) Q =    [3 - (1/10)(4/5)] = [73/25]
```

For a single column, `QᵀQ = I` means unit length. With several columns it also means that
different columns have inner product zero. The same example proves that `(1, 1)ᵀ` fails: its Gram
matrix is `[2]`.

Momentum is zero and the gradient is `Q`, so the fresh momentum buffer is already `Q`. The example
uses the identity backend only because this particular buffer is already normalized. Identity does
not turn arbitrary gradients into orthonormal directions.

| Theorem in `Concrete` | What Lean proves |
| --- | --- |
| `direction_has_exact_gram` | The proposed direction satisfies `QᵀQ = I`. |
| `unnormalized_not_exact_gram` | The direction `(1, 1)ᵀ` cannot satisfy that condition. |
| `fresh_buffer_eq_direction` | The update's fresh momentum buffer is exactly `Q`. |
| `step_certified` | The inputs meet `Muon.ExactCertifiedStep` with no backend assumption. |
| `updated_parameters_eq` | Consuming the certificate's parameter equation yields `(97/50, 73/25)ᵀ`. |

The proofs are over exact real numbers. They establish the direction and update equations and
say nothing about decreasing loss, convergence, speed, or native/CUDA numerical agreement.

### Backend obligations

Two conditional theorems follow `Concrete` and show how a caller obtains the same kind of
certificate from a backend. The caller must supply the evidence in the middle column:

| Backend | Required evidence | Result |
| --- | --- | --- |
| QR | Positive QR pivots for the fresh momentum buffer | Exact `QᵀQ = I`. |
| Newton–Schulz residual check | An entrywise bound on `QᵀQ - I` | Approximate Gram certificate with the stated tolerance. |

Choosing a Newton–Schulz iteration count doesn't by itself prove the residual small.
`qr_step_exact_gram` and `newtonSchulz_step_approx_gram` take that evidence and extract the
direction's Gram certificate and the parameter equation. For a fresh state, apply the same theorems
to the zero momentum buffer that `Muon.init` sets up. For a buffer satisfying
`Muon.NewtonSchulz.FixedPoint`, `Muon.NewtonSchulz.FixedPoint.orthogonalizer_apply` proves that the
orthogonalizer returns it unchanged.

Runtime code configures Muon through `optim.muon`, and the theorems are stated about `Muon.State`
and `Muon.update`. `TorchLean/NN/Optimizer/Update.lean` defines the state, `Muon.init` and
`Muon.update`, `TorchLean/NN/Optimizer/Muon.lean` the Gram conditions, and
`TorchLean/NN/Optimizer/Muon/` the QR and Newton–Schulz backends and the step certificates.
`TorchLean/NN/Optimizer/TensorOptimizer.lean` packages Muon as a `TensorOptimizer` and holds its
comparison laws.
