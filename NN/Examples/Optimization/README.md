# Optimization Examples

A **Muon step certificate** here is an ordinary Lean proof value about one optimizer update.
It records that the chosen matrix direction has orthonormal columns, that the backend actually
returned that direction, and that the new state and parameters follow the update rule. It is not
an exported certificate file or a claim that training will succeed.

## Start with the numbers

Read the `Concrete` namespace at the start of `MuonCertificates.lean`:

```text
                  [3/5]                       [2]
direction Q =     [4/5]       parameters P =   [3]

QᵀQ = [(3/5)² + (4/5)²] = [1]

                       [2 - (1/10)(3/5)]   [97/50]
P' = P - (1/10) Q =    [3 - (1/10)(4/5)] = [73/25]
```

For a single column, `QᵀQ = I` means unit length. With several columns it additionally means that
different columns have inner product zero. The same example proves that `(1, 1)ᵀ` fails: its Gram
matrix is `[2]`.

Momentum is zero and the gradient is `Q`, so the fresh momentum buffer is already `Q`. The example
uses the identity backend only because this particular buffer is already normalized. Identity does
not turn arbitrary gradients into orthonormal directions.

| Theorem in `Concrete` | What Lean proves |
| --- | --- |
| `direction_has_exact_gram` | The proposed direction satisfies `QᵀQ = I`. |
| `unnormalized_direction_rejected` | The direction `(1, 1)ᵀ` cannot satisfy that condition. |
| `fresh_buffer_eq_direction` | This update's new momentum buffer is exactly `Q`. |
| `step_certified` | The concrete inputs satisfy the real `Optim.Muon.ExactCertifiedStep` API, with no remaining backend assumption. |
| `updated_parameters_eq` | Consuming the certificate's parameter equation yields `(97/50, 73/25)ᵀ`. |

These are proofs over exact real numbers. They establish the direction and update equations;
they do not establish decreasing loss, convergence, speed, or native/CUDA numerical agreement.

## Then read the backend examples

The remaining theorems show how a caller obtains the same kind of certificate from a backend.
They are conditional: the caller must supply the obligation in the middle column.

| Backend | Required evidence | Result |
| --- | --- | --- |
| QR | Positive QR pivots for the fresh momentum buffer | Exact `QᵀQ = I`. |
| Newton–Schulz residual check | An entrywise bound on `QᵀQ - I` | Approximate Gram certificate with the stated tolerance. |

Selecting a Newton–Schulz iteration count alone does not prove its residual is small.
The two consumers preserve those hypotheses and extract the direction and parameter equation.
For initialized-state and fixed-point variants, use the library theorems in `Optim.Muon`.

## Build and continue

```bash
lake build NN.Examples.Optimization
```

This directory contains proof tutorials, so it has no CLI or loss-curve output. Runtime users
configure Muon through `TorchLean.optim.muon.optimizer`; proof examples use `Optim.Muon`.
The reusable definitions and theorems live in `NN/MLTheory/Optimization/Muon.lean` and
`NN/MLTheory/Optimization/OptimizerLaws.lean`.
