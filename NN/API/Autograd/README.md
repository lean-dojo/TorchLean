# Autograd

First decide whether the derivative itself is the result you want:

| Goal | Use |
| --- | --- |
| Differentiate a function of one tensor | `autograd.grad` |
| Differentiate a model loss with respect to model state | `autograd.model.grad` |
| Pull an output gradient back to model state and input | `autograd.model.vjp` |
| Repeated or mixed input-directional derivatives | `autograd.model.derivative` |
| Parameter and input gradients through those derivatives | `autograd.model.derivativeVjp` |
| Differentiate a real loss on complex parameters | `autograd.complex.grad` |
| Update model parameters over data | `Trainer`, not a direct autograd call |

`autograd` returns values and derivatives. It does not mutate parameters or run an optimizer.
`Trainer` uses model autograd internally, then owns updates, batching, devices, and reports.

Application code imports:

```lean
import NN.API
open TorchLean
```

Lean namespace completion lists the operations themselves:

- `autograd.` exposes `grad`, `vjp`, `jacfwd`, `jacrev`, and `hessian`;
- `autograd.model.` exposes `grad`, `vjp`, `jacrev`, `jvp`, `hvp`, `derivative`,
  `derivativeVjp`, and `Loss`.

The selected operation's signature then shows its named options. For example, `grad` displays
`(value : Bool := false)`; there is no separate compound operation to discover.

## Tensor Functions

An `autograd.Function input output` is a differentiable tensor program with no
model parameters. Write its body with `nn.functional` operations, which record how inputs
produce outputs. It is not an arbitrary `Tensor → Tensor` function with a derivative inferred
from its Lean source.

```lean
def meanSquare {shape : Shape} : autograd.Function shape [] :=
  fun x => do
    let squared ← nn.functional.square x
    nn.functional.mean squared

#eval show IO Unit from do
  let x : Tensor Float [3] := [1.0, 2.0, 3.0]
  let (gradient, value) ← autograd.grad meanSquare x (value := true)
  IO.println s!"value = {value}, gradient = {reprStr gradient}"
```

`value` is a rank-zero `Tensor α []`, so it prints directly and remains in the same tensor API as
every other result. Use `Tensor.item` only at an explicit host-scalar boundary such as a log message
or loop counter decision.

Use `grad` for scalar outputs; set `(value := true)` when the same evaluation should also return
the function value. Use `vjp`, `jacrev`, `jacfwd`, or `hessian` only when the corresponding
derivative object is actually needed.

## Model Losses

`autograd.model.State model α` has the same typed tensor layout as the model's state. Differentiation
returns a new state-shaped gradient value; there is no mutable `.grad` field.

```lean
def model : nn.Sequential [2] [1] := nn.build 0 (nn.linear 2 1)

#eval show IO Unit from do
  let state : autograd.model.State model Float := autograd.model.initialState model
  let input : Tensor Float [2] := [0.5, -1.0]
  let target : Tensor Float [1] := [0.25]
  let (gradient, lossValue) ← autograd.model.grad model
    autograd.model.Loss.mse state input target (value := true)
  IO.println s!"loss = {lossValue}, gradient = {reprStr gradient}"
```

Use:

- `grad` when only the model-state gradient is needed;
- `grad (value := true)` when both the model-state gradient and loss are needed from one evaluation;
- `vjp` for state and input gradients induced by an upstream output gradient;
- `jvp` for a directional derivative through model state;
- `hvp` for a Hessian-vector product;
- `Loss.detach loss` to stop gradients through the model output.

For an upstream model-output gradient:

```lean
-- Inside a `do` block, with an output-shaped tensor `seed`:
let (gradient, inputGradient) ← autograd.model.vjp model state input seed
IO.println s!"state gradient = {reprStr gradient}"
IO.println s!"input gradient = {reprStr inputGradient}"
```

Plain `Tensor` values do not own a mutable gradient field. Session inputs default to no gradient;
model parameters default to trainable unless a layer marks them frozen.

Model loss transforms differentiate the model state because that is the quantity an optimizer
updates. They do not also differentiate the input and target implicitly. Use `vjp` when an explicit
output gradient should be pulled back to the input, or define a scalar tensor function whose chosen
argument is the differentiation input.

## Neural Fields and PDE Residuals

`autograd.model.derivative model state input directions` takes input-directional derivatives
while holding the parameter state fixed. An empty list evaluates the model, `[dx]` computes
one derivative, and `[dx, dy]` computes a mixed derivative. Directions have the model's input
shape: they need not be coordinate basis vectors, and neither dimension nor derivative order is
fixed by the API. Evaluation uses nested dual scalars rather than finite differences.

To train on a derivative, use `autograd.model.derivativeVjp` with the same arguments and an
output cotangent. It returns parameter and input gradients. For a squared PDE residual
`(u_xx - f)^2 / 2`, pass `u_xx - f` as that cotangent; the parameter result is the contribution
of this residual to the training gradient. Directions and the supplied cotangent are held
constant in the pullback. Separate boundary terms use the ordinary model VJP.

The transforms use evaluation mode and the existing scalar/graph derivative conventions.
Choose a sufficiently smooth model for the equation: a ReLU MLP's second derivative is not
a useful classical curvature model. Each call lowers the model at the required nested scalar
type; high derivative orders can be expensive.

Run a complete Lean-only example with `scripts/lake.sh exe torchlean pinn`. It trains a tanh MLP
on `u'' = -2`, `u(-1) = u(1) = 0`, without supplying solution labels. This is sampled training,
not a proof of the PDE between collocation points.

## Real Losses on Complex Parameters

Complex parameters have two real coordinates. For a real loss `L`, the update direction is
`dL/dre + i*dL/dim`; this also handles conjugation and squared magnitude, which are not complex
analytic functions. `autograd.complex.Objective shapes` describes such a loss on
`nn.State (Complex α) shapes`. Write the objective polymorphically in the component type so it
can run with dual numbers as well as ordinary scalars.

`autograd.complex.grad objective state` returns the state-shaped gradient. Pass `(value := true)`
to also return the real loss. `autograd.complex.jvp objective state direction` evaluates a chosen
direction in one pass. These operations keep both components; there is no conversion to `Float`.
An SGD update uses `nn.sgdStep model (TorchLean.Complex.ofReal rate) state gradient`. The model's
frozen-parameter flags still apply.

This implementation takes two forward directional passes per complex parameter entry. It is
useful for small models and checking gradients, not a fast complex reverse-mode engine. The
objective must give the same calculation on every pass: do not resample data or dropout masks
inside it. Branch cuts and nonsmooth points retain the scalar implementation's conventions.

The component type needs `Storage`, `Context`, and `Atan2`. Native `Float` and `Float32` work;
executable binary32 has a host-angle adapter. A wider configured binary type needs an appropriate
`Atan2` instance before it can supply the full complex `Context`. Checkpoint encoding does not
require that angle operation and supports wide complex components directly.

Run `scripts/lake.sh exe torchlean complex_regression`. This trains complex binary32 weights
and bias against a real squared-residual loss, then prints both components of a held-out prediction.
An optional step count and path save and reload the complete state:

```bash
scripts/lake.sh exe torchlean complex_regression 50 /tmp/complex-regression.state
```

`Checkpoint.State.load (α := Complex Float32) model path` rejects a different component format
or tensor layout. The ordinary supervised `Trainer` remains a real-data API; this example uses
explicit state throughout.

## Training

Ordinary training code should use `Trainer`. It runs model autograd internally and adds the parts
that are not differentiation:

- datasets and batching;
- optimizer and scheduler state;
- eager or typed-graph execution;
- device selection;
- checkpoints, logs, and reports.

Direct `autograd.model` calls are most useful for custom objectives, gradient inspection, and
derivative tests.

## Runtime Domains

Raw `Runtime.log` requires positive inputs. Public autograd transforms check this condition and
raise an `IO` error for nonpositive values or NaNs, including derivative-only calls and empty
Jacobian results. `Runtime.safeLog` applies the explicit epsilon-protected operation instead.
This runtime policy does not change the mathematical scalar backend's totalized `log` definition.

## What The Types Guarantee

Input, output, target, parameter, and derivative shapes are tracked in Lean types. A returned model
gradient has the model state's tensor layout.

Executing an autograd program computes floating-point values; execution alone is not a theorem that
the result equals a real-analysis derivative. The corresponding semantic and runtime-correctness
proofs live under `NN/Proofs/Autograd`.

For exact-real graphs, `TypedGraphWithData.jvpChecked_fderiv` and
`TypedGraphWithData.vjpChecked_adjoint_fderiv` connect successful checked execution to mathlib's
`fderiv` of the selected forward output. They require a proof graph describing the same node data
and derivative-correctness certificates at the input and intermediate values. Passing runtime
validation does not supply those certificates.

If you build with the existing proof-carrying `Proofs.Autograd.DGraph`, composition keeps those
certificates. `DGraph.toTypedGraph` selects an output for the checked runtime, and
`DGraph.vjpChecked_adjoint_fderiv` uses the stored proofs without asking you to prove the new
composition's backward pass again. Import `NN.Proofs.Autograd.Runtime.Link.GraphComposition` for
this connection. New primitives still need their own local derivative proof; arbitrary Lean
functions are not automatically converted into certified graph nodes.
The [autograd deep dive](../../Examples/DeepDives/AutogradTransforms.lean) shows this for a custom
squaring operation composed twice, with arbitrary tensor shape.

For proof automation, import `NN.Tactic.Autograd` and use `by autograd`. It combines mathlib's
scalar derivative rules, lifts them to elementwise tensor certificates, and assembles registered
certificates for explicit graphs. Register a new proved rule with `@[autograd]`; use `autograd?`
to inspect the proof script. Domain assumptions still have to be proved. The
[tactic guide](../../Tactic/README.md) gives scalar, tensor, and validation examples.

The graph results above cover first-order differentiation. `NN.Proofs.Autograd.Dual` additionally
proves mixed-derivative semantics for supported nested scalar operations at every finite order,
using mathlib's `iteratedFDeriv`. `DualTensor` lifts them to the existing Euclidean tensor space;
the deep dive proves an elementwise `exp(x*x)` computation for every shape and derivative order.
The scalar rules also cover `tanh`, `sinh`, and `cosh`, including composition with smooth input
functions and higher derivatives of their pullback expressions. These are proofs of the existing
nested-dual operations, not finite-difference approximations or a replacement evaluator.
Its tensor seeding and coefficient extraction are the same runtime functions used by
`model.derivative` and `model.derivativeVjp`. `Runtime.Link.HigherOrder` proves that these
derivatives propagate through forward graph evaluation when each node preserves them; the result
includes every saved intermediate and any selected graph output. `HigherOrderReverse` extends
this to reverse accumulation with smoothly varying cotangents. `HigherOrderFDeriv` combines
that result with the graph's first-order certificate, proving higher derivatives of the exact
adjoint derivative. None of these theorems identifies floating-point results with exact real
derivatives.

For the pure runtime graph API, `TypedGraphWithData.tangent_vjp` connects nested `vjpWithSeed`
execution to the iterated derivative of the real graph's VJP. Its hypotheses check both the node
laws and agreement of the recorded shapes and selected output.

`TypedGraphWithData.tangent_vjpChecked_adjoint_fderiv` covers successful checked reverse
execution too, including tape lowering and recovery of typed gradients. The bookkeeping theorem
preserves addition order and works with arbitrary scalar storage, addition, and zero; only the
derivative interpretation specializes to reals.

With a fixed output cotangent and direction tuple,
`TypedGraphWithData.tangent_vjpChecked_iteratedFDeriv` identifies that result with the pullback
of the higher derivative itself. The interchange proof requires `n + 1` continuous derivatives
and works at every finite order. It uses mathlib's second-derivative symmetry, not an assumption
that every smooth function is analytic. The separate calculus theorem applies to general real
Hilbert spaces; derivative interchange without adjoints also covers complex normed spaces.

`NN.Proofs.Autograd.Model` connects the public `model.derivative` IO call to `iteratedFDeriv`.
It includes fixed state, the supplied direction list, and the returned graph. You must prove that
lowering returns that graph and that its operations preserve jets.

`model.derivativeVjp_eq` in `NN.Proofs.Autograd.Model.Reverse` covers the public reverse call,
including validation, execution, and the returned state/input pair. It proves that constant
parameter seeds and the supplied input directions are the correct full-context jet. Parameter
directions are zero for the input derivatives, but the final pullback still differentiates with
respect to parameters and inputs. The theorem requires successful lowering and checked execution,
the two graph certificates, and `n + 1` continuous derivatives. It works for every direction-list
length, including zero, and can be applied by `autograd`.

General automatic certification of model recording and `hessian` still need their full
connections, as do the remaining nonlinear scalar rules.

The deep dive also uses the actual graph recorder for `square` followed by `exp`. It proves that
recording succeeds and that nested-dual execution computes every mixed derivative of the real
graph's forward function. The scalar backend stays abstract in the recording proof; only the
derivative interpretation specializes to exact reals.

The same deep dive proves `squareModel_derivative` for the public IO API, including successful
model validation and recording, at every shape and finite order. This example needs no assumed
lowering result. For compositional execution proofs, `NN.Proofs.Autograd.Model.Composition`
provides equations for sequential models that preserve state order and training-mode buffer
updates without unfolding the backend implementation.

Runnable examples:

```bash
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean autograd_transforms
```
