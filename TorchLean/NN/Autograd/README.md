# Autograd

This folder holds the `autograd` transforms:

- `Function.lean`: `autograd.grad`, `autograd.vjp`, `autograd.jacrev`, `autograd.jacfwd` and
  `autograd.hessian` for one-argument tensor programs. `Frontend.lean` is the internal frontend
  that records ordinary scalar and tensor calculations on the autograd tape.
- `Model.lean`: derivatives of a model with respect to its state and its input
  (`autograd.model.grad` and the rest). In `Model/`, `Reverse.lean` proves the higher-derivative
  theorems of the public transforms and `Seeding.lean` builds fixed-state derivative seeds.
- `Differential.lean`: coordinate derivatives of neural fields (`autograd.model.derivative`,
  `autograd.model.derivativeVJP`).
- `Complex.lean`: real losses on complex parameters (`autograd.complex.grad`).

First decide whether the derivative itself is the result you want:

| Goal | Use |
| --- | --- |
| Differentiate a function of one tensor | `autograd.grad` |
| Differentiate a model loss with respect to model state | `autograd.model.grad` |
| Pull an output gradient back to model state and input | `autograd.model.vjp` |
| Repeated or mixed input-directional derivatives | `autograd.model.derivative` |
| Parameter and input gradients through those derivatives | `autograd.model.derivativeVJP` |
| Differentiate a real loss on complex parameters | `autograd.complex.grad` |
| Update model parameters over data | `Trainer`, not a direct autograd call |

`autograd` returns values and derivatives and never mutates parameters or runs an optimizer.
`Trainer` uses model autograd internally, then owns updates, batching, devices and reports.

Application code imports:

```lean
import TorchLean.NN.Builder
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Model
open TorchLean
```

Lean namespace completion lists the operations themselves:

- `autograd.` exposes `grad`, `vjp`, `jacfwd`, `jacrev` and `hessian`.
- `autograd.model.` exposes `grad`, `vjp`, `jacrev`, `jvp`, `hvp`, `derivative`,
  `derivativeVJP` and `Loss`.

The selected operation's signature then shows its named options. For example, `grad` displays
`(value : Bool := false)`, so there is no separate compound operation to discover.

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

`value` is a rank-zero `Tensor α []`, so it prints as is and stays in the same tensor API as every
other result. Use `Tensor.item` only at a host-scalar boundary such as a log message or a
loop-counter decision.

Use `grad` for scalar outputs, and set `(value := true)` when the same evaluation should also return
the function value. Use `vjp`, `jacrev`, `jacfwd` or `hessian` only when that derivative object is
needed.

## Model Losses

`autograd.model.State model α` has the same typed tensor layout as the model's state.
Differentiation returns a fresh state-shaped gradient value, and there is no mutable `.grad`
field.

```lean
def model : nn.Sequential [2] [1] := nn.build 0 (nn.linear 2 1)

#eval show IO Unit from do
  let state : autograd.model.State model Float := nn.initialState model
  let input : Tensor Float [2] := [0.5, -1.0]
  let target : Tensor Float [1] := [0.25]
  let (gradient, lossValue) ← autograd.model.grad model
    autograd.model.Loss.mse state input target (value := true)
  IO.println s!"loss = {lossValue}, gradient = {reprStr gradient}"
```

Use:

- `grad` when only the model-state gradient is needed
- `grad (value := true)` when both the model-state gradient and loss are needed from one
  evaluation
- `vjp` for state and input gradients induced by an upstream output gradient
- `jvp` for a directional derivative through model state
- `hvp` for a Hessian-vector product
- `Loss.detach loss` to stop gradients through the model output

For an upstream model-output gradient:

```lean
-- Inside a `do` block, with an output-shaped tensor `seed`:
let (gradient, inputGradient) ← autograd.model.vjp model state input seed
IO.println s!"state gradient = {reprStr gradient}"
IO.println s!"input gradient = {reprStr inputGradient}"
```

Plain `Tensor` values have no mutable gradient field. Session inputs default to no gradient, and
model parameters default to trainable unless a layer marks them frozen.

Model loss transforms differentiate the model state, because that is what an optimizer updates.
They don't differentiate the input or the target. Use `vjp` to pull an output gradient back to the
input, or define a scalar tensor function whose argument is the differentiation input.

## Neural Fields and PDE Residuals

`autograd.model.derivative model state input directions` takes input-directional derivatives
while holding the parameter state fixed. Directions are a tensor whose leading axis selects the
derivative order: an empty leading axis evaluates the model, `[dx]` computes one derivative,
and `[dx, dy]` computes a mixed derivative. Each row has the model's input shape. Directions
need not be coordinate basis vectors, and the API fixes neither the dimension nor the derivative
order. Evaluation propagates nested dual scalars and uses no finite differences.

To train on a derivative, use `autograd.model.derivativeVJP` with the same arguments and an
output cotangent. It returns parameter and input gradients. For a squared PDE residual
`(u_xx - f)^2 / 2`, pass `u_xx - f` as that cotangent, and the parameter result is this
residual's contribution to the training gradient. Directions and the supplied cotangent are held
constant in the pullback. Separate boundary terms use the ordinary model VJP.

The derivative transforms use evaluation mode. `autograd.model.derivative` differentiates the
forward evaluation over nested dual numbers (`derivative_eq`), not the registered JVP rules, so it
can differ from `jacfwdInput` and the model VJP: a frozen reference keeps its tangent, a tie of
three or more entries in a max or min reduction weights the tangents by fold position (1/4, 1/4
and 1/2 for three entries) instead of taking their mean. Like the model VJP, it runs each node's
`NodeData.validate` check and throws the first failure.
Choose a model smooth enough for the equation: a ReLU MLP's second derivative is not a useful
model of classical curvature. Each call lowers the model at the required nested scalar type, so
high derivative orders can be expensive.

Run a complete Lean-only example with `scripts/lake.sh exe torchlean pinn`. It trains a tanh MLP on
`u'' = -2`, `u(-1) = u(1) = 0` without solution labels. The loss samples the equation residual at
five interior collocation points and the boundary residual at both endpoints, so training proves
nothing about the PDE between those points.

## Real Losses on Complex Parameters

Complex parameters have two real coordinates. For a real loss `L`, the update direction is
`dL/dre + i*dL/dim`, which also handles conjugation and squared magnitude (neither is complex
analytic). `autograd.complex.Objective shapes` describes such a loss on
`TensorPack (Complex α) shapes`. Write the objective polymorphically in the component type so it
can run with dual numbers as well as ordinary scalars.

`autograd.complex.grad objective state` returns the state-shaped gradient. Pass `(value := true)`
to also return the real loss. `autograd.complex.jvp objective state direction` evaluates a chosen
direction in one pass. They keep real and imaginary components throughout, without converting to
`Float`. An SGD update uses `nn.sgdStep model (TorchLean.Complex.ofReal rate) state gradient`, and
the model's frozen-parameter flags apply here too.

The implementation takes two forward directional passes per complex parameter entry. It suits
small models and gradient checks, since the cost grows with the number of parameter entries. The
objective must compute the same thing on every pass, so don't resample data or dropout masks
inside it. Branch cuts and nonsmooth points follow the scalar implementation's conventions. On the
cut of the complex `sqrt`, a negative real part with a zero imaginary part, the tangent is the
derivative `dz / (2 √z)` of the side that the sign of the zero selects.

The component type needs `Storage`, `Context` and `InverseTrig`. `Float` and `Float32` take their
inverse trigonometric functions from the C library, and every configured binary format has an
`InverseTrig` instance (`configuredBinaryInverseTrig`, each function the real one rounded once).
The complex type over each of them has the full `Context`. Checkpoints don't need the angle
operations and store complex components of every width without changing their precision.

Run `scripts/lake.sh exe torchlean complex_regression`. It trains complex binary32 weights and
bias against a real squared-residual loss, then prints both components of a held-out
prediction. `--steps N` sets the step count (50 by default) and `--save-checkpoint PATH` saves
the complete state:

```bash
scripts/lake.sh exe torchlean complex_regression \
  --steps 50 --save-checkpoint /tmp/complex-regression.state
```

`Checkpoint.State.load (α := Complex Float32) (nn.Sequential.stateShapes model) path` rejects a
different component format or tensor layout. The supervised `Trainer` is a real-data API, so the
example threads the state through by hand.

## Training

Ordinary training code should use `Trainer`. It runs model autograd internally and adds the parts
that are not differentiation:

- datasets and batching
- optimizer and scheduler state
- device selection
- checkpoints, logs and reports

Direct `autograd.model` calls are most useful for custom objectives, gradient inspection, and
derivative tests.

## Runtime Domains

`Op.log` checks no input values (`Op.validate_eq_ok`), and the autograd transforms add no check.
At `Float` and `Float32` it gives `-∞` at `+0` and at `-0`, NaN below zero and at NaN, and `+∞`
at `+∞` (IEEE 754-2019 9.2 and 6.2). Over `ℝ` it is `Real.log`, with `Real.log 0 = 0`. The
pullback of `Op.log` is `g / x`. `Op.safeLog` computes `log (softplus x + ε)` with a frozen
scalar `ε`, for models that need an epsilon-protected log-like operation.

## What The Types Guarantee

Input, output, target, parameter, and derivative shapes are tracked in Lean types. A returned model
gradient has the model state's tensor layout.

Running an autograd program computes floating-point values. A run doesn't prove that the result
equals a real-analysis derivative. The semantic and runtime-correctness proofs live under
`TorchLean/NN`.

For exact-real graphs, `WithData.jvpChecked_eq_fderiv` and
`WithData.vjpChecked_eq_fderiv` connect successful checked execution to mathlib's
`fderiv` of the selected forward output. They require a proof graph describing the same node data
and derivative-correctness certificates at the input and intermediate values. Passing runtime
validation does not supply those certificates.

Graphs built with the proof-carrying `Tape.DGraph` keep those certificates under
composition. `DGraph.toTypedGraph` selects an output for the checked runtime, and
`DGraph.vjpChecked_eq_fderiv` uses the stored proofs, so the composed backward pass needs no
new proof. Import `TorchLean.NN.Tape.DGraph.Checked` for this connection.
Each new primitive needs its own local derivative proof, and arbitrary Lean functions don't become
certified graph nodes automatically.
The [autograd deep dive](../../../TorchLeanExamples/DeepDives/AutogradTransforms.lean) shows this
for a custom squaring operation composed twice, on tensors of any shape.

For proof automation, import `TorchLean.NN.Tactic.Autograd` and use `by autograd`. It combines
mathlib's scalar derivative rules, lifts them to elementwise tensor certificates, and assembles
registered certificates for recorded graph nodes. Register a new proved rule with `@[autograd]`, and
use `autograd?` to inspect the proof script. Domain hypotheses of nonsmooth operations need their
own proofs. The [tactic guide](../Tactic/README.md) gives scalar and tensor examples.

The graph results above cover first-order differentiation. `TorchLean.NN.Dual.Jet`
proves mixed-derivative semantics for the supported nested scalar operations at every finite order,
using mathlib's `iteratedFDeriv`, and `DualTensor` lifts them to the Euclidean tensor space. The
deep dive proves an elementwise `exp(x*x)` computation for every shape and derivative order. The
scalar rules also cover `tanh`, `sinh` and `cosh`, including composition with smooth input
functions and higher derivatives of their pullback expressions. The theorems are about the
nested-dual operations the runtime executes: tensor seeding and coefficient extraction are the
same functions that `model.derivative` and `model.derivativeVJP` call.

`TorchLean.NN.Tape.Lowering.HigherOrder` proves that these derivatives propagate through forward
graph evaluation when each node preserves them, for every saved intermediate and any selected graph
output. `TorchLean.NN.Tape.Lowering.HigherOrderReverse` extends this to reverse accumulation with
smoothly varying cotangents. `TorchLean.NN.TypedGraph.WithData.HigherOrder` combines that result
with the graph's first-order certificate and proves higher derivatives of the exact adjoint
derivative. None of these theorems identifies
floating-point results with exact real derivatives.

For the pure runtime graph API, `WithData.tangent_vjp` connects nested `vjp`
execution to the iterated derivative of the real graph's VJP. Its hypotheses check both the node
laws and agreement of the recorded shapes and selected output.

`WithData.tangent_vjpChecked_eq_fderiv` covers successful checked reverse
execution too, including tape lowering and recovery of typed gradients. The bookkeeping theorem
preserves addition order and holds for any scalar storage, addition and zero. Only the
derivative interpretation specializes to reals.

With a fixed output cotangent and direction tuple,
`WithData.tangent_vjpChecked_iteratedFDeriv` identifies that result with the pullback
of the higher derivative itself. The interchange proof needs `n + 1` continuous derivatives and
works at every finite order. It uses mathlib's symmetry of second derivatives and assumes no
analyticity. A standalone calculus theorem covers general real Hilbert spaces, and derivative
interchange without adjoints also covers complex normed spaces.

`TorchLean.NN.Autograd.Model.Reverse` connects the public `model.derivative` IO call to
`iteratedFDeriv`. It includes fixed state, the supplied direction tensor, and the returned graph.
You must prove that lowering returns that graph and that its operations preserve jets.

`model.derivativeVJP_eq` in `TorchLean.NN.Autograd.Model.Reverse` covers the public reverse
call, including validation, execution, and the returned state/input pair. It proves that constant
parameter seeds and the supplied input directions are the correct full-context jet. Parameter
directions are zero for the input derivatives, but the final pullback differentiates with
respect to both parameters and inputs. The theorem requires successful lowering and checked
execution, the two graph certificates, and `n + 1` continuous derivatives. Directions have a
leading axis of length `n`, followed by the input shape. It works for every finite order,
including an empty leading axis for order zero, and can be applied by `autograd`.

Automatic certification has no such connection for general model recording, `hessian` or the
remaining nonlinear scalar rules.

The deep dive also applies the graph recorder to `square` followed by `exp`. It proves that
recording succeeds and that nested-dual execution computes every mixed derivative of the real
graph's forward function. The scalar backend stays abstract in the recording proof, and only the
derivative interpretation specializes to exact reals.

The same deep dive proves `squareModel_derivative` for the public IO API, including successful
model validation and recording, at every shape and finite order. This example needs no assumed
lowering result. For compositional execution proofs,
`TorchLean.NN.Sequential.Composition` provides equations for sequential models that
preserve state order and training-mode buffer updates without unfolding the backend implementation.

Runnable examples:

```bash
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean autograd_transforms
```
