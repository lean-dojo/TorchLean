# Autograd

First decide whether the derivative itself is the result you want:

| Goal | Use |
| --- | --- |
| Differentiate a function of one tensor | `autograd.grad` |
| Differentiate a model loss with respect to model state | `autograd.model.grad` |
| Pull an output gradient back to model state and input | `autograd.model.vjp` |
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
- `autograd.model.` exposes `grad`, `vjp`, `jacrev`, `jvp`, `hvp`, and `Loss`.

The selected operation's signature then shows its named options. For example, `grad` displays
`(value : Bool := false)`; there is no separate compound operation to discover.

## Tensor Functions

An `autograd.Function input output` is a differentiable tensor program with no
model parameters.

```lean
def meanSquare : autograd.Function [3] [] :=
  fun x => do
    let squared ← nn.functional.square x
    nn.functional.mean squared

let x : Tensor Float [3] := [1.0, 2.0, 3.0]
let (gradient, value) ← autograd.grad meanSquare x (value := true)
#eval value
#eval gradient
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
let state : autograd.model.State model Float :=
  autograd.model.initialState model
let (gradient, lossValue) ←
  autograd.model.grad
    model autograd.model.Loss.meanSquaredError state input target
    (value := true)
#eval lossValue
#eval gradient
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
let (gradient, inputGradient) ←
  autograd.model.vjp model state input seed
#eval gradient
#eval inputGradient
```

Plain `Tensor` values do not own a mutable gradient field. Session inputs default to no gradient;
model parameters default to trainable unless a layer marks them frozen.

Model loss transforms differentiate the model state because that is the quantity an optimizer
updates. They do not also differentiate the input and target implicitly. Use `vjp` when an explicit
output gradient should be pulled back to the input, or define a scalar tensor function whose chosen
argument is the differentiation input.

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

Runnable examples:

```bash
lake exe torchlean quickstart_autograd
lake exe torchlean autograd_transforms
```
