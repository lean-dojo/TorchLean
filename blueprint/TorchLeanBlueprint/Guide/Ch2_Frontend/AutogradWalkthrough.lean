import VersoManual

open Verso.Genre Manual

#doc (Manual) "Autograd Walkthrough" =>
%%%
tag := "autograd-walkthrough"
%%%

Most ML users first meet automatic differentiation through `loss.backward()`. TorchLean keeps that
workflow, but it does not hide the objects involved. A gradient call names the function being
differentiated, the input shape, the output shape, and, for vector outputs, the cotangent seed that
chooses which linear combination of outputs to differentiate.

The objects to watch are:

- the function or model being differentiated,
- the input and parameter shapes,
- the seed for a vector-Jacobian product when one is needed,
- the gradient tensors returned by the runtime.

# A Scalar Function

The smallest case is a scalar valued tensor function.

```
import NN

open Spec
open Tensor
open NN.Tensor
open NN.API

def sumsq : autograd.fn1.Fn (Shape.Vec 2) Shape.scalar :=
  fun x => do
    let y ← nn.functional.square x
    nn.functional.mean y

def demo : IO Unit := do
  let x : Tensor Float (Shape.Vec 2) := tensorND! [2] [0.5, -1.2]
  let g ← autograd.fn1.grad (α := Float) sumsq x
  IO.println s!"grad = {Spec.pretty g}"
```

The shape tells us that `sumsq` consumes a vector of length two and returns a scalar. Because the
output is scalar, `autograd.fn1.grad` returns a tensor with the same shape as the input.

For `x = (x_1, x_2)`, the function is:

$$`\operatorname{mean}(x^2) = (x_1^2 + x_2^2)/2`

So the mathematical gradient is:

$$`\nabla_x \operatorname{mean}(x^2) = x`

so the returned gradient has the same two coordinates as `x`.

# Value And Gradient Together

For debugging it is often useful to compute the value and gradient in one call:

```
let (value, grad) ← autograd.fn1.valueAndGrad (α := Float) sumsq x
```

This mirrors the common workflow:

```
y = f(x)
dy_dx = grad(y, x)
```

TorchLean keeps both results ordinary values. There is no hidden `.grad` field that must be cleared
before the next step.

# Vector Outputs Need A Seed

For scalar outputs, there is a single gradient. For vector outputs, there is not. If
`f(x) = [f_1(x), f_2(x)]`, then “the gradient of `f`” could mean the gradient of `f_1`, the
gradient of `f_2`, their sum, or some weighted combination. A vector-Jacobian product supplies that
choice.

```
let dx ← autograd.fn1.vjp (α := Float) f x seedOut
```

Informally:

$$`\operatorname{vjp}(f,x,\bar y) = J_f(x)^\mathsf{T}\bar y`

Here `seedOut` is the output cotangent. It has the same shape as the output of `f`, and the returned
tensor has the same shape as `x`.

This is exactly the operation reverse mode needs. Each local operation sends an output cotangent
back to input cotangents; composing those local rules gives the full gradient.

This is one of the places where TorchLean's shape discipline pays off. The type of the derivative
tells you what kind of object you should expect back.

# Model Parameters

Training usually differentiates a loss with respect to parameters, not only with respect to an
input tensor. The model-level API uses the same idea, but the returned gradient has the same
parameter structure as the model.

The usual calls are:

- `autograd.model.gradParams` for gradients of the loss with respect to parameters,
- `autograd.model.gradInputs` for gradients with respect to input and target tensors,
- `autograd.model.valueAndGradParams` when the loss value and parameter gradients are both needed.

Those declarations live in [NN.API.Public](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean), while the lower runtime
implementation lives in [NN.API.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Runtime.lean) and
[NN.Runtime.Autograd.TorchLean.Autodiff](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Autodiff.lean).

# Jacobians, JVPs, And HVPs

TorchLean also exposes analysis tools beyond the basic training gradient:

- `autograd.fn1.jacfwd` for a forward mode Jacobian,
- `autograd.fn1.jacrev` for a reverse mode Jacobian,
- `autograd.fn1.hessian` for scalar valued functions,
- `autograd.model.jvpParams` for a directional derivative of a scalar loss,
- `autograd.model.hvpParams` for Hessian vector products over parameters.

Read the APIs this way:

- `grad` returns a scalar-output gradient, the common case for losses.
- `vjp` returns `J_f(x)^T seedOut`, the reverse-mode object for vector outputs.
- `jacfwd` is useful when the input dimension is small.
- `jacrev` is useful when the output dimension is small.
- `hvpParams` gives a Hessian-vector product for curvature diagnostics.

These are useful when a chapter talks about sensitivity, curvature, verification, or second order
diagnostics. They are also a good way to see why the typed shape discipline matters: every returned
object has the shape dictated by the derivative it represents.

# From Local Rules To Global Backprop

Autograd correctness has a simple informal shape.

1. Each primitive operation has a forward meaning.
2. Each primitive operation has a local VJP or JVP rule.
3. If those local rules match the derivative of the forward operation, then the composed reverse
   pass computes the adjoint derivative of the whole graph.

The guide states this informally because it is the right mental model for users. The proof chapters
then point to the exact Lean theorems that make the statement precise.

The main proof trail starts here:

- [NN.Proofs.Autograd.Overview API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Overview.lean)
- [NN.Proofs.Autograd.Tape.Algebra.Soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean)
- [NN.Proofs.Autograd.Runtime.Link API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Runtime/Link.lean)

# A Practical Rule

Use the smallest API that matches the question.

- For a gradient of `Tensor σ -> scalar`, start with `autograd.fn1.grad`.
- For a value and gradient together, start with `autograd.fn1.valueAndGrad`.
- For a vector output with a chosen cotangent, start with `autograd.fn1.vjp`.
- For a model loss gradient with respect to parameters, start with `autograd.model.gradParams`.
- For minibatch training, start with `train.fitLoader` or the quickstart examples.

For a runnable tour, open
[NN.Examples.Quickstart.AutogradBasics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean). It prints
gradients, VJPs, Jacobians, Hessian vector products, and parameter gradients in one place.
