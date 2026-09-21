/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Autograd.Function
public import NN.API.Autograd.Model
public import NN.API.Autograd.Differential
public import NN.API.Autograd.Complex

/-!
# Automatic Differentiation

The common transforms are named directly:

- `autograd.grad` differentiates scalar tensor functions and accepts `(value := true)`;
- `autograd.model.grad` differentiates model losses and accepts `(value := true)`;
- `Trainer` is the ordinary training API and adds optimizer updates, batching, devices, and logs.

Namespace completion after `autograd.` exposes the function transforms. Completion after
`autograd.model.` exposes the model transforms and loss namespace. Named options such as `value`
appear in each operation's signature instead of creating additional `AndValue` declarations.

Start with `NN.Examples.Quickstart.AutogradBasics`. Full Jacobians, JVPs, Hessians, and
Hessian-vector products are kept in `NN.Examples.DeepDives.AutogradTransforms`.
-/
