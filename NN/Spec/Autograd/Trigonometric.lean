/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Core.TensorOps

/-!
# Trigonometric derivatives

Sine and cosine act independently on each tensor entry, with angles measured in radians.
Their Jacobians are diagonal, so the same pointwise derivative multiplication computes a JVP
or a VJP. The scalar operations come from `MathFunctions`, which also supplies their derivatives
when the scalar type is a dual number.
-/

@[expose] public section

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/--
Elementwise sine with VJP `cos(x) * dLdy`.

Evaluating the cosine at the original input preserves its sign across periods; recovering it
from the sine output would lose that information.
-/
def sinOp {s : Shape} : OpSpec α s s :=
  { forward := sinSpec
    backward := fun x dLdy => mulSpec (cosSpec x) dLdy }

/-- Elementwise cosine with VJP `-sin(x) * dLdy`, for inputs measured in radians. -/
def cosOp {s : Shape} : OpSpec α s s :=
  { forward := cosSpec
    backward := fun x dLdy => mulSpec (negSpec (sinSpec x)) dLdy }

end Spec
