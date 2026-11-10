/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autograd.Function
public import TorchLean.NN.Tactic.Autograd
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting

/-!
# A scaled exponential

`exponential scale` is `x ↦ exp (scale * x)` as an `autograd.Function` on scalars, for an integer
`scale`. `TorchLean.Op.exp_backward_eq_adjoint` is the real-arithmetic theorem behind its gradient:
at every shape, the arm of `Op.exp` has the slope map of `Real.exp` as derivative, and its pullback
is the adjoint.
-/

@[expose] public section

namespace TorchLean.Examples.Transcendentals

/--
$f(x)=e^{\mathrm{scale}\,x}$ for any integer scale.

The default uses the exponential directly, without adding an identity-scale node to the tape.
Other scales are converted through the scalar backend's natural-number and negation operations.
-/
def exponential (scale : Int := 1) : autograd.Function [] [] :=
  fun x =>
    if scale = 1 then
      Ops.apply .exp (x, ())
    else do
      let u ← Ops.apply .scale
          (x, .data (Ops.dataConst (Tensor.scalar (match scale with
        | .ofNat n => Nat.cast n
        | .negSucc n => -Nat.cast (n + 1)))), ())
      Ops.apply .exp (u, ())

end TorchLean.Examples.Transcendentals
