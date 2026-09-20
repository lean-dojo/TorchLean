/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Loss
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Spec.Core.Context.Rational

/-!
# WeightedMseTest

Exact rational regression checks for weighted mean-squared error.

The fixture gives zero weight to two deliberately large errors. It checks both the scalar loss and
the prediction gradient, so a regression in masking, normalization, or differentiation is visible
without floating-point tolerance.
-/

open scoped Spec.RationalAlgebraic

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tests
namespace Rationals
namespace WeightedMse

open Runtime.Autograd

/-- Four-coordinate fixture shape. -/
abbrev shape : Shape := [4]

/-- Prediction containing large errors at coordinates excluded by the weights. -/
def prediction : Tensor ℚ shape := [1, 100, 3, 200]

/-- Zero reconstruction target. -/
def target : Tensor ℚ shape := [0, 0, 0, 0]

/-- Normalized weights selecting only the first and third coordinates. -/
def weights : Tensor ℚ shape := [1 / 2, 0, 1 / 2, 0]

/-- Exact prediction gradient `2 * weights * (prediction - target)`. -/
def expectedGradient : Tensor ℚ shape := [1, 0, 3, 0]

/-- Evaluate the weighted loss and its prediction gradient through the eager runtime. -/
def evaluate : IO (ℚ × Tensor ℚ shape) := do
  let session ← Torch.Internal.EagerSession.new (α := ℚ)
  let predictionRef ← Torch.Internal.EagerSession.input
    (α := ℚ) (sh := shape) session prediction
    (name := some "prediction") (requiresGrad := true)
  let action : Torch.Internal.EagerM ℚ (Torch.TensorRef ℚ Shape.scalar) := do
    let targetRef ← Torch.Ops.const
      (m := Torch.Internal.EagerM ℚ) (α := ℚ) (s := shape) target
    let weightsRef ← Torch.Ops.const
      (m := Torch.Internal.EagerM ℚ) (α := ℚ) (s := shape) weights
    TorchLean.Loss.mseWeighted
      (m := Torch.Internal.EagerM ℚ) (α := ℚ) (s := shape)
      predictionRef targetRef weightsRef
  let lossRef ← action session
  let loss ← Torch.Internal.EagerSession.getValue
    (α := ℚ) (sh := Shape.scalar) session lossRef
  let gradients ← Torch.Internal.EagerSession.backwardDenseAll
    (α := ℚ) (sh := Shape.scalar) session lossRef (Tensor.scalar 1)
  let predictionGradient ← Torch.Internal.EagerSession.grad gradients predictionRef
  pure (Tensor.item loss, predictionGradient)

/-- Run the exact weighted-MSE regression check. -/
def run : IO Unit := do
  let (loss, gradient) ← evaluate
  unless loss = 5 do
    throw <| IO.userError s!"weighted_mse_test (Rat): expected loss 5, got {loss}"
  unless Tensor.to gradient (Array ℚ) = Tensor.to expectedGradient (Array ℚ) do
    throw <| IO.userError
      s!"weighted_mse_test (Rat): expected gradient {expectedGradient}, got {gradient}"
  IO.println "weighted_mse_test (Rat): OK"

end WeightedMse
end Rationals
end Tests
