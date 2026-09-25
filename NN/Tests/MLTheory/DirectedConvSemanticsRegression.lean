/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.MLTheory.CROWN.Extras.FP32
import NN.MLTheory.CROWN.Proofs.Conv

/-!
# Convolution coefficient regressions

A stored FP32 weight need not lie on its rounding grid. Positive dilation must preserve that
weight in the affine matrix. Zero dilation can identify several kernel coordinates and must
retain their sum in the real converter.
-/

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor

noncomputable section

private def oneExtent : Tensor Nat [1] := Tensor.full [1] 1
private def zeroExtent : Tensor Nat [1] := Tensor.full [1] 0

private def thirdLayer : ConvSpec 1 1 1 oneExtent oneExtent zeroExtent FP32 :=
  { kernel := Tensor.full _ ⟨(1 : ℝ) / 3⟩
    bias := Tensor.full [1] 0 }

/-- The actual FP32 affine converter preserves an off-grid singleton weight exactly. -/
theorem convLinearMatrix_fp32_third :
    LawfulBoundOps.toReal
      (getAtOrZero
        (convLinearMatrix (inSpatial := oneExtent) thirdLayer oneExtent zeroExtent 1 .scalar)
        [0, 0]) = (1 : ℝ) / 3 := by
  rfl

/-- Two real kernel coordinates that coincide under zero dilation still contribute twice. -/
theorem convKernelCoefficient_zero_dilation :
    convKernelCoefficient [2] [0] [0] [1] [0] [0] (fun _ => (1 : ℝ) / 3) = 2 / 3 := by
  norm_num [convKernelCoefficient, Spec.Conv.Internal.foldlIndices, List.finRange_succ,
    Spec.Conv.Internal.mkDilatedInputIdx?]

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
