/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay

/-!
# Softmax replay in binary32

Kernel-checked binary32 inputs on which softmax takes its replay branches.

* `absorbed`: the ordinary sum `1 + 2^-25 + 2^-25` stalls at one.
* `overflowing`: the ordinary sum of two largest finite values overflows.
* The logits `[9, -9, -9]` of Barron, Gnanapragasam and Tschannen: softmax replays because its
  max-shifted sum stalls at one. The softmax pullback of a one-hot upstream at these logits keeps
  its ordinary result.
* A pullback whose centered upstream overflows.

`roundoff` bounds softmax and its pullback through the absolute forms of the theorems of
`TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay`, for every format.
-/

public section

namespace TorchLean.SoftmaxReplay

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open TorchLean.Tensor

/-- The largest finite binary32 value, `(2^24 - 1) * 2^104`. -/
theorem toRat?_maxFinite :
    ExecFloat.Binary.toRat? (ExecFloat.Binary.maxFinite : Binary32) = some (16777215 * 2 ^ 104) ∧
      Model.toRat? (Model.posMaxFinite (ExecFloat.Binary.descriptor 8 23)) =
        some (16777215 * 2 ^ 104) := by
  decide +kernel

/-! ## A stalled sum -/

/-- One, then two weights that binary32 addition to one absorbs. -/
def absorbed : Tensor Binary32 [3] :=
  Tensor.ofFn fun i => if i = 0 then 1 else ExecFloat.Binary.ofRat (1 / 2 ^ 25)

/-- The ordinary sum stalls at one, so its quotients would return the stored weights. The replay
rounds the exact weights `1 / (1 + 2^-24)` and `2^-25 / (1 + 2^-24)` instead. -/
theorem absorbed_replay :
    ExecFloat.Binary.toRat? (Tensor.sum absorbed) = some 1 ∧
      ExecFloat.Binary.toRat? (absorbed.getScalar 0 / Tensor.sum absorbed) = some 1 ∧
      ExecFloat.Binary.toRat? ((Softmax.normalizeWeights absorbed).getScalar 0) =
        some (16777215 / 2 ^ 24) ∧
      ExecFloat.Binary.toRat? ((Softmax.normalizeWeights absorbed).getScalar 1) =
        some (16777215 / 2 ^ 49) := by
  decide +kernel

/-! ## A discarded overflowing denominator -/

/-- Two largest finite values. -/
def overflowing : Tensor Binary32 [2] := Tensor.ofFn fun _ => ExecFloat.Binary.maxFinite

/-- The ordinary sum overflows and its quotients vanish. The replay returns one half twice. -/
theorem overflowing_replay :
    ExecFloat.Binary.toRat? (Tensor.sum overflowing) = none ∧
      ExecFloat.Binary.toRat? (overflowing.getScalar 0 / Tensor.sum overflowing) = some 0 ∧
      ExecFloat.Binary.toRat? ((Softmax.normalizeWeights overflowing).getScalar 0) =
        some (1 / 2) ∧
      ExecFloat.Binary.toRat? ((Softmax.normalizeWeights overflowing).getScalar 1) =
        some (1 / 2) := by
  decide +kernel

/-! ## The paper's logits -/

/-- Logits `[9, -9, -9]`. -/
def logits : Tensor Binary32 [3] := Tensor.ofFn fun i => if i = 0 then 9 else -9

/-- The max-shifted sum `1 + 2 exp (-18)` stalls at one, so softmax replays: the target weight is
the largest binary32 value below one, and the others keep their exponentials. -/
theorem logits_replay :
    ExecFloat.Binary.toRat? (Tensor.sum (Softmax.maxShiftedExpVec logits)) = some 1 ∧
      ExecFloat.Binary.toRat? ((Softmax.vec logits).getScalar 0) =
        some (16777215 / 2 ^ 24) ∧
      ExecFloat.Binary.toRat? ((Softmax.vec logits).getScalar 1) =
        ExecFloat.Binary.toRat? ((Softmax.maxShiftedExpVec logits).getScalar 1) := by
  decide +kernel

/-- The one-hot upstream of class `0`. -/
def target : Tensor Binary32 [3] := Tensor.ofFn fun i => if i = 0 then 1 else 0

/-- Against the one-hot upstream, no ordinary pullback coordinate is flagged, so the softmax VJP
keeps its ordinary result. -/
theorem logits_pullback_ordinary :
    let weights := Softmax.vec logits
    Softmax.pullback weights target (target.getScalar 0) =
      weights * ((target - Tensor.full _ (target.getScalar 0)) -
        (Tensor.full _ (Tensor.sum ((target - Tensor.full _ (target.getScalar 0)) *
            weights)))) := by
  decide +kernel

/-! ## An overflowing centered upstream -/

/-- Equal weights. -/
def halves : Tensor Binary32 [2] := Tensor.ofFn fun _ => ExecFloat.Binary.ofRat (1 / 2)

/-- The largest finite value and its negation. -/
def extremes : Tensor Binary32 [2] :=
  Tensor.ofFn fun i => if i = 0 then ExecFloat.Binary.maxFinite else ExecFloat.Binary.maxFinite true

/-- Centering at the first upstream value overflows, so the ordinary pullback is not finite.
The replay returns the exact VJP, half the largest finite value with either sign. -/
theorem extremes_replay :
    let centered := extremes - Tensor.full _ (extremes.getScalar 0)
    let ordinary := halves *
      (centered - Tensor.full _ (Tensor.sum (centered * halves)))
    let pullback := Softmax.pullback halves extremes (extremes.getScalar 0)
    ExecFloat.Binary.isFinite (ordinary.getScalar 0) = false ∧
      ExecFloat.Binary.toRat? (pullback.getScalar 0) = some (16777215 * 2 ^ 103) ∧
      ExecFloat.Binary.toRat? (pullback.getScalar 1) = some (-(16777215 * 2 ^ 103)) := by
  decide +kernel

/-! ## Automation -/

section Configured

variable {format : FloatFormat}

local notation "Value" => ExecFloat.Binary format.expWidth format.fracWidth
  format.encoding format.exponentBias format.expWidth_ge_two format.fracWidth_pos
  format.exponentBias_pos format.exponentBias_le_maxFinite

/-- `roundoff` reads softmax through the normalization rule and composes it with the error of the
max-shifted exponentials, against the exact softmax of the stored logits. -/
roundoff theorem softmax_forward_error :=
  fun {n : Nat} (input : Tensor Value [n + 1]) (index : Fin (n + 1))
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal ((Softmax.vec input).getScalar index)

/-- `roundoff` analyzes both branches of the pullback: the ordinary arithmetic under its flag
check, and the replay through its rule. -/
roundoff theorem softmax_pullback_error :=
  fun {n : Nat} (weights gradient : Tensor Value [n]) (anchor : Value) (index : Fin n)
    (_hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      ((Softmax.pullback weights gradient anchor).getScalar index)

end Configured

end TorchLean.SoftmaxReplay
