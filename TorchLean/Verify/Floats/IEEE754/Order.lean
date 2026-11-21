/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Order
public import TorchLean.Verify.Floats.IEEE754.Reductions

/-!
# The rounding error of an ordered sum

A sum in a reduction order, `Tensor.ReduceOrder.fold` with executable addition, evaluates the
order's tree (`fold_add_eq_evalSum`). In every format of FloatLib's configured binary family on
an IEEE descriptor:

- `abs_fold_add_sub_le`: for every order that covers the `n` entries, a finite ordered sum is
  within `((1 + u) ^ d - 1) · A` of the start value plus every entry, where `d` is the depth of
  the order's tree, `u = Model.unitRoundoffAt format` and `A` is the sum of the absolute values of
  the start value and the entries. It is `IEEE754.evalSum_enclosure_depth` on that tree.
- `abs_fold_pairwise_sub_le`: for `pairwise` the depth is at most `⌈log₂ (n + 1)⌉`. The left fold
  has depth `n`, the exponent of `evalSum_enclosure`.

Every entry of `reduceIn order .sum`, `sumIn`, `contractIn`, `poolSumIn` and `scanIn order (· + ·)`
is such a fold from zero, so the bound holds entrywise for each of them. The leaves of a
contraction are its rounded products, so there the bound is relative to the exact sum of those
products, as for `exactSumDot`. Products, maxima and minima have no bound here.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Numerics (ReductionTree)

namespace TorchLean.IEEE754

open FloatLib.Floats.Formats.BinaryInterchange
open Tensor (ReduceOrder)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

omit [ExecFloat.ModelCodec plan (Model format) code] in
/-- An ordered sum evaluates the order's tree, with the start value at `none` and entry `i` at
`some i`. -/
theorem fold_add_eq_evalSum [ExecFloat.Add (Configured.Family format code plan)]
    (order : ReduceOrder) (n : Nat) (start : Value) (value : Fin n → Value) :
    order.fold n (· + ·) start value = evalSum ((order.tree n).map (·.elim start value)) := by
  simp only [ReduceOrder.fold, evalSum, ReductionTree.eval_map]
  rfl

/-- A finite ordered sum in an IEEE format, in an order that covers `n` entries with a tree of
depth `d`, is within `((1 + u) ^ d - 1) · A` of the start value plus every entry, with `A` the sum
of their absolute values. -/
theorem abs_fold_add_sub_le [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add] {order : ReduceOrder} {n : Nat}
    (hcov : order.Covers n) (start : Value) (value : Fin n → Value)
    (hresult : isFinite (order.fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    |(toModel (order.fold n (· + ·) start value)).toReal -
        ((toModel start).toReal + ∑ i, (toModel (value i)).toReal)| ≤
      ((1 + Model.unitRoundoffAt format) ^ (order.tree n).depth - 1) *
        (|(toModel start).toReal| + ∑ i, |(toModel (value i)).toReal|) := by
  rw [fold_add_eq_evalSum] at hresult ⊢
  have hsum : ∀ g : Value → ℝ, ((order.tree n).map (·.elim start value)).eval (· + ·) g =
      g start + ∑ i, g (value i) := by
    intro g
    rw [ReductionTree.eval_add_eq_sum, ReductionTree.leaves_map, List.map_map,
      (List.Perm.map _ hcov).sum_eq, Fin.sum_univ_def]
    simp [Function.comp_def]
  have bound := evalSum_enclosure_depth _ (finiteEvalSumTree_of_isFinite _ hresult hformat) hformat
  rw [ReductionTree.depth_map, exactSum, hsum, sumAbs, ReductionTree.sumAbs, hsum] at bound
  exact bound

/-- A finite pairwise sum of `n` entries in an IEEE format is within
`((1 + u) ^ ⌈log₂ (n + 1)⌉ - 1) · A` of the start value plus every entry. -/
theorem abs_fold_pairwise_sub_le [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add] (n : Nat) (start : Value)
    (value : Fin n → Value)
    (hresult : isFinite (ReduceOrder.pairwise.fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    |(toModel (ReduceOrder.pairwise.fold n (· + ·) start value)).toReal -
        ((toModel start).toReal + ∑ i, (toModel (value i)).toReal)| ≤
      ((1 + Model.unitRoundoffAt format) ^ Nat.clog 2 (n + 1) - 1) *
        (|(toModel start).toReal| + ∑ i, |(toModel (value i)).toReal|) :=
  (abs_fold_add_sub_le (ReduceOrder.covers_pairwise n) start value hresult hformat).trans
    (mul_le_mul_of_nonneg_right
      (sub_le_sub_right (pow_le_pow_right₀
        (le_add_of_nonneg_right (Model.unitRoundoffAt_pos format).le)
        (ReduceOrder.depth_pairwise_le n)) 1)
      (by positivity))

end TorchLean.IEEE754
