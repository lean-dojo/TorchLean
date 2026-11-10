/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Lowering.Reduce.Index
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Tactic.Einops.Attr

/-!
# The einops mean at a scalar with a `Context`

Where the elaborator finds a `Context` but no division ring of characteristic zero, as at `Float32`
or at a scalar variable `α` with `[Context α]`, `reduce x "..." by mean` elaborates to
`Lowering.reduceFold (· + ·) 0 (fun total n => TorchLean.Context.divNat total n)`, to its
flat-reader form `Lowering.reduceFoldView`, or to a native kernel with the first as reference
(`TorchLean.Native.tensorKernel_correct`, `TorchLean.Native.certifiedTensor_correct`).
`Lowering.reduceFold_ordered_correct` and `Lowering.reduceFoldView_ordered_correct`
rewrite both forms to `Semantics.denoteOrderedReduce` with the same three arguments: the row-major
sum of each fiber, divided once by the fiber size as `Tensor.mean` divides. The two theorems of
this module assume a nonempty fiber, which the elaborator certifies for `mean`. On an empty fiber
the fold divides `0` by `0`, where `Tensor.mean` divides by one.

* `Semantics.denoteOrderedReduce_divNat_eq_mean`: under `LawfulContext` this mean is
  `Reduction.mean`.
* `Semantics.denoteOrderedReduce_divNat_eq_castShape_mean`: a reduction of every axis is
  `Tensor.mean` over every axis, cast to the checked output shape.
-/

@[expose] public section

namespace TorchLean.Einops

namespace Semantics

open Shape (Coord)
open Tensor (Rep)

/-- Under `LawfulContext`, the ordered reduction that adds each fiber with the `Context` addition
from the `Context` zero and divides once with `Context.divNat` is `Reduction.mean` on nonempty
fibers. -/
@[grind =, einops =, einops_simps] theorem denoteOrderedReduce_divNat_eq_mean {α : Type}
    [Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α] {context : Context α}
    [@LawfulContext α context _ _ _]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) :
    denoteOrderedReduce
        (fun total value => @HAdd.hAdd α α α (@instHAdd α context.toAdd) total value)
        (@OfNat.ofNat α 0 (@Zero.toOfNat0 α context.toZero))
        (fun total cardinality => context.divNat total cardinality)
        checked hKind inputTensor =
      denoteReduceNonempty Reduction.mean checked hKind hPositive inputTensor := by
  have hAdd : context.toAdd = Distrib.toAdd := by
    show Add.mk (fun x y => @HAdd.hAdd α α α (@instHAdd α context.toAdd) x y) =
      Add.mk (fun x y => @HAdd.hAdd α α α (@instHAdd α Distrib.toAdd) x y)
    simp only [LawfulContext.add_eq]
  have hZero : context.toZero = MulZeroClass.toZero := by
    show Zero.mk (@Zero.zero α context.toZero) = Zero.mk (@Zero.zero α MulZeroClass.toZero)
    rw [LawfulContext.zero_eq]
  rw [hAdd, hZero]
  simp only [LawfulContext.divNat_eq]
  rw [← Lowering.reduceFold_ordered_correct]
  exact Lowering.reduceFold_mean checked hKind hPositive inputTensor

/-- An ordered reduction of every axis with a nonempty fiber that adds from `0` and divides once
with `Context.divNat` is `Tensor.mean` over every axis, cast to the checked output shape. -/
theorem denoteOrderedReduce_divNat_eq_castShape_mean {α : Type} [Storage α] [Context α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hAll : checked.value.normalized.outputAxes = [])
    (hPositive : 0 < checked.reductionFiberSize)
    {keepdim : Bool}
    (hShape : Shape.reduce checked.value.normalized.input
        (List.finRange (Shape.rank checked.value.normalized.input)) keepdim =
      checked.value.output)
    (inputTensor : checked.InputTensor α) :
    denoteOrderedReduce (· + ·) 0 (fun total cardinality => Context.divNat total cardinality)
        checked hKind inputTensor =
      Tensor.castShape (Tensor.mean inputTensor (List.finRange _) keepdim) hShape := by
  have hReduced : checked.reducedAxes = checked.value.normalized.inputAxes := by
    simp [Transform.reducedAxes, hAll]
  have hFiber : checked.reductionFiberSize = Shape.size checked.value.normalized.input := by
    rw [checked.input_size_eq_elementary, ← Transform.reductionShape_size,
      Transform.reductionShape, hReduced]
  have hSizes : Shape.size checked.reductionShape = Shape.size checked.value.normalized.input := by
    rw [Transform.reductionShape_size, hFiber]
  have hOutput : Shape.size checked.value.output = 1 := by
    rw [← hShape, Shape.size_reduce_finRange]
  rw [← Lowering.reduceFold_ordered_correct]
  apply Rep.ext
  intro outputCoordinate
  rw [Tensor.castShape_apply]
  simp only [Lowering.reduceFold, Rep.get_ofFlatFn, Tensor.mean, Tensor.map_apply,
    Tensor.reduce_sum_finRange_apply, Shape.selectAxes_finRange]
  congr 1
  · -- With no output axis, the reduction reads the input in its own row-major order.
    have hIndex : ∀ r, (Lowering.Reduction.inputFlatIndex checked
        (Coord.unlinearize (Coord.linearize outputCoordinate)) r).val = r.val := by
      intro r
      rw [Lowering.Reduction.inputFlatIndex_val,
        Lowering.Reduction.linearIndex]
      have hZero : (Coord.linearize outputCoordinate).val = 0 := by
        have := (Coord.linearize outputCoordinate).isLt
        omega
      rw [hZero, Nat.mul_zero, Nat.add_zero]
      have hr := r.isLt
      generalize r.val = i at hr ⊢
      rw [hSizes, checked.input_size_eq_elementary] at hr
      rw [hAll, List.nil_append, hReduced]
      have h := Rearrangement.linearize_select checked.value.axisLength
        checked.value.normalized.inputAxes checked.value.normalized.inputAxes
        (fun _ h => h) ⟨i, hr⟩
      rw [AxisTuple.select_self checked.valid.normalization.input_nodup,
        Equiv.symm_apply_apply, Coord.linearize_unlinearize] at h
      exact h.symm
    rw [Lowering.Reduction.foldl, Lowering.Reduction.foldlFromFlat,
      Rep.foldl_eq_fin_foldl, Fin.foldl_congr hSizes]
    congr 1
    funext total r
    exact congrArg (total + ·) (congrArg inputTensor.getFlat (Fin.ext (hIndex _)))
  · rw [ite_eq_right (by omega), hFiber]

end Semantics

end TorchLean.Einops
