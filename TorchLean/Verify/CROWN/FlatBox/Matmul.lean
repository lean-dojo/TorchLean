/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.MatmulLayout
public import TorchLean.Verify.CROWN.FlatBox.Real
public import TorchLean.Verify.CROWN.BoundOps.Interval

/-!
# The interval matrix product on flat boxes

`binaryMatmulBox` is the directed interval contraction of two flat boxes over the layout
`IR.MatmulDims`, with the scalar indexing and summation order of `IR.MatmulDims.matmulFlat`.
`boxScale` multiplies each entry of a box by the interval scalar in the first entry of a second
box. Over a lawful backend, `Directed.binaryMatmulBox_encloses` proves that the first
encloses `IR.MatmulDims.matmulFlat` on the real vectors (`pointVector`) of enclosed operands, and
`Directed.boxScale_encloses` proves the same for the scaled values.
-/

public section

namespace TorchLean.CROWN

open Tensor

variable {α : Type} [Storage α] [Context α]
variable [BoundOps α]

open BoundOps


/-- Interval box of the matrix product over `dims`. Each output entry sums the `intervalMul`
products of the operand entries that `IR.MatmulDims.matmulFlat` pairs, in its order, rounding the
lower sum down and the upper sum up. -/
@[expose] def Graph.binaryMatmulBox (dims : IR.MatmulDims) (left right : FlatBox α) :
    FlatBox α :=
  let endpoints (index : Fin dims.outShape.size) : α × α :=
    (List.range dims.inner).foldl (fun acc inner =>
      let leftIndex := dims.leftIndex index.val inner
      let rightIndex := dims.rightIndex index.val inner
      let product := intervalMul (α := α)
        (getAtOrZero left.lo [leftIndex]) (getAtOrZero left.hi [leftIndex])
        (getAtOrZero right.lo [rightIndex]) (getAtOrZero right.hi [rightIndex])
      (addDown acc.1 product.1, addUp acc.2 product.2)) (0, 0)
  { dim := dims.outShape.size
    lo := Tensor.ofFn fun index => (endpoints index).1
    hi := Tensor.ofFn fun index => (endpoints index).2 }

/-- The box `B` scaled entrywise by the interval in the first entry of `S`. -/
@[expose] def Graph.boxScale (B S : FlatBox α) : FlatBox α :=
  let product (i : Fin B.dim) := intervalMul (α := α) (Tensor.getScalar B.lo i)
    (Tensor.getScalar B.hi i) (getAtOrZero S.lo [0]) (getAtOrZero S.hi [0])
  { dim := B.dim
    lo := Tensor.ofFn fun i => (product i).1
    hi := Tensor.ofFn fun i => (product i).2 }

namespace Directed

open Graph (intervalMul_encloses)

noncomputable section

open Graph (binaryMatmulBox boxScale)

variable [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The real vector of length `n` whose `i`-th coordinate is `f i`. -/
@[expose] def pointVector (n : Nat) (f : Nat → ℝ) : Tensor ℝ [n] :=
  Tensor.ofFn fun i => f i.val

/-- Zero-extended reads keep an enclosure: past index `n` the box endpoints and the vector all read
as zero. -/
theorem rowEncloses_read {box : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (h : RowEncloses box n f) (i : Nat) :
    value (getAtOrZero box.lo [i]) ≤ getAtOrZero (pointVector n f) [i] ∧
      getAtOrZero (pointVector n f) [i] ≤ value (getAtOrZero box.hi [i]) := by
  rcases box with ⟨d, lo, hi⟩
  obtain ⟨hd, h⟩ := h
  change d = n at hd
  subst d
  by_cases hi' : i < n
  · simpa [getAtOrZero_dim_cons, hi', pointVector, Tensor.getScalar_ofFn] using h ⟨i, hi'⟩
  · simp [getAtOrZero_dim_cons, hi', LawfulBoundOps.toReal_zero]

/-- If `left` and `right` enclose `x` and `y`, then `binaryMatmulBox layout left right` encloses
`MatmulDims.matmulFlat layout` on the real vectors of `x` and `y`. -/
theorem binaryMatmulBox_encloses (layout : IR.MatmulDims)
    {left right : FlatBox α} {n m : Nat} {x y : Nat → ℝ}
    (hx : RowEncloses left n x) (hy : RowEncloses right m y) :
    RowEncloses (binaryMatmulBox layout left right) layout.outShape.size
      (fun i => getAtOrZero
        (IR.MatmulDims.matmulFlat layout (pointVector n x) (pointVector m y)) [i]) := by
  refine ⟨rfl, ?_⟩
  intro output
  simp only [binaryMatmulBox, IR.MatmulDims.matmulFlat, Tensor.getAtOrZero_eq_getScalar,
    Tensor.getScalar_ofFn]
  let p (inner : Nat) : α × α := Graph.intervalMul
    (getAtOrZero left.lo [layout.leftIndex output.val inner])
    (getAtOrZero left.hi [layout.leftIndex output.val inner])
    (getAtOrZero right.lo [layout.rightIndex output.val inner])
    (getAtOrZero right.hi [layout.rightIndex output.val inner])
  have h := foldl_encloses (List.range layout.inner) (fun inner => (p inner).1)
    (fun inner => (p inner).2)
    (fun inner => getAtOrZero (pointVector n x) [layout.leftIndex output.val inner] *
      getAtOrZero (pointVector m y) [layout.rightIndex output.val inner]) 0 0 0
    (by simp [LawfulBoundOps.toReal_zero]) (by simp [LawfulBoundOps.toReal_zero])
    fun inner _ => intervalMul_encloses
      (rowEncloses_read hx (layout.leftIndex output.val inner)).1
      (rowEncloses_read hx (layout.leftIndex output.val inner)).2
      (rowEncloses_read hy (layout.rightIndex output.val inner)).1
      (rowEncloses_read hy (layout.rightIndex output.val inner)).2
  -- The pair fold is the pair of the endpoint folds, and the real fold is the list sum.
  rw [zero_add, List.sum_eq_foldl, List.foldl_map,
    List.foldl_hom Prod.fst (g₁ := fun a t => (BoundOps.addDown a.1 (p t).1,
      BoundOps.addUp a.2 (p t).2)) (g₂ := fun a t => BoundOps.addDown a (p t).1)
      (init := (0, 0)) fun _ _ => rfl,
    List.foldl_hom Prod.snd (g₁ := fun a t => (BoundOps.addDown a.1 (p t).1,
      BoundOps.addUp a.2 (p t).2)) (g₂ := fun a t => BoundOps.addUp a (p t).2)
      (init := (0, 0)) fun _ _ => rfl] at h
  exact h

/-- The real vector of the zero-extended reads of a vector is the vector. -/
theorem pointVector_getAtOrZero {n : Nat} (v : Tensor ℝ [n]) :
    pointVector n (fun i => getAtOrZero v [i]) = v := by
  simp only [pointVector, Tensor.getAtOrZero_eq_getScalar, Tensor.ofFn_getScalar]

/-- If `B` encloses `x` and the one-entry box `S` encloses `y`, then `boxScale B S` encloses
`fun i => x i * y 0`. -/
theorem boxScale_encloses {B S : FlatBox α} {n : Nat} {x y : Nat → ℝ}
    (hx : RowEncloses B n x) (hy : RowEncloses S 1 y) :
    RowEncloses (boxScale B S) n fun i => x i * y 0 := by
  rcases B with ⟨d, lo, hi⟩
  obtain ⟨hd, h⟩ := hx
  change d = n at hd
  subst d
  refine ⟨rfl, fun i => ?_⟩
  have hi := h i
  simp only [Tensor.getAtOrZero_eq_getScalar] at hi
  simp only [boxScale, Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_ofFn]
  exact intervalMul_encloses hi.1 hi.2 (hy.2 ⟨0, Nat.one_pos⟩).1 (hy.2 ⟨0, Nat.one_pos⟩).2

end

end Directed

end TorchLean.CROWN
