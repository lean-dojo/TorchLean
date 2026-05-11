/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Flatbox
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Slice / gather / split operator bounds

This file provides IBP and affine transfer rules for a small subset of indexing-like operations:
- `Slice`: extract a contiguous range `[start, stop)` from a flattened vector,
- `Gather`: select entries by *static* indices (`List Nat`), and
- `Split`: split a flattened vector into a list of parts.

Important limitation: this does **not** model tensor-valued index dtypes inside the differentiable
graph (i.e. no PyTorch-style `LongTensor` indexing/gather/scatter driven by data tensors).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators.Slice

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

  /-- View a `(.dim n .scalar)` tensor as its underlying `Fin n → Tensor α .scalar` function. -/
  def getDimScalarFn {n : Nat} (t : Tensor α (.dim n .scalar)) : Fin n → Tensor α .scalar :=
    match t with
    | .dim f => f

/-- IBP for Slice: extract elements [start, stop) from a flattened vector.
    Slice is a linear operation, so bounds propagate exactly.
-/
def ibpSlice (xB : FlatBox α) (start stop : Nat) : FlatBox α :=
  let outDim := stop - start
  if start < stop ∧ stop ≤ xB.dim then
    let flo := getDimScalarFn xB.lo
    let fhi := getDimScalarFn xB.hi
    let outLo := Tensor.dim (fun i : Fin outDim =>
      let idx := start + i.val
      if hidx : idx < xB.dim then
        flo ⟨idx, hidx⟩
      else
        Tensor.scalar Numbers.zero)
    let outHi := Tensor.dim (fun i : Fin outDim =>
      let idx := start + i.val
      if hidx : idx < xB.dim then
        fhi ⟨idx, hidx⟩
      else
        Tensor.scalar Numbers.zero)
    { dim := outDim, lo := outLo, hi := outHi }
  else
    -- Invalid slice: return empty box
    { dim := 0
    , lo := Tensor.dim (fun i : Fin 0 => i.elim0)
    , hi := Tensor.dim (fun i : Fin 0 => i.elim0) }

/-- IBP for Gather: index into a vector using integer indices.
    Given input x and indices i, output[j] = x[indices[j]].
    Since indices are concrete, this is a permutation/selection.
-/
def ibpGather (xB : FlatBox α) (indices : List Nat) : FlatBox α :=
  let outDim := indices.length
  let flo := getDimScalarFn xB.lo
  let fhi := getDimScalarFn xB.hi
  let outLo := Tensor.dim (fun j : Fin outDim =>
    match indices[j.val]? with
    | some idx =>
      if hidx : idx < xB.dim then
        flo ⟨idx, hidx⟩
      else
        Tensor.scalar Numbers.zero
    | none => Tensor.scalar Numbers.zero)
  let outHi := Tensor.dim (fun j : Fin outDim =>
    match indices[j.val]? with
    | some idx =>
      if hidx : idx < xB.dim then
        fhi ⟨idx, hidx⟩
      else
        Tensor.scalar Numbers.zero
    | none => Tensor.scalar Numbers.zero)
  { dim := outDim, lo := outLo, hi := outHi }

/-- IBP for Split: split a vector into multiple parts.
    Returns a list of FlatBoxes, one for each split.
-/
def ibpSplit (xB : FlatBox α) (splitSizes : List Nat) : List (FlatBox α) :=
  let flo := getDimScalarFn xB.lo
  let fhi := getDimScalarFn xB.hi
  let rec buildSplits (remaining : List Nat) (offset : Nat) : List (FlatBox α) :=
    match remaining with
    | [] => []
    | size :: rest =>
      let box := {
        dim := size
        lo := Tensor.dim (fun i : Fin size =>
          let idx := offset + i.val
          if hidx : idx < xB.dim then
            flo ⟨idx, hidx⟩
          else
            Tensor.scalar Numbers.zero)
        hi := Tensor.dim (fun i : Fin size =>
          let idx := offset + i.val
          if hidx : idx < xB.dim then
            fhi ⟨idx, hidx⟩
          else
            Tensor.scalar Numbers.zero)
      }
      box :: buildSplits rest (offset + size)
  buildSplits splitSizes 0

/-- Affine bounds for Slice: extract subvector of affine form.
    If aff represents y = A·x + c, then slice just selects rows of A and c.
-/
def affSlice {inDim outDim : Nat} (start : Nat)
    (aff : AffineVec α inDim outDim) :
    (sliceSize : Nat) → AffineVec α inDim sliceSize :=
  fun sliceSize =>
    match aff.A, aff.c with
    | .dim rows, .dim cv =>
      let A' := Tensor.dim (fun i : Fin sliceSize =>
        let srcIdx := start + i.val
        if hsrc : srcIdx < outDim then
          rows ⟨srcIdx, hsrc⟩
        else
          -- Out of bounds: return zero row
          Tensor.dim (fun _ : Fin inDim => Tensor.scalar Numbers.zero))
      let c' := Tensor.dim (fun i : Fin sliceSize =>
        let srcIdx := start + i.val
        if hsrc : srcIdx < outDim then
          cv ⟨srcIdx, hsrc⟩
        else
          Tensor.scalar Numbers.zero)
      { A := A', c := c' }

/-- Affine bounds for Gather: permute/select rows of affine form. -/
def affGather {inDim outDim : Nat} (indices : List Nat)
    (aff : AffineVec α inDim outDim) : AffineVec α inDim indices.length :=
  match aff.A, aff.c with
  | .dim rows, .dim cv =>
    let A' := Tensor.dim (fun j : Fin indices.length =>
      match indices[j.val]? with
      | some idx =>
        if hidx : idx < outDim then
          rows ⟨idx, hidx⟩
        else
          Tensor.dim (fun _ : Fin inDim => Tensor.scalar Numbers.zero)
      | none => Tensor.dim (fun _ : Fin inDim => Tensor.scalar Numbers.zero))
    let c' := Tensor.dim (fun j : Fin indices.length =>
      match indices[j.val]? with
      | some idx =>
        if hidx : idx < outDim then
          cv ⟨idx, hidx⟩
        else
          Tensor.scalar Numbers.zero
      | none => Tensor.scalar Numbers.zero)
    { A := A', c := c' }

/-- Derivative bounds for Slice: derivatives just slice through. -/
def derivSlice (dB : FlatBox α) (start stop : Nat) : FlatBox α :=
  ibpSlice dB start stop

/-- Derivative bounds for Gather: derivatives follow the same indexing. -/
def derivGather (dB : FlatBox α) (indices : List Nat) : FlatBox α :=
  ibpGather dB indices

/-- Concatenate multiple FlatBoxes into one. -/
def ibpConcat (boxes : List (FlatBox α)) : FlatBox α :=
  let totalDim := boxes.foldl (fun acc b => acc + b.dim) 0
  if h : totalDim > 0 then
    let buildConcat := boxes.foldl (fun (acc : Array α × Array α) b =>
      let (loArr, hiArr) := acc
      let flo := getDimScalarFn b.lo
      let fhi := getDimScalarFn b.hi
      let newLo := (List.finRange b.dim).foldl (fun arr i =>
        match flo i with
        | .scalar v => arr.push v
      ) loArr
      let newHi := (List.finRange b.dim).foldl (fun arr i =>
        match fhi i with
        | .scalar v => arr.push v
      ) hiArr
      (newLo, newHi)
    ) (#[], #[])
    let (loArr, hiArr) := buildConcat
    { dim := totalDim
    , lo := Tensor.dim (fun i : Fin totalDim =>
        Tensor.scalar (if h : i.val < loArr.size then loArr[i.val] else Numbers.zero))
    , hi := Tensor.dim (fun i : Fin totalDim =>
        Tensor.scalar (if h : i.val < hiArr.size then hiArr[i.val] else Numbers.zero))
    }
  else
    { dim := 0
    , lo := Tensor.dim (fun i : Fin 0 => i.elim0)
    , hi := Tensor.dim (fun i : Fin 0 => i.elim0) }

namespace Theorems

/-- Slice output dimension matches stop - start for valid inputs. -/
theorem ibp_slice_dim (xB : FlatBox α) (start stop : Nat)
    (hvalid : start < stop ∧ stop ≤ xB.dim) :
    (ibpSlice xB start stop).dim = stop - start := by
  simp only [ibpSlice, hvalid]
  rfl

/-- Slice produces valid FlatBox structure. -/
theorem ibp_slice_returns_box (xB : FlatBox α) (start stop : Nat) :
    ∃ dim lo hi, ibpSlice xB start stop = { dim := dim, lo := lo, hi := hi } := by
  exact ⟨(ibpSlice xB start stop).dim, (ibpSlice xB start stop).lo, (ibpSlice xB start stop).hi,
    rfl⟩

/-- Gather output dimension equals indices length. -/
theorem ibp_gather_dim (xB : FlatBox α) (indices : List Nat) :
    (ibpGather xB indices).dim = indices.length := by
  unfold ibpGather
  rfl

/-- Split produces list of valid FlatBoxes. -/
theorem ibp_split_returns_list (xB : FlatBox α) (splitSizes : List Nat) :
    ∀ box ∈ ibpSplit xB splitSizes, ∃ dim lo hi, box = { dim := dim, lo := lo, hi := hi } := by
  intro box hbox
  exact ⟨box.dim, box.lo, box.hi, rfl⟩

/-- Concat output dimension is sum of input dimensions. -/
theorem ibp_concat_dim (boxes : List (FlatBox α)) :
    (ibpConcat boxes).dim = boxes.foldl (fun acc b => acc + b.dim) 0 ∨
    (ibpConcat boxes).dim = 0 := by
  classical
  by_cases hpos : (boxes.foldl (fun acc b => acc + b.dim) 0) > 0
  · left
    simp [ibpConcat, hpos]
  · right
    simp [ibpConcat, hpos]

end Theorems

end NN.MLTheory.CROWN.Operators.Slice
