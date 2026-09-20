/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Traversal
public import NN.Tensor.Internal.Representation.Basic.Reindex -- shake: keep

/-!
# Certified loops used by generated einsum kernels

Literal einsums execute concrete traversals with `USize` counters whenever
the corresponding semantic length fits every Lean target. This module keeps
the loop implementation and its correctness theorems separate from syntax
elaboration.

Each optimized primitive has a theorem identifying it with the standard
`Fin.foldl`, `Array.ofFn`, or row-major coordinate operation used by the
mathematical lowering. The proof arguments erase during code generation, so
the generated loop carries only its native bound, counter, and accumulator.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u

/--
Recombining a native quotient and remainder recovers the original index.

This theorem is kept out of the global simplifier. The einsum compiler uses
it only for row-major indices that it generated itself.
-/
theorem native_div_mul_add_mod (index divisor : USize) :
    index / divisor * divisor + index % divisor = index := by
  apply USize.toNat.inj
  simp [USize.toNat_add, USize.toNat_mul, Nat.div_add_mod',
    Nat.mod_eq_of_lt index.toNat_lt_size]

/--
Converting a native sum to `Nat` preserves a certified nonwrapping sum.
-/
theorem native_add_toNat_of_eq
    (left right : USize) (leftValue rightValue : Nat)
    (hLeft : left.toNat = leftValue)
    (hRight : right.toNat = rightValue)
    (hBound : leftValue + rightValue < USize.size) :
    (left + right).toNat = leftValue + rightValue := by
  have hBound' :
      leftValue + rightValue < 2 ^ System.Platform.numBits := by
    simpa [USize.size] using hBound
  simp [USize.toNat_add, hLeft, hRight, Nat.mod_eq_of_lt hBound']

/--
Converting a native product to `Nat` preserves a certified nonwrapping
product.
-/
theorem native_mul_toNat_of_eq
    (left right : USize) (leftValue rightValue : Nat)
    (hLeft : left.toNat = leftValue)
    (hRight : right.toNat = rightValue)
    (hBound : leftValue * rightValue < USize.size) :
    (left * right).toNat = leftValue * rightValue := by
  have hBound' :
      leftValue * rightValue < 2 ^ System.Platform.numBits := by
    simpa [USize.size] using hBound
  simp [USize.toNat_mul, hLeft, hRight, Nat.mod_eq_of_lt hBound']

/-- Native division respects certified equalities of both operands. -/
theorem native_div_toNat_of_eq
    (left right : USize) (leftValue rightValue : Nat)
    (hLeft : left.toNat = leftValue)
    (hRight : right.toNat = rightValue) :
    (left / right).toNat = leftValue / rightValue := by
  simp [USize.toNat_div, hLeft, hRight]

/-- Native remainder respects certified equalities of both operands. -/
theorem native_mod_toNat_of_eq
    (left right : USize) (leftValue rightValue : Nat)
    (hLeft : left.toNat = leftValue)
    (hRight : right.toNat = rightValue) :
    (left % right).toNat = leftValue % rightValue := by
  simp [USize.toNat_mod, hLeft, hRight]

/--
Native subtraction respects certified operand values when the subtrahend is
no larger than the minuend.
-/
theorem native_sub_toNat_of_eq
    (left right : USize) (leftValue rightValue : Nat)
    (hLeft : left.toNat = leftValue)
    (hRight : right.toNat = rightValue)
    (hLe : rightValue ≤ leftValue) :
    (left - right).toNat = leftValue - rightValue := by
  have hNative : right ≤ left := by
    rw [USize.le_iff_toNat_le, hLeft, hRight]
    exact hLe
  rw [USize.toNat_sub_of_le left right hNative, hLeft, hRight]

/--
A native strict comparison transports to the corresponding natural-number
comparison when both operand values are certified.
-/
theorem nat_lt_of_native_lt_of_eq
    (left right : USize) (leftValue rightValue : Nat)
    (hLeft : left.toNat = leftValue)
    (hRight : right.toNat = rightValue)
    (hLt : left < right) :
    leftValue < rightValue := by
  have hLtNat := USize.lt_iff_toNat_lt.mp hLt
  simpa only [hLeft, hRight] using hLtNat

/--
The false branch of a native strict comparison transports to the corresponding
natural-number lower bound when both operand values are certified.
-/
theorem nat_le_of_not_native_lt_of_eq
    (left right : USize) (leftValue rightValue : Nat)
    (hLeft : left.toNat = leftValue)
    (hRight : right.toNat = rightValue)
    (hNotLt : ¬ left < right) :
    rightValue ≤ leftValue := by
  apply Nat.le_of_not_gt
  intro hLt
  apply hNotLt
  apply USize.lt_iff_toNat_lt.mpr
  simpa only [hLeft, hRight] using hLt

/--
Decode and immediately re-encode a native row-major coordinate before adding
an invariant index base.
-/
theorem native_recombine_div_mod
    (index divisor product base : USize)
    (hProduct : product = index / divisor * divisor) :
    index % divisor + (product + base) =
      index + base := by
  rw [hProduct, ← USize.add_assoc,
    USize.add_comm (index % divisor), native_div_mul_add_mod]

/--
Decode and immediately re-encode a native row-major coordinate with a stride
before adding an invariant index base.

This is the form produced for a flattened contraction coordinate inside an
operand whose contracted axes are not the final physical dimensions.
-/
theorem native_recombine_scaled_div_mod
    (scale stride index divisor base : USize)
    (hStride : stride = scale * divisor) :
    scale * (index % divisor) +
        (stride * (index / divisor) + base) =
      scale * index + base := by
  rw [hStride]
  calc
    _ = (scale * (index % divisor) +
          (scale * divisor) * (index / divisor)) + base := by
      rw [USize.add_assoc]
    _ = scale *
        ((index / divisor) * divisor + index % divisor) + base := by
      congr 1
      rw [USize.mul_add, USize.mul_assoc,
        USize.mul_comm divisor (index / divisor), USize.add_comm]
    _ = scale * index + base := by
      rw [native_div_mul_add_mod]

/--
Dividing a bounded native product index by its row width produces a valid
outer coordinate.
-/
theorem native_div_lt_of_lt_mul
    (outer inner : Nat) (index innerBound : USize)
    (hInnerBound : innerBound.toNat = inner)
    (hIndex : index.toNat < outer * inner) :
    (index / innerBound).toNat < outer := by
  rw [USize.toNat_div, hInnerBound]
  exact Nat.div_lt_of_lt_mul (Nat.mul_comm outer inner ▸ hIndex)

/--
Taking a native remainder by a positive row width produces a valid inner
coordinate.
-/
theorem native_mod_lt_of_pos
    (inner : Nat) (index innerBound : USize)
    (hInnerBound : innerBound.toNat = inner)
    (hInner : 0 < inner) :
    (index % innerBound).toNat < inner := by
  rw [USize.toNat_mod, hInnerBound]
  exact Nat.mod_lt _ hInner

/--
Native quotient decoding agrees with the quotient coordinate of a finite
row-major product index.
-/
theorem native_fin_div_eq_divNat
    (outer inner : Nat) (index innerBound : USize)
    (hInnerBound : innerBound.toNat = inner)
    (hIndex : index.toNat < outer * inner) :
    (⟨(index / innerBound).toNat,
        native_div_lt_of_lt_mul outer inner index innerBound
          hInnerBound hIndex⟩ : Fin outer) =
      (⟨index.toNat, hIndex⟩ : Fin (outer * inner)).divNat := by
  apply Fin.ext
  simp [hInnerBound]

/--
Native remainder decoding agrees with the remainder coordinate of a finite
row-major product index.
-/
theorem native_fin_mod_eq_modNat
    (outer inner : Nat) (index innerBound : USize)
    (hInnerBound : innerBound.toNat = inner)
    (hIndex : index.toNat < outer * inner)
    (hInner : 0 < inner) :
    (⟨(index % innerBound).toNat,
        native_mod_lt_of_pos inner index innerBound hInnerBound hInner⟩ :
      Fin inner) =
      (⟨index.toNat, hIndex⟩ : Fin (outer * inner)).modNat := by
  apply Fin.ext
  simp [hInnerBound]

/--
Projecting the state of a finite fold is equivalent to folding the projected
state whenever one step commutes with the projection.
-/
theorem fin_foldl_project
    {State Value : Type*} (length : Nat)
    (project : State → Value)
    (stateStep : State → Fin length → State)
    (valueStep : Value → Fin length → Value)
    (initial : State)
    (hStep :
      ∀ state index,
        valueStep (project state) index =
          project (stateStep state index)) :
    project (Fin.foldl length stateStep initial) =
      Fin.foldl length valueStep (project initial) := by
  rw [Fin.foldl_eq_foldl_finRange, Fin.foldl_eq_foldl_finRange]
  exact (List.foldl_hom project hStep).symm

/--
Fold over a tensor shape in row-major coordinate order.

This definition is the proof-level reference for generated loop nests. The
einsum compiler emits the same recursion directly, choosing a native counter
independently for each concrete axis.
-/
def coordinateFoldl {α : Type u} :
    (shape : Shape) → (α → Coord shape → α) → α → α
  | [], step, initial => step initial PUnit.unit
  | length :: shape, step, initial =>
      Fin.foldl length
        (fun value head =>
          coordinateFoldl shape
            (fun value tail => step value (head, tail))
            value)
        initial

/-- A four-element finite fold is the corresponding sequence of four updates. -/
theorem fin_foldl_four
    {α : Type u} (step : α → Fin 4 → α) (initial : α) :
    Fin.foldl 4 step initial =
      step (step (step (step initial 0) 1) 2) 3 := by
  simp only [Fin.foldl_succ, Fin.foldl_zero]
  congr

/-- An eight-element finite fold is the corresponding sequence of eight updates. -/
theorem fin_foldl_eight
    {α : Type u} (step : α → Fin 8 → α) (initial : α) :
    Fin.foldl 8 step initial =
      step (step (step (step (step (step (step (step initial 0) 1) 2) 3)
        4) 5) 6) 7 := by
  simp only [Fin.foldl_succ, Fin.foldl_zero]
  congr

/--
Converting a row-major product coordinate to `USize` exposes the native
multiplication and addition used to construct it.

The semantic certificate retains `Fin.mkDivMod`; generated indices use the
equivalent word-sized arithmetic.
-/
theorem toUSize_mkDivMod
    {blocks width : Nat} (block : Fin blocks) (lane : Fin width) :
    USize.ofNat (Fin.mkDivMod block lane).val =
      USize.ofNat width * USize.ofNat block.val +
        USize.ofNat lane.val := by
  simp only [Fin.coe_mkDivMod, USize.ofNat_add, USize.ofNat_mul]

/--
Fixed-width blocks followed by a tail enumerate exactly the original finite
index interval. The theorem changes only loop nesting; it preserves the order
in which `step` observes indices.
-/
theorem fin_foldl_tiles
    {α : Type u} (blocks width tail : Nat)
    (step : α → Fin (blocks * width + tail) → α)
    (initial : α) :
    Fin.foldl tail
        (fun value index =>
          step value (index.natAdd (blocks * width)))
        (Fin.foldl blocks
          (fun value block =>
            Fin.foldl width
              (fun value lane =>
                step value <|
                  (Fin.mkDivMod block lane).castLE
                    (Nat.le_add_right (blocks * width) tail))
              value)
          initial) =
      Fin.foldl (blocks * width + tail) step initial := by
  rw [fin_foldl_product]
  simpa only [Fin.divNat_mkDivMod_modNat] using
    (Fin.foldl_add step initial).symm

/--
The recursive coordinate fold is exactly the standard flat finite fold.

This theorem supplies the arbitrary-rank certificate for generated output
loops. In particular, the empty shape executes once and any zero-length axis
executes no leaves.
-/
theorem coordinateFoldl_eq_fin_foldl
    {α : Type u} (shape : Shape)
    (step : α → Coord shape → α) (initial : α) :
    coordinateFoldl shape step initial =
      Fin.foldl (Shape.size shape)
        (fun value index =>
          step value (Coord.unlinearize index))
        initial := by
  induction shape generalizing initial with
  | nil =>
      simp [coordinateFoldl, Shape.size, Fin.foldl_succ,
        Fin.foldl_zero, Coord.unlinearize, Coord.equivFin]
  | cons length shape inductionHypothesis =>
      simp only [coordinateFoldl, Shape.size_cons]
      simp_rw [inductionHypothesis]
      rw [fin_foldl_product]
      apply congrArg (fun foldStep =>
        Fin.foldl (length * Shape.size shape) foldStep initial)
      funext value index
      rfl

/-- A shape containing no coordinates leaves a fold accumulator unchanged. -/
theorem coordinateFoldl_eq_initial_of_size_eq_zero
    {α : Type u} (shape : Shape) (step : α → Coord shape → α) (initial : α)
    (hSize : Shape.size shape = 0) : coordinateFoldl shape step initial = initial := by
  rw [coordinateFoldl_eq_fin_foldl]
  simp [hSize, Fin.foldl_zero]

/--
Projecting the state of a row-major coordinate fold is equivalent to folding
the projected state when every coordinate update commutes with the projection.
-/
theorem coordinateFoldl_project
    {State Value : Type*} (shape : Shape)
    (project : State → Value)
    (stateStep : State → Coord shape → State)
    (valueStep : Value → Coord shape → Value)
    (initial : State)
    (hStep :
      ∀ state coordinate,
        valueStep (project state) coordinate =
          project (stateStep state coordinate)) :
    project (coordinateFoldl shape stateStep initial) =
      coordinateFoldl shape valueStep (project initial) := by
  rw [coordinateFoldl_eq_fin_foldl, coordinateFoldl_eq_fin_foldl]
  apply fin_foldl_project
  intro state index
  exact hStep state (Coord.unlinearize index)

/--
Pushing entries through the row-major coordinate fold constructs the standard
flat function array.
-/
theorem coordinateFoldl_push_linearized_eq_array_ofFn
    {α : Type u} (shape : Shape)
    (values : Fin (Shape.size shape) → α) :
    coordinateFoldl shape
        (fun output coordinate =>
          output.push (values (Coord.linearize coordinate)))
        (Array.emptyWithCapacity (Shape.size shape)) =
      Array.ofFn values := by
  rw [coordinateFoldl_eq_fin_foldl]
  simpa only [Coord.linearize_unlinearize] using
    Storage.fin_foldl_push_eq_array_ofFn (Shape.size shape) values

/--
Observing a physical-buffer coordinate fold gives the standard flat function
array.
-/
theorem coordinateFoldl_storagePush_linearized_toArray_eq_array_ofFn
    {α : Type u} [storage : Storage α] (shape : Shape)
    (values : Fin (Shape.size shape) → α) :
    storage.toArray
        (coordinateFoldl shape
          (fun output coordinate =>
            storage.push output (values (Coord.linearize coordinate)))
          (storage.emptyWithCapacity (Shape.size shape))) =
      Array.ofFn values := by
  rw [coordinateFoldl_project shape storage.toArray
    (fun output coordinate =>
      storage.push output (values (Coord.linearize coordinate)))
    (fun output coordinate =>
      output.push (values (Coord.linearize coordinate)))
    (storage.emptyWithCapacity (Shape.size shape))]
  · rw [storage.toArray_emptyWithCapacity,
      coordinateFoldl_push_linearized_eq_array_ofFn]
  · intro output coordinate
    exact (storage.toArray_push output _).symm

/--
A one-axis output loop observes directly as its row-major function array.

This specialized certificate avoids normalizing the recursive coordinate fold
when a tiled output compiler already emits the single `Fin.foldl` explicitly.
-/
theorem fin_foldl_storagePush_rankOne_toArray_eq_array_ofFn
    {α : Type u} [storage : Storage α]
    (length : Nat) (values : Fin (Shape.size [length]) → α) :
    storage.toArray
        (Fin.foldl length
          (fun output index =>
            storage.push output
              (values (Coord.linearize (s := [length])
                (index, PUnit.unit))))
          (storage.emptyWithCapacity (Shape.size [length]))) =
      Array.ofFn values := by
  rw [Storage.toArray_finFoldl_push_eq_array_ofFn]
  apply Array.ext
  · calc
      (Array.ofFn fun index : Fin length =>
          values (Coord.linearize (s := [length])
            (index, PUnit.unit))).size = length := Array.size_ofFn
      _ = Shape.size [length] := (Nat.mul_one length).symm
      _ = (Array.ofFn values).size := Array.size_ofFn.symm
  · intro index hLeft hRight
    simp only [Array.getElem_ofFn]
    apply congrArg values
    apply Fin.ext
    calc
      (Coord.linearize (s := [length])
          (⟨index, by simpa [Array.size_ofFn] using hLeft⟩,
            PUnit.unit)).val =
          (Coord.linearize (s := []) PUnit.unit).val +
            Shape.size [] * index :=
        Coord.linearize_cons_val _ _
      _ = index := by
        simp [Coord.linearize, Coord.equivFin, Shape.size]

/--
Transport a certified one-axis physical-buffer loop directly to its ordinary
array observation.
-/
theorem storage_toArray_eq_array_ofFn_of_rankOne_eq
    {α : Type u} [storage : Storage α]
    (length : Nat) (values : Fin (Shape.size [length]) → α)
    (output : storage.Buffer)
    (hOutput :
      output =
        Fin.foldl length
          (fun result index =>
            storage.push result
              (values (Coord.linearize (s := [length])
                (index, PUnit.unit))))
          (storage.emptyWithCapacity (Shape.size [length]))) :
    storage.toArray output = Array.ofFn values := by
  rw [hOutput]
  exact
    fin_foldl_storagePush_rankOne_toArray_eq_array_ofFn
      length values

end TorchLean.Tensor.Internal.Elab.Impl
