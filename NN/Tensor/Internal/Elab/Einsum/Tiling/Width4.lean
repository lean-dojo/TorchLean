/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Tiling.Semantics
import Mathlib.Tactic.FinCases

/-!
# Four-register output tiling

This module implements one concrete scalar-register lowering selected by
static cost analysis. Its update step refines the arbitrary-width semantics
from `Tiling.Semantics`; the width is an implementation choice rather than a
semantic restriction.

Every lane observes contracted coordinates in the same row-major order as the
scalar lowering, so tiling requires no reassociation, commutativity, or
distributivity argument.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u

/-- Select one of four values by a statically bounded lane index. -/
def selectTile4 {α : Type u}
    (value0 value1 value2 value3 : α) : Fin 4 → α
  | lane =>
      match lane.val with
      | 0 => value0
      | 1 => value1
      | 2 => value2
      | _ => value3

/--
Selecting a concrete four-lane contraction after summing each lane agrees
with summing the selected lane. This representation lemma preserves the
coordinate order of `Semantics.coordinateSum`.
-/
theorem coordinateSum_selectTile4
    {α : Type u} [Add α] [OfNat α 0] (shape : Shape)
    (value0 value1 value2 value3 : Coord shape → α) (initial : α) :
    (fun lane =>
      Semantics.coordinateSum shape
        (fun coordinate =>
          selectTile4
            (value0 coordinate) (value1 coordinate)
            (value2 coordinate) (value3 coordinate) lane)
        initial) =
      selectTile4
        (Semantics.coordinateSum shape value0 initial)
        (Semantics.coordinateSum shape value1 initial)
        (Semantics.coordinateSum shape value2 initial)
        (Semantics.coordinateSum shape value3 initial) := by
  funext lane
  fin_cases lane <;> rfl

/--
Update the proof-facing state of a four-lane tile.

For concrete contractions, `compileCoordinateFold` recognizes this exact
update and first emits `nativeFinSum4`, whose four totals are separate loop
arguments. The surrounding output pass fuses an immediately consumed result
into `nativeFinSum4Push`.
-/
@[inline] def updateTile4
    {α : Type u} [Add α] (totals : Vector α 4)
    (value0 value1 value2 value3 : α) : Vector α 4 :=
  let totals := totals.set 0 (totals[0] + value0)
  let totals := totals.set 1 (totals[1] + value1)
  let totals := totals.set 2 (totals[2] + value2)
  totals.set 3 (totals[3] + value3)

/-- Append one completed four-lane tile to a row-major output buffer. -/
@[inline] def pushTile4
    {α : Type u} [storage : Storage α]
    (output : storage.Buffer) (values : Vector α 4) : storage.Buffer :=
  storage.push
    (storage.push
      (storage.push
        (storage.push output values[0]) values[1]) values[2]) values[3]

/-- Pointwise equal lane values determine equal four-lane vectors. -/
theorem ofFn_selectTile4_congr
    {α : Type u}
    (value0 value1 value2 value3
      reference0 reference1 reference2 reference3 : α)
    (h0 : value0 = reference0) (h1 : value1 = reference1)
    (h2 : value2 = reference2) (h3 : value3 = reference3) :
    Vector.ofFn (selectTile4 value0 value1 value2 value3) =
      Vector.ofFn
        (selectTile4 reference0 reference1 reference2 reference3) := by
  subst_vars
  rfl

/-- Appending a four-lane function vector emits its values in lane order. -/
theorem pushTile4_ofFn
    {α : Type u} [storage : Storage α] (output : storage.Buffer)
    (value0 value1 value2 value3 : α) :
    pushTile4 output
        (Vector.ofFn (selectTile4 value0 value1 value2 value3)) =
      storage.push
        (storage.push
          (storage.push
            (storage.push output value0) value1) value2) value3 := by
  simp only [pushTile4, Vector.getElem_ofFn, selectTile4]

/-- Projecting a four-lane update gives the corresponding scalar update. -/
theorem updateTile4_get
    {α : Type u} [Add α] (totals : Vector α 4)
    (value0 value1 value2 value3 : α) (lane : Fin 4) :
    (updateTile4 totals value0 value1 value2 value3)[lane] =
      totals[lane] +
        selectTile4 value0 value1 value2 value3 lane := by
  fin_cases lane <;>
    simp only [Fin.getElem_fin, updateTile4, selectTile4,
      Vector.getElem_set, Nat.reduceEqDiff, reduceIte]

/--
The four-register update implements the width-polymorphic tile update.
-/
theorem updateTile4_eq_updateTile
    {α : Type u} [Add α] (totals : Vector α 4)
    (value0 value1 value2 value3 : α) :
    updateTile4 totals value0 value1 value2 value3 =
      updateTile totals
        (selectTile4 value0 value1 value2 value3) := by
  apply Vector.ext
  intro lane hLane
  have hFour :=
    updateTile4_get totals value0 value1 value2 value3 ⟨lane, hLane⟩
  have hGeneric :=
    updateTile_get totals
      (selectTile4 value0 value1 value2 value3) ⟨lane, hLane⟩
  simp only [Fin.getElem_fin] at hFour hGeneric
  rw [hFour, hGeneric]

/--
Run a native finite sum while carrying four lane totals as separate scalar
arguments.

Each recursive step advances one contraction coordinate. This keeps the
generated callback small enough for Lean's native compiler to inline it into
the loop, matching the eight-lane implementation.
-/
@[specialize] def nativeFinSum4Loop
    {α : Type u} [Add α] (length : Nat)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize) (total0 total1 total2 total3 : α) : Vector α 4 :=
  if hIndex : index < bound then
    have _hIndexNat : index.toNat < length := by
      have := USize.lt_iff_toNat_lt.mp hIndex
      lia
    nativeFinSum4Loop length value0 value1 value2 value3
      bound hBound (index + 1)
      (total0 + value0 index _hIndexNat)
      (total1 + value1 index _hIndexNat)
      (total2 + value2 index _hIndexNat)
      (total3 + value3 index _hIndexNat)
  else
    #v[total0, total1, total2, total3]
termination_by length - index.toNat
decreasing_by
  have _hIndexNat : index.toNat < bound.toNat :=
    USize.lt_iff_toNat_lt.mp hIndex
  have hNextFits : index.toNat + 1 < USize.size := by
    have hBoundFits := USize.toNat_lt_size bound
    lia
  simp only [USize.toNat_add, USize.toNat_one,
    Nat.mod_eq_of_lt hNextFits]
  lia

/--
Sum four lanes over a native finite interval without vector operations inside
the loop.
-/
@[inline] def nativeFinSum4
    {α : Type u} [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (initial : Vector α 4) : Vector α 4 :=
  nativeFinSum4Loop length value0 value1 value2 value3
    bound hBound 0 initial[0] initial[1] initial[2] initial[3]

/--
Run four scalar accumulators and append their final values directly to an
existing output buffer.
-/
@[specialize] def nativeFinSum4PushLoop
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (output : storage.Buffer) (index : USize)
    (total0 total1 total2 total3 : α) : storage.Buffer :=
  if hIndex : index < bound then
    have _hIndexNat : index.toNat < length := by
      have := USize.lt_iff_toNat_lt.mp hIndex
      lia
    nativeFinSum4PushLoop length value0 value1 value2 value3
      bound hBound output (index + 1)
      (total0 + value0 index _hIndexNat)
      (total1 + value1 index _hIndexNat)
      (total2 + value2 index _hIndexNat)
      (total3 + value3 index _hIndexNat)
  else
    let output := storage.push output total0
    let output := storage.push output total1
    let output := storage.push output total2
    storage.push output total3
termination_by length - index.toNat
decreasing_by
  have _hIndexNat : index.toNat < bound.toNat :=
    USize.lt_iff_toNat_lt.mp hIndex
  have hNextFits : index.toNat + 1 < USize.size := by
    have hBoundFits := USize.toNat_lt_size bound
    lia
  simp only [USize.toNat_add, USize.toNat_one,
    Nat.mod_eq_of_lt hNextFits]
  lia

/-- Sum four lanes and append them without materializing the result vector. -/
@[inline] def nativeFinSum4Push
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (output : storage.Buffer) (initial : α) : storage.Buffer :=
  nativeFinSum4PushLoop length value0 value1 value2 value3
    bound hBound output 0 initial initial initial initial

/--
Starting from any native index and four scalar totals, the scalar-state loop
agrees with the corresponding vector-state native fold.
-/
private theorem nativeFinSum4Loop_eq_nativeFinFoldlLoop
    {α : Type u} [Add α] (length : Nat)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize) (total0 total1 total2 total3 : α) :
    nativeFinSum4Loop length value0 value1 value2 value3
        bound hBound index
        total0 total1 total2 total3 =
      nativeFinFoldlLoop length
        (fun totals index hIndex =>
          updateTile4 totals
            (value0 index hIndex) (value1 index hIndex)
            (value2 index hIndex) (value3 index hIndex))
        bound hBound index #v[total0, total1, total2, total3] := by
  refine nativeFinSum4Loop.induct
    length value0 value1 value2 value3
    bound hBound
    (motive := fun index total0 total1 total2 total3 =>
      nativeFinSum4Loop length value0 value1 value2 value3
          bound hBound index
          total0 total1 total2 total3 =
        nativeFinFoldlLoop length
          (fun totals index hIndex =>
            updateTile4 totals
              (value0 index hIndex) (value1 index hIndex)
              (value2 index hIndex) (value3 index hIndex))
          bound hBound index #v[total0, total1, total2, total3])
    ?_ ?_ index total0 total1 total2 total3
  · intro index total0 total1 total2 total3
      hIndex _hIndexNat inductionHypothesis
    rw [nativeFinSum4Loop.eq_1, dite_eq_left hIndex]
    rw [nativeFinFoldlLoop.eq_1, dite_eq_left hIndex]
    simpa [updateTile4] using inductionHypothesis
  · intro index total0 total1 total2 total3 hIndex
    rw [nativeFinSum4Loop.eq_1, dite_eq_right hIndex]
    rw [nativeFinFoldlLoop.eq_1, dite_eq_right hIndex]

/--
The four-scalar native loop is exactly the ordinary native fold using
`updateTile4`. Every lane therefore retains its original reduction order.
-/
theorem nativeFinSum4_eq_nativeFinFoldl
    {α : Type u} [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (initial : Vector α 4) :
    nativeFinSum4 length bound hBound value0 value1 value2 value3 initial =
      nativeFinFoldl length bound hBound
        (fun totals index hIndex =>
          updateTile4 totals
            (value0 index hIndex) (value1 index hIndex)
            (value2 index hIndex) (value3 index hIndex))
        initial := by
  rw [nativeFinSum4, nativeFinFoldl]
  rw [nativeFinSum4Loop_eq_nativeFinFoldlLoop]
  have hInitial :
      #v[initial[0], initial[1], initial[2], initial[3]] = initial := by
    apply Vector.ext
    intro lane hLane
    have hLaneCases :
        lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 := by
      omega
    rcases hLaneCases with rfl | rfl | rfl | rfl <;> rfl
  rw [hInitial]

/--
Directly appending the scalar loop totals is exactly `pushTile4` applied to
the vector-valued loop result.
-/
theorem nativeFinSum4Push_eq_pushTile4
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (output : storage.Buffer) (initial : α) :
    nativeFinSum4Push length bound hBound
        value0 value1 value2 value3 output initial =
      pushTile4 output (
        nativeFinSum4 length bound hBound
          value0 value1 value2 value3
          (Vector.replicate 4 initial)) := by
  rw [nativeFinSum4Push, nativeFinSum4]
  refine nativeFinSum4PushLoop.induct
    length value0 value1 value2 value3
    bound hBound output
    (motive := fun index total0 total1 total2 total3 =>
      nativeFinSum4PushLoop length value0 value1 value2 value3
          bound hBound output index total0 total1 total2 total3 =
        pushTile4 output (
          nativeFinSum4Loop length value0 value1 value2 value3
            bound hBound index total0 total1 total2 total3))
    ?_ ?_ 0 initial initial initial initial
  · intro index total0 total1 total2 total3
      hIndex _hIndexNat inductionHypothesis
    rw [nativeFinSum4PushLoop.eq_1, dite_eq_left hIndex]
    rw [nativeFinSum4Loop.eq_1, dite_eq_left hIndex]
    exact inductionHypothesis
  · intro index total0 total1 total2 total3 hIndex
    rw [nativeFinSum4PushLoop.eq_1, dite_eq_right hIndex]
    rw [nativeFinSum4Loop.eq_1, dite_eq_right hIndex]
    rfl

/--
Run four scalar accumulators, apply one terminal function to each total, and
append the results directly to an existing output buffer.

The terminal functions are never called inside the contraction loop. This is
the execution path used when semiring laws move contraction-invariant factors
outside a shared tiled sum.
-/
@[specialize] def nativeFinSum4FinalizePushLoop
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (finalize0 finalize1 finalize2 finalize3 : α → α)
    (bound : USize) (hBound : bound.toNat = length)
    (output : storage.Buffer) (index : USize)
    (total0 total1 total2 total3 : α) : storage.Buffer :=
  if hIndex : index < bound then
    have _hIndexNat : index.toNat < length := by
      have := USize.lt_iff_toNat_lt.mp hIndex
      lia
    nativeFinSum4FinalizePushLoop length
      value0 value1 value2 value3
      finalize0 finalize1 finalize2 finalize3
      bound hBound output (index + 1)
      (total0 + value0 index _hIndexNat)
      (total1 + value1 index _hIndexNat)
      (total2 + value2 index _hIndexNat)
      (total3 + value3 index _hIndexNat)
  else
    let output := storage.push output (finalize0 total0)
    let output := storage.push output (finalize1 total1)
    let output := storage.push output (finalize2 total2)
    storage.push output (finalize3 total3)
termination_by length - index.toNat
decreasing_by
  have _hIndexNat : index.toNat < bound.toNat :=
    USize.lt_iff_toNat_lt.mp hIndex
  have hNextFits : index.toNat + 1 < USize.size := by
    have hBoundFits := USize.toNat_lt_size bound
    lia
  simp only [USize.toNat_add, USize.toNat_one,
    Nat.mod_eq_of_lt hNextFits]
  lia

/--
Sum four lanes and apply their terminal functions directly at the output
buffer boundary.
-/
@[inline] def nativeFinSum4FinalizePush
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (finalize0 finalize1 finalize2 finalize3 : α → α)
    (output : storage.Buffer) (initial : α) : storage.Buffer :=
  nativeFinSum4FinalizePushLoop length
    value0 value1 value2 value3
    finalize0 finalize1 finalize2 finalize3
    bound hBound output 0 initial initial initial initial

/--
The finalized four-lane append is exactly the vector-valued contraction
followed by lane-wise finalization and `pushTile4`.
-/
theorem nativeFinSum4FinalizePush_eq_pushTile4
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 :
      (index : USize) → index.toNat < length → α)
    (finalize0 finalize1 finalize2 finalize3 : α → α)
    (output : storage.Buffer) (initial : α) :
    nativeFinSum4FinalizePush length bound hBound
        value0 value1 value2 value3
        finalize0 finalize1 finalize2 finalize3 output initial =
      let totals :=
        nativeFinSum4 length bound hBound
          value0 value1 value2 value3
          (Vector.replicate 4 initial)
      pushTile4 output <| Vector.ofFn <|
        selectTile4
          (finalize0 totals[0]) (finalize1 totals[1])
          (finalize2 totals[2]) (finalize3 totals[3]) := by
  rw [nativeFinSum4FinalizePush, nativeFinSum4]
  refine nativeFinSum4FinalizePushLoop.induct
    length value0 value1 value2 value3
    finalize0 finalize1 finalize2
    bound hBound output
    (motive := fun index total0 total1 total2 total3 =>
      nativeFinSum4FinalizePushLoop length
          value0 value1 value2 value3
          finalize0 finalize1 finalize2 finalize3
          bound hBound output index total0 total1 total2 total3 =
        let totals :=
          nativeFinSum4Loop length value0 value1 value2 value3
            bound hBound index total0 total1 total2 total3
        pushTile4 output <| Vector.ofFn <|
          selectTile4
            (finalize0 totals[0]) (finalize1 totals[1])
            (finalize2 totals[2]) (finalize3 totals[3]))
    ?_ ?_ 0 initial initial initial initial
  · intro index total0 total1 total2 total3
      hIndex _hIndexNat inductionHypothesis
    rw [nativeFinSum4FinalizePushLoop.eq_1, dite_eq_left hIndex]
    rw [nativeFinSum4Loop.eq_1, dite_eq_left hIndex]
    exact inductionHypothesis
  · intro index total0 total1 total2 total3 hIndex
    rw [nativeFinSum4FinalizePushLoop.eq_1, dite_eq_right hIndex]
    rw [nativeFinSum4Loop.eq_1, dite_eq_right hIndex]
    simp [pushTile4, selectTile4]

end TorchLean.Tensor.Internal.Elab.Impl
