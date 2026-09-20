/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Tiling.Semantics
import Mathlib.Tactic.FinCases

/-!
# Eight-register output tiling

This module implements the wider concrete scalar-register lowering selected
for large contractions. Its update step refines the arbitrary-width semantics
from `Tiling.Semantics`; the width is an implementation choice rather than a
semantic restriction.

Every lane observes contracted coordinates in the same row-major order as the
scalar lowering, so tiling requires no reassociation, commutativity, or
distributivity argument.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u

/-- Select one of eight values by a statically bounded lane index. -/
def selectTile8 {α : Type u}
    (value0 value1 value2 value3 value4 value5 value6 value7 : α) :
    Fin 8 → α
  | lane =>
      match lane.val with
      | 0 => value0
      | 1 => value1
      | 2 => value2
      | 3 => value3
      | 4 => value4
      | 5 => value5
      | 6 => value6
      | _ => value7

/--
Selecting a concrete eight-lane contraction after summing each lane agrees
with summing the selected lane.
-/
theorem coordinateSum_selectTile8
    {α : Type u} [Add α] [OfNat α 0] (shape : Shape)
    (value0 value1 value2 value3
      value4 value5 value6 value7 : Coord shape → α) (initial : α) :
    (fun lane =>
      Semantics.coordinateSum shape
        (fun coordinate =>
          selectTile8
            (value0 coordinate) (value1 coordinate)
            (value2 coordinate) (value3 coordinate)
            (value4 coordinate) (value5 coordinate)
            (value6 coordinate) (value7 coordinate) lane)
        initial) =
      selectTile8
        (Semantics.coordinateSum shape value0 initial)
        (Semantics.coordinateSum shape value1 initial)
        (Semantics.coordinateSum shape value2 initial)
        (Semantics.coordinateSum shape value3 initial)
        (Semantics.coordinateSum shape value4 initial)
        (Semantics.coordinateSum shape value5 initial)
        (Semantics.coordinateSum shape value6 initial)
        (Semantics.coordinateSum shape value7 initial) := by
  funext lane
  fin_cases lane <;> rfl

/--
Update the proof-facing state of an eight-lane tile.

For large concrete contractions, `compileCoordinateFold` recognizes this
exact update and first emits `nativeFinSum8`, which carries the eight totals
as separate loop arguments. The surrounding output pass fuses an immediately
consumed result into `nativeFinSum8Push`. The scalar-state theorems prove both
forms compute this vector update exactly.
-/
@[inline] def updateTile8
    {α : Type u} [Add α] (totals : Vector α 8)
    (value0 value1 value2 value3 value4 value5 value6 value7 : α) :
    Vector α 8 :=
  let totals := totals.set 0 (totals[0] + value0)
  let totals := totals.set 1 (totals[1] + value1)
  let totals := totals.set 2 (totals[2] + value2)
  let totals := totals.set 3 (totals[3] + value3)
  let totals := totals.set 4 (totals[4] + value4)
  let totals := totals.set 5 (totals[5] + value5)
  let totals := totals.set 6 (totals[6] + value6)
  totals.set 7 (totals[7] + value7)

/-- Append one completed eight-lane tile to a row-major output buffer. -/
@[inline] def pushTile8
    {α : Type u} [storage : Storage α]
    (output : storage.Buffer) (values : Vector α 8) : storage.Buffer :=
  let output := storage.push output values[0]
  let output := storage.push output values[1]
  let output := storage.push output values[2]
  let output := storage.push output values[3]
  let output := storage.push output values[4]
  let output := storage.push output values[5]
  let output := storage.push output values[6]
  storage.push output values[7]

/-- Pointwise equal lane values determine equal eight-lane vectors. -/
theorem ofFn_selectTile8_congr
    {α : Type u}
    (value0 value1 value2 value3 value4 value5 value6 value7
      reference0 reference1 reference2 reference3
      reference4 reference5 reference6 reference7 : α)
    (h0 : value0 = reference0) (h1 : value1 = reference1)
    (h2 : value2 = reference2) (h3 : value3 = reference3)
    (h4 : value4 = reference4) (h5 : value5 = reference5)
    (h6 : value6 = reference6) (h7 : value7 = reference7) :
    Vector.ofFn
        (selectTile8 value0 value1 value2 value3
          value4 value5 value6 value7) =
      Vector.ofFn
        (selectTile8 reference0 reference1 reference2 reference3
          reference4 reference5 reference6 reference7) := by
  subst_vars
  rfl

/-- Appending an eight-lane function vector emits its values in lane order. -/
theorem pushTile8_ofFn
    {α : Type u} [storage : Storage α] (output : storage.Buffer)
    (value0 value1 value2 value3 value4 value5 value6 value7 : α) :
    pushTile8 output
        (Vector.ofFn
          (selectTile8 value0 value1 value2 value3
            value4 value5 value6 value7)) =
      let output := storage.push output value0
      let output := storage.push output value1
      let output := storage.push output value2
      let output := storage.push output value3
      let output := storage.push output value4
      let output := storage.push output value5
      let output := storage.push output value6
      storage.push output value7 := by
  simp only [pushTile8, Vector.getElem_ofFn, selectTile8]

/-- Projecting an eight-lane update gives the corresponding scalar update. -/
theorem updateTile8_get
    {α : Type u} [Add α] (totals : Vector α 8)
    (value0 value1 value2 value3 value4 value5 value6 value7 : α)
    (lane : Fin 8) :
    (updateTile8 totals value0 value1 value2 value3
      value4 value5 value6 value7)[lane] =
      totals[lane] +
        selectTile8 value0 value1 value2 value3
          value4 value5 value6 value7 lane := by
  fin_cases lane <;>
    simp only [Fin.getElem_fin, updateTile8, selectTile8,
      Vector.getElem_set, Nat.reduceEqDiff, reduceIte]

/--
The eight-register update implements the width-polymorphic tile update.
-/
theorem updateTile8_eq_updateTile
    {α : Type u} [Add α] (totals : Vector α 8)
    (value0 value1 value2 value3 value4 value5 value6 value7 : α) :
    updateTile8 totals value0 value1 value2 value3
        value4 value5 value6 value7 =
      updateTile totals (
        selectTile8 value0 value1 value2 value3
          value4 value5 value6 value7) := by
  apply Vector.ext
  intro lane hLane
  have hEight :=
    updateTile8_get totals value0 value1 value2 value3
      value4 value5 value6 value7 ⟨lane, hLane⟩
  have hGeneric :=
    updateTile_get totals
      (selectTile8 value0 value1 value2 value3
        value4 value5 value6 value7) ⟨lane, hLane⟩
  simp only [Fin.getElem_fin] at hEight hGeneric
  rw [hEight, hGeneric]

/--
Run a native finite sum while carrying eight lane totals as separate scalar
arguments.

The compiler selects this loop when eight neighboring outputs are available
and shared coordinate work amortizes the additional live state. Both concrete
tile widths advance one contraction coordinate per recursive step; the
eight-lane loop keeps twice as many independent accumulators live.
-/
@[specialize] def nativeFinSum8Loop
    {α : Type u} [Add α] (length : Nat)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize)
    (total0 total1 total2 total3 total4 total5 total6 total7 : α) :
    Vector α 8 :=
  if hIndex : index < bound then
    have _hIndexNat : index.toNat < length := by
      have := USize.lt_iff_toNat_lt.mp hIndex
      lia
    nativeFinSum8Loop length
      value0 value1 value2 value3 value4 value5 value6 value7
      bound hBound (index + 1)
      (total0 + value0 index _hIndexNat)
      (total1 + value1 index _hIndexNat)
      (total2 + value2 index _hIndexNat)
      (total3 + value3 index _hIndexNat)
      (total4 + value4 index _hIndexNat)
      (total5 + value5 index _hIndexNat)
      (total6 + value6 index _hIndexNat)
      (total7 + value7 index _hIndexNat)
  else
    #v[total0, total1, total2, total3, total4, total5, total6, total7]
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
Sum eight lanes over a native finite interval without vector operations inside
the loop.
-/
@[inline] def nativeFinSum8
    {α : Type u} [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (initial : Vector α 8) : Vector α 8 :=
  nativeFinSum8Loop length
    value0 value1 value2 value3 value4 value5 value6 value7
    bound hBound 0
    initial[0] initial[1] initial[2] initial[3]
    initial[4] initial[5] initial[6] initial[7]

/--
Run the eight scalar accumulators and append their final values directly to an
existing output buffer.

The terminal pushes avoid constructing a temporary result vector between the
contraction loop and the surrounding output loop.
-/
@[specialize] def nativeFinSum8PushLoop
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (output : storage.Buffer) (index : USize)
    (total0 total1 total2 total3 total4 total5 total6 total7 : α) :
    storage.Buffer :=
  if hIndex : index < bound then
    have _hIndexNat : index.toNat < length := by
      have := USize.lt_iff_toNat_lt.mp hIndex
      lia
    nativeFinSum8PushLoop length
      value0 value1 value2 value3 value4 value5 value6 value7
      bound hBound output (index + 1)
      (total0 + value0 index _hIndexNat)
      (total1 + value1 index _hIndexNat)
      (total2 + value2 index _hIndexNat)
      (total3 + value3 index _hIndexNat)
      (total4 + value4 index _hIndexNat)
      (total5 + value5 index _hIndexNat)
      (total6 + value6 index _hIndexNat)
      (total7 + value7 index _hIndexNat)
  else
    let output := storage.push output total0
    let output := storage.push output total1
    let output := storage.push output total2
    let output := storage.push output total3
    let output := storage.push output total4
    let output := storage.push output total5
    let output := storage.push output total6
    storage.push output total7
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
Sum eight lanes and append them without materializing the intermediate vector.
-/
@[inline] def nativeFinSum8Push
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (output : storage.Buffer) (initial : α) : storage.Buffer :=
  nativeFinSum8PushLoop length
    value0 value1 value2 value3 value4 value5 value6 value7
    bound hBound output 0
    initial initial initial initial
    initial initial initial initial

/--
Starting from any native index and eight scalar totals, the scalar-state loop
agrees with the corresponding vector-state native fold.
-/
private theorem nativeFinSum8Loop_eq_nativeFinFoldlLoop
    {α : Type u} [Add α] (length : Nat)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize)
    (total0 total1 total2 total3 total4 total5 total6 total7 : α) :
    nativeFinSum8Loop length
        value0 value1 value2 value3 value4 value5 value6 value7
        bound hBound index
        total0 total1 total2 total3 total4 total5 total6 total7 =
      nativeFinFoldlLoop length
        (fun totals index hIndex =>
          updateTile8 totals
            (value0 index hIndex) (value1 index hIndex)
            (value2 index hIndex) (value3 index hIndex)
            (value4 index hIndex) (value5 index hIndex)
            (value6 index hIndex) (value7 index hIndex))
        bound hBound index
        #v[total0, total1, total2, total3,
          total4, total5, total6, total7] := by
  refine nativeFinSum8Loop.induct
    length value0 value1 value2 value3
    value4 value5 value6 value7 bound hBound
    (motive := fun index
        total0 total1 total2 total3 total4 total5 total6 total7 =>
      nativeFinSum8Loop length
          value0 value1 value2 value3 value4 value5 value6 value7
          bound hBound index
          total0 total1 total2 total3 total4 total5 total6 total7 =
        nativeFinFoldlLoop length
          (fun totals index hIndex =>
            updateTile8 totals
              (value0 index hIndex) (value1 index hIndex)
              (value2 index hIndex) (value3 index hIndex)
              (value4 index hIndex) (value5 index hIndex)
              (value6 index hIndex) (value7 index hIndex))
          bound hBound index
          #v[total0, total1, total2, total3,
            total4, total5, total6, total7])
    ?_ ?_ index
    total0 total1 total2 total3 total4 total5 total6 total7
  · intro index
      total0 total1 total2 total3 total4 total5 total6 total7
      hIndex _hIndexNat inductionHypothesis
    rw [nativeFinSum8Loop.eq_1, dite_eq_left hIndex]
    rw [nativeFinFoldlLoop.eq_1, dite_eq_left hIndex]
    simpa [updateTile8] using inductionHypothesis
  · intro index
      total0 total1 total2 total3 total4 total5 total6 total7 hIndex
    rw [nativeFinSum8Loop.eq_1, dite_eq_right hIndex]
    rw [nativeFinFoldlLoop.eq_1, dite_eq_right hIndex]

/--
The eight-scalar native loop is exactly the ordinary native fold using
`updateTile8`. In particular, every lane retains its original reduction order.
-/
theorem nativeFinSum8_eq_nativeFinFoldl
    {α : Type u} [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (initial : Vector α 8) :
    nativeFinSum8 length bound hBound
        value0 value1 value2 value3
        value4 value5 value6 value7 initial =
      nativeFinFoldl length bound hBound
        (fun totals index hIndex =>
          updateTile8 totals
            (value0 index hIndex) (value1 index hIndex)
            (value2 index hIndex) (value3 index hIndex)
            (value4 index hIndex) (value5 index hIndex)
            (value6 index hIndex) (value7 index hIndex))
        initial := by
  rw [nativeFinSum8, nativeFinFoldl]
  rw [nativeFinSum8Loop_eq_nativeFinFoldlLoop]
  have hInitial :
      #v[initial[0], initial[1], initial[2], initial[3],
        initial[4], initial[5], initial[6], initial[7]] = initial := by
    apply Vector.ext
    intro lane hLane
    have hLaneCases :
        lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 ∨
          lane = 4 ∨ lane = 5 ∨ lane = 6 ∨ lane = 7 := by
      omega
    rcases hLaneCases with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> rfl
  rw [hInitial]

/--
Directly appending the scalar loop totals is exactly `pushTile8` applied to
the vector-valued loop result.
-/
theorem nativeFinSum8Push_eq_pushTile8
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (output : storage.Buffer) (initial : α) :
    nativeFinSum8Push length bound hBound
        value0 value1 value2 value3
        value4 value5 value6 value7 output initial =
      pushTile8 output (
        nativeFinSum8 length bound hBound
          value0 value1 value2 value3
          value4 value5 value6 value7
          (Vector.replicate 8 initial)) := by
  rw [nativeFinSum8Push, nativeFinSum8]
  refine nativeFinSum8PushLoop.induct
    length value0 value1 value2 value3 value4 value5 value6 value7
    bound hBound output
    (motive := fun index
        total0 total1 total2 total3 total4 total5 total6 total7 =>
      nativeFinSum8PushLoop length
          value0 value1 value2 value3 value4 value5 value6 value7
          bound hBound output index
          total0 total1 total2 total3 total4 total5 total6 total7 =
        pushTile8 output (
          nativeFinSum8Loop length
            value0 value1 value2 value3 value4 value5 value6 value7
            bound hBound index
            total0 total1 total2 total3
            total4 total5 total6 total7))
    ?_ ?_ 0
    initial initial initial initial
    initial initial initial initial
  · intro index
      total0 total1 total2 total3 total4 total5 total6 total7
      hIndex _hIndexNat inductionHypothesis
    rw [nativeFinSum8PushLoop.eq_1, dite_eq_left hIndex]
    rw [nativeFinSum8Loop.eq_1, dite_eq_left hIndex]
    exact inductionHypothesis
  · intro index
      total0 total1 total2 total3 total4 total5 total6 total7 hIndex
    rw [nativeFinSum8PushLoop.eq_1, dite_eq_right hIndex]
    rw [nativeFinSum8Loop.eq_1, dite_eq_right hIndex]
    rfl

/--
Run eight scalar accumulators, apply one terminal function to each total, and
append the results directly to an existing output buffer.

The terminal functions run only after the contraction has completed. This
keeps factored semiring operands out of the hot loop without introducing an
intermediate result vector.
-/
@[specialize] def nativeFinSum8FinalizePushLoop
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (finalize0 finalize1 finalize2 finalize3
      finalize4 finalize5 finalize6 finalize7 : α → α)
    (bound : USize) (hBound : bound.toNat = length)
    (output : storage.Buffer) (index : USize)
    (total0 total1 total2 total3 total4 total5 total6 total7 : α) :
    storage.Buffer :=
  if hIndex : index < bound then
    have _hIndexNat : index.toNat < length := by
      have := USize.lt_iff_toNat_lt.mp hIndex
      lia
    nativeFinSum8FinalizePushLoop length
      value0 value1 value2 value3 value4 value5 value6 value7
      finalize0 finalize1 finalize2 finalize3
      finalize4 finalize5 finalize6 finalize7
      bound hBound output (index + 1)
      (total0 + value0 index _hIndexNat)
      (total1 + value1 index _hIndexNat)
      (total2 + value2 index _hIndexNat)
      (total3 + value3 index _hIndexNat)
      (total4 + value4 index _hIndexNat)
      (total5 + value5 index _hIndexNat)
      (total6 + value6 index _hIndexNat)
      (total7 + value7 index _hIndexNat)
  else
    let output := storage.push output (finalize0 total0)
    let output := storage.push output (finalize1 total1)
    let output := storage.push output (finalize2 total2)
    let output := storage.push output (finalize3 total3)
    let output := storage.push output (finalize4 total4)
    let output := storage.push output (finalize5 total5)
    let output := storage.push output (finalize6 total6)
    storage.push output (finalize7 total7)
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
Sum eight lanes and apply their terminal functions directly at the output
buffer boundary.
-/
@[inline] def nativeFinSum8FinalizePush
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (finalize0 finalize1 finalize2 finalize3
      finalize4 finalize5 finalize6 finalize7 : α → α)
    (output : storage.Buffer) (initial : α) : storage.Buffer :=
  nativeFinSum8FinalizePushLoop length
    value0 value1 value2 value3 value4 value5 value6 value7
    finalize0 finalize1 finalize2 finalize3
    finalize4 finalize5 finalize6 finalize7
    bound hBound output 0
    initial initial initial initial
    initial initial initial initial

/--
Applying terminal functions at the end of the eight-scalar loop is equivalent
to mapping those functions over the vector-valued loop result before append.
-/
theorem nativeFinSum8FinalizePush_eq_pushTile8
    {α : Type u} [storage : Storage α] [Add α] (length : Nat)
    (bound : USize) (hBound : bound.toNat = length)
    (value0 value1 value2 value3 value4 value5 value6 value7 :
      (index : USize) → index.toNat < length → α)
    (finalize0 finalize1 finalize2 finalize3
      finalize4 finalize5 finalize6 finalize7 : α → α)
    (output : storage.Buffer) (initial : α) :
    nativeFinSum8FinalizePush length bound hBound
        value0 value1 value2 value3
        value4 value5 value6 value7
        finalize0 finalize1 finalize2 finalize3
        finalize4 finalize5 finalize6 finalize7 output initial =
      let totals :=
        nativeFinSum8 length bound hBound
          value0 value1 value2 value3
          value4 value5 value6 value7
          (Vector.replicate 8 initial)
      pushTile8 output <| Vector.ofFn <|
        selectTile8
          (finalize0 totals[0]) (finalize1 totals[1])
          (finalize2 totals[2]) (finalize3 totals[3])
          (finalize4 totals[4]) (finalize5 totals[5])
          (finalize6 totals[6]) (finalize7 totals[7]) := by
  rw [nativeFinSum8FinalizePush, nativeFinSum8]
  refine nativeFinSum8FinalizePushLoop.induct
    length value0 value1 value2 value3 value4 value5 value6 value7
    finalize0 finalize1 finalize2 finalize3
    finalize4 finalize5 finalize6
    bound hBound output
    (motive := fun index
        total0 total1 total2 total3 total4 total5 total6 total7 =>
      nativeFinSum8FinalizePushLoop length
          value0 value1 value2 value3 value4 value5 value6 value7
          finalize0 finalize1 finalize2 finalize3
          finalize4 finalize5 finalize6 finalize7
          bound hBound output index
          total0 total1 total2 total3 total4 total5 total6 total7 =
        let totals :=
          nativeFinSum8Loop length
            value0 value1 value2 value3 value4 value5 value6 value7
            bound hBound index
            total0 total1 total2 total3
            total4 total5 total6 total7
        pushTile8 output <| Vector.ofFn <|
          selectTile8
            (finalize0 totals[0]) (finalize1 totals[1])
            (finalize2 totals[2]) (finalize3 totals[3])
            (finalize4 totals[4]) (finalize5 totals[5])
            (finalize6 totals[6]) (finalize7 totals[7]))
    ?_ ?_ 0
    initial initial initial initial
    initial initial initial initial
  · intro index
      total0 total1 total2 total3 total4 total5 total6 total7
      hIndex _hIndexNat inductionHypothesis
    rw [nativeFinSum8FinalizePushLoop.eq_1, dite_eq_left hIndex]
    rw [nativeFinSum8Loop.eq_1, dite_eq_left hIndex]
    exact inductionHypothesis
  · intro index
      total0 total1 total2 total3 total4 total5 total6 total7 hIndex
    rw [nativeFinSum8FinalizePushLoop.eq_1, dite_eq_right hIndex]
    rw [nativeFinSum8Loop.eq_1, dite_eq_right hIndex]
    simp [pushTile8, selectTile8]

end TorchLean.Tensor.Internal.Elab.Impl
