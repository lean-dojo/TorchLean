/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Pointwise

/-!
# Tensor Buffer Traversal

Laws that turn a fold over a tensor's physical buffer into a `Fin.foldl` over its row-major flat
indices, or into a fold over its coordinates. Reductions execute on the specialized storage, and
proofs reason about the ordinary finite traversal, which visits the entries in the same order.
The coordinate fold `Shape.Coord.foldl` visits the coordinates of a shape in the same row-major
order, as `Tensor.Rep.foldl_coordinates_eq_foldl` proves.
-/

@[expose] public section

namespace TorchLean

universe u

private theorem Einops.foldl_flatten_ofFn
    {α β : Type*} (outer inner : Nat)
    (entries : Fin outer → Fin inner → β)
    (step : α → β → α) (initial : α) :
    (List.ofFn fun outerIndex =>
        List.ofFn fun innerIndex =>
          entries outerIndex innerIndex).flatten.foldl step initial =
      Fin.foldl outer
        (fun value outerIndex =>
          Fin.foldl inner
            (fun value innerIndex =>
              step value (entries outerIndex innerIndex))
            value)
        initial := by
  rw [List.foldl_flatten]
  simp only [List.ofFn_eq_map, List.foldl_map,
    Fin.foldl_eq_foldl_finRange]

/-- Nested finite folds are one row-major fold over the product index. -/
theorem Native.fin_foldl_product
    {α : Type u} (outer inner : Nat)
    (step : α → Fin outer → Fin inner → α) (initial : α) :
    Fin.foldl outer
        (fun value outerIndex =>
          Fin.foldl inner
            (fun value innerIndex =>
              step value outerIndex innerIndex)
            value)
        initial =
      Fin.foldl (outer * inner)
        (fun value index =>
          step value index.divNat index.modNat)
        initial := by
  symm
  rw [Fin.foldl_eq_foldl_finRange,
    ← List.ofFn_id (outer * inner), List.ofFn_mul,
    Einops.foldl_flatten_ofFn]
  apply congrArg fun outerStep =>
    Fin.foldl outer outerStep initial
  funext value outerIndex
  apply congrArg fun innerStep =>
    Fin.foldl inner innerStep value
  funext innerValue innerIndex
  have hIndex :
      (⟨outerIndex * inner + innerIndex, by
        calc
          ↑outerIndex * inner + innerIndex <
              (outerIndex + 1) * inner := by
            rw [Nat.add_mul, Nat.one_mul]
            exact Nat.add_lt_add_left innerIndex.isLt _
          _ ≤ outer * inner :=
            Nat.mul_le_mul_right inner outerIndex.isLt⟩ :
        Fin (outer * inner)) =
        Fin.mkDivMod outerIndex innerIndex := by
    apply Fin.ext
    simp [Nat.mul_comm, Nat.add_comm]
  rw [hIndex]
  simp only [id_eq, Fin.divNat_mkDivMod, Fin.modNat_mkDivMod]

/--
Fold over the coordinates of a shape in row-major order.

Einsum contraction adds its values in this order, and the einsum compiler proves each loop nest it
generates equal to this fold.
-/
def Shape.Coord.foldl {α : Type u} :
    (shape : Shape) → (α → Coord shape → α) → α → α
  | [], step, initial => step initial PUnit.unit
  | length :: shape, step, initial =>
      Fin.foldl length
        (fun value head =>
          foldl shape
            (fun value tail => step value (head, tail))
            value)
        initial

namespace Tensor.Rep

open Shape (Coord)

/-- A packed fold is the finite fold over the tensor's flat row-major indices. -/
theorem foldl_eq_fin_foldl {α β : Type u} [Storage α] {s : Shape}
    (step : β → α → β) (initial : β) (x : Rep α s) :
    foldl step initial x =
      Fin.foldl (Shape.size s)
        (fun value index => step value (x.getFlat index)) initial := by
  rw [foldl_eq_data_foldl, ← Array.foldl_toList]
  have hData : x.data = Array.ofFn x.getFlat := by
    apply Array.ext
    · simp
    · intro index hLeft hRight
      let flatIndex : Fin (Shape.size s) := ⟨index, by simpa using hLeft⟩
      simpa [flatIndex] using x.data_getFlat flatIndex
  rw [hData, Array.toList_ofFn, List.ofFn_eq_map, List.foldl_map,
    ← Fin.foldl_eq_foldl_finRange]

/-- A map `state` that turns each `step` into `step'` on the mapped element turns the whole fold of
`input` into the fold of `input.map element`. Both folds visit the entries in the same order, and
the state and element types may differ between the two sides. -/
theorem map_foldl {α β γ δ : Type u} [Storage α] [Storage β] {s : Shape}
    (input : Rep α s) (step : γ → α → γ) (initial : γ)
    (element : α → β) (state : γ → δ) (step' : δ → β → δ)
    (hstep : ∀ accumulator value,
      state (step accumulator value) = step' (state accumulator) (element value)) :
    state (foldl step initial input) = foldl step' (state initial) (map element input) := by
  simp only [foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  symm
  apply List.foldl_hom state
  intro accumulator index
  simpa only [map, getFlat_ofFlatFn] using (hstep accumulator (input.getFlat index)).symm

/-- Folding a mapped tensor is folding the original tensor with the map fused into the step, in the
same order. -/
theorem foldl_map {α β γ : Type u} [Storage α] [Storage β] {s : Shape}
    (f : α → β) (input : Rep α s) (step : γ → β → γ) (initial : γ) :
    foldl step initial (map f input) =
      foldl (fun state value => step state (f value)) initial input := by
  simpa using (map_foldl input (fun state value => step state (f value)) initial
    f id step (fun _ _ => rfl)).symm

/-- The tensor that stores each coordinate of `shape` at that coordinate. Folding it visits the
coordinates in the row-major order of a tensor reduction. -/
def coordinates (shape : Shape) : Tensor (Coord shape) shape := ofFn id

/-- Flat and structured coordinates observe the same stored entry. -/
theorem getFlat_eq_get {α : Type u} [Storage α] {shape : Shape}
    (input : Rep α shape) (index : Fin (Shape.size shape)) :
    input.getFlat index = input.get (Coord.unlinearize index) := by
  simp only [get, Coord.linearize_unlinearize]

/-- Folding a tensor is folding its coordinates and reading the entry at each one. Rewriting with it
exposes the scalar computation of whatever produced the tensor, without treating that output as an
opaque source tensor. -/
theorem foldl_eq_coordinates {α β : Type} [Storage α] {shape : Shape}
    (step : β → α → β) (initial : β) (input : Rep α shape) :
    foldl step initial input =
      (coordinates shape).foldl
        (fun state coordinate => step state (input coordinate)) initial := by
  simp only [foldl_eq_fin_foldl, coordinates, ofFn, getFlat_ofFlatFn, id_eq,
    get, Coord.linearize_unlinearize]

/-- A generated tensor is its generator mapped over the coordinates tensor. Rewriting with it
exposes the generator's arithmetic without evaluating or unrolling any entry. -/
theorem ofFn_eq_map_coordinates {α : Type} [Storage α] {shape : Shape}
    (values : Coord shape → α) :
    ofFn values = (coordinates shape).map values := by
  apply Rep.ext
  intro coordinate
  simp [coordinates]

/-- A `zipWith` is one map over the coordinates tensor that reads both inputs at each coordinate,
even when the two inputs use different storage. -/
theorem zipWith_eq_map_coordinates {α β γ : Type} [Storage α] [Storage β] [Storage γ]
    {shape : Shape} (f : α → β → γ) (left : Rep α shape) (right : Rep β shape) :
    zipWith f left right = (coordinates shape).map (fun i => f (left i) (right i)) := by
  apply Rep.ext
  intro coordinate
  simp [coordinates]

/-- If `relation` holds between the initial states and every pair of steps at the same coordinate
preserves it, then it holds between the two folds. The relation can carry an error budget or
range information. -/
theorem foldl_rel {α β γ δ : Type u} [Storage α] [Storage β] {s : Shape}
    (input : Rep α s) (input' : Rep β s) (step : γ → α → γ) (step' : δ → β → δ)
    (relation : γ → δ → Prop) (initial : γ) (initial' : δ)
    (hinit : relation initial initial')
    (hstep : ∀ (index : Coord s) accumulator accumulator',
      relation accumulator accumulator' →
        relation (step accumulator (input index)) (step' accumulator' (input' index))) :
    relation (foldl step initial input) (foldl step' initial' input') := by
  simp only [foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  apply List.foldl_rel hinit
  intro index _ accumulator accumulator' hacc
  simpa only [get, Coord.linearize_unlinearize] using
    hstep (Coord.unlinearize index) accumulator accumulator' hacc

/-- Flat lookup into a stack selects the corresponding row and inner flat index. -/
theorem getFlat_stack_product {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) (outer : Fin n)
    (inner : Fin (Shape.size s)) :
    (dim components).getFlat
        (Fin.cast (Shape.size_cons n s).symm (finProdFinEquiv (outer, inner))) =
      (components outer).getFlat inner := by
  let index : Fin (Shape.size (n :: s)) :=
    Fin.cast (Shape.size_cons n s).symm (finProdFinEquiv (outer, inner))
  have hCoordinate :
      Coord.unlinearize (s := n :: s) index =
        (outer, Coord.unlinearize (s := s) inner) := by
    apply Coord.linearize_injective
    rw [Coord.linearize_unlinearize, Coord.linearize_cons,
      Coord.linearize_unlinearize]
    apply Fin.ext
    simp [index]
  calc
    (dim components).getFlat index =
        (dim components) (Coord.unlinearize (s := n :: s) index) := by
      change _ = (dim components).getFlat
        (Coord.linearize (Coord.unlinearize index))
      rw [Coord.linearize_unlinearize]
    _ = (components outer) (Coord.unlinearize inner) := by
      rw [hCoordinate, dim_apply]
    _ = (components outer).getFlat inner := by
      change (components outer).getFlat (Coord.linearize (Coord.unlinearize inner)) = _
      rw [Coord.linearize_unlinearize]

/-- Folding a stack is folding each row in leading-axis order. -/
theorem foldl_stack {α β : Type u} [Storage α] {n : Nat} {s : Shape}
    (step : β → α → β) (initial : β)
    (components : Fin n → Rep α s) :
    (dim components).foldl step initial =
      Fin.foldl n
        (fun value component => foldl step value (components component))
        initial := by
  rw [foldl_eq_fin_foldl]
  symm
  simp_rw [foldl_eq_fin_foldl]
  rw [Native.fin_foldl_product]
  apply congrArg (fun next => Fin.foldl (n * Shape.size s) next initial)
  funext value index
  apply congrArg (step value)
  have hProduct :
      finProdFinEquiv (index.divNat, index.modNat) = index := by
    exact (finProdFinEquiv :
      Fin n × Fin (Shape.size s) ≃ Fin (n * Shape.size s)).apply_symm_apply index
  have hIndex :
      Fin.cast (Shape.size_cons n s).symm
          (finProdFinEquiv (index.divNat, index.modNat)) =
        index := by
    apply Fin.ext
    simpa using congrArg Fin.val hProduct
  have hStack :=
    getFlat_stack_product components index.divNat index.modNat
  rw [hIndex] at hStack
  exact hStack.symm

/-- Folding the coordinate tensor of a shape is the row-major coordinate fold `Coord.foldl`. -/
theorem foldl_coordinates_eq_foldl {β : Type} :
    (s : Shape) → (step : β → Coord s → β) → (initial : β) →
      (coordinates s).foldl step initial = Coord.foldl s step initial
  | [], step, initial => by
      rw [foldl_eq_fin_foldl]
      rfl
  | n :: s, step, initial => by
      have stacked : coordinates (n :: s) =
          dim fun index => (coordinates s).map fun coordinate =>
            ((index, coordinate) : Coord (n :: s)) := by
        apply Rep.ext
        rintro ⟨index, coordinate⟩
        simp [coordinates]
      rw [stacked, foldl_stack, Coord.foldl]
      congr 1
      funext state index
      rw [foldl_map]
      exact foldl_coordinates_eq_foldl s _ state

/-- The coordinates of the rank-zero shape are folded in one step. -/
theorem foldl_coordinates_nil {β : Type} (step : β → Coord [] → β) (initial : β) :
    (coordinates []).foldl step initial = step initial PUnit.unit :=
  foldl_coordinates_eq_foldl [] step initial

/-- Folding the coordinates of a shape with a leading axis folds, for each index of that axis in
order, the coordinates of the remaining shape. -/
theorem foldl_coordinates_cons {β : Type} (n : Nat) (s : Shape)
    (step : β → Coord (n :: s) → β) (initial : β) :
    (coordinates (n :: s)).foldl step initial =
      (List.finRange n).foldl (fun state index =>
        (coordinates s).foldl (fun state coordinate => step state (index, coordinate)) state)
        initial := by
  rw [foldl_coordinates_eq_foldl, Coord.foldl, Fin.foldl_eq_foldl_finRange]
  simp only [foldl_coordinates_eq_foldl]

end Tensor.Rep

end TorchLean
