/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Logic.Function.Iterate
public import TorchLean.NN.TypedGraph.Context

/-!
# Deferred updates and gradient contexts

A `GradientContext` stores the last materialized gradient at each context position with the
version of the shared `Deferred.History` it has seen. A uniform contribution only extends the
history, and a gradient replays the updates since its version when it is read
(`GradientContext.lookup`), so every scalar addition keeps the order of the dense pass.

Repeated updates stop early at a checked fixed point (`Deferred.repeatStable`), using decidable
equality and no additive identity. Repetition counts are `Nat.iterate` (`step^[count]`), which
applies the updates in chronological order.

## Main declarations

- `TorchLean.Tape.Deferred.History`, with `Deferred.push` and `Deferred.applyRecent`.
- `TorchLean.Tape.GradientContext`, with `GradientContext.lookup`,
  `GradientContext.addContributions` and `GradientContext.toPack`.
-/

@[expose] public section

namespace TorchLean.Tape

namespace Deferred

/-- Stop repeating an operation only after checking that its actual result is a fixed point. -/
def repeatStable {β : Type} [DecidableEq β] (step : β → β) : Nat → β → β
  | 0, value => value
  | count + 1, value =>
      let next := step value
      if next = value then next else repeatStable step count next

/-- Fixed-point compression preserves every finite sequence of repeated operations. -/
theorem repeatStable_eq {β : Type} [DecidableEq β] (step : β → β)
    (count : Nat) (value : β) :
    repeatStable step count value = step^[count] value := by
  induction count generalizing value with
  | zero => rfl
  | succ count ih =>
      rw [Function.iterate_succ_apply]
      simp only [repeatStable]
      split
      next fixed => rw [fixed, Function.iterate_fixed fixed]
      next _ => exact ih (step value)

/-- Consecutive occurrences of the same uniform contribution share one run. -/
abbrev History (α : Type) := List (α × Nat)

/-- Append one update to a history stored with its newest run first. -/
def push {α : Type} [DecidableEq α] (history : History α) (value : α) : History α :=
  match history with
  | [] => [(value, 1)]
  | (previous, count) :: rest =>
      if previous = value then (previous, count + 1) :: rest
      else (value, 1) :: history

/-- Apply the most recent `count` updates, in their original chronological order. -/
def applyRecent {α β : Type} (step : α → β → β) :
    History α → Nat → β → β
  | [], _, value => value
  | (uniform, runLength) :: rest, count, value =>
      (step uniform)^[min count runLength] (applyRecent step rest (count - runLength) value)

/-- Applying the zero most recent updates of a history leaves the value unchanged. -/
@[simp] theorem applyRecent_zero {α β : Type} (step : α → β → β)
    (history : History α) (value : β) : applyRecent step history 0 value = value := by
  induction history with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨uniform, count⟩
      simpa only [applyRecent, Nat.zero_min, Nat.zero_sub, Function.iterate_zero_apply] using ih

/-- Extending the history performs exactly one more update, after all previous updates. -/
theorem applyRecent_push {α β : Type} [DecidableEq α] (step : α → β → β)
    (history : History α) (uniform : α) (count : Nat) (value : β) :
    applyRecent step (push history uniform) (count + 1) value =
      step uniform (applyRecent step history count value) := by
  cases history with
  | nil => simp [push, applyRecent]
  | cons entry rest =>
      obtain ⟨previous, runLength⟩ := entry
      simp only [push]
      split
      next same =>
        subst previous
        have lengths : min (count + 1) (runLength + 1) = min count runLength + 1 := by omega
        simp only [applyRecent, Nat.add_sub_add_right, lengths, Function.iterate_succ_apply']
      next _ =>
        simp only [applyRecent, Nat.add_sub_cancel, Nat.min_eq_right (by omega : 1 ≤ count + 1),
          Function.iterate_one]

/-- Execute a deferred history with checked fixed-point compression in each run. -/
def applyRecentStable {α β : Type} [DecidableEq β] (step : α → β → β) :
    History α → Nat → β → β
  | [], _, value => value
  | (uniform, runLength) :: rest, count, value =>
      repeatStable (step uniform) (min count runLength)
        (applyRecentStable step rest (count - runLength) value)

/-- Compressed history execution preserves the complete uncompressed update sequence. -/
theorem applyRecentStable_eq {α β : Type} [DecidableEq β] (step : α → β → β)
    (history : History α) (count : Nat) (value : β) :
    applyRecentStable step history count value = applyRecent step history count value := by
  induction history generalizing count with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨uniform, runLength⟩
      simp only [applyRecentStable, applyRecent, ih, repeatStable_eq]

end Deferred

namespace Deferred

open TensorPack (getIdx)

variable {α : Type} [Storage α] [Add α] [DecidableEq α]

/-- Add a uniform tensor on the right, exactly as in dense context accumulation. -/
def addUniform {shape : Shape} (uniform : α) (value : Tensor α shape) : Tensor α shape :=
  value + Tensor.full shape uniform

/-- Materialize a typed boundary in one traversal, with constant-time version lookup. -/
def materializePack (history : History α) (clock : Nat) (epochs : Array Nat) :
    {shapes : List Shape} → TensorPack α shapes → Nat →
      TensorPack α shapes
  | [], .nil, _ => .nil
  | _ :: _, .cons value rest, offset =>
      .cons (applyRecentStable addUniform history (clock - epochs[offset]?.getD 0) value)
        (materializePack history clock epochs rest (offset + 1))

/-- The entry of `materializePack` at an index replays the updates issued since the epoch stored at
`offset` plus that index. -/
theorem getIdx_materializePack (history : History α) (clock : Nat) (epochs : Array Nat)
    {shapes : List Shape} (xs : TensorPack α shapes) (offset : Nat)
    {shape : Shape} (idx : Idx shapes shape) :
    getIdx (materializePack history clock epochs xs offset) idx =
      applyRecentStable addUniform history (clock - epochs[offset + idx.i.val]?.getD 0)
        (getIdx xs idx) := by
  obtain ⟨i, rfl⟩ := idx
  induction shapes generalizing offset with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
      cases xs with
      | cons value rest =>
          obtain ⟨i, bound⟩ := i
          cases i with
          | zero => rfl
          | succ i =>
              simpa [materializePack, getIdx, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
                using ih rest (offset + 1) ⟨i, Nat.lt_of_succ_lt_succ bound⟩

/-- Materializing `xs.snoc value` materializes `xs` and appends `value` with the updates issued
since the epoch stored at its position. -/
theorem materializePack_snoc (history : History α) (clock : Nat) (epochs : Array Nat)
    {shapes : List Shape} (xs : TensorPack α shapes) (offset : Nat)
    {shape : Shape} (value : Tensor α shape) :
    materializePack history clock epochs (xs.snoc value) offset =
      (materializePack history clock epochs xs offset).snoc
        (applyRecentStable addUniform history (clock - epochs[offset + shapes.length]?.getD 0)
          value) := by
  induction shapes generalizing offset with
  | nil => cases xs; simp [materializePack, Pack.snoc]
  | cons head shapes ih =>
      cases xs with
      | cons value rest =>
          simpa [materializePack, Pack.snoc, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using congrArg (Pack.cons
              (applyRecentStable addUniform history (clock - epochs[offset]?.getD 0) value))
                (ih rest (offset + 1))

end Deferred

/-- Array gradients and a shared chronological history of deferred uniform additions. -/
structure GradientContext (α : Type) [Storage α] (shapes : List Shape) where
  /-- Last materialized tensor at each active position. -/
  stored : TensorContext α shapes
  /-- History version already applied to each tensor. Finalized suffix entries may remain here. -/
  epochs : Array Nat
  /-- Number of uniform additions issued so far. -/
  clock : Nat
  /-- Consecutive identical contributions share one run. -/
  history : Deferred.History α
  /-- Every active tensor has a version slot. -/
  covered : stored.values.size ≤ epochs.size
  /-- No tensor version refers to an update that has not happened. -/
  bounded : ∀ i (bound : i < epochs.size), epochs[i] ≤ clock

namespace GradientContext

variable {α : Type} [Storage α] {shapes : List Shape}

/-- Start with a materialized seed and no deferred additions. -/
def ofPack (seed : TensorPack α shapes) : GradientContext α shapes :=
  let stored := TensorContext.ofPack seed
  { stored := stored
    epochs := Array.replicate stored.values.size 0
    clock := 0
    history := []
    covered := by simp
    bounded := by intro i bound; simp }

variable [Add α] [DecidableEq α]

/-- Read one gradient, replaying precisely the updates since its saved version. -/
def lookup (ctx : GradientContext α shapes) : TensorLookup α shapes :=
  ⟨fun idx => Deferred.applyRecentStable Deferred.addUniform ctx.history
    (ctx.clock - ctx.epochs[idx.i.val]?.getD 0) (ctx.stored.lookup.read idx)⟩

/-- Materialize only when crossing back into the public typed pack interface. -/
def toPack (ctx : GradientContext α shapes) : TensorPack α shapes :=
  Deferred.materializePack ctx.history ctx.clock ctx.epochs ctx.stored.toPack 0

/-- The reader of the materialized pack of a gradient context is its lookup. -/
@[simp] theorem lookup_toPack (ctx : GradientContext α shapes) :
    TensorLookup.ofPack ctx.toPack = ctx.lookup := by
  apply TensorLookup.ext
  funext shape idx
  simp only [TensorLookup.ofPack, toPack, Deferred.getIdx_materializePack, Nat.zero_add, lookup]
  rw [← TensorContext.lookup_toPack]
  rfl

/-- The gradient context of a seed pack reads the seed tensors. -/
@[simp] theorem lookup_ofPack (seed : TensorPack α shapes) :
    (ofPack seed).lookup = TensorLookup.ofPack seed := by
  apply TensorLookup.ext
  funext shape idx
  simp only [lookup, ofPack, Deferred.applyRecentStable, TensorContext.lookup_ofPack]

/-- Materializing the gradient context of a seed pack gives the seed. -/
@[simp] theorem toPack_ofPack (seed : TensorPack α shapes) :
    (ofPack seed).toPack = seed := by
  apply TensorPack.ext_getIdx
  intro shape idx
  exact congrArg (fun (reader : TensorLookup α shapes) => reader.read idx)
    ((lookup_toPack _).trans (lookup_ofPack seed))

/-- Defer one uniform contribution to the active prefix. -/
def addUniform (ctx : GradientContext α shapes) (uniform : α) : GradientContext α shapes :=
  { ctx with
    clock := ctx.clock + 1
    history := Deferred.push ctx.history uniform
    bounded := fun i bound => Nat.le_trans (ctx.bounded i bound) (Nat.le_succ _) }

/-- After `addUniform uniform` each entry reads as the entry of `ctx` plus the constant tensor of
`uniform`. -/
theorem lookup_addUniform (ctx : GradientContext α shapes) (uniform : α)
    {shape : Shape} (idx : Idx shapes shape) :
    (ctx.addUniform uniform).lookup.read idx =
      Deferred.addUniform uniform (ctx.lookup.read idx) := by
  have bound : idx.i.val < ctx.epochs.size :=
    Nat.lt_of_lt_of_le (by rw [TensorContext.size_values]; exact idx.i.isLt) ctx.covered
  have version : ctx.epochs[idx.i.val]?.getD 0 ≤ ctx.clock := by
    simpa only [Array.getElem?_eq_getElem bound, Option.getD_some] using ctx.bounded _ bound
  have count : ctx.clock + 1 - ctx.epochs[idx.i.val]?.getD 0 =
      (ctx.clock - ctx.epochs[idx.i.val]?.getD 0) + 1 := by omega
  simp only [lookup, addUniform, Deferred.applyRecentStable_eq, count, Deferred.applyRecent_push]

/-- Save an already materialized gradient and its current history version. -/
def set (ctx : GradientContext α shapes) {shape : Shape} (idx : Idx shapes shape)
    (value : Tensor α shape) : GradientContext α shapes :=
  -- `bound` discharges the in-bounds auto-parameter of `epochs.set` below.
  have bound : idx.i.val < ctx.epochs.size :=
    Nat.lt_of_lt_of_le (by rw [TensorContext.size_values]; exact idx.i.isLt) ctx.covered
  { ctx with
    stored := ctx.stored.set idx value
    epochs := ctx.epochs.set idx.i.val ctx.clock
    covered := by simpa only [TensorContext.set, Array.size_set] using ctx.covered
    bounded := by
      intro i within
      simp only [Array.getElem_set]
      split
      · exact Nat.le_refl _
      · exact ctx.bounded i (by simpa only [Array.size_set] using within) }

/-- Saving a gradient at an index updates the lookup of the context at that index. -/
@[simp] theorem lookup_set (ctx : GradientContext α shapes) {shape : Shape}
    (idx : Idx shapes shape) (value : Tensor α shape) :
    (ctx.set idx value).lookup = ctx.lookup.set idx value := by
  apply TensorLookup.ext
  funext otherShape other
  have bound : other.i.val < ctx.epochs.size :=
    Nat.lt_of_lt_of_le (by rw [TensorContext.size_values]; exact other.i.isLt) ctx.covered
  by_cases same : idx.i.val = other.i.val
  · have shapes := TensorLookup.shape_eq_of_index_eq idx other same
    cases shapes
    have positions : idx = other := by
      cases idx
      cases other
      simp only [Idx.mk.injEq]
      exact Fin.ext same
    subst other
    simp [lookup, set, Deferred.applyRecentStable_eq]
  · simp only [lookup, set, TensorContext.lookup_set, TensorLookup.read_set_other _ _ _ _ same,
      Array.getElem?_set, same, ite_false]

/-- A materialized update contains no reference to the full previous gradient context. -/
abbrev Update (α : Type) [Storage α] (shapes : List Shape) :=
  (shape : Shape) × (Idx shapes shape × Tensor α shape)

/-- Compute every exceptional result before modifying the gradient array. -/
def snapshot (indices : List (SomeIdx shapes)) (source : TensorLookup α shapes) :
    List (Update α shapes) :=
  indices.map (fun ⟨shape, idx⟩ => ⟨shape, idx, source.read idx⟩)

/-- Install materialized exceptions in their existing order. Duplicate writes are identical. -/
def setMany (ctx : GradientContext α shapes) : List (Update α shapes) → GradientContext α shapes
  | [] => ctx
  | ⟨_, idx, value⟩ :: rest => setMany (ctx.set idx value) rest

/-- After installing a snapshot of `source` at `indices`, the positions in `indices` read from
`source` and the other positions keep their gradients. -/
theorem lookup_setMany_snapshot (ctx : GradientContext α shapes)
    (indices : List (SomeIdx shapes)) (source : TensorLookup α shapes)
    {shape : Shape} (other : Idx shapes shape) :
    (ctx.setMany (snapshot indices source)).lookup.read other =
      if other.i.val ∈ indices.map (fun entry => entry.2.i.val)
      then source.read other else ctx.lookup.read other := by
  induction indices generalizing ctx with
  | nil => simp [snapshot, setMany]
  | cons entry indices ih =>
      obtain ⟨entryShape, idx⟩ := entry
      change ((ctx.set idx (source.read idx)).setMany (snapshot indices source)).lookup.read
          other =
        if other.i.val ∈ idx.i.val :: indices.map (fun entry => entry.2.i.val)
        then source.read other else ctx.lookup.read other
      rw [ih, lookup_set, TensorLookup.read_set_from]
      simp only [List.mem_cons]
      by_cases head : other.i.val = idx.i.val <;>
        by_cases tail : other.i.val ∈ indices.map (fun entry => entry.2.i.val) <;>
          simp only [head, tail] <;> simp

/-- Accumulate a node's complete local contribution, preserving each dense addition. -/
def addContributions (ctx : GradientContext α shapes) (contribution : Contributions α shapes) :
    GradientContext α shapes :=
  let updates := snapshot contribution.support (TensorLookup.add ctx.lookup contribution.lookup)
  (ctx.addUniform contribution.uniform).setMany updates

/-- Accumulating a contribution adds its lookup to the lookup of the context. -/
theorem lookup_addContributions (ctx : GradientContext α shapes)
    (contribution : Contributions α shapes) :
    (ctx.addContributions contribution).lookup =
      TensorLookup.add ctx.lookup contribution.lookup := by
  apply TensorLookup.ext
  funext shape idx
  simp only [addContributions, lookup_setMany_snapshot]
  split
  · rfl
  next absent =>
    have outside := contribution.outside idx (by
      intro entry member same
      exact absent (List.mem_map.mpr ⟨entry, member, same⟩))
    rw [lookup_addUniform]
    simp only [TensorLookup.add, outside, Deferred.addUniform]

/-- Compact accumulation equals full pack addition without an additive-identity assumption. -/
theorem toPack_addContributions (ctx : GradientContext α shapes)
    (contribution : Contributions α shapes) :
    (ctx.addContributions contribution).toPack =
      TensorPack.add ctx.toPack (contribution.dense ()) := by
  apply TensorPack.ext_getIdx
  intro shape idx
  have same : TensorLookup.ofPack (ctx.addContributions contribution).toPack =
      TensorLookup.ofPack (TensorPack.add ctx.toPack (contribution.dense ())) := by
    rw [lookup_toPack, lookup_addContributions, TensorLookup.ofPack_add,
      lookup_toPack, contribution.ofPack_dense]
  exact congrArg (fun (reader : TensorLookup α shapes) => reader.read idx) same

/-- Add a dense contribution, the path of a VJP without a compact program. The context is
materialized as a pack and stored again (`toPack_addDense`). -/
def addDense (ctx : GradientContext α shapes) (contribution : TensorPack α shapes) :
    GradientContext α shapes :=
  ofPack (TensorPack.add ctx.toPack contribution)

/-- Adding a dense contribution materializes to the sum of the materialized context and the
contribution. -/
@[simp] theorem toPack_addDense (ctx : GradientContext α shapes)
    (contribution : TensorPack α shapes) :
    (ctx.addDense contribution).toPack = TensorPack.add ctx.toPack contribution :=
  toPack_ofPack _

/-- Transport the typed active prefix without touching gradient storage. -/
def cast {other : List Shape} (same : shapes = other) (ctx : GradientContext α shapes) :
    GradientContext α other := same ▸ ctx

/-- Materializing a cast gradient context gives the cast of its materialized pack. -/
@[simp] theorem toPack_cast {other : List Shape} (same : shapes = other)
    (ctx : GradientContext α shapes) :
    (ctx.cast same).toPack = Pack.cast same ctx.toPack := by
  cases same
  rfl

/-- Finalize one output gradient and remove its slot from the active prefix. -/
def pop {shape : Shape} (ctx : GradientContext α (shapes ++ [shape])) :
    GradientContext α shapes × Tensor α shape :=
  -- Read the output index from the remaining array. Reading the original array's size below
  -- would keep it shared across `pop`, forcing a full copy on every reverse step.
  let parts := ctx.stored.pop
  ({ stored := parts.1
     epochs := ctx.epochs
     clock := ctx.clock
     history := ctx.history
     covered := by
       have covered := ctx.covered
       rw [TensorContext.size_values] at covered ⊢
       simp only [List.length_append, List.length_singleton] at covered
       omega
     bounded := ctx.bounded },
   Deferred.applyRecentStable Deferred.addUniform ctx.history
     (ctx.clock - ctx.epochs[parts.1.values.size]?.getD 0) parts.2)

/-- Removing a deferred gradient context agrees with removing the fully materialized pack. -/
theorem toPack_pop {shape : Shape} (ctx : GradientContext α (shapes ++ [shape])) :
    ((ctx.pop).1.toPack, (ctx.pop).2) = ctx.toPack.unsnoc := by
  have whole : ctx.stored.toPack =
      ((ctx.stored.pop).1.toPack).snoc (ctx.stored.pop).2 := by
    have parts := TensorContext.toPack_pop ctx.stored
    conv_lhs => rw [← Pack.snoc_unsnoc ctx.stored.toPack, ← parts]
  simp only [pop, toPack]
  rw [whole, Deferred.materializePack_snoc, Pack.unsnoc_snoc]
  simp only [TensorContext.size_values, Nat.zero_add]

end GradientContext

end TorchLean.Tape
