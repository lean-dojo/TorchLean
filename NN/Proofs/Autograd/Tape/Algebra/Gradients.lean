/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Contributions
public import NN.Proofs.Autograd.Tape.Algebra.Deferred

/-!
Gradient references carry the version of their last materialization. A uniform contribution updates
one shared history; selected parent gradients replay their pending updates before receiving the
node's complete local contribution. All scalar additions retain their dense evaluation order.
-/

@[expose] public section

namespace Proofs.Autograd.Algebra

open Spec TorchLean

namespace Deferred

variable {α : Type} [Storage α] [Add α] [DecidableEq α]

/-- Add a uniform tensor on the right, exactly as in dense context accumulation. -/
def addUniform {shape : Shape} (uniform : α) (value : Tensor α shape) : Tensor α shape :=
  Tensor.addSpec value (Tensor.full shape uniform)

/-- Materialize a typed boundary in one traversal, with constant-time version lookup. -/
def materializePack (history : History α) (clock : Nat) (epochs : Array Nat) :
    {shapes : List Shape} → TorchLean.TensorPack α shapes → Nat →
      TorchLean.TensorPack α shapes
  | [], .nil, _ => .nil
  | _ :: _, .cons value rest, offset =>
      .cons (applyRecentStable addUniform history (clock - epochs[offset]?.getD 0) value)
        (materializePack history clock epochs rest (offset + 1))

theorem getIdx_materializePack (history : History α) (clock : Nat) (epochs : Array Nat)
    {shapes : List Shape} (xs : TorchLean.TensorPack α shapes) (offset : Nat)
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

theorem materializePack_snoc (history : History α) (clock : Nat) (epochs : Array Nat)
    {shapes : List Shape} (xs : TorchLean.TensorPack α shapes) (offset : Nat)
    {shape : Shape} (value : Tensor α shape) :
    materializePack history clock epochs (xs.snoc value) offset =
      (materializePack history clock epochs xs offset).snoc
        (applyRecentStable addUniform history (clock - epochs[offset + shapes.length]?.getD 0)
          value) := by
  induction shapes generalizing offset with
  | nil => cases xs; simp [materializePack, TorchLean.TensorPack.snoc]
  | cons head shapes ih =>
      cases xs with
      | cons value rest =>
          simpa [materializePack, TorchLean.TensorPack.snoc, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using congrArg (TorchLean.TensorPack.cons
              (applyRecentStable addUniform history (clock - epochs[offset]?.getD 0) value))
                (ih rest (offset + 1))

end Deferred

/-- Array gradients and a shared chronological history of deferred uniform additions. -/
structure GradientContext (α : Type) [Storage α] (shapes : List Shape) where
  /-- Most recently materialized tensor at each active position. -/
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
def ofPack (seed : TorchLean.TensorPack α shapes) : GradientContext α shapes :=
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
def toPack (ctx : GradientContext α shapes) : TorchLean.TensorPack α shapes :=
  Deferred.materializePack ctx.history ctx.clock ctx.epochs ctx.stored.toPack 0

@[simp] theorem lookup_toPack (ctx : GradientContext α shapes) :
    TensorLookup.ofPack ctx.toPack = ctx.lookup := by
  apply TensorLookup.ext
  funext shape idx
  simp only [TensorLookup.ofPack, toPack, Deferred.getIdx_materializePack, Nat.zero_add, lookup]
  rw [← TensorContext.lookup_toPack]
  rfl

@[simp] theorem lookup_ofPack (seed : TorchLean.TensorPack α shapes) :
    (ofPack seed).lookup = TensorLookup.ofPack seed := by
  apply TensorLookup.ext
  funext shape idx
  simp only [lookup, ofPack, Deferred.applyRecentStable, TensorContext.lookup_ofPack]

@[simp] theorem toPack_ofPack (seed : TorchLean.TensorPack α shapes) :
    (ofPack seed).toPack = seed := by
  apply TorchLean.TensorPack.ext_getIdx
  intro shape idx
  exact congrArg (fun (reader : TensorLookup α shapes) => reader.read idx)
    ((lookup_toPack _).trans (lookup_ofPack seed))

/-- Defer one uniform contribution to the active prefix. -/
def addUniform (ctx : GradientContext α shapes) (uniform : α) : GradientContext α shapes :=
  { ctx with
    clock := ctx.clock + 1
    history := Deferred.push ctx.history uniform
    bounded := fun i bound => Nat.le_trans (ctx.bounded i bound) (Nat.le_succ _) }

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
      TorchLean.TensorPack.add ctx.toPack (contribution.dense ()) := by
  apply TorchLean.TensorPack.ext_getIdx
  intro shape idx
  have same : TensorLookup.ofPack (ctx.addContributions contribution).toPack =
      TensorLookup.ofPack (TorchLean.TensorPack.add ctx.toPack (contribution.dense ())) := by
    rw [lookup_toPack, lookup_addContributions, TensorLookup.ofPack_add,
      lookup_toPack, contribution.correct]
  exact congrArg (fun (reader : TensorLookup α shapes) => reader.read idx) same

/-- Custom dense VJPs remain supported through an exact materializing adapter. -/
def addDense (ctx : GradientContext α shapes) (contribution : TorchLean.TensorPack α shapes) :
    GradientContext α shapes :=
  ofPack (TorchLean.TensorPack.add ctx.toPack contribution)

@[simp] theorem toPack_addDense (ctx : GradientContext α shapes)
    (contribution : TorchLean.TensorPack α shapes) :
    (ctx.addDense contribution).toPack = TorchLean.TensorPack.add ctx.toPack contribution :=
  toPack_ofPack _

/-- Transport the typed active prefix without touching gradient storage. -/
def cast {other : List Shape} (same : shapes = other) (ctx : GradientContext α shapes) :
    GradientContext α other := same ▸ ctx

@[simp] theorem toPack_cast {other : List Shape} (same : shapes = other)
    (ctx : GradientContext α shapes) :
    (ctx.cast same).toPack = TorchLean.TensorPack.cast same ctx.toPack := by
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
    conv_lhs => rw [← TorchLean.TensorPack.snoc_unsnoc ctx.stored.toPack, ← parts]
  simp only [pop, toPack]
  rw [whole, Deferred.materializePack_snoc, TorchLean.TensorPack.unsnoc_snoc]
  simp only [TensorContext.size_values, Nat.zero_add]

end GradientContext

end Proofs.Autograd.Algebra
