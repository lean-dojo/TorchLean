/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPBasic
public import NN.MLTheory.CROWN.Proofs.LayerNormEnclosure

/-!
# Tensor enclosure for structural interval transfers

Coordinatewise enclosure is preserved by changes of shape, broadcasting, and directed axis
summation. These lemmas use the tensor operations executed by the forward IBP dispatcher.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- A shaped endpoint pair encloses a real tensor at every coordinate. -/
def TensorEncloses {s : Shape} (lo hi : Tensor α s) (x : Tensor ℝ s) : Prop :=
  ∀ c : s.Coord, value (lo c) ≤ x c ∧ x c ≤ value (hi c)

theorem TensorEncloses.cast {s t : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) (hst : s = t) :
    TensorEncloses (hst ▸ lo) (hst ▸ hi) (hst ▸ x) := by
  subst t
  exact h

/-- Reconstruct a real tensor from its row-major graph coordinates. -/
def realTensor (s : Shape) (f : Nat → ℝ) : Tensor ℝ s :=
  Tensor.unflattenSpec s (Tensor.ofFn fun i => f i.val)

/-- Read the flat coordinates of a real tensor, extending them by zero out of bounds. -/
def tensorValues {s : Shape} (x : Tensor ℝ s) (i : Nat) : ℝ :=
  getAtOrZero (Tensor.flattenSpec x) [i]

@[simp] theorem tensorValues_fin {s : Shape} (x : Tensor ℝ s) (i : Fin s.size) :
    tensorValues x i.val = (Tensor.flattenSpec x).getScalar i :=
  read_fin _ i

@[simp] theorem getScalar_flatten_realTensor (s : Shape) (f : Nat → ℝ) (i : Fin s.size) :
    (Tensor.flattenSpec (realTensor s f)).getScalar i = f i.val := by
  simp [realTensor]

theorem TensorEncloses.unstack {n : Nat} {s : Shape}
    {lo hi : Tensor α (.dim n s)} {x : Tensor ℝ (.dim n s)}
    (h : TensorEncloses lo hi x) (i : Fin n) :
    TensorEncloses (lo.unstack i) (hi.unstack i) (x.unstack i) := by
  intro c
  simpa only [Tensor.unstack, Tensor.Internal.Rep.unstack_apply] using h (i, c)

theorem TensorEncloses.dim {n : Nat} {s : Shape}
    {lo hi : Fin n → Tensor α s} {x : Fin n → Tensor ℝ s}
    (h : ∀ i, TensorEncloses (lo i) (hi i) (x i)) :
    TensorEncloses (Tensor.dim lo) (Tensor.dim hi) (Tensor.dim x) := by
  rintro ⟨i, c⟩
  simpa only [Tensor.dim, Tensor.Internal.Rep.stack_apply] using h i c

theorem TensorEncloses.reshape {s t : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x)
    (hsize : Tensor.Internal.Shape.size s.toList = Tensor.Internal.Shape.size t.toList) :
    TensorEncloses (Tensor.Internal.Rep.reshape hsize lo)
      (Tensor.Internal.Rep.reshape hsize hi) (Tensor.Internal.Rep.reshape hsize x) := by
  intro c
  simpa only [Tensor.Internal.Rep.reshape_apply_coordEquiv] using
    h (Tensor.Internal.Rep.reshapeCoordEquiv hsize c)

theorem TensorEncloses.flatten {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) :
    TensorEncloses (Tensor.flattenSpec lo) (Tensor.flattenSpec hi) (Tensor.flattenSpec x) :=
  h.reshape (Tensor.size_toList_flatten s)

theorem TensorEncloses.unflatten {s : Shape} {lo hi : Tensor α [s.size]}
    {x : Tensor ℝ [s.size]} (h : TensorEncloses lo hi x) :
    TensorEncloses (Tensor.unflattenSpec s lo) (Tensor.unflattenSpec s hi)
      (Tensor.unflattenSpec s x) :=
  h.reshape (Tensor.size_toList_flatten s).symm

theorem tensorEncloses_vector_iff {n : Nat} {lo hi : Tensor α [n]} {x : Tensor ℝ [n]} :
    TensorEncloses lo hi x ↔
      ∀ i, value (lo.getScalar i) ≤ x.getScalar i ∧
        x.getScalar i ≤ value (hi.getScalar i) := by
  simp only [TensorEncloses, getScalar_eq_apply, Shape.Coord, Prod.forall]
  constructor
  · exact fun h i => h i PUnit.unit
  · intro h i c
    cases c
    exact h i

/-- Flattened endpoint bounds recover enclosure of the shaped real input. -/
theorem tensorEncloses_ibpUnflatten {s : Shape} {B : FlatBox α} {f : Nat → ℝ}
    (hd : B.dim = s.size) (hB : RowEncloses B s.size f) :
    TensorEncloses (ibpUnflatten B.dim B.lo hd) (ibpUnflatten B.dim B.hi hd)
      (realTensor s f) := by
  obtain ⟨d, lo, hi⟩ := B
  dsimp only at hd
  subst d
  simp only [ibpUnflatten, eq_mp_eq_cast, cast_eq, realTensor]
  apply TensorEncloses.unflatten
  rw [tensorEncloses_vector_iff]
  simpa only [getScalar_ofFn] using rowEncloses_iff.mp hB

/-- A shaped enclosure gives the flat graph enclosure when the real coordinates agree. -/
theorem TensorEncloses.row {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) {f : Nat → ℝ}
    (hf : ∀ i : Fin s.size, f i.val = (Tensor.flattenSpec x).getScalar i) :
    RowEncloses { dim := s.size, lo := Tensor.flattenSpec lo, hi := Tensor.flattenSpec hi }
      s.size f := by
  rw [rowEncloses_iff]
  intro i
  rw [hf i]
  exact tensorEncloses_vector_iff.mp h.flatten i

theorem TensorEncloses.tensorValues {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) :
    RowEncloses { dim := s.size, lo := Tensor.flattenSpec lo, hi := Tensor.flattenSpec hi }
      s.size (tensorValues x) :=
  h.row (tensorValues_fin x)

theorem TensorEncloses.getAtOrZero {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) (indices : List Nat) :
    value (getAtOrZero lo indices) ≤ getAtOrZero x indices ∧
      getAtOrZero x indices ≤ value (getAtOrZero hi indices) := by
  induction s generalizing indices with
  | scalar =>
      cases indices with
      | nil => exact h PUnit.unit
      | cons i indices => simp [LawfulBoundOps.toReal_zero (α := α)]
  | dim n s ih =>
      cases indices with
      | nil => simp [LawfulBoundOps.toReal_zero (α := α)]
      | cons i indices =>
          simp only [get_at_or_zero_dim_cons]
          split
          · exact ih (h.unstack _) indices
          · simp [LawfulBoundOps.toReal_zero (α := α)]

theorem TensorEncloses.broadcastPadded {s t : Shape} {lo hi : Tensor α s}
    {x : Tensor ℝ s} (h : TensorEncloses lo hi x) (k : Nat)
    (hb : List.Forall₂ (fun a b => a = b ∨ a = 1) (Shape.padLeft k s).toList t.toList) :
    TensorEncloses (Broadcasting.Internal.broadcastPadded k hb lo)
      (Broadcasting.Internal.broadcastPadded k hb hi)
      (Broadcasting.Internal.broadcastPadded k hb x) := by
  induction t generalizing s k with
  | scalar =>
      cases k with
      | succ k => cases hb
      | zero =>
          cases s with
          | scalar =>
              simpa only [Broadcasting.Internal.broadcastPadded_zero_scalar] using h
          | dim n s => cases hb
  | dim n t ih =>
      cases k with
      | succ k =>
          simp only [Broadcasting.Internal.broadcastPadded_succ]
          exact TensorEncloses.dim fun _ => ih h k _
      | zero =>
          cases s with
          | scalar => cases hb
          | dim m s =>
              rcases (List.forall₂_cons.mp hb).1 with heq | hone
              · subst m
                simp only [Broadcasting.Internal.broadcastPadded_zero_dim_eq]
                exact TensorEncloses.dim fun i => ih (h.unstack i) 0 _
              · subst m
                simp only [Broadcasting.Internal.broadcastPadded_zero_dim_one]
                exact TensorEncloses.dim fun _ => ih (h.unstack 0) 0 _

theorem TensorEncloses.broadcastTo {s t : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) (hb : Shape.CanBroadcastTo s t) :
    TensorEncloses (Tensor.broadcastTo hb lo) (Tensor.broadcastTo hb hi)
      (Tensor.broadcastTo hb x) :=
  h.broadcastPadded _ hb.forall₂_toList

/-- A packed directed sum encloses the corresponding exact tensor sum. -/
theorem TensorEncloses.sum {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) :
    value (Tensor.foldlSpec BoundOps.addDown 0 lo) ≤ Tensor.sumSpec x ∧
      Tensor.sumSpec x ≤ value (Tensor.foldlSpec BoundOps.addUp 0 hi) := by
  have hc (i : Fin (Tensor.Internal.Shape.size s.toList)) :
      value (lo.getFlat i) ≤ x.getFlat i ∧ x.getFlat i ≤ value (hi.getFlat i) := by
    simpa only [Tensor.Internal.Rep.get, Tensor.Internal.Coord.linearize_unlinearize] using
      h (Tensor.Internal.Coord.unlinearize i)
  simpa only [Tensor.sumSpec, Tensor.foldlSpec, Tensor.Internal.Rep.foldl_eq_fin_foldl,
    Fin.foldl_eq_foldl_finRange, List.finRange_foldl_add_eq_finset_sum] using
    sum_encloses lo.getFlat hi.getFlat x.getFlat hc

theorem TensorEncloses.reduceOuterSum {s : Shape} {n : Nat}
    {lo hi : Tensor α (.dim n s)} {x : Tensor ℝ (.dim n s)}
    (h : TensorEncloses lo hi x) :
    TensorEncloses
      (Reduction.Internal.reduceOuterAxis (fun row => Tensor.foldlSpec BoundOps.addDown 0 row) lo)
      (Reduction.Internal.reduceOuterAxis (fun row => Tensor.foldlSpec BoundOps.addUp 0 row) hi)
      (Reduction.Internal.reduceOuterAxis Tensor.sumSpec x) := by
  induction s with
  | scalar =>
      intro c
      cases c
      simpa only [Reduction.Internal.reduceOuterAxis, Tensor.scalar_apply] using h.sum
  | dim m s ih =>
      apply TensorEncloses.dim
      intro j
      apply ih
      apply TensorEncloses.dim
      intro i
      exact (h.unstack i).unstack j

/-- The induction follows both the selected axis and every surviving leading dimension. -/
theorem TensorEncloses.reduceSum {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
    (h : TensorEncloses lo hi x) (axis : Nat) :
    TensorEncloses
      (Tensor.reduceDim (fun row => Tensor.foldlSpec BoundOps.addDown 0 row) axis lo)
      (Tensor.reduceDim (fun row => Tensor.foldlSpec BoundOps.addUp 0 row) axis hi)
      (Tensor.reduceDim Tensor.sumSpec axis x) := by
  induction s generalizing axis with
  | scalar => exact h
  | dim n s ih =>
      cases axis with
      | zero => exact h.reduceOuterSum
      | succ axis =>
          apply TensorEncloses.dim
          intro i
          exact ih (h.unstack i) axis

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
