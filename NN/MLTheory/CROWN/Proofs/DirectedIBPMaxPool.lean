/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPPermutation

/-!
# Directed enclosure for max pooling

The window fold skips padded cells and uses zero only when the entire fold is empty. Its
comparison selects an endpoint exactly; the proof needs the scalar order law, not rounded
addition or an assumption that pooling windows contain positive values.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN
open Spec.Pooling.Internal

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Three optional window values have matching presence and enclosing scalar values. -/
inductive OptionEncloses : Option α → Option α → Option ℝ → Prop
  | none : OptionEncloses none none none
  | some {lo hi : α} {x : ℝ} (bounds : value lo ≤ x ∧ x ≤ value hi) :
      OptionEncloses (some lo) (some hi) (some x)

theorem OptionEncloses.getD_zero {lo hi : Option α} {x : Option ℝ}
    (h : OptionEncloses lo hi x) :
    value (lo.getD 0) ≤ x.getD 0 ∧ x.getD 0 ≤ value (hi.getD 0) := by
  cases h with
  | none => simp [LawfulBoundOps.toReal_zero (α := α)]
  | some h => exact h

/-- A selected maximum denotes the maximum of the two scalar values. -/
theorem value_selectedMax (a b : α) :
    value (if b > a then b else a) = max (value a) (value b) := by
  by_cases h : b > a
  · rw [ite_eq_left h, max_eq_right (le_of_lt ((LawfulBoundOps.lt_iff a b).mp h))]
  · rw [ite_eq_right h, max_eq_left]
    exact le_of_not_gt (fun hlt => h ((LawfulBoundOps.lt_iff a b).mpr hlt))

theorem selectedMax_real (a b : ℝ) :
    (if b > a then b else a) = max a b := by
  by_cases h : b > a
  · rw [ite_eq_left h, max_eq_right (le_of_lt h)]
  · rw [ite_eq_right h, max_eq_left (le_of_not_gt h)]

/-- The max-pool fold step, with absent padded candidates skipped. -/
def poolMaxStep {β : Type} [Context β] (best candidate : Option β) : Option β :=
  match candidate, best with
  | none, _ => best
  | some x, none => some x
  | some x, some b => if x > b then some x else best

theorem OptionEncloses.poolMaxStep {lo hi clo chi : Option α} {x cx : Option ℝ}
    (h : OptionEncloses lo hi x) (hc : OptionEncloses clo chi cx) :
    OptionEncloses (poolMaxStep lo clo) (poolMaxStep hi chi) (poolMaxStep x cx) := by
  cases hc with
  | none => exact h
  | @some cl cu c hc =>
      cases h with
      | none => exact .some hc
      | @some l u z h =>
          change OptionEncloses
            (if cl > l then Option.some cl else Option.some l)
            (if cu > u then Option.some cu else Option.some u)
            (if c > z then Option.some c else Option.some z)
          rw [← apply_ite Option.some, ← apply_ite Option.some, ← apply_ite Option.some]
          apply OptionEncloses.some
          simp only [value_selectedMax, selectedMax_real]
          exact ⟨max_le_max h.1 hc.1, max_le_max h.2 hc.2⟩

/-- A ternary relation preserved by each list step is preserved by the complete fold. -/
theorem foldl_rel₃ {ι β γ δ : Type} (R : β → γ → δ → Prop)
    (indices : List ι) {fl : β → ι → β} {fu : γ → ι → γ} {fx : δ → ι → δ}
    (hstep : ∀ l u x i, R l u x → R (fl l i) (fu u i) (fx x i))
    {l : β} {u : γ} {x : δ} (h : R l u x) :
    R (indices.foldl fl l) (indices.foldl fu u) (indices.foldl fx x) := by
  induction indices generalizing l u x with
  | nil => exact h
  | cons i indices ih => exact ih (hstep l u x i h)

/-- The spatial iterator preserves a relation independently of rank and empty extents. -/
theorem foldlIndices_rel₃ {β γ δ : Type} (R : β → γ → δ → Prop)
    (dims : List Nat)
    {fl : β → List Nat → β} {fu : γ → List Nat → γ} {fx : δ → List Nat → δ}
    (hstep : ∀ l u x i, R l u x → R (fl l i) (fu u i) (fx x i))
    {l : β} {u : γ} {x : δ} (h : R l u x) :
    R (Spec.Conv.Internal.foldlIndices dims l fl)
      (Spec.Conv.Internal.foldlIndices dims u fu)
      (Spec.Conv.Internal.foldlIndices dims x fx) := by
  induction dims generalizing fl fu fx l u x with
  | nil => exact hstep l u x [] h
  | cons n dims ih =>
      apply foldl_rel₃ R (List.finRange n) _ h
      intro l u x head hacc
      exact ih (fun l u x tail hacc => hstep l u x (head.val :: tail) hacc) hacc

theorem TensorEncloses.paddedAverage {d : Nat} {spatial : Tensor Nat [d]}
    {lo hi : Tensor α (Shape.ofList spatial.data.toList)}
    {x : Tensor ℝ (Shape.ofList spatial.data.toList)}
    (h : TensorEncloses lo hi x) (out window stride padding : List Nat) :
    value (getPaddedAverageInputVal lo out window stride padding) ≤
        getPaddedAverageInputVal x out window stride padding ∧
      getPaddedAverageInputVal x out window stride padding ≤
        value (getPaddedAverageInputVal hi out window stride padding) := by
  unfold getPaddedAverageInputVal
  cases paddedCoords? out window stride with
  | none => simp [LawfulBoundOps.toReal_zero (α := α)]
  | some padded =>
      simp only
      cases hu : unpadCoords? padded padding with
      | none => simp [LawfulBoundOps.toReal_zero (α := α)]
      | some original => exact h.getAtOrZero original

theorem TensorEncloses.paddedMax {d : Nat} {spatial : Tensor Nat [d]}
    {lo hi : Tensor α (Shape.ofList spatial.data.toList)}
    {x : Tensor ℝ (Shape.ofList spatial.data.toList)}
    (h : TensorEncloses lo hi x) (out window stride padding : List Nat) :
    OptionEncloses (getPaddedMaxInputVal? lo out window stride padding)
      (getPaddedMaxInputVal? hi out window stride padding)
      (getPaddedMaxInputVal? x out window stride padding) := by
  unfold getPaddedMaxInputVal?
  cases paddedCoords? out window stride with
  | none => exact .none
  | some padded =>
      simp only
      cases unpadCoords? padded padding with
      | none => exact .none
      | some original =>
          simp only
          split
          · exact .some (h.getAtOrZero original)
          · exact .none

theorem TensorEncloses.maxPoolValue {d : Nat} {spatial : Tensor Nat [d]}
    {lo hi : Tensor α (Shape.ofList spatial.data.toList)}
    {x : Tensor ℝ (Shape.ofList spatial.data.toList)}
    (h : TensorEncloses lo hi x) (out kernel stride padding : List Nat) :
    value (maxPoolValue lo out kernel stride padding) ≤
        maxPoolValue x out kernel stride padding ∧
      maxPoolValue x out kernel stride padding ≤
        value (maxPoolValue hi out kernel stride padding) := by
  apply OptionEncloses.getD_zero
  apply foldlIndices_rel₃ OptionEncloses kernel _ OptionEncloses.none
  intro l u z window hacc
  exact hacc.poolMaxStep (h.paddedMax out window stride padding)

theorem TensorEncloses.maxPoolSpatial {d : Nat}
    {kernel stride padding spatial : Tensor Nat [d]}
    {hk : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hs : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hk hs)
    {lo hi : Tensor α (Shape.ofList spatial.data.toList)}
    {x : Tensor ℝ (Shape.ofList spatial.data.toList)} (h : TensorEncloses lo hi x) :
    TensorEncloses (Spec.maxPoolSpatialSpec layer lo) (Spec.maxPoolSpatialSpec layer hi)
      (Spec.maxPoolSpatialSpec layer x) := by
  intro c
  simpa only [Spec.maxPoolSpatialSpec, Tensor.generate, Tensor.Internal.Rep.get_ofFn] using
    h.maxPoolValue (Shape.Coord.toList _ c) kernel.data.toList stride.data.toList
      padding.data.toList

theorem TensorEncloses.mapLeading (leading : Shape) {s t : Shape}
    {fl fu : Tensor α s → Tensor α t} {fx : Tensor ℝ s → Tensor ℝ t}
    (hstep : ∀ {lo hi x}, TensorEncloses lo hi x → TensorEncloses (fl lo) (fu hi) (fx x))
    {lo hi : Tensor α (leading.concat s)} {x : Tensor ℝ (leading.concat s)}
    (h : TensorEncloses lo hi x) :
    TensorEncloses (Tensor.mapLeading leading fl lo) (Tensor.mapLeading leading fu hi)
      (Tensor.mapLeading leading fx x) := by
  induction leading with
  | scalar => exact hstep h
  | dim n leading ih =>
      exact TensorEncloses.dim fun i => ih (h.unstack i)

/-- The arbitrary-rank real max-pool evaluator is enclosed by the two endpoint evaluations. -/
theorem evalMaxPool_encloses (config : NN.IR.WindowConfig) :
    TensorTransferEncloses (NN.IR.Graph.evalMaxPool (α := α) config)
      (NN.IR.Graph.evalMaxPool (α := ℝ) config) := by
  intro lo hi x h outLo outHi outX hlo hhi hx
  cases h with
  | @mk s lo hi x h =>
      unfold NN.IR.Graph.evalMaxPool at hlo hhi hx
      cases hp : NN.IR.OpContracts.planPool "max_pool" config s with
      | error e => simp only [hp, Bind.bind, Except.bind, reduceCtorEq] at hlo
      | ok plan =>
          simp only [hp, Bind.bind, Except.bind, Pure.pure, Except.pure,
            Except.ok.injEq] at hlo hhi hx
          subst outLo outHi outX
          apply SomeTensorEncloses.mk
          apply TensorEncloses.mapLeading plan.leading
            (fun h => h.maxPoolSpatial {})
          exact h.cast plan.concat_eq.symm

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
