/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Graph.Forward

/-!
# Sparse VJP Contexts

Builders of shape-indexed contexts (`TensorPack.setIdx`, `TensorPack.set2Idx`) and error lists
(`EList.setIdx`, `EList.set2Idx`) that are zero except at one or two positions. In
`TorchLean.Verify.RuntimeApprox.NF.BackwardOps.Context`, `NF.approxCtx_setIdx` bounds the error of
a write at one position and `NF.approxCtx_set2Idx_ne` that of a write at two distinct positions.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

/-! ## Sparse Contexts For Local VJPs -/

namespace TensorPack

/-- Set a single `Idx` position in a `TorchLean.TensorPack`, filling all other entries with
zeros. -/
def setIdx {α : Type} [Storage α] [Zero α] :
    {Γ : List Shape} → {s : Shape} → Idx Γ s → Tensor α s → TensorPack α Γ
  | [], _, idx, _t => nomatch idx.i
  | s0 :: Γ, s, ⟨⟨0, _⟩, hshape⟩, t =>
      let t0 : Tensor α s0 :=
        Tensor.castShape t (by simpa using hshape.symm)
      .cons t0 (TensorPack.zero (α := α) (ss := Γ))
  | s0 :: Γ, s, ⟨⟨Nat.succ i, hi⟩, hshape⟩, t =>
      .cons (Tensor.full s0 (0 : α))
        (setIdx (α := α) (Γ := Γ) (s := s)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using hshape⟩ t)

/-- Set two `Idx` positions in a `TorchLean.TensorPack`, adding the contributions when the
positions coincide and filling all other entries with zeros. -/
def set2Idx {α : Type} [Storage α] [Zero α] [Add α] :
    {Γ : List Shape} → {s₁ s₂ : Shape} →
      Idx Γ s₁ → Tensor α s₁ → Idx Γ s₂ → Tensor α s₂ → TensorPack α Γ
  | [], _, _, idx, _t₁, _jdx, _t₂ => nomatch idx.i
  | s0 :: Γ, s₁, s₂, ⟨⟨0, _⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂ =>
      let t₁0 : Tensor α s0 :=
        Tensor.castShape t₁ (by simpa using h₁.symm)
      let t₂0 : Tensor α s0 :=
        Tensor.castShape t₂ (by simpa using h₂.symm)
      .cons (t₁0 + t₂0) (TensorPack.zero (α := α) (ss := Γ))
  | s0 :: Γ, s₁, s₂, ⟨⟨0, _⟩, h₁⟩, t₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂ =>
      let t₁0 : Tensor α s0 :=
        Tensor.castShape t₁ (by simpa using h₁.symm)
      .cons t₁0
        (setIdx (α := α) (Γ := Γ) (s := s₂) ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ t₂)
  | s0 :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂ =>
      let t₂0 : Tensor α s0 :=
        Tensor.castShape t₂ (by simpa using h₂.symm)
      .cons t₂0
        (setIdx (α := α) (Γ := Γ) (s := s₁) ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ t₁)
  | s0 :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂ =>
      .cons (Tensor.full s0 (0 : α))
        (set2Idx (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ t₁
          ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ t₂)

end TensorPack

namespace EList

/-- Set a single `Idx` position in an `EList`, filling all other entries with zeros. -/
def setIdx : {Γ : List Shape} → {s : Shape} → Idx Γ s → ℝ → EList Γ
  | [], _, idx, _e => nomatch idx.i
  | _ :: Γ, _s, ⟨⟨0, _⟩, _⟩, e => .cons e (Pack.fill (fun _ => (0 : ℝ)) (ss := Γ))
  | _ :: Γ, s, ⟨⟨Nat.succ i, hi⟩, hshape⟩, e =>
      .cons 0 (setIdx (Γ := Γ) (s := s) ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using hshape⟩ e)

/-- Set two indices in an `EList`, using the supplied combined value `eBoth` when they
coincide and zero elsewhere. -/
def set2Idx : {Γ : List Shape} → {s₁ s₂ : Shape} →
    Idx Γ s₁ → ℝ → Idx Γ s₂ → ℝ → ℝ → EList Γ
  | [], _, _, idx, _e₁, _jdx, _e₂, _eBoth => nomatch idx.i
  | _ :: Γ, _s₁, _s₂, ⟨⟨0, _⟩, _⟩, _e₁, ⟨⟨0, _⟩, _⟩, _e₂, eBoth =>
      .cons eBoth (Pack.fill (fun _ => (0 : ℝ)) (ss := Γ))
  | _ :: Γ, s₁, s₂, ⟨⟨0, _⟩, _⟩, e₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, e₂, eBoth =>
      .cons e₁ (setIdx (Γ := Γ) (s := s₂) ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ e₂)
  | _ :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, e₁, ⟨⟨0, _⟩, _⟩, e₂, _eBoth =>
      .cons e₂ (setIdx (Γ := Γ) (s := s₁) ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ e₁)
  | _ :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, e₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, e₂, eBoth =>
      .cons 0
        (set2Idx (Γ := Γ) (s₁ := s₁) (s₂ := s₂)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ e₁
          ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ e₂ eBoth)

end EList

end

end TorchLean
