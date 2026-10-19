/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox

/-!
# Sparse VJP Contexts

Shape-indexed context builders used by the NF reverse-mode proofs. These helpers write only the
positions touched by a local VJP rule and leave every other component at zero, which lets the sparse
cases avoid unnecessary rounded additions.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec
open Proofs.Autograd.Algebra

noncomputable section

/-! ## Sparse Contexts For Local VJPs -/

namespace SparseContext

/-- Set a single `Idx` position in a `TorchLean.TensorPack`, filling all other entries with
zeros. -/
def setIdx {α : Type} [TorchLean.Storage α] [Zero α] :
    {Γ : List Shape} → {s : Shape} → Idx Γ s → Tensor α s → TorchLean.TensorPack α Γ
  | [], _, idx, _t => nomatch idx.i
  | s0 :: Γ, s, ⟨⟨0, _⟩, hshape⟩, t =>
      let t0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (source := s) s0 (by simpa using hshape.symm) t
      .cons t0 (TorchLean.TensorPack.zero (α := α) (ss := Γ))
  | s0 :: Γ, s, ⟨⟨Nat.succ i, hi⟩, hshape⟩, t =>
      .cons (Tensor.full s0 (0 : α))
        (setIdx (α := α) (Γ := Γ) (s := s)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using hshape⟩ t)

/-- Set two indices; if they coincide, add the contributions at that position. -/
def set2Idx {α : Type} [TorchLean.Storage α] [Zero α] [Add α] :
    {Γ : List Shape} → {s₁ s₂ : Shape} →
      Idx Γ s₁ → Tensor α s₁ → Idx Γ s₂ → Tensor α s₂ → TorchLean.TensorPack α Γ
  | [], _, _, idx, _t₁, _jdx, _t₂ => nomatch idx.i
  | s0 :: Γ, s₁, s₂, ⟨⟨0, _⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂ =>
      let t₁0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (source := s₁) s0 (by simpa using h₁.symm) t₁
      let t₂0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (source := s₂) s0 (by simpa using h₂.symm) t₂
      .cons (addSpec t₁0 t₂0) (TorchLean.TensorPack.zero (α := α) (ss := Γ))
  | s0 :: Γ, s₁, s₂, ⟨⟨0, _⟩, h₁⟩, t₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂ =>
      let t₁0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (source := s₁) s0 (by simpa using h₁.symm) t₁
      .cons t₁0
        (setIdx (α := α) (Γ := Γ) (s := s₂) ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ t₂)
  | s0 :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂ =>
      let t₂0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (source := s₂) s0 (by simpa using h₂.symm) t₂
      .cons t₂0
        (setIdx (α := α) (Γ := Γ) (s := s₁) ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ t₁)
  | s0 :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂ =>
      .cons (Tensor.full s0 (0 : α))
        (set2Idx (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ t₁
          ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ t₂)

end SparseContext

namespace EList

/-- An `EList` filled with zeros, used for sparse error-bound contexts. -/
def zeros : {ss : List Shape} → EList ss
  | [] => .nil
  | _ :: ss => .cons 0 (zeros (ss := ss))

/-- Set a single `Idx` position in an `EList`, filling all other entries with zeros. -/
def setIdx : {Γ : List Shape} → {s : Shape} → Idx Γ s → ℝ → EList Γ
  | [], _, idx, _e => nomatch idx.i
  | _ :: Γ, _s, ⟨⟨0, _⟩, _⟩, e => .cons e (zeros (ss := Γ))
  | _ :: Γ, s, ⟨⟨Nat.succ i, hi⟩, hshape⟩, e =>
      .cons 0 (setIdx (Γ := Γ) (s := s) ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using hshape⟩ e)

/-- Set two indices in an `EList`; if they coincide, use the supplied combined value `eBoth`. -/
def set2Idx : {Γ : List Shape} → {s₁ s₂ : Shape} →
    Idx Γ s₁ → ℝ → Idx Γ s₂ → ℝ → ℝ → EList Γ
  | [], _, _, idx, _e₁, _jdx, _e₂, _eBoth => nomatch idx.i
  | _ :: Γ, _s₁, _s₂, ⟨⟨0, _⟩, _⟩, _e₁, ⟨⟨0, _⟩, _⟩, _e₂, eBoth =>
      .cons eBoth (zeros (ss := Γ))
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

end RuntimeApprox
end Proofs
