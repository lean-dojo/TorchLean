/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.BackwardApprox
public import NN.Proofs.RuntimeApprox.NF.Linalg
public import NN.Proofs.RuntimeApprox.NF.Ops

/-!
# BackwardOps

NF (rounded) backend: backward/VJP runtime→spec approximation lemmas.

This file instantiates the backend-agnostic reverse framework
`NN.Proofs.RuntimeApprox.Graph.BackwardApprox` for the proof-relevant rounded runtime `NF`.

Scope:
- a context-wise addition bound (`ctxAddBound`) + soundness lemma (`approxCtx_add`);
- `RevNode` constructors for a core set of primitive ops (arithmetic + common activations);
- linear-algebra reverse nodes for `mat_vec_mul_spec` and `mat_mul_spec`.

Notes:
- These are spec-level statements over `ℝ`.
- The NF runtime is a Lean model (rounding after each primitive op); relating it to hardware/`Float`
  remains a trusted interface boundary.

## PyTorch correspondence / citations
This provides per-op VJP rules and error bounds analogous to what PyTorch Autograd computes during
backward passes on a computation graph.
https://pytorch.org/docs/stable/autograd.html

## Map of this file
- Sparse-context helpers (`TList.setIdx`, `set2Idx`, `set3IdxNe`, ...) used to assemble local VJP
  contributions while avoiding extra NF rounding from context-wise `+` in the sparse cases.
- `ctxAddBound` / `approxCtx_add`: a controlled “add contexts” bound used when contributions must
  be accumulated.
- Per-op `RevNode` constructors (arithmetic, activations, reductions, softmax, etc.).
- Linear-algebra reverse nodes (`matVecMulRevNode`, `matMulRevNode`).
- `backprop_approx`: the end-to-end composition theorem, obtained by
  instantiating the backend-agnostic framework.

## References
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018) (VJP
  framing).
- Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library* (NeurIPS
  2019).
- IEEE 754-2019 and Higham, *Accuracy and Stability of Numerical Algorithms* (rounding/error
  composition background).
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

/-! ## Sparse Contexts For Local VJPs -/

namespace TList

/-- A `TList` filled with zeros (shape-wise), used to build sparse contexts for local VJPs. -/
def zeros {α : Type} [Zero α] : {ss : List Shape} → TList α ss
  | [] => .nil
  | s :: ss => .cons (Spec.fill (0 : α) s) (zeros (ss := ss))

/-- Set a single `Idx` position in a `TList`, filling all other entries with zeros. -/
def setIdx {α : Type} [Zero α] : {Γ : List Shape} → {s : Shape} → Idx Γ s → Tensor α s → TList α Γ
  | [], _, idx, _t => nomatch idx.i
  | s0 :: Γ, s, ⟨⟨0, _⟩, hshape⟩, t =>
      let t0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s) (t := s0) (by simpa using hshape.symm) t
      .cons t0 (zeros (α := α) (ss := Γ))
  | s0 :: Γ, s, ⟨⟨Nat.succ i, hi⟩, hshape⟩, t =>
      .cons (Spec.fill (0 : α) s0)
        (setIdx (α := α) (Γ := Γ) (s := s)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using hshape⟩ t)

/-- Set two indices; if they coincide, add the contributions at that position. -/
def set2Idx {α : Type} [Zero α] [Add α] :
    {Γ : List Shape} → {s₁ s₂ : Shape} →
      Idx Γ s₁ → Tensor α s₁ → Idx Γ s₂ → Tensor α s₂ → TList α Γ
  | [], _, _, idx, _t₁, _jdx, _t₂ => nomatch idx.i
  | s0 :: Γ, s₁, s₂, ⟨⟨0, _⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂ =>
      let t₁0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s₁) (t := s0) (by simpa using h₁.symm) t₁
      let t₂0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s₂) (t := s0) (by simpa using h₂.symm) t₂
      .cons (addSpec t₁0 t₂0) (zeros (α := α) (ss := Γ))
  | s0 :: Γ, s₁, s₂, ⟨⟨0, _⟩, h₁⟩, t₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂ =>
      let t₁0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s₁) (t := s0) (by simpa using h₁.symm) t₁
      .cons t₁0
        (setIdx (α := α) (Γ := Γ) (s := s₂) ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ t₂)
  | s0 :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂ =>
      let t₂0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s₂) (t := s0) (by simpa using h₂.symm) t₂
      .cons t₂0
        (setIdx (α := α) (Γ := Γ) (s := s₁) ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ t₁)
  | s0 :: Γ, s₁, s₂, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂ =>
      .cons (Spec.fill (0 : α) s0)
        (set2Idx (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ t₁
          ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ t₂)

end TList

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
  | _ :: Γ, _s₁, _s₂, ⟨⟨0, _⟩, _⟩, e₁, ⟨⟨0, _⟩, _⟩, _e₂, eBoth =>
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

namespace TList

/-- Set three indices when the positions are pairwise distinct.

This avoids any context-wise addition: only the three targeted positions are written,
and all others are `0`. This is important for NF, where even `x + 0` would incur rounding. -/
def set3IdxNe {α : Type} [Zero α] [Add α] :
    {Γ : List Shape} → {s₁ s₂ s₃ : Shape} →
      (a : Idx Γ s₁) → Tensor α s₁ →
      (b : Idx Γ s₂) → Tensor α s₂ →
      (c : Idx Γ s₃) → Tensor α s₃ →
      a.i ≠ b.i → a.i ≠ c.i → b.i ≠ c.i →
      TList α Γ
  | [], _, _, _, a, _t₁, _b, _t₂, _c, _t₃, _hab, _hac, _hbc => nomatch a.i
  | s0 :: Γ, s₁, s₂, s₃, ⟨⟨0, _⟩, h₁⟩, t₁, ⟨⟨0, _⟩, _h₂⟩, _t₂, _c, _t₃, hab, _hac, _hbc =>
      False.elim (hab rfl)
  | s0 :: Γ, s₁, s₂, s₃, ⟨⟨0, _⟩, h₁⟩, t₁, _b, _t₂, ⟨⟨0, _⟩, _h₃⟩, _t₃, _hab, hac, _hbc =>
      False.elim (hac rfl)
  | s0 :: Γ, s₁, s₂, s₃, ⟨⟨0, _⟩, h₁⟩, t₁,
      ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂,
      ⟨⟨Nat.succ k, hk⟩, h₃⟩, t₃,
      _hab, _hac, _hbc =>
      let t₁0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s₁) (t := s0) (by simpa using h₁.symm) t₁
      let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩
      let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ hk⟩, by simpa using h₃⟩
      .cons t₁0 (TList.set2Idx (α := α) (Γ := Γ) (s₁ := s₂) (s₂ := s₃) bTail t₂ cTail t₃)
  | s0 :: Γ, s₁, s₂, s₃, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂, ⟨⟨0, _⟩, _h₃⟩, _t₃,
      _hab, _hac, hbc =>
      False.elim (hbc rfl)
  | s0 :: Γ, s₁, s₂, s₃, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨0, _⟩, h₂⟩, t₂,
      ⟨⟨Nat.succ k, hk⟩, h₃⟩, t₃,
      _hab, _hac, _hbc =>
      let t₂0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s₂) (t := s0) (by simpa using h₂.symm) t₂
      let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩
      let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ hk⟩, by simpa using h₃⟩
      .cons t₂0 (TList.set2Idx (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₃) aTail t₁ cTail t₃)
  | s0 :: Γ, s₁, s₂, s₃, ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂, ⟨⟨0, _⟩, h₃⟩, t₃,
      hab, _hac, _hbc =>
      let t₃0 : Tensor α s0 :=
        Spec.tensorCast (α := α) (s := s₃) (t := s0) (by simpa using h₃.symm) t₃
      let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩
      let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩
      have habTail : aTail.i ≠ bTail.i := by
        intro h
        apply hab
        apply Fin.ext
        simpa using congrArg Fin.val h
      .cons t₃0 (TList.set2Idx (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) aTail t₁ bTail t₂)
  | s0 :: Γ, s₁, s₂, s₃,
      ⟨⟨Nat.succ i, hi⟩, h₁⟩, t₁,
      ⟨⟨Nat.succ j, hj⟩, h₂⟩, t₂,
      ⟨⟨Nat.succ k, hk⟩, h₃⟩, t₃,
      hab, hac, hbc =>
      .cons (Spec.fill (0 : α) s0)
        (set3IdxNe (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ t₁
          ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ t₂
          ⟨⟨k, Nat.lt_of_succ_lt_succ hk⟩, by simpa using h₃⟩ t₃
          (by
            intro h
            apply hab
            apply Fin.ext
            simpa using congrArg Fin.val h)
          (by
            intro h
            apply hac
            apply Fin.ext
            simpa using congrArg Fin.val h)
          (by
            intro h
            apply hbc
            apply Fin.ext
            simpa using congrArg Fin.val h))

end TList

namespace EList

/-- Error list for `TList.set3Idx_ne`: set the three designated positions, `0` elsewhere. -/
def set3IdxNe :
    {Γ : List Shape} → {s₁ s₂ s₃ : Shape} →
      (a : Idx Γ s₁) → ℝ → (b : Idx Γ s₂) → ℝ → (c : Idx Γ s₃) → ℝ →
      a.i ≠ b.i → a.i ≠ c.i → b.i ≠ c.i →
      EList Γ
  | [], _, _, _, a, _e₁, _b, _e₂, _c, _e₃, _hab, _hac, _hbc => nomatch a.i
  | _ :: Γ, _s₁, _s₂, _s₃, ⟨⟨0, _⟩, _⟩, _e₁, ⟨⟨0, _⟩, _⟩, _e₂, _c, _e₃, hab, _hac, _hbc =>
      False.elim (hab rfl)
  | _ :: Γ, _s₁, _s₂, _s₃, ⟨⟨0, _⟩, _⟩, _e₁, _b, _e₂, ⟨⟨0, _⟩, _⟩, _e₃, _hab, hac, _hbc =>
      False.elim (hac rfl)
  | _ :: Γ, s₁, s₂, s₃, ⟨⟨0, _⟩, _h₁⟩, e₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, e₂, ⟨⟨Nat.succ k, hk⟩, h₃⟩, e₃,
      _hab, _hac, _hbc =>
      let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩
      let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ hk⟩, by simpa using h₃⟩
      .cons e₁ (EList.set2Idx (Γ := Γ) (s₁ := s₂) (s₂ := s₃) bTail e₂ cTail e₃ 0)
  | _ :: Γ, s₁, s₂, s₃, ⟨⟨Nat.succ i, hi⟩, h₁⟩, e₁, ⟨⟨0, _⟩, _h₂⟩, e₂, ⟨⟨0, _⟩, _h₃⟩, _e₃,
      _hab, _hac, hbc =>
      False.elim (hbc rfl)
  | _ :: Γ, s₁, s₂, s₃, ⟨⟨Nat.succ i, hi⟩, h₁⟩, e₁, ⟨⟨0, _⟩, h₂⟩, e₂, ⟨⟨Nat.succ k, hk⟩, h₃⟩, e₃,
      _hab, _hac, _hbc =>
      let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩
      let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ hk⟩, by simpa using h₃⟩
      .cons e₂ (EList.set2Idx (Γ := Γ) (s₁ := s₁) (s₂ := s₃) aTail e₁ cTail e₃ 0)
  | _ :: Γ, s₁, s₂, s₃, ⟨⟨Nat.succ i, hi⟩, h₁⟩, e₁, ⟨⟨Nat.succ j, hj⟩, h₂⟩, e₂, ⟨⟨0, _⟩, _h₃⟩, e₃,
      hab, _hac, _hbc =>
      let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩
      let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩
      have habTail : aTail.i ≠ bTail.i := by
        intro h
        apply hab
        apply Fin.ext
        simpa using congrArg Fin.val h
      .cons e₃ (EList.set2Idx (Γ := Γ) (s₁ := s₁) (s₂ := s₂) aTail e₁ bTail e₂ 0)
  | _ :: Γ, s₁, s₂, s₃,
      ⟨⟨Nat.succ i, hi⟩, h₁⟩, e₁,
      ⟨⟨Nat.succ j, hj⟩, h₂⟩, e₂,
      ⟨⟨Nat.succ k, hk⟩, h₃⟩, e₃,
      hab, hac, hbc =>
      .cons 0
        (set3IdxNe (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃)
          ⟨⟨i, Nat.lt_of_succ_lt_succ hi⟩, by simpa using h₁⟩ e₁
          ⟨⟨j, Nat.lt_of_succ_lt_succ hj⟩, by simpa using h₂⟩ e₂
          ⟨⟨k, Nat.lt_of_succ_lt_succ hk⟩, by simpa using h₃⟩ e₃
          (by
            intro h
            apply hab
            apply Fin.ext
            simpa using congrArg Fin.val h)
          (by
            intro h
            apply hac
            apply Fin.ext
            simpa using congrArg Fin.val h)
          (by
            intro h
            apply hbc
            apply Fin.ext
            simpa using congrArg Fin.val h))

end EList

/-! ## NF Backend Instantiation -/

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- `toSpec` is already defined in `NN.Proofs.RuntimeApprox.NF.Ops`.

private lemma toSpec_one_bound :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) - (1 : ℝ)) ≤
      neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2 := by
  -- `1 : R` is `NF.ofReal 1`, so this is the standard single-step rounding error bound.
  simpa [NFBackend.toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.ofReal,
    TorchLean.Floats.NF.roundR,
    Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ))

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
lemma approxT_fill_const {cS : ℝ} {cR : R} {eps : ℝ} (h : abs (toSpec (β := β) (fexp := fexp) (rnd
  := rnd) cR - cS) ≤ eps) :
    ∀ {s : Shape}, approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.fill cS s) (Spec.fill cR s) eps := by
  intro s
  induction s with
  | scalar =>
      simpa [Spec.fill] using (approxT_scalar_iff (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (x := cS) (xR := cR) (eps := eps)
          |>.2 h)
  | dim n s ih =>
      cases n with
      | zero =>
          -- vacuous: `Fin 0` is empty, and the `foldl max` is `0`.
          have heps : 0 ≤ eps := le_trans (abs_nonneg _) h
          simpa [Spec.fill, approxT, approxWith, tensorToSpec, linfNorm,
            RuntimeApprox.linfNorm,
            tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
            tensorLinfNorm, Spec.mapTensor] using heps
      | succ n =>
          -- Each component satisfies the IH; take the `foldl max` upper bound.
          have heps : 0 ≤ eps := le_trans (abs_nonneg _) h
          -- Unfold `approxT` for the outer `.dim`.
          -- Reduce to a `foldl max` bound over component distances.
          have hcomp :
              ∀ i : Fin (Nat.succ n),
                tensorDistance (α := SpecScalar) linfNorm
                    (Spec.fill cS s)
                    (tensorToSpec (α := R)
                      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Spec.fill cR s))
                  ≤ eps := by
            intro i
            -- This is exactly the IH at the inner shape (independent of `i`).
            simpa [approxT, approxWith] using ih
          have hfold :=
            List.foldl_max_le_of_le (List.finRange (Nat.succ n))
              (fun i =>
                tensorDistance (α := SpecScalar) linfNorm
                    (Spec.fill cS s)
                    (tensorToSpec (α := R)
                      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Spec.fill cR s)))
              (acc := (0 : ℝ)) (eps := eps) heps (by
                intro i hi
                simpa using hcomp i)
          -- Finish by rewriting back to the dim tensor form.
          simpa [Spec.fill, approxT, approxWith, tensorToSpec, linfNorm,
            RuntimeApprox.linfNorm,
            tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
            tensorLinfNorm, Spec.mapTensor] using hfold

lemma approxT_fill_one :
    ∀ {s : Shape}, approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s) (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward /
        2) := by
  intro s
  exact
    approxT_fill_const (β := β) (fexp := fexp) (rnd := rnd)
      (cS := (1 : ℝ)) (cR := (1 : R))
      (eps := neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
      (toSpec_one_bound (β := β) (fexp := fexp) (rnd := rnd)) (s := s)

lemma approxT_fill_zero :
    ∀ {s : Shape}, approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.fill (0 : ℝ) s) (Spec.fill (0 : R) s) 0 := by
  intro s
  refine approxT_fill_const (β := β) (fexp := fexp) (rnd := rnd) (cS := (0 : ℝ)) (cR := (0 : R))
    (eps := 0) ?_ (s := s)
  simp

lemma idx_shape_eq_of_i_eq {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    (h : a.i = b.i) : s₁ = s₂ := by
  have : Γ.get a.i = Γ.get b.i := by simp [h]
  calc
    s₁ = Γ.get a.i := by simpa using a.h.symm
    _ = Γ.get b.i := this
    _ = s₂ := by simpa using b.h

/--
Cast a tensor across a shape equality induced by equal `Idx` positions.

Given `a : Idx Γ s₁`, `b : Idx Γ s₂`, and `h : a.i = b.i`, this produces a function
`Tensor α s₂ → Tensor α s₁` that casts along the implied equality `s₁ = s₂`.
-/
def tensorCastOfIdxEq {α : Type} {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    (h : a.i = b.i) : Tensor α s₂ → Tensor α s₁ :=
  Spec.tensorCast (α := α) (s := s₂) (t := s₁) (idx_shape_eq_of_i_eq (Γ := Γ) (a := a) (b := b)
    h).symm

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
lemma approxT_tensor_cast {s t : Shape} (h : s = t)
    {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.tensorCast (α := SpecScalar) (s := s) (t := t) h xS)
      (Spec.tensorCast (α := R) (s := s) (t := t) h xR)
      eps := by
  cases h
  simpa [Spec.tensorCast] using hx

lemma approxCtx_zeros {Γ : List Shape} :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.zeros (α := SpecScalar) (ss := Γ))
      (TList.zeros (α := R) (ss := Γ))
      (EList.zeros (ss := Γ)) := by
  induction Γ with
  | nil =>
      simp [TList.zeros, EList.zeros, approxCtx]
  | cons s Γ ih =>
      refine And.intro ?_ ih
      simpa [TList.zeros, EList.zeros] using (approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := s))

lemma approxCtx_setIdx {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    {tS : SpecTensor s} {tR : Tensor R s} {eps : ℝ}
    (ht : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) tS tR eps) :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) idx tS)
      (TList.setIdx (α := R) (Γ := Γ) (s := s) idx tR)
      (EList.setIdx (Γ := Γ) (s := s) idx eps) := by
  classical
  cases idx with
  | mk i hshape =>
      induction Γ with
      | nil =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
      | cons s0 Γ ih =>
          cases i with
          | mk iVal hiVal =>
              cases iVal with
              | zero =>
                  -- head is the distinguished index
                  cases hshape
                  refine And.intro ?_ ?_
                  · simpa [TList.setIdx, EList.setIdx] using ht
                  · simpa [TList.setIdx, EList.setIdx] using
                      (approxCtx_zeros (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ))
              | succ j =>
                  have hshape' : Γ.get ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩ = s := by
                    simpa using hshape
                  have iht := ih (i := ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩) (hshape := hshape')
                  refine And.intro ?_ ?_
                  · simpa [TList.setIdx, EList.setIdx] using
                      (approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s0))
                  · simpa [TList.setIdx, EList.setIdx] using iht

lemma approxCtx_set2Idx_ne {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    {t₁S : SpecTensor s₁} {t₁R : Tensor R s₁} {eps₁ : ℝ}
    {t₂S : SpecTensor s₂} {t₂R : Tensor R s₂} {eps₂ : ℝ}
    (h₁ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₁S t₁R eps₁)
    (h₂ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₂S t₂R eps₂)
    (hne : a.i ≠ b.i) :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a t₁S b t₂S)
      (TList.set2Idx (α := R) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a t₁R b t₂R)
      (EList.set2Idx (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a eps₁ b eps₂ 0) := by
  classical
  induction Γ with
  | nil =>
      cases a with
      | mk i _ =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γ ih =>
      cases a with
      | mk ia haShape =>
          cases b with
          | mk ib hbShape =>
              cases ia with
              | mk iaVal iaLt =>
                  cases ib with
                  | mk ibVal ibLt =>
                      cases iaVal with
                      | zero =>
                          cases ibVal with
                          | zero =>
                              exact False.elim (hne (by rfl))
                          | succ j =>
                              cases haShape
                              refine And.intro ?_ ?_
                              · simpa [TList.set2Idx, EList.set2Idx] using h₁
                              ·
                                let bTail : Idx Γ s₂ :=
                                  ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩, by simpa using hbShape⟩
                                have := approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
                                  (Γ := Γ) (s := s₂) bTail (tS := t₂S) (tR := t₂R) (eps := eps₂) h₂
                                simpa [TList.set2Idx, EList.set2Idx, bTail] using this
                      | succ i =>
                          cases ibVal with
                          | zero =>
                              cases hbShape
                              refine And.intro ?_ ?_
                              · simpa [TList.set2Idx, EList.set2Idx] using h₂
                              ·
                                let aTail : Idx Γ s₁ :=
                                  ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩, by simpa using haShape⟩
                                have := approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
                                  (Γ := Γ) (s := s₁) aTail (tS := t₁S) (tR := t₁R) (eps := eps₁) h₁
                                simpa [TList.set2Idx, EList.set2Idx, aTail] using this
                          | succ j =>
                              let aTail : Idx Γ s₁ :=
                                ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩, by simpa using haShape⟩
                              let bTail : Idx Γ s₂ :=
                                ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩, by simpa using hbShape⟩
                              have hneTail : aTail.i ≠ bTail.i := by
                                intro hij
                                apply hne
                                apply Fin.ext
                                have : i = j := by
                                  simpa [aTail, bTail] using congrArg Fin.val hij
                                simp [this]
                              refine And.intro ?_ ?_
                              · simpa [TList.set2Idx, EList.set2Idx] using
                                  (approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s0))
                              ·
                                have := ih (a := aTail) (b := bTail) hneTail
                                simpa [TList.set2Idx, EList.set2Idx, aTail, bTail] using this

lemma approxCtx_set3Idx_ne {Γ : List Shape} {s₁ s₂ s₃ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂) (c :
  Idx Γ s₃)
    {t₁S : SpecTensor s₁} {t₁R : Tensor R s₁} {eps₁ : ℝ}
    {t₂S : SpecTensor s₂} {t₂R : Tensor R s₂} {eps₂ : ℝ}
    {t₃S : SpecTensor s₃} {t₃R : Tensor R s₃} {eps₃ : ℝ}
    (h₁ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₁S t₁R eps₁)
    (h₂ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₂S t₂R eps₂)
    (h₃ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₃S t₃R eps₃)
    (hab : a.i ≠ b.i) (hac : a.i ≠ c.i) (hbc : b.i ≠ c.i) :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.set3IdxNe (α := SpecScalar) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) a t₁S b t₂S c
        t₃S hab hac hbc)
      (TList.set3IdxNe (α := R) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) a t₁R b t₂R c t₃R hab hac
        hbc)
      (EList.set3IdxNe (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) a eps₁ b eps₂ c eps₃ hab hac hbc)
        := by
  classical
  induction Γ with
  | nil =>
      cases a with
      | mk i _ =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γ ih =>
      cases a with
      | mk ia haShape =>
          cases b with
          | mk ib hbShape =>
              cases c with
              | mk ic hcShape =>
                  cases ia with
                  | mk iaVal iaLt =>
                      cases ib with
                      | mk ibVal ibLt =>
                          cases ic with
                          | mk icVal icLt =>
                              cases iaVal with
                              | zero =>
                                  cases ibVal with
                                  | zero =>
                                      exact False.elim (hab rfl)
                                  | succ j =>
                                      cases icVal with
                                      | zero =>
                                          exact False.elim (hac rfl)
                                      | succ k =>
                                          cases haShape
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using h₁
                                          ·
                                            let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ
                                              ibLt⟩, by simpa using hbShape⟩
                                            let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ
                                              icLt⟩, by simpa using hcShape⟩
                                            have hbcTail : bTail.i ≠ cTail.i := by
                                              intro hij
                                              apply hbc
                                              apply Fin.ext
                                              have : j = k := by
                                                simpa [bTail, cTail] using congrArg Fin.val hij
                                              simp [this]
                                            have :=
                                              approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd :=
                                                rnd)
                                                (Γ := Γ) (s₁ := s₂) (s₂ := s₃) bTail cTail
                                                (t₁S := t₂S) (t₁R := t₂R) (eps₁ := eps₂)
                                                (t₂S := t₃S) (t₂R := t₃R) (eps₂ := eps₃)
                                                h₂ h₃ hbcTail
                                            simpa [TList.set3IdxNe, EList.set3IdxNe, bTail, cTail]
                                              using this
                              | succ i =>
                                  cases ibVal with
                                  | zero =>
                                      cases icVal with
                                      | zero =>
                                          exact False.elim (hbc rfl)
                                      | succ k =>
                                          cases hbShape
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using h₂
                                          ·
                                            let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ
                                              iaLt⟩, by simpa using haShape⟩
                                            let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ
                                              icLt⟩, by simpa using hcShape⟩
                                            have hacTail : aTail.i ≠ cTail.i := by
                                              intro hij
                                              apply hac
                                              apply Fin.ext
                                              have : i = k := by
                                                simpa [aTail, cTail] using congrArg Fin.val hij
                                              simp [this]
                                            have :=
                                              approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd :=
                                                rnd)
                                                (Γ := Γ) (s₁ := s₁) (s₂ := s₃) aTail cTail
                                                (t₁S := t₁S) (t₁R := t₁R) (eps₁ := eps₁)
                                                (t₂S := t₃S) (t₂R := t₃R) (eps₂ := eps₃)
                                                h₁ h₃ hacTail
                                            simpa [TList.set3IdxNe, EList.set3IdxNe, aTail, cTail]
                                              using this
                                  | succ j =>
                                      cases icVal with
                                      | zero =>
                                          cases hcShape
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using h₃
                                          ·
                                            let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ
                                              iaLt⟩, by simpa using haShape⟩
                                            let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ
                                              ibLt⟩, by simpa using hbShape⟩
                                            have habTail : aTail.i ≠ bTail.i := by
                                              intro hij
                                              apply hab
                                              apply Fin.ext
                                              have : i = j := by
                                                simpa [aTail, bTail] using congrArg Fin.val hij
                                              simp [this]
                                            have :=
                                              approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd :=
                                                rnd)
                                                (Γ := Γ) (s₁ := s₁) (s₂ := s₂) aTail bTail
                                                (t₁S := t₁S) (t₁R := t₁R) (eps₁ := eps₁)
                                                (t₂S := t₂S) (t₂R := t₂R) (eps₂ := eps₂)
                                                h₁ h₂ habTail
                                            simpa [TList.set3IdxNe, EList.set3IdxNe, aTail, bTail]
                                              using this
                                      | succ k =>
                                          let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩,
                                            by simpa using haShape⟩
                                          let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩,
                                            by simpa using hbShape⟩
                                          let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ icLt⟩,
                                            by simpa using hcShape⟩
                                          have habTail : aTail.i ≠ bTail.i := by
                                            intro hij
                                            apply hab
                                            apply Fin.ext
                                            have : i = j := by
                                              simpa [aTail, bTail] using congrArg Fin.val hij
                                            simp [this]
                                          have hacTail : aTail.i ≠ cTail.i := by
                                            intro hij
                                            apply hac
                                            apply Fin.ext
                                            have : i = k := by
                                              simpa [aTail, cTail] using congrArg Fin.val hij
                                            simp [this]
                                          have hbcTail : bTail.i ≠ cTail.i := by
                                            intro hij
                                            apply hbc
                                            apply Fin.ext
                                            have : j = k := by
                                              simpa [bTail, cTail] using congrArg Fin.val hij
                                            simp [this]
                                          have iht := ih (a := aTail) (b := bTail) (c := cTail)
                                            habTail hacTail hbcTail
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using
                                              (approxT_fill_zero (β := β) (fexp := fexp) (rnd :=
                                                rnd) (s := s0))
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe, aTail, bTail,
                                            cTail] using iht

-- ---------------------------------------------------------------------------
-- Context-wise addition bound (used by global backprop accumulation)
-- ---------------------------------------------------------------------------

/--
Context-wise addition bound (NF runtime vs spec).

This produces an `EList` of `linf_norm` bounds for adding two contexts elementwise, and is used when
reverse-mode accumulation must combine contributions from multiple consumers.
-/
def ctxAddBound : {Δ : List Shape} → EList Δ → EList Δ → TList R Δ → TList R Δ → EList Δ
  | [], .nil, .nil, .nil, .nil => .nil
  | _ :: ss, .cons ex exs, .cons ey eys, .cons x xs, .cons y ys =>
      .cons (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := _) ex ey x y))
        (ctxAddBound (Δ := ss) exs eys xs ys)

/--
Soundness of context-wise addition under `approxCtx`.

If `xS ~ xR ± epsx` and `yS ~ yR ± epsy`, then `(xS + yS) ~ (xR + yR)` with error bounded by
`ctxAddBound epsx epsy xR yR`.
-/
theorem approxCtx_add {Δ : List Shape} :
    ∀ (xS yS : TList SpecScalar Δ) (xR yR : TList R Δ) (epsx epsy : EList Δ),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (TList.add (α := SpecScalar) xS yS)
          (TList.add (α := R) xR yR)
          (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd) epsx epsy xR yR) := by
  intro xS yS xR yR epsx epsy hx hy
  induction Δ with
  | nil =>
      cases xS
      cases yS
      cases xR
      cases yR
      cases epsx
      cases epsy
      simp [TList.add, ctxAddBound, approxCtx]
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases yS with
          | cons ySh ySt =>
              cases xR with
              | cons xRh xRt =>
                  cases yR with
                  | cons yRh yRt =>
                      cases epsx with
                      | cons ex exs =>
                          cases epsy with
                          | cons ey eys =>
                              refine And.intro ?_ ?_
                              · -- head uses `approxT_add_spec`
                                have hx0 : approxT (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) xSh xRh ex :=
                                  hx.1
                                have hy0 : approxT (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) ySh yRh ey :=
                                  hy.1
                                simpa [TList.add, ctxAddBound] using
                                  (approxT_add_spec (β := β) (fexp := fexp) (rnd := rnd)
                                    (s := s) (xS := xSh) (yS := ySh) (xR := xRh) (yR := yRh)
                                    (epsx := ex) (epsy := ey) hx0 hy0)
                              · -- tail by IH
                                have hxT : approxCtx (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) xSt xRt exs :=
                                  hx.2
                                have hyT : approxCtx (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) ySt yRt eys :=
                                  hy.2
                                simpa [TList.add, ctxAddBound] using
                                  ih (xS := xSt) (yS := ySt) (xR := xRt) (yR := yRt) (epsx := exs)
                                    (epsy := eys) hxT hyT

-- ---------------------------------------------------------------------------
-- Reverse nodes (RevNode constructors)
-- ---------------------------------------------------------------------------

/--
Reverse node for addition: `z = a + b`.

VJP is `(δ ↦ (δ, δ))`, with a special-case when `a` and `b` are the same context index.
-/
def addRevNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := addNode (β := β) (fexp := fexp) (rnd := rnd) a b
      vjpSpec := fun _ctx δ =>
        if h : a.i = b.i then
          TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (addSpec δ δ)
        else
          TList.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b δ
      vjpRuntime := fun _ctx δ =>
        if h : a.i = b.i then
          TList.setIdx (α := R) (Γ := Γ) (s := s) a (addSpec δ δ)
        else
          TList.set2Idx (α := R) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b δ
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        if h : a.i = b.i then
          let epsBoth :=
            linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ epsδ δR
              δR)
          EList.setIdx (Γ := Γ) (s := s) a epsBoth
        else
          EList.set2Idx (Γ := Γ) (s₁ := s) (s₂ := s) a epsδ b epsδ 0
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  classical
  by_cases hEq : a.i = b.i
  · have hsum :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec δS δS) (addSpec δR δR)
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ epsδ δR
            δR)) := by
      simpa using
        (approxT_add_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := δS) (xR := δR) (yR := δR)
          (epsx := epsδ) (epsy := epsδ) hδ hδ)
    have hctx' :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := s) a
        (tS := addSpec δS δS) (tR := addSpec δR δR)
        (eps := linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ epsδ
          δR δR)) hsum
    simpa [hEq] using hctx'
  · have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := s) (s₂ := s) a b
        (t₁S := δS) (t₁R := δR) (eps₁ := epsδ)
        (t₂S := δS) (t₂R := δR) (eps₂ := epsδ)
        hδ hδ hEq
    simpa [hEq] using hctx'

/--
Reverse node for subtraction: `z = a - b`.

VJP is `(δ ↦ (δ, -δ))`, with a special-case when `a` and `b` are the same context index.
-/
def subRevNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := subNode (β := β) (fexp := fexp) (rnd := rnd) a b
      vjpSpec := fun _ctx δ =>
        if h : a.i = b.i then
          TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (Spec.fill (0 : SpecScalar) s)
        else
          TList.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b (negSpec δ)
      vjpRuntime := fun _ctx δ =>
        if h : a.i = b.i then
          TList.setIdx (α := R) (Γ := Γ) (s := s) a (Spec.fill (0 : R) s)
        else
          TList.set2Idx (α := R) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b (negSpec δ)
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        if h : a.i = b.i then
          EList.setIdx (Γ := Γ) (s := s) a 0
        else
          let epsNeg :=
            linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR)
          EList.set2Idx (Γ := Γ) (s₁ := s) (s₂ := s) a epsδ b epsNeg 0
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  classical
  by_cases hEq : a.i = b.i
  · have h0 :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := s) a
        (tS := Spec.fill (0 : SpecScalar) s) (tR := Spec.fill (0 : R) s)
        (eps := 0) (approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s))
    simpa [hEq] using h0
  · have hneg :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (negSpec δS) (negSpec δR)
          (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR)) := by
      simpa using
        (approxT_neg_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (xR := δR) (eps := epsδ) hδ)
    have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := s) (s₂ := s) a b
        (t₁S := δS) (t₁R := δR) (eps₁ := epsδ)
        (t₂S := negSpec δS) (t₂R := negSpec δR)
        (eps₂ := linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR))
        hδ hneg hEq
    simpa [hEq] using hctx'

/--
Reverse node for multiplication: `z = a * b`.

VJP is `(δ ↦ (δ*b, δ*a))`, with rounding-aware bounds produced by the NF backend.
-/
def mulRevNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := mulNode (β := β) (fexp := fexp) (rnd := rnd) a b
      vjpSpec := fun ctx δ =>
        if h : a.i = b.i then
          let x := getIdx (α := SpecScalar) ctx a
          let u := mulSpec δ x
          TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (addSpec u u)
        else
          let xa := getIdx (α := SpecScalar) ctx a
          let xb := getIdx (α := SpecScalar) ctx b
          TList.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s) (s₂ := s) a (mulSpec δ xb) b (mulSpec
            δ xa)
      vjpRuntime := fun ctx δ =>
        if h : a.i = b.i then
          let x := getIdx (α := R) ctx a
          let u := mulSpec δ x
          TList.setIdx (α := R) (Γ := Γ) (s := s) a (addSpec u u)
        else
          let xa := getIdx (α := R) ctx a
          let xb := getIdx (α := R) ctx b
          TList.set2Idx (α := R) (Γ := Γ) (s₁ := s) (s₂ := s) a (mulSpec δ xb) b (mulSpec δ xa)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        if h : a.i = b.i then
          let xR := getIdx (α := R) ctxR a
          let uR := mulSpec δR xR
          let epsU :=
            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR xR)
          let epsBoth :=
            linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsU epsU uR
              uR)
          EList.setIdx (Γ := Γ) (s := s) a epsBoth
        else
          let xaR := getIdx (α := R) ctxR a
          let xbR := getIdx (α := R) ctxR b
          let epsA :=
            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx b) δR xbR)
          let epsB :=
            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR xaR)
          EList.set2Idx (Γ := Γ) (s₁ := s) (s₂ := s) a epsA b epsB 0
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  classical
  have ha :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hb :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx b
  by_cases hEq : a.i = b.i
  · -- x*x case: contributions add
    have hu :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (mulSpec δR (getIdx (α := R) ctxR a))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a))) := by
      simpa using
        (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := getIdx (α := SpecScalar) ctxS a)
          (xR := δR) (yR := getIdx (α := R) ctxR a)
          (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx a) hδ ha)
    have hsum :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec (mulSpec δS (getIdx (α := SpecScalar) ctxS a)) (mulSpec δS (getIdx (α :=
            SpecScalar) ctxS a)))
          (addSpec (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s)
            (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
            (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
            (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR a)))) := by
      simpa using
        (approxT_add_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (xS := mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (yS := mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (xR := mulSpec δR (getIdx (α := R) ctxR a))
          (yR := mulSpec δR (getIdx (α := R) ctxR a))
          (epsx := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          (epsy := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          hu hu)
    have hctx' :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := s) a
        (tS := addSpec (mulSpec δS (getIdx (α := SpecScalar) ctxS a)) (mulSpec δS (getIdx (α :=
          SpecScalar) ctxS a)))
        (tR := addSpec (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR
          a)))
        (eps := linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR a)))) hsum
    simpa [hEq] using hctx'
  · have hA :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec δS (getIdx (α := SpecScalar) ctxS b))
          (mulSpec δR (getIdx (α := R) ctxR b))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx b) δR (getIdx (α := R) ctxR b))) := by
      simpa using
        (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := getIdx (α := SpecScalar) ctxS b)
          (xR := δR) (yR := getIdx (α := R) ctxR b)
          (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx b) hδ hb)
    have hB :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (mulSpec δR (getIdx (α := R) ctxR a))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a))) := by
      simpa using
        (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := getIdx (α := SpecScalar) ctxS a)
          (xR := δR) (yR := getIdx (α := R) ctxR a)
          (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx a) hδ ha)
    have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := s) (s₂ := s) a b
        (t₁S := mulSpec δS (getIdx (α := SpecScalar) ctxS b))
        (t₁R := mulSpec δR (getIdx (α := R) ctxR b))
        (eps₁ := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx b) δR (getIdx (α := R) ctxR b)))
        (t₂S := mulSpec δS (getIdx (α := SpecScalar) ctxS a))
        (t₂R := mulSpec δR (getIdx (α := R) ctxR a))
        (eps₂ := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
        hA hB hEq
    simpa [hEq] using hctx'

/--
Reverse node for scaling by a constant: `z = c * a`.
-/
def scaleRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (c : R) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := scaleNode (β := β) (fexp := fexp) (rnd := rnd) a c
      vjpSpec := fun _ctx δ =>
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a
          (scaleSpec (α := SpecScalar) (s := s) δ (toSpec (β := β) (fexp := fexp) (rnd := rnd) c))
      vjpRuntime := fun _ctx δ =>
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (scaleSpec (α := R) (s := s) δ c)
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        EList.setIdx (Γ := Γ) (s := s) a
          (linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ c δR))
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  have hscale :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (scaleSpec (α := SpecScalar) (s := s) δS (toSpec (β := β) (fexp := fexp) (rnd := rnd) c))
        (scaleSpec (α := R) (s := s) δR c)
        (linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ c δR)) :=
          by
    simpa using
      (approxT_scale_spec (β := β) (fexp := fexp) (rnd := rnd) (c := c)
        (s := s) (xS := δS) (xR := δR) (eps := epsδ) hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := scaleSpec (α := SpecScalar) (s := s) δS (toSpec (β := β) (fexp := fexp) (rnd := rnd)
        c))
      (tR := scaleSpec (α := R) (s := s) δR c)
      (eps := linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ c
        δR)) hscale
  simpa using hctx'

/--
Reverse node for negation: `z = -a`.
-/
def negRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := negNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun _ctx δ =>
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (negSpec δ)
      vjpRuntime := fun _ctx δ =>
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (negSpec δ)
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        EList.setIdx (Γ := Γ) (s := s) a (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd
          := rnd) (s := s) epsδ δR))
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  have hneg :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (negSpec δS) (negSpec δR)
        (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR)) := by
    simpa using
      (approxT_neg_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s) (xS := δS) (xR := δR) (eps := epsδ) hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a (tS := negSpec δS) (tR := negSpec δR)
      (eps := linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR))
        hneg
  simpa using hctx'

/--
Reverse node for `exp`.
-/
def expRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := expNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let ex := mapSpec (s := s) MathFunctions.exp x
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec ex δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let ex := mapSpec (s := s) MathFunctions.exp x
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec ex δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let exR := mapSpec (s := s) MathFunctions.exp xR
        let epsEx :=
          linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsEx epsδ exR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hex :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a))
        (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
        (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxT_exp_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hout :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a)) δS)
        (mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a)) δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsδ
          (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
          δR)) := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a))
        (yS := δS)
        (xR := mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
        (yR := δR)
        (epsx := linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := epsδ)
        hex hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a)) δS)
      (tR := mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a)) δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        epsδ
        (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
        δR)) hout
  simpa using hctx'

/--
Reverse node for `tanh`.
-/
def tanhRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := tanhNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let t := mapSpec (s := s) MathFunctions.tanh x
        let df := subSpec (Spec.fill (1 : ℝ) s) (mulSpec t t)
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let t := mapSpec (s := s) MathFunctions.tanh x
        let df := subSpec (Spec.fill (1 : R) s) (mulSpec t t)
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let tR := mapSpec (s := s) MathFunctions.tanh xR
        let epsT :=
          linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsSq :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsT epsT tR tR)
        let onesR : Tensor R s := Spec.fill (1 : R) s
        let epsDf :=
          linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2) epsSq onesR (mulSpec tR
              tR))
        let dfR := subSpec onesR (mulSpec tR tR)
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have ht :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
        (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
        (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxT_tanh_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hsq :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a)))
        (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))) := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
        (yS := mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
        (xR := mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
        (yR := mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
        (epsx := linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        ht ht)
  have hones :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s)
        (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2) :=
    approxT_fill_one (β := β) (fexp := fexp) (rnd := rnd) (s := s)
  have hdf :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (subSpec (Spec.fill (1 : ℝ) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
        (subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))) := by
    simpa using
      (approxT_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := Spec.fill (1 : ℝ) s)
        (yS := mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a)))
        (xR := Spec.fill (1 : R) s)
        (yR := mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))
        (epsx := neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
        (epsy := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        hones hsq)
  have hout :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (subSpec (Spec.fill (1 : ℝ) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
          δS)
        (mulSpec
          (subSpec (Spec.fill (1 : R) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
            (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
            (Spec.fill (1 : R) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))))
          epsδ
          (subSpec (Spec.fill (1 : R) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          δR)) := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := subSpec (Spec.fill (1 : ℝ) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
        (yS := δS)
        (xR := subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        (yR := δR)
        (epsx := linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))))
        (epsy := epsδ)
        hdf hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (subSpec (Spec.fill (1 : ℝ) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
        δS)
      (tR := mulSpec
        (subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))))
        epsδ
        (subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        δR)) hout
  simpa using hctx'

/--
Reverse node for `sigmoid`.
-/
def sigmoidRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := sigmoidNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let sS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) x
        let df := mulSpec sS (subSpec (Spec.fill (1 : ℝ) s) sS)
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let sR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) x
        let df := mulSpec sR (subSpec (Spec.fill (1 : R) s) sR)
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let sR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let epsS :=
          linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOne : ℝ := neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2
        let epsOneMinus :=
          linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsOne epsS (Spec.fill (1 : R) s) sR)
        let oneMinusR := subSpec (Spec.fill (1 : R) s) sR
        let epsDf :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsS epsOneMinus sR oneMinusR)
        let dfR := mulSpec sR oneMinusR
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hs :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxT_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hones :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s)
        (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2) :=
    approxT_fill_one (β := β) (fexp := fexp) (rnd := rnd) (s := s)
  have honeMinus :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))) :=
            by
    simpa using
      (approxT_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := Spec.fill (1 : ℝ) s)
        (yS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (xR := Spec.fill (1 : R) s)
        (yR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (epsx := neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
        (epsy := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        hones hs)
  have hdf :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))))
              := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (xR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))
        (epsx := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        hs honeMinus)
  have hout :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (subSpec (Spec.fill (1 : ℝ) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))))
          δS)
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
              (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR
                a)))))
          epsδ
          (mulSpec
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)) := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (yS := δS)
        (xR := mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        (yR := δR)
        (epsx := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))))
        (epsy := epsδ)
        hdf hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        δS)
      (tR := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))))
        epsδ
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)) hout
  simpa using hctx'

/--
Reverse node for `softplus`.
-/
def softplusRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := softplusNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let sig := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) x
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec sig δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let sig := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) x
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec sig δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let sigR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let epsSig :=
          linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsSig epsδ sigR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hsig :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxT_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hout :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α :=
          SpecScalar) ctxS a)) δS)
        (mulSpec (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR
          a)) δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsδ
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          δR)) := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := δS)
        (xR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := δR)
        (epsx := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := epsδ)
        hsig hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        δS)
      (tR := mulSpec
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        epsδ
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        δR)) hout
  simpa using hctx'

/--
Reverse node for a log with an explicit stabilization parameter `ε` (to avoid `log 0`-style issues).
-/
def safeLogRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  let epsR : R := TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) ε
  let epsErr : ℝ := neuralUlp β fexp ε TrainingPhase.forward / 2
  refine
    { toFwdNode := safeLogSoftplusNode (β := β) (fexp := fexp) (rnd := rnd) a ε hε
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let num := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) x
        let sp := mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) x
        let denom := addSpec sp (Spec.fill ε s)
        let df := map2Spec (s := s) (safeDiv (ε := ε)) num denom
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let xR := getIdx (α := R) ctx a
        let numR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let spR := mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) xR
        let denomR := addSpec spR (Spec.fill epsR s)
        let dfR := map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) numR denomR
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec dfR δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let numR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let spR := mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) xR
        let epsNum :=
          linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsSp :=
          linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let denomR := addSpec spR (Spec.fill epsR s)
        let epsDen :=
          linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsSp epsErr spR (Spec.fill epsR s))
        let epsDf :=
          linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) ε epsNum epsDen numR denomR)
        let dfR := map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) numR denomR
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a

  have hnum :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxT_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)

  have hsp :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
          a))
        (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxT_softplus_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)

  have heps_val :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) epsR - ε) ≤ epsErr := by
    simpa [epsR, epsErr, toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.ofReal,
      TorchLean.Floats.NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) ε)
  have heps :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill ε s) (Spec.fill epsR s) epsErr := by
    simpa [epsErr] using
      (approxT_fill_const (β := β) (fexp := fexp) (rnd := rnd)
        (cS := ε) (cR := epsR) (eps := epsErr) heps_val (s := s))

  have hden :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (addSpec
          (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (Spec.fill ε s))
        (addSpec
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s))
        (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsErr
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s))) := by
    simpa using
      (approxT_add_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := Spec.fill ε s)
        (xR := mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
          ctxR a))
        (yR := Spec.fill epsR s)
        (epsx := linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := epsErr)
        hsp heps)

  have hdf :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (map2Spec (s := s) (safeDiv (ε := ε))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (addSpec
            (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (Spec.fill ε s)))
        (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
        (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            epsErr
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))) := by
    simpa using
      (approxT_safeDiv_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
        (xS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := addSpec
          (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (Spec.fill ε s))
        (xR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := addSpec
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s))
        (epsx := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsErr
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s)))
        hnum hden)

  have hout :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (map2Spec (s := s) (safeDiv (ε := ε))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (addSpec
              (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))
              (Spec.fill ε s)))
          δS)
        (mulSpec
          (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) ε
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              epsErr
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s))))
          epsδ
          (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
          δR)) := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS :=
          map2Spec (s := s) (safeDiv (ε := ε))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (addSpec
              (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))
              (Spec.fill ε s)))
        (yS := δS)
        (xR :=
          map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
        (yR := δR)
        (epsx := linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            epsErr
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s))))
        (epsy := epsδ)
        hdf hδ)

  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (map2Spec (s := s) (safeDiv (ε := ε))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (addSpec
            (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (Spec.fill ε s)))
        δS)
      (tR := mulSpec
        (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            epsErr
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s))))
        epsδ
        (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
        δR)) hout
  simpa using hctx'

/--
Reverse node for the scalar logistic-compatible `softmax` node, using the analytic ℝ derivative plus NF
error bounds.
-/
def softmaxRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := softmaxNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let sS := mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) x
        let df := mulSpec sS (subSpec (Spec.fill (1 : ℝ) s) sS)
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let sR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) x
        let df := mulSpec sR (subSpec (Spec.fill (1 : R) s) sR)
        TList.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let sR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) xR
        let epsS :=
          linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOne : ℝ := neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2
        let epsOneMinus :=
          linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsOne epsS (Spec.fill (1 : R) s) sR)
        let oneMinusR := subSpec (Spec.fill (1 : R) s) sR
        let epsDf :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsS epsOneMinus sR oneMinusR)
        let dfR := mulSpec sR oneMinusR
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hs :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxT_softmax_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hones :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s)
        (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2) :=
    approxT_fill_one (β := β) (fexp := fexp) (rnd := rnd) (s := s)
  have honeMinus :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))) :=
            by
    simpa using
      (approxT_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := Spec.fill (1 : ℝ) s)
        (yS := mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (xR := Spec.fill (1 : R) s)
        (yR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
        (epsx := neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
        (epsy := linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        hones hs)
  have hdf :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))))
              := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (xR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))
        (epsx := linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        hs honeMinus)
  have hout :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (subSpec (Spec.fill (1 : ℝ) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))))
          δS)
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
              (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR
                a)))))
          epsδ
          (mulSpec
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)) := by
    simpa using
      (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (yS := δS)
        (xR := mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        (yR := δR)
        (epsx := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))))
        (epsy := epsδ)
        hdf hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        δS)
      (tR := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))))
        epsδ
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)) hout
  simpa using hctx'

/--
Reverse node for ReLU, using the standard piecewise derivative/VJP.
-/
def reluRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := reluNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let gated := map2Spec (fun d x => if x > 0 then d else 0) δ x
        TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a gated
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let gated := map2Spec (fun d x => if x > 0 then d else 0) δ x
        TList.setIdx (α := R) (Γ := Γ) (s := s) a gated
      vjpBound := fun _epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let bndT : SpecTensor s :=
          map2Spec (fun a _b => abs a + epsδ)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δR)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
        EList.setIdx (Γ := Γ) (s := s) a (linfNorm bndT)
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hgate :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (map2Spec (fun d x => if x > 0 then d else 0) δS (getIdx (α := SpecScalar) ctxS a))
        (map2Spec (fun d x => if x > 0 then d else 0) δR (getIdx (α := R) ctxR a))
        (linfNorm
          (map2Spec (fun a _b => abs a + epsδ)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δR)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (getIdx
              (α := R) ctxR a)))) := by
    -- Use the generic `map2` lifting lemma with a conservative scalar bound.
    simpa using
      (approxT_map2_spec_of_scalar_bound (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (s := s)
        (fS := fun d x => if x > 0 then d else 0)
        (fR := fun d x => if x > 0 then d else 0)
        (bnd := fun a _b epsδ _epsx => abs a + epsδ)
        (xS := δS) (yS := getIdx (α := SpecScalar) ctxS a)
        (xR := δR) (yR := getIdx (α := R) ctxR a)
        (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx a)
        hδ hx (by
          intro d x dR xR hd hx'
          by_cases hxS : x > 0 <;> by_cases hxR : xR > 0
          · -- both on
            simpa [hxS, hxR] using le_trans hd (le_add_of_nonneg_left (abs_nonneg _))
          · -- spec on, runtime off
            have hδ' : abs d ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
              have hdiff : abs (d - toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) ≤ epsδ := by
                simpa [abs_sub_comm] using hd
              calc
                abs d = abs ((d - toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + toSpec (β := β)
                  (fexp := fexp) (rnd := rnd) dR) := by ring_nf
                _ ≤ abs (d - toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + abs (toSpec (β := β)
                  (fexp := fexp) (rnd := rnd) dR) := abs_add_le _ _
                _ ≤ epsδ + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) := add_le_add hdiff
                  (le_rfl)
                _ = abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by ring_nf
            -- output diff is |0 - d| = |d|
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                = abs d := by
                  simp [hxS, hxR]
            -- bound by |toSpec dR| + epsδ
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
                  simpa [this] using hδ'
            simpa [hxS, hxR] using this
          · -- spec off, runtime on
            -- output diff is |toSpec dR - 0| = |toSpec dR|
            have heps : 0 ≤ epsδ := le_trans (abs_nonneg _) hd
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                = abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) := by
                  simp [hxS, hxR]
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
                  simpa [this] using
                    (le_add_of_nonneg_right (a := abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                      dR)) heps)
            simpa [hxS, hxR] using this
          · -- both off
            have heps : 0 ≤ epsδ := le_trans (abs_nonneg _) hd
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                = 0 := by simp [hxS, hxR]
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
                  simpa [this] using add_nonneg (abs_nonneg _) heps
            simpa [hxS, hxR] using this))
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := map2Spec (fun d x => if x > 0 then d else 0) δS (getIdx (α := SpecScalar) ctxS a))
      (tR := map2Spec (fun d x => if x > 0 then d else 0) δR (getIdx (α := R) ctxR a))
      (eps := linfNorm
        (map2Spec (fun a _b => abs a + epsδ)
          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δR)
          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (getIdx
            (α := R) ctxR a)))) hgate
  simpa using hctx'

/--
Reverse node for reduction `sum`, sending the upstream gradient back along the broadcasted shape.
-/
def sumRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ Shape.scalar :=
by
  classical
  refine
    { toFwdNode := sumNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun _ctx δ =>
        match δ with
        | Tensor.scalar d =>
            TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (Spec.fill d s)
      vjpRuntime := fun _ctx δ =>
        match δ with
        | Tensor.scalar d =>
            TList.setIdx (α := R) (Γ := Γ) (s := s) a (Spec.fill d s)
      vjpBound := fun _epsCtx _ctxR epsδ _δR =>
        EList.setIdx (Γ := Γ) (s := s) a epsδ
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  cases δS with
  | scalar dS =>
      cases δR with
      | scalar dR =>
          have hd :
              abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR - dS) ≤ epsδ := by
            simpa using
              (approxT_scalar_iff (α := R)
                (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (x := dS) (xR := dR) (eps := epsδ)).1 hδ
          have hfill :
              approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (Spec.fill dS s) (Spec.fill dR s) epsδ :=
            approxT_fill_const (β := β) (fexp := fexp) (rnd := rnd) (cS := dS) (cR := dR) (eps :=
              epsδ) hd
              (s := s)
          have hctx' :=
            approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
              (Γ := Γ) (s := s) a (tS := Spec.fill dS s) (tR := Spec.fill dR s) (eps := epsδ) hfill
          simpa using hctx'

-- ---------------------------------------------------------------------------
-- Linear algebra reverse nodes (`mat_vec_mul_spec`, `mat_mul_spec`)
-- ---------------------------------------------------------------------------

/--
Reverse node for matrix-vector multiplication (`mat_vec_mul_spec`).

VJP uses the standard adjoint identities: `δW = δ ⊗ x` and `δx = Wᵀ δ` (expressed in tensor form),
with NF error bounds layered over the primitive ops.
-/
def matVecMulRevNode {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (v : Idx Γ (.dim n .scalar)) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim m .scalar) :=
by
  classical
  refine
    { toFwdNode := matVecMulNode (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ) (m := m) (n := n) A v
      vjpSpec := fun ctx δ =>
        let vS := getIdx (α := SpecScalar) ctx v
        let δcol := Spec.Tensor.expandToColSpec (α := SpecScalar) (n := m) (s := Shape.scalar) δ
        let vcol := Spec.Tensor.expandToColSpec (α := SpecScalar) (n := n) (s := Shape.scalar) vS
        let dA :=
          Spec.matMulSpec (α := SpecScalar)
            δcol (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := 1) vcol)
        let dV :=
          Spec.matVecMulSpec (α := SpecScalar)
            (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
              SpecScalar) ctx A))
            δ
        TList.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n
          .scalar)) A dA v dV
      vjpRuntime := fun ctx δ =>
        let vR := getIdx (α := R) ctx v
        let δcol := Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δ
        let vcol := Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) vR
        let dA :=
          Spec.matMulSpec (α := R)
            δcol (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1) vcol)
        let dV :=
          Spec.matVecMulSpec (α := R)
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctx A))
            δ
        TList.set2Idx (α := R) (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n .scalar)) A
          dA v dV
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let vR := getIdx (α := R) ctxR v
        let δcolR := Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δR
        let vcolR := Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) vR
        let vrowR := Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1) vcolR
        let epsV := getIdxEps (Γ := Γ) (s := (.dim n .scalar)) epsCtx v
        let epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A
        let dABound :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := 1) (p := n)
              epsδ epsV δcolR vrowR)
        let AT_R := Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R)
          ctxR A)
        let dVBound :=
          linfNorm
            (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m)
              epsA epsδ AT_R δR)
        EList.set2Idx (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n .scalar)) A dABound
          v dVBound 0
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  -- Approximate `v` and `A` from the context.
  have hv := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    v
  have hA := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    A

  -- `dA = mat_mul (expand δ) (transpose (expand v))`.
  have hδcol :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
        (Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δR)
        epsδ :=
    approxT_expand_to_col_spec (β := β) (fexp := fexp) (rnd := rnd) (n := m) (s := Shape.scalar) (xS
      := δS)
      (xR := δR) (eps := epsδ) hδ

  have hvcol :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx (α :=
          SpecScalar) ctxS v))
        (Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R) ctxR
          v))
        (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) :=
    approxT_expand_to_col_spec (β := β) (fexp := fexp) (rnd := rnd) (n := n) (s := Shape.scalar)
      (xS := getIdx (α := SpecScalar) ctxS v) (xR := getIdx (α := R) ctxR v)
      (eps := getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) hv

  have hvrow :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := 1)
          (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx (α
            := SpecScalar) ctxS v)))
        (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1)
          (Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
            ctxR v)))
        (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) :=
    approxT_matrix_transpose_spec (β := β) (fexp := fexp) (rnd := rnd) (m := n) (n := 1) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) hvcol

  have hdA :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matMulSpec (α := SpecScalar)
          (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
          (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := 1)
            (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx
              (α := SpecScalar) ctxS v))))
        (Spec.matMulSpec (α := R)
          (Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δR)
          (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1)
            (Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
              ctxR v))))
        (linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := 1) (p := n)
            epsδ (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v)
            (Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δR)
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1)
              (Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
                ctxR v))))) := by
    simpa using
      (approxT_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := 1) (p := n)
        (AS := Spec.Tensor.expandToColSpec (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
        (BS := Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := 1)
          (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx (α
            := SpecScalar) ctxS v)))
        (AR := Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δR)
        (BR := Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1)
          (Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
            ctxR v)))
        (epsA := epsδ) (epsB := getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) hδcol hvrow)

  -- `dV = mat_vec_mul (transpose A) δ`.
  have hAT :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
          SpecScalar) ctxS A))
        (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
        (getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) :=
    approxT_matrix_transpose_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) hA

  have hdV :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matVecMulSpec (α := SpecScalar)
          (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
            SpecScalar) ctxS A)) δS)
        (Spec.matVecMulSpec (α := R)
          (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
            δR)
        (linfNorm
          (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := n) (n := m)
            (getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) epsδ
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
              δR)) := by
    simpa using
      (approxT_mat_vec_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := n) (n := m)
        (AS := Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
          SpecScalar) ctxS A))
        (vS := δS)
        (AR := Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
          A))
        (vR := δR)
        (epsA := getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) (epsV := epsδ)
        hAT hδ)

  have hne : A.i ≠ v.i := by
    intro hEq
    -- Shapes would have to coincide.
    have hshapeEq :
        Shape.dim m (Shape.dim n Shape.scalar) = Shape.dim n Shape.scalar := by
      have : Γ.get A.i = Γ.get v.i := by simp [hEq]
      calc
        Shape.dim m (Shape.dim n Shape.scalar) = Γ.get A.i := by simpa using A.h.symm
        _ = Γ.get v.i := this
        _ = Shape.dim n Shape.scalar := by simpa using v.h
    -- Contradiction by constructor discrimination.
    cases hshapeEq

  have hctx' :=
    approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n .scalar)) A v
      (t₁S :=
        Spec.matMulSpec (α := SpecScalar)
          (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
          (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := 1)
            (Spec.Tensor.expandToColSpec (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx
              (α := SpecScalar) ctxS v))))
      (t₁R :=
        Spec.matMulSpec (α := R)
          (Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δR)
          (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1)
            (Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
              ctxR v))))
      (eps₁ :=
        let vR := getIdx (α := R) ctxR v
        let δcolR := Spec.Tensor.expandToColSpec (α := R) (n := m) (s := Shape.scalar) δR
        let vcolR := Spec.Tensor.expandToColSpec (α := R) (n := n) (s := Shape.scalar) vR
        let vrowR := Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := 1) vcolR
        linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := 1) (p := n)
            epsδ (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v)
            δcolR vrowR))
      (t₂S :=
        Spec.matVecMulSpec (α := SpecScalar)
          (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
            SpecScalar) ctxS A))
          δS)
      (t₂R :=
        Spec.matVecMulSpec (α := R)
          (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
          δR)
      (eps₂ :=
        linfNorm
          (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := n) (n := m)
            (getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) epsδ
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
              δR))
      hdA hdV hne
  simpa using hctx'

/--
Reverse node for matrix multiplication (`mat_mul_spec`).

VJP uses the standard identities `δA = δC * Bᵀ` and `δB = Aᵀ * δC` (in appropriate shapes),
with NF error bounds layered over the primitive ops.
-/
def matMulRevNode {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim m (.dim p
      .scalar)) :=
by
  classical
  refine
    { toFwdNode := matMulNode (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ) (m := m) (n := n) (p :=
      p) A B
      vjpSpec := fun ctx δ =>
        if h : A.i = B.i then
          -- both contributions land in the same slot
          let A0 := getIdx (α := SpecScalar) ctx A
          let δA := Spec.matMulSpec (α := SpecScalar) δ (Spec.Tensor.matrixTransposeSpec (α :=
            SpecScalar) (m := n) (n := p) (getIdx (α := SpecScalar) ctx B))
          let δB := Spec.matMulSpec (α := SpecScalar) (Spec.Tensor.matrixTransposeSpec (α :=
            SpecScalar) (m := m) (n := n) A0) δ
          let δB' := tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) h δB
          TList.setIdx (α := SpecScalar) (Γ := Γ) (s := (.dim m (.dim n .scalar))) A (addSpec δA
            δB')
        else
          let δA := Spec.matMulSpec (α := SpecScalar) δ (Spec.Tensor.matrixTransposeSpec (α :=
            SpecScalar) (m := n) (n := p) (getIdx (α := SpecScalar) ctx B))
          let δB := Spec.matMulSpec (α := SpecScalar) (Spec.Tensor.matrixTransposeSpec (α :=
            SpecScalar) (m := m) (n := n) (getIdx (α := SpecScalar) ctx A)) δ
          TList.set2Idx (α := SpecScalar) (Γ := Γ)
            (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar))) A δA B δB
      vjpRuntime := fun ctx δ =>
        if h : A.i = B.i then
          let A0 := getIdx (α := R) ctx A
          let δA := Spec.matMulSpec (α := R) δ (Spec.Tensor.matrixTransposeSpec (α := R) (m :=
            n) (n := p) (getIdx (α := R) ctx B))
          let δB := Spec.matMulSpec (α := R) (Spec.Tensor.matrixTransposeSpec (α := R) (m := m)
            (n := n) A0) δ
          let δB' := tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) h δB
          TList.setIdx (α := R) (Γ := Γ) (s := (.dim m (.dim n .scalar))) A (addSpec δA δB')
        else
          let δA := Spec.matMulSpec (α := R) δ (Spec.Tensor.matrixTransposeSpec (α := R) (m :=
            n) (n := p) (getIdx (α := R) ctx B))
          let δB := Spec.matMulSpec (α := R) (Spec.Tensor.matrixTransposeSpec (α := R) (m := m)
            (n := n) (getIdx (α := R) ctx A)) δ
          TList.set2Idx (α := R) (Γ := Γ)
            (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar))) A δA B δB
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A
        let epsB := getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B
        let BT_R := Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R)
          ctxR B)
        let AT_R := Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R)
          ctxR A)
        let δA_R := Spec.matMulSpec (α := R) δR BT_R
        let δB_R := Spec.matMulSpec (α := R) AT_R δR
        let epsδA :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := p) (p := n)
              epsδ epsB δR BT_R)
        let epsδB :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              epsA epsδ AT_R δR)
        if h : A.i = B.i then
          let δB_R' := tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) h δB_R
          let epsSum :=
            linfNorm
              (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := (.dim m (.dim n .scalar))) epsδA epsδB δA_R δB_R')
          EList.setIdx (Γ := Γ) (s := (.dim m (.dim n .scalar))) A epsSum
        else
          EList.set2Idx (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar)))
            A epsδA B epsδB 0
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  classical
  have hA := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    A
  have hB := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    B

  have hBT :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := p) (getIdx (α :=
          SpecScalar) ctxS B))
        (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR B))
        (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) :=
    approxT_matrix_transpose_spec (β := β) (fexp := fexp) (rnd := rnd) (m := n) (n := p) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) hB

  have hAT :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
          SpecScalar) ctxS A))
        (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
        (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) :=
    approxT_matrix_transpose_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) hA

  have hdA :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matMulSpec (α := SpecScalar) δS
          (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := p) (getIdx (α :=
            SpecScalar) ctxS B)))
        (Spec.matMulSpec (α := R) δR
          (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR B)))
        (linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := p) (p := n)
            epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
              B)))) := by
    simpa using
      (approxT_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := p) (p := n)
        (AS := δS)
        (BS := Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := p) (getIdx (α :=
          SpecScalar) ctxS B))
        (AR := δR)
        (BR := Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
          B))
        (epsA := epsδ) (epsB := getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) hδ
          hBT)

  have hdB :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matMulSpec (α := SpecScalar)
          (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
            SpecScalar) ctxS A)) δS)
        (Spec.matMulSpec (α := R)
          (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
            δR)
        (linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := n) (n := m) (p := p)
            (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
              δR)) := by
    simpa using
      (approxT_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := n) (n := m) (p := p)
        (AS := Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
          SpecScalar) ctxS A))
        (BS := δS)
        (AR := Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
          A))
        (BR := δR)
        (epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) (epsB := epsδ) hAT
          hδ)

  by_cases hEq : A.i = B.i
  · -- contributions add in one slot
    have hdB' :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
            (Spec.matMulSpec (α := SpecScalar)
              (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
                SpecScalar) ctxS A)) δS))
          (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
            (Spec.matMulSpec (α := R)
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                A)) δR))
          (linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                A)) δR)) := by
      simpa [tensorCastOfIdxEq, idx_shape_eq_of_i_eq] using
        (approxT_tensor_cast (β := β) (fexp := fexp) (rnd := rnd)
          (h := (idx_shape_eq_of_i_eq (Γ := Γ) (a := A) (b := B) hEq).symm)
          (xS :=
            Spec.matMulSpec (α := SpecScalar)
              (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
                SpecScalar) ctxS A)) δS)
          (xR :=
            Spec.matMulSpec (α := R)
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                A)) δR)
          (eps :=
            linfNorm
              (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (m := n) (n := m) (p := p)
                (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
                (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                  A)) δR))
          hdB)

    have hsum :
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec
            (Spec.matMulSpec (α := SpecScalar) δS
              (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := p) (getIdx (α :=
                SpecScalar) ctxS B)))
            (tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := SpecScalar)
                (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
                  SpecScalar) ctxS A)) δS)))
          (addSpec
            (Spec.matMulSpec (α := R) δR
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
                B)))
            (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := R)
                (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                  A)) δR)))
          (linfNorm
            (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := (.dim m (.dim n .scalar)))
              (linfNorm
                (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (m := m) (n := p) (p := n)
                  epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
                  (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R)
                    ctxR B))))
              (linfNorm
                (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (m := n) (n := m) (p := p)
                  (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
                  (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R)
                    ctxR A)) δR))
              (Spec.matMulSpec (α := R) δR
                (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
                  B)))
              (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
                (Spec.matMulSpec (α := R)
                  (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R)
                    ctxR A)) δR)))) := by
      simpa using
        (approxT_add_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := (.dim m (.dim n .scalar)))
          (xS :=
            Spec.matMulSpec (α := SpecScalar) δS
              (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := p) (getIdx (α :=
                SpecScalar) ctxS B)))
          (yS :=
            tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := SpecScalar)
                (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
                  SpecScalar) ctxS A)) δS))
          (xR :=
            Spec.matMulSpec (α := R) δR
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
                B)))
          (yR :=
            tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := R)
                (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                  A)) δR))
          (epsx :=
            linfNorm
              (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (m := m) (n := p) (p := n)
                epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
                (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
                  B))))
          (epsy :=
            linfNorm
              (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (m := n) (n := m) (p := p)
                (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
                (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                  A)) δR))
          hdA hdB')

    let epsSum : ℝ :=
      linfNorm
        (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := (.dim m (.dim n .scalar)))
          (linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := p) (p := n)
              epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
                B))))
          (linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                A)) δR))
          (Spec.matMulSpec (α := R) δR
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR B)))
          (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
            (Spec.matMulSpec (α := R)
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                A)) δR)))
    have hctx' :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := (.dim m (.dim n .scalar))) A
        (tS :=
          addSpec
            (Spec.matMulSpec (α := SpecScalar) δS
              (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := p) (getIdx (α :=
                SpecScalar) ctxS B)))
            (tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := SpecScalar)
                (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
                  SpecScalar) ctxS A)) δS)))
        (tR :=
          addSpec
            (Spec.matMulSpec (α := R) δR
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
                B)))
            (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := R)
                (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                  A)) δR)))
        (eps := epsSum) (by simpa [epsSum] using hsum)
    simpa [hEq, epsSum, tensorCastOfIdxEq, idx_shape_eq_of_i_eq] using hctx'
  · -- disjoint indices: use `set2Idx`
    have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar))) A B
        (t₁S :=
          Spec.matMulSpec (α := SpecScalar) δS
            (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := n) (n := p) (getIdx (α :=
              SpecScalar) ctxS B)))
        (t₁R :=
          Spec.matMulSpec (α := R) δR
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR B)))
        (eps₁ :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := p) (p := n)
              epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := n) (n := p) (getIdx (α := R) ctxR
                B))))
        (t₂S :=
          Spec.matMulSpec (α := SpecScalar)
            (Spec.Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α :=
              SpecScalar) ctxS A)) δS)
        (t₂R :=
          Spec.matMulSpec (α := R)
            (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR A))
              δR)
        (eps₂ :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
              (Spec.Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctxR
                A)) δR))
        hdA hdB hEq
    simpa [hEq] using hctx'

-- ---------------------------------------------------------------------------
-- Global reverse-mode bound (specialized to NF accumulation)
-- ---------------------------------------------------------------------------

/--
End-to-end NF reverse-mode soundness for a well-typed reverse graph.

This is the main composition theorem: if each node in the graph has a sound `RevNode` instance,
then the whole backpropagated context is an `approxCtx` enclosure of the spec backpropagation.
-/
theorem backprop_approx {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ ss) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList R Γ) (epsIn : EList Γ)
      (seedS : TList SpecScalar (Γ ++ ss)) (seedR : TList R (Γ ++ ss)) (epsSeed : EList (Γ ++ ss)),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsIn →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) seedS seedR epsSeed
        →
        approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (RevGraph.backpropSpec g xS seedS)
          (RevGraph.backpropRuntime g xR seedR)
          (RevGraph.backpropBounds g epsIn xR epsSeed seedR (ctxAddBound (β := β) (fexp := fexp)
            (rnd := rnd))) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  simpa using
    (RevGraph.backprop_approx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) g
      (addBound := ctxAddBound (β := β) (fexp := fexp) (rnd := rnd))
      (addSound := fun {Δ} => approxCtx_add (β := β) (fexp := fexp) (rnd := rnd) (Δ := Δ))
      xS xR epsIn seedS seedR epsSeed hx hseed)

end NFBackend

end

end RuntimeApprox
end Proofs
