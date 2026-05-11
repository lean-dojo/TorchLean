/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.NNReal.Defs
public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox

/-!
# ScaleApprox

Scale-aware approximation helpers.

This module adds an *optional* layer that tracks a per-tensor "scale bound" (a nonnegative bound
on `linf_norm`) alongside the existing `eps` error bounds.

It is designed to be used to *derive* readable abs+rel tolerances from existing eps-style proofs:
given an error budget `eps` and a scale bound `B`, we can form an `ApproxTol` whose `rel` component
is computed from `(eps / B)` (with safe handling of `B = 0`).

Nothing here changes existing forward/backward frameworks; it only provides new predicates and
lemmas you can opt into.

## PyTorch correspondence / citations
This is the proof-oriented analogue of reasoning with a magnitude/scale estimate (e.g. `‖x‖∞ ≤ B`)
to turn an absolute error budget into an `rtol`-style relative tolerance.
https://pytorch.org/docs/stable/generated/torch.allclose.html
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open NN.MLTheory.Robustness.Spec
open scoped NNReal

noncomputable section

-- ---------------------------------------------------------------------------
-- Scale vectors aligned with contexts
-- ---------------------------------------------------------------------------

/-- Nonnegative scale bounds aligned with a context shape list. -/
inductive BList : List Shape → Type where
  | nil : BList []
  | cons {s : Shape} {ss : List Shape} : ℝ≥0 → BList ss → BList (s :: ss)

namespace BList

/-- Transport a `BList` along an equality of the underlying shape lists. -/
def cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : BList ss₁) : BList ss₂ :=
  Eq.mp (congrArg BList h) xs

@[simp] theorem cast_rfl {ss : List Shape} (xs : BList ss) :
    cast (ss₁ := ss) (ss₂ := ss) rfl xs = xs := by
  cases xs <;> rfl

/-- Append one additional scale bound at the end of a `BList`. -/
def snoc {τ : Shape} : {ss : List Shape} → BList ss → ℝ≥0 → BList (ss ++ [τ])
  | [], .nil, e => .cons e .nil
  | _ :: ss, .cons x xs, e => .cons x (snoc (ss := ss) xs e)

/-- Split a `BList (ss ++ [τ])` into its prefix and the last bound. -/
def unsnoc {τ : Shape} : {ss : List Shape} → BList (ss ++ [τ]) → BList ss × ℝ≥0
  | [], .cons e .nil => (.nil, e)
  | _ :: ss, .cons x xs =>
      let (ys, last) := unsnoc (ss := ss) (τ := τ) xs
      (.cons x ys, last)

@[simp] theorem unsnoc_snoc {ss : List Shape} {τ : Shape} (xs : BList ss) (e : ℝ≥0) :
    unsnoc (ss := ss) (τ := τ) (snoc (ss := ss) (τ := τ) xs e) = (xs, e) := by
  induction ss with
  | nil =>
      cases xs
      simp [snoc, unsnoc]
  | cons s ss ih =>
      cases xs with
      | cons x xt =>
          simp [snoc, unsnoc, ih]

/-- Get the `i`th scale bound from a `BList` (using the `Fin ss.length` index). -/
def get : {ss : List Shape} → BList ss → (i : Fin ss.length) → ℝ≥0
  | [], .nil, i => nomatch i
  | _ :: _, .cons x _xs, ⟨0, _⟩ => x
  | _ :: ss, .cons _x xs, ⟨Nat.succ i, hi⟩ =>
      get (ss := ss) xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩

end BList

-- ---------------------------------------------------------------------------
-- Scale predicates (single tensor and contexts)
-- ---------------------------------------------------------------------------

/-- A scale bound says both spec and runtime (mapped to spec) norms are bounded by `B`. -/
def scaleWith {α : Type} {s : Shape}
    (toSpec : α → SpecScalar)
    (norm : ∀ {s : Shape}, SpecTensor s → SpecScalar)
    (spec : SpecTensor s)
    (runtime : Tensor α s)
    (B : ℝ≥0) : Prop :=
  let runtimeS := tensorToSpec toSpec runtime
  norm spec ≤ B ∧ norm runtimeS ≤ B

/-- Default scale predicate on tensors (uses `linf_norm`). -/
def scaleT {α : Type} {s : Shape}
    (toSpec : α → SpecScalar)
    (spec : SpecTensor s)
    (runtime : Tensor α s)
    (B : ℝ≥0) : Prop :=
  scaleWith (toSpec := toSpec) (norm := linfNorm) spec runtime B

/-- Context-level scale predicate aligned with a `BList`. -/
def scaleCtx {α : Type} (toSpec : α → SpecScalar) : {ss : List Shape} →
    TList SpecScalar ss → TList α ss → BList ss → Prop
  | [], .nil, .nil, .nil => True
  | _ :: ss, .cons x xs, .cons y ys, .cons b bs =>
      scaleT (α := α) (toSpec := toSpec) x y b ∧ scaleCtx (ss := ss) toSpec xs ys bs

lemma scaleCtx_cast {α : Type} {toSpec : α → SpecScalar} {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    {xS : TList SpecScalar ss₁} {xR : TList α ss₁} {bs : BList ss₁} :
    scaleCtx (α := α) toSpec xS xR bs →
      scaleCtx (α := α) toSpec
        (TList.cast (α := SpecScalar) (ss₁ := ss₁) (ss₂ := ss₂) h xS)
        (TList.cast (α := α) (ss₁ := ss₁) (ss₂ := ss₂) h xR)
        (BList.cast (ss₁ := ss₁) (ss₂ := ss₂) h bs) := by
  cases h
  simp

lemma scaleCtx_snoc {α : Type} {toSpec : α → SpecScalar} {ss : List Shape} {τ : Shape}
    {xS : TList SpecScalar ss} {xR : TList α ss} {bs : BList ss}
    (hx : scaleCtx (α := α) toSpec xS xR bs)
    {yS : SpecTensor τ} {yR : Tensor α τ} {b : ℝ≥0}
    (hy : scaleT (α := α) (toSpec := toSpec) yS yR b) :
    scaleCtx (α := α) toSpec
      (TList.snoc (α := SpecScalar) (ss := ss) xS yS)
      (TList.snoc (α := α) (ss := ss) xR yR)
      (BList.snoc (ss := ss) (τ := τ) bs b) := by
  induction ss with
  | nil =>
      cases xS
      cases xR
      cases bs
      simpa [TList.snoc, BList.snoc, scaleCtx] using And.intro hy True.intro
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases bs with
              | cons bh bt =>
                  have hx' : scaleCtx (α := α) toSpec xSt xRt bt := hx.2
                  have ih' := ih hx'
                  exact And.intro hx.1 ih'

lemma scaleCtx_unsnoc {α : Type} {toSpec : α → SpecScalar} {ss : List Shape} {τ : Shape}
    {xS : TList SpecScalar (ss ++ [τ])} {xR : TList α (ss ++ [τ])} {bs : BList (ss ++ [τ])} :
    scaleCtx (α := α) toSpec xS xR bs →
      scaleCtx (α := α) toSpec
          (TList.unsnoc (α := SpecScalar) (ss := ss) (τ := τ) xS).1
          (TList.unsnoc (α := α) (ss := ss) (τ := τ) xR).1
          (BList.unsnoc (ss := ss) (τ := τ) bs).1
        ∧
      scaleT (α := α) (toSpec := toSpec)
          (TList.unsnoc (α := SpecScalar) (ss := ss) (τ := τ) xS).2
          (TList.unsnoc (α := α) (ss := ss) (τ := τ) xR).2
          (BList.unsnoc (ss := ss) (τ := τ) bs).2 := by
  intro h
  induction ss with
  | nil =>
      cases xS with
      | cons tS xsS =>
          cases xsS with
          | nil =>
              cases xR with
              | cons tR xsR =>
                  cases xsR with
                  | nil =>
                      cases bs with
                      | cons b bs' =>
                          cases bs' with
                          | nil =>
                              refine And.intro ?_ ?_
                              · simp [TList.unsnoc, BList.unsnoc, scaleCtx]
                              · simpa [TList.unsnoc, BList.unsnoc, scaleCtx, scaleT, scaleWith]
                                using h.1
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases bs with
              | cons bh bt =>
                  have ht := ih (xS := xSt) (xR := xRt) (bs := bt) h.2
                  refine And.intro ?_ ht.2
                  -- prepend the head back on the prefix result
                  simpa [TList.unsnoc, BList.unsnoc, scaleCtx] using And.intro h.1 ht.1

lemma scaleCtx_get {α : Type} {toSpec : α → SpecScalar} {Γ : List Shape}
    {xS : TList SpecScalar Γ} {xR : TList α Γ} {bs : BList Γ}
    (h : scaleCtx (α := α) toSpec xS xR bs) (i : Fin Γ.length) :
    scaleT (α := α) (toSpec := toSpec)
      (TList.get (α := SpecScalar) xS i)
      (TList.get (α := α) xR i)
      (BList.get bs i) := by
  induction Γ with
  | nil =>
      cases i with
      | mk val isLt =>
          exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γ ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases bs with
              | cons bh bt =>
                  cases i with
                  | mk iVal hiVal =>
                      cases iVal with
                      | zero =>
                          simpa [TList.get, BList.get, scaleCtx, scaleT, scaleWith] using h.1
                      | succ j =>
                          have := ih (xS := xSt) (xR := xRt) (bs := bt) h.2
                            ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩
                          simpa [TList.get, BList.get] using this

-- ---------------------------------------------------------------------------
-- Derive abs+rel tolerances from (eps, scale)
-- ---------------------------------------------------------------------------

/-- A derived tolerance from an absolute error `eps` and a scale bound `B`.

We always keep the absolute component (`abs = eps`) for safety. The relative component is
computed as `eps / B` (with a `B = 0` guard) and then clamped to be nonnegative via `toNNReal`
inside `ApproxTol.ofReal`.
-/
def tolFromEpsScale (eps : ℝ) (B : ℝ≥0) : ApproxTol :=
  let rel : ℝ := if (B : ℝ) = 0 then 0 else eps / (B : ℝ)
  ApproxTol.ofReal eps rel 1

lemma absOnly_le_tolFromEpsScale (eps : ℝ) (B : ℝ≥0) :
    (ApproxTol.absOnly eps).abs ≤ (tolFromEpsScale eps B).abs ∧
    (ApproxTol.absOnly eps).rel ≤ (tolFromEpsScale eps B).rel ∧
    (ApproxTol.absOnly eps).slack ≤ (tolFromEpsScale eps B).slack := by
  constructor
  · simp [ApproxTol.absOnly, tolFromEpsScale, ApproxTol.ofReal]
  constructor
  · -- `0 ≤ Real.toNNReal _`
    simp [ApproxTol.absOnly, tolFromEpsScale, ApproxTol.ofReal]
  · simp [ApproxTol.absOnly, tolFromEpsScale, ApproxTol.ofReal]

lemma approxTTol_from_scale {α : Type} {s : Shape} {toSpec : α → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} (eps : ℝ) (B : ℝ≥0)
    (h : approxT (α := α) (toSpec := toSpec) spec runtime eps) :
    approxTTol (α := α) (toSpec := toSpec) spec runtime (tolFromEpsScale eps B) := by
  -- `approxT` -> `absOnly eps`, then enlarge tolerance (abs+rel) via monotonicity.
  have habsOnly : approxTTol (α := α) (toSpec := toSpec) spec runtime (ApproxTol.absOnly eps) := by
    -- use eps->absOnly lift lemma for `approx_with`
    have : approxWith (α := α) (toSpec := toSpec) (norm := linfNorm) spec runtime eps := by
      simpa [approxT] using h
    simpa [approxTTol] using
      (approx_with_to_approx_with_tol_absOnly (toSpec := toSpec) (norm := linfNorm)
        (spec := spec) (runtime := runtime) eps this)
  rcases absOnly_le_tolFromEpsScale eps B with ⟨habs, hrel, hslack⟩
  exact approxTTol_mono (α := α) (toSpec := toSpec) (spec := spec) (runtime := runtime)
    (tol₁ := ApproxTol.absOnly eps) (tol₂ := tolFromEpsScale eps B) habs hrel hslack habsOnly

lemma approxCtx_get_tolFromEpsScale {α : Type} {toSpec : α → SpecScalar} {Γ : List Shape}
    {xS : TList SpecScalar Γ} {xR : TList α Γ} {eps : EList Γ} {bs : BList Γ}
    (hε : approxCtx (α := α) toSpec xS xR eps) (_hB : scaleCtx (α := α) toSpec xS xR bs)
    (i : Fin Γ.length) :
    approxTTol (α := α) (toSpec := toSpec)
      (TList.get (α := SpecScalar) xS i)
      (TList.get (α := α) xR i)
      (tolFromEpsScale (EList.get eps i) (BList.get bs i)) := by
  have hi : approxT (α := α) (toSpec := toSpec)
      (TList.get (α := SpecScalar) xS i)
      (TList.get (α := α) xR i)
      (EList.get eps i) :=
    approxCtx_get (α := α) (toSpec := toSpec) (xS := xS) (xR := xR) (eps := eps) hε i
  exact approxTTol_from_scale (α := α) (toSpec := toSpec)
    (spec := TList.get (α := SpecScalar) xS i)
    (runtime := TList.get (α := α) xR i)
    (eps := EList.get eps i) (B := BList.get bs i) hi

end

end RuntimeApprox
end Proofs
