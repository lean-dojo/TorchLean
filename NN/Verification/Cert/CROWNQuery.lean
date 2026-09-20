/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.RationalReflection
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Activation

/-!
# Exact CROWN output-query checks

Recompute affine bounds for dense/ReLU networks and prove strict or non-strict output inequalities
over an input box. The arithmetic is rational; the conclusion uses TorchLean's existing real module
semantics. Every transfer and final query check is proved here, with no external soundness premise.
-/

@[expose] public section

namespace NN.Verification.CROWNQuery

open _root_.Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open NN.Verification.Cert.RationalReflection
open NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness
open NN.MLTheory.CROWN.Graph.CertSoundness
open NN.MLTheory.CROWN.Graph.CrownCertSoundness
open scoped _root_.Spec.RationalAlgebraic

/-- Check all upper endpoints against zero, using strict inequalities when requested. -/
def checkUpper {n m : Nat} (strict : Bool) (upper : AffineVec ℚ n m)
    (input : Box ℚ [n]) : Bool :=
  let bounds := IBP.linear upper.A input ⟨upper.c, upper.c⟩
  (List.finRange m).all fun i =>
    if strict then decide (bounds.hi.getScalar i < 0)
    else decide (bounds.hi.getScalar i ≤ 0)

/-- Every accepted upper form is below zero throughout the real input box. -/
theorem checkUpper_sound {n m : Nat} (strict : Bool) (upper : AffineVec ℚ n m)
    (input : Box ℚ [n]) (h : checkUpper strict upper input = true)
    (x : Tensor ℝ [n])
    (hx : Theorems.Semantics.encloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ x) (i : Fin m) :
    if strict then (affineEvalAt (realAffine upper) x).getScalar i < 0
    else (affineEvalAt (realAffine upper) x).getScalar i ≤ 0 := by
  have hbound := Theorems.ibp_linear_sound_real
    (realTensor upper.A) (realBox input) (Box.point (realTensor upper.c))
    x (realTensor upper.c) hx (Box.contains_point_self _)
  have hi := (hbound i).2
  change (affineEvalAt (realAffine upper) x).getScalar i ≤
    (IBP.linear (realTensor upper.A) (realBox input)
      (realBox ⟨upper.c, upper.c⟩)).hi.getScalar i at hi
  have hcheck := (List.all_eq_true.mp h) i (List.mem_finRange i)
  have hmap := congrArg (fun b : Box ℝ [m] => b.hi.getScalar i)
    (realBox_linear upper.A input ⟨upper.c, upper.c⟩)
  simp only [realBox, realTensor_getScalar] at hmap
  change (affineEvalAt (realAffine upper) x).getScalar i ≤ _ at hi
  simp only [realBox] at hi
  rw [← hmap] at hi
  cases strict with
  | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at hcheck ⊢
      have hq := of_decide_eq_true hcheck
      have hr : (((IBP.linear upper.A input ⟨upper.c, upper.c⟩).hi.getScalar i : ℚ)
        : ℝ) ≤ 0 := by exact_mod_cast hq
      exact hi.trans hr
  | true =>
      simp only [↓reduceIte] at hcheck ⊢
      have hq := of_decide_eq_true hcheck
      have hr : (((IBP.linear upper.A input ⟨upper.c, upper.c⟩).hi.getScalar i : ℚ)
        : ℝ) < 0 := by exact_mod_cast hq
      exact hi.trans_lt hr

/-- Affine bounds with their input and output dimensions checked by Lean. -/
structure Bounds (n m : Nat) where
  lo : AffineVec ℚ n m
  hi : AffineVec ℚ n m

def Bounds.flat {n m : Nat} (b : Bounds n m) : FlatAffineBounds ℚ :=
  ⟨n, m, b.lo, b.hi⟩

/-- The bounds enclose a value at a particular real input. -/
def Bounds.Encloses {n m : Nat} (b : Bounds n m) (x : Tensor ℝ [n])
    (y : Tensor ℝ [m]) : Prop :=
  ∀ i, (affineEvalAt (realAffine b.lo) x).getScalar i ≤ y.getScalar i ∧
    y.getScalar i ≤ (affineEvalAt (realAffine b.hi) x).getScalar i

def Bounds.identity (n : Nat) : Bounds n n :=
  ⟨NN.MLTheory.CROWN.Cert.affIdentity n, NN.MLTheory.CROWN.Cert.affIdentity n⟩

def Bounds.linear {n m k : Nat} (b : Bounds n m) (w : Tensor ℚ [k, m])
    (bias : Tensor ℚ [k]) : Bounds n k :=
  let out := NN.MLTheory.CROWN.Cert.linearBoundsFromAffine w bias b.flat rfl
  ⟨out.loAff, out.hiAff⟩

/-- Bound each affine form over the original input box. -/
def Bounds.interval {n m : Nat} (b : Bounds n m) (input : Box ℚ [n]) : Box ℚ [m] :=
  ⟨(IBP.linear b.lo.A input ⟨b.lo.c, b.lo.c⟩).lo,
   (IBP.linear b.hi.A input ⟨b.hi.c, b.hi.c⟩).hi⟩

def Bounds.relu {n m : Nat} (b : Bounds n m) (input : Box ℚ [n])
    (alpha : Tensor ℚ [m]) : Bounds n m :=
  let pre := b.interval input
  ⟨Runtime.Ops.ReLU.propagateAffine
      (NN.MLTheory.CROWN.Cert.alphaRelaxLowerVec pre.lo pre.hi alpha) b.lo,
   Runtime.Ops.ReLU.propagateAffine (Runtime.Ops.ReLU.relaxVector pre.lo pre.hi) b.hi⟩

theorem Bounds.identity_encloses {n : Nat} (x : Tensor ℝ [n]) :
    (Bounds.identity n).Encloses x x := by
  have h := boundsEvalAt_bounds_identity x
  have hr := realAffineBounds_identity n
  simp only [realAffineBounds, NN.MLTheory.CROWN.Cert.boundsIdentity] at hr
  injection hr with _ _ hloAff hhiAff
  simp only [boundsEvalAt, NN.MLTheory.CROWN.Cert.boundsIdentity] at h
  injection h with _ hlo hhi
  intro i
  constructor
  · simpa [identity, ← hloAff] using le_of_eq (congrArg (fun t => t.getScalar i) hlo)
  · simpa [identity, ← hhiAff] using le_of_eq (congrArg (fun t => t.getScalar i) hhi).symm

theorem Bounds.linear_encloses {n m k : Nat} (b : Bounds n m)
    (w : Tensor ℚ [k, m]) (bias : Tensor ℚ [k]) (x : Tensor ℝ [n])
    (y : Tensor ℝ [m]) (h : b.Encloses x y) :
    (b.linear w bias).Encloses x
      (Tensor.addSpec (matVecMulSpec (realTensor w) y) (realTensor bias)) := by
  have hbox := (encloses_iff_getScalar _ _ _).mpr h
  have hs := encloses_linear_signSplit (realTensor w) (realTensor bias) _ _ y hbox
  have hs' := (encloses_iff_getScalar _ _ _).mp hs
  rw [← affineEvalAt_linear_pair (IBP.matPos (realTensor w)) (IBP.matNeg (realTensor w))
    (realAffine b.lo) (realAffine b.hi) (realTensor bias) x] at hs'
  rw [← affineEvalAt_linear_pair (IBP.matPos (realTensor w)) (IBP.matNeg (realTensor w))
    (realAffine b.hi) (realAffine b.lo) (realTensor bias) x] at hs'
  simp only [Encloses, linear, flat, NN.MLTheory.CROWN.Cert.linearBoundsFromAffine,
    realAffine, realTensor_add, realTensor_matMul,
    realTensor_matVecMul, realTensor_matPos, realTensor_matNeg]
  exact hs'

theorem affine_interval_encloses {n m : Nat} (a : AffineVec ℚ n m)
    (input : Box ℚ [n]) (x : Tensor ℝ [n])
    (hx : Theorems.Semantics.encloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ x) :
    ∀ i, (realTensor (IBP.linear a.A input ⟨a.c, a.c⟩).lo).getScalar i ≤
      (affineEvalAt (realAffine a) x).getScalar i ∧
      (affineEvalAt (realAffine a) x).getScalar i ≤
        (realTensor (IBP.linear a.A input ⟨a.c, a.c⟩).hi).getScalar i := by
  have h := Theorems.ibp_linear_sound_real
    (realTensor a.A) (realBox input) (Box.point (realTensor a.c))
    x (realTensor a.c) hx (Box.contains_point_self _)
  have hm := realBox_linear a.A input ⟨a.c, a.c⟩
  have hlo := congrArg Box.lo hm
  have hhi := congrArg Box.hi hm
  simp only [realBox] at hlo hhi
  intro i
  have hi := h i
  change (IBP.linear (realTensor a.A) (realBox input)
      (Box.point (realTensor a.c))).lo.getScalar i ≤ _ ∧
    _ ≤ (IBP.linear (realTensor a.A) (realBox input)
      (Box.point (realTensor a.c))).hi.getScalar i at hi
  simp only [realBox, Box.point] at hi
  rw [← hlo, ← hhi] at hi
  exact hi

theorem Bounds.interval_encloses {n m : Nat} (b : Bounds n m) (input : Box ℚ [n])
    (x : Tensor ℝ [n]) (y : Tensor ℝ [m])
    (hx : Theorems.Semantics.encloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ x) (hy : b.Encloses x y) :
    ∀ i, (realTensor (b.interval input).lo).getScalar i ≤ y.getScalar i ∧
      y.getScalar i ≤ (realTensor (b.interval input).hi).getScalar i := by
  intro i
  exact ⟨((affine_interval_encloses b.lo input x hx i).1).trans (hy i).1,
    (hy i).2 |>.trans (affine_interval_encloses b.hi input x hx i).2⟩

def checkAlpha {m : Nat} (alpha : Tensor ℚ [m]) : Bool :=
  (List.finRange m).all fun i => decide (0 ≤ alpha.getScalar i ∧ alpha.getScalar i ≤ 1)

theorem checkAlpha_sound {m : Nat} (alpha : Tensor ℚ [m]) (h : checkAlpha alpha = true)
    (i : Fin m) : 0 ≤ (realTensor alpha).getScalar i ∧ (realTensor alpha).getScalar i ≤ 1 := by
  have hq := of_decide_eq_true (List.all_eq_true.mp h i (List.mem_finRange i))
  simp only [realTensor_getScalar]
  exact_mod_cast hq

theorem Bounds.relu_encloses {n m : Nat} (b : Bounds n m) (input : Box ℚ [n])
    (alpha : Tensor ℚ [m]) (ha : checkAlpha alpha = true)
    (x : Tensor ℝ [n]) (y : Tensor ℝ [m])
    (hx : Theorems.Semantics.encloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ x) (hy : b.Encloses x y) :
    (b.relu input alpha).Encloses x (Activation.reluSpec y) := by
  have hpre := b.interval_encloses input x y hx hy
  intro i
  have hai := checkAlpha_sound alpha ha i
  have hl := NN.MLTheory.CROWN.Proofs.alphaRelaxLowerScalar_sound
    (lowerBound := (hpre i).1) (upperBound := (hpre i).2)
    (alphaNonnegative := hai.1) (alphaAtMostOne := hai.2)
  have hu := relu_relax_scalar_upper_real_runtime _ _ _ (hpre i).1 (hpre i).2
  have sl := alphaRelaxLowerScalar_slope_nonneg
    ((realTensor (b.interval input).lo).getScalar i)
    ((realTensor (b.interval input).hi).getScalar i)
    ((realTensor alpha).getScalar i) hai.1
  have su := relax_scalar_slope_nonneg
    ((realTensor (b.interval input).lo).getScalar i)
    ((realTensor (b.interval input).hi).getScalar i)
  dsimp only [relu]
  simp only [realAffine_propagate, realRelax_lowerVec, realRelax_upperVec,
    getScalar_affineEvalAt_relu_propagate_affine, getScalar_alphaRelaxLowerVec,
    getScalar_runtime_relu_relax_vector, getScalar_relu_spec]
  exact ⟨(add_le_add (mul_le_mul_of_nonneg_left (hy i).1 sl) le_rfl).trans hl,
    hu.trans (add_le_add (mul_le_mul_of_nonneg_left (hy i).2 su) le_rfl)⟩

/-- Dense/ReLU networks with a proposed lower-relaxation slope at each ReLU. -/
inductive Network : Nat → Nat → Type where
  | linear {n m : Nat} (layer : LinearSpec ℚ n m) : Network n m
  | relu {n : Nat} (alpha : Tensor ℚ [n]) : Network n n
  | comp {n m k : Nat} : Network n m → Network m k → Network n k

/-- Use the same linear and activation modules as ordinary TorchLean models. -/
noncomputable def Network.model {n m : Nat} : Network n m → Module.Chain ℝ [n] [m]
  | .linear layer => .single (Module.linear ⟨realTensor layer.weights, realTensor layer.bias⟩)
  | .relu _ => .single (Module.relu [n])
  | .comp first rest => .comp first.model rest.model

/-- Recompute affine bounds with existing CROWN transfers and exact rational arithmetic. -/
def Network.propagate {n m k : Nat} (net : Network m k) (b : Bounds n m)
    (input : Box ℚ [n]) : Option (Bounds n k) :=
  match net with
  | .linear layer => some (b.linear layer.weights layer.bias)
  | .relu alpha => if checkAlpha alpha then some (b.relu input alpha) else none
  | .comp first rest => do
      let mid ← first.propagate b input
      rest.propagate mid input

/-- Replayed bounds enclose the whole network, not just individual certificate entries. -/
theorem Network.propagate_sound {n m k : Nat} (net : Network m k) (b : Bounds n m)
    (input : Box ℚ [n]) (out : Bounds n k) (h : net.propagate b input = some out)
    (x : Tensor ℝ [n]) (y : Tensor ℝ [m])
    (hx : Theorems.Semantics.encloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ x) (hy : b.Encloses x y) :
    out.Encloses x (net.model.forward y) := by
  induction net with
  | linear layer =>
      cases Option.some.inj h
      exact b.linear_encloses layer.weights layer.bias x y hy
  | relu alpha =>
      simp only [propagate] at h
      split at h
      · rename_i ha
        cases Option.some.inj h
        exact b.relu_encloses input alpha ha x y hx hy
      · contradiction
  | comp first rest hf hr =>
      simp only [propagate, Option.bind_eq_bind] at h
      cases hm : first.propagate b input with
      | none => simp [hm] at h
      | some mid =>
          have ht : rest.propagate mid input = some out := by simpa [hm] using h
          exact hr mid out ht (first.model.forward y) (hf b mid hm y hy)

/-- A conjunction of affine output inequalities, interpreted as `weights * output + bias`. -/
structure Query (n m : Nat) where
  network : Network n m
  input : Box ℚ [n]
  numConstraints : Nat
  inequalities : LinearSpec ℚ m numConstraints
  strict : Bool

/-- Reject reversed boxes and empty input, output, or query dimensions. -/
def Query.wellFormed {n m : Nat} (q : Query n m) : Bool :=
  decide (0 < n ∧ 0 < m ∧ 0 < q.numConstraints) &&
    (List.finRange n).all fun i => decide (q.input.lo.getScalar i ≤ q.input.hi.getScalar i)

/-- An executable safety check; no externally supplied affine bounds are trusted. -/
def Query.check {n m : Nat} (q : Query n m) : Bool :=
  q.wellFormed && match q.network.propagate (Bounds.identity n) q.input with
  | none => false
  | some out => checkUpper q.strict
      (out.linear q.inequalities.weights q.inequalities.bias).hi q.input

/-- Every real input in the decoded box satisfies every decoded output inequality. -/
def Query.Safe {n m : Nat} (q : Query n m) : Prop :=
  ∀ x : Tensor ℝ [n],
    Theorems.Semantics.encloses ⟨n, realTensor q.input.lo, realTensor q.input.hi⟩ x →
    ∀ i : Fin q.numConstraints,
      let result := Tensor.addSpec
        (matVecMulSpec (realTensor q.inequalities.weights) (q.network.model.forward x))
        (realTensor q.inequalities.bias)
      if q.strict then result.getScalar i < 0 else result.getScalar i ≤ 0

/-- Acceptance implies query-level soundness with no unproved transfer or coverage premise. -/
theorem Query.check_sound {n m : Nat} (q : Query n m) (h : q.check = true) : q.Safe := by
  simp only [check, Bool.and_eq_true] at h
  obtain ⟨_, hc⟩ := h
  cases hp : q.network.propagate (Bounds.identity n) q.input with
  | none => simp [hp] at hc
  | some out =>
      have hg : checkUpper q.strict
          (out.linear q.inequalities.weights q.inequalities.bias).hi q.input = true := by
        simpa [hp] using hc
      intro x hx i
      have hn := q.network.propagate_sound (Bounds.identity n) q.input out hp x x hx
        (Bounds.identity_encloses x)
      have hq := out.linear_encloses q.inequalities.weights q.inequalities.bias x
        (q.network.model.forward x) hn
      have hu := checkUpper_sound q.strict _ q.input hg x hx i
      have hi := (hq i).2
      cases hstrict : q.strict <;> simp only [hstrict, Bool.false_eq_true, ↓reduceIte] at hu ⊢
      · exact hi.trans hu
      · exact hi.trans_lt hu

end NN.Verification.CROWNQuery
