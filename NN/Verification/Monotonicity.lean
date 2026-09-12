/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
public import NN.Spec.Module.Activation
public import NN.Spec.Module.Linear
public import NN.Spec.Core.Context.Rational
public import NN.Spec.Core.Context.Real
public import Mathlib.Data.Rat.Cast.Order

/-!
# Exact monotonicity certificates

A certificate records linear and ReLU modules using exact rational parameters. Its interpretation
uses the existing `Spec.Module.Chain` semantics over the reals. Checking nonnegative weights is a
sufficient, incomplete test for global componentwise monotonicity. Biases are unrestricted.

This checks a universal real-valued hyperproperty, not a floating-point deployment guarantee or a
CROWN certificate. It does not enumerate activation regions.
-/

@[expose] public section

namespace NN.Verification.Monotonicity

open _root_.Spec TorchLean TorchLean.Tensor
open scoped BigOperators
open scoped Spec.RationalAlgebraic

/-- Componentwise order preservation for a tensor map. -/
def PreservesOrder {α : Type} [Storage α] [LE α] {s t : Shape}
    (f : Tensor α s → Tensor α t) : Prop :=
  ∀ x y, Tensor.Forall₂ (· ≤ ·) x y → Tensor.Forall₂ (· ≤ ·) (f x) (f y)

/-- Serializable evidence for a composition of existing linear and ReLU modules. -/
inductive Certificate : Shape → Shape → Type where
  | linear {n m : Nat} (layer : LinearSpec ℚ n m) : Certificate [n] [m]
  | relu (s : Shape) : Certificate s s
  | comp {s t u : Shape} : Certificate s t → Certificate t u → Certificate s u

/-- Interpret exact rational layer parameters as real numbers. -/
noncomputable def realLayer {n m : Nat} (layer : LinearSpec ℚ n m) : LinearSpec ℝ n m :=
  { weights := Tensor.map (fun q : ℚ => (q : ℝ)) layer.weights
    bias := Tensor.map (fun q : ℚ => (q : ℝ)) layer.bias }

/-- The certificate's model is an ordinary TorchLean mathematical module chain. -/
noncomputable def Certificate.model {s t : Shape} : Certificate s t → Module.Chain ℝ s t
  | .linear layer => .single (Module.linear (realLayer layer))
  | .relu s => .single (Module.relu s)
  | .comp a b => .comp a.model b.model

/-- Execute the same recorded linear/ReLU model using exact rational arithmetic. -/
def Certificate.rationalModel {s t : Shape} : Certificate s t → Module.Chain ℚ s t
  | .linear layer => .single (Module.linear layer)
  | .relu s => .single (Module.relu s)
  | .comp a b => .comp a.rationalModel b.rationalModel

/-- Exact executable nonnegativity check for a weight matrix. -/
def checkWeights {n m : Nat} (weights : Tensor ℚ [m, n]) : Bool :=
  (List.finRange m).all fun i =>
    (List.finRange n).all fun j => decide (0 ≤ get2 weights i j)

/-- Accept only certificates whose linear modules have nonnegative weights. -/
def check {s t : Shape} : Certificate s t → Bool
  | .linear layer => checkWeights layer.weights
  | .relu _ => true
  | .comp a b => check a && check b

private theorem vector_order_iff {α : Type} [Storage α] [LE α] {n : Nat} (x y : Tensor α [n]) :
    Tensor.Forall₂ (· ≤ ·) x y ↔ ∀ i, x.getScalar i ≤ y.getScalar i := by
  rfl

private theorem scalar_add {α : Type} [Storage α] [Add α] {n : Nat}
    (x y : Tensor α [n]) (i : Fin n) :
    (Tensor.addSpec x y).getScalar i = x.getScalar i + y.getScalar i := by
  simp [Tensor.addSpec]

/-- A successful exact weight check supplies nonnegativity over the real interpretation. -/
theorem checkWeights_sound {n m : Nat} {weights : Tensor ℚ [m, n]}
    (h : checkWeights weights = true) (i : Fin m) (j : Fin n) :
    0 ≤ get2 (Tensor.map (fun q : ℚ => (q : ℝ)) weights) i j := by
  simp only [checkWeights, List.all_eq_true] at h
  have hq : 0 ≤ get2 weights i j :=
    of_decide_eq_true (h i (List.mem_finRange i) j (List.mem_finRange j))
  have hr : (0 : ℝ) ≤ ((get2 weights i j : ℚ) : ℝ) := by exact_mod_cast hq
  simpa [get2_eq_getScalar_get, Spec.get, Tensor.unstack_map] using hr

/-- A linear module with nonnegative weights preserves componentwise real order. -/
theorem linear_preserves_order {α : Type} [Storage α] [CommRing α] [LinearOrder α]
    [IsStrictOrderedRing α] {n m : Nat} (layer : LinearSpec α n m)
    (hw : ∀ i j, 0 ≤ get2 layer.weights i j) :
    PreservesOrder (Spec.linearSpec layer) := by
  intro x y hxy
  apply (vector_order_iff _ _).mpr
  intro i
  simp only [Spec.linearSpec, scalar_add, Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec]
  refine add_le_add ?_ le_rfl
  exact Finset.sum_le_sum fun j _ =>
    mul_le_mul_of_nonneg_left ((vector_order_iff x y).mp hxy j) (hw i j)

/-- ReLU preserves componentwise order at every tensor rank. -/
theorem relu_preserves_order {α : Type} [Storage α] [LinearOrder α] [Zero α] (s : Shape) :
    PreservesOrder (Activation.reluSpec (α := α) (s := s)) := by
  intro x y hxy
  induction s with
  | scalar =>
      simpa [Tensor.Forall₂, Activation.reluSpec, Tensor.mapSpec,
        Activation.Math.reluSpec_eq_max] using max_le_max hxy (le_refl (0 : α))
  | dim n s ih =>
      intro i
      simpa [Activation.reluSpec, Tensor.mapSpec, ← Tensor.unstack_map] using
        ih (x.unstack i) (y.unstack i) (hxy i)

/-- Acceptance proves global monotonicity of the recorded TorchLean model over real inputs. -/
theorem check_sound {s t : Shape} (cert : Certificate s t)
    (h : check cert = true) : PreservesOrder cert.model.forward := by
  induction cert with
  | linear layer =>
      exact linear_preserves_order (realLayer layer) (checkWeights_sound h)
  | relu s => exact relu_preserves_order s
  | comp a b ha hb =>
      simp only [check, Bool.and_eq_true] at h
      intro x y hxy
      exact hb h.2 _ _ (ha h.1 x y hxy)

/-- Acceptance also proves order preservation for the exact rational execution semantics. -/
theorem check_sound_rat {s t : Shape} (cert : Certificate s t)
    (h : check cert = true) : PreservesOrder cert.rationalModel.forward := by
  induction cert with
  | linear layer =>
      apply linear_preserves_order layer
      intro i j
      simp only [check, checkWeights, List.all_eq_true] at h
      exact of_decide_eq_true (h i (List.mem_finRange i) j (List.mem_finRange j))
  | relu s => exact relu_preserves_order s
  | comp a b ha hb =>
      simp only [check, Bool.and_eq_true] at h
      intro x y hxy
      exact hb h.2 _ _ (ha h.1 x y hxy)

end NN.Verification.Monotonicity
