/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Algebra
public import TorchLean.NN.Sequential
public import TorchLean.NN.Layer.Activation
public import TorchLean.NN.Layer.Recurrent
public import TorchLean.NN.Sequential.Composition
public import TorchLean.Tensor.Context.Rational
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Contract

/-!
# Exact monotonicity certificates

A certificate records linear and ReLU layers using exact rational parameters. Its model is an
`nn.Sequential` evaluated by `nn.Sequential.eval` over any linearly ordered field with a lawful
`Context`, such as the reals and the rationals. Nonnegative weights are a sufficient but
incomplete test for global componentwise monotonicity, and biases are unrestricted.

Monotonicity relates pairs of inputs (a hyperproperty), and `check_sound` proves it for every pair
in exact arithmetic, which leaves floating-point evaluation outside its scope. The check reads only
the weights, so it enumerates no activation regions and builds no CROWN certificate.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean.Monotonicity

open Tensor
open nn
open scoped RationalAlgebraic

/-- Componentwise order preservation for a tensor map. -/
def PreservesOrder {α : Type} [Storage α] [LE α] {s t : Shape}
    (f : Tensor α s → Tensor α t) : Prop :=
  ∀ x y, Tensor.Forall₂ (· ≤ ·) x y → Tensor.Forall₂ (· ≤ ·) (f x) (f y)

/-- Serializable evidence for a composition of linear and ReLU layers. -/
inductive Certificate : Shape → Shape → Type where
  /-- A dense layer. -/
  | linear {n m : Nat} (layer : Linear ℚ n m) : Certificate [n] [m]
  /-- A ReLU. -/
  | relu (s : Shape) : Certificate s s
  /-- One certificate after another. -/
  | comp {s t u : Shape} : Certificate s t → Certificate t u → Certificate s u

/-- The certificate's model: its linear and ReLU layers in sequence. -/
def Certificate.model {s t : Shape} : Certificate s t → nn.Sequential s t
  | @Certificate.linear n m _ => .fromLayer (Layer.linear n m)
  | .relu _ => .fromLayer (Layer.op .relu)
  | .comp a b => a.model >>> b.model

/-- The certificate's parameters at the scalar `α`, each rational cast to `α`. -/
def Certificate.params {α : Type} [Storage α] [RatCast α] {s t : Shape} :
    (cert : Certificate s t) → TensorPack α (nn.Sequential.stateShapes cert.model)
  | .linear layer => .cons (Tensor.map (fun q : ℚ => (q : α)) layer.weights)
      (.cons (Tensor.map (fun q : ℚ => (q : α)) layer.bias) .nil)
  | .relu _ => .nil
  | .comp a b => (nn.Sequential.stateShapes_comp a.model b.model).symm ▸ a.params.append b.params

/-- Exact executable nonnegativity check for a weight matrix. -/
def checkWeights {n m : Nat} (weights : Tensor ℚ [m, n]) : Bool :=
  (List.finRange m).all fun i =>
    (List.finRange n).all fun j => decide (0 ≤ get2 weights i j)

/-- Accept only certificates whose linear layers have nonnegative weights. -/
def Certificate.check {s t : Shape} : Certificate s t → Bool
  | .linear layer => checkWeights layer.weights
  | .relu _ => true
  | .comp a b => check a && check b

private theorem vector_order_iff {α : Type} [Storage α] [LE α] {n : Nat} (x y : Tensor α [n]) :
    Tensor.Forall₂ (· ≤ ·) x y ↔ ∀ i, x.getScalar i ≤ y.getScalar i := by
  rfl

/-- A successful exact weight check certifies that every rational entry is nonnegative. -/
theorem checkWeights_nonneg {n m : Nat} {weights : Tensor ℚ [m, n]}
    (h : checkWeights weights = true) (i : Fin m) (j : Fin n) : 0 ≤ get2 weights i j := by
  simp only [checkWeights, List.all_eq_true] at h
  exact of_decide_eq_true (h i (List.mem_finRange i) j (List.mem_finRange j))

/-- A linear layer with nonnegative weights preserves componentwise order. -/
theorem preservesOrder_linear {α : Type} [Storage α] [CommRing α] [LinearOrder α]
    [IsStrictOrderedRing α] {n m : Nat} (layer : Linear α n m)
    (hw : ∀ i j, 0 ≤ get2 layer.weights i j) :
    PreservesOrder (Linear.forward layer) := by
  intro x y hxy
  apply (vector_order_iff _ _).mpr
  intro i
  simp only [Linear.forward, Tensor.add_def, Tensor.getScalar_zipWith,
    Tensor.getScalar_contract_matVec_eq_sum]
  refine add_le_add ?_ le_rfl
  exact Finset.sum_le_sum fun j _ =>
    mul_le_mul_of_nonneg_left ((vector_order_iff x y).mp hxy j) (hw i j)

/-- ReLU preserves componentwise order at every tensor rank. -/
theorem preservesOrder_relu {α : Type} [Storage α] [LinearOrder α] [Zero α] (s : Shape) :
    PreservesOrder (Tensor.relu (α := α) (s := s)) := by
  intro x y hxy
  induction s with
  | scalar =>
      simpa [Tensor.Forall₂, Tensor.relu,
        Activation.relu_eq_max] using max_le_max hxy (le_refl (0 : α))
  | dim n s ih =>
      intro i
      simpa [Tensor.relu] using
        ih (x.unstack i) (y.unstack i) (hxy i)

/-
`nn.Sequential.eval` computes with the `Context` operations. In a lawful context these are the
ordered field operations, so `check_sound` rewrites the instances before applying the two lemmas
above.
-/
section

variable {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α] {context : Context α}
  [@LawfulContext α context _ _ _]

private theorem context_toAdd_eq : context.toAdd = Distrib.toAdd := by
  show Add.mk (fun x y => @HAdd.hAdd α α α (@instHAdd α context.toAdd) x y) =
    Add.mk (fun x y => @HAdd.hAdd α α α (@instHAdd α Distrib.toAdd) x y)
  simp only [LawfulContext.add_eq]

private theorem context_toMul_eq : context.toMul = Distrib.toMul := by
  show Mul.mk (fun x y => @HMul.hMul α α α (@instHMul α context.toMul) x y) =
    Mul.mk (fun x y => @HMul.hMul α α α (@instHMul α Distrib.toMul) x y)
  simp only [LawfulContext.mul_eq]

private theorem context_toZero_eq : context.toZero = MulZeroClass.toZero := by
  show Zero.mk (@Zero.zero α context.toZero) = Zero.mk (@Zero.zero α MulZeroClass.toZero)
  rw [LawfulContext.zero_eq]

private theorem context_toMax_eq : context.toMax = LinearOrder.toMax := by
  show Max.mk (fun x y => @Max.max α context.toMax x y) =
    Max.mk (fun x y => @Max.max α LinearOrder.toMax x y)
  simp only [LawfulContext.max_eq]

private theorem context_toBEq_eq : context.toBEq = instBEqOfDecidableEq := by
  show BEq.mk (fun x y => @BEq.beq α context.toBEq x y) =
    BEq.mk (fun x y => @BEq.beq α instBEqOfDecidableEq x y)
  congr
  funext x y
  exact Bool.eq_iff_iff.mpr
    ((LawfulContext.beq_iff x y).trans (@beq_iff_eq α instBEqOfDecidableEq _ x y).symm)

private theorem context_toLE_eq : context.toLE = Preorder.toLE := by
  show LE.mk (fun x y => @LE.le α context.toLE x y) =
    LE.mk (fun x y => @LE.le α Preorder.toLE x y)
  simp only [LawfulContext.le_iff]

private theorem context_toRatCast_eq : context.toRatCast = DivisionRing.toRatCast := by
  show RatCast.mk (fun q => @Rat.cast α context.toRatCast q) =
    RatCast.mk (fun q => @Rat.cast α DivisionRing.toRatCast q)
  simp only [LawfulContext.ratCast_eq]

end

/-- Acceptance proves global monotonicity of the recorded TorchLean model over any linearly
ordered field with a lawful `Context`, such as `ℝ`, or `ℚ` with `RationalAlgebraic` open. -/
theorem Certificate.check_sound {α : Type} [Storage α] [Field α] [LinearOrder α]
    [IsStrictOrderedRing α]
    {context : Context α} [@LawfulContext α context _ _ _] {s t : Shape}
    (cert : Certificate s t) (h : check cert = true) :
    PreservesOrder (cert.model.eval (cert.params (α := α))) := by
  induction cert with
  | @linear n m layer =>
      change @PreservesOrder α _ context.toLE [n] [m]
        (@Linear.forward α _ context.toAdd context.toMul context.toZero n m
          ⟨Tensor.map (fun q : ℚ => @Rat.cast α context.toRatCast q) layer.weights,
            Tensor.map (fun q : ℚ => @Rat.cast α context.toRatCast q) layer.bias⟩)
      rw [context_toLE_eq (context := context), context_toAdd_eq (context := context),
        context_toMul_eq (context := context), context_toZero_eq (context := context),
        context_toRatCast_eq (context := context)]
      refine preservesOrder_linear _ fun i j => ?_
      simpa [get2_eq_getScalar_unstack, Tensor.unstack_map] using
        (Rat.cast_nonneg (K := α)).mpr (checkWeights_nonneg h i j)
  | relu s =>
      change @PreservesOrder α _ context.toLE s s
        (@Tensor.relu α _ context.toZero context.toMax context.toBEq s)
      rw [context_toLE_eq (context := context), context_toMax_eq (context := context),
        context_toBEq_eq (context := context)]
      exact preservesOrder_relu s
  | comp a b ha hb =>
      simp only [check, Bool.and_eq_true] at h
      intro x y hxy
      simp only [Certificate.model, Certificate.params, nn.Sequential.eval_comp]
      exact hb h.2 _ _ (ha h.1 x y hxy)

end TorchLean.Monotonicity
