/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# `NN.MLTheory.Robustness.Spec`

Scalar-polymorphic definitions of norms/distances and basic robustness vocabulary on
TorchLean's shape-indexed tensors.
-/

@[expose] public section

open Spec

namespace NN.MLTheory.Robustness.Spec

/-!
# Robustness specifications (polymorphic)

This file defines a small vocabulary for specifying **robustness properties** of tensor-valued
functions.

All definitions are **scalar-polymorphic** in `α` via `[Context α]`, so the same spec can be
instantiated for:

- `ℝ` (paper-style theorems),
- `Float` (fast, executable sanity checks),
- `Interval` (sound enclosures for verification),
- executable IEEE-754 backends (bit-level runtime semantics).

We keep this module definition-focused: whether these norms/distances satisfy the usual metric
laws depends on additional algebraic/order assumptions on `α`, and those theorems belong in
dedicated proof developments.

The `Float` specializations of these definitions live in `NN.MLTheory.Robustness.Runtime`.

Verified bounds/certificates are proved in dedicated developments (e.g. Lipschitz bounds in
`NN.Proofs.Analysis.Lipschitz`, and certified robustness procedures in `NN.MLTheory.CROWN`).

## References

- Adversarial examples and threat models: Szegedy et al. (2013/2014); Goodfellow, Shlens & Szegedy
  (2015, FGSM); Madry et al. (2017).
- Certified robustness / verification: Wong & Kolter (2018); Cohen, Rosenfeld & Kolter (2019).
- Lipschitz-based viewpoints (one entry point): Hein & Andriushchenko (2017).
-/

variable {α : Type} [Context α]

/-! ## Norms on spec tensors -/

/--
`L∞` norm of a shape-indexed tensor.

If you flatten the tensor entries into a vector `tᵢ`, this is `maxᵢ |tᵢ|`.
-/
def tensorLinfNorm {s : Shape} (t : Tensor α s) : α :=
  match s with
  | .scalar => match t with
    | .scalar x => MathFunctions.abs x
  | .dim n _inner_s => match t with
    | .dim f =>
      (List.finRange n).foldl
        (fun acc i => max acc (tensorLinfNorm (f i)))
        0

/--
`L2` (Euclidean) norm of a shape-indexed tensor.

If you flatten the tensor entries into a vector `tᵢ`, this is `sqrt (∑ᵢ tᵢ²)`.
-/
def tensorL2Norm {s : Shape} (t : Tensor α s) : α :=
  MathFunctions.sqrt (tensor_l2_norm_squared t)
where
  tensor_l2_norm_squared {s : Shape} (t : Tensor α s) : α :=
    match s with
    | .scalar => match t with
      | .scalar x => x * x
    | .dim n _inner_s => match t with
      | .dim f =>
        (List.finRange n).foldl
          (fun acc i => acc + tensor_l2_norm_squared (f i))
          0

/-! ## Distances and balls -/

/--
Distance induced by a tensor norm:

`dist(t1,t2) = ‖t1 - t2‖`.
-/
def tensorDistance (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (t1 t2 : Tensor α s) : α :=
  norm (tensor_sub t1 t2)
where
  tensor_sub {s : Shape} : Tensor α s → Tensor α s → Tensor α s
    | .scalar x, .scalar y => .scalar (x - y)
    | .dim f1, .dim f2 => .dim (fun i => tensor_sub (f1 i) (f2 i))

@[simp] theorem tensor_distance_tensor_sub_eq_sub_spec {s : Shape} (t1 t2 : Tensor α s) :
    tensorDistance.tensor_sub t1 t2 = Spec.Tensor.subSpec t1 t2 := by
  induction s with
  | scalar =>
    cases t1 with
    | scalar x =>
      cases t2 with
      | scalar y =>
        rfl
  | dim n inner ih =>
    cases t1 with
    | dim f1 =>
      cases t2 with
      | dim f2 =>
        -- Reduce to pointwise equality of the recursive calls.
        apply congrArg Tensor.dim
        funext i
        exact ih (t1 := f1 i) (t2 := f2 i)

@[simp] theorem tensor_distance_eq_norm_sub_spec (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (t1 t2 : Tensor α s) :
    tensorDistance (α := α) norm t1 t2 = norm (Spec.Tensor.subSpec t1 t2) := by
  simp [tensorDistance]

/--
Closed `ε`-ball around `center` for the given norm:

`{ t | dist(center,t) ≤ ε }`.
-/
def tensorBall (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (center : Tensor α s) (ε : α) : Set (Tensor α s) :=
  {t | tensorDistance norm center t ≤ ε}

/-! ## Continuity / robustness specifications -/

/--
Lipschitz continuity (global), phrased using `tensor_distance`.

If `f` is `L`-Lipschitz and `tensor_distance norm₁ x₀ x ≤ ε`, then
`tensor_distance norm₂ (f x₀) (f x) ≤ L*ε`.
-/
def isLipschitzContinuous {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (L : α) : Prop :=
  ∀ x y : Tensor α s₁,
    tensorDistance norm₂ (f x) (f y) ≤ L * tensorDistance norm₁ x y

/--
Local Lipschitz continuity within the `ε`-ball around `x₀`.
-/
def isLocallyLipschitz {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (x₀ : Tensor α s₁) (ε : α) (L : α) : Prop :=
  ∀ x y : Tensor α s₁,
    x ∈ tensorBall norm₁ x₀ ε → y ∈ tensorBall norm₁ x₀ ε →
    tensorDistance norm₂ (f x) (f y) ≤ L * tensorDistance norm₁ x y

/--
Adversarial robustness at a point `x₀`.

`f` is `(ε,δ)`-robust at `x₀` if every input within distance `ε` of `x₀` maps to an output within
distance `δ` of `f x₀`.
-/
def isAdversariallyRobust {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (x₀ : Tensor α s₁) (ε δ : α) : Prop :=
  ∀ x : Tensor α s₁,
    tensorDistance norm₁ x₀ x ≤ ε →
    tensorDistance norm₂ (f x₀) (f x) ≤ δ

/--
Certified robustness for a classifier: the prediction is constant on the `ε`-ball around `x₀`.

For neural networks, `classifier` is typically `argmax` on a logits tensor.
-/
def isCertifiedRobust {s : Shape}
    (classifier : Tensor α s → Nat)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (x₀ : Tensor α s) (ε : α) : Prop :=
  ∀ x : Tensor α s,
    x ∈ tensorBall norm x₀ ε →
    classifier x = classifier x₀

/--
Uniform adversarial robustness over a finite list of inputs.
-/
def isUniformlyRobust {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (dataset : List (Tensor α s₁)) (ε δ : α) : Prop :=
  ∀ x₀ ∈ dataset, isAdversariallyRobust f norm₁ norm₂ x₀ ε δ

/--
Contraction mapping under a norm: `f` shrinks distances by `contraction_factor < 1`.

This is a standard sufficient condition for convergence of iterated dynamics and robustness of
fixed points.
-/
def isContractive {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (contraction_factor : α) : Prop :=
  contraction_factor < 1 ∧
  isLipschitzContinuous f norm norm contraction_factor

/--
Sensitivity ratio for a specific additive perturbation.

This is the local “output change divided by input change” quantity:

`‖f(x) - f(x + perturbation)‖ / ‖perturbation‖`.
-/
def sensitivity {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (x : Tensor α s₁) (perturbation : Tensor α s₁) : α :=
  let output_change := tensorDistance norm₂ (f x) (f (Spec.Tensor.addSpec x perturbation))
  let input_change := norm₁ perturbation
  output_change / input_change



end NN.MLTheory.Robustness.Spec
