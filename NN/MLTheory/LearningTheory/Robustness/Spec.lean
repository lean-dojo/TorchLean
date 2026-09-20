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

open Spec TorchLean

namespace NN.MLTheory.Robustness.Spec

/-!
# Robustness specifications (polymorphic)

This file defines reusable vocabulary for specifying **robustness properties** of tensor-valued
functions.

All definitions are **scalar-polymorphic** in `α` via `[TorchLean.Storage α] [Context α]`, so the
same spec can be instantiated for:

- `ℝ` (paper-style theorems),
- `Float` (fast, executable consistency checks),
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

variable {α : Type} [TorchLean.Storage α] [Context α]

/-! ## Norms on spec tensors -/

/--
$L^\infty$ norm of a shape-indexed tensor.

If you flatten the tensor entries into a vector $t_i$, this is $\max_i |t_i|$.
-/
def tensorLinfNorm {s : Shape} (t : Tensor α s) : α :=
  match s with
  | .scalar => MathFunctions.abs t.item
  | .dim n _inner_s =>
    (List.finRange n).foldl
      (fun acc i => max acc (tensorLinfNorm (t.unstack i)))
      0

/--
$L^2$ (Euclidean) norm of a shape-indexed tensor.

If you flatten the tensor entries into a vector $t_i$, this is
$\sqrt{\sum_i t_i^2}$.
-/
def tensorL2Norm {s : Shape} (t : Tensor α s) : α :=
  MathFunctions.sqrt (tensorL2NormSquared t)
where
  tensorL2NormSquared {s : Shape} (t : Tensor α s) : α :=
    match s with
    | .scalar => t.item * t.item
    | .dim n _inner_s =>
      (List.finRange n).foldl
        (fun acc i => acc + tensorL2NormSquared (t.unstack i))
        0

/-! ## Distances and balls -/

/--
Distance induced by a tensor norm:

$$
\operatorname{dist}(t_1,t_2)=\lVert t_1-t_2\rVert.
$$
-/
def tensorDistance (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (t1 t2 : Tensor α s) : α :=
  norm (tensorSub t1 t2)
where
  tensorSub {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
    TorchLean.Tensor.subSpec t1 t2

/-- The compatibility subtraction helper delegates to the shared tensor subtraction. -/
@[simp] theorem tensor_distance_tensor_sub_eq_sub_spec {s : Shape} (t1 t2 : Tensor α s) :
    tensorDistance.tensorSub t1 t2 = TorchLean.Tensor.subSpec t1 t2 := by
  rfl

/--
Distance is the norm of the library difference: the form every downstream robustness proof uses.
-/
@[simp] theorem tensor_distance_eq_norm_sub_spec (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (t1 t2 : Tensor α s) :
    tensorDistance (α := α) norm t1 t2 = norm (TorchLean.Tensor.subSpec t1 t2) := by
  simp [tensorDistance]

/--
Closed $\varepsilon$-ball around `center` for the given norm:

$$
\{t\mid \operatorname{dist}(\mathrm{center},t)\leq\varepsilon\}.
$$
-/
def tensorBall (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (center : Tensor α s) (ε : α) : Set (Tensor α s) :=
  {t | tensorDistance norm center t ≤ ε}

/-! ## Continuity / robustness specifications -/

/--
Lipschitz continuity (global), phrased using `tensorDistance`.

If $f$ is $L$-Lipschitz and $d_1(x_0,x)\leq\varepsilon$, then
$d_2(f(x_0),f(x))\leq L\varepsilon$.
-/
def isLipschitzContinuous {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (L : α) : Prop :=
  ∀ x y : Tensor α s₁,
    tensorDistance norm₂ (f x) (f y) ≤ L * tensorDistance norm₁ x y

/--
Local Lipschitz continuity within the $\varepsilon$-ball around $x_0$.
-/
def IsLocallyLipschitz {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (x₀ : Tensor α s₁) (ε : α) (L : α) : Prop :=
  ∀ x y : Tensor α s₁,
    x ∈ tensorBall norm₁ x₀ ε → y ∈ tensorBall norm₁ x₀ ε →
    tensorDistance norm₂ (f x) (f y) ≤ L * tensorDistance norm₁ x y

/--
Adversarial robustness at a point $x_0$.

$f$ is $(\varepsilon,\delta)$-robust at $x_0$ if every input within distance $\varepsilon$ of
$x_0$ maps to an output within distance $\delta$ of $f(x_0)$.
-/
def IsAdversariallyRobust {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (x₀ : Tensor α s₁) (ε δ : α) : Prop :=
  ∀ x : Tensor α s₁,
    tensorDistance norm₁ x₀ x ≤ ε →
    tensorDistance norm₂ (f x₀) (f x) ≤ δ

/--
Certified robustness for a classifier at a nonnegative radius: the prediction is constant on the
$\varepsilon$-ball around $x_0$.

For neural networks, `classifier` is typically `argmax` on a logits tensor.
-/
def IsCertifiedRobust {s : Shape}
    (classifier : Tensor α s → Nat)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (x₀ : Tensor α s) (ε : α) : Prop :=
  0 ≤ ε ∧
    ∀ x : Tensor α s,
      x ∈ tensorBall norm x₀ ε →
      classifier x = classifier x₀

/-- Uniform adversarial robustness over a finite array of inputs. -/
def IsUniformlyRobust {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (dataset : Array (Tensor α s₁)) (ε δ : α) : Prop :=
  ∀ x₀ ∈ dataset, IsAdversariallyRobust f norm₁ norm₂ x₀ ε δ

/--
Contraction mapping under a norm: $f$ shrinks distances by a factor $c<1$.

This is a standard sufficient condition for convergence of iterated dynamics and robustness of
fixed points.
-/
def isContractive {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (contractionFactor : α) : Prop :=
  contractionFactor < 1 ∧
  isLipschitzContinuous f norm norm contractionFactor

/--
Sensitivity ratio for a specific additive perturbation.

This is the local “output change divided by input change” quantity:

$$
\frac{\lVert f(x)-f(x+\mathrm{perturbation})\rVert}
     {\lVert\mathrm{perturbation}\rVert}.
$$
-/
def sensitivity {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (x : Tensor α s₁) (perturbation : Tensor α s₁) : α :=
  let outputChange := tensorDistance norm₂ (f x) (f (TorchLean.Tensor.addSpec x perturbation))
  let inputChange := norm₁ perturbation
  outputChange / inputChange



end NN.MLTheory.Robustness.Spec
