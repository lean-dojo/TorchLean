/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Arithmetic

/-!
# Robustness definitions

Scalar-polymorphic definitions of norms/distances and basic robustness vocabulary on
TorchLean's shape-indexed tensors.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Robustness

/-!
# Robustness specifications (polymorphic)

The definitions here give a reusable vocabulary for **robustness properties** of tensor-valued
functions: the L∞ norm, distances and balls, Lipschitz, adversarial-robustness,
certified-robustness and contraction predicates, and a sensitivity ratio.

Every definition is **scalar-polymorphic** in `α` via `[TorchLean.Storage α] [Context α]`, so the
same spec can be instantiated for:

- `ℝ` (paper-style theorems),
- `Float` (fast, executable consistency checks),
- executable IEEE-754 backends (bit-level runtime semantics).

Apart from one `rfl` unfolding, the module holds definitions. Whether this norm and these distances
satisfy the usual metric laws depends on further algebraic and order assumptions on `α`, and those
theorems belong in dedicated proof developments.

At `Float`, the fold of `linfNorm` skips a NaN entry (`max 0 NaN` is `0`).
Executable `Float` code uses `Tensor.maxAbs` and `Tensor.maxAbsDiff` for the L∞ norm and distance,
which propagate NaN, and `TorchLean.Robustness` holds the scaled Euclidean norm
`l2NormFloat`.
Verified bounds and certificates are proved elsewhere, for example Lipschitz bounds in
`TorchLean.NN.Lipschitz.Network` and certified robustness procedures in
`CROWN`.

## References

- Adversarial examples and threat models: Szegedy et al. (2013/2014), Goodfellow, Shlens & Szegedy
  (2015, FGSM) and Madry et al. (2017).
- Certified robustness / verification: Wong & Kolter (2018) and Cohen, Rosenfeld & Kolter (2019).
- Lipschitz-based viewpoints (one entry point): Hein & Andriushchenko (2017).
-/

variable {α : Type} [Storage α] [Context α]

/-! ## Norms on spec tensors -/

/--
$L^\infty$ norm of a shape-indexed tensor.

On the flattened entries $t_i$ it is $\max_i |t_i|$, and zero for a tensor without entries.
-/
def linfNorm {s : Shape} (t : Tensor α s) : α :=
  match s with
  | .scalar => MathFunctions.abs t.item
  | .dim n _inner_s =>
    (List.finRange n).foldl
      (fun acc i => max acc (linfNorm (t.unstack i)))
      0

/-! ## Distances and balls -/

/--
Distance induced by a tensor norm:

$$
\operatorname{dist}(t_1,t_2)=\lVert t_1-t_2\rVert.
$$
-/
def distance (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (t1 t2 : Tensor α s) : α :=
  norm (t1 - t2)

/--
Distance is the norm of the library difference `t1 - t2`, the form downstream robustness proofs
rewrite to.
-/
@[simp] theorem distance_eq_norm_sub (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (t1 t2 : Tensor α s) :
    distance (α := α) norm t1 t2 = norm (t1 - t2) := by
  rfl

/--
Closed $\varepsilon$-ball around `center` for the given norm:

$$
\{t\mid \operatorname{dist}(\mathrm{center},t)\leq\varepsilon\}.
$$
-/
def ball (norm : ∀ {s : Shape}, Tensor α s → α) {s : Shape}
    (center : Tensor α s) (ε : α) : Set (Tensor α s) :=
  {t | distance norm center t ≤ ε}

/-! ## Continuity / robustness specifications -/

/--
Global Lipschitz continuity under `distance`: $d_2(f(x),f(y))\leq L\,d_1(x,y)$ for all $x$
and $y$.

For real scalars and $L\geq 0$, it follows that $d_1(x_0,x)\leq\varepsilon$ implies
$d_2(f(x_0),f(x))\leq L\varepsilon$.
-/
def IsLipschitzContinuous {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (L : α) : Prop :=
  ∀ x y : Tensor α s₁,
    distance norm₂ (f x) (f y) ≤ L * distance norm₁ x y

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
    distance norm₁ x₀ x ≤ ε →
    distance norm₂ (f x₀) (f x) ≤ δ

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
      x ∈ ball norm x₀ ε →
      classifier x = classifier x₀

/--
Contraction mapping under a norm: $f$ is `IsLipschitzContinuous` with a factor $c<1$, so it
shrinks distances.

Contractions are the standard sufficient condition for convergence of iterated dynamics and for
robustness of fixed points.
-/
def IsContractive {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (contractionFactor : α) : Prop :=
  contractionFactor < 1 ∧
  IsLipschitzContinuous f norm norm contractionFactor

/--
Sensitivity ratio for a specific additive perturbation, the local “output change divided by input
change” quantity:

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
  let outputChange := distance norm₂ (f x) (f (x + perturbation))
  let inputChange := norm₁ perturbation
  outputChange / inputChange

end TorchLean.Robustness
