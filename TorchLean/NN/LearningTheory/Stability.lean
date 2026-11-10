/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Measure.FiniteMeasurePi
public import TorchLean.Tensor.Datatype

/-!
# Algorithmic stability (learning theory)

The definitions here are a small, reusable vocabulary for algorithmic stability: datasets as
tensors, replace-one and remove-one perturbations, empirical error, deterministic replace-one
uniform stability (`UniformStableReplace`) and IID sampling of datasets. The stability definition
follows the standard test-point-wise inequality from the literature, stated so that it connects
easily to concrete algorithms and bounds.

The definitions commit to no particular hypothesis-class structure. Stability serves classical ERM
analyses and modern training procedures alike, and the right ambient structure depends on the
application.

## Scope and design notes

### Datasets as tensors

A dataset of size `n` is a **length-`n` spec tensor** `Dataset n Z := TorchLean.Tensor Z [n]`.

- The learning-theory layer thereby shares TorchLean’s core shape-indexed tensor datatype
  (`TorchLean.Tensor.Datatype`), and the type enforces that a dataset has exactly `n`
  elements.
- The tensor keeps its entries in one row-major buffer, and `Dataset.toFn` and `Dataset.ofFn` give
  its `Fin n → Z` view. Stability needs only two things from a dataset: (a) access to coordinate
  `i : Fin n` and (b) the replace-one and remove-one perturbations.
- The theorems here are simp lemmas about datasets and the measurability of `Dataset.ofFn`. The
  example `TorchLeanExamples.DeepDives.Stability.RidgeRegression1D` proves a concrete bound with
  these definitions.

### Measures and expectations

For sampling, we assume `[MeasurableSpace Z]` and use mathlib's `ProbabilityMeasure`:
`iid μ n` is the product distribution on datasets `S : Dataset n Z`.

## References

Algorithmic stability is a standard toolbox for generalization bounds. Classic and modern
references include:

- Bousquet & Elisseeff (2002), “Stability and Generalization”.
- Shalev-Shwartz et al. (2010), “Learnability, Stability and Uniform Convergence”.
- Hardt, Recht & Singer (2016), “Train faster, generalize better: Stability of stochastic gradient
  descent”.
- For a textbook perspective relating stability, uniform convergence, and generalization, see
  Shalev-Shwartz & Ben-David (2014), “Understanding Machine Learning: From Theory to Algorithms”.
-/

@[expose] public section

noncomputable section

open scoped BigOperators

namespace TorchLean.LearningTheory

variable {Z H : Type}

/-! ## Datasets -/

/--
A dataset of size `n` with examples in `Z`.

The leading dimension records the sample count in the type.
-/
abbrev Dataset (n : Nat) (Z : Type) : Type :=
  Tensor Z [n]

namespace Dataset

variable {n : Nat} {Z : Type}

/--
View a dataset tensor as a function `Fin n → Z`.

`toFn` applies `TorchLean.Tensor.vectorEquiv`, and the function view is used to:

- define replace/remove operations via `Function.update` and `Fin.removeNth`, and
- transport the standard product measurable space / IID sampling measure to the tensor type.
-/
abbrev toFn (S : Dataset n Z) : Fin n → Z :=
  Tensor.vectorEquiv (α := Z) n S

/-- Build a dataset tensor from a function `Fin n → Z`. -/
abbrev ofFn (f : Fin n → Z) : Dataset n Z :=
  (Tensor.vectorEquiv (α := Z) n).symm f

/-- Reading back a dataset built from a function recovers the function. -/
@[simp] theorem toFn_ofFn (f : Fin n → Z) :
    toFn (n := n) (Z := Z) (ofFn (n := n) (Z := Z) f) = f := by
  simp [toFn, ofFn]

/-- The other round trip. Together with `toFn_ofFn` this is what lets stability arguments move
freely between the tensor representation of a sample and the function view the measure theory
prefers. -/
@[simp] theorem ofFn_toFn (S : Dataset n Z) :
    ofFn (n := n) (Z := Z) (toFn (n := n) (Z := Z) S) = S := by
  simp [toFn, ofFn]

/-- The function view of a dataset built from `f` agrees with `f` at every coordinate. -/
@[simp] theorem toFn_ofFn_apply (f : Fin n → Z) (i : Fin n) :
    toFn (n := n) (Z := Z) (ofFn (n := n) (Z := Z) f) i = f i :=
  congrFun ((Tensor.vectorEquiv n).apply_symm_apply f) i

section Measure

variable [MeasurableSpace Z]

/--
The measurable space on dataset tensors is the one transported from the standard product
measurable space on functions `Fin n → Z`.

IID sampling (`iid` below) and the standard stability definitions therefore work with their
measure-theoretic content unchanged, since the transport is only a representation choice.
-/
instance : MeasurableSpace (Dataset n Z) :=
  (inferInstance : MeasurableSpace (Fin n → Z)).comap (toFn (n := n) (Z := Z))

/-- `ofFn` is measurable, so the representation choice is invisible to the measure theory. -/
theorem measurable_ofFn : Measurable (ofFn (n := n) (Z := Z) : (Fin n → Z) → Dataset n Z) := by
  -- In the `comap` measurable space, a function into `Dataset n Z` is measurable iff composing with
  -- `toFn` is measurable.
  -- Here, `toFn ∘ ofFn = id`.
  apply
    (measurable_comap_iff (f := (ofFn (n := n) (Z := Z) : (Fin n → Z) → Dataset n Z))
      (g := (toFn (n := n) (Z := Z) : Dataset n Z → (Fin n → Z)))).2
  have hcomp :
      (toFn (n := n) (Z := Z) : Dataset n Z → (Fin n → Z)) ∘
          (ofFn (n := n) (Z := Z) : (Fin n → Z) → Dataset n Z) =
        id := by
    funext f
    exact toFn_ofFn f
  rw [hcomp]
  exact measurable_id

end Measure

end Dataset

/--
Replace the example at index `i` with `z'`.

Uniform stability definitions use this standard “replace-one” perturbation.
-/
def Dataset.replaceAt {n : Nat} (S : Dataset n Z) (i : Fin n) (z' : Z) : Dataset n Z :=
  Dataset.ofFn (n := n) (Z := Z) (Function.update (Dataset.toFn (n := n) (Z := Z) S) i z')

/-- Reading a replaced dataset returns the replacement at that coordinate and the original
example everywhere else. -/
@[simp] theorem Dataset.toFn_replaceAt {n : Nat} (S : Dataset n Z) (i j : Fin n) (z' : Z) :
    Dataset.toFn (replaceAt S i z') j = if j = i then z' else Dataset.toFn S j := by
  rw [replaceAt, Dataset.toFn_ofFn_apply]
  exact Function.update_apply _ _ _ _

/--
Remove the example at index `i` from a dataset of size `n+1`.

This is `Fin.removeNth i` on the function view, which reindexes the rest by `Fin.succAbove`.
-/
def Dataset.removeAt {n : Nat} (S : Dataset (n + 1) Z) (i : Fin (n + 1)) : Dataset n Z :=
  Dataset.ofFn (n := n) (Z := Z) (Fin.removeNth i (Dataset.toFn (n := n + 1) (Z := Z) S))

/-- Reading a shortened dataset skips the removed index, which is what `Fin.succAbove` encodes. -/
@[simp] theorem Dataset.toFn_removeAt {n : Nat} (S : Dataset (n + 1) Z) (i : Fin (n + 1))
    (j : Fin n) :
    Dataset.toFn (removeAt S i) j = Dataset.toFn S (i.succAbove j) := by
  exact Dataset.toFn_ofFn_apply _ _

/-! ## Learning algorithms and loss -/

/--
A deterministic learning algorithm mapping datasets to hypotheses.

Stating stability needs only this interface: an “algorithm” is a function `Dataset n Z → H`.
-/
abbrev LearningMap (n : Nat) (Z H : Type) : Type :=
  Dataset n Z → H

/--
A real-valued loss function.

We fix the codomain to `ℝ` to match the standard stability literature.
-/
abbrev Loss (H Z : Type) : Type :=
  H → Z → ℝ

/-! ## Errors -/

/--
Empirical error (average loss on a dataset).

The $1/n$ normalization is written out so downstream lemmas can control constants. At `n = 0` the
expression is zero, because Lean's real division gives $1/0=0$ and the sum is empty.
-/
def empiricalError {n : Nat} (ℓ : Loss H Z) (h : H) (S : Dataset n Z) : ℝ :=
  (1 / (n : ℝ)) * ∑ i : Fin n, ℓ h (Dataset.toFn (n := n) (Z := Z) S i)

/-! ## Deterministic replace-one stability -/

/--
Deterministic **replace-one uniform stability** (a common core notion).

`UniformStableReplace A ℓ β` says that replacing any one example of any training set changes the
loss at every test point by at most $\beta$.
-/
def UniformStableReplace {n : Nat} (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop :=
  ∀ (S : Dataset n Z) (i : Fin n) (z z' : Z),
    |ℓ (A S) z - ℓ (A (Dataset.replaceAt S i z')) z| ≤ β

/-! ## IID sampling helper -/

section Measure

variable [MeasurableSpace Z]

/--
IID sampling: product distribution on datasets.

If `μ` is a distribution over examples `Z`, then `iid μ n` is the distribution over datasets of size
`n` obtained by sampling each coordinate independently from `μ`.

The product distribution is naturally defined on functions `Fin n → Z`, and `Dataset.ofFn`
transports it to the shape-indexed tensor representation.
-/
def iid (μ : MeasureTheory.ProbabilityMeasure Z) (n : Nat) : MeasureTheory.ProbabilityMeasure
  (Dataset n Z) :=
  let ν : MeasureTheory.ProbabilityMeasure (Fin n → Z) :=
    MeasureTheory.ProbabilityMeasure.pi fun _ : Fin n => μ
  ν.map (Dataset.ofFn (n := n) (Z := Z))

end Measure

end TorchLean.LearningTheory
