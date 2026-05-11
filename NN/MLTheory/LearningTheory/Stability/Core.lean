/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Mathlib.MeasureTheory.Measure.FiniteMeasurePi
public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
public import NN.Spec.Core.Tensor.Vec

/-!
# Algorithmic stability (learning theory)

This file defines the core *notions* of algorithmic stability that commonly appear in the
learning-theory literature:

- replace-one dataset perturbations,
- (expected / probabilistic) hypothesis and error stability, and
- uniform stability-style definitions that quantify over all test points.

The goal here is to provide a small, reusable vocabulary that downstream developments can reuse.
The definitions follow the standard event-wise / test-point-wise inequalities from the literature,
stated in a way that is easy to connect to concrete algorithms and bounds.

We keep the definitions general and avoid committing to a particular hypothesis class structure:
stability is useful both for classical ERM analyses and for modern training procedures, and the
right ambient structure depends on the application.

## Scope and design notes

### Datasets as tensors

We represent a dataset of size `n` as a **length-`n` spec tensor**

  `Dataset n Z := Spec.Tensor Z (.dim n .scalar)`.

  This integrates the learning-theory layer with TorchLean’s core, shape-indexed tensor datatype
  (`NN.Spec.Core.Tensor.Core`) and keeps the “dataset has exactly `n` elements” invariant enforced
  by the type.
- Even though the underlying tensor representation is functional (`Fin n → ...`), we treat datasets
  abstractly: what matters for stability is that we can (a) access coordinate `i : Fin n` and (b)
  perform replace-one / remove-one perturbations.
- The stability notions here are *definitions only* (plus a few helper constructions like IID
  sampling). Concrete bounds are proved in separate files, e.g.
  `NN.MLTheory.LearningTheory.Stability.RidgeRegression1D`.

### Measures and expectations

For the probabilistic notions, we assume `[MeasurableSpace Z]` and phrase expectations using
mathlib's `ProbabilityMeasure`. In particular:

- `(iid μ n)` is the product distribution on datasets (samples `S : Dataset n Z`),
- `∫ z, ... ∂μ` and `∫ S, ... ∂(iid μ n)` are Bochner integrals in `ℝ`.

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

namespace NN.MLTheory.LearningTheory.Stability

open Spec

variable {Z H : Type}

/-! ## Datasets -/

/--
A dataset of size `n` with examples in `Z`.

We model datasets as **vector tensors** `Spec.Tensor Z (.dim n .scalar)`.

This matches the rest of TorchLean’s codebase, where “a length-`n` vector” is represented as a
shape-indexed tensor.
-/
abbrev Dataset (n : Nat) (Z : Type) : Type :=
  Spec.Vec n Z

namespace Dataset

variable {n : Nat} {Z : Type}

/--
View a dataset tensor as a function `Fin n → Z`.

This is definitional content via `Spec.Tensor.dimScalarEquiv`, and is used to:

- define replace/remove operations via `Function.update` and `Fin.succAbove`, and
- transport the standard product measurable space / IID sampling measure to the tensor type.
-/
abbrev toFn (S : Dataset n Z) : Fin n → Z :=
  Spec.Vec.toFn (n := n) (α := Z) S

/-- Build a dataset tensor from a function `Fin n → Z`. -/
abbrev ofFn (f : Fin n → Z) : Dataset n Z :=
  Spec.Vec.ofFn (n := n) (α := Z) f

@[simp] theorem toFn_ofFn (f : Fin n → Z) : toFn (n := n) (Z := Z) (ofFn (n := n) (Z := Z) f) = f :=
  by
  simp [toFn, ofFn]

@[simp] theorem ofFn_toFn (S : Dataset n Z) : ofFn (n := n) (Z := Z) (toFn (n := n) (Z := Z) S) = S
  := by
  simp [toFn, ofFn]

/-- Coordinate access for dataset tensors. -/
abbrev get (S : Dataset n Z) (i : Fin n) : Z :=
  toFn (n := n) (Z := Z) S i

@[simp] theorem get_ofFn (f : Fin n → Z) (i : Fin n) :
    get (n := n) (Z := Z) (ofFn (n := n) (Z := Z) f) i = f i := by
  simp [get, toFn, ofFn]

section Measure

variable [MeasurableSpace Z]

/--
The measurable space on dataset tensors is the one transported from the standard product
measurable space on functions `Fin n → Z`.

This makes IID sampling (`iid` below) and the standard stability definitions work without changing
their measure-theoretic content; it is just a representation choice.
-/
instance : MeasurableSpace (Dataset n Z) :=
  (inferInstance : MeasurableSpace (Fin n → Z)).comap (toFn (n := n) (Z := Z))

theorem measurable_toFn : Measurable (toFn (n := n) (Z := Z) : Dataset n Z → (Fin n → Z)) :=
  comap_measurable _

theorem measurable_ofFn : Measurable (ofFn (n := n) (Z := Z) : (Fin n → Z) → Dataset n Z) := by
  -- In the `comap` measurable space, a function into `Dataset n Z` is measurable iff composing with
  -- `toFn` is measurable.
  -- Here, `toFn ∘ ofFn = id`.
  simpa [Function.comp, ofFn, toFn] using
    (measurable_comap_iff (f := (ofFn (n := n) (Z := Z) : (Fin n → Z) → Dataset n Z))
      (g := (toFn (n := n) (Z := Z) : Dataset n Z → (Fin n → Z)))).2
      (by simpa using (measurable_id : Measurable (fun x : Fin n → Z => x)))

end Measure

end Dataset

/--
Replace the example at index `i` with `z'`.

This is the standard “replace-one” perturbation used in uniform stability definitions.
-/
def replaceAt {n : Nat} [DecidableEq (Fin n)] (S : Dataset n Z) (i : Fin n) (z' : Z) : Dataset n Z
  :=
  Dataset.ofFn (n := n) (Z := Z) (Function.update (Dataset.toFn (n := n) (Z := Z) S) i z')

/--
Remove the example at index `i` from a dataset of size `n+1`.

This uses `Fin.succAbove` to reindex the remaining elements into `Fin n`.
-/
def removeAt {n : Nat} (S : Dataset (n + 1) Z) (i : Fin (n + 1)) : Dataset n Z :=
  Dataset.ofFn (n := n) (Z := Z) (fun j => Dataset.get (n := n + 1) (Z := Z) S (i.succAbove j))

/-! ## Learning algorithms and loss -/

/--
A deterministic learning algorithm mapping datasets to hypotheses.

This is the interface needed to state stability: an “algorithm” is just a function
`Dataset n Z → H`.
-/
abbrev LearningMap (n : Nat) (Z H : Type) : Type :=
  Dataset n Z → H

/--
A real-valued loss function.

We fix the codomain to `ℝ` to match the standard stability literature and to make integration
(`trueError`) straightforward.
-/
abbrev Loss (H Z : Type) : Type :=
  H → Z → ℝ

/-! ## Errors -/

/--
Empirical error (average loss on a dataset).

We write this with an explicit `(1 / n)` normalization so downstream lemmas can control constants.
-/
def empiricalError {n : Nat} [Fintype (Fin n)] (ℓ : Loss H Z) (h : H) (S : Dataset n Z) : ℝ :=
  (1 / (n : ℝ)) * ∑ i : Fin n, ℓ h (Dataset.get (n := n) (Z := Z) S i)

section Measure

variable [MeasurableSpace Z]

/--
True (population) error under a data distribution `μ`.

This is the expected loss `𝔼_{z∼μ}[ℓ(h,z)]`.
-/
def trueError (μ : MeasureTheory.ProbabilityMeasure Z) (ℓ : Loss H Z) (h : H) : ℝ :=
  ∫ z, ℓ h z ∂μ

/-! ## Deterministic replace-one stability -/

/--
Deterministic **replace-one uniform stability** (a common core notion).

`UniformStableReplace A ℓ β` means that if you replace one example in the training set, then the
loss on *any* test point changes by at most `β`.

This is the most “pointwise” notion in this file; the probabilistic notions below integrate or
take suprema in various ways.
-/
def UniformStableReplace {n : Nat} [DecidableEq (Fin n)]
    (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop :=
  ∀ (S : Dataset n Z) (i : Fin n) (z z' : Z),
    |ℓ (A S) z - ℓ (A (replaceAt S i z')) z| ≤ β

/-! ## IID sampling helper -/

/--
IID sampling: product distribution on datasets.

If `μ` is a distribution over examples `Z`, then `iid μ n` is the distribution over datasets of size
`n` obtained by sampling each coordinate independently from `μ`.

Implementation note: our dataset type is a tensor `Spec.Vec n Z`, but the product distribution is
most naturally defined on functions `Fin n → Z`. We transport it across the `Vec.ofFn` equivalence.
-/
def iid (μ : MeasureTheory.ProbabilityMeasure Z) (n : Nat) : MeasureTheory.ProbabilityMeasure
  (Dataset n Z) :=
  let ν : MeasureTheory.ProbabilityMeasure (Fin n → Z) :=
    MeasureTheory.ProbabilityMeasure.pi fun _ : Fin n => μ
  have h : Measurable (Dataset.ofFn (n := n) (Z := Z) : (Fin n → Z) → Dataset n Z) :=
    Dataset.measurable_ofFn (n := n) (Z := Z)
  ν.map (f := (Dataset.ofFn (n := n) (Z := Z))) h.aemeasurable

/-! ## Expected/probabilistic stability notions -/

/--
Expected (integrated) hypothesis stability.

This integrates the pointwise loss change over:

1. a random dataset `S ∼ iid μ n`,
2. a fresh replacement example `z' ∼ μ`, and
3. an independent test point `z ∼ μ`.

This corresponds to one of the standard “expected” stability notions in the literature.
-/
def HypothesisStability {n : Nat} [DecidableEq (Fin n)]
    (μ : MeasureTheory.ProbabilityMeasure Z) (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop
      :=
  ∀ i : Fin n,
    (∫ S, (∫ z', (∫ z, |ℓ (A S) z - ℓ (A (replaceAt S i z')) z| ∂μ) ∂μ) ∂(iid μ n)) ≤ β

/--
Pointwise hypothesis stability.

This is like `HypothesisStability`, but the “test point” is taken to be the `i`-th training example
itself (the coordinate being replaced).
-/
def PointwiseHypothesisStability {n : Nat} [DecidableEq (Fin n)]
    (μ : MeasureTheory.ProbabilityMeasure Z) (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop
      :=
  ∀ i : Fin n,
    (∫ S, (∫ z',
      |ℓ (A S) (Dataset.get (n := n) (Z := Z) S i) -
        ℓ (A (replaceAt S i z')) (Dataset.get (n := n) (Z := Z) S i)| ∂μ) ∂(iid μ n)) ≤ β

/--
Error stability (population error stability).

This measures how much the **true error** `trueError μ ℓ` changes under a replace-one perturbation.
-/
def ErrorStability {n : Nat} [DecidableEq (Fin n)]
    (μ : MeasureTheory.ProbabilityMeasure Z) (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop
      :=
  ∀ i : Fin n,
    (∫ S, (∫ z',
      |trueError μ ℓ (A S) - trueError μ ℓ (A (replaceAt S i z'))| ∂μ) ∂(iid μ n)) ≤ β

/--
Uniform stability (expected supremum over test points).

For each random dataset `S` and random replacement `z'`, we take the supremum over all test points
`z : Z` of the absolute loss change, then integrate. We make the usual boundedness side condition
explicit: every range whose supremum appears must be bounded above. This avoids relying on
`sSup` outside its mathematically meaningful domain.
-/
def uniformStabilityRange {n : Nat} [DecidableEq (Fin n)]
    (A : LearningMap n Z H) (ℓ : Loss H Z)
    (i : Fin n) (S : Dataset n Z) (z' : Z) : Set ℝ :=
  Set.range fun z : Z => |ℓ (A S) z - ℓ (A (replaceAt S i z')) z|

/--
Boundedness side condition for uniform-stability suprema.

The standard literature often assumes bounded losses up front. TorchLean keeps this as an explicit
predicate so downstream theorems can either prove it from a bounded-loss hypothesis or carry it as a
transparent assumption.
-/
def UniformStabilityRangeBdd {n : Nat} [DecidableEq (Fin n)]
    (A : LearningMap n Z H) (ℓ : Loss H Z) : Prop :=
  ∀ i : Fin n, ∀ S : Dataset n Z, ∀ z' : Z,
    BddAbove (uniformStabilityRange (n := n) (Z := Z) (H := H) A ℓ i S z')

/-- Supremum term used in `UniformStability` once boundedness is available. -/
def uniformStabilitySup {n : Nat} [DecidableEq (Fin n)]
    (A : LearningMap n Z H) (ℓ : Loss H Z)
    (i : Fin n) (S : Dataset n Z) (z' : Z) : ℝ :=
  sSup (uniformStabilityRange (n := n) (Z := Z) (H := H) A ℓ i S z')

/--
Uniform stability (expected supremum over test points).

For each random dataset `S` and random replacement `z'`, we take the supremum over all test points
`z : Z` of the absolute loss change, then integrate. The first conjunct records the boundedness
needed for those suprema to be mathematically disciplined.
-/
def UniformStability {n : Nat} [DecidableEq (Fin n)]
    (μ : MeasureTheory.ProbabilityMeasure Z) (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop
      :=
  UniformStabilityRangeBdd (n := n) (Z := Z) (H := H) A ℓ ∧
  ∀ i : Fin n,
    (∫ S, (∫ z',
      uniformStabilitySup (n := n) (Z := Z) (H := H) A ℓ i S z' ∂μ) ∂(iid μ n)) ≤ β

/--
Probabilistic uniform stability.

This is a “high probability” analogue of `UniformStability`: with probability at least `1-δ` over
datasets `S`, the (integrated) uniform stability quantity is at most `β`. As above, boundedness of
the pointwise ranges is part of the definition rather than an implicit side condition.
-/
def ProbUniformStability {n : Nat} [DecidableEq (Fin n)]
    (μ : MeasureTheory.ProbabilityMeasure Z) (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) (δ :
      ENNReal) : Prop :=
  UniformStabilityRangeBdd (n := n) (Z := Z) (H := H) A ℓ ∧
  ∀ i : Fin n,
    (MeasureTheory.ProbabilityMeasure.toMeasure (iid μ n) {S |
      (∫ z', uniformStabilitySup (n := n) (Z := Z) (H := H) A ℓ i S z' ∂μ) ≤ β}) ≥
        ((1 : ENNReal) - δ)

/-! ## Leave-one-out (CV-LOO) style quantities -/

/--
Leave-one-out (LOO) estimate, phrased using `removeAt`.

For each index `i`, train on the dataset with the `i`-th example removed, and evaluate loss on the
held-out example. Then average over `i`.
-/
def looEstimate {n : Nat} [DecidableEq (Fin (n + 1))] [Fintype (Fin (n + 1))] [Fintype (Fin n)]
    (A : LearningMap n Z H) (ℓ : Loss H Z) (S : Dataset (n + 1) Z) : ℝ :=
  (1 / ((n + 1 : Nat) : ℝ)) * ∑ i : Fin (n + 1),
    ℓ (A (removeAt S i)) (Dataset.get (n := n + 1) (Z := Z) S i)

/--
Cross-validation leave-one-out stability.

This measures how much the LOO estimate changes when one example is replaced.
-/
def CVlooStability {n : Nat} [DecidableEq (Fin (n + 1))] [Fintype (Fin (n + 1))] [Fintype (Fin n)]
    (μ : MeasureTheory.ProbabilityMeasure Z) (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop
      :=
  ∀ i : Fin (n + 1),
    (∫ S, (∫ z',
      |looEstimate A ℓ S - looEstimate A ℓ (replaceAt S i z')| ∂μ) ∂(iid μ (n + 1))) ≤ β

/--
Expected LOO vs true error stability.

This compares the LOO estimate on a dataset to the true error of (one particular) leave-one-out
trained hypothesis.
-/
def ElooErrStability {n : Nat} [DecidableEq (Fin (n + 1))] [Fintype (Fin (n + 1))] [Fintype (Fin n)]
    (μ : MeasureTheory.ProbabilityMeasure Z) (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop
      :=
  (∫ S, |looEstimate A ℓ S - trueError μ ℓ (A (removeAt S 0))| ∂(iid μ (n + 1))) ≤ β

end Measure

end NN.MLTheory.LearningTheory.Stability
/-!
The remaining definitions in this file are various stability notions appearing in the literature.
We keep them in a single place so downstream theorems can reference a shared vocabulary.
-/
