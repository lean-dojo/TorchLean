/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Measure.FiniteMeasurePi
public import NN.Spec.Core.Tensor.Core

/-!
# Algorithmic stability (learning theory)

This file defines the vocabulary for algorithmic stability: datasets as tensors, replace-one and
remove-one perturbations, empirical error, deterministic replace-one uniform stability
(`UniformStableReplace`), and IID sampling of datasets.

The goal here is to provide a small, reusable vocabulary that downstream developments can reuse.
The definitions follow the standard event-wise / test-point-wise inequalities from the literature,
stated in a way that is easy to connect to concrete algorithms and bounds.

We keep the definitions general and avoid committing to a particular hypothesis class structure:
stability is useful both for classical ERM analyses and for modern training procedures, and the
right ambient structure depends on the application.

## Scope and design notes

### Datasets as tensors

We represent a dataset of size `n` as a **length-`n` spec tensor**

  `Dataset n Z := TorchLean.Tensor Z [n]`.

  This integrates the learning-theory layer with TorchLean’s core, shape-indexed tensor datatype
  (`NN.Spec.Core.Tensor.Core`) and keeps the “dataset has exactly `n` elements” invariant enforced
  by the type.
- Even though the underlying tensor representation is functional (`Fin n → ...`), we treat datasets
  abstractly: what matters for stability is that we can (a) access coordinate `i : Fin n` and (b)
  perform replace-one / remove-one perturbations.
- This file holds definitions only. Concrete bounds are proved in separate files, e.g.
  `NN.MLTheory.LearningTheory.Stability.RidgeRegression1D`.

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

namespace NN.MLTheory.LearningTheory.Stability

open Spec TorchLean

variable {Z H : Type}

/-! ## Datasets -/

/--
A dataset of size `n` with examples in `Z`.

The leading dimension records the sample count in the type.
-/
abbrev Dataset (n : Nat) (Z : Type) : Type :=
  TorchLean.Tensor Z [n]

namespace Dataset

variable {n : Nat} {Z : Type}

/--
View a dataset tensor as a function `Fin n → Z`.

This is definitional content via `TorchLean.Tensor.vectorEquiv`, and is used to:

- define replace/remove operations via `Function.update` and `Fin.succAbove`, and
- transport the standard product measurable space / IID sampling measure to the tensor type.
-/
abbrev toFn (S : Dataset n Z) : Fin n → Z :=
  (TorchLean.Tensor.vectorEquiv (α := Z) n).toFun S

/-- Build a dataset tensor from a function `Fin n → Z`. -/
abbrev ofFn (f : Fin n → Z) : Dataset n Z :=
  (TorchLean.Tensor.vectorEquiv (α := Z) n).invFun f

/-- Reading back a dataset built from a function recovers the function. -/
@[simp] theorem toFn_ofFn (f : Fin n → Z) : toFn (n := n) (Z := Z) (ofFn (n := n) (Z := Z) f) = f :=
  by
  simp [toFn, ofFn]

/-- The other round trip. Together with `toFn_ofFn` this is what lets stability arguments move
freely
between the tensor representation of a sample and the function view the measure theory prefers. -/
@[simp] theorem ofFn_toFn (S : Dataset n Z) : ofFn (n := n) (Z := Z) (toFn (n := n) (Z := Z) S) = S
  := by
  simp [toFn, ofFn]

/-- Coordinate access for dataset tensors. -/
abbrev get (S : Dataset n Z) (i : Fin n) : Z :=
  toFn (n := n) (Z := Z) S i

/-- Coordinate access on a dataset built from a function is just application. -/
@[simp] theorem get_ofFn (f : Fin n → Z) (i : Fin n) :
    get (n := n) (Z := Z) (ofFn (n := n) (Z := Z) f) i = f i := by
  change Tensor.vectorEquiv n ((Tensor.vectorEquiv n).symm f) i = f i
  exact congrFun ((Tensor.vectorEquiv n).apply_symm_apply f) i

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

/-- `toFn` is measurable by construction: the measurable space on datasets is its `comap`. -/
theorem measurable_toFn : Measurable (toFn (n := n) (Z := Z) : Dataset n Z → (Fin n → Z)) :=
  comap_measurable _

/-- `ofFn` is measurable too, so the representation choice is invisible to the measure theory. -/
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

This is the standard “replace-one” perturbation used in uniform stability definitions.
-/
def replaceAt {n : Nat} [DecidableEq (Fin n)] (S : Dataset n Z) (i : Fin n) (z' : Z) : Dataset n Z
  :=
  Dataset.ofFn (n := n) (Z := Z) (Function.update (Dataset.toFn (n := n) (Z := Z) S) i z')

/-- Reading a replaced dataset returns the replacement at that coordinate and the original
example everywhere else. -/
@[simp] theorem get_replaceAt {n : Nat} [DecidableEq (Fin n)]
    (S : Dataset n Z) (i j : Fin n) (z' : Z) :
    Dataset.get (replaceAt S i z') j = if j = i then z' else Dataset.get S j := by
  simp [replaceAt, Dataset.get, Function.update_apply]

/--
Remove the example at index `i` from a dataset of size `n+1`.

This uses `Fin.succAbove` to reindex the remaining elements into `Fin n`.
-/
def removeAt {n : Nat} (S : Dataset (n + 1) Z) (i : Fin (n + 1)) : Dataset n Z :=
  Dataset.ofFn (n := n) (Z := Z) (fun j => Dataset.get (n := n + 1) (Z := Z) S (i.succAbove j))

/-- Reading a shortened dataset skips the removed index, which is what `Fin.succAbove` encodes. -/
@[simp] theorem get_removeAt {n : Nat} (S : Dataset (n + 1) Z) (i : Fin (n + 1)) (j : Fin n) :
    Dataset.get (removeAt S i) j = Dataset.get S (i.succAbove j) := by
  simp [removeAt]

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

We fix the codomain to `ℝ` to match the standard stability literature.
-/
abbrev Loss (H Z : Type) : Type :=
  H → Z → ℝ

/-! ## Errors -/

/--
Empirical error (average loss on a dataset).

We write this with an explicit $1/n$ normalization so downstream lemmas can control constants.
At `n = 0`, the totalized real expression is zero.
-/
def empiricalError {n : Nat} [Fintype (Fin n)] (ℓ : Loss H Z) (h : H) (S : Dataset n Z) : ℝ :=
  (1 / (n : ℝ)) * ∑ i : Fin n, ℓ h (Dataset.get (n := n) (Z := Z) S i)

/-! ## Deterministic replace-one stability -/

/--
Deterministic **replace-one uniform stability** (a common core notion).

`UniformStableReplace A ℓ β` means that if you replace one example in the training set, then the
loss on *any* test point changes by at most $\beta$.
-/
def UniformStableReplace {n : Nat} [DecidableEq (Fin n)]
    (A : LearningMap n Z H) (ℓ : Loss H Z) (β : ℝ) : Prop :=
  ∀ (S : Dataset n Z) (i : Fin n) (z z' : Z),
    |ℓ (A S) z - ℓ (A (replaceAt S i z')) z| ≤ β

/-! ## IID sampling helper -/

section Measure

variable [MeasurableSpace Z]

/--
IID sampling: product distribution on datasets.

If `μ` is a distribution over examples `Z`, then `iid μ n` is the distribution over datasets of size
`n` obtained by sampling each coordinate independently from `μ`.

The product distribution is naturally defined on functions `Fin n → Z`; `Dataset.ofFn` transports
it to the shape-indexed tensor representation.
-/
def iid (μ : MeasureTheory.ProbabilityMeasure Z) (n : Nat) : MeasureTheory.ProbabilityMeasure
  (Dataset n Z) :=
  let ν : MeasureTheory.ProbabilityMeasure (Fin n → Z) :=
    MeasureTheory.ProbabilityMeasure.pi fun _ : Fin n => μ
  ν.map (Dataset.ofFn (n := n) (Z := Z))

end Measure

end NN.MLTheory.LearningTheory.Stability
/-!
The definitions above provide a shared vocabulary for downstream stability theorems.
-/
