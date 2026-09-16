/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.Finite
public import NN.Proofs.RuntimeApprox.Reductions.Tree

/-!
# Reductions

Deployment-aware reduction semantics for `ExecFloat.Binary 8 23`.

In deployed environments, reductions (sums / dot-products / matmul accumulations) may be evaluated
using different valid parenthesizations and, depending on the implementation, different leaf
orders. Since floating-point addition is not associative, distinct reduction schedules can produce
distinct results.

This module models a reduction as "any result produced by a valid reduction tree over the same
leaves" and proves a standard forward-error enclosure that holds uniformly over all such trees.

### What this file proves

Assuming a local rounded-addition model with parameter `u`, we derive a global enclosure whose
parameters depend only on:

- `n` (the number of leaves), and
- $A=\sum_i|\mathtt{leaf}_i|$ (a scale factor).

This matches the standard $\gamma_k$-style summation bounds where order-dependence is absorbed by
$A$
(sum of absolute values) and a growth factor in `n`.

For `ExecFloat.Binary 8 23`, we connect the executable `add` to the real model `fp32Round (a + b)`,
but only on
the *finite* branch. If `Inf`/`NaN`/overflow is possible, the right semantics is the special-value
semantics from FloatLib, so this file keeps those cases out of scope via `FiniteEval*`.

### Inspiration and related work

This is a direct formalization of standard numerical-analysis results for parallel sums:

- David Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic”
  (1991). DOI: 10.1145/103162.103163
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002),
  especially the summation chapter and the $\gamma_k$-style bounds.
- Sylvie Boldo and Guillaume Melquiond, “Flocq: A Unified Library for Proving Floating-Point
  Algorithms in Coq” (ARITH 2011). DOI: 10.1109/ARITH.2011.40

Instead of bounding nondeterminism, another design direction is to eliminate it via reproducible
accumulators/binned sums. That literature is a complement to this file:

- James Demmel and Hong Diep Nguyen, “Fast Reproducible Floating-Point Summation” (ARITH 2013).
  DOI: 10.1109/ARITH.2013.9
- Peter Ahrens, James Demmel, and Hong Diep Nguyen, “Algorithms for Efficient Reproducible Floating
  Point Summation” (ACM TOMS 2020). DOI: 10.1145/3389360
- ReproBLAS (binned reproducible BLAS): https://bebop.cs.berkeley.edu/reproblas/
- ExBLAS (reproducible BLAS kernels): https://github.com/riakymch/exblas
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats
open FloatLib.Numerics FloatLib.Floats.Formats.Flocq

open ReductionBound

/-! ## `ExecFloat.Binary 8 23`: evaluation + nondeterminism (expression tree + permutation) -/

namespace IEEE32Exec

noncomputable section

/--
Evaluate a reduction tree using the executable `ExecFloat.add`.

This is the “concrete” semantics: it includes IEEE special values, finite rounding, etc.
Error analysis uses a separate real-valued interpretation (`evalRealIEEE`).
-/
def evalIEEE : SumTree (ExecFloat.Binary 8 23) → (ExecFloat.Binary 8 23)
  | .leaf x => x
  | .node a b => ExecFloat.add (evalIEEE a) (evalIEEE b)

/--
`FiniteEvalSumTree t` means:

- every leaf is finite, and
- every internal `add` evaluates on the finite branch (no `Inf`/`NaN` result).

We need this hypothesis when relating the executable semantics to the real model
`fp32Round (toReal a + toReal b)`: if an `add` can overflow to `Inf` or produce a NaN, the correct
semantic layer is FloatLib's special-value semantics, not a small-error enclosure.
-/
def FiniteEvalSumTree : SumTree (ExecFloat.Binary 8 23) → Prop
  | .leaf x => isFinite x = true
  | .node a b =>
      FiniteEvalSumTree a ∧ FiniteEvalSumTree b ∧ isFinite (ExecFloat.add (evalIEEE a) (evalIEEE b))
        = true

/--
If the whole reduction evaluates on the finite branch, then the final result is finite.
-/
theorem isFinite_evalIEEE_of_FiniteEvalSumTree :
    ∀ t : SumTree (ExecFloat.Binary 8 23), FiniteEvalSumTree t → isFinite (evalIEEE t) = true
  | .leaf x, hx => by simpa [evalIEEE, FiniteEvalSumTree] using hx
  | .node a b, hx => by simpa [evalIEEE] using hx.2.2

/--
Nondeterministic sum result relation.

`sumTreeResult xs r` means: there exists a reduction tree `t` whose leaves are a permutation of `xs`
and such that evaluating `t` using executable `add` produces `r` *and* stays finite throughout.
-/
def sumTreeResult (xs : Array (ExecFloat.Binary 8 23)) (r : ExecFloat.Binary 8 23) : Prop :=
  ∃ t : SumTree (ExecFloat.Binary 8 23),
    List.Perm t.leaves.toList xs.toList ∧ evalIEEE t = r ∧ FiniteEvalSumTree t

/--
Real-valued “finite-branch model” of `evalIEEE`.

Every internal node uses `fp32Round (a+b)`. This is the usual floating-point model
  (round-to-nearest)
*when the result stays finite*; our bridge lemmas justify this model under `FiniteEvalSumTree`.
-/
def evalRealIEEE (t : SumTree (ExecFloat.Binary 8 23)) : ℝ :=
  evalRound (fun a b => fp32Round (a + b)) (fun x => (toModel x).toReal) t

/-- Every internal rounded sum has the canonical effective mantissa/exponent representation. -/
theorem evalRealIEEE_node_eq_computed (a b : SumTree (ExecFloat.Binary 8 23)) :
    evalRealIEEE (.node a b) =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 (evalRealIEEE a + evalRealIEEE b))
        exponent := cexp binaryRadix fexp32 (evalRealIEEE a + evalRealIEEE b) } := by
  change fp32Round (evalRealIEEE a + evalRealIEEE b) = _
  exact fp32Round_eq_computed (evalRealIEEE a + evalRealIEEE b)

/-- Exact real sum of the decoded leaves. -/
def exactSumIEEE (t : SumTree (ExecFloat.Binary 8 23)) : ℝ :=
  exactSum (Model.toReal ∘ toModel) t

/-- Leaf scale for sums: `Σ |toReal leaf|`. -/
def sumAbsIEEE (t : SumTree (ExecFloat.Binary 8 23)) : ℝ :=
  sumAbs (Model.toReal ∘ toModel) t

/--
Under `FiniteEvalSumTree`, the decoded executable sum agrees with the real-valued rounding model.

Informal: as long as all intermediate `add`s stay finite, `ExecFloat.add` refines
`fp32Round (toReal a + toReal b)` at each node, so the whole tree refines `evalRealIEEE`.
-/
theorem toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree :
    ∀ t : SumTree (ExecFloat.Binary 8 23), FiniteEvalSumTree t → (toModel (evalIEEE t)).toReal =
      evalRealIEEE t
  | .leaf x, _ => by simp [evalIEEE, evalRealIEEE, evalRound]
  | .node a b, h => by
      have ha : FiniteEvalSumTree a := h.1
      have hb : FiniteEvalSumTree b := h.2.1
      have hfin : isFinite (ExecFloat.add (evalIEEE a) (evalIEEE b)) = true := h.2.2
      have hrecA : (toModel (evalIEEE a)).toReal = evalRealIEEE a := by
        simpa using (toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree a ha)
      have hrecB : (toModel (evalIEEE b)).toReal = evalRealIEEE b := by
        simpa using (toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree b hb)
      have hadd :
          (toModel (ExecFloat.add (evalIEEE a) (evalIEEE b))).toReal =
            fp32Round ((toModel (evalIEEE a)).toReal + (toModel (evalIEEE b)).toReal) :=
        toReal_add_eq_fp32Round_of_isFinite (x := evalIEEE a) (y := evalIEEE b) hfin
      calc
        (Model.toReal ∘ toModel) (evalIEEE (SumTree.node a b))
            = (toModel (ExecFloat.add (evalIEEE a) (evalIEEE b))).toReal := by simp [evalIEEE]
        _ = fp32Round ((toModel (evalIEEE a)).toReal + (toModel (evalIEEE b)).toReal) := hadd
        _ = fp32Round (evalRealIEEE a + evalRealIEEE b) := by rw [hrecA, hrecB]
        _ = evalRealIEEE (SumTree.node a b) := by simp [evalRealIEEE, evalRound]

/--
Enclosure theorem for nondeterministic FP32 sums.

`sumTreeResult xs r` means: `r` is the result of adding the elements of `xs` using executable
`ExecFloat.add`, but with an evaluation order that is allowed to vary (a reduction tree plus a
permutation of leaves).

The conclusion produces a witness tree `t` and a bound on how far `toReal r` can deviate from the
exact real sum of `t`’s leaves (measured against the leaf scale `sumAbsIEEE t`).
-/
theorem sumTreeResult_enclosure
    (xs : Array (ExecFloat.Binary 8 23)) (r : ExecFloat.Binary 8 23)
    (hres : sumTreeResult xs r)
    (u : ℝ)
    (H : RelativeLocalAddBound (fun a b => fp32Round (a + b)) u)
    (hu : 0 ≤ u) :
    ∃ t : SumTree (ExecFloat.Binary 8 23),
      List.Perm t.leaves.toList xs.toList ∧ evalIEEE t = r ∧
      _root_.abs ((toModel r).toReal - exactSumIEEE t) ≤ (growth u t.leafCount - 1) * sumAbsIEEE t
        := by
  rcases hres with ⟨t, hperm, hr, hfin⟩
  refine ⟨t, hperm, hr, ?_⟩
  have hto : (toModel (evalIEEE t)).toReal = evalRealIEEE t :=
    toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree t hfin
  have hE :
      _root_.abs (evalRealIEEE t - exactSumIEEE t) ≤ (growth u t.leafCount - 1) * sumAbsIEEE t := by
    exact evalRound_enclosure_of_relativeLocalAddBound
      (α := ExecFloat.Binary 8 23) (roundAdd := fun a b => fp32Round (a + b))
      (leafVal := fun x : ExecFloat.Binary 8 23 => (toModel x).toReal) (u := u) H hu t
  -- rewrite `evalRealIEEE t` as `toReal r` using `r = evalIEEE t`.
  have htr : (toModel r).toReal = evalRealIEEE t := by simpa [hr] using hto
  -- avoid `simp` here: `simp` unfolds `toReal` and obscures rewriting.
  rw [htr]
  exact hE

/-!
## Dot-product accumulation (sum of products)

The definitions below model computations such as:

- dot products `Σᵢ xᵢ * yᵢ`, and
- the inner accumulations that show up in `matmul` / conv / attention scores.

On real hardware, the *sum* part is where a lot of nondeterminism creeps in:
threads compute partial sums and then reduce them with a tree-shaped schedule. Different valid
schedules correspond to different parenthesizations and different orders of the same leaf terms.

We model that explicitly:

- leaves are pairs `(x, y)`, interpreted as the executable product `mul x y`,
- internal nodes add those products using executable `add`.

Two important notes (to avoid over-claiming):

1) Our enclosure bounds the *accumulation* error (the rounded adds) relative to the real
   sum of the **already-rounded** products `toReal (mul x y)`. If you want a bound relative to the
   exact real dot product `Σᵢ (toReal xᵢ) * (toReal yᵢ)`, you also need a per-product error bound
     for
   `mul` (provided elsewhere).
2) Some runtimes use fused multiply-add (FMA) for dot products. `ExecFloat.Binary 8 23` has an
`fma`, but this
   particular model uses the more basic “mul then add” semantics.
-/

/-!
### Executable semantics

`evalDotIEEE` is the concrete reduction semantics: it returns an `ExecFloat.Binary 8 23` result and
therefore
includes all IEEE special-value behavior and rounding.
-/

/--
Evaluate a dot-product reduction tree using executable `mul` at leaves and `add` at internal nodes.

This is the “concrete” semantics of a sum-of-products accumulation.
-/
def evalDotIEEE : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23)) → (ExecFloat.Binary 8
  23)
  | .leaf (x, y) => ExecFloat.mul x y
  | .node a b => ExecFloat.add (evalDotIEEE a) (evalDotIEEE b)

/--
`FiniteEvalDot t` means:

- each leaf product `mul x y` is finite, and
- each internal accumulation `add` stays finite.

This is the exact analogue of `FiniteEvalSumTree`, but for the sum-of-products setting. We need it
whenever we want to interpret an executable dot-product accumulation as “a real sum + small rounded
add errors”.
-/
def FiniteEvalDot : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23)) → Prop
  | .leaf (x, y) => isFinite (ExecFloat.mul x y) = true
  | .node a b =>
      FiniteEvalDot a ∧ FiniteEvalDot b ∧ isFinite (ExecFloat.add (evalDotIEEE a) (evalDotIEEE b)) =
        true

/-- If a dot-product reduction stays finite at every step, then the final result is finite. -/
theorem isFinite_evalDotIEEE_of_FiniteEvalDot :
    ∀ t : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23)), FiniteEvalDot t → isFinite
      (evalDotIEEE t) = true
  | .leaf _, hx => by simpa [evalDotIEEE, FiniteEvalDot] using hx
  | .node a b, hx => by simpa [evalDotIEEE] using hx.2.2

/--
Nondeterministic dot-product result relation.

`dotTreeResult xs r` means: there exists a reduction tree over leaf pairs in `xs` (up to
  permutation)
whose executable evaluation `evalDotIEEE` yields `r` and stays finite throughout.
-/
def dotTreeResult (xs : Array ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23))) (r :
  ExecFloat.Binary 8 23) : Prop :=
  ∃ t : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23)),
    List.Perm t.leaves.toList xs.toList ∧ evalDotIEEE t = r ∧ FiniteEvalDot t

/--
Real-valued finite-branch model of `evalDotIEEE`.

- Leaves contribute `toReal (mul x y)` (the real meaning of the executable product).
- Internal nodes add those contributions using `fp32Round (a+b)`.

This matches the standard floating-point model for accumulation, under `FiniteEvalDot`.
-/
def evalRealDotIEEE (t : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23))) : ℝ :=
  evalRound (fun a b => fp32Round (a + b)) (fun p => (toModel (ExecFloat.mul p.1 p.2)).toReal) t

/-- A finite executable dot-product leaf has the effective rounded-product representation. -/
theorem evalRealDotIEEE_leaf_eq_computed (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.mul x y) = true) :
    evalRealDotIEEE (.leaf (x, y)) =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 ((toModel x).toReal * (toModel y).toReal))
        exponent := cexp binaryRadix fexp32 ((toModel x).toReal * (toModel y).toReal) } := by
  change (toModel (ExecFloat.mul x y)).toReal = _
  exact toReal_mul_eq_computed_of_isFinite x y hfin

/-- Every dot-product accumulation node has the effective rounded-sum representation. -/
theorem evalRealDotIEEE_node_eq_computed
    (a b : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23))) :
    evalRealDotIEEE (.node a b) =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32
            (evalRealDotIEEE a + evalRealDotIEEE b))
        exponent := cexp binaryRadix fexp32
          (evalRealDotIEEE a + evalRealDotIEEE b) } := by
  change fp32Round (evalRealDotIEEE a + evalRealDotIEEE b) = _
  exact fp32Round_eq_computed (evalRealDotIEEE a + evalRealDotIEEE b)

/--
Exact real sum of the rounded leaf products.

This is *not* the exact real dot product of the original inputs; it is the exact sum of the values
that `evalDotIEEE` would compute at the leaves (after `mul` rounding), ignoring only the rounding
introduced by the accumulation adds.
-/
def exactSumDotIEEE (t : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23))) : ℝ :=
  exactSum (fun p => (toModel (ExecFloat.mul p.1 p.2)).toReal) t

/--
Leaf scale for dot-product accumulation: `Σ |toReal (mul x y)|`.

This is the “A” term that appears in the forward-error bound for reductions.
-/
def sumAbsDotIEEE (t : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23))) : ℝ :=
  sumAbs (fun p => (toModel (ExecFloat.mul p.1 p.2)).toReal) t

/--
Bridge lemma: on the finite branch, `toReal` of the executable dot-product reduction agrees with
the real-valued model `evalRealDotIEEE`.
-/
theorem toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot :
    ∀ t : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23)),
      FiniteEvalDot t → (toModel (evalDotIEEE t)).toReal = evalRealDotIEEE t
  | .leaf (x, y), _ => by simp [evalDotIEEE, evalRealDotIEEE, evalRound]
  | .node a b, h => by
      have ha : FiniteEvalDot a := h.1
      have hb : FiniteEvalDot b := h.2.1
      have hfin : isFinite (ExecFloat.add (evalDotIEEE a) (evalDotIEEE b)) = true := h.2.2
      have hrecA : (toModel (evalDotIEEE a)).toReal = evalRealDotIEEE a :=
        toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot a ha
      have hrecB : (toModel (evalDotIEEE b)).toReal = evalRealDotIEEE b :=
        toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot b hb
      have hadd :
          (toModel (ExecFloat.add (evalDotIEEE a) (evalDotIEEE b))).toReal =
            fp32Round ((toModel (evalDotIEEE a)).toReal + (toModel (evalDotIEEE b)).toReal) :=
        toReal_add_eq_fp32Round_of_isFinite (x := evalDotIEEE a) (y := evalDotIEEE b) hfin
      calc
        (Model.toReal ∘ toModel) (evalDotIEEE (SumTree.node a b))
            = (toModel (ExecFloat.add (evalDotIEEE a) (evalDotIEEE b))).toReal := by simp
              [evalDotIEEE]
        _ = fp32Round ((toModel (evalDotIEEE a)).toReal + (toModel (evalDotIEEE b)).toReal) := hadd
        _ = fp32Round (evalRealDotIEEE a + evalRealDotIEEE b) := by rw [hrecA, hrecB]
        _ = evalRealDotIEEE (SumTree.node a b) := by simp [evalRealDotIEEE, evalRound]

/--
Enclosure theorem for nondeterministic dot-product accumulation.

`dotTreeResult xs r` means: the runtime computed `r` by taking the array of pairs `xs`, multiplying
each pair, and then summing the products using *some* reduction tree whose leaves are a permutation
of `xs`.

The conclusion gives a witness tree `t` for that schedule and an order-independent enclosure:
the accumulation result is close (in $\mathbb{R}$) to the exact sum of leaf products, with the same
growth factor bound as in the plain-sum case.
-/
theorem dotTreeResult_enclosure
    (xs : Array ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23))) (r : ExecFloat.Binary 8 23)
    (hres : dotTreeResult xs r)
    (u : ℝ)
    (H : RelativeLocalAddBound (fun a b => fp32Round (a + b)) u)
    (hu : 0 ≤ u) :
    ∃ t : SumTree ((ExecFloat.Binary 8 23) × (ExecFloat.Binary 8 23)),
      List.Perm t.leaves.toList xs.toList ∧ evalDotIEEE t = r ∧
      _root_.abs ((toModel r).toReal - exactSumDotIEEE t) ≤ (growth u t.leafCount - 1) *
        sumAbsDotIEEE t := by
  rcases hres with ⟨t, hperm, hr, hfin⟩
  refine ⟨t, hperm, hr, ?_⟩
  have hto : (toModel (evalDotIEEE t)).toReal = evalRealDotIEEE t :=
    toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot t hfin
  have hE :
      _root_.abs (evalRealDotIEEE t - exactSumDotIEEE t) ≤ (growth u t.leafCount - 1) *
        sumAbsDotIEEE t := by
    simpa [evalRealDotIEEE, exactSumDotIEEE, sumAbsDotIEEE] using
      (evalRound_enclosure_of_relativeLocalAddBound (roundAdd := fun a b => fp32Round (a + b))
        (leafVal := fun p => (toModel (ExecFloat.mul p.1 p.2)).toReal) (u := u) H hu t)
  have htr : (toModel r).toReal = evalRealDotIEEE t := by simpa [hr] using hto
  rw [htr]
  exact hE

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
