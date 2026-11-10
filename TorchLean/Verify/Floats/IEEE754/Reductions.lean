/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Floats.IEEE754
public import FloatLib.Numerics.Reduction.Error
import FloatLib.Floats.Formats.BinaryInterchange.Configured.Reduction.Tree

/-!
# Reductions

Deployment-aware reduction semantics for configured IEEE formats such as `ExecFloat.Binary 8 23`.

A deployed reduction (a sum, a dot product or a matmul accumulation) may use any valid
parenthesization and, depending on the implementation, any leaf order. Floating-point addition is
not associative, so different reduction schedules can produce different results.

We model a reduction as any result of a valid FloatLib `ReductionTree` schedule over the same
leaves. A standard forward-error enclosure holds uniformly over all such trees.

We stated these by hand over FloatLib's trees, which in any leaf order are the general class (3.1)
of Higham (1993): `n - 1` additions $T_k = T_{k_1} + T_{k_2}$ of two earlier terms, inputs or
partial sums. That class holds recursive summation in every order, pairwise summation, the
insertion method and the method that sums the positive and the negative terms apart. Kahan's
compensated summation, the last method the paper examines, falls outside it and outside our trees.

### What this file proves

For a schedule that stays finite, we bound the distance to the exact real sum by an enclosure
whose parameters depend only on:

- `n` (the number of leaves), and
- $A=\sum_i|\mathtt{leaf}_i|$ (a scale factor).

The bound uses the geometric factor $(1+u)^{n-1}-1$, not the rational $\gamma_k$ bound, with the
format's unit roundoff `u = Model.unitRoundoffAt format` (`2⁻²⁴` for binary32). A tree with $n$
leaves performs `t.nodeCount = n - 1` additions, so the statements write the factor as
`(1 + u) ^ t.nodeCount - 1`. It comes from FloatLib's
`Model.ReductionTree.abs_toReal_eval_sub_sum_le_geometric`, which bounds each addition on the
tree's own representable operands. The enclosure therefore covers subnormal sums under the
finite-execution hypothesis alone. These are mathematical enclosures, not native-kernel checks.

Higham (1993) proves his (2.6), $|\hat S_n - S_n| \le \gamma_{n-1}\sum_i|x_i|$ with
$\gamma_k = ku/(1-ku)$, for recursive summation in any order, from the standard model (1.2) with
no underflow, and the paper says every method it examines meets it. We chose the geometric factor
because it's never above $\gamma_{n-1}$ and needs no $(n-1)u < 1$.

The bridge lemma relates the executable `add` to the real model `Model.roundAt format (a + b)` only
on the *finite* branch. Where `Inf`, `NaN` or overflow can occur, the right semantics is FloatLib's
special-value semantics, and the `FiniteEval*` hypotheses keep those cases out of scope. In an IEEE
format a finite result already evaluates on the finite branch (`finiteEvalSumTree_of_isFinite`).

### Inspiration and related work

The enclosures formalize standard numerical-analysis results for parallel sums:

- David Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic”
  (1991). DOI: 10.1145/103162.103163
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002),
  especially Chapter 4 (Summation) and the $\gamma_k$-style bounds.
  DOI: 10.1137/1.9780898718027
- Nicholas J. Higham, “The Accuracy of Floating Point Summation”, SIAM J. Sci. Comput. 14(4),
  783-799 (1993). DOI: 10.1137/0914050
- Sylvie Boldo and Guillaume Melquiond, “Flocq: A Unified Library for Proving Floating-Point
  Algorithms in Coq” (ARITH 2011). DOI: 10.1109/ARITH.2011.40

A different design removes the nondeterminism with reproducible accumulators or binned sums. That
literature complements this file:

- James Demmel and Hong Diep Nguyen, “Fast Reproducible Floating-Point Summation” (ARITH 2013).
  DOI: 10.1109/ARITH.2013.9
- Willow Ahrens, James Demmel, and Hong Diep Nguyen, “Algorithms for Efficient Reproducible Floating
  Point Summation” (ACM TOMS 2020). DOI: 10.1145/3389360
- ReproBLAS (binned reproducible BLAS): https://bebop.cs.berkeley.edu/reproblas/
- ExBLAS (reproducible BLAS kernels): https://github.com/riakymch/exblas
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean.IEEE754

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange

/-- Relabelling the leaves keeps the number of additions. -/
private theorem nodeCount_map {α β : Type*} (t : ReductionTree α) (f : α → β) :
    (t.map f).nodeCount = t.nodeCount := by
  induction t with
  | leaf x => rfl
  | node a b ha hb => simp only [ReductionTree.map, ReductionTree.nodeCount, ha, hb]

/-! ## Configured formats: evaluation + nondeterminism (expression tree + permutation) -/

noncomputable section

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
Evaluate a reduction tree using the executable `ExecFloat.add`.

This is the “concrete” semantics: it includes IEEE special values, finite rounding, etc.
On a finite execution, `toReal_evalSum` identifies its
decoded value with the real-valued model that computes `Model.roundAt format (a + b)` at each node.
-/
def evalSum [ExecFloat.Add (Configured.Family format code plan)] (t : ReductionTree Value) :
    Value :=
  t.eval ExecFloat.add id

/--
`FiniteSum t` means:

- every leaf is finite, and
- every internal `add` evaluates on the finite branch (no `Inf`/`NaN` result).

Relating the executable semantics to the real model `Model.roundAt format (toReal a + toReal b)`
needs this hypothesis, because an `add` that overflows to `Inf` or produces a NaN falls under
FloatLib's special-value semantics, where no small-error enclosure holds.
-/
def FiniteSum [ExecFloat.Add (Configured.Family format code plan)] :
    ReductionTree Value → Prop
  | .leaf x => isFinite x = true
  | .node a b =>
      FiniteSum a ∧ FiniteSum b ∧
        isFinite (ExecFloat.add (evalSum a) (evalSum b)) = true

/--
If the whole reduction evaluates on the finite branch, then the final result is finite.
-/
theorem isFinite_evalSum
    [ExecFloat.Add (Configured.Family format code plan)] :
    ∀ t : ReductionTree Value, FiniteSum t → isFinite (evalSum t) = true
  | .leaf _, hx => hx
  | .node _ _, hx => hx.2.2

/-- In an IEEE format a sum schedule with a finite result evaluates on the finite branch: an
addition with a NaN or an infinite operand has no finite result. -/
theorem finiteEvalSumTree_of_isFinite [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add] (t : ReductionTree Value)
    (hresult : isFinite (evalSum t) = true) (hformat : format.isIEEE = true := by ieee_format) :
    FiniteSum t := by
  induction t with
  | leaf x => exact hresult
  | node a b ha hb =>
    have hmodel :
        Model.isFinite (Model.add (toModel (evalSum a)) (toModel (evalSum b))) = true := by
      rw [← ExecFloat.Binary.toModel_add (evalSum a) (evalSum b) .nearest]
      exact hresult
    obtain ⟨hfa, hfb⟩ := Model.isFinite_of_isFinite_add hformat hmodel
    exact ⟨ha hfa, hb hfb, hresult⟩

/--
Nondeterministic sum result relation.

`SumResult xs r` means: there exists a reduction tree `t` whose leaves are a permutation of `xs`
and such that evaluating `t` using executable `add` produces `r` *and* stays finite throughout.
-/
def SumResult [ExecFloat.Add (Configured.Family format code plan)]
    (xs : Array Value) (r : Value) : Prop :=
  ∃ t : ReductionTree Value, t.leaves.Perm xs.toList ∧ evalSum t = r ∧ FiniteSum t

/-- Exact real sum of the decoded leaves. -/
def exactSum (t : ReductionTree Value) : ℝ :=
  t.eval (· + ·) (Model.toReal ∘ toModel)

/-- Leaf scale for sums: `Σ |toReal leaf|`. -/
def sumAbs (t : ReductionTree Value) : ℝ :=
  t.sumAbs (Model.toReal ∘ toModel)

/-- Decoding commutes with executable evaluation. -/
private theorem toModel_evalSum [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add] (t : ReductionTree Value) :
    toModel (evalSum t) = (t.map toModel).eval Model.add id := by
  induction t with
  | leaf x => rfl
  | node a b ha hb =>
    change toModel (ExecFloat.add (evalSum a) (evalSum b)) =
      Model.add ((a.map toModel).eval Model.add id) ((b.map toModel).eval Model.add id)
    rw [← ha, ← hb]
    exact ExecFloat.Binary.toModel_add (evalSum a) (evalSum b) .nearest

/-- A finite executable sum is a finite execution of the decoded model tree. -/
private theorem finiteEval_map_toModel [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add] (t : ReductionTree Value)
    (finite : FiniteSum t) : Model.ReductionTree.FiniteEval (t.map toModel) := by
  induction t with
  | leaf x => exact finite
  | node a b ha hb =>
    refine ⟨ha finite.1, hb finite.2.1, ?_⟩
    change Model.isFinite (((ReductionTree.node a b).map toModel).eval Model.add id) = true
    rw [← toModel_evalSum (.node a b)]
    exact finite.2.2

/-- Under `FiniteSum`, the decoded executable sum equals the real-valued rounding model of
the decoded tree, which computes `Model.roundAt format (a + b)` at each node. -/
theorem toReal_evalSum
    [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add]
    (t : ReductionTree Value) (finite : FiniteSum t)
    (hformat : format.isIEEE = true := by ieee_format) :
    (toModel (evalSum t)).toReal =
      (t.map toModel).eval (fun a b => Model.roundAt format (a + b)) Model.toReal := by
  rw [toModel_evalSum]
  exact Model.ReductionTree.toReal_eval (t.map toModel) hformat (finiteEval_map_toModel t finite)

/-- A finite sum has the usual unit-roundoff enclosure of its format, including subnormal sums.

FloatLib bounds addition on this tree's own representable operands, so the result needs no
rounding assumption beyond finiteness.
-/
theorem evalSum_enclosure
    [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add] (t : ReductionTree Value)
    (finite : FiniteSum t) (hformat : format.isIEEE = true := by ieee_format) :
    |(toModel (evalSum t)).toReal - exactSum t| ≤
      ((1 + Model.unitRoundoffAt format) ^ t.nodeCount - 1) * sumAbs t := by
  have bound := Model.ReductionTree.abs_toReal_eval_sub_sum_le_geometric
    (t.map toModel) hformat (finiteEval_map_toModel t finite)
  rw [← toModel_evalSum t, ReductionTree.leaves_map, List.map_map,
    ← ReductionTree.eval_add_eq_sum, nodeCount_map, ReductionTree.sumAbs,
    ReductionTree.eval_map] at bound
  exact bound

/-- Every finite sum schedule has the tree-local unit-roundoff enclosure of its format. -/
theorem SumResult.enclosure
    [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add]
    (xs : Array Value) (r : Value) (result : SumResult xs r)
    (hformat : format.isIEEE = true := by ieee_format) :
    ∃ t : ReductionTree Value,
      t.leaves.Perm xs.toList ∧ evalSum t = r ∧
      |(toModel r).toReal - exactSum t| ≤
        ((1 + Model.unitRoundoffAt format) ^ t.nodeCount - 1) * sumAbs t := by
  obtain ⟨t, permutation, rfl, finite⟩ := result
  exact ⟨t, permutation, rfl, evalSum_enclosure t finite hformat⟩

/-!
## Dot-product accumulation (sum of products)

Leaves are operand pairs, evaluated by executable multiplication, and internal nodes use executable
addition. The enclosure bounds the accumulation error relative to the exact sum of the
already-rounded products, and a bound against the original real dot product also needs
multiplication error bounds. The model has no fused multiply-add and says nothing about which
schedule a native matmul, convolution or attention kernel uses.
-/

/--
Evaluate a dot-product reduction tree using executable `mul` at leaves and `add` at internal nodes.
-/
def evalDot [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)] (t : ReductionTree (Value × Value)) :
    Value :=
  t.eval ExecFloat.add fun p => ExecFloat.mul p.1 p.2

/--
`FiniteDot t` means:

- each leaf product `mul x y` is finite, and
- each internal accumulation `add` stays finite.

The predicate concerns the computed products and the accumulation results and puts no separate
finiteness condition on the operand pairs.
-/
def FiniteDot [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)] : ReductionTree (Value × Value) → Prop
  | .leaf (x, y) => isFinite (ExecFloat.mul x y) = true
  | .node a b =>
      FiniteDot a ∧ FiniteDot b ∧
        isFinite (ExecFloat.add (evalDot a) (evalDot b)) = true

/-- If a dot-product reduction stays finite at every step, then the final result is finite. -/
theorem isFinite_evalDot [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)] :
    ∀ t : ReductionTree (Value × Value), FiniteDot t → isFinite (evalDot t) = true
  | .leaf (_, _), hx => hx
  | .node _ _, hx => hx.2.2

/--
Nondeterministic dot-product result relation.

`DotResult xs r` means: there exists a reduction tree over leaf pairs in `xs` (up to
permutation) whose executable evaluation `evalDot` yields `r` and stays finite throughout.
-/
def DotResult [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)] (xs : Array (Value × Value))
    (r : Value) : Prop :=
  ∃ t : ReductionTree (Value × Value),
    t.leaves.Perm xs.toList ∧ evalDot t = r ∧ FiniteDot t

/--
Real-valued finite-branch model of `evalDot`.

- Leaves contribute `toReal (mul x y)` (the real meaning of the executable product).
- Internal nodes add those contributions using `Model.roundAt format (a+b)`.

Under `FiniteDot` it equals the decoded `evalDot t`, by
`toReal_evalDot`.
-/
def evalDotReal [ExecFloat.Mul (Configured.Family format code plan)]
    (t : ReductionTree (Value × Value)) : ℝ :=
  t.eval (fun a b => Model.roundAt format (a + b))
    (fun p => (toModel (ExecFloat.mul p.1 p.2)).toReal)

/--
Exact real sum of the rounded leaf products.

Each leaf contributes the product `toReal (mul x y)` that `evalDot` computes, after `mul`
rounding, and only the rounding of the accumulation adds is left out. The exact dot product of the
original inputs differs from this sum by the multiplication errors.
-/
def exactSumDot [ExecFloat.Mul (Configured.Family format code plan)]
    (t : ReductionTree (Value × Value)) : ℝ :=
  t.eval (· + ·) (fun p => (toModel (ExecFloat.mul p.1 p.2)).toReal)

/--
Leaf scale for dot-product accumulation: `Σ |toReal (mul x y)|`.

It is the “A” term of the forward-error bound for reductions.
-/
def sumAbsDot [ExecFloat.Mul (Configured.Family format code plan)]
    (t : ReductionTree (Value × Value)) : ℝ :=
  t.sumAbs (fun p => (toModel (ExecFloat.mul p.1 p.2)).toReal)

/--
Bridge lemma: on the finite branch, `toReal` of the executable dot-product reduction agrees with
the real-valued model `evalDotReal`.
-/
theorem toReal_evalDot
    [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add]
    [ExecFloat.Mul (Configured.Family format code plan)]
    (t : ReductionTree (Value × Value)) (finite : FiniteDot t)
    (hformat : format.isIEEE = true := by ieee_format) :
    (toModel (evalDot t)).toReal = evalDotReal t := by
  induction t with
  | leaf p => rfl
  | node a b ha hb =>
      change (toModel (ExecFloat.add (evalDot a) (evalDot b))).toReal =
        Model.roundAt format (evalDotReal a + evalDotReal b)
      have hround : (toModel (ExecFloat.add (evalDot a) (evalDot b))).toReal =
          Model.roundAt format
            ((toModel (evalDot a)).toReal + (toModel (evalDot b)).toReal) :=
        ExecFloat.Binary.toReal_add_eq_roundAt finite.2.2 hformat
      rw [hround, ha finite.1, hb finite.2.1]

/-- Finite products and finite accumulation make the tree of computed products a finite sum. -/
private theorem finiteSum_map_mul [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)] (t : ReductionTree (Value × Value))
    (finite : FiniteDot t) :
    FiniteSum (t.map fun p => ExecFloat.mul p.1 p.2) := by
  induction t with
  | leaf p => exact finite
  | node a b ha hb =>
    refine ⟨ha finite.1, hb finite.2.1, ?_⟩
    rw [evalSum, evalSum, ReductionTree.eval_map, ReductionTree.eval_map]
    exact finite.2.2

/-- Every finite dot-product schedule has the unit-roundoff enclosure of its format, which needs
only finite products and finite intermediate sums. The reference value `exactSumDot t` is the
exact sum of the already-rounded products. -/
theorem DotResult.enclosure
    [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add]
    [ExecFloat.Mul (Configured.Family format code plan)]
    (xs : Array (Value × Value)) (r : Value) (result : DotResult xs r)
    (hformat : format.isIEEE = true := by ieee_format) :
    ∃ t : ReductionTree (Value × Value),
      t.leaves.Perm xs.toList ∧ evalDot t = r ∧
      |(toModel r).toReal - exactSumDot t| ≤
        ((1 + Model.unitRoundoffAt format) ^ t.nodeCount - 1) * sumAbsDot t := by
  obtain ⟨t, permutation, rfl, finite⟩ := result
  refine ⟨t, permutation, rfl, ?_⟩
  have bound := evalSum_enclosure (t.map fun p => ExecFloat.mul p.1 p.2)
    (finiteSum_map_mul t finite) hformat
  rw [evalSum, exactSum, sumAbs, ReductionTree.sumAbs, ReductionTree.eval_map,
    ReductionTree.eval_map, ReductionTree.eval_map, nodeCount_map] at bound
  exact bound

end

end TorchLean.IEEE754
