/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.GroupWithZero.Unbundled.Basic
public import Mathlib.Data.List.Permutation
public import NN.Floats.IEEEExec.BridgeFP32Total
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Reductions

Deployment-aware reduction semantics for `IEEE32Exec`.

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
- `A = Σ |leaf_i|` (a scale factor).

This matches the standard "γ_k-style" summation bounds where order-dependence is absorbed by `A`
(sum of absolute values) and a growth factor in `n`.

For `IEEE32Exec`, we connect the executable `add` to the real model `fp32Round (a + b)`, but only on
the *finite* branch. If `Inf`/`NaN`/overflow is possible, the right semantics is the special-value
semantics from `SpecialRules.lean`, so this file keeps those cases out of scope via `FiniteEval*`.

### Inspiration and related work

This is a direct formalization of standard numerical-analysis results for parallel sums:

- David Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic”
  (1991). DOI: 10.1145/103162.103163
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM (2002),
  especially the summation chapter and the `γ_k`-style bounds.
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


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

/-! ## Generic reduction trees over `ℝ` -/

universe u

/--
A binary reduction schedule.

Leaves contain inputs of type `α`; internal nodes indicate “evaluate left and right subtrees, then
combine”. Different trees represent different valid parenthesizations for a parallel reduction.
-/
inductive SumTree (α : Type u) where
  | leaf : α → SumTree α
  | node : SumTree α → SumTree α → SumTree α
  deriving Repr

namespace SumTree

variable {α : Type u}

/--
The leaves of the reduction tree, read left-to-right.

We use `List.Perm` on `leaves` later to model “the same multiset of inputs, possibly reordered by
parallel scheduling”.
-/
def leaves : SumTree α → List α
  | leaf x => [x]
  | node a b => leaves a ++ leaves b

/-- Number of leaves in the tree (the “reduction length”). -/
def leafCount : SumTree α → Nat
  | leaf _ => 1
  | node a b => leafCount a + leafCount b

/-- A reduction tree always has at least one leaf. -/
theorem leafCount_pos (t : SumTree α) : 0 < t.leafCount := by
  induction t with
  | leaf => simp [leafCount]
  | node a b ihA ihB =>
      -- `0 < a + b` since `0 < a`.
      simpa [leafCount] using Nat.add_pos_left ihA (leafCount b)

/-- A reduction tree has `leafCount ≥ 1` (as a `Nat` inequality). -/
theorem leafCount_ge_one (t : SumTree α) : 1 ≤ t.leafCount :=
  Nat.succ_le_iff.mp (leafCount_pos t)

end SumTree

namespace ReductionBound

/-- Growth factor `(1+u)^(n-1)` for a reduction with `n` leaves. -/
noncomputable def growth (u : ℝ) (n : Nat) : ℝ :=
  (1 + u) ^ (n - 1)

/-- Base case: a “reduction” with one leaf has growth factor `1`. -/
theorem growth_one (u : ℝ) : growth u 1 = 1 := by simp [growth]

/--
Monotonicity of the growth factor in the number of leaves.

When `u ≥ 0` (which is the only meaningful regime for an error parameter), longer reductions have
larger or equal worst-case amplification.
-/
theorem growth_mono (u : ℝ) (hu : 0 ≤ u) : Monotone (fun n => growth u n) := by
  intro m n hmn
  dsimp [growth]
  have hbase : (1 : ℝ) ≤ 1 + u := by linarith
  exact pow_le_pow_right₀ hbase (Nat.sub_le_sub_right hmn 1)

end ReductionBound

open ReductionBound

variable {α : Type u}

/--
Evaluate a reduction tree using a given “rounded add” at internal nodes.

`evalRound roundAdd leafVal t` maps leaves via `leafVal`, and combines subresults using
`roundAdd`. This abstracts the idea of evaluating a parallel sum with a fixed local rounding model.
-/
def evalRound (roundAdd : ℝ → ℝ → ℝ) (leafVal : α → ℝ) : SumTree α → ℝ
  | .leaf x => leafVal x
  | .node a b => roundAdd (evalRound roundAdd leafVal a) (evalRound roundAdd leafVal b)

/-- Exact real evaluation of the tree (just `+` at internal nodes). -/
def exactSum (leafVal : α → ℝ) : SumTree α → ℝ
  | .leaf x => leafVal x
  | .node a b => exactSum leafVal a + exactSum leafVal b

/--
Sum of absolute values of leaf contributions.

This is the standard “scale” that appears in forward-error bounds for floating-point reductions.
-/
def sumAbs (leafVal : α → ℝ) : SumTree α → ℝ
  | .leaf x => _root_.abs (leafVal x)
  | .node a b => sumAbs leafVal a + sumAbs leafVal b

/-- `sumAbs leafVal t` is always nonnegative. -/
theorem sumAbs_nonneg (leafVal : α → ℝ) (t : SumTree α) : 0 ≤ sumAbs leafVal t := by
  induction t with
  | leaf x => simp [sumAbs]
  | node a b ihA ihB => simpa [sumAbs] using add_nonneg ihA ihB

/--
Triangle-inequality bound: the absolute value of the exact sum is at most the sum of absolute
values.

This is the standard inequality `|Σ a_i| ≤ Σ |a_i|` proved by induction on the tree shape.
-/
theorem abs_exactSum_le_sumAbs (leafVal : α → ℝ) (t : SumTree α) :
    _root_.abs (exactSum leafVal t) ≤ sumAbs leafVal t := by
  induction t with
  | leaf x => simp [exactSum, sumAbs]
  | node a b ihA ihB =>
      have h1 :
          _root_.abs (exactSum leafVal a + exactSum leafVal b) ≤
            _root_.abs (exactSum leafVal a) + _root_.abs (exactSum leafVal b) := by
        simpa using abs_add_le (exactSum leafVal a) (exactSum leafVal b)
      have h2 :
          _root_.abs (exactSum leafVal a) + _root_.abs (exactSum leafVal b) ≤
            sumAbs leafVal a + sumAbs leafVal b := by
        exact add_le_add ihA ihB
      simpa [exactSum, sumAbs] using h1.trans h2

/--
Local rounded-addition assumption for reductions.

This matches the usual “unit roundoff” envelope:
`roundAdd a b = (a+b) + e`, with `|e| ≤ u*(|a|+|b|)`.
-/
def LocalAddBound (roundAdd : ℝ → ℝ → ℝ) (u : ℝ) : Prop :=
  ∀ a b : ℝ, _root_.abs (roundAdd a b - (a + b)) ≤ u * (_root_.abs a + _root_.abs b)

/--
Order-independent enclosure for any reduction tree evaluated with `roundAdd`.

Let `A = Σ |leaf_i|` be the sum of absolute values of leaves.
For `n` leaves, the rounded evaluation is within `(growth u n - 1) * A` of the exact real sum.
-/
theorem evalRound_enclosure_of_LocalAddBound
    (roundAdd : ℝ → ℝ → ℝ) (leafVal : α → ℝ) (u : ℝ)
    (H : LocalAddBound roundAdd u) (hu : 0 ≤ u) :
    ∀ t : SumTree α,
      _root_.abs (evalRound roundAdd leafVal t - exactSum leafVal t) ≤
        (growth u t.leafCount - 1) * sumAbs leafVal t := by
  intro t
  induction t with
  | leaf x =>
      simp [evalRound, exactSum, sumAbs, SumTree.leafCount, growth]
  | node a b ihA ihB =>
      -- Proof strategy:
      --
      -- 1) Use the local bound `H` to control the fresh rounding error at the root node.
      -- 2) Use the inductive hypotheses to control the accumulated errors in each subtree.
      -- 3) Relate intermediate values to the leaf-scale `Aa`/`Ab` via triangle inequalities.
      -- 4) Algebraically combine coefficients, producing the clean factor `(1+u)^(n-1) - 1`.
      set ra := evalRound roundAdd leafVal a
      set rb := evalRound roundAdd leafVal b
      set Sa := exactSum leafVal a
      set Sb := exactSum leafVal b
      set Aa := sumAbs leafVal a
      set Ab := sumAbs leafVal b
      have hAa : 0 ≤ Aa := sumAbs_nonneg leafVal a
      have hAb : 0 ≤ Ab := sumAbs_nonneg leafVal b
      have hnA : 1 ≤ a.leafCount := SumTree.leafCount_ge_one a
      have hnB : 1 ≤ b.leafCount := SumTree.leafCount_ge_one b
      have hn2 : 2 ≤ (a.leafCount + b.leafCount) := by
        simpa using Nat.add_le_add hnA hnB
      set n : Nat := a.leafCount + b.leafCount
      have hn : (SumTree.leafCount (SumTree.node a b)) = n := by rfl

      have hmono : Monotone (fun k => growth u k) := growth_mono (u := u) (hu := hu)
      have hga : growth u a.leafCount ≤ growth u (n - 1) := by
        -- `a.leafCount ≤ n - 1` because `b.leafCount ≥ 1`.
        have : a.leafCount ≤ n - 1 := by
          have : a.leafCount + 1 ≤ n := by
            simpa [n] using Nat.add_le_add_left hnB a.leafCount
          exact Nat.le_pred_of_lt (Nat.lt_of_lt_of_le (Nat.lt_succ_self a.leafCount) this)
        exact hmono this
      have hgb : growth u b.leafCount ≤ growth u (n - 1) := by
        have : b.leafCount ≤ n - 1 := by
          have : b.leafCount + 1 ≤ n := by
            simpa [n] using Nat.add_le_add_right hnA b.leafCount
          exact Nat.le_pred_of_lt (Nat.lt_of_lt_of_le (Nat.lt_succ_self b.leafCount) this)
        exact hmono this

      have hEa : _root_.abs (ra - Sa) ≤ (growth u a.leafCount - 1) * Aa := by
        simpa [ra, Sa, Aa] using ihA
      have hEb : _root_.abs (rb - Sb) ≤ (growth u b.leafCount - 1) * Ab := by
        simpa [rb, Sb, Ab] using ihB

      have hra : _root_.abs ra ≤ growth u (n - 1) * Aa := by
        -- `|ra| ≤ |Sa| + |ra-Sa| ≤ Aa + (growth-1)Aa = growth*Aa`, then monotone growth.
        have hSa : _root_.abs Sa ≤ Aa := abs_exactSum_le_sumAbs leafVal a
        have htri : _root_.abs ra ≤ _root_.abs Sa + _root_.abs (ra - Sa) := by
          have h : Sa + (ra - Sa) = ra := by abel
          simpa [h] using (abs_add_le Sa (ra - Sa))
        have hab : _root_.abs Sa + _root_.abs (ra - Sa) ≤ Aa + ((growth u a.leafCount - 1) * Aa) :=
          add_le_add hSa hEa
        have hgA : (growth u a.leafCount) * Aa ≤ (growth u (n - 1)) * Aa :=
          mul_le_mul_of_nonneg_right hga hAa
        calc
          _root_.abs ra ≤ _root_.abs Sa + _root_.abs (ra - Sa) := htri
          _ ≤ Aa + ((growth u a.leafCount - 1) * Aa) := hab
          _ = (growth u a.leafCount) * Aa := by ring
          _ ≤ (growth u (n - 1)) * Aa := hgA
      have hrb : _root_.abs rb ≤ growth u (n - 1) * Ab := by
        have hSb : _root_.abs Sb ≤ Ab := abs_exactSum_le_sumAbs leafVal b
        have htri : _root_.abs rb ≤ _root_.abs Sb + _root_.abs (rb - Sb) := by
          have h : Sb + (rb - Sb) = rb := by abel
          simpa [h] using (abs_add_le Sb (rb - Sb))
        have hab : _root_.abs Sb + _root_.abs (rb - Sb) ≤ Ab + ((growth u b.leafCount - 1) * Ab) :=
          add_le_add hSb hEb
        have hgB : (growth u b.leafCount) * Ab ≤ (growth u (n - 1)) * Ab :=
          mul_le_mul_of_nonneg_right hgb hAb
        calc
          _root_.abs rb ≤ _root_.abs Sb + _root_.abs (rb - Sb) := htri
          _ ≤ Ab + ((growth u b.leafCount - 1) * Ab) := hab
          _ = (growth u b.leafCount) * Ab := by ring
          _ ≤ (growth u (n - 1)) * Ab := hgB

      have hround : _root_.abs (roundAdd ra rb - (ra + rb)) ≤ u * (_root_.abs ra + _root_.abs rb) :=
        H ra rb

      have hdecomp :
          roundAdd ra rb - (Sa + Sb) = (roundAdd ra rb - (ra + rb)) + ((ra - Sa) + (rb - Sb)) := by
        abel

      have hEab : _root_.abs (ra - Sa) + _root_.abs (rb - Sb) ≤ (growth u (n - 1) - 1) * (Aa + Ab)
        := by
        have hEa' : _root_.abs (ra - Sa) ≤ (growth u (n - 1) - 1) * Aa := by
          have : (growth u a.leafCount - 1) * Aa ≤ (growth u (n - 1) - 1) * Aa := by
            have : growth u a.leafCount - 1 ≤ growth u (n - 1) - 1 := by linarith [hga]
            exact mul_le_mul_of_nonneg_right this hAa
          exact hEa.trans this
        have hEb' : _root_.abs (rb - Sb) ≤ (growth u (n - 1) - 1) * Ab := by
          have : (growth u b.leafCount - 1) * Ab ≤ (growth u (n - 1) - 1) * Ab := by
            have : growth u b.leafCount - 1 ≤ growth u (n - 1) - 1 := by linarith [hgb]
            exact mul_le_mul_of_nonneg_right this hAb
          exact hEb.trans this
        have hab : _root_.abs (ra - Sa) + _root_.abs (rb - Sb) ≤
            (growth u (n - 1) - 1) * Aa + (growth u (n - 1) - 1) * Ab :=
          add_le_add hEa' hEb'
        simpa [mul_add] using hab.trans_eq (by ring : (growth u (n - 1) - 1) * Aa + (growth u (n -
          1) - 1) * Ab =
          (growth u (n - 1) - 1) * (Aa + Ab))

      have hR : _root_.abs ra + _root_.abs rb ≤ growth u (n - 1) * (Aa + Ab) := by
        have hab : _root_.abs ra + _root_.abs rb ≤ (growth u (n - 1) * Aa) + (growth u (n - 1) * Ab)
          :=
          add_le_add hra hrb
        simpa [mul_add] using hab.trans_eq (by ring : growth u (n - 1) * Aa + growth u (n - 1) * Ab
          =
          growth u (n - 1) * (Aa + Ab))

      have hcoef :
          u * growth u (n - 1) + (growth u (n - 1) - 1) = (growth u n - 1) := by
        -- `growth u n = (1+u)^(n-1) = (1+u)^(n-2) * (1+u) = growth u (n-1) * (1+u)`
        have hn_ge2 : 2 ≤ n := by simpa [n] using hn2
        have hn1 : n - 1 = (n - 2) + 1 := by
          obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_le hn_ge2
          -- Rewrite `n = 2 + d` and compute the truncated subtractions.
          rw [hd]
          simp [Nat.add_comm]
        -- turn `growth u n` into `growth u (n-1) * (1+u)`
        have : growth u n = growth u (n - 1) * (1 + u) := by
          -- both sides are `(1+u)^(n-2) * (1+u)` when `n ≥ 2`
          simp [growth, hn1, pow_succ]
        -- now finish by algebra
        calc
          u * growth u (n - 1) + (growth u (n - 1) - 1)
              = growth u (n - 1) * (1 + u) - 1 := by ring
          _ = growth u n - 1 := by simp [this]

      -- Finish.
      calc
        _root_.abs (evalRound roundAdd leafVal (SumTree.node a b) - exactSum leafVal (SumTree.node a
          b))
            = _root_.abs (roundAdd ra rb - (Sa + Sb)) := by
                simp [evalRound, exactSum, ra, rb, Sa, Sb]
        _ = _root_.abs ((roundAdd ra rb - (ra + rb)) + ((ra - Sa) + (rb - Sb))) := by
              simp [hdecomp]
        _ ≤ _root_.abs (roundAdd ra rb - (ra + rb)) + _root_.abs ((ra - Sa) + (rb - Sb)) := by
              simpa using abs_add_le (roundAdd ra rb - (ra + rb)) ((ra - Sa) + (rb - Sb))
        _ ≤ u * (_root_.abs ra + _root_.abs rb) + (_root_.abs (ra - Sa) + _root_.abs (rb - Sb)) :=
          by
              -- bound the rounding error and split the sub-error sum
              have habsErr : _root_.abs ((ra - Sa) + (rb - Sb)) ≤ _root_.abs (ra - Sa) + _root_.abs
                (rb - Sb) := by
                simpa using abs_add_le (ra - Sa) (rb - Sb)
              exact add_le_add hround habsErr
        _ ≤ (u * (growth u (n - 1) * (Aa + Ab))) + ((growth u (n - 1) - 1) * (Aa + Ab)) := by
              have hmul :
                  u * (_root_.abs ra + _root_.abs rb) ≤ u * (growth u (n - 1) * (Aa + Ab)) :=
                mul_le_mul_of_nonneg_left hR hu
              exact add_le_add hmul hEab
        _ = (growth u n - 1) * (Aa + Ab) := by
              -- apply the coefficient identity
              calc
                u * (growth u (n - 1) * (Aa + Ab)) + (growth u (n - 1) - 1) * (Aa + Ab)
                    = (u * growth u (n - 1) + (growth u (n - 1) - 1)) * (Aa + Ab) := by ring
                _ = (growth u n - 1) * (Aa + Ab) := by simp [hcoef]
        _ = (growth u (SumTree.leafCount (SumTree.node a b)) - 1) * sumAbs leafVal (SumTree.node a
          b) := by
              simp [SumTree.leafCount, sumAbs, n, Aa, Ab]

/-! ## `IEEE32Exec`: evaluation + nondeterminism (expression tree + permutation) -/

namespace IEEE32Exec

noncomputable section

/--
Evaluate a reduction tree using the executable `IEEE32Exec.add`.

This is the “concrete” semantics: it includes IEEE special values, finite rounding, etc.
For error analysis we will separately define a real-valued interpretation (`evalRealIEEE`).
-/
def evalIEEE : SumTree IEEE32Exec → IEEE32Exec
  | .leaf x => x
  | .node a b => add (evalIEEE a) (evalIEEE b)

/--
`FiniteEvalSumTree t` means:

- every leaf is finite, and
- every internal `add` evaluates on the finite branch (no `Inf`/`NaN` result).

We need this hypothesis when relating the executable semantics to the real model
`fp32Round (toReal a + toReal b)`: if an `add` can overflow to `Inf` or produce a NaN, the correct
semantic layer is the special-value one from `SpecialRules.lean`, not a small-error enclosure.
-/
def FiniteEvalSumTree : SumTree IEEE32Exec → Prop
  | .leaf x => isFinite x = true
  | .node a b =>
      FiniteEvalSumTree a ∧ FiniteEvalSumTree b ∧ isFinite (add (evalIEEE a) (evalIEEE b)) = true

/--
If the whole reduction evaluates on the finite branch, then the final result is finite.
-/
theorem isFinite_evalIEEE_of_FiniteEvalSumTree :
    ∀ t : SumTree IEEE32Exec, FiniteEvalSumTree t → isFinite (evalIEEE t) = true
  | .leaf x, hx => by simpa [evalIEEE] using hx
  | .node a b, hx => by simpa [evalIEEE] using hx.2.2

/--
Nondeterministic sum result relation.

`sumTreeResult xs r` means: there exists a reduction tree `t` whose leaves are a permutation of `xs`
and such that evaluating `t` using executable `add` produces `r` *and* stays finite throughout.
-/
def sumTreeResult (xs : List IEEE32Exec) (r : IEEE32Exec) : Prop :=
  ∃ t : SumTree IEEE32Exec, List.Perm t.leaves xs ∧ evalIEEE t = r ∧ FiniteEvalSumTree t

/--
Real-valued “finite-branch model” of `evalIEEE`.

Every internal node uses `fp32Round (a+b)`. This is the usual floating-point model
  (round-to-nearest)
*when the result stays finite*; our bridge lemmas justify this model under `FiniteEvalSumTree`.
-/
def evalRealIEEE (t : SumTree IEEE32Exec) : ℝ :=
  evalRound (fun a b => fp32Round (a + b)) toReal t

/-- Exact real sum of the decoded leaves. -/
def exactSumIEEE (t : SumTree IEEE32Exec) : ℝ :=
  exactSum toReal t

/-- Leaf scale for sums: `Σ |toReal leaf|`. -/
def sumAbsIEEE (t : SumTree IEEE32Exec) : ℝ :=
  sumAbs toReal t

/--
Under `FiniteEvalSumTree`, the decoded executable sum agrees with the real-valued rounding model.

Informal: as long as all intermediate `add`s stay finite, `IEEE32Exec.add` refines
`fp32Round (toReal a + toReal b)` at each node, so the whole tree refines `evalRealIEEE`.
-/
theorem toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree :
    ∀ t : SumTree IEEE32Exec, FiniteEvalSumTree t → toReal (evalIEEE t) = evalRealIEEE t
  | .leaf x, _ => by simp [evalIEEE, evalRealIEEE, evalRound]
  | .node a b, h => by
      have ha : FiniteEvalSumTree a := h.1
      have hb : FiniteEvalSumTree b := h.2.1
      have hfin : isFinite (add (evalIEEE a) (evalIEEE b)) = true := h.2.2
      have hrecA : toReal (evalIEEE a) = evalRealIEEE a := by
        simpa using (toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree a ha)
      have hrecB : toReal (evalIEEE b) = evalRealIEEE b := by
        simpa using (toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree b hb)
      have hadd :
          toReal (add (evalIEEE a) (evalIEEE b)) =
            fp32Round (toReal (evalIEEE a) + toReal (evalIEEE b)) :=
        toReal_add_eq_fp32Round_of_isFinite (x := evalIEEE a) (y := evalIEEE b) hfin
      calc
        toReal (evalIEEE (SumTree.node a b))
            = toReal (add (evalIEEE a) (evalIEEE b)) := by simp [evalIEEE]
        _ = fp32Round (toReal (evalIEEE a) + toReal (evalIEEE b)) := hadd
        _ = fp32Round (evalRealIEEE a + evalRealIEEE b) := by rw [hrecA, hrecB]
        _ = evalRealIEEE (SumTree.node a b) := by simp [evalRealIEEE, evalRound]

/--
Enclosure theorem for nondeterministic FP32 sums.

`sumTreeResult xs r` means: `r` is the result of adding the elements of `xs` using executable
`IEEE32Exec.add`, but with an evaluation order that is allowed to vary (a reduction tree plus a
permutation of leaves).

The conclusion produces a witness tree `t` and a bound on how far `toReal r` can deviate from the
exact real sum of `t`’s leaves (measured against the leaf scale `sumAbsIEEE t`).
-/
theorem sumTreeResult_enclosure
    (xs : List IEEE32Exec) (r : IEEE32Exec)
    (hres : sumTreeResult xs r)
    (u : ℝ)
    (H : LocalAddBound (fun a b => fp32Round (a + b)) u)
    (hu : 0 ≤ u) :
    ∃ t : SumTree IEEE32Exec,
      List.Perm t.leaves xs ∧ evalIEEE t = r ∧
      _root_.abs (toReal r - exactSumIEEE t) ≤ (growth u t.leafCount - 1) * sumAbsIEEE t := by
  rcases hres with ⟨t, hperm, hr, hfin⟩
  refine ⟨t, hperm, hr, ?_⟩
  have hto : toReal (evalIEEE t) = evalRealIEEE t :=
    toReal_evalIEEE_eq_evalRealIEEE_of_FiniteEvalSumTree t hfin
  have hE :
      _root_.abs (evalRealIEEE t - exactSumIEEE t) ≤ (growth u t.leafCount - 1) * sumAbsIEEE t := by
    simpa [evalRealIEEE, exactSumIEEE, sumAbsIEEE] using
      (evalRound_enclosure_of_LocalAddBound (roundAdd := fun a b => fp32Round (a + b))
        (leafVal := toReal) (u := u) H hu t)
  -- rewrite `evalRealIEEE t` as `toReal r` using `r = evalIEEE t`.
  have htr : toReal r = evalRealIEEE t := by simpa [hr] using hto
  -- avoid `simp` here: `simp` unfolds `toReal` and obscures rewriting.
  rw [htr]
  exact hE

/-!
## Dot-product accumulation (sum of products)

This section models the shape of computations like:

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
2) Some runtimes use fused multiply-add (FMA) for dot products. `IEEE32Exec` has an `fma`, but this
   particular model uses the more basic “mul then add” semantics.
-/

/-!
### Executable semantics

`evalDotIEEE` is the concrete reduction semantics: it returns an `IEEE32Exec` result and therefore
includes all IEEE special-value behavior and rounding.
-/

/--
Evaluate a dot-product reduction tree using executable `mul` at leaves and `add` at internal nodes.

This is the “concrete” semantics of a sum-of-products accumulation.
-/
def evalDotIEEE : SumTree (IEEE32Exec × IEEE32Exec) → IEEE32Exec
  | .leaf (x, y) => mul x y
  | .node a b => add (evalDotIEEE a) (evalDotIEEE b)

/--
`FiniteEvalDot t` means:

- each leaf product `mul x y` is finite, and
- each internal accumulation `add` stays finite.

This is the exact analogue of `FiniteEvalSumTree`, but for the sum-of-products setting. We need it
whenever we want to interpret an executable dot-product accumulation as “a real sum + small rounded
add errors”.
-/
def FiniteEvalDot : SumTree (IEEE32Exec × IEEE32Exec) → Prop
  | .leaf (x, y) => isFinite (mul x y) = true
  | .node a b =>
      FiniteEvalDot a ∧ FiniteEvalDot b ∧ isFinite (add (evalDotIEEE a) (evalDotIEEE b)) = true

/-- If a dot-product reduction stays finite at every step, then the final result is finite. -/
theorem isFinite_evalDotIEEE_of_FiniteEvalDot :
    ∀ t : SumTree (IEEE32Exec × IEEE32Exec), FiniteEvalDot t → isFinite (evalDotIEEE t) = true
  | .leaf _, hx => by simpa [evalDotIEEE] using hx
  | .node a b, hx => by simpa [evalDotIEEE] using hx.2.2

/--
Nondeterministic dot-product result relation.

`dotTreeResult xs r` means: there exists a reduction tree over leaf pairs in `xs` (up to
  permutation)
whose executable evaluation `evalDotIEEE` yields `r` and stays finite throughout.
-/
def dotTreeResult (xs : List (IEEE32Exec × IEEE32Exec)) (r : IEEE32Exec) : Prop :=
  ∃ t : SumTree (IEEE32Exec × IEEE32Exec),
    List.Perm t.leaves xs ∧ evalDotIEEE t = r ∧ FiniteEvalDot t

/--
Real-valued finite-branch model of `evalDotIEEE`.

- Leaves contribute `toReal (mul x y)` (the real meaning of the executable product).
- Internal nodes add those contributions using `fp32Round (a+b)`.

This matches the standard floating-point model for accumulation, under `FiniteEvalDot`.
-/
def evalRealDotIEEE (t : SumTree (IEEE32Exec × IEEE32Exec)) : ℝ :=
  evalRound (fun a b => fp32Round (a + b)) (fun p => toReal (mul p.1 p.2)) t

/--
Exact real sum of the rounded leaf products.

This is *not* the exact real dot product of the original inputs; it is the exact sum of the values
that `evalDotIEEE` would compute at the leaves (after `mul` rounding), ignoring only the rounding
introduced by the accumulation adds.
-/
def exactSumDotIEEE (t : SumTree (IEEE32Exec × IEEE32Exec)) : ℝ :=
  exactSum (fun p => toReal (mul p.1 p.2)) t

/--
Leaf scale for dot-product accumulation: `Σ |toReal (mul x y)|`.

This is the “A” term that appears in the forward-error bound for reductions.
-/
def sumAbsDotIEEE (t : SumTree (IEEE32Exec × IEEE32Exec)) : ℝ :=
  sumAbs (fun p => toReal (mul p.1 p.2)) t

/--
Bridge lemma: on the finite branch, `toReal` of the executable dot-product reduction agrees with
the real-valued model `evalRealDotIEEE`.
-/
theorem toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot :
    ∀ t : SumTree (IEEE32Exec × IEEE32Exec),
      FiniteEvalDot t → toReal (evalDotIEEE t) = evalRealDotIEEE t
  | .leaf (x, y), _ => by simp [evalDotIEEE, evalRealDotIEEE, evalRound]
  | .node a b, h => by
      have ha : FiniteEvalDot a := h.1
      have hb : FiniteEvalDot b := h.2.1
      have hfin : isFinite (add (evalDotIEEE a) (evalDotIEEE b)) = true := h.2.2
      have hrecA : toReal (evalDotIEEE a) = evalRealDotIEEE a :=
        toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot a ha
      have hrecB : toReal (evalDotIEEE b) = evalRealDotIEEE b :=
        toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot b hb
      have hadd :
          toReal (add (evalDotIEEE a) (evalDotIEEE b)) =
            fp32Round (toReal (evalDotIEEE a) + toReal (evalDotIEEE b)) :=
        toReal_add_eq_fp32Round_of_isFinite (x := evalDotIEEE a) (y := evalDotIEEE b) hfin
      calc
        toReal (evalDotIEEE (SumTree.node a b))
            = toReal (add (evalDotIEEE a) (evalDotIEEE b)) := by simp [evalDotIEEE]
        _ = fp32Round (toReal (evalDotIEEE a) + toReal (evalDotIEEE b)) := hadd
        _ = fp32Round (evalRealDotIEEE a + evalRealDotIEEE b) := by rw [hrecA, hrecB]
        _ = evalRealDotIEEE (SumTree.node a b) := by simp [evalRealDotIEEE, evalRound]

/--
Enclosure theorem for nondeterministic dot-product accumulation.

`dotTreeResult xs r` means: the runtime computed `r` by taking the list of pairs `xs`, multiplying
each pair, and then summing the products using *some* reduction tree whose leaves are a permutation
of `xs`.

The conclusion gives a witness tree `t` for that schedule and an order-independent enclosure:
the accumulation result is close (in `ℝ`) to the exact sum of leaf products, with the same growth
factor bound as in the plain-sum case.
-/
theorem dotTreeResult_enclosure
    (xs : List (IEEE32Exec × IEEE32Exec)) (r : IEEE32Exec)
    (hres : dotTreeResult xs r)
    (u : ℝ)
    (H : LocalAddBound (fun a b => fp32Round (a + b)) u)
    (hu : 0 ≤ u) :
    ∃ t : SumTree (IEEE32Exec × IEEE32Exec),
      List.Perm t.leaves xs ∧ evalDotIEEE t = r ∧
      _root_.abs (toReal r - exactSumDotIEEE t) ≤ (growth u t.leafCount - 1) * sumAbsDotIEEE t := by
  rcases hres with ⟨t, hperm, hr, hfin⟩
  refine ⟨t, hperm, hr, ?_⟩
  have hto : toReal (evalDotIEEE t) = evalRealDotIEEE t :=
    toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot t hfin
  have hE :
      _root_.abs (evalRealDotIEEE t - exactSumDotIEEE t) ≤ (growth u t.leafCount - 1) *
        sumAbsDotIEEE t := by
    simpa [evalRealDotIEEE, exactSumDotIEEE, sumAbsDotIEEE] using
      (evalRound_enclosure_of_LocalAddBound (roundAdd := fun a b => fp32Round (a + b))
        (leafVal := fun p => toReal (mul p.1 p.2)) (u := u) H hu t)
  have htr : toReal r = evalRealDotIEEE t := by simpa [hr] using hto
  rw [htr]
  exact hE

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
