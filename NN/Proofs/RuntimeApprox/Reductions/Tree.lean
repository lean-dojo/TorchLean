/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.GroupWithZero.Basic
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Reduction-tree error enclosure

A binary tree records the parenthesization of a rounded sum. Under the explicit local
`RelativeLocalAddBound` hypothesis, every such tree satisfies the same leaf-count and
absolute-leaf-scale error enclosure. The argument is independent of a floating-point format.

Binary32 does not satisfy the usual relative bound globally: gradual underflow needs an
absolute-error term. The executable specializations retain the local-bound hypothesis and
check finite intermediates. No unconditional unit-roundoff claim is made here.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

/-! ## Generic reduction trees over $\mathbb{R}$ -/

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

The numerical collection is an `Array`. The schedule relation below converts arrays to lists only
inside `List.Perm`, reusing mathlib's established permutation theory without exposing list-backed
numerical data in the API.
-/
def leaves : SumTree α → Array α
  | leaf x => #[x]
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

/-- A reduction tree has $\mathtt{leafCount}\ge 1$ (as a `Nat` inequality). -/
theorem leafCount_ge_one (t : SumTree α) : 1 ≤ t.leafCount :=
  Nat.succ_le_iff.mp (leafCount_pos t)

end SumTree

namespace ReductionBound

/-- Growth factor `(1+u)^(n-1)` for a reduction with `n` leaves. -/
noncomputable def growth (u : ℝ) (n : Nat) : ℝ :=
  (1 + u) ^ (n - 1)

/--
Monotonicity of the growth factor in the number of leaves.

When $u\ge 0$ (which is the only meaningful regime for an error parameter), longer reductions have
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

This is the standard inequality
$\left|\sum_i a_i\right|\le\sum_i|a_i|$, proved by induction on the tree shape.
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
Relative local rounded-addition assumption for reductions.

This is the usual normal-range unit-roundoff envelope:
$$
\operatorname{roundAdd}(a,b)=(a+b)+e,
\qquad
|e|\le u(|a|+|b|).
$$
Binary32 does **not** satisfy it globally with $u=2^{-24}$: subnormal results require an
absolute-error term. Consequently the executable
theorems below take this predicate as an explicit hypothesis; it must be established from
normal-range intermediate sums or replaced by an absolute/mixed analysis.
-/
def RelativeLocalAddBound (roundAdd : ℝ → ℝ → ℝ) (u : ℝ) : Prop :=
  ∀ a b : ℝ, _root_.abs (roundAdd a b - (a + b)) ≤ u * (_root_.abs a + _root_.abs b)

/--
Order-independent enclosure for any reduction tree evaluated with `roundAdd`.

Let $A=\sum_i|\mathtt{leaf}_i|$ be the sum of absolute values of leaves.
For $n$ leaves, the rounded evaluation is within
$(\operatorname{growth}(u,n)-1)A$ of the exact real sum.
-/
theorem evalRound_enclosure_of_relativeLocalAddBound
    (roundAdd : ℝ → ℝ → ℝ) (leafVal : α → ℝ) (u : ℝ)
    (H : RelativeLocalAddBound roundAdd u) (hu : 0 ≤ u) :
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
            simpa [n, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
              Nat.add_le_add_right hnA b.leafCount
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

end TorchLean.Floats.IEEE754
