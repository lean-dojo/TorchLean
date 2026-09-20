/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Basic.Real.Basic
import FloatLib.Numerics.Reduction.Error

/-!
# Reduction-tree error enclosure

A binary tree records the parenthesization of a rounded sum. Under the explicit local
`RelativeLocalAddBound` hypothesis, every such tree satisfies the same leaf-count and
absolute-leaf-scale error enclosure. The argument is independent of a floating-point format.
We retain the array-facing schedule API and use FloatLib's generic reduction-tree error bound.

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

end SumTree

namespace ReductionBound

/-- Growth factor `(1+u)^(n-1)` for a reduction with `n` leaves. -/
noncomputable def growth (u : ℝ) (n : Nat) : ℝ :=
  (1 + u) ^ (n - 1)

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

-- Keep the array-facing schedule API while sharing FloatLib's error analysis.
private def toReductionTree : SumTree α → FloatLib.Numerics.ReductionTree α
  | .leaf x => .leaf x
  | .node a b => .node (toReductionTree a) (toReductionTree b)

private theorem eval_toReductionTree (t : SumTree α)
    (combine : ℝ → ℝ → ℝ) (value : α → ℝ) :
    (toReductionTree t).eval combine value = evalRound combine value t := by
  induction t with
  | leaf x => rfl
  | node a b ha hb =>
    simp only [toReductionTree, FloatLib.Numerics.ReductionTree.eval, evalRound, ha, hb]

private theorem nodeCount_toReductionTree (t : SumTree α) :
    (toReductionTree t).nodeCount = t.leafCount - 1 := by
  induction t with
  | leaf x => rfl
  | node a b ha hb =>
    simp only [toReductionTree, FloatLib.Numerics.ReductionTree.nodeCount, SumTree.leafCount,
      ha, hb]
    have := a.leafCount_pos
    have := b.leafCount_pos
    omega

private theorem evalRound_add (value : α → ℝ) (t : SumTree α) :
    evalRound (· + ·) value t = exactSum value t := by
  induction t with
  | leaf x => rfl
  | node a b ha hb => simp only [evalRound, exactSum, ha, hb]

private theorem exactSum_abs (value : α → ℝ) (t : SumTree α) :
    exactSum (fun x => |value x|) t = sumAbs value t := by
  induction t with
  | leaf x => rfl
  | node a b ha hb => simp only [exactSum, sumAbs, ha, hb]

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
  have localBound (r : FloatLib.Numerics.ReductionTree α) :
      r.AllNodes roundAdd leafVal
        (fun a b => |roundAdd a b - (a + b)| ≤ u * (|a| + |b|)) := by
    induction r with
    | leaf x => trivial
    | node a b ha hb => exact ⟨ha, hb, H _ _⟩
  have bound := (toReductionTree t).abs_eval_sub_exact_le_geometric
    roundAdd leafVal id u hu (localBound _)
  simpa only [id_eq, Function.id_comp, FloatLib.Numerics.ReductionTree.sumAbs,
    eval_toReductionTree, nodeCount_toReductionTree, evalRound_add, exactSum_abs,
    growth] using bound

end TorchLean.Floats.IEEE754
