/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.Enclosure.Interval.Basic
public import Mathlib.Algebra.Order.Monoid.Unbundled.Basic

/-!
# Exact endpoint interval addition

Intervals are FloatLib's endpoint pairs `FloatLib.Numerics.Interval`, whose own sums round outward
(`FloatLib.Numerics.Interval.add?`). `intervalAdd` adds the endpoints with the carrier's own `+`,
which is exact in carriers such as `ℚ` and `ℝ`, and `intervalAdd_sound` shows that over an ordered
additive carrier the result contains every sum of members.

`intervalAdd` is the interval addition of Moore, Kearfott and Cloud (2009, Introduction to Interval
Analysis, SIAM, DOI 10.1137/1.9780898717716), and `add?` computes the same endpoints before it
rounds them.
-/

@[expose] public section

namespace TorchLean.Numeric

variable {α : Type*}

/-- Endpoint addition $[a,b]+[c,d]=[a+c,b+d]$ in the carrier's own arithmetic, with no outward
rounding. -/
def intervalAdd [Add α] (I J : FloatLib.Numerics.Interval α) : FloatLib.Numerics.Interval α where
  lo := I.lo + J.lo
  hi := I.hi + J.hi

/-- If $x\in[a,b]$ and $y\in[c,d]$, then $x+y\in[a+c,b+d]$, in any preorder whose addition is
monotone on both sides. -/
theorem intervalAdd_sound [Add α] [Preorder α] [AddLeftMono α] [AddRightMono α]
    {I J : FloatLib.Numerics.Interval α} {x y : α}
    (hx : I.lo ≤ x ∧ x ≤ I.hi) (hy : J.lo ≤ y ∧ y ≤ J.hi) :
    (intervalAdd I J).lo ≤ x + y ∧ x + y ≤ (intervalAdd I J).hi :=
  ⟨add_le_add hx.1 hy.1, add_le_add hx.2 hy.2⟩

end TorchLean.Numeric
