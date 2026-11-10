/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Hylo

/-!
# Sample hylomorphisms

Recursive functions whose calls do not follow a value, each a `Hylo` built from branches with its
certificate proved here through `Hylo.correct_branches`. The Fibonacci numbers in a scalar format
make a call tree as large as the numbers themselves. The sum of a range of an input by halves is
pairwise summation (Higham 1993), whose rounding error grows with the depth of the call tree
rather than with its size. A failed read is the result of the function. The composite midpoint
rule for `x ^ 2` by bisection passes scalars to its calls: the left end and the width of the
interval of each call.
-/

@[expose] public section

namespace TorchLean.Examples.Tree

open Kernel

variable {α : Type}

/-- The Fibonacci numbers in arithmetic `A`, from `zero` and `one`. -/
def fib (A : Arithmetic α) (zero one : α) : Nat → α
  | 0 => zero
  | 1 => one
  | n + 2 => A.binary .add (fib A zero one (n + 1)) (fib A zero one n)

namespace Fib

/-- The argument `n` of a call. -/
def n : Expr α (Slots 1 0) .index :=
  .var (Slots.indexVar 0)

/-- The reference: `fib` of the argument. -/
def reference (A : Arithmetic α) (zero one : α) : Reader α → Env α (Slots 1 0) → Except Error α :=
  fun _ env => .ok (fib A zero one (env.get (Slots.indexVar 0)).toNat)

/-- Below `2` a call makes no calls and gives `zero` at `0` and `one` at `1`. -/
def small (zero one : α) : Expr α (Slots 1 0) .predicate × Hylo.Branch α 1 0 :=
  (.indexCompare .lt n (.index 2),
    ⟨[], .cond (.indexCompare .eq (.var (Row.indexVar 0)) (.index 0)) (.scalar zero)
      (.scalar one)⟩)

/-- From `2` on a call makes the calls at `n - 1` and `n - 2` and adds their results. -/
def large : Hylo.Branch α 1 0 :=
  ⟨[⟨fun _ => .indexBinary .sub n (.index 1), Fin.elim0⟩,
      ⟨fun _ => .indexBinary .sub n (.index 2), Fin.elim0⟩],
    .binary .add (.load 0 (.var .zero)) (.load 0 (.indexBinary .add (.var .zero) (.index 1)))⟩

/-- The certificate of the branches of `fib`. -/
theorem correct (A : Arithmetic α) (zero one : α) :
    Hylo.Branches.Correct 1 0 A 0 (reference A zero one) (fun _ _ => True) [small zero one]
      large := by
  refine ⟨rfl, ⟨by simp [UInt64.size], ?_⟩, ⟨by simp [large, UInt64.size], ?_⟩⟩
  · rintro read R index value first - ⟨-, h⟩ - -
    simp only [n, Expr.eval, Slots.get_env_indexVar, Compare.index, Bind.bind,
      Except.bind, Pure.pure, Except.pure, Except.ok.injEq, decide_eq_true_eq] at h
    have hk : (index 0).toNat < 2 := by simpa [UInt64.lt_iff_toNat_lt] using h
    simp only [reference, Expr.eval, Row.get_env_indexVar, Slots.get_env_indexVar,
      Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure]
    obtain h0 | h1 : (index 0).toNat = 0 ∨ (index 0).toNat = 1 := by omega
    · have : index 0 = 0 := UInt64.toNat_inj.1 (by simpa using h0)
      simp [this, fib]
    · have : index 0 = 1 := UInt64.toNat_inj.1 (by simpa using h1)
      simp [this, fib]
  · rintro read R index value first - ⟨-, h⟩ hfirst hcall
    simp only [n, Expr.eval, Slots.get_env_indexVar, Compare.index, Bind.bind,
      Except.bind, Pure.pure, Except.pure, Except.ok.injEq, decide_eq_false_iff_not,
      UInt64.not_lt] at h
    have hk : 2 ≤ (index 0).toNat := by simpa [UInt64.le_iff_toNat_le] using h
    obtain ⟨i0, v0, hi0, -, h0⟩ := hcall 0 (by simp [large])
    obtain ⟨i1, v1, hi1, -, h1⟩ := hcall 1 (by simp [large])
    have e0 := hi0 0
    have e1 := hi1 0
    simp only [large, n, List.getElem_cons_zero, List.getElem_cons_succ, Expr.eval,
      Slots.get_env_indexVar, Bind.bind, Except.bind, Pure.pure, Except.pure, IndexOp.eval,
      Except.ok.injEq] at e0 e1
    simp only [reference, Slots.get_env_indexVar, Nat.add_zero] at h0 h1
    simp only [large, reference, Expr.eval, Row.get_env_zero, Slots.get_env_indexVar, Bind.bind,
      Except.bind, Pure.pure, Except.pure, IndexOp.eval]
    rw [show first.toUInt64 + 1 = (first + 1).toUInt64 by simp [Nat.toUInt64_eq], h0, h1, ← e0,
      ← e1, UInt64.toNat_sub_of_le _ _ (by simpa [UInt64.le_iff_toNat_le] using (by omega)),
      UInt64.toNat_sub_of_le _ _ h]
    obtain ⟨k, hk'⟩ : ∃ k, (index 0).toNat = k + 2 := ⟨(index 0).toNat - 2, by omega⟩
    simp [hk', fib]

end Fib

/-- The hylomorphism of the Fibonacci numbers in the device arithmetic of `α`, from `zero` and
`one`: a call holds `n` in its index slot and its result is `fib n`. -/
def fibHylo [Scalar α] (zero one : α) : Hylo α 1 0 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 0
  expr := Hylo.stepOf [Fib.small zero one] Fib.large
  native := rfl
  reference := Fib.reference Scalar.arithmetic zero one
  calls := Hylo.callsOf [Fib.small zero one] Fib.large
  argIndex := Hylo.argIndexOf [Fib.small zero one] Fib.large
  argValue := Hylo.argValueOf (α := α) [Fib.small zero one] Fib.large
  correct := Hylo.correct_branches (Fib.correct Scalar.arithmetic zero one)

/-- The sum of entries `lo` to `hi - 1` of input `0` in arithmetic `A`, by halves: the sum of the
first half and then of the second, so the additions make a balanced tree. An empty range gives
`zero`, and a failed read is the result. -/
def pairwise (A : Arithmetic α) (zero : α) (read : Reader α) (lo hi : Nat) : Except Error α :=
  if hi ≤ lo then .ok zero
  else if hi - lo = 1 then read 0 lo.toUInt64
  else do
    let a ← pairwise A zero read lo (lo + (hi - lo) / 2)
    let b ← pairwise A zero read (lo + (hi - lo) / 2) hi
    return A.binary .add a b
termination_by hi - lo

namespace Pairwise

/-- The first entry `lo` of the range of a call. -/
def lo : Expr α (Slots 2 0) .index :=
  .var (Slots.indexVar 0)

/-- The end `hi` of the range of a call. -/
def hi : Expr α (Slots 2 0) .index :=
  .var (Slots.indexVar 1)

/-- The middle of the range of a call. -/
def mid : Expr α (Slots 2 0) .index :=
  .indexBinary .add lo (.indexBinary .div (.indexBinary .sub hi lo) (.index 2))

/-- The reference: `pairwise` over the range of the call. -/
def reference (A : Arithmetic α) (zero : α) : Reader α → Env α (Slots 2 0) → Except Error α :=
  fun read env => pairwise A zero read (env.get (Slots.indexVar 0)).toNat
    (env.get (Slots.indexVar 1)).toNat

/-- An empty range makes no calls and gives `zero`. -/
def empty (zero : α) : Expr α (Slots 2 0) .predicate × Hylo.Branch α 2 0 :=
  (.indexCompare .le hi lo, ⟨[], .scalar zero⟩)

/-- A range of one entry makes no calls and reads its entry. -/
def single : Expr α (Slots 2 0) .predicate × Hylo.Branch α 2 0 :=
  (.indexCompare .eq (.indexBinary .sub hi lo) (.index 1), ⟨[], .load 0 (.var (Row.indexVar 0))⟩)

/-- A longer range makes the calls on its two halves and adds their results. -/
def halves : Hylo.Branch α 2 0 :=
  ⟨[⟨fun i => if i = 0 then lo else mid, Fin.elim0⟩,
      ⟨fun i => if i = 0 then mid else hi, Fin.elim0⟩],
    .binary .add (.load 1 (.var .zero)) (.load 1 (.indexBinary .add (.var .zero) (.index 1)))⟩

/-- The certificate of the branches of `pairwise`. -/
theorem correct (A : Arithmetic α) (zero : α) :
    Hylo.Branches.Correct 2 0 A 1 (reference A zero) (fun _ _ => True) [empty zero, single]
      halves := by
  refine ⟨rfl, ⟨by simp [UInt64.size], ?_⟩, rfl, ⟨by simp [UInt64.size], ?_⟩,
    ⟨by simp [halves, UInt64.size], ?_⟩⟩
  · rintro read R index value first - ⟨-, h⟩ - -
    simp only [lo, hi, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one,
      Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq,
      decide_eq_true_eq] at h
    have hle : (index 1).toNat ≤ (index 0).toNat := by simpa [UInt64.le_iff_toNat_le] using h
    simp only [reference, Expr.eval, Slots.get_env_indexVar]
    rw [pairwise]
    simp [hle]
  · rintro read R index value first hR ⟨⟨-, h1⟩, h2⟩ - -
    simp only [lo, hi, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one,
      Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq,
      decide_eq_false_iff_not, UInt64.not_le, IndexOp.eval, beq_iff_eq] at h1 h2
    have hlt : (index 0).toNat < (index 1).toNat := by simpa [UInt64.lt_iff_toNat_lt] using h1
    have hone : (index 1).toNat - (index 0).toNat = 1 := by
      simpa [UInt64.toNat_sub_of_le _ _ (UInt64.le_of_lt h1)] using congrArg UInt64.toNat h2
    simp only [reference, Expr.eval, Row.get_env_indexVar, Slots.get_env_indexVar, Fin.val_zero,
      Fin.val_one, Bind.bind, Except.bind, hR 0 Nat.one_pos]
    rw [pairwise, ite_eq_right (by omega), ite_eq_left hone, Nat.toUInt64_eq, UInt64.ofNat_toNat]
  · rintro read R index value first hR ⟨⟨-, h1⟩, h2⟩ hfirst hcall
    simp only [lo, hi, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one,
      Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq,
      decide_eq_false_iff_not, UInt64.not_le, IndexOp.eval, beq_eq_false_iff_ne] at h1 h2
    have hlt : (index 0).toNat < (index 1).toNat := by simpa [UInt64.lt_iff_toNat_lt] using h1
    have hsub : (index 1 - index 0).toNat = (index 1).toNat - (index 0).toNat :=
      UInt64.toNat_sub_of_le _ _ (UInt64.le_of_lt h1)
    have hne : (index 1).toNat - (index 0).toNat ≠ 1 := fun h =>
      h2 (UInt64.toNat_inj.1 (hsub.trans (h.trans rfl)))
    have hbound := (index 1).toNat_lt
    have hmid : (index 0 + (index 1 - index 0) / 2).toNat =
        (index 0).toNat + ((index 1).toNat - (index 0).toNat) / 2 := by
      rw [UInt64.toNat_add, UInt64.toNat_div, hsub]
      simp only [UInt64.reduceToNat]
      omega
    obtain ⟨i0, v0, hi0, -, e0⟩ := hcall 0 (by simp [halves])
    obtain ⟨i1, v1, hi1, -, e1⟩ := hcall 1 (by simp [halves])
    have a0 := hi0 0
    have b0 := hi0 1
    have a1 := hi1 0
    have b1 := hi1 1
    simp only [halves, lo, hi, mid, List.getElem_cons_zero, List.getElem_cons_succ, Expr.eval,
      Slots.get_env_indexVar, Fin.val_zero, Fin.val_one, Bind.bind, Except.bind, Pure.pure,
      Except.pure, IndexOp.eval, Except.ok.injEq, Fin.isValue, one_ne_zero, ↓reduceIte]
      at a0 b0 a1 b1
    simp only [reference, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one, Nat.add_zero]
      at e0 e1
    simp only [halves, reference, Expr.eval, Row.get_env_zero, Slots.get_env_indexVar,
      Fin.val_zero, Fin.val_one, Bind.bind, Except.bind, Pure.pure, Except.pure, IndexOp.eval]
    rw [show first.toUInt64 + 1 = (first + 1).toUInt64 by simp [Nat.toUInt64_eq], e0, e1,
      ← a0, ← b0, ← a1, ← b1, hmid, pairwise.eq_1 A zero read (index 0).toNat (index 1).toNat,
      ite_eq_right (by omega), ite_eq_right hne]
    rfl

end Pairwise

/-- The hylomorphism of the sum by halves of a range of input `0`, in the device arithmetic of
`α`: a call holds the first entry and the end of its range in its index slots. -/
def pairwiseHylo [Scalar α] (zero : α) : Hylo α 2 0 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 1
  expr := Hylo.stepOf [Pairwise.empty zero, Pairwise.single] Pairwise.halves
  native := rfl
  reference := Pairwise.reference Scalar.arithmetic zero
  calls := Hylo.callsOf [Pairwise.empty zero, Pairwise.single] Pairwise.halves
  argIndex := Hylo.argIndexOf [Pairwise.empty zero, Pairwise.single] Pairwise.halves
  argValue := Hylo.argValueOf (α := α) [Pairwise.empty zero, Pairwise.single] Pairwise.halves
  correct := Hylo.correct_branches (Pairwise.correct Scalar.arithmetic zero)

/-- The composite midpoint rule for `x ^ 2` over the interval from `a` of width `w` with `2 ^ d`
panels, in arithmetic `A`, by bisection: a panel gives `w * (a + w / 2) ^ 2`, and the results of
the two halves of an interval are added. `two` is the scalar `2`. -/
def midpoint (A : Arithmetic α) (two : α) : Nat → α → α → α
  | 0, a, w =>
    let c := A.binary .add a (A.binary .div w two)
    A.binary .mul w (A.binary .mul c c)
  | d + 1, a, w =>
    let h := A.binary .div w two
    A.binary .add (midpoint A two d a h) (midpoint A two d (A.binary .add a h) h)

namespace Midpoint

/-- The depth `d` of a call. -/
def depth : Expr α (Slots 1 2) .index :=
  .var (Slots.indexVar 0)

/-- Half the width of the interval of a call. -/
def half (two : α) : Expr α (Slots 1 2) .scalar :=
  .binary .div (.var (Slots.scalarVar 1)) (.scalar two)

/-- The reference: `midpoint` at the depth and the interval of the call. -/
def reference (A : Arithmetic α) (two : α) : Reader α → Env α (Slots 1 2) → Except Error α :=
  fun _ env => .ok (midpoint A two (env.get (Slots.indexVar 0)).toNat
    (env.get (Slots.scalarVar 0)) (env.get (Slots.scalarVar 1)))

/-- At depth `0` a call makes no calls and gives its panel. -/
def panel (two : α) : Expr α (Slots 1 2) .predicate × Hylo.Branch α 1 2 :=
  (.indexCompare .eq depth (.index 0),
    ⟨[], .letIn (.binary .add (.var (Row.scalarVar 0))
        (.binary .div (.var (Row.scalarVar 1)) (.scalar two)))
      (.binary .mul (.var (.succ (Row.scalarVar 1))) (.binary .mul (.var .zero) (.var .zero)))⟩)

/-- Deeper, a call makes the calls on the two halves of its interval and adds their results. -/
def halves (two : α) : Hylo.Branch α 1 2 :=
  ⟨[⟨fun _ => .indexBinary .sub depth (.index 1),
        fun j => if j = 0 then .var (Slots.scalarVar 0) else half two⟩,
      ⟨fun _ => .indexBinary .sub depth (.index 1),
        fun j => if j = 0 then .binary .add (.var (Slots.scalarVar 0)) (half two) else half two⟩],
    .binary .add (.load 0 (.var .zero)) (.load 0 (.indexBinary .add (.var .zero) (.index 1)))⟩

/-- The certificate of the branches of `midpoint`. -/
theorem correct (A : Arithmetic α) (two : α) :
    Hylo.Branches.Correct 1 2 A 0 (reference A two) (fun _ _ => True) [panel two]
      (halves two) := by
  refine ⟨rfl, ⟨by simp [UInt64.size], ?_⟩, ⟨by simp [halves, UInt64.size], ?_⟩⟩
  · rintro read R index value first - ⟨-, h⟩ - -
    simp only [depth, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Compare.index,
      Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq, beq_iff_eq] at h
    simp only [reference, Expr.eval, Row.get_env_scalarVar, Slots.get_env_indexVar,
      Slots.get_env_scalarVar, Env.get_push_zero, Env.get_push_succ, Fin.val_zero, Fin.val_one,
      Bind.bind, Except.bind, Pure.pure, Except.pure, h]
    simp [midpoint]
  · rintro read R index value first - ⟨-, h⟩ hfirst hcall
    simp only [depth, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Compare.index,
      Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq, beq_eq_false_iff_ne] at h
    have hk : 1 ≤ (index 0).toNat :=
      Nat.pos_of_ne_zero fun h0 => h (UInt64.toNat_inj.1 (by simpa using h0))
    obtain ⟨i0, v0, hi0, hv0, e0⟩ := hcall 0 (by simp [halves])
    obtain ⟨i1, v1, hi1, hv1, e1⟩ := hcall 1 (by simp [halves])
    have a0 := hi0 0
    have b0 := hv0 0
    have c0 := hv0 1
    have a1 := hi1 0
    have b1 := hv1 0
    have c1 := hv1 1
    simp only [halves, depth, half, List.getElem_cons_zero, List.getElem_cons_succ, Expr.eval,
      Slots.get_env_indexVar, Slots.get_env_scalarVar, Fin.val_zero, Fin.val_one, Bind.bind,
      Except.bind, Pure.pure, Except.pure, IndexOp.eval, Except.ok.injEq, Fin.isValue,
      one_ne_zero, ↓reduceIte] at a0 b0 c0 a1 b1 c1
    simp only [reference, Slots.get_env_indexVar, Slots.get_env_scalarVar, Fin.val_zero,
      Fin.val_one, Nat.add_zero] at e0 e1
    simp only [halves, reference, Expr.eval, Row.get_env_zero, Slots.get_env_indexVar,
      Slots.get_env_scalarVar, Fin.val_zero, Fin.val_one, Bind.bind, Except.bind, Pure.pure,
      Except.pure, IndexOp.eval]
    rw [show first.toUInt64 + 1 = (first + 1).toUInt64 by simp [Nat.toUInt64_eq], e0, e1, ← a0,
      ← b0, ← c0, ← a1, ← b1, ← c1,
      UInt64.toNat_sub_of_le _ _ (by simpa [UInt64.le_iff_toNat_le] using hk)]
    obtain ⟨k, hk'⟩ : ∃ k, (index 0).toNat = k + 1 := ⟨(index 0).toNat - 1, by omega⟩
    simp [hk', midpoint]

end Midpoint

/-- The hylomorphism of the composite midpoint rule for `x ^ 2` by bisection, in the device
arithmetic of `α`: a call holds its depth in its index slot and the left end and the width of its
interval in its scalar slots. -/
def midpointHylo [Scalar α] (two : α) : Hylo α 1 2 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 0
  expr := Hylo.stepOf [Midpoint.panel two] (Midpoint.halves two)
  native := rfl
  reference := Midpoint.reference Scalar.arithmetic two
  calls := Hylo.callsOf [Midpoint.panel two] (Midpoint.halves two)
  argIndex := Hylo.argIndexOf [Midpoint.panel two] (Midpoint.halves two)
  argValue := Hylo.argValueOf [Midpoint.panel two] (Midpoint.halves two)
  correct := Hylo.correct_branches (Midpoint.correct Scalar.arithmetic two)

end TorchLean.Examples.Tree
