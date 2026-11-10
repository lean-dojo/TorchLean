/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Unfold
public import TorchLeanExamples.Tree.Types

/-!
# Sample unfolds

Unfolds into the sample tree types, each an `Unfold` with its certificate `Unfold.Correct` proved
here. The balanced binary tree of a range of an input has the middle entry of the range at its
root and the trees of the two halves around it, so a sorted input gives a balanced search tree.
Its nodes read the input, and a range past the end of the input fails with the first missing
entry. The binomial tree of order `k` (Vuillemin 1978) is a rose tree of `2 ^ k` nodes whose root
has the binomial trees of orders `k - 1` down to `0` as its children, with the scalar of every
node the scalar of its parent times a constant. The seed of a node decides how many children it
has, so the lists of children of one level have many lengths.
-/

@[expose] public section

namespace TorchLean.Examples.Tree

open Kernel Sig.Inductive

variable {α : Type}

namespace BTree

/-- The balanced binary tree of entries `lo` to `hi - 1` of input `0`: the middle entry at the
root, the tree of the entries before it on the left and the tree of the entries after it on the
right. The reads follow the order of the entries, so the first failed read is the result. -/
def ofRange (read : Reader α) (lo hi : Nat) : Except Error (BTree α) :=
  if hi ≤ lo then .ok .leaf
  else do
    let l ← ofRange read lo (lo + (hi - lo) / 2)
    let x ← read 0 (lo + (hi - lo) / 2).toUInt64
    let r ← ofRange read (lo + (hi - lo) / 2 + 1) hi
    return .node l x r
termination_by hi - lo

end BTree

namespace OfRange

/-- The first entry `lo` of the range of a row. -/
def lo : Expr α (Row 2 0) .index :=
  .var (Row.indexVar 0)

/-- The end `hi` of the range of a row. -/
def hi : Expr α (Row 2 0) .index :=
  .var (Row.indexVar 1)

/-- The middle entry of the range of a row. -/
def mid : Expr α (Row 2 0) .index :=
  .indexBinary .add lo (.indexBinary .div (.indexBinary .sub hi lo) (.index 2))

/-- Whether the range of a row is empty. -/
def empty : Expr α (Row 2 0) .predicate :=
  .indexCompare .le hi lo

/-- Whether a call is the first call of its row. -/
def isFirst : Expr α (Row 2 0) .predicate :=
  .indexCompare .eq (.var .zero) (.index 0)

/-- An empty range makes no calls, and a longer range makes one call for each half. -/
def calls : Expr α (Slots 2 0) .index :=
  .cond (.indexCompare .le (.var (Slots.indexVar 1)) (.var (Slots.indexVar 0))) (.index 0)
    (.index 2)

/-- The range of a call: the entries before the middle one for the first call, and the entries
after it for the second. -/
def argIndex (i : Fin 2) : Expr α (Row 2 0) .index :=
  if i = 0 then .cond isFirst lo (.indexBinary .add mid (.index 1)) else .cond isFirst mid hi

/-- The constructor number of the node of a row: `0` for a leaf and `1` for a node. -/
def tag : Expr α (Row 2 0) .index :=
  .cond empty (.index 0) (.index 1)

/-- Index column `i` of the node of a row: the row of child `i` from the first child, or `0` for a
leaf. -/
def child (i : Fin 2) : Expr α (Row 2 0) .index :=
  .cond empty (.index 0) (.indexBinary .add (.var .zero) (.index i.val.toUInt64))

/-- The scalar column of the node of a row: the middle entry, or `default` for a leaf. -/
def node [Inhabited α] : Expr α (Row 2 0) .scalar :=
  .cond empty (.scalar default) (.load 0 mid)

/-- The reference: `BTree.ofRange` over the range of the seed. -/
def reference :
    Reader α → Env α (Slots 2 0) → Except Error (Σ _ : Fin BTree.sig.sorts, BTree α) :=
  fun read env => (BTree.ofRange read (env.get (Slots.indexVar 0)).toNat
    (env.get (Slots.indexVar 1)).toNat).map (⟨0, ·⟩)

/-- The certificate of the unfold of `BTree.ofRange`. -/
theorem correct [Inhabited α] (A : Arithmetic α) :
    Unfold.Correct BTree.sig (fun _ => BTree α) A reference calls argIndex Fin.elim0 tag child
      (fun _ => node) := by
  intro read index value first ys hcalls hfirst hargs tag' index'' value'' htag hindex hscalar
  have hsize : UInt64.size = 2 ^ 64 := rfl
  simp only [calls, Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one, Compare.index,
    Bind.bind, Except.bind, Pure.pure, Except.pure] at hcalls
  simp only [tag, empty, hi, lo, Expr.eval, Row.get_env_indexVar, Fin.val_zero, Fin.val_one,
    Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure] at htag
  have hb := (index 1).toNat_lt
  by_cases hle : index 1 ≤ index 0
  · have hle' : (index 1).toNat ≤ (index 0).toNat := UInt64.le_iff_toNat_le.1 hle
    simp only [hle, decide_true, Bool.cond_true, Except.ok.injEq] at hcalls htag
    have hlen : ys.length = 0 := by
      have := congrArg UInt64.toNat hcalls
      rw [Nat.toUInt64_eq, UInt64.toNat_ofNat'] at this
      simp only [UInt64.toNat_zero] at this
      omega
    obtain rfl := List.eq_nil_of_length_eq_zero hlen
    refine ⟨⟨0, .leaf⟩, ?_, rfl, ?_, fun i => ?_, fun i => ?_⟩
    · simp only [reference, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one]
      rw [BTree.ofRange, ite_eq_left hle']
      rfl
    · rw [← htag]
      rfl
    · have h := hindex i
      simp only [child, empty, hi, lo, Expr.eval, Row.get_env_indexVar, Fin.val_zero, Fin.val_one,
        Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure, hle, decide_true,
        Bool.cond_true, Except.ok.injEq] at h
      rw [← h]
      show (0 : UInt64) = (#[] : Array UInt64).getD i 0
      simp
    · have h := hscalar i
      simp only [node, empty, hi, lo, Expr.eval, Row.get_env_indexVar, Fin.val_zero, Fin.val_one,
        Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure, hle, decide_true,
        Bool.cond_true, Except.ok.injEq] at h
      rw [← h]
      show (default : α) = ([] : List α).getD i default
      simp
  · have hlt : (index 0).toNat < (index 1).toNat := by
      simpa [UInt64.le_iff_toNat_le] using hle
    simp only [hle, decide_false, Bool.cond_false, Except.ok.injEq] at hcalls htag
    have hlen : ys.length = 2 := by
      have := congrArg UInt64.toNat hcalls
      rw [Nat.toUInt64_eq, UInt64.toNat_ofNat'] at this
      simp only [UInt64.reduceToNat] at this
      omega
    obtain ⟨y0, y1, rfl⟩ := List.length_eq_two.1 hlen
    have hsub : (index 1 - index 0).toNat = (index 1).toNat - (index 0).toNat :=
      UInt64.toNat_sub_of_le _ _ (UInt64.le_of_lt (by simpa using hle))
    have hmid : (index 0 + (index 1 - index 0) / 2).toNat =
        (index 0).toNat + ((index 1).toNat - (index 0).toNat) / 2 := by
      rw [UInt64.toNat_add, UInt64.toNat_div, hsub]
      simp only [UInt64.reduceToNat]
      omega
    have hmid1 : (index 0 + (index 1 - index 0) / 2 + 1).toNat =
        (index 0).toNat + ((index 1).toNat - (index 0).toNat) / 2 + 1 := by
      rw [UInt64.toNat_add, hmid]
      simp only [UInt64.reduceToNat]
      omega
    obtain ⟨i0, v0, hi0, -, e0⟩ := hargs 0 (by simp)
    obtain ⟨i1, v1, hi1, -, e1⟩ := hargs 1 (by simp)
    have a0 := hi0 0
    have b0 := hi0 1
    have a1 := hi1 0
    have b1 := hi1 1
    simp only [argIndex, isFirst, mid, lo, hi, Expr.eval, Row.get_env_zero, Row.get_env_indexVar,
      Fin.val_zero, Fin.val_one, Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure,
      IndexOp.eval, Fin.isValue, one_ne_zero, ↓reduceIte] at a0 b0 a1 b1
    have h10 : ((1 : UInt64) == 0) = false := by decide
    simp only [Nat.toUInt64_eq, UInt64.reduceOfNat, beq_self_eq_true, Bool.cond_true,
      Except.ok.injEq] at a0 b0
    simp only [Nat.toUInt64_eq, UInt64.reduceOfNat, h10, Bool.cond_false, Except.ok.injEq] at a1 b1
    simp only [reference, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one,
      List.getElem_cons_zero, List.getElem_cons_succ] at e0 e1
    rw [← a0, ← b0, hmid] at e0
    rw [← a1, ← b1, hmid1] at e1
    have hv := hscalar ⟨0, by decide⟩
    simp only [node, empty, mid, hi, lo, Expr.eval, Row.get_env_indexVar, Fin.val_zero,
      Fin.val_one, Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure, IndexOp.eval,
      hle, decide_false, Bool.cond_false] at hv
    have hmid' : ((index 0).toNat + ((index 1).toNat - (index 0).toNat) / 2).toUInt64 =
        index 0 + (index 1 - index 0) / 2 := by
      rw [← hmid, Nat.toUInt64_eq, UInt64.ofNat_toNat]
    cases hl : BTree.ofRange read (index 0).toNat
        ((index 0).toNat + ((index 1).toNat - (index 0).toNat) / 2) with
    | error e =>
      rw [hl] at e0
      cases e0
    | ok l =>
    cases hr : BTree.ofRange read ((index 0).toNat + ((index 1).toNat - (index 0).toNat) / 2 + 1)
        (index 1).toNat with
    | error e =>
      rw [hr] at e1
      cases e1
    | ok r =>
    rw [hl] at e0
    rw [hr] at e1
    simp only [Except.map, Except.ok.injEq] at e0 e1
    subst e0 e1
    refine ⟨⟨0, .node l (value'' ⟨0, by decide⟩) r⟩, ?_, rfl, ?_, ?_, ?_⟩
    · simp only [reference, Slots.get_env_indexVar, Fin.val_zero, Fin.val_one]
      rw [BTree.ofRange, ite_eq_right (by omega)]
      simp only [hl, hr, hmid', hv, Bind.bind, Except.bind, Pure.pure, Except.pure]
      rfl
    · rw [← htag]
      rfl
    · rintro ⟨i, hi2⟩
      have h := hindex ⟨i, hi2⟩
      simp only [child, empty, hi, lo, Expr.eval, Row.get_env_zero, Row.get_env_indexVar,
        Fin.val_zero, Fin.val_one, Compare.index, Bind.bind, Except.bind, Pure.pure, Except.pure,
        IndexOp.eval, hle, decide_false, Bool.cond_false, Except.ok.injEq] at h
      rw [← h]
      show _ = (#[first.toUInt64] ++ (#[(first + 1).toUInt64] ++ #[])).getD i 0
      obtain rfl | rfl : i = 0 ∨ i = 1 := by
        have : i < 2 := hi2
        omega
      · simp
      · simp [Nat.toUInt64_eq]
    · rintro ⟨i, hi1⟩
      obtain rfl : i = 0 := by
        have : i < 1 := hi1
        omega
      show _ = [value'' ⟨0, by decide⟩].getD 0 default
      rfl

end OfRange

/-- The unfold of the balanced binary trees of ranges of input `0`, in the device arithmetic of
`α`: a seed holds the first entry and the end of its range in its index slots. -/
def ofRangeUnfold [Scalar α] [Inhabited α] : Unfold BTree.sig α (fun _ => BTree α) 2 0 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 1
  calls := OfRange.calls
  argIndex := OfRange.argIndex
  argValue := Fin.elim0
  native := rfl
  reference := OfRange.reference
  tag := OfRange.tag
  nodeIndex := OfRange.child
  nodeScalar _ := OfRange.node
  correct := OfRange.correct Scalar.arithmetic

namespace Rose

/-- The binomial tree of order `k` in arithmetic `A` with the scalar `x` at its root: the children
of a node of order `k` are binomial trees of orders `k - 1` down to `0`, each with the scalar of
its parent times `c`, so the tree has `2 ^ k` nodes. -/
def binomial (A : Arithmetic α) (c : α) (k : Nat) (x : α) : Rose α :=
  .node x (List.ofFn fun j : Fin k => binomial A c (k - 1 - j) (A.binary .mul x c))
termination_by k
decreasing_by
  have := j.isLt
  omega

end Rose

namespace Binomial

/-- The order of call `j` of a row of order `k`: `k - 1 - j`. -/
def argIndex : Expr α (Row 1 1) .index :=
  .indexBinary .sub (.indexBinary .sub (.var (Row.indexVar 0)) (.index 1)) (.var .zero)

/-- The scalar of every call of a row: the scalar of the row times `c`. -/
def argValue (c : α) : Expr α (Row 1 1) .scalar :=
  .binary .mul (.var (Row.scalarVar 0)) (.scalar c)

/-- Index column `i` of the node of a row: the position of its first child, and then its number
of children, which is its order. -/
def slot (i : Fin 2) : Expr α (Row 1 1) .index :=
  if i.val = 0 then .var .zero else .var (Row.indexVar 0)

/-- The reference: `Rose.binomial` at the order and the scalar of the seed. -/
def reference (A : Arithmetic α) (c : α) :
    Reader α → Env α (Slots 1 1) → Except Error (Σ _ : Fin Rose.sig.sorts, Rose α) :=
  fun _ env => .ok ⟨0, Rose.binomial A c (env.get (Slots.indexVar 0)).toNat
    (env.get (Slots.scalarVar 0))⟩

/-- The certificate of the unfold of `Rose.binomial`. -/
theorem correct [Inhabited α] (A : Arithmetic α) (c : α) :
    Unfold.Correct Rose.sig (fun _ => Rose α) A (reference A c) (.var (Slots.indexVar 0))
      (fun _ => argIndex) (fun _ => argValue c) (.index 0) slot
      (fun _ => .var (Row.scalarVar 0)) := by
  intro read index value first ys hcalls hfirst hargs tag' index'' value'' htag hindex hscalar
  have hsize : UInt64.size = 2 ^ 64 := rfl
  simp only [Expr.eval, Slots.get_env_indexVar, Fin.val_zero, Except.ok.injEq] at hcalls htag
  have hn : (index 0).toNat = ys.length := by
    rw [hcalls, Nat.toUInt64_eq, UInt64.toNat_ofNat']
    omega
  have h0 := hindex ⟨0, by decide⟩
  have h1 := hindex ⟨1, by decide⟩
  have hv := hscalar ⟨0, by decide⟩
  simp only [slot, Expr.eval, Row.get_env_zero, Row.get_env_indexVar, Row.get_env_scalarVar,
    Fin.val_zero, Except.ok.injEq, ↓reduceIte, Nat.one_ne_zero] at h0 h1 hv
  have hbin : Rose.binomial A c (index 0).toNat (value 0) = .node (value 0)
      (List.ofFn fun j : Fin (index 0).toNat =>
        Rose.binomial A c ((index 0).toNat - 1 - j) (A.binary .mul (value 0) c)) := by
    rw [Rose.binomial]
  refine ⟨⟨0, Rose.binomial A c (index 0).toNat (value 0)⟩, rfl, ?_, ?_, ?_, ?_⟩
  · rw [hbin]
    change List.map _ (List.ofFn _) ++ [] = ys
    rw [List.append_nil]
    apply List.ext_getElem (by simp [hn])
    intro j hj hj'
    obtain ⟨i', v', hi', hv', e'⟩ := hargs j hj'
    have a := hi' 0
    have b := hv' 0
    simp only [argIndex, argValue, Expr.eval, Row.get_env_zero, Row.get_env_indexVar,
      Row.get_env_scalarVar, Fin.val_zero, Bind.bind, Except.bind, Pure.pure, Except.pure,
      IndexOp.eval, Except.ok.injEq] at a b
    simp only [reference, Slots.get_env_indexVar, Slots.get_env_scalarVar, Fin.val_zero,
      Except.ok.injEq] at e'
    have hj2 : (index 0 - 1 - j.toUInt64).toNat = (index 0).toNat - 1 - j := by
      have := (index 0).toNat_lt
      rw [UInt64.toNat_sub, UInt64.toNat_sub, Nat.toUInt64_eq, UInt64.toNat_ofNat']
      simp only [UInt64.reduceToNat]
      omega
    rw [← e', ← a, ← b, hj2]
    simp only [List.getElem_map, List.getElem_ofFn]
  · rw [← htag, hbin]
    rfl
  · rintro ⟨i, hi⟩
    have hi2 : i < 2 := hi
    rw [hbin]
    obtain rfl | rfl : i = 0 ∨ i = 1 := by omega
    · rw [← h0]
      show _ = (#[first.toUInt64, (List.ofFn _).length.toUInt64] ++ #[]).getD 0 0
      simp
    · rw [← h1]
      show _ = (#[first.toUInt64, (List.ofFn _).length.toUInt64] ++ #[]).getD 1 0
      simp
  · rintro ⟨i, hi⟩
    obtain rfl : i = 0 := by
      have hi1 : i < 1 := hi
      omega
    rw [← hv, hbin]
    rfl

end Binomial

/-- The unfold of binomial trees in the device arithmetic of `α`, with the scalars of the children
of a node the scalar of the node times `c`: a seed holds the order of its tree in its index slot
and the scalar of the root in its scalar slot. -/
def binomialUnfold [Scalar α] [Inhabited α] (c : α) : Unfold Rose.sig α (fun _ => Rose α) 1 1 where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 0
  calls := .var (Slots.indexVar 0)
  argIndex _ := Binomial.argIndex
  argValue _ := Binomial.argValue c
  native := rfl
  reference := Binomial.reference Scalar.arithmetic c
  tag := .index 0
  nodeIndex := Binomial.slot
  nodeScalar _ := .var (Row.scalarVar 0)
  correct := Binomial.correct Scalar.arithmetic c

end TorchLean.Examples.Tree
