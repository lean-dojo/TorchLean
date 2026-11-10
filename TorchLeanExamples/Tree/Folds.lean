/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Table
public import TorchLean.CUDA.Tree.Run
public import TorchLeanExamples.Tree.Types

/-!
# Sample folds

Folds over the sample tree types, each with its certificate proved here: the sum of the scalars of
a binary tree, a sum that scales each scalar by a weight read from an input, the sum of the scalars
of a rose tree, whose step loops over the children of a node, and the value of an arithmetic term
whose variables are read from an environment. A failed read is the result of the fold. For the
binary64 sum, `BTree.denotes_sumFold` instantiates `Step.denotes_source`: the row kernel that the
device runs computes the rows of every level from the results below. The fold `BTree.tableFold`
sums what a table gives at the scalars of a binary tree, with the keys and the values of the table
as inputs of the fold. For a table of `f`, `BTree.eval_tableFold` proves that over the arena of a
tree it gives the sum of the tree mapped by `f`, so a fold can apply FloatLib's correctly rounded
functions at every node.
-/

@[expose] public section

namespace TorchLean.Examples.Tree

open Kernel

variable {α : Type}

namespace BTree

/-- The sum of the scalars of a binary tree in arithmetic `A`, with `zero` at the leaves: the left
subtree, then the node's scalar, then the right subtree. -/
def sum (A : Arithmetic α) (zero : α) : BTree α → α
  | .leaf => zero
  | .node l x r => A.binary .add (A.binary .add (sum A zero l) x) (sum A zero r)

/-- The sum in arithmetic `A` of the scalars of a binary tree, each scaled by element `0` of input
`0`. The weight is read after the left subtree and before the right one, so the first failed read
is the result. -/
def weighted (A : Arithmetic α) (zero : α) (read : Reader α) : BTree α → Except Error α
  | .leaf => .ok zero
  | .node l x r => do
    let a ← weighted A zero read l
    let w ← read 0 0
    let b ← weighted A zero read r
    return A.binary .add (A.binary .add a (A.binary .mul x w)) b

/-- The step of the sum on a node: the children's results, input `0`, around the node's scalar. -/
def sumNode : Expr α BTree.sig.row .scalar :=
  .binary .add
    (.binary .add (.load 0 (.var (Row.indexVar ⟨0, by decide⟩)))
      (.var (Row.scalarVar ⟨0, by decide⟩)))
    (.load 0 (.var (Row.indexVar ⟨1, by decide⟩)))

/-- The step of the weighted sum on a node: the children's results are input `1`, and the weight
is element `0` of input `0`. -/
def weightedNode : Expr α BTree.sig.row .scalar :=
  .binary .add
    (.binary .add (.load 1 (.var (Row.indexVar ⟨0, by decide⟩)))
      (.binary .mul (.var (Row.scalarVar ⟨0, by decide⟩)) (.load 0 (.index 0))))
    (.load 1 (.var (Row.indexVar ⟨1, by decide⟩)))

/-- The certificate of the sum: a leaf gives `zero`, and a node adds its children's results around
its scalar. -/
theorem sum_cases (A : Arithmetic α) (zero : α) :
    Fold.Cases BTree.sig A 0 (fun _ _ t => .ok (sum A zero t)) 0 [.scalar zero, sumNode] := by
  refine ⟨fun _ => ?_, fun _ => ?_, trivial⟩
  · rintro read R index value ⟨⟩ - -
    rfl
  · rintro read R index value ⟨l, x, r, ⟨⟩⟩ - ⟨hl, rfl, hr, -⟩
    simp only [Nat.zero_add] at hl hr
    simp only [sumNode, Expr.eval, Row.get_env_indexVar, Row.get_env_scalarVar, hl, hr,
      Bind.bind, Except.bind, Pure.pure, Except.pure]
    rfl

/-- The certificate of the weighted sum: a leaf gives `zero`, and a node reads its left child's
result, the weight and its right child's result in that order. -/
theorem weighted_cases (A : Arithmetic α) (zero : α) :
    Fold.Cases BTree.sig A 1 (fun read _ t => weighted A zero read t) 0
      [.scalar zero, weightedNode] := by
  refine ⟨fun _ => ?_, fun _ => ?_, trivial⟩
  · rintro read R index value ⟨⟩ - -
    rfl
  · rintro read R index value ⟨l, x, r, ⟨⟩⟩ hR ⟨hl, rfl, hr, -⟩
    simp only [Nat.zero_add] at hl hr
    have h0 : R 0 = read 0 := hR 0 Nat.one_pos
    show _ = weighted A zero read (.node l (value 0) r)
    simp only [weightedNode, weighted, Expr.eval, Row.get_env_indexVar,
      Row.get_env_scalarVar, hl, hr, h0, Bind.bind, Except.bind, Pure.pure, Except.pure]
    cases weighted A zero read l <;> cases read 0 0 <;> cases weighted A zero read r <;> rfl

/-- The fold that sums the scalars of a binary tree in the device arithmetic of `α`. -/
def sumFold [Scalar α] (zero : α) : Fold BTree.sig α (fun _ => BTree α) where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 0
  reference _ _ t := .ok (sum Scalar.arithmetic zero t)
  expr := Fold.select 0 (.scalar zero) [sumNode]
  correct := Fold.correct_select (sum_cases Scalar.arithmetic zero) rfl
  native := rfl

/-- The fold that sums the scalars of a binary tree, each scaled by a weight read from input `0`,
in the device arithmetic of `α`. -/
def weightedFold [Scalar α] (zero : α) : Fold BTree.sig α (fun _ => BTree α) where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 1
  reference read _ t := weighted Scalar.arithmetic zero read t
  expr := Fold.select 0 (.scalar zero) [weightedNode]
  correct := Fold.correct_select (weighted_cases Scalar.arithmetic zero) rfl
  native := rfl

/-- The row kernel of the binary64 sum, as `Step.source` renders it, computes the rows of every
level from the results of the next level, failed reads included. -/
theorem denotes_sumFold {text : String} (hSource : (sumFold (0.0 : Float)).source = .ok text)
    (read : Reader Float) (level : Arena.Level Float) (below : Array Float)
    (hBelow : below.size < UInt64.size) :
    CUDA.RowSource.Denotes .binary64 (sumFold (0.0 : Float)).arithmetic 1
      (fun op => if op = 0 then below.size.toUInt64 else 0)
      (fun op i => if op = 0 then below[i.toNat]?.getD default else 0)
      (fun index => level.env BTree.sig.indexSlots BTree.sig.scalarSlots index.toNat) text
      (fun recorded block blockSize thread count _ =>
        let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
        if index ≥ count then some (.ok none)
        else if recorded then some (.ok none)
        else some (((sumFold (0.0 : Float)).expr.eval (sumFold (0.0 : Float)).arithmetic
          (Step.reader read 0 (below.map .ok))
          (level.env BTree.sig.indexSlots BTree.sig.scalarSlots index.toNat)).map some)) :=
  Step.denotes_source (format := .binary64) (sumFold (0.0 : Float)).toStep rfl Scalar.bits_float
    hSource read (fun _ => 0) (fun _ _ => 0) (fun _ h => absurd h (Nat.not_lt_zero _)) level
    below hBelow

/-- The sum in arithmetic `A` of what the table `t` gives at the scalars of a binary tree, with
`zero` at the leaves, where the table reads its keys from input `0` and its values from input `1`
of `read`. The order is that of `sum`, the left subtree, then the node, then the right subtree, so
the first failed read is the result. -/
def tableSum (A : Arithmetic α) (zero : α) (t : Table α) (read : Reader α) :
    BTree α → Except Error α
  | .leaf => .ok zero
  | .node l x r => do
    let a ← tableSum A zero t read l
    let y ← t.evalWith A (read 0) (read 1) x
    let b ← tableSum A zero t read r
    return A.binary .add (A.binary .add a y) b

/-- The step of `tableSum` on a node: the children's results, input `2`, around the table `t`
applied to the node's scalar, with the keys of `t` as input `0` and its values as input `1`. -/
def tableNode (t : Table α) : Expr α BTree.sig.row .scalar :=
  .binary .add
    (.binary .add (.load 2 (.var (Row.indexVar ⟨0, by decide⟩)))
      (t.apply 0 1 (.var (Row.scalarVar ⟨0, by decide⟩))))
    (.load 2 (.var (Row.indexVar ⟨1, by decide⟩)))

/-- The certificate of `tableSum`: a leaf gives `zero`, and a node reads its left child's result,
the table at its scalar and its right child's result in that order. -/
theorem tableSum_cases (A : Arithmetic α) (zero : α) (t : Table α) :
    Fold.Cases BTree.sig A 2 (fun read _ s => tableSum A zero t read s) 0
      [.scalar zero, tableNode t] := by
  refine ⟨fun _ => ?_, fun _ => ?_, trivial⟩
  · rintro read R index value ⟨⟩ - -
    rfl
  · rintro read R index value ⟨l, x, r, ⟨⟩⟩ hR ⟨hl, rfl, hr, -⟩
    simp only [Nat.zero_add] at hl hr
    have h0 : R 0 = read 0 := hR 0 (by decide)
    have h1 : R 1 = read 1 := hR 1 (by decide)
    show _ = tableSum A zero t read (.node l (value 0) r)
    simp only [tableNode, tableSum, Expr.eval, Table.eval_apply, Row.get_env_indexVar,
      Row.get_env_scalarVar, hl, hr, h0, h1, Bind.bind, Except.bind, Pure.pure, Except.pure]
    cases tableSum A zero t read l <;> cases t.evalWith A (read 0) (read 1) (value 0) <;>
      cases tableSum A zero t read r <;> rfl

/-- The fold of `tableSum` with the table `t` in the device arithmetic of `α`, which reads the keys
of `t` from input `0` and its values from input `1`. -/
def tableFold [Scalar α] (zero : α) (t : Table α) : Fold BTree.sig α (fun _ => BTree α) where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 2
  reference read _ s := tableSum Scalar.arithmetic zero t read s
  expr := Fold.select 0 (.scalar zero) [tableNode t]
  correct := Fold.correct_select (tableSum_cases Scalar.arithmetic zero t) rfl
  native := rfl

/-- Over a reader that holds the keys of a table of `f` as input `0` and its values as input `1`,
`tableSum` is the sum of the tree mapped by `f`. -/
theorem tableSum_of_holds {A : Arithmetic α} {f : α → α} (T : Table.Of A f) (zero : α)
    {read : Reader α} (hk : read.Holds 0 T.table.keys) (hv : read.Holds 1 T.table.values) :
    ∀ s : BTree α, tableSum A zero T.table read s = .ok (sum A zero (s.map f))
  | .leaf => rfl
  | .node l x r => by
    simp only [tableSum, sum, map, tableSum_of_holds T zero hk hv l,
      tableSum_of_holds T zero hk hv r, T.evalWith_eq hk hv x, Bind.bind, Except.bind,
      Pure.pure, Except.pure]

/-- Over a reader that holds the keys of a table of `f` as input `0` and its values as input `1`,
the fold of the table over the arena of a binary tree of fewer than `2 ^ 64` nodes gives the sum
of the tree mapped by `f`. -/
theorem eval_tableFold [Scalar α] [Inhabited α] {f : α → α} (T : Table.Of Scalar.arithmetic f)
    (zero : α) {read : Reader α} (hk : read.Holds 0 T.table.keys)
    (hv : read.Holds 1 T.table.values) (t : BTree α)
    (ht : Sig.Inductive.count (σ := BTree.sig) (T := fun _ => BTree α) (s := 0) t <
      UInt64.size) :
    (tableFold zero T.table).eval read
        (Arena.encode (σ := BTree.sig) (T := fun _ => BTree α) (s := 0) t) =
      .ok (sum Scalar.arithmetic zero (t.map f)) :=
  ((tableFold zero T.table).eval_encode read (s := 0) t ht).trans
    (tableSum_of_holds T zero hk hv t)

end BTree

namespace Rose

mutual
/-- The sum of the scalars of a rose tree in arithmetic `A`: the node's scalar, then the sums of
the children from the first to the last. -/
def sum (A : Arithmetic α) : Rose α → α
  | .node x cs => sumList A x cs

/-- The sums of the trees `cs` added to `acc` from the first to the last. -/
def sumList (A : Arithmetic α) (acc : α) : List (Rose α) → α
  | [] => acc
  | c :: cs => sumList A (A.binary .add acc (sum A c)) cs
end

/-- The step of the sum on a node: a loop over the children that adds their results, input `0`
from the row of the first child on, to the node's scalar. -/
def sumNode : Expr α Rose.sig.row .scalar :=
  .fold (.var (Row.indexVar ⟨1, by decide⟩)) (.var (Row.scalarVar ⟨0, by decide⟩))
    (.binary .add (.var .zero)
      (.load 0 (.indexBinary .add (.var (.succ (.succ (Row.indexVar ⟨0, by decide⟩))))
        (.var (.succ .zero)))))

/-- A loop whose step `i` adds the read of row `start + i` of input `0` adds the sums of the trees
`cs` to `acc` in order, once those reads from step `i` on give the sums. -/
theorem iterate_sum (A : Arithmetic α) (R : Reader α) (start : UInt64)
    (step : UInt64 → α → Except Error α)
    (hstep : ∀ i acc, step i acc = (do let y ← R 0 (start + i); return A.binary .add acc y)) :
    ∀ (cs : List (Rose α)) (i : UInt64) (acc : α),
      (∀ m (h : m < cs.length), R 0 (start + (i + m.toUInt64)) = .ok (sum A cs[m])) →
      iterate step cs.length i acc = .ok (sumList A acc cs)
  | [], _, _, _ => rfl
  | c :: cs, i, acc, h => by
    have h0 : R 0 (start + i) = .ok (sum A c) := by
      simpa using h 0 (Nat.zero_lt_succ cs.length)
    rw [List.length_cons, iterate, hstep, h0, sumList]
    refine iterate_sum A R start step hstep cs (i + 1) _ fun m hm => ?_
    have hm' := h (m + 1) (by simp; omega)
    rw [show i + (m + 1).toUInt64 = i + 1 + m.toUInt64 by
      simp only [Nat.toUInt64, UInt64.ofNat_add, UInt64.ofNat_one]
      rw [UInt64.add_assoc, UInt64.add_comm 1 (UInt64.ofNat m)]] at hm'
    simpa using hm'

/-- The certificate of the sum: a node's loop adds its children's results to its scalar from the
first child to the last. -/
theorem sum_cases (A : Arithmetic α) :
    Fold.Cases Rose.sig A 0 (fun _ _ t => .ok (sum A t)) 0 [sumNode] := by
  refine ⟨fun _ => ?_, trivial⟩
  rintro read R index value ⟨x, cs, ⟨⟩⟩ - ⟨rfl, hlength, hcs, -⟩
  show _ = Except.ok (sumList A (value 0) cs)
  have hloop := iterate_sum A R (index 0) _ (fun i acc => rfl) cs 0 (value 0) fun m hm => by
    simpa using hcs m hm
  rw [← hloop, ← hlength]
  simp only [sumNode, Expr.eval, Row.get_env_indexVar, Row.get_env_scalarVar,
    Env.get_push_zero, Env.get_push_succ, IndexOp.eval, Bind.bind, Except.bind, Pure.pure,
    Except.pure]

/-- The fold that sums the scalars of a rose tree in the device arithmetic of `α`. -/
def sumFold [Scalar α] : Fold Rose.sig α (fun _ => Rose α) where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 0
  reference _ _ t := .ok (sum Scalar.arithmetic t)
  expr := Fold.select 0 sumNode []
  correct := Fold.correct_select (sum_cases Scalar.arithmetic) rfl
  native := rfl

end Rose

namespace Term

/-- The value of a term in arithmetic `A`, with variable `v` read as element `v` of input `0`,
each operation evaluating its first operand and then its second. -/
def eval (A : Arithmetic α) (read : Reader α) : Term α → Except Error α
  | .lit x => .ok x
  | .var v => read 0 v
  | .add a b => do
    let x ← eval A read a
    let y ← eval A read b
    return A.binary .add x y
  | .mul a b => do
    let x ← eval A read a
    let y ← eval A read b
    return A.binary .mul x y

/-- The step of a literal: its scalar. -/
def litNode : Expr α Term.sig.row .scalar := .var (Row.scalarVar ⟨0, by decide⟩)

/-- The step of a variable: element `v` of input `0`, for the index `v` of the variable. -/
def varNode : Expr α Term.sig.row .scalar := .load 0 (.var (Row.indexVar ⟨0, by decide⟩))

/-- The step of a sum or a product: the results of the operands, input `1`, combined by `op`. -/
def opNode (op : ScalarOp) : Expr α Term.sig.row .scalar :=
  .binary op (.load 1 (.var (Row.indexVar ⟨0, by decide⟩)))
    (.load 1 (.var (Row.indexVar ⟨1, by decide⟩)))

/-- The certificate of the evaluation: a literal gives its scalar, a variable its read, and a sum
or a product combines its operands' results in order. -/
theorem eval_cases (A : Arithmetic α) :
    Fold.Cases Term.sig A 1 (fun read _ t => eval A read t) 0
      [litNode, varNode, opNode .add, opNode .mul] := by
  refine ⟨fun _ => ?_, fun _ => ?_, fun _ => ?_, fun _ => ?_, trivial⟩
  · rintro read R index value ⟨x, ⟨⟩⟩ - ⟨rfl, -⟩
    simp [litNode, Expr.eval]
    rfl
  · rintro read R index value ⟨v, ⟨⟩⟩ hR ⟨rfl, -⟩
    show _ = read 0 (index 0)
    simp only [varNode, Expr.eval, Row.get_env_indexVar, hR 0 Nat.one_pos, Bind.bind,
      Except.bind]
  · rintro read R index value ⟨a, b, ⟨⟩⟩ - ⟨ha, hb, -⟩
    simp only [Nat.zero_add] at ha hb
    show _ = eval A read (.add a b)
    simp only [opNode, eval, Expr.eval, Row.get_env_indexVar, ha, hb, Bind.bind, Except.bind,
      Pure.pure, Except.pure]
  · rintro read R index value ⟨a, b, ⟨⟩⟩ - ⟨ha, hb, -⟩
    simp only [Nat.zero_add] at ha hb
    show _ = eval A read (.mul a b)
    simp only [opNode, eval, Expr.eval, Row.get_env_indexVar, ha, hb, Bind.bind, Except.bind,
      Pure.pure, Except.pure]

/-- The fold that evaluates a term in the device arithmetic of `α`, with the environment of its
variables as input `0`. -/
def evalFold [Scalar α] : Fold Term.sig α (fun _ => Term α) where
  scalar := inferInstance
  arithmetic := Scalar.arithmetic
  inputs := 1
  reference read _ t := eval Scalar.arithmetic read t
  expr := Fold.select 0 litNode [varNode, opNode .add, opNode .mul]
  correct := Fold.correct_select (eval_cases Scalar.arithmetic) rfl
  native := rfl

end Term

end TorchLean.Examples.Tree
