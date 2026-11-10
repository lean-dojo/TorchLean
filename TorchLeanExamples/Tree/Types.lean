/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Derive

/-!
# Sample tree types

Binary trees with a scalar at each node, rose trees with a scalar and a list of children at each
node, and arithmetic terms with variables, described to the tree kernels by
`deriving Sig.Inductive`. The generators build perfect trees, caterpillars, and pseudorandom trees
and terms of a given number of nodes for the tests and the benchmarks.
-/

@[expose] public section

namespace TorchLean.Examples.Tree

open Kernel

/-- Binary trees with a scalar at each node. -/
inductive BTree (α : Type) where
  | leaf
  | node (l : BTree α) (x : α) (r : BTree α)
  deriving Repr, Inhabited, DecidableEq, Sig.Inductive

namespace BTree

variable {α β : Type}

/-- The tree with `f` applied to each scalar. -/
def map (f : α → β) : BTree α → BTree β
  | .leaf => .leaf
  | .node l x r => .node (l.map f) (f x) (r.map f)

/-- The scalars of a tree, in order. -/
def toList : BTree α → List α
  | .leaf => []
  | .node l x r => l.toList ++ x :: r.toList

/-- The perfect tree of depth `d`, with `2 ^ d - 1` nodes whose scalars are `x`. -/
def perfect (x : α) : Nat → BTree α
  | 0 => .leaf
  | d + 1 => .node (perfect x d) x (perfect x d)

/-- The caterpillar of `n` nodes: a left spine of nodes with scalars `x`, each with a leaf on the
right, so that the tree has depth `n`. -/
def caterpillar (x : α) : Nat → BTree α
  | 0 => .leaf
  | n + 1 => .node (caterpillar x n) x .leaf

/-- One step of the SplitMix64 generator. -/
def mix (z : UInt64) : UInt64 :=
  let z := z + 0x9e3779b97f4a7c15
  let z := (z ^^^ (z >>> 30)) * 0xbf58476d1ce4e5b9
  let z := (z ^^^ (z >>> 27)) * 0x94d049bb133111eb
  z ^^^ (z >>> 31)

/-- A pseudorandom tree of `n` nodes from `seed`: the left subtree gets a uniform share of the
other nodes, and node scalars are `scalar` of pseudorandom words. -/
def random (scalar : UInt64 → α) : (n : Nat) → (seed : UInt64) → BTree α
  | 0, _ => .leaf
  | n + 1, seed =>
    let k := (mix seed).toNat % (n + 1)
    .node (random scalar k (mix (seed ^^^ 1))) (scalar (mix (seed ^^^ 2)))
      (random scalar (n - k) (mix (seed ^^^ 3)))
termination_by n => n
decreasing_by
  · exact Nat.mod_lt _ (Nat.succ_pos n)
  · omega

end BTree

/-- Rose trees: a scalar and a list of children at each node. -/
inductive Rose (α : Type) where
  | node (x : α) (cs : List (Rose α))
  deriving Repr, Inhabited, BEq, Sig.Inductive

namespace Rose

variable {α β : Type}

mutual
/-- The tree with `f` applied to each scalar. -/
def map (f : α → β) : Rose α → Rose β
  | .node x cs => .node (f x) (mapList f cs)

/-- The trees with `f` applied to each scalar. -/
def mapList (f : α → β) : List (Rose α) → List (Rose β)
  | [] => []
  | c :: cs => c.map f :: mapList f cs
end

/-- The forest of a binary tree read with first children on the left and next siblings on the
right. -/
def ofBTree : BTree α → List (Rose α)
  | .leaf => []
  | .node l x r => .node x (ofBTree l) :: ofBTree r

/-- A pseudorandom rose tree of `n + 1` nodes from `seed`, with node scalars `scalar` of
pseudorandom words. -/
def random (scalar : UInt64 → α) (n : Nat) (seed : UInt64) : Rose α :=
  .node (scalar seed) (ofBTree (BTree.random scalar n (BTree.mix seed)))

end Rose

/-- Arithmetic terms over scalars: literals, variables that name an element of an environment,
sums and products. -/
inductive Term (α : Type) where
  | lit (x : α)
  | var (v : UInt64)
  | add (a b : Term α)
  | mul (a b : Term α)
  deriving Repr, Inhabited, BEq, Sig.Inductive

namespace Term

variable {α : Type}

/-- A pseudorandom term with `n` sums and products from `seed`: the left operand gets a uniform
share of the other operations, and a leaf is a literal `scalar w` of a pseudorandom word `w` or a
variable whose index is a pseudorandom word modulo `vars`, with even odds. -/
def random (scalar : UInt64 → α) (vars : UInt64) : (n : Nat) → (seed : UInt64) → Term α
  | 0, seed =>
    let z := BTree.mix seed
    if z % 2 = 0 then .lit (scalar (BTree.mix z)) else .var (BTree.mix z % vars)
  | n + 1, seed =>
    let k := (BTree.mix seed).toNat % (n + 1)
    let a := random scalar vars k (BTree.mix (seed ^^^ 1))
    let b := random scalar vars (n - k) (BTree.mix (seed ^^^ 3))
    if BTree.mix (seed ^^^ 2) % 2 = 0 then .add a b else .mul a b
termination_by n => n
decreasing_by
  · exact Nat.mod_lt _ (Nat.succ_pos n)
  · omega

end Term

end TorchLean.Examples.Tree
