/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Elab
public import TorchLeanTest.Tree.Fold

/-!
# Folds from recursive functions

The checks compile functions with `Fold.of` and compare each fold with the function it comes
from, by the encodings of the results. Over binary trees the functions compute sizes, sums,
heights, counts, weighted sums and guarded sums, with `let`, `if` on comparisons in both
directions, `if` chains and `if`s nested in both branches, six independent `if`s in one body, a
dependent `if`, a `match` on a `Bool`, a fixed leading argument that is closed, captured from a
parameter or holds a comparison, a `match` in the body, a definition by cases without recursion,
`Except` without a reader, binders named like the ones the elaborator makes, and reads of two
inputs, in binary64, binary32 and binary16. Over a derived type whose constructors interleave
index, child and scalar fields, over the expressions of a type derived here, which has negations,
and over the terms of the examples, they are evaluators that read their variables from an input,
and one of them skips an operand. The inputs hold every entry that a function reads, some of
them, or none.

On the host the fold of each function must give the result of the function. On the device the
schedule must finish exactly when the function has a result on the value of every row of the
arena, with the results of the host schedule, and `Fold.run` must give the result of the
function, failures included. The fold of the term evaluator must agree with the fold of the
examples written by hand, and the reference of a fold is the function itself, by `rfl`. The
functions that `Fold.of` rejects are checked by their messages when this file compiles.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Elaborated

open Kernel Examples.Tree Arenas FloatLib.Floats
open Folds (allSame arenaT checkRun checkSource environment moderate same)

variable {α : Type}

/-! ### Binary trees -/

/-- The number of nodes of a tree. -/
def size : BTree Float → Float
  | .leaf => 0
  | .node l _ r => size l + 1 + size r

/-- The fold of `size`. -/
def sizeFold := Fold.of size

/-- The sum of the scalars of a tree. -/
def sum : BTree Float → Float
  | .leaf => 0
  | .node l x r => sum l + x + sum r

/-- The fold of `sum`. -/
def sumFold := Fold.of sum

/-- The height of a tree, with the heights of the children bound by `let`. -/
def height : BTree Float → Float
  | .leaf => 0
  | .node l _ r =>
    let a := height l
    let b := height r
    if a < b then b + 1 else a + 1

/-- The fold of `height`. -/
def heightFold := Fold.of height

/-- The height of a tree, with the recursive calls in the comparison and its branches. -/
def height' : BTree Float → Float
  | .leaf => 0
  | .node l _ r => if height' l < height' r then height' r + 1 else height' l + 1

/-- The fold of `height'`. -/
def heightFold' := Fold.of height'

/-- The number of nodes of a tree, with the match in the body. -/
def nodes (t : BTree Float) : Float :=
  match t with
  | .leaf => 0
  | .node l _ r => nodes l + 1 + nodes r

/-- The fold of `nodes`, given in eta-expanded form. -/
def nodesFold := Fold.of fun t => nodes t

/-- The number of scalars of a tree above `0.5`. -/
def countAbove : BTree Float → Float
  | .leaf => 0
  | .node l x r => countAbove l + (if x > 0.5 then 1 else 0) + countAbove r

/-- The fold of `countAbove`. -/
def countAboveFold := Fold.of countAbove

/-- A function with the fixed leading argument `c`. -/
def scaled (c : Float) : BTree Float → Float
  | .leaf => c
  | .node l x r => scaled c l * c + x - scaled c r

/-- The fold of `scaled 1.5`. -/
def scaledFold := Fold.of (scaled 1.5)

/-- A function with divisions, subtractions and a comparison of the results of the children. -/
def mixed : BTree Float → Float
  | .leaf => 1
  | .node l x r =>
    let a := mixed l
    let b := mixed r
    if a < b then a / (b + 1) - x else (a - b) * 0.5

/-- The fold of `mixed`. -/
def mixedFold := Fold.of mixed

/-- A chain of `if`s: the class of the scalar at each node. -/
def classes : BTree Float → Float
  | .leaf => 0
  | .node l x r => classes l + (if x < 0.25 then 1 else if x < 0.75 then 2 else 3) + classes r

/-- The fold of `classes`. -/
def classesFold := Fold.of classes

/-- `if`s nested in both branches. -/
def quadrants : BTree Float → Float
  | .leaf => 0
  | .node l x r =>
    quadrants l + (if x < 1 then (if x < 0.5 then 1 else 2) else (if x < 2 then 3 else 4)) +
      quadrants r

/-- The fold of `quadrants`. -/
def quadrantsFold := Fold.of quadrants

/-- Six independent `if`s in one body: the number of thresholds below the scalar. -/
def thresholds : BTree Float → Float
  | .leaf => 0
  | .node l x r =>
    thresholds l + (if x > 0.1 then 1 else 0) + (if x > 0.2 then 1 else 0) +
      (if x ≥ 0.3 then 1 else 0) + (if x ≥ 0.4 then 1 else 0) + (if 0.5 ≤ x then 1 else 0) +
      (if 0.6 < x then 1 else 0) + thresholds r

/-- The fold of `thresholds`. -/
def thresholdsFold := Fold.of thresholds

/-- The height of a tree with `≥` on the results of the children, which appear in the condition
and in both branches. -/
def taller : BTree Float → Float
  | .leaf => 0
  | .node l _ r => if taller l ≥ taller r then taller l + 1 else taller r + 1

/-- The fold of `taller`. -/
def tallerFold := Fold.of taller

/-- A dependent `if` whose branches do not use the proof. -/
def dependent : BTree Float → Float
  | .leaf => 0
  | .node l x r => dependent l + (if _h : x < 0.5 then 1 else 0) + dependent r

/-- The fold of `dependent`. -/
def dependentFold := Fold.of dependent

/-- A `match` on a `Bool`. -/
def matched : BTree Float → Float
  | .leaf => 0
  | .node l x r => matched l + (match decide (x < 0.5) with | true => 1 | false => 0) + matched r

/-- The fold of `matched`. -/
def matchedFold := Fold.of matched

/-- The number of scalars below `c`, with the fixed leading argument `c` in the comparison. -/
def below (c : Float) : BTree Float → Float
  | .leaf => 0
  | .node l x r => below c l + (if x < c then 1 else 0) + below c r

/-- The fold of `below` at a fixed argument that holds a comparison. -/
def belowFold := Fold.of (below (if (1 : Float) > 0 then 0.5 else 0.3))

/-- The fold of `scaled c` for a captured parameter `c`. -/
def scaledOf (c : Float) := Fold.of (scaled c)

/-- The scalar at the root, a definition by cases without recursion. -/
def root : BTree Float → Float
  | .leaf => 7
  | .node _ x _ => x

/-- The fold of `root`. -/
def rootFold := Fold.of root

/-- A sum in `Except` without a reader. -/
def checked : BTree Float → Except Error Float
  | .leaf => .ok 0
  | .node l x r => do
    let a ← checked l
    let b ← checked r
    return a + x + b

/-- The fold of `checked`. -/
def checkedFold := Fold.of checked

/-- A sum with binders named like the ones the elaborator makes, one of them shadowing a child
by its result. -/
def named : BTree Float → Float
  | .leaf => 0
  | .node l x r =>
    let l_result := named l
    let l := x * 2
    let tag := named r
    l_result + l + tag

/-- The fold of `named`. -/
def namedFold := Fold.of named

/-- The sum of the scalars of a tree weighted by entry `0` of input `0`, which it reads between
the children. -/
def weighted (read : Reader Float) : BTree Float → Except Error Float
  | .leaf => .ok 0
  | .node l x r => do
    let a ← weighted read l
    let w ← read 0 0
    let b ← weighted read r
    return a + x * w + b

/-- The fold of `weighted`. -/
def weightedFold := Fold.of weighted

/-- A function that reads entry `0` of input `1` at the leaves, and at a node whose scalar is
below `0.25` reads entry `7` of input `0` in place of the right child, so that whether it fails
depends on the scalars. -/
def guarded (read : Reader Float) : BTree Float → Except Error Float
  | .leaf => read 1 0
  | .node l x r => do
    let a ← guarded read l
    if x < 0.25 then
      let w ← read 0 7
      return a * w
    else
      let b ← guarded read r
      return a - b + x

/-- The fold of `guarded`. -/
def guardedFold := Fold.of guarded

-- A fold reads the inputs that its function reads, and then the results of the children.
example : sizeFold.inputs = 0 ∧ weightedFold.inputs = 1 ∧ guardedFold.inputs = 2 :=
  ⟨rfl, rfl, rfl⟩

-- The reference of a fold is the function itself, with `Except.ok` around a pure result.
example : sizeFold.reference = fun _ _ t => .ok (size t) := rfl
example : scaledFold.reference = fun _ _ t => .ok (scaled 1.5 t) := rfl
example : weightedFold.reference = fun read _ t => weighted read t := rfl
example (c : Float) : (scaledOf c).reference = fun _ _ t => .ok (scaled c t) := rfl

/-! ### Binary trees in other formats -/

/-- The sum of the scalars of a tree in binary32. -/
def sum32 : BTree Float32 → Float32
  | .leaf => 0
  | .node l x r => sum32 l + x + sum32 r

/-- The fold of `sum32`. -/
def sum32Fold := Fold.of sum32

/-- The weighted sum of `weighted` in binary32. -/
def weighted32 (read : Reader Float32) : BTree Float32 → Except Error Float32
  | .leaf => .ok 0
  | .node l x r => do
    let a ← weighted32 read l
    let w ← read 0 0
    let b ← weighted32 read r
    return a + x * w + b

/-- The fold of `weighted32`. -/
def weighted32Fold := Fold.of weighted32

/-- The sum of the squares of the scalars of a tree in binary16. -/
def sumSquares16 : BTree (ExecFloat.Binary 5 10) → ExecFloat.Binary 5 10
  | .leaf => 0
  | .node l x r => sumSquares16 l + x * x + sumSquares16 r

/-- The fold of `sumSquares16`. -/
def sumSquares16Fold := Fold.of sumSquares16

/-- A height in binary16 that adds a quarter of the scalar on one branch. -/
def height16 : BTree (ExecFloat.Binary 5 10) → ExecFloat.Binary 5 10
  | .leaf => 0
  | .node l x r =>
    let a := height16 l
    let b := height16 r
    if a < b then b + 1 else a + x / 4

/-- The fold of `height16`. -/
def height16Fold := Fold.of height16

/-- A function in binary16 that reads entry `1` of input `0` at the leaves. -/
def weighted16 (read : Reader (ExecFloat.Binary 5 10)) :
    BTree (ExecFloat.Binary 5 10) → Except Error (ExecFloat.Binary 5 10)
  | .leaf => read 0 1
  | .node l x r => do
    let a ← weighted16 read l
    let b ← weighted16 read r
    return a - x * b

/-- The fold of `weighted16`. -/
def weighted16Fold := Fold.of weighted16

/-- A height in binary32 with an `if` chain, `≥` on the results of the children and `>` on the
scalar. -/
def height32 : BTree Float32 → Float32
  | .leaf => 0
  | .node l x r =>
    let a := height32 l
    let b := height32 r
    if a ≥ b then a + 1.5 else if x > 0.25 then b + 1 else b

/-- The fold of `height32`. -/
def height32Fold := Fold.of height32

/-- A function in binary16 with an `if` chain, `≤`, `>`, negation and division by a literal. -/
def signed16 : BTree (ExecFloat.Binary 5 10) → ExecFloat.Binary 5 10
  | .leaf => 1
  | .node l x r =>
    let a := signed16 l
    let b := signed16 r
    if a ≤ b then -(a / 2) + x else if x > 0 then b else -b

/-- The fold of `signed16`. -/
def signed16Fold := Fold.of signed16

/-! ### Interleaved fields -/

/-- A type whose constructors interleave index, child and scalar fields. -/
inductive Mixed where
  | tip (i : UInt64) (x : Float)
  | fork (i : UInt64) (a : Mixed) (x : Float) (j : UInt64) (b : Mixed) (y : Float)
  deriving Sig.Inductive

namespace Mixed

/-- A weight that compares the index fields and uses every scalar. -/
def weight : Mixed → Float
  | .tip i x => if i < 2 then x else -x
  | .fork i a x j b y => if i ≤ j then a.weight + x - b.weight * y else a.weight * y + b.weight + x

/-- The fold of `Mixed.weight`. -/
def weightFold := Fold.of weight

/-- A value of each shape: tips below and above the threshold, forks in both orders. -/
def values : List (String × Mixed) :=
  [("tip", .tip 1 2.5), ("tip above", .tip 5 2.5),
    ("fork", .fork 1 (.tip 0 1) 0.5 3 (.tip 7 2) 1.5),
    ("fork reversed", .fork 4 (.fork 1 (.tip 0 1) 0.5 3 (.tip 7 2) 1.5) 0.25 2 (.tip 1 3) 2)]

end Mixed

/-! ### Expressions -/

/-- Expressions over `Float` with sums, products and negations, whose variables name entries of
input `0`. -/
inductive Ast where
  | lit (x : Float)
  | var (i : UInt64)
  | add (a b : Ast)
  | mul (a b : Ast)
  | neg (a : Ast)
  deriving Sig.Inductive

namespace Ast

/-- The value of an expression. -/
def eval (read : Reader Float) : Ast → Except Error Float
  | .lit x => .ok x
  | .var i => read 0 i
  | .add a b => do return (← eval read a) + (← eval read b)
  | .mul a b => do return (← eval read a) * (← eval read b)
  | .neg a => do return -(← eval read a)

/-- The fold of `Ast.eval`. -/
def evalFold := Fold.of eval

/-- The value of an expression that skips the right operand of a sum whose left operand is
negative, so that a failing right operand only matters when the left one is not. -/
def short (read : Reader Float) : Ast → Except Error Float
  | .lit x => .ok x
  | .var i => read 0 i
  | .add a b => do
    let x ← short read a
    if x < 0 then return x else return x + (← short read b)
  | .mul a b => do
    let x ← short read a
    let y ← short read b
    return x * y
  | .neg a => do return -(← short read a)

/-- The fold of `Ast.short`. -/
def shortFold := Fold.of short

/-- A chain of `n` sums, each of half the chain below and a negated variable. -/
def chain : Nat → Ast
  | 0 => .var 3
  | n + 1 => .add (.mul (.lit 0.5) (chain n)) (.neg (.var (n.toUInt64 % 11)))

/-- The expression of depth `3 * n` whose subexpressions at depth `3 * k` are two copies of the
one below, the second times a variable. -/
def balanced : Nat → Ast
  | 0 => .lit 1.25
  | n + 1 =>
    .add (.mul (.lit 0.5) (balanced n)) (.neg (.mul (.var (n.toUInt64 % 7)) (balanced n)))

/-- A pseudorandom expression of depth at most `n` from `seed`, with variables below `6`. -/
def random : Nat → UInt64 → Ast
  | 0, seed =>
    let z := BTree.mix seed
    if z % 2 = 0 then .lit (moderate z) else .var (z % 6)
  | n + 1, seed =>
    let z := BTree.mix seed
    let a := random n (BTree.mix (seed ^^^ 1))
    match (z % 8).toNat with
    | 0 => .lit (moderate z)
    | 1 => .var (z % 6)
    | 2 | 3 | 4 => .add a (random n (BTree.mix (seed ^^^ 3)))
    | 5 | 6 => .mul a (random n (BTree.mix (seed ^^^ 3)))
    | _ => .neg a

end Ast

/-- The expressions of the checks: a literal, a variable, small sums and negations, a chain of
1000 sums, an expression of depth 36 and pseudorandom expressions. -/
def asts : List (String × Ast) :=
  [("literal", .lit 2.5), ("variable", .var 4), ("negated sum", .neg (.add (.var 1) (.lit 3))),
    ("sum of a negation", .add (.neg (.lit 1)) (.var 1)), ("chain 1000", Ast.chain 1000),
    ("balanced 12", Ast.balanced 12)] ++
  (List.range 6).map fun i => (s!"random {i}", Ast.random 18 i.toUInt64)

/-! ### Terms -/

/-- The value of a term of the examples, written as a recursive function. -/
def termEval (read : Reader Float) : Term Float → Except Error Float
  | .lit x => pure x
  | .var v => read 0 v
  | .add a b => do
    let x ← termEval read a
    let y ← termEval read b
    pure (x + y)
  | .mul a b => do
    let x ← termEval read a
    let y ← termEval read b
    pure (x * y)

/-- The fold of `termEval`. -/
def termFold := Fold.of termEval

/-! ### Checks -/

/-- Whether a function has a result on the value of every row of an arena, read as sort `s`,
which is when the schedule of its fold finishes. -/
def allRows {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T] (s : Fin σ.sorts)
    (f : T s → Except Error α) (a : Arena α) : Bool :=
  a.levels.toList.zipIdx.all fun (level, r) => (List.range level.tag.size).all fun k =>
    (a.decodeAt T a.levels.size r k s).any (f · |>.toBool)

/-- Checks a fold against its function on named values of sort `s`, under each array of inputs.
On the host the fold must give the result of the function. With `device`, the checks of
`Folds.checkRun` must pass, with the schedule finishing exactly when the function has a result on
the value of every row. -/
@[noinline, nospecialize] def check {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T]
    [Scalar α] [Inhabited α] (label : String) (device : Bool) (F : Fold σ α T) (s : Fin σ.sorts)
    (f : Reader α → T s → Except Error α) (values : List (String × T s))
    (inputs : List (Array (Array α))) : IO Unit := do
  for (name, t) in values do
    let a := Arena.encode (σ := σ) (T := T) t
    for xs in inputs do
      let read := Reader.ofArrays xs
      let label := s!"{label} {name} with {xs.size} inputs"
      if device then
        checkRun label F a xs (allRows s (f read) a) (some #[f read t])
      else
        ensure (same (F.eval read a) (f read t)) s!"{label}: the fold differs from its function"

/-- The inputs of the binary tree functions: none, a weight alone, a weight and entry `0` of
input `1`, and every entry that they read. -/
def treeInputs : List (Array (Array Float)) :=
  [#[], #[#[0.75]], #[#[0.75], #[2.0]], #[(Array.range 8).map (·.toFloat * 0.5), #[2.0]]]

/-- The environments of the expressions and terms: none, the first four variables, and all of
them. -/
def environments : List (Array (Array Float)) :=
  [#[], #[(environment (α := Float)).extract 0 4], #[environment]]

/-- Checks the fold of every function against the function, on the host or on the device. -/
def checkAll (device : Bool) : IO Unit := do
  let trees := Folds.btrees (α := Float)
  check "size" device sizeFold 0 (fun _ t => .ok (size t)) trees [#[]]
  check "sum" device sumFold 0 (fun _ t => .ok (sum t)) trees [#[]]
  check "height" device heightFold 0 (fun _ t => .ok (height t)) trees [#[]]
  check "height'" device heightFold' 0 (fun _ t => .ok (height' t)) trees [#[]]
  check "nodes" device nodesFold 0 (fun _ t => .ok (nodes t)) trees [#[]]
  check "countAbove" device countAboveFold 0 (fun _ t => .ok (countAbove t)) trees [#[]]
  check "scaled" device scaledFold 0 (fun _ t => .ok (scaled 1.5 t)) trees [#[]]
  check "mixed" device mixedFold 0 (fun _ t => .ok (mixed t)) trees [#[]]
  check "classes" device classesFold 0 (fun _ t => .ok (classes t)) trees [#[]]
  check "quadrants" device quadrantsFold 0 (fun _ t => .ok (quadrants t)) trees [#[]]
  check "thresholds" device thresholdsFold 0 (fun _ t => .ok (thresholds t)) trees [#[]]
  check "taller" device tallerFold 0 (fun _ t => .ok (taller t)) trees [#[]]
  check "dependent" device dependentFold 0 (fun _ t => .ok (dependent t)) trees [#[]]
  check "matched" device matchedFold 0 (fun _ t => .ok (matched t)) trees [#[]]
  check "below" device belowFold 0 (fun _ t => .ok (below 0.5 t)) trees [#[]]
  check "scaledOf" device (scaledOf 2.5) 0 (fun _ t => .ok (scaled 2.5 t)) trees [#[]]
  check "root" device rootFold 0 (fun _ t => .ok (root t)) trees [#[]]
  check "checked" device checkedFold 0 (fun _ t => checked t) trees [#[]]
  check "named" device namedFold 0 (fun _ t => .ok (named t)) trees [#[]]
  check "weighted" device weightedFold 0 weighted trees treeInputs
  check "guarded" device guardedFold 0 guarded trees treeInputs
  let w : Float32 := moderate 3
  check "sum32" device sum32Fold 0 (fun _ t => .ok (sum32 t)) Folds.btrees [#[]]
  check "weighted32" device weighted32Fold 0 weighted32 Folds.btrees [#[], #[#[w]]]
  let v : ExecFloat.Binary 5 10 := moderate 4
  check "sumSquares16" device sumSquares16Fold 0 (fun _ t => .ok (sumSquares16 t)) Folds.btrees
    [#[]]
  check "height16" device height16Fold 0 (fun _ t => .ok (height16 t)) Folds.btrees [#[]]
  check "weighted16" device weighted16Fold 0 weighted16 Folds.btrees
    [#[], #[#[v]], #[#[v, moderate 5]]]
  check "height32" device height32Fold 0 (fun _ t => .ok (height32 t)) Folds.btrees [#[]]
  check "signed16" device signed16Fold 0 (fun _ t => .ok (signed16 t)) Folds.btrees [#[]]
  check "weight" device Mixed.weightFold 0 (fun _ t => .ok (Mixed.weight t)) Mixed.values [#[]]
  check "eval" device Ast.evalFold 0 Ast.eval asts environments
  check "short" device Ast.shortFold 0 Ast.short asts environments
  check "termEval" device termFold 0 termEval (Folds.terms (α := Float)) environments

/-- The host checks: the fold of every function against the function, the fold of the term
evaluator against the fold written by hand, and the sources of two folds that read inputs. -/
def run : IO Unit := do
  checkAll false
  for (name, t) in Folds.terms (α := Float) do
    for xs in environments do
      let read := Reader.ofArrays xs
      ensure (allSame (termFold.results read (arenaT t) 0)
          (Term.evalFold.results read (arenaT t) 0))
        s!"term {name}: the fold of termEval differs from the fold written by hand"
  checkSource "guarded" guardedFold
  checkSource "weighted16" weighted16Fold
  IO.println "  folds from functions: trees, expressions and terms passed on the host"

/-- The device checks: the fold of every function against the function, on the device. -/
def runCUDA : IO Unit := do
  checkAll true
  IO.println "  folds from functions: device folds match their functions"

/-! ### Rejected functions -/

/-- A function with a recursive call on a leaf rather than a child. -/
def badCall : BTree Float → Float
  | .leaf => 0
  | .node l _ _ => badCall l + badCall .leaf
termination_by t => t.toList.length
decreasing_by all_goals simp [BTree.toList]

/--
error: Fold.of: in the case BTree.node, the recursive call
  badCall BTree.leaf
must pass the same leading arguments and a child of the constructor
-/
#guard_msgs in
example := Fold.of badCall

/-- A function that uses a child as a value. -/
def badChild : BTree Float → Float
  | .leaf => 0
  | .node l _ r => badChild l + r.toList.length.toFloat

/--
error: Fold.of: in the case BTree.node, the child r is used as a value in
  r.toList
Children may only be passed to recursive calls.
-/
#guard_msgs in
example := Fold.of badChild

/-- A function with an operation outside the expression language. -/
def badOp : BTree Float → Float
  | .leaf => 0
  | .node l x r => badOp l + Float.sin x + badOp r

/-- error: Fold.of: in the case BTree.node: unsupported computation x.sin -/
#guard_msgs in
example := Fold.of badOp

/-- A function whose result is not a scalar. -/
def badType : BTree Float → Nat
  | .leaf => 0
  | .node l _ r => badType l + 1 + badType r

/--
error: Fold.of: the function must return Float or Except Error Float, where Float is the scalar
type of BTree Float, got
  ℕ
-/
#guard_msgs in
example := Fold.of badType

/-- A function with a recursive call on a grandchild, through a nested match. -/
def badGrandchild : BTree Float → Float
  | .leaf => 0
  | .node l _ r => badGrandchild l + match r with
    | .leaf => 0
    | .node rl _ _ => badGrandchild rl

/--
error: Fold.of: no equation of badGrandchild covers the whole case
  badGrandchild (l.node x r)
The function must match on the constructor only, and children may only be passed to recursive
calls.
-/
#guard_msgs in
example := Fold.of badGrandchild

/-- A function whose recursive call changes a leading argument. -/
def badLead (c : Float) : BTree Float → Float
  | .leaf => c
  | .node l _ r => badLead (c * 2) l + badLead c r

/--
error: Fold.of: in the case BTree.node, the recursive call
  badLead (1 * 2) l
must pass the same leading arguments and a child of the constructor
-/
#guard_msgs in
example := Fold.of (badLead 1)

/-- A function that reads an input whose number is computed. -/
def badRead (read : Reader Float) : BTree Float → Except Error Float
  | .leaf => .ok 0
  | .node l x r => do
    let a ← badRead read l
    let w ← read x.toUInt64.toNat 0
    let b ← badRead read r
    return a + x * w + b

/--
error: Fold.of: in the case BTree.node, the read
  read x.toUInt64.toNat 0
must use a literal input number
-/
#guard_msgs in
example := Fold.of badRead

/--
error: Fold.of: expected Fold BTree.sig Float32 fun x ↦ BTree Float32, produced Fold BTree.sig
Float fun s ↦ BTree Float
-/
#guard_msgs in
example : Fold BTree.sig Float32 (fun _ => BTree Float32) := Fold.of size

/-- A function that is not recursive. -/
def notRecursive (t : BTree Float) : Float := t.toList.length.toFloat

/--
error: Fold.of: we expected a recursive definition or a definition by cases on the tree, got
  notRecursive t
-/
#guard_msgs in
example := Fold.of notRecursive

/-- A function with `max`, which the expression language lacks. -/
def badMax : BTree Float → Float
  | .leaf => 0
  | .node l _ r => max (badMax l) (badMax r) + 1

/--
error: Fold.of: in the case BTree.node: unsupported computation instMaxFloat.1 l_result r_result
-/
#guard_msgs in
example := Fold.of badMax

mutual

/-- The number of nodes of a rose tree. -/
def roseSize : Rose Float → Float
  | .node _ cs => 1 + roseSizes cs

/-- The number of nodes of a list of rose trees. -/
def roseSizes : List (Rose Float) → Float
  | [] => 0
  | c :: cs => roseSize c + roseSizes cs

end

/-- error: Fold.of: the list field cs of Rose.node is not supported -/
#guard_msgs in
example := Fold.of roseSize

/-- Binary trees without a derived signature. -/
inductive Plain where
  | leaf
  | node (a b : Plain)

/-- The number of leaves of a tree without a derived signature. -/
def Plain.leaves : Plain → Float
  | .leaf => 1
  | .node a b => a.leaves + b.leaves

/-- error: Fold.of: Plain has no derived signature. Derive `Sig.Inductive` for it. -/
#guard_msgs in
example := Fold.of Plain.leaves

/-- A function that reads input `256`. -/
def bigInput (read : Reader Float) : BTree Float → Except Error Float
  | .leaf => read 256 0
  | .node l x r => do return (← bigInput read l) + x + (← bigInput read r)

/--
error: Fold.of: in the case BTree.leaf, the read
  read 256 0
uses an input number of 256 or more, which we do not support
-/
#guard_msgs in
example := Fold.of bigInput

/-- A function that throws. -/
def thrower : BTree Float → Except Error Float
  | .leaf => .ok 0
  | .node l x r => do
    let a ← thrower l
    if x < 0 then throw (.bounds 0 0 0) else return a + x + (← thrower r)

/--
error: Fold.of: in the case BTree.node, the function throws
  throw (Error.bounds 0 0 0)
A fold may fail only where it reads an input or the result of a child.
-/
#guard_msgs in
example := Fold.of thrower

mutual
/-- Forests of a mutual block. -/
inductive Forest where
  | nil
  | cons (t : Tree₁) (f : Forest)
  deriving Sig.Inductive

/-- Trees of a mutual block. -/
inductive Tree₁ where
  | node (x : Float) (f : Forest)
  deriving Sig.Inductive
end

mutual
/-- The sum of the scalars of a forest. -/
def Forest.total : Forest → Float
  | .nil => 0
  | .cons t f => t.total + f.total

/-- The sum of the scalars of a tree of the mutual block. -/
def Tree₁.total : Tree₁ → Float
  | .node x f => x + f.total
end

/-- error: Fold.of: Forest is part of a mutual block, which we do not support -/
#guard_msgs in
example := Fold.of Forest.total

/-- A function with the tree before a scalar argument. -/
def treeFirst : BTree Float → Float → Float
  | .leaf, c => c
  | .node l x r, c => treeFirst l c + x + treeFirst r c

/--
error: Fold.of: the tree must be the last argument of the function, got
  BTree Float → Float → Float
-/
#guard_msgs in
example := Fold.of treeFirst

/-- A function over a scalar type without a `Scalar` instance. -/
def sumNat : BTree Nat → Nat
  | .leaf => 0
  | .node l x r => sumNat l + x + sumNat r

/-- error: Fold.of: the scalar type ℕ of BTree ℕ has no `Scalar` instance -/
#guard_msgs in
example := Fold.of sumNat

end TorchLean.Test.Tree.Elaborated
