/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLeanExamples.Tree.Unfolds
public import TorchLeanTest.Tree.Hylo

/-!
# Unfolds

The checks run the unfolds of `TorchLeanExamples.Tree.Unfolds` from batches of seeds and compare
every value with an independent construction by its full encoding. The balanced tree of a range
is built from the list of its entries by halving the list, and the binomial tree of order `k + 1`
by linking two binomial trees of order `k` (Vuillemin 1978): the root of one gets the other as its
first child. The input of the ranges starts with the special encodings of the format, so the
nodes hold zeros, subnormals, infinities and NaNs as they were read.

On the host `Unfold.build` must give the arena of the expected values at a capacity of exactly
their number of nodes, and fail one row below it, and the values decoded from the arena and the
results of `Unfold.results` must be the expected values, in binary64, binary32 and a 12-bit
format. A range past the end of its input, or without an input, fails with its first missing
entry, and the other ranges of its batch keep their values. An empty batch builds an arena without
levels. The sums and the weighted sums of the balanced trees and the sums of the binomial trees,
as folds over the built arena, must be those of the expected values, and so must the weighted
sums without the weight, which fail on every tree with a node.

On the device `Unfold.buildDevice` must give the arena of `Unfold.build` at the capacity
`min capacity Expansion.deviceCapacity` bit for bit, or fail with it, and `Unfold.run` must give
the expected values, failures included. The folds run over the arena that `Unfold.buildDevice`
keeps in device memory: `Step.runDevice` must give the folds of the expected values there, or give
up when one of them fails, and `Fold.run` must give the folds of the expected values. The checks
run in binary64, binary32, binary16 and configured 12-bit, 17-bit, 65-bit and 256-bit formats, and
in binary64 also for a range of `2 ^ 20` entries and a binomial tree of order 20. Seeds without the
slots of the unfold are refused. Node kernels that read the input on every node, leaves included,
must give up on the last level when its last leaf reads past the end of the input, and give the
host columns otherwise, and so must node columns of tags alone.

Without LibTorch `Unfold.run` must give the expected values on the host.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Unfolds

open Kernel Examples.Tree Arenas Folds Hylos

variable {α : Type}

/-- The balanced binary tree of a list: its middle entry at the root, the tree of the entries
before it on the left and the tree of the entries after it on the right. -/
def balanced (xs : List α) : BTree α :=
  if h : xs.length / 2 < xs.length then
    .node (balanced (xs.take (xs.length / 2))) xs[xs.length / 2]
      (balanced (xs.drop (xs.length / 2 + 1)))
  else .leaf
termination_by xs.length
decreasing_by
  all_goals
    simp only [List.length_take, List.length_drop]
    omega

/-- The binomial tree of order `k` in arithmetic `A` with the scalar `x` at its root, by linking:
the tree of order `k + 1` is the tree of order `k` whose root gets the tree of order `k` of the
scalar `x * c` as its first child. -/
def linked (A : Arithmetic α) (c : α) : Nat → α → Rose α
  | 0, x => .node x []
  | k + 1, x =>
    match linked A c k x with
    | .node y cs => .node y (linked A c k (A.binary .mul x c) :: cs)

/-- Whether two arrays of values have the same size and, entry by entry, the same failure or
values of the same sort with the same arena, the scalars by their encodings. -/
def sameValues [Scalar α] [Inhabited α] {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T]
    (xs ys : Array (Except Error (Σ i, T i))) : Bool :=
  xs.size == ys.size && (xs.zip ys).all fun
    | (.ok x, .ok y) => x.1 == y.1 &&
      sameLevels (Arena.ofValues (σ := σ) (T := T) [x]).levels.toList
        (Arena.ofValues (σ := σ) (T := T) [y]).levels.toList
    | (.error e, .error e') => e == e'
    | _ => false

/-- Checks an unfold from the seeds `roots` with the arrays `inputs` as the inputs, at the
capacity `capacity`. `Unfold.build` must succeed when `builds` holds and fail otherwise, with the
arena of `expected` when it succeeds, and the decoded values and `Unfold.results` must be
`expected`. With `device`, the build is at the capacity `min capacity Expansion.deviceCapacity`,
`Unfold.buildDevice` must give its arena bit for bit or fail with it, and `Unfold.run` must give
`expected`. -/
@[noinline, nospecialize] def checkUnfold {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T]
    [Inhabited α] {m s : Nat} (device : Bool) (label : String) (U : Unfold σ α T m s)
    (capacity : Nat) (roots : Arena.Level α) (inputs : Array (Array α)) (builds : Bool)
    (expected : Array (Except Error (Σ i, T i))) : IO Unit :=
  letI := U.scalar
  do
    let read := Reader.ofArrays inputs
    let bound := if device then min capacity Expansion.deviceCapacity else capacity
    let host := U.build read bound roots
    ensure (host.isSome == builds)
      s!"{label}: the build {if builds then "failed" else "finished"}"
    if let some a := host then
      let some xs := expected.mapM Except.toOption
        | throw <| IO.userError s!"{label}: the build finished with a failing value"
      ensure (sameLevels a.levels.toList
          (Arena.ofValues (σ := σ) (T := T) xs.toList).levels.toList)
        s!"{label}: the built arena is not the arena of the expected values"
      ensure (sameValues (U.decodeRoots read roots a) expected)
        s!"{label}: the decoded values differ from the expected values"
    ensure (sameValues (U.results read capacity roots) expected)
      s!"{label}: the results differ from the expected values"
    if device then
      let built ← U.buildDevice capacity roots inputs
      ensure (built.isSome == builds)
        s!"{label}: the device build {if builds then "failed" else "finished"}"
      if let (some a, some d) := (host, built) then
        ensure (sameLevels a.levels.toList (← d.download).levels.toList)
          s!"{label}: the device arena differs from the host build"
      ensure (sameValues (← U.run capacity roots inputs) expected)
        s!"{label}: the values of run differ from the expected values"

/-- Checks a fold over the arena of an unfold from the seeds `roots` with the arrays `inputs` as
the inputs: over the arena of `Unfold.build` the fold with the arrays `foldInputs` must give
`expected`, and with `device` so must `Fold.run` over the arena of `Unfold.buildDevice` in device
memory, and `Step.runDevice` there, which must give up exactly when a result of `expected`
fails. -/
@[noinline, nospecialize] def checkFused {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T]
    [Inhabited α] {m s : Nat} (device : Bool) (label : String) (U : Unfold σ α T m s)
    (F : Fold σ α T) (hs : U.scalar = F.scalar) (capacity : Nat) (roots : Arena.Level α)
    (inputs foldInputs : Array (Array α)) (expected : Array (Except Error α)) : IO Unit :=
  letI := U.scalar
  do
    let some a := U.build (Reader.ofArrays inputs) capacity roots
      | throw <| IO.userError s!"{label}: the build failed"
    ensure (allSame (F.results (Reader.ofArrays foldInputs) a 0) expected)
      s!"{label}: the fold over the built arena differs from the expected results"
    if device then
      let some d ← U.buildDevice capacity roots inputs
        | throw <| IO.userError s!"{label}: the device build failed"
      let folds := expected.all (· matches .ok _)
      let results ← F.runDevice (hs ▸ d) foldInputs
      ensure (results.isSome == folds)
        s!"{label}: the fold over the device arena {if folds then "gave up" else "finished"}"
      if let some vs := results then
        ensure (allSame (vs.map .ok) expected)
          s!"{label}: the fold over the device arena differs from the expected results"
      ensure (allSame (← F.run (hs ▸ d) foldInputs) expected)
        s!"{label}: the results of run over the device arena differ from the expected results"

/-- The values of some trees, of sort `0`. -/
def values {T : Type} (ts : Array T) : Array (Except Error (Σ _ : Fin 1, T)) :=
  ts.map fun t => .ok ⟨0, t⟩

/-- The checks for scalars of one type, with `device` also on the device: balanced trees of
ranges of an input at the capacity of their nodes and one row below it, past the end of the input
and without one, binomial trees at the capacity of their nodes and one row below it, an empty
batch, and folds over the built arenas, one of them failing. -/
@[noinline, nospecialize] def checkAll [Scalar α] [Inhabited α] (device : Bool) (label : String) :
    IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let zero : α := Numeric.Word.ofNatBits 0
  let xs : Array α :=
    (specialScalars ++ (List.range 10000).map fun k => moderate (k + 77).toUInt64).toArray
  let n := xs.size
  let ranges : Array (Nat × Nat) :=
    #[(0, n), (0, 1), (5, 5), (9, 3), (1, n - 1), (17, 1000), (n - 1, n)]
  let rangeRoots (rs : Array (Nat × Nat)) : Arena.Level α :=
    roots rs.size #[rs.map (·.1.toUInt64), rs.map (·.2.toUInt64)] #[]
  let slice (lo hi : Nat) : BTree α := balanced ((xs.toList.drop lo).take (hi - lo))
  let trees := ranges.map fun (lo, hi) => slice lo hi
  let nodes := (ranges.map fun (lo, hi) => 2 * (hi - lo) + 1).foldl (· + ·) 0
  checkUnfold device s!"{label} balanced trees" ofRangeUnfold nodes (rangeRoots ranges) #[xs] true
    (values trees)
  checkUnfold device s!"{label} balanced trees below capacity" ofRangeUnfold (nodes - 1)
    (rangeRoots ranges) #[xs] false (values trees)
  checkUnfold device s!"{label} range past the end" ofRangeUnfold (2 ^ 20)
    (rangeRoots #[(0, 10), (n - 2, n + 3)]) #[xs] false
    #[.ok ⟨0, slice 0 10⟩, .error (.bounds 0 n.toUInt64 n)]
  checkUnfold device s!"{label} range without an input" ofRangeUnfold (2 ^ 20)
    (rangeRoots #[(4, 4), (0, 3)]) #[] false #[.ok ⟨0, .leaf⟩, .error (.bounds 0 0 0)]
  checkUnfold device s!"{label} empty batch" ofRangeUnfold 0 (rangeRoots #[]) #[xs] true #[]
  let c : α := moderate 11
  let orders : Array Nat := #[0, 1, 2, 5, 10, 12]
  let seeds : Array α := orders.map fun k => moderate (k + 900).toUInt64
  let binomialRoots := roots orders.size #[orders.map (·.toUInt64)] #[seeds]
  let rows := (orders.map (2 ^ ·)).foldl (· + ·) 0
  let roses := (orders.zip seeds).map fun (k, x) => linked A c k x
  checkUnfold device s!"{label} binomial trees" (binomialUnfold c) rows binomialRoots #[] true
    (values roses)
  checkUnfold device s!"{label} binomial trees below capacity" (binomialUnfold c) (rows - 1)
    binomialRoots #[] false (values roses)
  checkFused device s!"{label} sums of balanced trees" ofRangeUnfold (BTree.sumFold zero) rfl
    (2 ^ 20) (rangeRoots ranges) #[xs] #[] (trees.map fun t => .ok (BTree.sum A zero t))
  let w : α := moderate 3
  checkFused device s!"{label} weighted sums of balanced trees" ofRangeUnfold
    (BTree.weightedFold zero) rfl (2 ^ 20) (rangeRoots ranges) #[xs] #[#[w]]
    (trees.map (BTree.weighted A zero (Reader.ofArrays #[#[w]])))
  checkFused device s!"{label} weighted sums of balanced trees without the weight" ofRangeUnfold
    (BTree.weightedFold zero) rfl (2 ^ 20) (rangeRoots ranges) #[xs] #[]
    (trees.map (BTree.weighted A zero (Reader.ofArrays #[])))
  checkFused device s!"{label} sums of binomial trees" (binomialUnfold c) Rose.sumFold rfl
    (2 ^ 20) binomialRoots #[] #[] (roses.map fun t => .ok (Rose.sum A t))

/-- The host checks: the cases of `checkAll` in three formats. -/
def run : IO Unit := do
  checkAll (α := Float) false "binary64"
  checkAll (α := Float32) false "binary32"
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 6) false "12-bit"
  IO.println "  tree unfolds: balanced and binomial trees and folds over them passed on the host"

/-- The checks of builds without LibTorch: `Unfold.run` gives the expected values on the host, for
balanced trees of ranges of an input at the capacity of their nodes, one row below it and above
`Expansion.deviceCapacity`, for a range past the end of the input, and for a binomial tree. -/
def runUnlinked : IO Unit := do
  let A : Arithmetic Float := Scalar.arithmetic
  let xs : Array Float := (Array.range 100).map fun k => (k % 7).toFloat
  let n := xs.size
  let ranges : Array (Nat × Nat) := #[(0, n), (5, 5), (17, 60)]
  let r := roots ranges.size #[ranges.map (·.1.toUInt64), ranges.map (·.2.toUInt64)] #[]
  let trees := values (ranges.map fun (lo, hi) => balanced ((xs.toList.drop lo).take (hi - lo)))
  let nodes := (ranges.map fun (lo, hi) => 2 * (hi - lo) + 1).foldl (· + ·) 0
  for capacity in [nodes, nodes - 1, 2 ^ 40] do
    ensure (sameValues (σ := BTree.sig) (← ofRangeUnfold.run capacity r #[xs]) trees)
      s!"balanced trees at capacity {capacity}: the values of run differ from the trees"
  let past := roots 2 #[#[0, (n - 2).toUInt64], #[10, (n + 3).toUInt64]] #[]
  ensure (sameValues (σ := BTree.sig) (← ofRangeUnfold.run (2 ^ 20) past #[xs])
      #[.ok ⟨0, balanced (xs.toList.take 10)⟩, .error (.bounds 0 n.toUInt64 n)])
    "range past the end: the values of run differ from the expected values"
  let order : Arena.Level Float := roots 1 #[#[10]] #[#[0.5]]
  ensure (sameValues (σ := Rose.sig) (← (binomialUnfold 1.0).run (2 ^ 10) order #[])
      (values #[linked A 1.0 10 0.5]))
    "binomial tree: the values of run differ from the binomial tree"
  IO.println "  tree unfolds: run builds the values on the host without LibTorch"

/-- The device check of node kernels on the last level, for the balanced trees of ranges of an
input whose every node, leaves included, holds the entry at the start of its range. On the range
`[n - 7, n)` of an input of `n` entries all the leaves are on the last level and the last one reads
entry `n`, so `Expansion.buildHandle` must give up as the host does, and on `[n - 8, n - 1)` it must
give the host node columns bit for bit. With node columns of tags alone both ranges must give the
host tags. -/
@[noinline, nospecialize] def checkLastLevel [Scalar α] [Inhabited α] (label : String) :
    IO Unit := do
  let U := ofRangeUnfold (α := α)
  let xs : Array α := (Array.range 100).map fun k => moderate (k + 1).toUInt64
  let n := xs.size
  let read := Reader.ofArrays #[xs]
  let first : Expr α (Row 2 0) .scalar := .load 0 (.var (Row.indexVar 0))
  for (lo, hi, builds) in [(n - 7, n, false), (n - 8, n - 1, true)] do
    let r := roots 1 #[#[lo.toUInt64], #[hi.toUInt64]] #[]
    let range := s!"{label} [{lo}, {hi})"
    let levels := U.expand read (2 ^ 10) r
    let host := levels.bind fun ls => ls.mapM fun level => do
      let tag ← U.column read level U.tag
      let index ← (Array.finRange BTree.sig.indexSlots).mapM fun i =>
        U.column read level (U.nodeIndex i)
      let x ← U.column read level first
      return (⟨tag, index, #[x]⟩ : Arena.Level α)
    ensure (host.isSome == builds)
      s!"{range}: the host nodes {if builds then "failed" else "were built"}"
    let built ← U.buildHandle (b := 1) (2 ^ 10) r #[xs] U.tag U.nodeIndex fun _ => first
    ensure (built.isSome == builds)
      s!"{range}: the device nodes {if builds then "failed" else "were built"}"
    if let (some hs, some (handle, sizes)) := (host, built) then
      let a ← (Arena.Device.ofHandle (α := α) handle sizes BTree.sig.indexSlots 1).download
      ensure (sameLevels a.levels.toList hs) s!"{range}: the device nodes differ from the host"
    let tags := levels.bind fun ls => ls.mapM fun level =>
      (U.column read level U.tag).map fun tag => (⟨tag, #[], #[]⟩ : Arena.Level α)
    let some ts := tags | throw <| IO.userError s!"{range}: the host tags failed"
    let some (handle, sizes) ← U.buildHandle (a := 0) (b := 0) (2 ^ 10) r #[xs] U.tag
        (fun i => i.elim0) (fun i => i.elim0)
      | throw <| IO.userError s!"{range}: the device tags failed"
    let a ← (Arena.Device.ofHandle (α := α) handle sizes 0 0).download
    ensure (sameLevels a.levels.toList ts) s!"{range}: the device tags differ from the host"

/-- The device checks: seeds of other slots are refused, the cases of `checkAll` in seven formats,
node kernels on the last level, and in binary64 a balanced tree of `2 ^ 20` entries, a binomial
tree of order 20 and their sums. -/
def runCUDA : IO Unit := do
  ensureFails "seeds of other slots" "slots of the expansion"
    ((ofRangeUnfold (α := Float)).buildDevice 10 (roots 2 #[#[1]] #[]) #[])
  ensureFails "seeds with scalar slots" "slots of the expansion"
    ((ofRangeUnfold (α := Float)).buildDevice 10 (roots 1 #[#[0], #[1]] #[#[1.0]]) #[])
  checkAll (α := Float) true "binary64"
  checkAll (α := Float32) true "binary32"
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 10) true "binary16"
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 6) true "12-bit"
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 5 11) true "17-bit"
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 11 53) true "65-bit"
  checkAll (α := FloatLib.Floats.ExecFloat.Binary 15 240) true "256-bit"
  checkLastLevel (α := Float) "binary64 last level"
  checkLastLevel (α := FloatLib.Floats.ExecFloat.Binary 5 6) "12-bit last level"
  let A : Arithmetic Float := Scalar.arithmetic
  let n := 2 ^ 20
  let xs : Array Float := (Array.range n).map fun k => (k % 7).toFloat
  let t := balanced xs.toList
  let big : Arena.Level Float := roots 1 #[#[0], #[n.toUInt64]] #[]
  checkUnfold true "binary64 balanced tree of 2 ^ 20 entries" ofRangeUnfold (2 * n + 1) big #[xs]
    true (values #[t])
  checkFused true "binary64 sum of a balanced tree of 2 ^ 20 entries" ofRangeUnfold
    (BTree.sumFold 0.0) rfl (2 * n + 1) big #[xs] #[] #[.ok (BTree.sum A 0.0 t)]
  let r := linked A 1.0 20 0.5
  let order : Arena.Level Float := roots 1 #[#[20]] #[#[0.5]]
  checkUnfold true "binary64 binomial tree of order 20" (binomialUnfold 1.0) (2 ^ 20) order #[]
    true (values #[r])
  checkFused true "binary64 sum of a binomial tree of order 20" (binomialUnfold 1.0) Rose.sumFold
    rfl (2 ^ 20) order #[] #[] #[.ok (2 ^ 19 : Float)]
  IO.println "  tree unfolds: device arenas match the host builds bit for bit, and folds over them \
    in device memory the folds of the values"

end TorchLean.Test.Tree.Unfolds
