/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.LibTorch.Controls
public import TorchLean.CUDA.Tree.Run
public import TorchLeanExamples.Tree.Folds
public import TorchLeanTest.Tree.Derive

/-!
# Folds over tree arenas

The checks run the folds of `TorchLeanExamples.Tree.Folds` over binary trees, rose trees and
arithmetic terms, and compare each result with the reference function by its full encoding.
Lean's `Float` and `Float32` hold a single NaN, so for them any two NaNs are the same result. Most
scalars have exponents near zero, so that most sums and products stay finite, and spines of
special encodings bring in zeros, subnormals, infinities and NaNs.

On the host the schedule runs in binary64, binary32 and a 12-bit format. A reader that fails gives
its failure at the root, every root of a batch gets the result of its tree, a variable past the
end of its environment fails with its index, and an arena whose rows share a child gets the result
of the tree it decodes to. A tag that names no constructor runs the last case, and columns
shorter than the tags give zeros and defaults, as `Fold.select` and `Arena.Level.env` say. The
rows of a level cut into runs of any length, as compiled code computes them in parallel, are the
rows computed one at a time, on the levels of a pseudorandom tree and on levels of sizes around
the run boundaries. The source of each fold is a row kernel with both guards, one column per slot
of a row, the inputs of the fold and the results below.

On the device each fold runs one row kernel per level, in binary64, binary32, binary16 and
configured 12-bit, 17-bit, 65-bit and 256-bit formats. `Step.runDevice` must finish on the device
with the results of the host schedule, or give up when a read fails on the host, and `Fold.run`
must give the host results in both cases, failures included: a missing weight, an empty weight
array, a variable past the end of its environment or at `2 ^ 64 - 1`, child rows past the next
level or at `2 ^ 63` and `2 ^ 64 - 1`, and rose lists that run past the next level, start at
`2 ^ 63` or `2 ^ 64 - 1` or count that many children. The results must also be those of the
reference functions in every format. A row that fails but that no root reaches, under a
level with no rows or not, leaves the root its result. A fold over an arena of other slots is
refused, and so is, with two devices visible, a fold over an arena on the other device.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Folds

open Kernel Examples.Tree Arenas

variable {α : Type}

/-- Whether a scalar is a NaN, read from its encoding. -/
def isNaN [Scalar α] (x : α) : Bool :=
  let (e, f) := fieldWidths (Scalar.precision α)
  bits x / 2 ^ f % 2 ^ e == 2 ^ e - 1 && bits x % 2 ^ f != 0

/-- Whether two results are the same failure or scalars with the same encoding. Two NaNs of a
native format are the same, since Lean's `Float` and `Float32` hold one NaN, so a NaN result in a
native format pins less than its encoding would. The configured formats compare every bit. -/
def same [Scalar α] : Except Error α → Except Error α → Bool
  | .ok x, .ok y =>
    bits x == bits y || (Scalar.precision α matches .native _) && isNaN x && isNaN y
  | .error e, .error e' => e == e'
  | _, _ => false

/-- Whether two arrays of results have the same size and, entry by entry, the same results. -/
def allSame [Scalar α] (xs ys : Array (Except Error α)) : Bool :=
  xs.size == ys.size && (xs.zip ys).all fun (x, y) => same x y

/-- A scalar with the sign and fraction of `scalarOf w` and a pseudorandom exponent from `-4` to
`3`. -/
def moderate [Scalar α] (w : UInt64) : α :=
  let (e, f) := fieldWidths (Scalar.precision α)
  let x := bits (scalarOf w : α)
  let exponent := 2 ^ (e - 1) - 5 + (BTree.mix w).toNat % 8
  Numeric.Word.ofNatBits (x / 2 ^ (e + f) % 2 * 2 ^ (e + f) + exponent * 2 ^ f + x % 2 ^ f)

/-- The special encodings of the format of `α` as scalars. -/
def specialScalars [Scalar α] : List α :=
  let (e, f) := fieldWidths (Scalar.precision α)
  (specials e f).map Numeric.Word.ofNatBits

/-- The reader whose input `0` holds `w` alone. -/
def weight (w : α) : Reader α := fun op i =>
  if op = 0 ∧ i = 0 then .ok w else .error (.bounds op i (if op = 0 then 1 else 0))

/-- The reader that fails on every read. -/
def failing : Reader α := fun op _ => .error (.input op)

/-- The binary trees of the checks: a leaf, a perfect tree, a deep caterpillar, a pseudorandom
tree and a spine of special encodings. -/
def btrees [Scalar α] : List (String × BTree α) :=
  [("leaf", .leaf), ("perfect 12", .perfect (moderate 7) 12),
    ("caterpillar 2000", .caterpillar (moderate 9) 2000),
    ("random 20000", .random moderate 20000 1), ("specials", spine specialScalars)]

/-- The rose trees of the checks: a single node, a pseudorandom tree, a node with 300 children
and a node whose children hold the special encodings. -/
def roses [Scalar α] : List (String × Rose α) :=
  [("single", .node (moderate 1) []), ("random 20000", .random moderate 20000 2),
    ("wide", .node (moderate 2) ((List.range 300).map fun i => .node (moderate i.toUInt64) [])),
    ("specials", .node (moderate 3) (specialScalars.map (.node · [])))]

/-- The terms of the checks over an environment of 64 variables: a literal, a variable,
pseudorandom terms, a chain of 2000 sums of variables, and sums and products of the special
encodings. -/
def terms [Scalar α] : List (String × Term α) :=
  [("literal", .lit (moderate 4)), ("variable", .var 63),
    ("random 3000", .random moderate 64 3000 3), ("random 30000", .random moderate 64 30000 4),
    ("chain 2000", (List.range 2000).foldl (fun t i => .add t (.var (i % 64).toUInt64)) (.var 0)),
    ("special sums", specialScalars.foldl (fun t x => .add t (.lit x)) (.lit (moderate 5))),
    ("special products", specialScalars.foldl (fun t x => .mul (.lit x) t) (.var 1))]

/-- The environment of the terms: 64 scalars. -/
def environment [Scalar α] : Array α := (Array.range 64).map fun i => moderate (i + 1000).toUInt64

/-- The arena of a term. -/
def arenaT [Inhabited α] (t : Term α) : Arena α :=
  Arena.encode (σ := Term.sig) (T := fun _ => Term α) (s := 0) t

/-- The arena of a batch of terms, one per row of level zero. -/
def batchT [Inhabited α] (ts : List (Term α)) : Arena α :=
  Arena.ofValues (σ := Term.sig) (T := fun _ => Term α) (ts.map (⟨0, ·⟩))

/-- The host checks for scalars of one type: the binary tree folds with a weight and with a
failing reader, the sums of every root of a batch, the rose tree sum, and terms over the
environment and past its end. -/
@[noinline, nospecialize] def checkTrees [Scalar α] [Inhabited α] (label : String) :
    IO Unit := do
  let zero : α := Numeric.Word.ofNatBits 0
  let A : Arithmetic α := Scalar.arithmetic
  let w : α := moderate 3
  for (name, t) in btrees (α := α) do
    let a := arenaB t
    ensure (same ((BTree.sumFold zero).eval failing a) (.ok (BTree.sum A zero t)))
      s!"{label} {name}: the sum differs from the reference"
    ensure (same ((BTree.weightedFold zero).eval (weight w) a)
        (BTree.weighted A zero (weight w) t))
      s!"{label} {name}: the weighted sum differs from the reference"
    let failed : Except Error α := if t matches .leaf then .ok zero else .error (.input 0)
    ensure (same ((BTree.weightedFold zero).eval failing a) failed)
      s!"{label} {name}: the weighted sum does not fail with the reader"
  let ts : List (BTree α) := [.leaf, .perfect (moderate 5) 4, .random moderate 300 23]
  let results := (BTree.sumFold zero).results failing (batchB ts) 0
  ensure (results.size == ts.length) s!"{label} batch: {results.size} results"
  for (t, k) in ts.zipIdx do
    ensure (same (results[k]?.getD (.error (.input 1))) (.ok (BTree.sum A zero t)))
      s!"{label} batch: root {k} does not get the sum of its tree"
  for (name, t) in roses (α := α) do
    ensure (same (Rose.sumFold.eval failing (arenaR t)) (.ok (Rose.sum A t)))
      s!"{label} rose {name}: the sum differs from the reference"
  let env := Reader.ofArrays #[environment (α := α)]
  for (name, t) in terms (α := α) do
    ensure (same (Term.evalFold.eval env (arenaT t)) (Term.eval A env t))
      s!"{label} term {name}: the value differs from the reference"
  let past : Term α := .add (.mul (.var 3) (.var 64)) (.var 65)
  ensure (same (Term.evalFold.eval env (arenaT past)) (.error (.bounds 0 64 64)))
    s!"{label}: a variable past the environment does not fail with its index"

/-- Checks that the source of a fold is a row kernel with both guards, a column read for each
slot of a row, the inputs of the fold and the results below as its inputs, and the output write.
-/
def checkSource {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T] (label : String)
    (F : Fold σ α T) : IO Unit := do
  match F.source with
  | .ok text =>
    let parts := ["__global__ void torchlean_kernel(", "if (index >= count) return;\n",
      CUDA.errorGuard, "output[index] = "] ++
      (List.range σ.row.length).map (s!"= column{·}[index];") ++
      (List.range (F.inputs + 1)).map fun i => s!"input{i}, unsigned long long size{i}"
    for part in parts do
      ensure (text.contains part) s!"{label}: the source lacks {part.quote}"
    ensure (!text.contains s!"column{σ.row.length}" && !text.contains s!"input{F.inputs + 1}")
      s!"{label}: the source has more columns or inputs than the fold"
  | .error e => throw <| IO.userError s!"{label}: no source: {e}"

/-- Checks that the rows of every level of a pseudorandom tree, cut into runs of 1, 2, 3, 7, 256
and `2 ^ 20` rows, are the rows computed one at a time. On levels of 0 to 4097 rows around the run
boundaries, whose row `k` has both children and its scalar at `k` over results `k`, runs of 1, 7,
256 and all rows must give `3 * k` at row `k`. -/
def checkRuns : IO Unit := do
  let F := BTree.weightedFold (0.0 : Float)
  let read := weight (moderate 3 : Float)
  let a := arenaB (BTree.random (moderate (α := Float)) 20000 1)
  for (level, r) in a.levels.toList.zipIdx do
    let below := F.results read a (r + 1)
    let rows := (Array.range level.tag.size).map fun k =>
      F.expr.eval F.arithmetic (Step.reader read F.inputs below)
        (level.env BTree.sig.indexSlots BTree.sig.scalarSlots k)
    for grain in [1, 2, 3, 7, 256, 2 ^ 20] do
      ensure (allSame (F.levelRuns grain read level below) rows)
        s!"runs of {grain} rows: level {r} differs from its rows one at a time"
  let G := BTree.sumFold (0.0 : Float)
  for n in [0, 1, 2, 255, 256, 257, 511, 512, 513, 1000, 4096, 4097] do
    let rows := (Array.range n).map (·.toUInt64)
    let nodes := level (Array.replicate n 1) #[rows, rows] #[rows.map (·.toFloat)]
    let below := (Array.range n).map fun k => .ok k.toFloat
    let expected := (Array.range n).map fun k => .ok (3 * k).toFloat
    ensure (allSame (G.level failing nodes below) expected) s!"a level of {n} rows differs"
    for grain in [1, 7, 256, n + 1] do
      ensure (allSame (G.levelRuns grain failing nodes below) expected)
        s!"runs of {grain} rows: a level of {n} rows differs"

/-- The host checks: an exact sum in binary64, the folds in three formats, an arena whose rows
share a child, levels cut into runs, the reader of arrays, and the sources of the folds. -/
def run : IO Unit := do
  ensure (same ((BTree.sumFold 0.0).eval failing (arenaB (.perfect (1.0 : Float) 12)))
    (.ok 4095.0)) "binary64: the perfect tree of ones does not sum to 4095"
  checkTrees (α := Float) "binary64"
  checkTrees (α := Float32) "binary32"
  checkTrees (α := FloatLib.Floats.ExecFloat.Binary 5 6) "12-bit"
  let leaves := level #[0, 0] #[#[0, 0], #[0, 0]] #[#[0, 0]]
  let shared : Arena Float := ⟨#[level #[1] #[#[0], #[0]] #[#[1.5]],
    level #[1] #[#[0], #[1]] #[#[2.5]], leaves]⟩
  ensure (same ((BTree.sumFold 0.0).eval failing shared) (.ok 6.5))
    "shared child: the sum is not that of the tree it decodes to"
  for tag in [2, 5, 0xFFFFFFFFFFFFFFFF] do
    ensure (same ((BTree.sumFold 0.0).eval failing ⟨#[level #[tag] #[#[0], #[1]] #[#[1.5]],
        leaves]⟩) (.ok 1.5)) s!"tag {tag}: the row does not run the last case"
  let short : List (String × Arena.Level Float × Float) := [
    ("scalar column", level #[1] #[#[0], #[1]] #[#[]], 0.0),
    ("index column", level #[1] #[#[0]] #[#[1.5]], 1.5),
    ("columns", level #[1] #[] #[], 0.0)]
  for (name, row, x) in short do
    ensure (same ((BTree.sumFold 0.0).eval failing ⟨#[row, leaves]⟩) (.ok x))
      s!"a short {name} does not give zeros and defaults"
  checkRuns
  for op in [0, 1, 2] do
    for i in [0, 1, 2, 0xFFFFFFFFFFFFFFFF] do
      ensure (same (Reader.ofArrays #[#[2.5]] op i) (weight 2.5 op i))
        s!"reader of arrays: input {op} at {i} is not the weight's"
  checkSource "binary64 sum" (BTree.sumFold (0.0 : Float))
  checkSource "binary32 weighted sum" (BTree.weightedFold (0.0 : Float32))
  checkSource "12-bit rose sum" (Rose.sumFold (α := FloatLib.Floats.ExecFloat.Binary 5 6))
  checkSource "256-bit term" (Term.evalFold (α := FloatLib.Floats.ExecFloat.Binary 15 240))
  IO.println "  tree folds: sums, weighted sums, rose sums and terms passed on the host"

/-- Checks a fold of an arena on the device against the host. The host schedule must succeed
when `succeeds` holds and fail otherwise, `Step.runDevice` must give its results or give up with
it, and `Fold.run` must give the results of `Step.results`, which must be `expected` if given. -/
@[noinline, nospecialize] def checkRun {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T]
    [Inhabited α] (label : String) (F : Fold σ α T) (a : Arena α) (inputs : Array (Array α))
    (succeeds : Bool) (expected : Option (Array (Except Error α)) := none) : IO Unit :=
  letI := F.scalar
  do
    let read := Reader.ofArrays inputs
    let host := F.schedule read a.levels.toList
    ensure (host.isSome == succeeds)
      s!"{label}: the host schedule {if succeeds then "fails" else "succeeds"}"
    let d ← Arena.upload σ a
    let device ← F.runDevice d inputs
    ensure (device.isSome == succeeds)
      s!"{label}: the device {if succeeds then "gave up" else "finished"}"
    if let (some vs, some ws) := (device, host) then
      ensure (allSame (vs.map .ok) (ws.map .ok))
        s!"{label}: the device results differ from the host schedule"
    ensure (allSame (← F.run d inputs) (F.results read a 0))
      s!"{label}: the results of run differ from the host results"
    if let some rs := expected then
      ensure (allSame (F.results read a 0) rs) s!"{label}: the results differ from the reference"

/-- The device checks for scalars of one type, each against the reference function: the binary
tree folds with and without a weight, batches, the rose tree sum, and terms over the environment
and past its end. -/
@[noinline, nospecialize] def checkDeviceFolds [Scalar α] [Inhabited α] (label : String) :
    IO Unit := do
  let zero : α := Numeric.Word.ofNatBits 0
  let A : Arithmetic α := Scalar.arithmetic
  let w : α := moderate 3
  for (name, t) in btrees (α := α) do
    let a := arenaB t
    checkRun s!"{label} {name} sum" (BTree.sumFold zero) a #[] true
      (some #[.ok (BTree.sum A zero t)])
    checkRun s!"{label} {name} weighted sum" (BTree.weightedFold zero) a #[#[w]] true
      (some #[BTree.weighted A zero (Reader.ofArrays #[#[w]]) t])
    checkRun s!"{label} {name} weighted sum without a weight" (BTree.weightedFold zero) a #[]
      (t matches .leaf) (some #[BTree.weighted A zero (Reader.ofArrays #[]) t])
  let ts : List (BTree α) := [.leaf, .perfect (moderate 5) 4, .random moderate 300 23]
  checkRun s!"{label} batch" (BTree.sumFold zero) (batchB ts) #[] true
    (some (ts.toArray.map (.ok <| BTree.sum A zero ·)))
  checkRun s!"{label} empty batch" (BTree.sumFold zero) (batchB ([] : List (BTree α))) #[] true
    (some #[])
  for (name, t) in roses (α := α) do
    checkRun s!"{label} rose {name}" Rose.sumFold (arenaR t) #[] true
      (some #[.ok (Rose.sum A t)])
  let env := #[environment (α := α)]
  for (name, t) in terms (α := α) do
    checkRun s!"{label} term {name}" Term.evalFold (arenaT t) env true
      (some #[Term.eval A (Reader.ofArrays env) t])
  let past : List (Term α) := [.random moderate 64 300 6, .random moderate 100 3000 5]
  checkRun s!"{label} terms past the environment" Term.evalFold (batchT past) env false
    (some (past.toArray.map (Term.eval A (Reader.ofArrays env))))
  let bare : List (Term α) := [.lit (moderate 6), .add (.var 0) (.lit (moderate 7))]
  checkRun s!"{label} terms without an environment" (Term.evalFold (α := α)) (batchT bare) #[]
    false (some (bare.toArray.map (Term.eval A (Reader.ofArrays #[]))))

/-- With two devices visible, checks that a fold refuses an arena on the other device than the
selected one, and that the arena downloads after the refusal. -/
def checkDevices : IO Unit := do
  if (← LibTorch.deviceCount) < 2 then
    IO.println "  tree folds: one device visible, the fold on another device is not checked"
    return
  let first ← LibTorch.getDevice
  let a := arenaB (BTree.perfect (1.5 : Float) 6)
  let d ← Arena.upload BTree.sig a
  LibTorch.setDevice (if first == 0 then 1 else 0)
  try
    ensureFails "fold on another device" "another device"
      ((BTree.sumFold (0.0 : Float)).runDevice d #[])
    ensure (decodeB (← d.download) == decodeB a) "another device: the arena does not download"
  finally
    LibTorch.setDevice first
  IO.println "  tree folds: a fold over an arena on another device is refused"

/-- The device checks: a fold over an arena of other slots is refused, an arena with a child row
past the next level and one whose rows share a child fold as on the host, a failing row that no
root reaches and a level without rows leave the root its result, the folds run in seven formats,
and a fold over an arena on another device is refused. -/
def runCUDA : IO Unit := do
  ensureFails "fold over keys" "other slots" do
    let d ← Arena.upload Derived.Keys.sig (Derived.arenaK64 (Derived.Keys.ofArray #[3, 1, 2]))
    (BTree.sumFold (0.0 : Float)).runDevice d #[]
  let leaves := level #[0, 0] #[#[0, 0], #[0, 0]] #[#[0, 0]]
  checkRun "child past the next level" (BTree.sumFold (0.0 : Float))
    ⟨#[level #[1] #[#[0], #[2]] #[#[1.5]], leaves]⟩ #[] false
  checkRun "shared child" (BTree.sumFold (0.0 : Float))
    ⟨#[level #[1] #[#[0], #[0]] #[#[1.5]], level #[1] #[#[0], #[1]] #[#[2.5]], leaves]⟩ #[] true
    (some #[.ok 6.5])
  -- Term rows: a literal root, a level without rows, and a variable that no root reaches.
  let root := level #[0] #[#[0], #[0]] #[#[7.0]]
  let lost := level #[1] #[#[0], #[0]] #[#[0.0]]
  let empty := level #[] #[#[], #[]] #[#[]]
  checkRun "unreachable failing row" Term.evalFold ⟨#[root, lost]⟩ #[] false
    (some #[.ok 7.0])
  checkRun "failing row under an empty level" Term.evalFold ⟨#[root, empty, lost]⟩ #[] false
    (some #[.ok 7.0])
  checkRun "empty level" Term.evalFold ⟨#[root, empty, root]⟩ #[] true
    (some #[.ok 7.0])
  checkRun "child at 2 ^ 64 - 1" (BTree.sumFold (0.0 : Float))
    ⟨#[level #[1] #[#[0], #[0xFFFFFFFFFFFFFFFF]] #[#[1.5]], leaves]⟩ #[] false
    (some #[.error (.bounds 0 0xFFFFFFFFFFFFFFFF 2)])
  checkRun "child at 2 ^ 63" (BTree.sumFold (0.0 : Float))
    ⟨#[level #[1] #[#[0x8000000000000000], #[0]] #[#[1.5]], leaves]⟩ #[] false
    (some #[.error (.bounds 0 0x8000000000000000 2)])
  -- Rose rows: a list from row `first` of `count` children over three leaves.
  let roseLeaves := level #[0, 0, 0] #[#[0, 0, 0], #[0, 0, 0]] #[#[10.0, 20.0, 30.0]]
  let lists : List (String × UInt64 × UInt64 × Except Error Float) := [
    ("to the end", 1, 2, .ok 51.0), ("one past the end", 1, 3, .error (.bounds 0 3 3)),
    ("from 2 ^ 64 - 1", 0xFFFFFFFFFFFFFFFF, 2, .error (.bounds 0 0xFFFFFFFFFFFFFFFF 3)),
    ("from 2 ^ 63", 0x8000000000000000, 1, .error (.bounds 0 0x8000000000000000 3)),
    ("of 2 ^ 64 - 1 children", 1, 0xFFFFFFFFFFFFFFFF, .error (.bounds 0 3 3)),
    ("of 2 ^ 63 children", 0, 0x8000000000000000, .error (.bounds 0 3 3))]
  for (name, first, count, result) in lists do
    checkRun s!"rose list {name}" Rose.sumFold
      ⟨#[level #[0] #[#[first], #[count]] #[#[1.0]], roseLeaves]⟩ #[] (result matches .ok _)
      (some #[result])
  let huge : Term Float := .add (.var 0xFFFFFFFFFFFFFFFF) (.lit 1.0)
  checkRun "variable 2 ^ 64 - 1" Term.evalFold (arenaT huge) #[environment] false
    (some #[.error (.bounds 0 0xFFFFFFFFFFFFFFFF 64)])
  checkRun "empty weight array" (BTree.weightedFold (0.0 : Float)) (arenaB (.perfect 1.5 3))
    #[#[]] false (some #[.error (.bounds 0 0 0)])
  checkRun "empty weight array, unread" (BTree.weightedFold (0.0 : Float)) (arenaB .leaf) #[#[]]
    true (some #[.ok 0.0])
  checkDevices
  checkDeviceFolds (α := Float) "binary64"
  checkDeviceFolds (α := Float32) "binary32"
  checkDeviceFolds (α := FloatLib.Floats.ExecFloat.Binary 5 10) "binary16"
  checkDeviceFolds (α := FloatLib.Floats.ExecFloat.Binary 5 6) "12-bit"
  checkDeviceFolds (α := FloatLib.Floats.ExecFloat.Binary 5 11) "17-bit"
  checkDeviceFolds (α := FloatLib.Floats.ExecFloat.Binary 11 53) "65-bit"
  checkDeviceFolds (α := FloatLib.Floats.ExecFloat.Binary 15 240) "256-bit"
  IO.println "  tree folds: device folds match the host schedule, failures through the host"

end TorchLean.Test.Tree.Folds
