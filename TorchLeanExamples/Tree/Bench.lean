/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Table
public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.Tree.Run
public import TorchLean.NN.Backend
public import TorchLeanExamples.Tree.Folds
public import TorchLeanExamples.Tree.Hylos
public import TorchLeanExamples.Tree.Unfolds

/-!
# Tree benchmark

`tree_bench` times the stages that take a pseudorandom binary tree of `Float` scalars to the
device and back: encoding its arena, taking the columns, packing the blocks, uploading, downloading
and decoding. Uploads include the columns and the packing, and downloads include reading the
blocks back. It then times two folds, the sum of the tree and the value of a pseudorandom term
whose variables are read from an environment: the recursive reference function, the schedule on
the host, and `Step.runDevice` on the uploaded arena, whose first run compiles the row kernel.
Next it times three hylomorphisms of `TorchLeanExamples.Tree.Hylos` with call trees of about the
same number of rows: the sum by halves of an input, a Fibonacci number and the midpoint rule. For
them the schedule on the host includes the expansion of the call tree, and `Hylo.runDevice`
expands it on the device and runs the step there. Then it times two unfolds with arenas of about
the same number of rows, each followed by a sum: the balanced tree of an input and a binomial
tree. For them it times the recursive reference function, the build on the host, and on the
device `Unfold.buildDevice` and the sum over the arena it leaves in device memory, next to the
encoding and upload of the reference value that the device build stands in for. Last it makes
the tables of FloatLib's correctly rounded `exp` in binary16 and `log` in bfloat16 with
`Table.make`, once, and times the function itself and the search of its table on the host on
pseudorandom words. With a device it times `Table.Of.runDevice` for each size, whose results must
be the function's bit for bit. After the sizes it times the fold `BTree.tableFold` of the binary16
table over a pseudorandom tree of `2 ^ 16` inner nodes: the sum of the tree mapped by `exp`, the
schedule on the host and, with a device, `Step.runDevice`, whose result must be the sum bit for
bit. With `--device cpu`, the default, only the host stages run.
-/

@[expose] public section

namespace TorchLean.Examples.Tree.Bench

open Kernel

/-- `x`, behind a call the compiler cannot see through, so that work on it starts after `_`. -/
@[noinline] def after {β : Type} (_ : Nat) (x : β) : β := x

/-- Does nothing with `_`, behind a call the compiler cannot see through, so that `_` is computed
before the call. -/
@[noinline] def touch {β : Type} (_ : β) : IO Unit := pure ()

/-- `f x` and the nanoseconds it took. -/
def timed {β γ : Type} (f : β → γ) (x : β) : IO (γ × Nat) := do
  let start ← IO.monoNanosNow
  let r := f (after start x)
  touch r
  let stop ← IO.monoNanosNow
  return (r, stop - start)

/-- The result of an action and the nanoseconds it took. -/
def timedIO {γ : Type} (action : IO γ) : IO (γ × Nat) := do
  let start ← IO.monoNanosNow
  let r ← action
  touch r
  let stop ← IO.monoNanosNow
  return (r, stop - start)

/-- Nanoseconds per row, with one decimal. -/
def perRow (ns n : Nat) : String :=
  let tenths := (10 * ns + n / 2) / n
  s!"{tenths / 10}.{tenths % 10}"

/-- The arena of a binary tree. -/
def arena (t : BTree Float) : Arena Float :=
  Arena.encode (σ := BTree.sig) (T := fun _ => BTree Float) (s := 0) t

/-- Times the stages for a tree of `2 ^ k` inner nodes and prints one line of nanoseconds per row,
where the rows are the inner nodes and the leaves. -/
def stage (k : Nat) (device : Bool) : IO Unit := do
  let t := BTree.random (fun w => Float.ofNat (w % 1000).toNat / 8) (2 ^ k) (k.toUInt64 + 1)
  touch t
  let (a, encode) ← timed arena t
  let (c, columns) ← timed (fun a : Arena Float => a.columns BTree.sig) a
  let ((words, scalars), pack) ← timed (fun c : Arena.Columns Float => (c.words, c.scalars 8)) c
  let (back, decode) ← timed (fun a : Arena Float => a.decode (σ := BTree.sig)
    (fun _ => BTree Float) 0) a
  unless back.isSome do throw <| IO.userError "tree_bench: the arena does not decode"
  let n := c.tag.size
  let mut line := s!"2^{k} inner nodes ({n} rows, {c.sizes.size} levels, " ++
    s!"{words.size + scalars.size} bytes): encode {perRow encode n}, " ++
    s!"columns {perRow columns n}, pack {perRow pack n}, decode {perRow decode n}"
  if device then
    let (d, upload) ← timedIO (Arena.upload BTree.sig a)
    let (a', download) ← timedIO d.download
    let c' := a'.columns BTree.sig
    unless c'.words.data == words.data && (c'.scalars 8).data == scalars.data do
      throw <| IO.userError "tree_bench: the device returned other blocks"
    line := line ++ s!", upload {perRow upload n}, download {perRow download n}"
  IO.println (line ++ " ns per row")

/-- Whether two binary64 results are equal by encoding or both NaN. -/
def agree (x y : Float) : Bool := x.toBits == y.toBits || x.isNaN && y.isNaN

/-- Times a fold over the arena `a` of `n` rows, whose reference result `x` took `reference`
nanoseconds: the schedule on the host with the reader of `inputs`, and with a device two runs of
`Step.runDevice`. Prints a line of nanoseconds per row, with the first device run in
milliseconds. -/
def foldStage {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive Float T] (name : String)
    (F : Fold σ Float T) (a : Arena Float) (n : Nat) (inputs : Array (Array Float))
    (x : Float) (reference : Nat) (device : Bool) : IO Unit :=
  letI := F.scalar
  do
    let (y, schedule) ← timed (F.eval (Reader.ofArrays inputs)) a
    unless (match y with | .ok y => agree x y | .error _ => false) do
      throw <| IO.userError s!"tree_bench: the {name} schedule differs from the reference"
    let mut line := s!"  {name}: reference {perRow reference n}, schedule {perRow schedule n}"
    if device then
      let d ← Arena.upload σ a
      let (cold, first) ← timedIO (F.runDevice d inputs)
      let (warm, second) ← timedIO (F.runDevice d inputs)
      for r in [cold, warm] do
        unless (match r with | some #[y] => agree x y | _ => false) do
          throw <| IO.userError s!"tree_bench: the device {name} differs from the reference"
      line := line ++ s!", device {perRow second n} (first run {first / 1000000} ms)"
    IO.println (line ++ " ns per row")

/-- Times the sum of a pseudorandom binary tree of `2 ^ k` inner nodes, and the value of a
pseudorandom term of `2 ^ k` sums and products over an environment of 4096 variables. -/
def folds (k : Nat) (device : Bool) : IO Unit := do
  let A : Arithmetic Float := Scalar.arithmetic
  let t := BTree.random (fun w => Float.ofNat (w % 1000).toNat / 8) (2 ^ k) (k.toUInt64 + 1)
  let a := arena t
  touch a
  let (x, reference) ← timed (BTree.sum A 0.0) t
  foldStage "sum" (BTree.sumFold 0.0) a (2 ^ (k + 1) + 1) #[] x reference device
  let term := Term.random (fun w => 1 + Float.ofNat (w % 64).toNat / 512) 4096 (2 ^ k)
    (k.toUInt64 + 2)
  let env := (Array.range 4096).map fun i => 1 - Float.ofNat i / 8192
  let a := Arena.encode (σ := Term.sig) (T := fun _ => Term Float) (s := 0) term
  touch a
  let (x, reference) ← timed (fun t => Term.eval A (Reader.ofArrays #[env]) t) term
  let .ok x := x | throw <| IO.userError "tree_bench: the term reads past its environment"
  foldStage "term" Term.evalFold a (2 ^ (k + 1) + 1) #[env] x reference device

/-- Times a hylomorphism from the roots `roots` with a call tree of `n` rows and the reference
result `x`, which took `reference` nanoseconds: the schedule on the host with the reader of
`inputs`, and with a device two runs of `Hylo.runDevice`. Prints a line of nanoseconds per row,
with the first device run in milliseconds. -/
def hyloStage {m s : Nat} (name : String) (H : Hylo Float m s) (roots : Arena.Level Float)
    (n : Nat) (inputs : Array (Array Float)) (x : Float) (reference : Nat) (device : Bool) :
    IO Unit := do
  let (y, schedule) ← timed (H.schedule (Reader.ofArrays inputs) Expansion.deviceCapacity) roots
  unless (match y with | some #[y] => agree x y | _ => false) do
    throw <| IO.userError s!"tree_bench: the {name} schedule differs from the reference"
  let mut line := s!"  {name}: reference {perRow reference n}, schedule {perRow schedule n}"
  if device then
    let (cold, first) ← timedIO (H.runDevice Expansion.deviceCapacity roots inputs)
    let (warm, second) ← timedIO (H.runDevice Expansion.deviceCapacity roots inputs)
    for r in [cold, warm] do
      unless (match r with | some #[y] => agree x y | _ => false) do
        throw <| IO.userError s!"tree_bench: the device {name} differs from the reference"
    line := line ++ s!", device {perRow second n} (first run {first / 1000000} ms)"
  IO.println (line ++ " ns per row")

/-- The least `j` whose Fibonacci call tree, of `2 * fib (j + 1) - 1` rows, has at least `rows`
rows, and its number of rows. -/
def fibSize (rows : Nat) : Nat × Nat := Id.run do
  let mut (j, a, b) := (0, 1, 1)
  while 2 * a - 1 < rows do
    (j, a, b) := (j + 1, b, a + b)
  return (j, 2 * a - 1)

/-- Times three hylomorphisms with call trees of about `2 ^ (k + 1)` rows: the sum by halves of
`2 ^ k` entries of an input, the Fibonacci number whose call tree is the smallest with at least
`2 ^ (k + 1) - 1` rows, and the midpoint rule with `2 ^ k` panels. -/
def hylos (k : Nat) (device : Bool) : IO Unit := do
  let A : Arithmetic Float := Scalar.arithmetic
  let n := 2 ^ k
  let xs := (Array.range n).map fun i => Float.ofNat (i * 7919 % 1000) / 8
  touch xs
  let (x, reference) ← timed (fun xs => pairwise A 0.0 (Reader.ofArrays #[xs]) 0 n) xs
  let .ok x := x | throw <| IO.userError "tree_bench: the sum by halves reads past its input"
  hyloStage "sum by halves" (pairwiseHylo 0.0) ⟨#[0], #[#[0], #[n.toUInt64]], #[]⟩ (2 * n - 1)
    #[xs] x reference device
  let (j, rows) := fibSize (2 * n - 1)
  let (x, reference) ← timed (fib A 0.0 1.0) j
  hyloStage s!"fib {j}" (fibHylo 0.0 1.0) ⟨#[0], #[#[j.toUInt64]], #[]⟩ rows #[] x reference
    device
  let (x, reference) ← timed (fun k => midpoint A 2.0 k 0.0 1.0) k
  hyloStage "midpoint rule" (midpointHylo 2.0) ⟨#[0], #[#[k.toUInt64]], #[#[0.0], #[1.0]]⟩
    (2 * n - 1) #[] x reference device

/-- Times an unfold from the seeds `roots` into an arena of `n` rows and a fold over it, whose
reference value `t` took `reference` nanoseconds and folds to `x`: the build on the host, and with
a device the encoding and upload of `t`, two runs of `Unfold.buildDevice` and two of
`Step.runDevice` over the arena of the second, whose results must be `x`, as must the sum over the
arena of the first. Prints a line of nanoseconds per row, with the first device build in
milliseconds. -/
def unfoldStage {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive Float T] {m s : Nat}
    (name : String) (U : Unfold σ Float T m s) (F : Fold σ Float T) (hs : U.scalar = F.scalar)
    (roots : Arena.Level Float) (n : Nat) (inputs : Array (Array Float)) (t : Σ i, T i)
    (x : Float) (reference : Nat) (device : Bool) : IO Unit :=
  letI := U.scalar
  do
    let (a, build) ← timed (U.build (Reader.ofArrays inputs) Expansion.deviceCapacity) roots
    let some a := a | throw <| IO.userError s!"tree_bench: the {name} build failed"
    unless (match F.eval (Reader.ofArrays #[]) a with | .ok y => agree x y | .error _ => false) do
      throw <| IO.userError s!"tree_bench: the sum of the {name} build differs from the reference"
    let mut line := s!"  {name}: reference {perRow reference n}, build {perRow build n}"
    if device then
      let (encoded, encode) ← timed (fun t => Arena.ofValues (σ := σ) (T := T) [t]) t
      let (_, upload) ← timedIO (Arena.upload σ encoded)
      let (d₀, first) ← timedIO (U.buildDevice Expansion.deviceCapacity roots inputs)
      let (d, second) ← timedIO (U.buildDevice Expansion.deviceCapacity roots inputs)
      let (some d₀, some d) := (d₀, d)
        | throw <| IO.userError s!"tree_bench: the device {name} build failed"
      let (cold, _) ← timedIO (F.runDevice (hs ▸ d) #[])
      let (warm, fold) ← timedIO (F.runDevice (hs ▸ d) #[])
      let earlier ← F.runDevice (hs ▸ d₀) #[]
      for r in [cold, warm, earlier] do
        unless (match r with | some #[y] => agree x y | _ => false) do
          throw <| IO.userError s!"tree_bench: the device sum of the {name} differs from the \
            reference"
      line := line ++ s!", encode and upload {perRow (encode + upload) n}, device build " ++
        s!"{perRow second n} (first run {first / 1000000} ms), device sum {perRow fold n}"
    IO.println (line ++ " ns per row")

/-- Times two unfolds into arenas of about `2 ^ (k + 1)` rows, each followed by a sum: the
balanced tree of `2 ^ k` entries of an input, and the binomial tree of order `k + 1` whose
children have half the scalar of their parent. -/
def unfolds (k : Nat) (device : Bool) : IO Unit := do
  let A : Arithmetic Float := Scalar.arithmetic
  let n := 2 ^ k
  let xs := (Array.range n).map fun i => Float.ofNat (i * 7919 % 1000) / 8
  touch xs
  let (t, reference) ← timed (fun xs => BTree.ofRange (Reader.ofArrays #[xs]) 0 n) xs
  let .ok t := t | throw <| IO.userError "tree_bench: the balanced tree reads past its input"
  unfoldStage "balanced tree" ofRangeUnfold (BTree.sumFold 0.0) rfl
    ⟨#[0], #[#[0], #[n.toUInt64]], #[]⟩ (2 * n + 1) #[xs] ⟨0, t⟩ (BTree.sum A 0.0 t) reference
    device
  let (r, reference) ← timed (fun k => Rose.binomial A 0.5 k 1.0) (k + 1)
  unfoldStage s!"binomial tree of order {k + 1}" (binomialUnfold 0.5) Rose.sumFold rfl
    ⟨#[0], #[#[(k + 1).toUInt64]], #[#[1.0]]⟩ (2 ^ (k + 1)) #[] ⟨0, r⟩ (Rose.sum A r) reference
    device

/-- The scalars of `n` pseudorandom words of `α`. -/
def randomWords {α : Type} [Numeric.Word α] (n : Nat) : Array α :=
  (Array.range n).map fun i =>
    Numeric.Word.ofNatBits ((BTree.mix i.toUInt64).toNat % 2 ^ Numeric.Word.width (α := α))

/-- Whether the results `ys` of a search are the scalars `zs`, bit for bit. -/
def sameWords {α : Type} [Numeric.Word α] (ys : Array (Except Error α)) (zs : Array α) : Bool :=
  ys.size == zs.size && (ys.zip zs).all fun (y, z) => match y with
    | .ok y => Numeric.Word.toBits y == Numeric.Word.toBits z
    | .error _ => false

/-- Makes the table of `f`, named `name`, and times `Table.make`, then `f` itself and the search
of the table on the host on `2 ^ 14` pseudorandom words, where the search must give `f` bit for
bit. Prints a line with the milliseconds of `Table.make` and the nanoseconds per value of the
others. -/
def makeTable {α : Type} [Scalar α] (name : String) (f : α → α) :
    IO (Table.Of (Scalar.arithmetic (α := α)) f) := do
  let start ← IO.monoNanosNow
  let T : Option (Table.Of Scalar.arithmetic f) := Table.make Scalar.arithmetic (after start f)
  touch T
  let stop ← IO.monoNanosNow
  let some T := T | throw <| IO.userError s!"tree_bench: Table.make found no table of {name}"
  let xs : Array α := randomWords (2 ^ 14)
  touch xs
  let (zs, direct) ← timed (fun xs : Array α => xs.map f) xs
  let (ys, search) ← timed (fun xs : Array α => xs.map (T.table.eval Scalar.arithmetic)) xs
  unless sameWords ys zs do
    throw <| IO.userError s!"tree_bench: the search of the {name} table differs from {name}"
  IO.println s!"  {name} table of {T.table.keys.size} keys: make {(stop - start) / 1000000} ms, \
    on the host {name} {perRow direct xs.size}, the search {perRow search xs.size} ns per value"
  return T

/-- Times two runs of `Table.Of.runDevice` with the table `T` of `f`, named `name`, over `2 ^ k`
pseudorandom words. The results must be `f` bit for bit, read from `results`, the results of `f`
at every word. Prints a line of nanoseconds per value, with the first run in milliseconds. -/
def tableStage {α : Type} [Scalar α] {f : α → α} (name : String)
    (T : Table.Of (Scalar.arithmetic (α := α)) f) (results : Array α) (k : Nat) : IO Unit := do
  let n := 2 ^ k
  let xs : Array α := randomWords n
  touch xs
  let (cold, first) ← timedIO (T.runDevice xs)
  let (warm, second) ← timedIO (T.runDevice xs)
  for ys in [cold, warm] do
    unless ys.size == n && (xs.zip ys).all fun (x, y) =>
        (results[(Numeric.Word.toBits x).toNat]?.map Numeric.Word.toBits) ==
          some (Numeric.Word.toBits y) do
      throw <| IO.userError s!"tree_bench: the device {name} table differs from {name}"
  IO.println s!"  {name} table, 2^{k} values: device {perRow second n} (first run \
    {first / 1000000} ms) ns per value"

/-- Times the fold `BTree.tableFold` of the table `T` of `f`, named `name`, over a pseudorandom
binary tree of `2 ^ k` inner nodes whose scalars `scalar` gives: the sum of the tree mapped by `f`,
the schedule on the host and, with a device, two runs of `Step.runDevice`, whose results must be
the sum bit for bit. Prints a line of nanoseconds per row, with the first device run in
milliseconds. -/
def tableFoldStage {α : Type} [Scalar α] [Inhabited α] {f : α → α} (name : String)
    (T : Table.Of (Scalar.arithmetic (α := α)) f) (scalar : UInt64 → α) (k : Nat)
    (device : Bool) : IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let zero : α := Numeric.Word.ofNatBits 0
  let t := BTree.random scalar (2 ^ k) (k.toUInt64 + 3)
  let a := Arena.encode (σ := BTree.sig) (T := fun _ => BTree α) (s := 0) t
  touch a
  let n := 2 ^ (k + 1) + 1
  let (x, reference) ← timed (fun t => BTree.sum A zero (t.map f)) t
  let F := BTree.tableFold zero T.table
  let inputs := #[T.table.keys, T.table.values]
  let (y, schedule) ← timed (F.eval (Reader.ofArrays inputs)) a
  unless sameWords #[y] #[x] do
    throw <| IO.userError s!"tree_bench: the {name} table fold differs from the reference"
  let mut line := s!"  {name} table fold: reference {perRow reference n}, " ++
    s!"schedule {perRow schedule n}"
  if device then
    let d ← Arena.upload BTree.sig a
    let (cold, first) ← timedIO (F.runDevice d inputs)
    let (warm, second) ← timedIO (F.runDevice d inputs)
    for r in [cold, warm] do
      unless (match r with | some #[y] => sameWords #[.ok y] #[x] | _ => false) do
        throw <| IO.userError s!"tree_bench: the device {name} table fold differs"
    line := line ++ s!", device {perRow second n} (first run {first / 1000000} ms)"
  IO.println (line ++ " ns per row")

/-- The usage text of `tree_bench`. -/
def usage : String :=
  "usage: tree_bench [--log2 K,K,...] [--device auto|" ++ Backend.Device.help ++ "]\n" ++
  "Times the arena stages of pseudorandom binary trees with 2^K inner nodes for each K\n" ++
  "(default 16,18,20,22, at most 24), on the host and, with a device, the upload and download,\n" ++
  "then a tree sum and a term of 2^K operations as folds, and a sum by halves of 2^K entries,\n" ++
  "a Fibonacci number and a midpoint rule of 2^K panels as hylomorphisms, and a balanced tree\n" ++
  "of 2^K entries and a binomial tree of order K+1 as unfolds, on the host and the device.\n" ++
  "Last the tables of a correctly rounded exp in binary16 and log in bfloat16 are made once\n" ++
  "and searched on the host, and with a device on 2^K pseudorandom words there, and the\n" ++
  "binary16 table is applied at every node of a fold over a tree of 2^16 inner nodes.\n" ++
  "`auto` uses the device when one is usable."

/-- The exponents of a `--log2` value: numbers up to 24, separated by commas. -/
def parseLog2 (v : String) : Except String (List Nat) :=
  (v.splitOn ",").mapM fun s => match s.toNat? with
    | some k => if k ≤ 24 then .ok k else .error s!"tree_bench: 2^{k} inner nodes is over 2^24"
    | none => .error s!"tree_bench: --log2 takes numbers separated by commas, not {v}"

/-- The exponents and the device choice (`auto`, `cpu` or `cuda`) of some arguments. -/
def parse : List String → List Nat → String → Except String (List Nat × String)
  | [], ks, device => .ok (ks, device)
  | "--log2" :: v :: rest, _, device => do parse rest (← parseLog2 v) device
  | "--device" :: "auto" :: rest, ks, _ => parse rest ks "auto"
  | "--device" :: v :: rest, ks, _ => do parse rest ks (← Backend.Device.parse v).cliName
  | arg :: _, _, _ => .error s!"tree_bench: unknown or incomplete argument {arg}"

/-- Makes the two tables, then runs the stages of each `k` in `--log2` and last the fold of the
binary16 table, on the host and, with a device, also on the device. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  match parse args [16, 18, 20, 22] "cpu" with
  | .error e => IO.eprintln (e ++ "\n" ++ usage); return 2
  | .ok (ks, choice) =>
    let device ← match choice with
      | "cuda" => do LibTorch.Buffer.requireNativeRuntime; pure true
      | "auto" => do
        match ← LibTorch.Buffer.RuntimeStatus.query with
        | .nativeAvailable => pure true
        | _ => pure false
      | _ => pure false
    let exp16 := fun x : FloatLib.Floats.ExecFloat.Binary 5 10 =>
      FloatLib.Floats.ExecFloat.Binary.CorrectlyRounded.exp x {}
    let log16 := fun x : FloatLib.Floats.ExecFloat.Binary 8 7 =>
      FloatLib.Floats.ExecFloat.Binary.CorrectlyRounded.log x {}
    let expTable ← makeTable "binary16 exp" exp16
    let logTable ← makeTable "bfloat16 log" log16
    let expResults := if device then Table.results exp16 else #[]
    let logResults := if device then Table.results log16 else #[]
    for k in ks do
      stage k device
      folds k device
      hylos k device
      unfolds k device
      if device then
        tableStage "binary16 exp" expTable expResults k
        tableStage "bfloat16 log" logTable logResults k
    tableFoldStage "binary16 exp" expTable
      (fun w => Numeric.Word.ofNatBits ((w % 0x3C00).toNat + 0x8000)) 16 device
    return 0

end TorchLean.Examples.Tree.Bench
