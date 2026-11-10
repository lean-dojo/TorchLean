/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Table
public import TorchLean.Tensor.ConfiguredBinary.Elementary
public import TorchLeanTest.Tree.Word

/-!
# Unary functions in kernels as checked tables

The host checks build the tables of FloatLib's correctly rounded `exp`, `expMinus1`, `log` and
`logPlus1` in the configured formats e4m3, e5m2, 12-bit, binary16 and bfloat16 and in the 8-bit
formats without infinities e4m3fn and e4m3fnuz, of TorchLean's correctly rounded `tan`, negation,
`0 - x`, the identity and a constant in the 8-bit formats, and of the reciprocal in e4m3 and
e5m2. `Table.make` checks each table at every word, signed zeros, infinities and NaN payloads
included. The returned `Table.Of` carries that result, and `Table.Of.evaluate_kernel` proves that
the kernel gives the function when its inputs hold the table.
Two kernels apply tables inside arithmetic, `log1p (exp x)` with two tables and `x * exp x`, and
give the composite at every word of the 12-bit format, binary16 and bfloat16. A read past the
input is the failure of the kernel. NaNs and zeros read no table, the other inputs read the keys,
and the keys are padded to a power of two. In an arithmetic of e4m3 where only the words `0` and
`1` are equal to themselves, the table of the identity has depth `0` and one key, gives the
identity at every word and finds word `1` without reading a key.

`Table.make` must refuse the formats of more than `Table.maxWidth` bits, a 21-bit format among
them, and the absolute value, which clears the sign of a NaN. `Table.check` must reject a table
with one wrong value and a table with its keys reversed, and `Table.Of.runDevice` must refuse a
scalar type whose words are wider than its storage before it calls the device. The fold
`BTree.tableFold` of the table of `exp` in the 12-bit format and binary16 gives the sum of each tree
of the fold checks mapped by `exp`, and of each tree of one node, and without the table it fails on
the keys.

The device checks run the kernel of every table over all the words of its format with
`Table.Of.runDevice`, and the composite kernels with `runBytes`, and compare the results with the
functions bit for bit, and every result word must fit the width of its format. The table of `exp`
in the 12-bit format and binary16 also runs over the first 1, 255 and 257 words and all but one,
which end inside a block of threads, and over no words. Without its keys, and with an argument read
past its input, the launch fails, and the NaNs and zeros give `exp` without keys and values. The
fold of the table runs on the device over the same trees as on the host, with the table and without
it. Without LibTorch, `Table.Of.runDevice` must fail, also for an empty array.
-/

@[expose] public section

namespace TorchLean.Test.Tree.Tables

open Kernel Examples.Tree FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange Arenas Words

variable {α : Type}

/-- Every scalar of type `α`, in the order of the words. -/
def allWords [Numeric.Word α] : Array α :=
  Array.ofFn fun i : Fin (2 ^ Numeric.Word.width (α := α)) => Numeric.Word.ofNatBits i.val

section Functions

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

/-- FloatLib's correctly rounded `exp`, `expMinus1`, `log` and `logPlus1` in a configured format,
by name. -/
def correctlyRounded : List (String × (ExecFloat (Configured.Family format code plan) →
    ExecFloat (Configured.Family format code plan))) :=
  [("exp", (ExecFloat.Binary.CorrectlyRounded.exp · {})),
    ("expm1", (ExecFloat.Binary.CorrectlyRounded.expMinus1 · {})),
    ("log", (ExecFloat.Binary.CorrectlyRounded.log · {})),
    ("log1p", (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {}))]

/-- The functions of `correctlyRounded` and TorchLean's correctly rounded tangent. -/
def withTan : List (String × (ExecFloat (Configured.Family format code plan) →
    ExecFloat (Configured.Family format code plan))) :=
  correctlyRounded ++ [("tan", ConfiguredBinary.tan)]

end Functions

/-- Negation, `0 - x`, the identity and the constant `one` in the arithmetic of `α`, by name. At
the NaNs or at the zeros their tables give the negation of the input, `zero` minus the input, the
input itself and one scalar. -/
def arithmeticFunctions [Scalar α] (one : α) : List (String × (α → α)) :=
  let A : Arithmetic α := Scalar.arithmetic
  [("neg", A.neg), ("0-x", A.binary .sub (Numeric.Word.ofNatBits 0)), ("id", id),
    ("one", fun _ => one)]

/-- The functions of `arithmeticFunctions` and the reciprocal, `one` divided by the input. -/
def withReciprocal [Scalar α] (one : α) : List (String × (α → α)) :=
  arithmeticFunctions one ++ [("1/x", (Scalar.arithmetic (α := α)).binary .div one)]

/-- The table of `f`, or an error naming `label` and `name`. -/
def tableOf [Scalar α] (label name : String) (f : α → α) :
    IO (Table.Of (Scalar.arithmetic (α := α)) f) :=
  match Table.make Scalar.arithmetic f with
  | some T => pure T
  | none => throw <| IO.userError s!"{label} {name}: Table.make found no table"

/-- The positions `i` below `xs.size` where `ys i` is not `f xs[i]` bit for bit. -/
def mismatches [Numeric.Word α] [Inhabited α] (f : α → α) (xs : Array α)
    (ys : Nat → Except Error α) : List Nat :=
  (List.range xs.size).filter fun i => !sameBits (ys i) (.ok (f xs[i]!))

/-- Fails unless `wrong`, the positions where a result differs, is empty. -/
def ensureNone (label : String) (wrong : List Nat) : IO Unit :=
  ensure wrong.isEmpty
    s!"{label}: wrong at {wrong.length} words, the first at position {wrong.head?.getD 0}"

/-- The kernel of `g (f x)` at entry `i` of input `0`, with the table of `f` read from inputs `1`
and `2` and the table of `g` from inputs `3` and `4`. -/
def composite (F G : Table α) : Expr α [.index] .scalar :=
  G.apply 3 4 (F.apply 1 2 (.load 0 (.var .zero)))

/-- The kernel of `x * f x` at entry `i` of input `0`, with the table of `f` read from inputs `1`
and `2`. -/
def timesSelf (F : Table α) : Expr α [.index] .scalar :=
  .letIn (.load 0 (.var .zero)) (.binary .mul (.var .zero) (F.apply 1 2 (.var .zero)))

/-- Require a checked table for every function of `fs`. `Table.make` checks every word, and
`Table.Of.evaluate_kernel` proves the kernel's correspondence from the returned certificate. -/
@[noinline, nospecialize] def checkTables [Scalar α] (label : String)
    (fs : List (String × (α → α))) : IO Unit := do
  for (name, f) in fs do
    discard <| tableOf label name f

/-- Checks the kernels `composite` and `timesSelf` with the tables of `f` and `g` at every word,
the failure of a read past the input, the reads of the table of `f`, none for NaNs and zeros and
the keys for the other words, and that its keys are padded past the words it searches. -/
@[noinline, nospecialize] def checkKernels [Scalar α] [Inhabited α] (label : String)
    (f g : α → α) : IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let xs : Array α := allWords
  let F ← tableOf label "first" f
  let G ← tableOf label "second" g
  let R := Reader.ofArrays #[xs, F.table.keys, F.table.values, G.table.keys, G.table.values]
  ensureNone s!"{label}: two tables in one kernel"
    (mismatches (fun x => g (f x)) xs fun i =>
      evaluate A (composite F.table G.table) R i.toUInt64)
  ensureNone s!"{label}: a table inside arithmetic"
    (mismatches (fun x => A.binary .mul x (f x)) xs fun i =>
      evaluate A (timesSelf F.table) R i.toUInt64)
  let n := xs.size.toUInt64
  ensure (sameBits (evaluate A F.table.kernel R n) (.error (.bounds 0 n xs.size)))
    s!"{label}: a read past the input is not the failure of the kernel"
  let skips := xs.filter fun x => !A.compare .eq x x || A.compare .eq x F.table.zero
  ensureNone s!"{label}: NaNs and zeros without the table"
    (mismatches f skips fun i => evaluate A F.table.kernel (Reader.ofArrays #[skips]) i.toUInt64)
  let searched := xs.filter fun x => A.compare .eq x x && !A.compare .eq x F.table.zero
  ensure (searched.all fun x => match evaluate A F.table.kernel (Reader.ofArrays #[#[x]]) 0 with
      | .error (.bounds 1 _ 0) => true
      | _ => false)
    s!"{label}: an input other than a NaN or a zero does not read the keys"
  ensure (F.table.keys.size == 2 ^ F.table.depth && searched.size < F.table.keys.size)
    s!"{label}: {F.table.keys.size} keys of depth {F.table.depth} for {searched.size} words"

/-- Checks a table of depth `0`. In an arithmetic of e4m3 where only the words `0` and `1` are
equal to themselves, the table of the identity has one key and gives the identity at every word.
At word `1` its kernel reads no key, and without its values it fails on the read of value `0`. -/
def checkDepthZero : IO Unit := do
  let B : Arithmetic (ExecFloat.Binary 4 3) := Scalar.arithmetic
  let A : Arithmetic (ExecFloat.Binary 4 3) := { B with
    compare := fun op x y => match op with
      | .eq => bits x == bits y && bits x ≤ 1
      | op => B.compare op x y }
  let some T := (Table.make A id : Option (Table.Of A id))
    | throw <| IO.userError "two ordinary words: Table.make found no table"
  ensure (T.table.depth == 0 && T.table.keys.size == 1)
    s!"two ordinary words: a table of depth {T.table.depth} with {T.table.keys.size} keys"
  let xs : Array (ExecFloat.Binary 4 3) := allWords
  let R := Reader.ofArrays #[xs, T.table.keys, T.table.values]
  ensureNone "two ordinary words: the kernel"
    (mismatches id xs fun i => evaluate A T.table.kernel R i.toUInt64)
  let one : ExecFloat.Binary 4 3 := Numeric.Word.ofNatBits 1
  ensure (sameBits (evaluate A T.table.kernel (Reader.ofArrays #[#[one], #[], T.table.values]) 0)
      (.ok one)) "two ordinary words: the search of depth 0 reads a key"
  ensure (sameBits (evaluate A T.table.kernel (Reader.ofArrays #[#[one]]) 0)
      (.error (.bounds 2 0 0))) "two ordinary words: the search of depth 0 reads no value"

/-- Checks the refusals of `Table.make`, `Table.check` and `Table.Of.runDevice`: no table for
formats wider than `Table.maxWidth` bits, a 21-bit format among them, or for the absolute value in
e5m2, `check` rejects a table of `exp` in e5m2 with one wrong value or with its keys reversed, and
`runDevice` refuses the words of a 12-bit format in the storage of an 8-bit one before it calls
the device. -/
def checkRefusals : IO Unit := do
  let exp65 := fun x : ExecFloat.Binary 11 53 => ExecFloat.Binary.CorrectlyRounded.exp x {}
  ensure (Table.make Scalar.arithmetic exp65).isNone "65-bit exp: a table"
  ensure (Table.make (α := ExecFloat.Binary 8 23) Scalar.arithmetic id).isNone
    "configured binary32: a table"
  ensure (Table.make (α := ExecFloat.Binary 8 12) Scalar.arithmetic id).isNone
    "21-bit format: a table"
  ensure (Table.make (α := FloatBits) Scalar.arithmetic id).isNone "binary64 words: a table"
  ensure (Table.make (α := Float32Bits) Scalar.arithmetic id).isNone "binary32 words: a table"
  let A : Arithmetic (ExecFloat.Binary 5 2) := Scalar.arithmetic
  let abs := fun x : ExecFloat.Binary 5 2 =>
    (Numeric.Word.ofNatBits (bits x % 0x80) : ExecFloat.Binary 5 2)
  ensure (Table.make A abs).isNone
    "e5m2 absolute value: a table, though it clears the signs of NaNs"
  let expE5M2 := fun x : ExecFloat.Binary 5 2 => ExecFloat.Binary.CorrectlyRounded.exp x {}
  let rs := Table.results expE5M2
  let t := Table.build A rs
  ensure (t.check A rs) "e5m2 exp: the built table fails its check"
  let i := t.keys.size / 2
  ensure (!Table.check { t with values := t.values.set! i (A.neg t.values[i]!) } A rs)
    "e5m2 exp: a table with a wrong value passes its check"
  ensure (!Table.check { t with keys := t.keys.reverse } A rs)
    "e5m2 exp: a table with reversed keys passes its check"
  let wide : Scalar (ExecFloat.Binary 5 6) :=
    { (inferInstance : Scalar (ExecFloat.Binary 5 6)) with
      precision := Scalar.precision (ExecFloat.Binary 5 2) }
  let T ← @tableOf _ wide "12-bit words in 8-bit storage" "id" id
  ensureFails "12-bit words in 8-bit storage" "wider than their storage"
    (@Table.Of.runDevice _ wide _ T allWords)

/-- The binary trees of one node, one for each word of `α`. -/
def singles [Numeric.Word α] : List (BTree α) :=
  (allWords (α := α)).toList.map fun x => .node .leaf x .leaf

/-- Checks the fold `BTree.tableFold` of the table of `f` with the word `0` as `zero`. Over the
trees of the fold checks and over the batch of the trees of one node it gives the sum of each tree
mapped by `f`. Without the table it gives `BTree.tableSum`, which fails on the keys at every tree
but a leaf. -/
@[noinline, nospecialize] def checkTreeTables [Scalar α] [Inhabited α] (label : String)
    (f : α → α) : IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let zero : α := Numeric.Word.ofNatBits 0
  let T ← tableOf label "tree fold" f
  let F := BTree.tableFold zero T.table
  let read := Reader.ofArrays #[T.table.keys, T.table.values]
  let empty := Reader.ofArrays #[]
  for (name, t) in Folds.btrees (α := α) do
    ensure (Folds.same (F.eval read (arenaB t)) (.ok (BTree.sum A zero (t.map f))))
      s!"{label} {name}: the fold of the table differs from the sum of the function"
    let r := F.eval empty (arenaB t)
    ensure (Folds.same r (BTree.tableSum A zero T.table empty t))
      s!"{label} {name}: the fold without the table differs from its reference"
    ensure (t matches .leaf || r matches .error (.bounds 0 _ 0))
      s!"{label} {name}: the fold without the table does not fail on its keys"
  let ts := singles (α := α)
  ensure (Folds.allSame (F.results read (batchB ts) 0)
      (ts.toArray.map fun t => .ok (BTree.sum A zero (t.map f))))
    s!"{label}: the fold of the table differs from the sum of the function at a tree of one node"

/-- The host checks: the tables of the correctly rounded functions in seven formats, of the
tangent and the functions of `arithmeticFunctions` in the four 8-bit formats and of the reciprocal
in two, the composite kernels, the reads and failures of the kernels, a table of depth `0`, the
refusals, and the folds of tables over trees. -/
def run : IO Unit := do
  checkTables (α := ExecFloat.Binary 4 3) "e4m3"
    (withTan ++ withReciprocal (Numeric.Word.ofNatBits 0x38))
  checkTables (α := ExecFloat.Binary 5 2) "e5m2"
    (withTan ++ withReciprocal (Numeric.Word.ofNatBits 0x3C))
  checkTables (α := ExecFloat.Binary 4 3 .finiteMaxNaN) "e4m3fn"
    (withTan ++ arithmeticFunctions (Numeric.Word.ofNatBits 0x38))
  checkTables (α := ExecFloat.Binary 4 3 .finiteUnsignedZero) "e4m3fnuz"
    (withTan ++ arithmeticFunctions (Numeric.Word.ofNatBits 0x40))
  checkTables (α := ExecFloat.Binary 5 6) "12-bit" correctlyRounded
  checkTables (α := ExecFloat.Binary 5 10) "binary16" correctlyRounded
  checkTables (α := ExecFloat.Binary 8 7) "bfloat16" correctlyRounded
  checkKernels (α := ExecFloat.Binary 5 6) "12-bit"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
    (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {})
  checkKernels (α := ExecFloat.Binary 5 10) "binary16"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
    (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {})
  checkKernels (α := ExecFloat.Binary 8 7) "bfloat16"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
    (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {})
  checkDepthZero
  checkRefusals
  checkTreeTables (α := ExecFloat.Binary 5 6) "12-bit" (ExecFloat.Binary.CorrectlyRounded.exp · {})
  checkTreeTables (α := ExecFloat.Binary 5 10) "binary16"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
  IO.println "  tables: kernel tables of exp, expm1, log and log1p in seven formats, of tan, \
    negation, 0 - x, the identity and a constant in four and of the reciprocal in two, bit for \
    bit in checked tables on the host, a table of depth 0, the refusals and folds of tables"

/-- The words of `xs` as a kernel input. -/
def inputWords [Scalar α] (xs : Array α) : ByteArray :=
  Bytes.pushWords .empty (Scalar.precision α).bytes bits xs

/-- The scalars of the words of a kernel output, and a size failure past its end. -/
def outputScalars [Scalar α] (out : ByteArray) (i : Nat) : Except Error α :=
  let width := (Scalar.precision α).bytes
  if (i + 1) * width ≤ out.size then
    .ok (Numeric.Word.ofNatBits (Bytes.wordFromLE out (i * width) width))
  else .error (.size out.size)

/-- Whether every word of a kernel output, in the storage of `α`, is below `2 ^ width`. -/
def wordsFit [Scalar α] (out : ByteArray) : Bool :=
  let width := (Scalar.precision α).bytes
  (List.range (out.size / width)).all fun i =>
    Bytes.wordFromLE out (i * width) width < 2 ^ Numeric.Word.width (α := α)

/-- Checks the table of each function of `fs` on the device: `Table.Of.runDevice` over every word
of `α` gives the function bit for bit. -/
@[noinline, nospecialize] def checkDeviceTables [Scalar α] [Inhabited α] (label : String)
    (fs : List (String × (α → α))) : IO Unit := do
  let xs : Array α := allWords
  for (name, f) in fs do
    let T ← tableOf label name f
    let ys ← T.runDevice xs
    ensure (ys.size == xs.size) s!"{label} {name}: the device returned {ys.size} results"
    ensureNone s!"{label} {name}: the device kernel" (mismatches f xs fun i => .ok ys[i]!)

/-- Checks the kernels `composite` and `timesSelf` with the tables of `f` and `g` on the device at
every word of `α`, and that the words of their results fit the width of `α`. -/
@[noinline, nospecialize] def checkDeviceKernels [Scalar α] [Inhabited α] (label : String)
    (f g : α → α) : IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let xs : Array α := allWords
  let F ← tableOf label "first" f
  let G ← tableOf label "second" g
  let inputs := #[inputWords xs, inputWords F.table.keys, inputWords F.table.values,
    inputWords G.table.keys, inputWords G.table.values]
  let size := xs.size * (Scalar.precision α).bytes
  let out ← runBytes inferInstance (composite F.table G.table) inputs xs.size.toUInt64
  ensure (out.size == size) s!"{label}: the device returned {out.size} bytes"
  ensure (wordsFit (α := α) out) s!"{label}: a word of two tables is wider than the format"
  ensureNone s!"{label}: two tables in one device kernel"
    (mismatches (fun x => g (f x)) xs (outputScalars out))
  let out ← runBytes inferInstance (timesSelf F.table) inputs xs.size.toUInt64
  ensure (out.size == size) s!"{label}: the device returned {out.size} bytes"
  ensure (wordsFit (α := α) out) s!"{label}: a word of x * f x is wider than the format"
  ensureNone s!"{label}: a table inside arithmetic on the device"
    (mismatches (fun x => A.binary .mul x (f x)) xs (outputScalars out))

/-- Checks the table of `f` on the device at the edges of a launch. Over the first 1, 255 and 257
words of `α` and all but one, which end inside a block of threads, and over no words it gives `f`
at each entry. Without its keys and with an argument read past its input the launch fails, the
NaNs and zeros give `f` without keys and values, and the words of the results fit the width of
`α`. -/
@[noinline, nospecialize] def checkDeviceEdges [Scalar α] [Inhabited α] (label : String)
    (f : α → α) : IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let xs : Array α := allWords
  let T ← tableOf label "edges" f
  for n in [1, 255, 257, xs.size - 1] do
    let ys ← T.runDevice (xs.extract 0 n)
    ensure (ys.size == n) s!"{label}: the device returned {ys.size} results for {n} words"
    ensureNone s!"{label}: the device kernel over {n} words"
      (mismatches f (xs.extract 0 n) fun i => .ok ys[i]!)
  ensure (← T.runDevice #[]).isEmpty s!"{label}: the device returned results for no words"
  let n := xs.size.toUInt64
  let keys := inputWords T.table.keys
  let values := inputWords T.table.values
  ensureFails s!"{label}: a table without keys" "input 1 index"
    (runBytes inferInstance T.table.kernel #[inputWords xs, .empty, values] n)
  let past := T.table.apply 1 2 (.load 0 (.indexBinary .add (.var .zero) (.index 1)))
  ensureFails s!"{label}: an argument past the input" "input 0 index"
    (runBytes inferInstance past #[inputWords xs, keys, values] n)
  let skips := xs.filter fun x => !A.compare .eq x x || A.compare .eq x T.table.zero
  let out ← runBytes inferInstance T.table.kernel #[inputWords skips, .empty, .empty]
    skips.size.toUInt64
  ensureNone s!"{label}: NaNs and zeros without the table on the device"
    (mismatches f skips (outputScalars out))
  let out ← runBytes inferInstance T.table.kernel #[inputWords xs, keys, values] n
  ensure (wordsFit (α := α) out) s!"{label}: a result word is wider than the format"

/-- Checks the fold `BTree.tableFold` of the table of `f` on the device over the trees of the fold
checks, with the table and without it, and over the batch of the trees of one node, against the
sums of the trees mapped by `f` and, without the table, against `BTree.tableSum`. -/
@[noinline, nospecialize] def checkDeviceTreeTables [Scalar α] [Inhabited α] (label : String)
    (f : α → α) : IO Unit := do
  let A : Arithmetic α := Scalar.arithmetic
  let zero : α := Numeric.Word.ofNatBits 0
  let T ← tableOf label "tree fold" f
  let F := BTree.tableFold zero T.table
  let inputs := #[T.table.keys, T.table.values]
  for (name, t) in Folds.btrees (α := α) do
    Folds.checkRun s!"{label} {name} with a table" F (arenaB t) inputs true
      (some #[.ok (BTree.sum A zero (t.map f))])
    Folds.checkRun s!"{label} {name} without the table" F (arenaB t) #[] (t matches .leaf)
      (some #[BTree.tableSum A zero T.table (Reader.ofArrays #[]) t])
  let ts := singles (α := α)
  Folds.checkRun s!"{label} trees of one node" F (batchB ts) inputs true
    (some (ts.toArray.map fun t => .ok (BTree.sum A zero (t.map f))))

/-- The device checks: the tables of the correctly rounded functions in seven formats, of the
tangent and the functions of `arithmeticFunctions` in the four 8-bit formats and of the reciprocal
in two, the composite kernels, the edges of a launch and the folds of tables over trees, bit for
bit. -/
def runCUDA : IO Unit := do
  checkDeviceTables (α := ExecFloat.Binary 4 3) "e4m3"
    (withTan ++ withReciprocal (Numeric.Word.ofNatBits 0x38))
  checkDeviceTables (α := ExecFloat.Binary 5 2) "e5m2"
    (withTan ++ withReciprocal (Numeric.Word.ofNatBits 0x3C))
  checkDeviceTables (α := ExecFloat.Binary 4 3 .finiteMaxNaN) "e4m3fn"
    (withTan ++ arithmeticFunctions (Numeric.Word.ofNatBits 0x38))
  checkDeviceTables (α := ExecFloat.Binary 4 3 .finiteUnsignedZero) "e4m3fnuz"
    (withTan ++ arithmeticFunctions (Numeric.Word.ofNatBits 0x40))
  checkDeviceTables (α := ExecFloat.Binary 5 6) "12-bit" correctlyRounded
  checkDeviceTables (α := ExecFloat.Binary 5 10) "binary16" correctlyRounded
  checkDeviceTables (α := ExecFloat.Binary 8 7) "bfloat16" correctlyRounded
  checkDeviceKernels (α := ExecFloat.Binary 5 6) "12-bit"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
    (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {})
  checkDeviceKernels (α := ExecFloat.Binary 5 10) "binary16"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
    (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {})
  checkDeviceKernels (α := ExecFloat.Binary 8 7) "bfloat16"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
    (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {})
  checkDeviceEdges (α := ExecFloat.Binary 5 6) "12-bit" (ExecFloat.Binary.CorrectlyRounded.exp · {})
  checkDeviceEdges (α := ExecFloat.Binary 5 10) "binary16"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
  checkDeviceTreeTables (α := ExecFloat.Binary 5 6) "12-bit"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
  checkDeviceTreeTables (α := ExecFloat.Binary 5 10) "binary16"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
  IO.println "  tables: device kernel tables of exp, expm1, log and log1p in seven formats, of \
    tan, negation, 0 - x, the identity and a constant in four and of the reciprocal in two give \
    the functions bit for bit at every word, at the edges of a launch and in folds over trees"

/-- The check of builds without LibTorch: `Table.Of.runDevice` fails, also for an empty array. -/
def runUnlinked : IO Unit := do
  let T ← tableOf (α := ExecFloat.Binary 5 2) "e5m2" "exp"
    (ExecFloat.Binary.CorrectlyRounded.exp · {})
  ensureFails "a table on the device without LibTorch" "built without LibTorch"
    (T.runDevice allWords)
  ensureFails "an empty array on the device without LibTorch" "built without LibTorch"
    (T.runDevice #[])
  IO.println "  tables: device tables refused without LibTorch"

end TorchLean.Test.Tree.Tables
