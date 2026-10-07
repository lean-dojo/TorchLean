/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
/-!
# Native binary32 parity: the test behind the CUDA float32 contract

`Runtime.Autograd.LibTorch.Float32Contract.NativePrimitiveAgreement` records agreement assumptions
for native `add`, `mul`, `div`, `fma`, and `sqrt` against `ExecFloat.Binary 8 23`: non-NaN results
must have identical bits, while two NaN results may have different encodings.
This regression harness compares selected host results with that reference; passing a finite set of
cases does not establish the universal contract or verify a GPU implementation.

Lean's native `Float32` calls host binary32 arithmetic for four operations. Core provides no
`Float32` fused-multiply-add primitive, so this harness emits reference FMA cases for
`scripts/checks/cuda.sh parity`. That script compares host `std::fma`, ATen tensor
operations, and TorchLean's production C interface with the emitted reference bits.

Lean's `Float32.ofBits` canonicalizes NaNs, while `ExecFloat.Binary 8 23` preserves their sign and
payload when quieting them. The native conversion therefore cannot serve as a payload-preserving
oracle. The sweep skips NaN inputs and reports only performed comparisons; `nanOracleReport`
displays the conversion behavior. This limitation concerns the host comparison, not a measured GPU
result.

Run:
  `scripts/lake.sh exe native_float32_parity`
  `scripts/lake.sh exe native_float32_parity --sweep 1000000`
  `scripts/lake.sh exe native_float32_parity --emit-cases`

The last form prints one case per line for the CUDA parity script to consume.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32)

open TorchLean

namespace NN.Tests.Floats.NativePrimitiveParity

/-- Zero-padded lowercase hex, so that bit patterns line up in a column when printed. -/
def hex (u : UInt32) : String :=
  "0x" ++ u.toBitVec.toHex

/--
One comparison to make: a label, and the operand bits.

`z` is only read by `fma`, so the binary and unary cases leave it at zero. Carrying it in the same
structure keeps a single case list, which is what the emitted file for the CUDA side wants.
-/
structure Case where
  /-- Why this case is in the list, printed next to the result. -/
  what : String
  /-- First operand bits. -/
  x : UInt32
  /-- Second operand bits, unused by `sqrt`. -/
  y : UInt32 := 0
  /-- Third operand bits, read only by `fma`. -/
  z : UInt32 := 0
deriving Inhabited

/--
The cases exercise rounding boundaries, signed zeros, subnormal results, overflow, and invalid
operations. Subnormal cases distinguish gradual underflow from flush-to-zero behavior. The fused
triples below separately distinguish one rounding from a multiply followed by an add.
-/
def curatedCases : Array Case := #[
  { what := "tie to even rounds down", x := 0x3F800000, y := 0x33800000 },
  { what := "tie away from even rounds up", x := 0x3F800001, y := 0x33800000 },
  { what := "subnormal plus subnormal", x := 0x00000001, y := 0x00000001 },
  { what := "overflow to infinity", x := 0x7F7FFFFF, y := 0x7F7FFFFF },
  { what := "cancellation to one ulp", x := 0x3F800000, y := 0xBF7FFFFF },
  { what := "signed zeros", x := 0x80000000, y := 0x00000000 },
  { what := "min normal halved is subnormal", x := 0x00800000, y := 0x3F000000 },
  { what := "min subnormal halved underflows", x := 0x00000001, y := 0x3F000000 },
  { what := "three ulps halved ties to even", x := 0x00000003, y := 0x3F000000 },
  { what := "one third is inexact", x := 0x3F800000, y := 0x40400000 },
  { what := "infinity over infinity", x := 0x7F800000, y := 0x7F800000 },
  { what := "largest finite over smallest", x := 0x7F7FFFFF, y := 0x00000001 },
  { what := "two is not a square", x := 0x40000000, y := 0x40000000 },
  { what := "square root of a subnormal", x := 0x00000001, y := 0x00000001 },
  { what := "square root of a negative", x := 0xBF800000, y := 0x3F800000 }
]

/--
Triples for the fused case. A true `fma` rounds once, so the interesting rows are those where the
exact product needs more than 24 bits and the addend then cancels part of it. `modelMulThenAdd`
computes the same expression with two roundings, which is what a machine without a fused instruction
would return.
-/
def curatedFmaCases : Array Case := #[
  { what := "ordinary fused product", x := 0x3F800000, y := 0x40000000, z := 0x3F800000 },
  { what := "exact product cancels", x := 0x3F800001, y := 0x3F800001, z := 0xBF800002 },
  { what := "product rounds away the addend", x := 0x4B800001, y := 0x4B800001, z := 0x3F800000 },
  { what := "half-ulp residual rounds away", x := 0x3F800001, y := 0x3F800001, z := 0xBF800000 },
  { what := "fused product underflows to zero", x := 0x00800000, y := 0x33800000,
    z := 0x00000000 }
]

/-- Bits of a `NaN`: exponent all ones and a nonzero fraction. -/
def isNaNBits (u : UInt32) : Bool :=
  u &&& 0x7F800000 == 0x7F800000 && u &&& 0x007FFFFF != 0

/-- The reference model's answer for each primitive, as bits. -/
def modelAdd (x y : UInt32) : UInt32 :=
  Binary.toBits32 (ExecFloat.add (ofBits32 x) (ofBits32 y))

@[inherit_doc modelAdd]
def modelMul (x y : UInt32) : UInt32 :=
  Binary.toBits32 (ExecFloat.mul (ofBits32 x) (ofBits32 y))

@[inherit_doc modelAdd]
def modelDiv (x y : UInt32) : UInt32 :=
  Binary.toBits32 (ExecFloat.div (ofBits32 x) (ofBits32 y))

@[inherit_doc modelAdd]
def modelSqrt (x : UInt32) : UInt32 :=
  Binary.toBits32 (Binary.sqrtWithRounding (rounding := .nearestEven) (ofBits32 x))

@[inherit_doc modelAdd]
def modelFma (x y z : UInt32) : UInt32 :=
  Binary.toBits32
    (Binary.fmaWithRounding (rounding := .nearestEven) (ofBits32 x) (ofBits32 y) (ofBits32 z))

/-- The same expression with two roundings, used to show what a missing `fma` would return. -/
def modelMulThenAdd (x y z : UInt32) : UInt32 :=
  modelAdd (modelMul x y) z

/-- The host machine's answer for each primitive, through Lean's native `Float32`. -/
def nativeAdd (x y : UInt32) : UInt32 :=
  (Float32.ofBits x + Float32.ofBits y).toBits

@[inherit_doc nativeAdd]
def nativeMul (x y : UInt32) : UInt32 :=
  (Float32.ofBits x * Float32.ofBits y).toBits

@[inherit_doc nativeAdd]
def nativeDiv (x y : UInt32) : UInt32 :=
  (Float32.ofBits x / Float32.ofBits y).toBits

@[inherit_doc nativeAdd]
def nativeSqrt (x : UInt32) : UInt32 :=
  (Float32.ofBits x).sqrt.toBits

/--
One curated case as two lines of output: what the case is for, and the four native results with a
mark for each one the reference model agrees with. Two lines rather than one so that the table stays
inside a terminal width, and so a reader can check a row by hand against a float table.
-/
def caseRows (c : Case) : String :=
  let mark (native model : UInt32) : String := if native == model then "=" else "!"
  s!"  [{c.what}] {hex c.x} {hex c.y}\n" ++
    s!"    add {hex (nativeAdd c.x c.y)}{mark (nativeAdd c.x c.y) (modelAdd c.x c.y)}" ++
    s!" mul {hex (nativeMul c.x c.y)}{mark (nativeMul c.x c.y) (modelMul c.x c.y)}" ++
    s!" div {hex (nativeDiv c.x c.y)}{mark (nativeDiv c.x c.y) (modelDiv c.x c.y)}" ++
    s!" sqrt {hex (nativeSqrt c.x)}{mark (nativeSqrt c.x) (modelSqrt c.x)}"

/-- A single disagreement, kept in enough detail to reproduce it from the printed line alone. -/
structure Mismatch where
  /-- Which contract field failed: `add`, `mul`, `div` or `sqrt`. -/
  op : String
  /-- The operand bits, already formatted. -/
  operands : String
  /-- What the machine returned. -/
  native : UInt32
  /-- What the reference model returned. -/
  model : UInt32
deriving Inhabited

/-- Rendering used both by the tutorial output and by test failures. -/
def Mismatch.render (m : Mismatch) : String :=
  s!"{m.op}({m.operands}): native {hex m.native} model {hex m.model}"

/--
Running counts for one primitive.

Only the first four mismatches are kept, so a large failing sweep still produces a bounded report.
-/
structure Tally where
  /-- Comparisons performed. -/
  checked : Nat := 0
  /-- Comparisons that disagreed. -/
  failed : Nat := 0
  /-- The first few disagreements, for the report. -/
  examples : Array Mismatch := #[]
deriving Inhabited

/-- Record one comparison. `mk` is only called when the result disagrees. -/
def Tally.push (t : Tally) (agrees : Bool) (mk : Unit → Mismatch) : Tally :=
  if agrees then
    { t with checked := t.checked + 1 }
  else
    { checked := t.checked + 1
      failed := t.failed + 1
      examples := if t.examples.size < 4 then t.examples.push (mk ()) else t.examples }

/-- One line per primitive: how many comparisons agreed, and the first disagreements if any. -/
def Tally.render (name : String) (t : Tally) : String :=
  let head := s!"  {name}: {t.checked - t.failed}/{t.checked} bit-exact"
  t.examples.foldl (fun acc m => acc ++ "\n    " ++ m.render) head

/-- Counts for the four primitives Lean can call natively. -/
structure Report where
  /-- Addition tally. -/
  add : Tally := {}
  /-- Multiplication tally. -/
  mul : Tally := {}
  /-- Division tally. -/
  div : Tally := {}
  /-- Square root tally. -/
  sqrt : Tally := {}
deriving Inhabited

/-- Compare one case, on all four natively callable primitives. -/
def Report.check (r : Report) (c : Case) : Report :=
  let pair := s!"{hex c.x}, {hex c.y}"
  let one := hex c.x
  { add :=
      r.add.push (nativeAdd c.x c.y == modelAdd c.x c.y) fun _ =>
        { op := "add", operands := pair, native := nativeAdd c.x c.y, model := modelAdd c.x c.y }
    mul :=
      r.mul.push (nativeMul c.x c.y == modelMul c.x c.y) fun _ =>
        { op := "mul", operands := pair, native := nativeMul c.x c.y, model := modelMul c.x c.y }
    div :=
      r.div.push (nativeDiv c.x c.y == modelDiv c.x c.y) fun _ =>
        { op := "div", operands := pair, native := nativeDiv c.x c.y, model := modelDiv c.x c.y }
    sqrt :=
      r.sqrt.push (nativeSqrt c.x == modelSqrt c.x) fun _ =>
        { op := "sqrt", operands := one, native := nativeSqrt c.x, model := modelSqrt c.x } }

/-- Did every comparison in this report agree? -/
def Report.clean (r : Report) : Bool :=
  r.add.failed == 0 && r.mul.failed == 0 && r.div.failed == 0 && r.sqrt.failed == 0

/-- Total comparisons, four per case. -/
def Report.checked (r : Report) : Nat :=
  r.add.checked + r.mul.checked + r.div.checked + r.sqrt.checked

/-- The four tally lines, in contract order. -/
def Report.render (r : Report) : String :=
  String.intercalate "\n"
    [ Tally.render "add" r.add
    , Tally.render "mul" r.mul
    , Tally.render "div" r.div
    , Tally.render "sqrt" r.sqrt ]

/-- Check a list of cases and summarize the result. -/
def runCases (cases : Array Case) : Report :=
  cases.foldl Report.check {}

/--
The one-line verdict used by the book and by the tests: how many comparisons ran, and whether all
of them were bit-exact.
-/
def summary (cases : Array Case) : String :=
  let r := runCases cases
  if r.clean then
    s!"{r.checked} comparisons, all bit-exact"
  else
    s!"{r.checked} comparisons, {r.add.failed + r.mul.failed + r.div.failed + r.sqrt.failed} failed"

/--
Marsaglia's xorshift32 generator (Marsaglia, "Xorshift RNGs", J. Stat. Soft. 8(14), 2003). A named
generator rather than `IO.rand` keeps a sweep reproducible, which matters when the report is the
evidence for a claim: a reader with the same seed sees the same inputs.
-/
def xorshift (s : UInt32) : UInt32 :=
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 17)
  s ^^^ (s <<< 5)

/--
Compare `draws` pseudorandom operand pairs, skipping `NaN` operands for the reason given in the
module docstring. Each accepted pair contributes one comparison per primitive; skipped draws
contribute none.
-/
def sweep (draws : Nat) (seed : UInt32) : Report := Id.run do
  let mut s : UInt32 := if seed == 0 then 1 else seed
  let mut r : Report := {}
  for _ in [0:draws] do
    s := xorshift s
    let x := s
    s := xorshift s
    let y := s
    unless isNaNBits x || isNaNBits y do
      r := r.check { what := "sweep", x := x, y := y }
  return r

/--
What a `NaN` input does to the two implementations, printed as evidence for the caveat in the module
docstring rather than as a parity check.
-/
def nanOracleReport (payload : UInt32) : String :=
  let native := (Float32.ofBits payload).toBits
  let model := (ofBits32 payload).toBits32
  s!"NaN {hex payload}: Float32 keeps {hex native}, configured binary32 keeps {hex model}"

/-- Command-line help for the native parity tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean native binary32 parity check"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe native_float32_parity [--sweep N] [--seed S] [--emit-cases]"
    , ""
    , "Options:"
    , "  --sweep N     draw N pseudorandom operand pairs, skipping NaN inputs (default 0)"
    , "  --seed S      seed for the sweep generator (default 1)"
    , "  --emit-cases  print cases and their reference bits, one per line, in the format read"
    , "                by scripts/checks/cuda.sh parity;" ++
        " --sweep N adds up to N random cases"
    ]

/-- The lines describing one case: one per primitive, each ending in the reference bits. -/
def emitCase (c : Case) : Array String := #[
  s!"add {hex c.x} {hex c.y} {hex (modelAdd c.x c.y)}",
  s!"mul {hex c.x} {hex c.y} {hex (modelMul c.x c.y)}",
  s!"div {hex c.x} {hex c.y} {hex (modelDiv c.x c.y)}",
  s!"sqrt {hex c.x} {hex (modelSqrt c.x)}",
  s!"fma {hex c.x} {hex c.y} {hex c.z} {hex (modelFma c.x c.y c.z)}"]

/--
Emit the curated cases, the curated fused triples, and up to `draws` random cases, each line
carrying the reference result bits. Random cases draw three operands and skip the entire case
if any operand is a NaN. This stream differs from `sweep`, which draws two operands per case.
-/
def emitLines (draws : Nat) (seed : UInt32) : Array String := Id.run do
  let mut out : Array String := #[]
  for c in curatedCases do
    out := out ++ emitCase { c with z := c.y }
  for c in curatedFmaCases do
    out := out.push s!"fma {hex c.x} {hex c.y} {hex c.z} {hex (modelFma c.x c.y c.z)}"
  let mut s : UInt32 := if seed == 0 then 1 else seed
  for _ in [0:draws] do
    s := xorshift s
    let x := s
    s := xorshift s
    let y := s
    s := xorshift s
    let z := s
    unless isNaNBits x || isNaNBits y || isNaNBits z do
      out := out ++ emitCase { what := "sweep", x := x, y := y, z := z }
  return out

/-- Entry point. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (emit, args) ← CLI.orThrow "native_float32_parity" (CLI.takeBoolFlag args "emit-cases")
  let (sweepCount, args) ←
    CLI.orThrow "native_float32_parity" (CLI.takeNatFlag args "sweep" (default := 0))
  let (seed, args) ← CLI.orThrow "native_float32_parity" (CLI.takeSeed args (default := 1))
  CLI.requireNoArgs "native_float32_parity" args
  if emit then
    for line in emitLines sweepCount (UInt32.ofNat seed) do
      IO.println line
    return
  IO.println "== native binary32 parity (Float32 versus configured binary32) =="
  IO.println s!"curated cases: {curatedCases.size}, four primitives each"
  for c in curatedCases do
    IO.println (caseRows c)
  let curated := runCases curatedCases
  IO.println curated.render
  if sweepCount > 0 then
    let swept := sweep sweepCount (UInt32.ofNat seed)
    IO.println s!"sweep: {sweepCount} operand pairs, seed {seed}"
    IO.println swept.render
    unless swept.clean do
      throw <| IO.userError "native binary32 parity: sweep disagrees with configured binary32"
  IO.println "fused multiply-add has no Float32 primitive in Lean; reference bits only:"
  for c in curatedFmaCases do
    IO.println s!"  [{c.what}] {hex c.x} {hex c.y} {hex c.z}"
    IO.println <|
      s!"    fma {hex (modelFma c.x c.y c.z)}" ++
        s!"  two roundings {hex (modelMulThenAdd c.x c.y c.z)}"
  IO.println "NaN inputs are an oracle limitation, not a kernel disagreement:"
  for payload in [0x7FAC6DE8, 0xFFAC6DE8, 0x7F800001] do
    IO.println s!"  {nanOracleReport (UInt32.ofNat payload)}"
  unless curated.clean do
    throw <| IO.userError "native binary32 parity: curated cases disagree with configured binary32"

end NN.Tests.Floats.NativePrimitiveParity
