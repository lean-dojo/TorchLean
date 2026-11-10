/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Command
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.NativeDispatch
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Rounding.Runtime
/-!
# Native binary32 parity: the test behind the native arithmetic contract

`TorchLean.LibTorch.Contract.Agreement` is the agreement assumption for a
native operation against FloatLib's: non-NaN results must have identical bits, while two NaN
results may have different encodings. This regression harness compares selected host results with
the binary32 reference `ExecFloat.Binary 8 23`. A pass is evidence for those inputs on this host's
arithmetic, and the contract itself, quantified over all inputs and covering the GPU, stays an
assumption.

Lean's native `Float32` calls host binary32 arithmetic for `add`, `mul`, `div` and `sqrt`, which
the harness compares directly. Core has no `Float32` fused multiply-add, so `fma` is checked on the
CUDA side only: `--emit-cases` prints the reference bits of all five operations, and
`scripts/checks/cuda.sh parity` compares host `std::fma`, ATen tensor operations and TorchLean's
production C interface with them.

`Float32.ofBits` canonicalizes NaNs, while `ExecFloat.Binary 8 23` keeps their sign and payload
when it quiets them, so the native conversion cannot serve as a payload-preserving oracle. The
random cases that `--emit-cases --sweep N` adds skip NaN operands. The gap is a limit of the host
oracle, and no kernel disagreement lies behind it.

Run:
  `scripts/lake.sh exe native_float32_parity`
  `scripts/lake.sh exe native_float32_parity --emit-cases`

The first form checks the curated cases with `run`, and the second prints one case per line for
the CUDA parity script to read.
-/

@[expose] section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32)

namespace TorchLean.Test.NativePrimitives

/-- Zero-padded lowercase hex, so that bit patterns line up in a column when printed. -/
def hex (u : UInt32) : String :=
  "0x" ++ u.toBitVec.toHex

/--
One test case: a label and the operand bits, shared by every primitive.

`z` is read only by `fma`, so the binary and unary cases leave it at zero. One structure for all of
them keeps a single case list, which is the shape the emitted file for the CUDA side needs.
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
Cases at rounding boundaries, signed zeros, subnormal results, overflow and invalid operations.
Each runs through all four primitives, and its label says what it exercises. For example
`0x3F800000 + 0x33800000` is `1 + 2^-24`, halfway between `1` and the next float, and ties to even
give `1`. The subnormal cases tell gradual underflow from flush-to-zero. The fused triples are in
`curatedFmaCases`.
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
exact product needs more than 24 bits and the addend then cancels part of it.
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
  Binary.toBits32 (Binary.sqrt (mode := .nearest) (ofBits32 x))

@[inherit_doc modelAdd]
def modelFma (x y z : UInt32) : UInt32 :=
  Binary.toBits32
    (Binary.fma (mode := .nearest) (ofBits32 x) (ofBits32 y) (ofBits32 z))

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

/-- The printed line: the operation and operands, then the native and the model result bits. -/
def Mismatch.render (m : Mismatch) : String :=
  s!"{m.op}({m.operands}): native {hex m.native} model {hex m.model}"

/--
Running counts for one primitive.

Only the first four mismatches are kept, so a failing run still produces a bounded report.
-/
structure Tally where
  /-- Comparisons performed. -/
  checked : Nat := 0
  /-- Comparisons that disagreed. -/
  failed : Nat := 0
  /-- The first four disagreements, for the report. -/
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

/-- Whether every comparison in the report agreed. -/
def Report.clean (r : Report) : Bool :=
  r.add.failed == 0 && r.mul.failed == 0 && r.div.failed == 0 && r.sqrt.failed == 0

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

/-- Throw unless native `Float32` addition, multiplication, division and square root match
configured binary32 bit for bit on every curated case. -/
public def run : IO Unit := private_decl% (do
  let curated := runCases curatedCases
  unless curated.clean do
    throw <| IO.userError
      s!"native binary32 parity: curated cases disagree with configured binary32\n{curated.render}")

/--
Marsaglia's xorshift32 generator (Marsaglia, "Xorshift RNGs", J. Stat. Soft. 8(14), 2003). A named
generator keeps a sweep reproducible where `IO.rand` would not, which matters when the report is
the evidence for a claim: a reader with the same seed sees the same inputs.
-/
def xorshift (s : UInt32) : UInt32 :=
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 17)
  s ^^^ (s <<< 5)

/-- Command-line help for the internal native-parity check. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean native binary32 parity check"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe native_float32_parity [--emit-cases [--sweep N] [--seed S]]"
    , ""
    , "Options:"
    , "  --emit-cases  print cases and their reference bits, one per line, in the format read"
    , "                by scripts/checks/cuda.sh parity"
    , "  --sweep N     with --emit-cases, add up to N random cases, skipping NaN inputs"
    , "                (default 0)"
    , "  --seed S      seed for the random cases (default 1)"
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
if any operand is a NaN.
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

/-- Check the curated cases with `run`, which fails if a comparison disagrees, and print how many
agree. `--emit-cases` prints the reference stream instead, and `--sweep N` adds up to `N` random
cases to that stream. -/
public def main (args : List String) : IO Unit := private_decl% (do
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
  run
  IO.println s!"native binary32 parity: {curatedCases.size} curated cases agree")

end TorchLean.Test.NativePrimitives
