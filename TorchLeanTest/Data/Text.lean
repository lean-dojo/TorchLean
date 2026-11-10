/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Parser
public import TorchLean.NN.Data.Image
public import TorchLean.NN.Data.CSV
public import TorchLeanExamples.Verification.ODE.TextCursor

/-!
# Text Data Boundaries

Regression checks for the one decimal parser and its users (CSV cells, CLI float flags and
certificate numbers), the CSV final newline and the PPM writer. The decimal parser rounds once,
keeps the sign of `-0` and rejects `_` separators. A newline-terminated CSV file has no empty last
row. A PPM with a NaN channel is never written, and its bytes round to nearest. A certificate number
is read as the pair of its roundings toward $-\infty$ and $+\infty$.
-/

@[expose] public section

namespace TorchLean.Test.TextData

/-- Fail with a labeled message unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw (IO.userError s!"text data: {label}")

/-- The decimal parser's value, or `none` on a rejected string. -/
def decimal? (text : String) : Option Float :=
  (Data.parseFloatString "test" text).toOption

/-- The bits of an accepted decimal, or a sentinel that matches no `Float`. -/
def decimalBits (text : String) : Option UInt64 :=
  (decimal? text).map Float.toBits

/-- Check the decimal parser and its three users. -/
def testDecimal : IO Unit := do
  expect "03.30142 rounds once" (decimalBits "03.30142" == some (3.30142 : Float).toBits)
  expect "0.1 rounds once" (decimalBits "0.1" == some (0.1 : Float).toBits)
  expect "1e-3 parses" (decimalBits "1e-3" == some (0.001 : Float).toBits)
  expect "-0 keeps its sign" (decimalBits "-0" == some (-0.0 : Float).toBits)
  expect "1. parses" (decimalBits "1." == some (1.0 : Float).toBits)
  expect "0.5_5 is rejected" (decimal? "0.5_5").isNone
  expect "1e999 is rejected" (decimal? "1e999").isNone
  expect "nan is rejected" (decimal? "nan").isNone
  expect "CLI float flag rounds once"
    ((CLI.parseFloatLit? "03.30142").map Float.toBits ==
      some (3.30142 : Float).toBits)
  expect "CLI float flag keeps -0"
    ((CLI.parseFloatLit? "-0").map Float.toBits == some (-0.0 : Float).toBits)
  expect "CLI float flag rejects separators" (CLI.parseFloatLit? "0.5_5").isNone
  let cursor : Examples.TextCursor := ⟨"3.30142)", 0⟩
  match Examples.TextCursor.parseFloatBounds cursor with
  | .ok ((lo, hi), rest) =>
      expect "certificate number rounds down and up"
        (lo.toBits == 0x400A694EE392E1EF && hi.toBits == 0x400A694EE392E1F0 &&
          rest.position.byteIdx == 7)
  | .error message => throw (IO.userError s!"certificate number: {message}")

/-- A final newline ends the last row rather than starting an empty one. -/
def testCSV (fixtureDirectory : System.FilePath) : IO Unit := do
  let path := fixtureDirectory / "fixapi_rows.csv"
  IO.FS.writeFile path "1,2\n3,4\n"
  match ← Data.CSV.read path { allowEmptyLines := false } with
  | .ok rows => expect "newline-terminated CSV has two rows" (rows.size == 2)
  | .error message => throw (IO.userError s!"newline-terminated CSV: {message}")
  IO.FS.writeFile path "1,2\n\n3,4\n"
  match ← Data.CSV.read path { allowEmptyLines := false } with
  | .ok _ => throw (IO.userError "CSV accepted a blank line with allowEmptyLines := false")
  | .error _ => pure ()

/-- A NaN channel leaves no file behind, and bytes round to nearest. -/
def testPPM (fixtureDirectory : System.FilePath) : IO Unit := do
  let path := fixtureDirectory / "fixapi_image.ppm"
  if ← path.pathExists then IO.FS.removeFile path
  let nan : Float := 0.0 / 0.0
  let bad : Tensor Float [1, 3, 1, 2] :=
    (Tensor.ofFn fun i : Fin 6 => if i.val == 5 then nan else 0.5).reshape [1, 3, 1, 2]
      (by decide)
  let rejected ← try
      Data.Image.writeFirstPPM path bad
      pure false
    catch _ => pure true
  expect "a NaN channel is rejected" rejected
  expect "a rejected image leaves no file" !(← path.pathExists)
  let good : Tensor Float [1, 3, 1, 1] :=
    (Tensor.ofFn fun i : Fin 3 => if i.val == 0 then 0.5 else if i.val == 1 then 0.9999 else 1.0)
      |>.reshape [1, 3, 1, 1] (by decide)
  Data.Image.writeFirstPPM path good
  let text ← IO.FS.readFile path
  expect s!"bytes round to nearest, got {text}" (text == "P3\n1 1\n255\n128 255 255\n")

/-- Run the text-data checks. -/
def run : IO Unit := do
  let fixtureDirectory : System.FilePath := ".lake/build/tmp"
  IO.FS.createDirAll fixtureDirectory
  testDecimal
  testCSV fixtureDirectory
  testPPM fixtureDirectory

end TorchLean.Test.TextData
