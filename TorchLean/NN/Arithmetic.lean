/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import FloatLib.Floats.Formats.IEEE754.Native
public import TorchLean.NN.Dual
public import TorchLean.Tensor.Complex
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.FromFloat
public import TorchLean.NN.CLI.Command

/-!
# Runtime Arithmetic

A runnable command chooses one element arithmetic for the whole run: native binary32, reference
IEEE binary32 in software, or complex numbers over that software binary32. `Arithmetic` records
the choice and `Arithmetic.withRuntime` runs a program at the matching element type. `FromFloat`,
defined in `TorchLean.Tensor.FromFloat`, converts Lean binary64 `Float` values in any IEEE
754-2019 rounding direction and supplies scalar-specific rounding for parameter validation.

`withRuntime` handles all three modes. The runtime-selected `trainer.open` and `trainer.train`
paths accept only the two real modes, and `trainer.openTyped` takes its scalar as an argument and
runs on CPU. Complex training uses `autograd.complex.grad` for a real objective and `nn.sgdStep`
on complex state, and its predictions and `Checkpoint.State` keep both components.

Model definitions are polymorphic over the `Context α` interface. The arithmetic choice
does not replace that mathematical interface. It selects a concrete executable element
representation that satisfies it. Proof-only models such as `ℝ` and rounded-real binary32 are
selected directly in theorems rather than through a command-line flag.
-/

@[expose] public section


namespace TorchLean

open FloatLib.Floats

/--
The element arithmetic of a runnable program, set by the `--arithmetic` flag or
`Trainer.RunConfig.arithmetic`. Where PyTorch picks a dtype per tensor, this one choice covers
every tensor in the run:

- `native` uses Lean's native `Float32` operations on CPU and binary32 CUDA storage on GPU,
- `ieee` uses FloatLib's configured software IEEE-754 binary32 arithmetic,
- `complex` uses TorchLean's complex representation with binary32 real and imaginary components.
-/
inductive Arithmetic where
  | native
  | ieee
  | complex
  deriving Repr, DecidableEq

namespace Arithmetic

/-- The `--arithmetic` value that names this mode, which `parse` reads back. -/
def cliName : Arithmetic → String
  | .native => "native"
  | .ieee => "ieee"
  | .complex => "complex"

/-- Print a one-line `[TorchLean] arithmetic: ...` banner for the selected mode. -/
def log : Arithmetic → IO Unit
  | .native =>
      IO.println "[TorchLean] arithmetic: native binary32"
  | .ieee =>
      IO.println "[TorchLean] arithmetic: IEEE-754 binary32 reference"
  | .complex =>
      IO.println "[TorchLean] arithmetic: complex over executable IEEE-754 binary32"

instance : ToString Arithmetic where
  toString := cliName

/-- Parse an `--arithmetic` value: `native`, `ieee` or `complex`. -/
def parse (value : String) : Except String Arithmetic :=
  match value with
  | "native" => pure .native
  | "ieee" => pure .ieee
  | "complex" => pure .complex
  | _ => throw s!"unknown --arithmetic {value} (supported: native | ieee | complex)"

/--
Parse and remove `--arithmetic` (as `--arithmetic ieee` or `--arithmetic=ieee`), returning the
mode and the other arguments in order. `default` applies when the flag is absent.
-/
def parseAndStrip (arguments : List String) (default : Arithmetic := .native) :
    Except String (Arithmetic × List String) :=
  TorchLean.CLI.takeParsedFlag arguments "arithmetic" default.cliName parse

/--
Run `continuation` at the element type that `arithmetic` selects: `Float32` for `native`, software
binary32 (`ExecFloat.Binary` with 8 exponent and 23 fraction bits) for `ieee`, and
`TorchLean.Complex` over that type for `complex`. A program generic in `α` passes itself as
`withRuntime arithmetic @run`.
-/
def withRuntime
    (arithmetic : Arithmetic)
    (continuation :
      ∀ {α : Type}, [TorchLean.Storage α] → [Context α] → [ToString α] → [FromFloat α] → IO Unit) :
    IO Unit := do
  match arithmetic with
  | .native =>
      continuation (α := Float32)
  | .ieee =>
      continuation (α := ExecFloat.Binary (exponentBits := 8) (fractionBits := 23))
  | .complex =>
      continuation
        (α := TorchLean.Complex (ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)))

end Arithmetic

end TorchLean
