/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import FloatLib.Floats.Formats.IEEE754.Native
public import NN.Runtime.Autograd.Model.Dual
public import NN.Spec.Core.Complex
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Core.FloatInstances.Angle
public import NN.API.CLI

/-!
# Runtime Arithmetic

Runtime commands may choose native binary32, reference IEEE binary32, or complex arithmetic.
`Arithmetic` records that semantic choice. `FromFloat` supplies conversion from Lean binary64
`Float` literals and scalar-specific rounding for parameter validation.

The generic runtime dispatcher supports all three modes. The runtime-selected `trainer.open` and
`trainer.train` paths accept the two real modes; `trainer.openTyped` chooses its scalar directly on
CPU. Explicit complex training uses `autograd.complex.grad` for a real objective and `nn.sgdStep` on
complex state; its predictions and `Checkpoint.State` retain both components.

Model definitions remain polymorphic over the existing `Context α` interface. The arithmetic choice
does not replace that mathematical interface; it selects a concrete executable element
representation that satisfies it. Proof-only models such as `ℝ` and rounded-real binary32 are
selected directly in theorems rather than through a command-line flag.
-/

@[expose] public section


namespace TorchLean
namespace Runtime

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/--
Conversion from Lean `Float` constants into a selected runtime arithmetic representation.
-/
class FromFloat (α : Type) where
  /-- Convert a Lean binary64 `Float` literal into this runtime element representation. -/
  ofFloat : Float → α
  /--
  Represent the rounding of `ofFloat` in binary64 for parameter-domain checks.

  The default validates only the supplied binary64 value, preserving custom scalar instances.
  Finite-precision instances override this to reject values that leave the required domain after
  conversion. This capability needs no arithmetic or order operations from `Context`.
  -/
  roundForValidation : Float → Float := id

/-- Convert a Lean `Float` literal into a TorchLean runtime arithmetic representation. -/
def ofFloat {α : Type} [FromFloat α] (x : Float) : α :=
  FromFloat.ofFloat x

/-- `Float` values inject into the same type by identity. -/
instance : FromFloat Float where
  ofFloat := id

/-- Round binary64 literals to Lean's native binary32 representation. -/
instance : FromFloat Float32 where
  ofFloat := Float.toFloat32
  roundForValidation x := x.toFloat32.toFloat

/-- Inject binary64 literals into the executable IEEE-754 binary32 backend. -/
instance : FromFloat (ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)) where
  ofFloat x := ExecFloat.Binary.ofFloat32 x.toFloat32
  roundForValidation x := x.toFloat32.toFloat

/--
Round the exact binary64 source value once into any configured binary format.

The binary32 specialization above retains its native conversion path. Validation converts the
rounded destination value back to binary64, exposing overflow, underflow, and domain changes.
Explicit scalar-valued tensors and state do not use this conversion.
-/
instance (priority := 100) configuredBinaryFromFloat
    {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] :
    FromFloat (ExecFloat (Configured.Family format code plan)) :=
  let convert : Float → ExecFloat (Configured.Family format code plan) := fun value =>
    ExecFloat.Binary.ofModel
      (Model.cast _ format (ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat value)))
  { ofFloat := convert
    roundForValidation value :=
      ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel
        (Model.cast format _ (ExecFloat.Binary.toModel (convert value)))) }

/--
Inject binary64 literals into the dual-number backend used by the runtime autograd engine.

We interpret a literal as a primal value with zero tangent/adjoint component.
-/
instance {α : Type} [FromFloat α] [Zero α] :
    FromFloat (Runtime.Autograd.Model.Dual α) where
  ofFloat x := Runtime.Autograd.Model.Dual.ofPrimal (ofFloat x)
  roundForValidation := FromFloat.roundForValidation (α := α)

/-- Inject binary64 literals into TorchLean's complex representation with zero imaginary part. -/
instance {α : Type} [FromFloat α] [Zero α] : FromFloat (TorchLean.Complex α) where
  ofFloat x := ⟨ofFloat (α := α) x, 0⟩
  roundForValidation := FromFloat.roundForValidation (α := α)

/--
Arithmetic semantics for runnable executables.

This is a runtime selection mechanism used by example programs; the core library itself is
parametric in the element type `α`.

Unlike a PyTorch per-tensor dtype, this choice fixes one arithmetic semantics for the complete run:

- `.native` uses Lean's native `Float32` operations on CPU and binary32 CUDA storage on GPU,
- `.ieee` uses FloatLib's configured software IEEE-754 binary32 arithmetic,
- `.complex` uses TorchLean's complex representation with binary32 real and imaginary components.
-/
inductive Arithmetic where
  | native
  | ieee
  | complex
  deriving Repr, DecidableEq

namespace Arithmetic

/-- Stable spelling used by command-line flags. -/
def cliName : Arithmetic → String
  | .native => "native"
  | .ieee => "ieee"
  | .complex => "complex"

/-- Log a short description of the selected arithmetic semantics. -/
def log : Arithmetic → IO Unit
  | .native =>
      IO.println "[TorchLean] arithmetic: native binary32"
  | .ieee =>
      IO.println "[TorchLean] arithmetic: IEEE-754 binary32 reference"
  | .complex =>
      IO.println "[TorchLean] arithmetic: complex over executable IEEE-754 binary32"

instance : ToString Arithmetic where
  toString := cliName

/-- Parse the value of `--arithmetic`. -/
def parse (value : String) : Except String Arithmetic :=
  match value with
  | "native" => pure .native
  | "ieee" => pure .ieee
  | "complex" => pure .complex
  | _ => throw s!"unknown --arithmetic {value} (supported: native | ieee | complex)"

/-- Parse and remove `--arithmetic`, using `default` when the flag is absent. -/
def parseAndStrip (arguments : List String) (default : Arithmetic := .native) :
    Except String (Arithmetic × List String) := do
  let (value?, remainingArguments) ←
    TorchLean.CLI.takeFlagValue? arguments "arithmetic"
  match value? with
  | none => pure (default, remainingArguments)
  | some value => pure (← parse value, remainingArguments)

/--
Run `continuation` under the type selected by `arithmetic`.
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

end Runtime
end TorchLean
