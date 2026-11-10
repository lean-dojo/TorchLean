/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Trainer

/-!
# Native float conversion

`run` checks that `ofFloat` on `Float32` at round-to-nearest, which goes through the native
binary64 to binary32 conversion, rounds 0.1 up to the word `0x3DCCCCCD`.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model)

namespace TorchLean.Test
namespace DirectedConversion

/-- Fail with a labeled message unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw (IO.userError s!"directed conversion: {label}")

/-- The binary32 word of a native conversion. -/
def native (x : Float) (mode : Model.IEEERoundingMode) : UInt32 :=
  (ofFloat (α := Float32) x mode).toBits

/-- Run the native conversion check. -/
def run : IO Unit := do
  expect "native RN(0.1) rounds up" (native 0.1 .nearest == 0x3DCCCCCD)

end DirectedConversion
end TorchLean.Test
