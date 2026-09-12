/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Arithmetic
public import NN.Verification.Builtin.Lowering.API
public import NN.MLTheory.CROWN.Extras.BoundOpsIEEE32Exec

/-!
# TorchLean Verification Lowering

Umbrella import for the graph builder and the public TorchLean-to-verifier-IR lowering API.
-/

@[expose] public section

namespace NN.Verification.Builtin

open NN.MLTheory.CROWN

/--
Dispatch an executable bound-propagation computation under arithmetic semantics with explicit
outward endpoint operations, including nonlinear division and normalization bounds.

Native `Float32` and the bit-level `IEEE32Exec` backend meet that interface. Complex
arithmetic is rejected instead of receiving an invalid real-valued bound implementation.
-/
def withBoundArithmetic
    (arithmetic : TorchLean.Runtime.Arithmetic)
    (k : ∀ {α : Type}, [TorchLean.Storage α] →
      [Context α] → [ToString α] → [TorchLean.Runtime.FromFloat α] →
      [BoundOps α] → [NonlinearBoundOps α] → IO Unit) :
    IO Unit := do
  match arithmetic with
  | .native =>
      k (α := Float32)
  | .ieee =>
      k (α := TorchLean.Floats.IEEE754.IEEE32Exec)
  | .complex =>
      throw <| IO.userError
        "bound propagation currently supports real-valued arithmetic only"

/-- Parse and log arithmetic semantics, then run `withBoundArithmetic`. -/
def runWithBoundArithmetic
    (title : String) (args : List String)
    (k : ∀ {α : Type}, [TorchLean.Storage α] →
      [Context α] → [ToString α] → [TorchLean.Runtime.FromFloat α] →
      [BoundOps α] → [NonlinearBoundOps α] → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (arithmetic, rest) ←
    match TorchLean.Runtime.Arithmetic.parseAndStrip args with
    | .ok parsed => pure parsed
    | .error msg => throw <| IO.userError msg
  unless rest.isEmpty do
    throw <| IO.userError s!"unexpected arguments: {rest}"
  TorchLean.Runtime.Arithmetic.log arithmetic
  withBoundArithmetic arithmetic (fun {α} => k (α := α))

end NN.Verification.Builtin
