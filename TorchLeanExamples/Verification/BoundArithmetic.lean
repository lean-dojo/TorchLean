/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Arithmetic

/-!
# Choosing the bound arithmetic of a workflow

`run` reads the arithmetic of a verification workflow from its command
arguments, rejects any other argument and runs the workflow under `withArithmetic`, on native
`Float32` or the bit-level `Binary32`.
-/

@[expose] public section

namespace TorchLean.Examples.BoundArithmetic

open CROWN

/-- Print the workflow's title, read its arithmetic from `args` and run `k` under it. -/
def run
    (title : String) (args : List String)
    (k : ∀ {α : Type}, [Storage α] →
      [Context α] → [ToString α] → [FromFloat α] →
      [BoundOps α] → [NonlinearBoundOps α] → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (arithmetic, rest) ← IO.ofExcept (Arithmetic.parseAndStrip args)
  unless rest.isEmpty do
    throw <| IO.userError s!"unexpected arguments: {rest}"
  Arithmetic.log arithmetic
  withArithmetic arithmetic (fun {α} => k (α := α))

end TorchLean.Examples.BoundArithmetic
