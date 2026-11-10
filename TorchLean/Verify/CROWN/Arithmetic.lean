/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Arithmetic
public import TorchLean.Verify.CROWN.Op.Imported

/-!
# Arithmetic selection for bound propagation

`withArithmetic` runs a bound-propagation computation under native `Float32` or the bit-level
`ExecFloat.Binary 8 23` and rejects complex arithmetic. This module publicly imports the bound
passes on imported graphs of `TorchLean.Verify.CROWN.Op.Imported`, and through it
`TorchLean.NN.IR.Lowering`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)

namespace TorchLean.CROWN

/--
Run an executable bound-propagation computation `k` at the scalar type of the chosen arithmetic,
whose instances supply directed endpoint operations (`BoundOps`) and nonlinear enclosures such as
division and square root (`NonlinearBoundOps`).

The `.native` semantics runs `k` at `Float32` and `.ieee` at the bit-level `ExecFloat.Binary 8 23`.
Complex arithmetic is refused with a user error, since a real-valued bound implementation would be
invalid for it. Both choices land on the binary32 grid by different routes: `.native` widens each
host result by one adjacent value (`instBoundOpsFloat32`), and `.ieee` rounds each operation down
or up (`instBoundOpsBinary32`).
-/
def withArithmetic
    (arithmetic : Arithmetic)
    (k : ∀ {α : Type}, [Storage α] →
      [Context α] → [ToString α] → [FromFloat α] →
      [BoundOps α] → [NonlinearBoundOps α] → IO Unit) :
    IO Unit := do
  match arithmetic with
  | .native =>
      k (α := Float32)
  | .ieee =>
      k (α := (ExecFloat.Binary 8 23))
  | .complex =>
      throw <| IO.userError
        "bound propagation supports real-valued arithmetic only"

end TorchLean.CROWN
