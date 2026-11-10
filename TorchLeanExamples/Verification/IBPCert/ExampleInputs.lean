/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.IBP

/-!
# IBP certificate example inputs

The deterministic center and input box of the IBP certificate example.
-/

@[expose] public section

namespace TorchLean.Examples.IBPCert

open CROWN
open TorchLean.Tensor

/-- Center vector `[1, 2, ..., dim]`, used by the IBP certificate example. -/
def naturalCenter (dim : Nat) : Tensor Float [dim] :=
  Tensor.generate [dim] fun
    | [i] => Float.ofNat (i + 1)
    | _ => 0.0

/-- The $L^\infty$ box of radius `eps` around the center vector $[1,2,\ldots,\mathrm{dim}]$. -/
def naturalBox (dim : Nat) (eps : Float) : FlatBox Float :=
  FlatBox.lInfBall (naturalCenter dim) eps

end TorchLean.Examples.IBPCert
