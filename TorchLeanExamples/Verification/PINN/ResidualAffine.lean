/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.ExecCROWN

/-!
# ResidualAffine

Affine relaxations for assembling PDE residual bounds: CROWN bounds on the network output `u(x)`,
from the box of every node (`crownUBoundsNodewise`) or from one objective
(`crownUBoundsBackward`), on the network's graph of registered operators.

References:
- CROWN / DeepPoly-style affine bounds: `https://arxiv.org/abs/1811.00866`
- alpha,beta-CROWN (context): `https://arxiv.org/abs/2103.06624`
-/

@[expose] public section


namespace TorchLean.Examples.PINN

open CROWN
open Graph
open TorchLean.Tensor

/-- CROWN bounds for the scalar output `u` (or the sum of outputs) of the graph `G` with payload
`P` from the box `xB` at its input: the output's box in full CROWN (`Op.crownBoxes`: each node's
box within its backward affine bounds on the input box), from the interval table `ibp`. -/
def crownUBoundsNodewise (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float)
    (ibp : Array (Option (FlatBox Float))) : Option (Float × Float) :=
  let result : Except String (FlatBox Float) := do
    let some inputId := Op.inputId? G 0 | throw "the graph has no input 0"
    Op.outputBox? G
      (Op.crownBoxes G P { inputId, inputDim := xB.dim } (Op.inputSeeds G [xB]) ibp xB) 0
  match result with
  | .ok outB =>
      let ulo := Tensor.sum outB.lo
      let uhi := Tensor.sum outB.hi
      some (ulo, uhi)
  | .error _ => none

/-- Objective-dependent (backward/dual) CROWN bounds for the scalar output `u` (or the sum of
outputs) of the graph `G` with payload `P` from the box `xB` at its input and the interval table
`ibp`. -/
def crownUBoundsBackward (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float)
    (ibp : Array (Option (FlatBox Float))) : Option (Float × Float) :=
  let outDim : Nat :=
    match Op.outputBox? G ibp 0 with
    | .ok B => B.dim
    | .error _ => 0
  let objV : Tensor Float [outDim] := Tensor.full (α := Float) [outDim] 1.0
  let obj : FlatTensor Float := { n := outDim, v := objV }
  match Op.backwardObjectiveBox? G P ibp xB 0 obj with
  | .ok outB =>
      let ulo := Tensor.sum outB.lo
      let uhi := Tensor.sum outB.hi
      some (ulo, uhi)
  | .error _ => none


end TorchLean.Examples.PINN
