/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Robustness.Digits

/-!
# Digits CROWN Verdict

`crownCertifiesLabel` must read each lower endpoint from the lower affine form. The graph below
computes `1 - relu x` and the constant `0.5` on `x ∈ [-2, 1]`. The upper affine form of the first
logit is the constant 1, so its minimum over the box is 1 even though the logit reaches 0 at
`x = 1`, where the constant logit wins.
-/

public section

namespace NN.Tests.Verification.DigitsCrown

open _root_.Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open NN.Verification.Robustness.Digits (crownCertifiesLabel)

private def graph : NN.IR.Graph := ⟨#[
  { id := 0, parents := #[], kind := .input, outShape := [1] },
  { id := 1, parents := #[0], kind := .relu, outShape := [1] },
  { id := 2, parents := #[1], kind := .linear, outShape := [2] }]⟩

private def box (lo hi : Float) : FlatBox Float :=
  { dim := 1, lo := [lo], hi := [hi] }

private def params (xB : FlatBox Float) : ParamStore Float :=
  { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float)).insert 0 xB
    linearWB := ({} : Std.HashMap Nat (LinParams Float)).insert 2
      { m := 2, n := 1, w := [[-1], [0]], b := [1, 0.5] } }

private def verdict (xB : FlatBox Float) (label : Nat) : IO Bool :=
  match crownCertifiesLabel graph (params xB) xB 0 2 label with
  | .ok ok => pure ok
  | .error e => throw <| IO.userError s!"digits CROWN verdict failed: {e}"

def run : IO Unit := do
  let crossing := box (-2) 1
  -- The upper form alone would give the first logit the point range [1, 1].
  let affines := runAffine graph (params crossing) { inputId := 0, inputDim := 1 }
    (runIBP graph (params crossing))
  let some upper := affines[2]!
    | throw <| IO.userError "digits CROWN: upper affine form missing"
  let separated :=
    if hIn : crossing.dim = upper.inDim then
      let upperOnly := upper.evalOnFlatBox crossing hIn
      if h0 : 0 < upper.outDim then upperOnly.lo.getScalar ⟨0, h0⟩ > 0.5 else false
    else false
  unless separated do
    throw <| IO.userError "digits CROWN: the test graph no longer separates the two forms"
  if ← verdict crossing 0 then
    throw <| IO.userError "digits CROWN certified label 0, which loses at x = 1"
  if ← verdict crossing 1 then
    throw <| IO.userError "digits CROWN certified label 1, which loses at x = -2"
  -- On x ∈ [-2, -1] the first logit is exactly 1 and does win.
  unless ← verdict (box (-2) (-1)) 0 do
    throw <| IO.userError "digits CROWN rejected label 0 on a box where it always wins"
  IO.println "digits CROWN verdict: ok"

end NN.Tests.Verification.DigitsCrown
