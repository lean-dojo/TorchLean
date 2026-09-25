/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.FiniteArtifact

/-! Focused acceptance, rounding, coverage, and malformed-artifact regressions. -/

public section

namespace NN.Tests.Verification.FiniteArtifact

open _root_.Spec TorchLean TorchLean.Tensor
open FloatLib.Floats (ExecFloat)
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open NN.Verification.Cert.NodeReplay
open NN.Verification.Cert.FiniteArtifact

private def expect (name : String) (actual expected : Bool) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError s!"finite binary32 artifact: {name}: got {actual}"

private def vec (x : ExecFloat.Binary 8 23) : Tensor (ExecFloat.Binary 8 23) [1] :=
  Tensor.ofFn fun _ => x

private def scalarLayer (weight bias : ExecFloat.Binary 8 23) :
    LinParams (ExecFloat.Binary 8 23) :=
  ⟨1, 1, Tensor.matrix fun _ _ => weight, vec bias⟩

private def graph : Graph :=
  ⟨#[
    ⟨0, #[], .input, [1]⟩,
    ⟨1, #[0], .linear, [2]⟩,
    ⟨2, #[1], .relu, [2]⟩,
    ⟨3, #[2], .linear, [1]⟩
  ]⟩

private def params : ParamStore (ExecFloat.Binary 8 23) :=
  { inputBoxes := ({} : Std.HashMap Nat _).insert 0 ⟨1, vec (-1), vec 1⟩
    linearWB := ({} : Std.HashMap Nat _)
      |>.insert 1 ⟨2, 1, Tensor.matrix (fun i _ => if i.val = 0 then 1 else -1),
        Tensor.full [2] 0⟩
      |>.insert 3 ⟨1, 2, Tensor.full [1, 2] 1, vec 0⟩ }

private def replay (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (inputDim : Nat) (alpha : Array (Option (FlatTensor (ExecFloat.Binary 8 23)))) :
    CROWNNodeCoreCertificate := Id.run do
  let ibp := runIBP g ps
  let ctx : AffineCtx := ⟨0, inputDim⟩
  let mut crown := Array.replicate g.nodes.size none
  for id in [:g.nodes.size] do
    let entry := NN.MLTheory.CROWN.Cert.alphaCrownStepNode?
      g.nodes ps ibp alpha crown ctx id
    crown := crown.set! id entry
  return ⟨ctx, ibp, crown, alpha⟩

private def certificate : CROWNNodeCoreCertificate :=
  replay graph params 1 #[none, none, some ⟨2, Tensor.full [2] 0.5⟩, none]

private def query (bound : ExecFloat.Binary 8 23) (strict : Bool) : OutputQuery :=
  ⟨3, scalarLayer 1 (-bound), strict⟩

private def roundingRegression : IO Unit := do
  let g : Graph := ⟨#[
    ⟨0, #[], .input, [1]⟩,
    ⟨1, #[0], .linear, [1]⟩,
    ⟨2, #[1], .relu, [1]⟩,
    ⟨3, #[2], .linear, [1]⟩
  ]⟩
  -- (1 + 2^-23)^2 = 1 + 2^-22 + 2^-46, whose nearest binary32 value is smaller.
  let w : ExecFloat.Binary 8 23 := 1.00000011920928955078125
  let ps : ParamStore (ExecFloat.Binary 8 23) :=
    { inputBoxes := ({} : Std.HashMap Nat _).insert 0 ⟨1, vec 1, vec 1⟩
      linearWB := ({} : Std.HashMap Nat _)
        |>.insert 1 (scalarLayer w 0)
        |>.insert 3 (scalarLayer w 0) }
  let cert := replay g ps 1 #[none, none, some ⟨1, vec 1⟩, none]
  expect "inward-rounded transcript still matches binary32 replay" (replayAccepts g ps cert) true
  let some decoded := decode g ps cert (query 2 true)
    | throw <| IO.userError "finite rounding example did not decode"
  expect "same-artifact real dominance rejects inward rounding" decoded.check false
  expect "combined checker rejects inward rounding" (accepts g ps cert (query 2 true)) false

/-- Run the bounded fragment checks without changing the shared test suite. -/
def run : IO Unit := do
  expect "finite linear/ReLU graph and strict margin"
    (accepts graph params certificate (query 1.5 true)) true
  expect "strict boundary" (accepts graph params certificate (query 1 true)) false
  expect "non-strict boundary" (accepts graph params certificate (query 1 false)) true
  expect "unsafe requested margin" (accepts graph params certificate (query 0.5 false)) false
  expect "output points at an intermediate node"
    (accepts graph params certificate { query 2 true with outputId := 2 }) false
  expect "output beyond graph"
    (accepts graph params certificate { query 2 true with outputId := 4 }) false
  expect "missing output entry"
    (accepts graph params { certificate with crown := certificate.crown.set! 3 none }
      (query 2 true)) false
  expect "missing intermediate entry"
    (accepts graph params { certificate with crown := certificate.crown.set! 1 none }
      (query 2 true)) false
  expect "certificate array too short"
    (accepts graph params { certificate with crown := certificate.crown.pop }
      (query 2 true)) false
  expect "wrong designated input"
    (accepts graph params { certificate with ctx := ⟨1, 1⟩ } (query 2 true)) false
  expect "wrong input dimension"
    (accepts graph params { certificate with ctx := ⟨0, 2⟩ } (query 2 true)) false
  let wrongBounds : FlatAffineBounds (ExecFloat.Binary 8 23) :=
    NN.MLTheory.CROWN.Cert.boundsIdentity 2
  expect "artifact shape mismatch"
    (accepts graph params
      { certificate with crown := certificate.crown.set! 3 (some wrongBounds) }
      (query 2 true)) false
  let nan : ExecFloat.Binary 8 23 := ExecFloat.Binary.canonicalNaN
  let inf : ExecFloat.Binary 8 23 := ExecFloat.Binary.infinity false
  for exceptional in [nan, inf] do
    expect "nonfinite input box"
      (accepts graph { params with
        inputBoxes := params.inputBoxes.insert 0 ⟨1, vec (-1), vec exceptional⟩ }
        certificate (query 2 true)) false
    expect "nonfinite graph parameter"
      (accepts graph { params with
        linearWB := params.linearWB.insert 3
          ⟨1, 2, Tensor.full [1, 2] exceptional, vec 0⟩ }
        certificate (query 2 true)) false
    expect "nonfinite query"
      (accepts graph params certificate (query exceptional true)) false
    let forged : FlatAffineBounds (ExecFloat.Binary 8 23) :=
      ⟨1, 1, ⟨Tensor.full [1, 1] exceptional, vec 0⟩,
        ⟨Tensor.full [1, 1] exceptional, vec 0⟩⟩
    expect "nonfinite artifact coefficient"
      (accepts graph params
        { certificate with crown := certificate.crown.set! 3 (some forged) }
        (query 2 true)) false
    expect "nonfinite alpha"
      (accepts graph params
        { certificate with
          alpha := certificate.alpha.set! 2 (some ⟨2, Tensor.full [2] exceptional⟩) }
        (query 2 true)) false
  for slope in [(-1 : ExecFloat.Binary 8 23), 2] do
    expect "alpha outside unit interval"
      (accepts graph params
        { certificate with
          alpha := certificate.alpha.set! 2 (some ⟨2, Tensor.full [2] slope⟩) }
        (query 2 true)) false
  expect "wrong alpha dimension"
    (accepts graph params
      { certificate with alpha := certificate.alpha.set! 2 (some ⟨1, vec 0.5⟩) }
      (query 2 true)) false
  expect "reversed input box"
    (accepts graph { params with
      inputBoxes := params.inputBoxes.insert 0 ⟨1, vec 1, vec (-1)⟩ }
      certificate (query 2 true)) false
  expect "missing graph parameter"
    (accepts graph { params with linearWB := params.linearWB.erase 1 }
      certificate (query 2 true)) false
  for node in [
      (⟨1, #[1], .linear, [2]⟩ : Node),
      ⟨1, #[0, 0], .linear, [2]⟩,
      ⟨7, #[0], .linear, [2]⟩,
      ⟨1, #[0], .linear, [1, 2]⟩,
      ⟨1, #[], .input, [2]⟩,
      ⟨1, #[0], .sigmoid, [2]⟩] do
    expect "unsupported topology, operation, or shape"
      (accepts ⟨graph.nodes.set! 1 node⟩ params certificate (query 2 true)) false
  expect "empty graph" (accepts ⟨#[]⟩ params certificate (query 2 true)) false
  let empty : Tensor (ExecFloat.Binary 8 23) [0] := Tensor.ofFn Fin.elim0
  expect "empty input dimension"
    (accepts ⟨#[⟨0, #[], .input, [0]⟩]⟩
      { params with inputBoxes := params.inputBoxes.insert 0 ⟨0, empty, empty⟩ }
      certificate (query 2 true)) false
  let emptyQuery : OutputQuery :=
    ⟨3, ⟨0, 1, Tensor.full [0, 1] 0, empty⟩, false⟩
  expect "empty query dimension" (accepts graph params certificate emptyQuery) false
  let wrongQuery : OutputQuery := ⟨3, ⟨1, 2, Tensor.full [1, 2] 1, vec (-2)⟩, true⟩
  expect "wrong query dimension" (accepts graph params certificate wrongQuery) false
  let multiple (second : ExecFloat.Binary 8 23) : OutputQuery :=
    ⟨3, ⟨2, 1, Tensor.full [2, 1] 1,
      Tensor.ofFn fun i => if i.val = 0 then -2 else second⟩, true⟩
  expect "every output constraint holds"
    (accepts graph params certificate (multiple (-2))) true
  expect "the second output constraint fails"
    (accepts graph params certificate (multiple (-0.5))) false
  let identityGraph : Graph := ⟨#[⟨0, #[], .input, [1]⟩]⟩
  let identityCert := replay identityGraph params 1 #[none]
  expect "input-only graph has a covered output"
    (accepts identityGraph params identityCert ⟨0, scalarLayer 1 (-2), true⟩) true
  let zeroOutputGraph : Graph := ⟨#[⟨0, #[], .input, [1]⟩, ⟨1, #[0], .linear, [0]⟩]⟩
  expect "empty linear output"
    (accepts zeroOutputGraph
      { params with linearWB := params.linearWB.insert 1 ⟨0, 1, Tensor.full [0, 1] 0, empty⟩ }
      certificate (query 2 true)) false
  roundingRegression
  IO.println "  Finite binary32 artifact: dominance, margins, coverage, and rejection tests passed"

end NN.Tests.Verification.FiniteArtifact
