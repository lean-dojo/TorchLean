/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Floats.Utils
public import NN.Verification.Cert.NodeReplay
public import NN.Verification.Cert.CROWNNodeCert
public import NN.Verification.Cert.IBPNodeCert
public import NN.Verification.ODE.Parse
public import NN.Verification.PINN.Core
public import NN.Verification.PINN.PdeParse
public import NN.Verification.Robustness.MarginCert
public import NN.Verification.VNNComp.Spec
public import Lean.Data.Json

/-!
# CertificatePreconditions

Regression checks for the executable certificate boundary.

These tests cover side conditions that are easy for an external producer to get wrong: α-CROWN
slopes must be in $[0,1]$, binary elementwise bounds must have matching flattened dimensions, the
true `log` relaxation must only be replayed on positive boxes, verification expressions must follow
their documented power/negation semantics, and interval comparisons must reject non-finite values.
-/

@[expose] public section

open Spec TorchLean
open Lean
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.Cert.NodeReplay
open TorchLean.Floats.IEEE754

namespace Tests
namespace Floats
namespace CertificatePreconditions

/-- Fail the test with `msg` unless a Boolean check succeeds. -/
def expect (msg : String) (b : Bool) : IO Unit := do
  unless b do
    throw <| IO.userError msg

/-- Require an `IO` action to reject malformed certificate input. -/
def expectRejected {α : Type} (msg : String) (act : IO α) : IO Unit := do
  let mut rejected := false
  try
    let _ ← act
  catch _ =>
    rejected := true
  unless rejected do
    throw <| IO.userError msg

/-- Parse JSON for a test fixture, turning parser errors into test failures. -/
def parseJson! (s : String) : IO Json := do
  match Json.parse s with
  | .ok j => pure j
  | .error e => throw <| IO.userError s!"bad test JSON: {e}"

/-- Parse a certificate input box and fail the test when the schema is rejected. -/
def parseInputRegion! (source : String) : IO NN.Verification.Json.BoxRegion := do
  let json ← parseJson! source
  let inputResult : Except String Json :=
    match json.getObjVal? "input" with
    | .ok input => .ok input
    | .error _ => json.getObjVal? "region"
  let input ←
    match inputResult with
    | .ok input => pure input
    | .error e => throw <| IO.userError s!"input-region parser rejected a valid fixture: {e}"
  match NN.Verification.Json.parseBoxRegion "input" input with
  | .ok region => pure region
  | .error e => throw <| IO.userError s!"input-region parser rejected a valid fixture: {e}"

/-- Whether a JSON value passes the complete PINN certificate schema checks. -/
def pinnCertificateAccepted (json : Json) : Bool :=
  match NN.Verification.PINN.parseCertificate json with
  | .ok _ => true
  | .error _ => false

/-- Compare a computed scalar interval pair with its exact expected endpoints. -/
def expectFloatPair (msg : String) (actual : Option (Float × Float))
    (expected : Float × Float) : IO Unit := do
  match actual with
  | some pair =>
      expect msg (pair.1 == expected.1 && pair.2 == expected.2)
  | none =>
      throw <| IO.userError s!"{msg}: expression evaluation failed"

/-- Parse and evaluate an ODE expression on the point interval $u=2$. -/
def evalODEAtTwo (source : String) : Option (Float × Float) := do
  let expr ← NN.Verification.ODE.Parse.parseExpr source |>.toOption
  NN.Verification.ODE.eval (fun x => x)
    { t := (0.0, 0.0), u := (2.0, 2.0) } expr

/-- Parse and evaluate a PDE expression on the point interval $u=2$. -/
def evalPDEAtTwo (source : String) : Option (Float × Float) := do
  let expr ← NN.Verification.PINN.PdeParse.parseExpr (fun _ => none) source |>.toOption
  NN.Verification.PINN.PdeAst.eval
    { u := some (2.0, 2.0), duX := none, duY := none, d2uX := none, d2uY := none }
    expr

def flatBox (lo hi : Fin 2 → Float) : FlatBox IEEE32Exec :=
  { dim := 2
    lo := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn lo)
    hi := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn hi) }

def flatBox3 : FlatBox IEEE32Exec :=
  { dim := 3
    lo := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn (fun _ : Fin 3 => 0.0))
    hi := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn (fun _ : Fin 3 => 1.0)) }

def inputNode (id dim : Nat) : NN.IR.Node :=
  { id := id, parents := #[], kind := .input, outShape := [dim] }

def addGraph : NN.IR.Graph :=
  { nodes := #[
      inputNode 0 2,
      inputNode 1 3,
      { id := 2, parents := #[0, 1], kind := .add, outShape := [2] }
    ] }

def logGraph : NN.IR.Graph :=
  { nodes := #[
      inputNode 0 2,
      { id := 1, parents := #[0], kind := .log, outShape := [2] }
    ] }

def run : IO Unit := do
  let centerRegion ← parseInputRegion!
    "{\"input\":{\"center\":[1.0,2.0,3.0],\"eps\":0.25}}"
  expect "center/epsilon input region did not infer its dimension"
    (centerRegion.dim == 3 && centerRegion.lo == #[0.75, 1.75, 2.75] &&
      centerRegion.hi == #[1.25, 2.25, 3.25])
  let boxRegion ← parseInputRegion!
    "{\"region\":{\"lo\":[-1.0,0.0],\"hi\":[1.0,2.0]}}"
  expect "endpoint input region did not infer its dimension"
    (boxRegion.dim == 2 && boxRegion.lo.size == 2 && boxRegion.hi.size == 2)
  expectRejected "malformed input dimension was treated as absent" <|
    parseInputRegion! "{\"input\":{\"dim\":\"two\",\"lo\":[0.0,0.0],\"hi\":[1.0,1.0]}}"
  expectRejected "declared input dimension did not constrain endpoint lengths" <|
    parseInputRegion! "{\"input\":{\"dim\":3,\"lo\":[0.0,0.0],\"hi\":[1.0,1.0]}}"
  expectRejected "mismatched input endpoint lengths were accepted" <|
    parseInputRegion! "{\"input\":{\"lo\":[0.0],\"hi\":[1.0,2.0]}}"
  expectRejected "negative input radius was accepted" <|
    parseInputRegion! "{\"input\":{\"center\":[0.0,0.0],\"eps\":-0.5}}"
  expectRejected "reversed input interval was accepted" <|
    parseInputRegion! "{\"input\":{\"lo\":[1.0],\"hi\":[0.0]}}"
  expectRejected "mixed input-region schemas were accepted" <|
    parseInputRegion!
      "{\"input\":{\"lo\":[0.0],\"hi\":[1.0],\"center\":[0.5],\"eps\":0.5}}"
  expectRejected "endpoint input region accepted an unrelated radius" <|
    parseInputRegion! "{\"input\":{\"lo\":[0.0],\"hi\":[1.0],\"eps\":0.5}}"

  expect "VNN-COMP accepted a reversed input box" <|
    !NN.Verification.Util.Tensor.boundsOrdered ([1.0, 0.0] : TorchLean.Tensor Float [2])
      [0.0, 1.0]
  expect "VNN-COMP rejected a well-formed input box" <|
    NN.Verification.Util.Tensor.boundsOrdered ([-1.0, 0.0] : TorchLean.Tensor Float [2])
      [1.0, 2.0]

  let lower : Tensor Float [2] := [0.0, 1.0]
  let threshold : Tensor Float [2] := [0.0, 0.0]
  expect "threshold refutation missed the second coordinate" <|
    NN.Verification.Util.Tensor.refutesThreshold lower threshold
  expect "threshold witness silently fell back to another coordinate" <|
    !NN.Verification.Util.Tensor.refutesThresholdAt lower threshold 0
  expect "threshold witness accepted an out-of-range index" <|
    !NN.Verification.Util.Tensor.refutesThresholdAt lower threshold 2
  let nan : Float := Float.ofBits 0x7ff8000000000000
  expect "threshold witness accepted NaN" <|
    !NN.Verification.Util.Tensor.refutesThresholdAt ([nan] : Tensor Float [1]) [0.0] 0
  expect "bound ordering accepted NaN" <|
    !NN.Verification.Util.Tensor.boundsOrdered ([nan] : Tensor Float [1]) [0.0]
  expect "vector decoding accepted a mismatched length" <|
    (NN.Verification.Util.Tensor.vecOfArray 2 #[1.0]).isNone
  expect "matrix decoding accepted a ragged row" <|
    (NN.Verification.Util.Tensor.matOfArray 2 2 #[#[1.0, 0.0], #[1.0]]).isNone
  let term : NN.Verification.VNNComp.VNNLib.Term :=
    { rows := 1, cols := 2, mat := [[2.0, -3.0]], rhs := [-5.0] }
  let box : FlatBox Float := { dim := 2, lo := [1.0, 1.0], hi := [2.0, 2.0] }
  expect "outward row bound failed to refute a separated threshold" <|
    NN.Verification.VNNComp.VNNLib.termRefutedByOutputBox box term
  expect "outward row bound accepted equality as a strict refutation" <|
    !NN.Verification.VNNComp.VNNLib.termRefutedByOutputBox box { term with rhs := [-4.0] }
  expect "output spec accepted a dimension mismatch" <|
    !NN.Verification.VNNComp.VNNLib.termRefutedByOutputBox
      { dim := 1, lo := [1.0], hi := [2.0] } term

  let reversedMarginEntry ← parseJson! <|
    "{\"label\":0,\"logits_lo\":[2.0,0.0],\"logits_hi\":[1.0,0.5]}"
  expectRejected "margin report accepted reversed logit bounds" <|
    NN.Verification.Robustness.MarginCert.checkOneExample 2 reversedMarginEntry

  let validPinnCert ← parseJson! <|
    "{\"pinn\":{\"pde\":\"u\",\"h\":0.1,\"eps\":0.0,\"points\":[0.0]}," ++
    "\"residual_bounds\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"residual_bounds_deriv\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"u_bounds\":[{\"x\":0.0," ++
    "\"u_minus\":{\"lo\":0.0,\"hi\":0.0}," ++
    "\"u\":{\"lo\":0.0,\"hi\":0.0}," ++
    "\"u_plus\":{\"lo\":0.0,\"hi\":0.0}}]}"
  expect "a well-formed PINN certificate was rejected" (pinnCertificateAccepted validPinnCert)
  let negativeSpacing ← parseJson! <|
    "{\"pinn\":{\"pde\":\"u\",\"h\":-0.1,\"eps\":0.0,\"points\":[0.0]}," ++
    "\"residual_bounds\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"residual_bounds_deriv\":{\"lo\":[0.0],\"hi\":[0.0]},\"u_bounds\":[]}"
  expect "a PINN certificate with nonpositive spacing was accepted"
    (!pinnCertificateAccepted negativeSpacing)
  let negativeRadius ← parseJson! <|
    "{\"pinn\":{\"pde\":\"u\",\"h\":0.1,\"eps\":-0.1,\"points\":[0.0]}," ++
    "\"residual_bounds\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"residual_bounds_deriv\":{\"lo\":[0.0],\"hi\":[0.0]},\"u_bounds\":[]}"
  expect "a PINN certificate with a negative perturbation radius was accepted"
    (!pinnCertificateAccepted negativeRadius)
  let missingUBounds ← parseJson! <|
    "{\"pinn\":{\"pde\":\"u\",\"h\":0.1,\"eps\":0.0,\"points\":[0.0]}," ++
    "\"residual_bounds\":{\"lo\":[0.0],\"hi\":[0.0]}," ++
    "\"residual_bounds_deriv\":{\"lo\":[0.0],\"hi\":[0.0]},\"u_bounds\":[]}"
  expect "a PINN certificate with incomplete u_bounds was accepted"
    (!pinnCertificateAccepted missingUBounds)
  let reversedPinnInterval ← parseJson! <|
    "{\"pinn\":{\"pde\":\"u\",\"h\":0.1,\"eps\":0.0,\"points\":[0.0]}," ++
    "\"residual_bounds\":{\"lo\":[1.0],\"hi\":[0.0]}," ++
    "\"residual_bounds_deriv\":{\"lo\":[0.0],\"hi\":[0.0]},\"u_bounds\":[]}"
  expect "a PINN certificate with a reversed residual interval was accepted"
    (!pinnCertificateAccepted reversedPinnInterval)

  IO.println "certificate_preconditions: begin"

  let goodAlpha ← parseJson! "[0.0, 0.75]"
  let some _ ← parseAlphaVec? 2 goodAlpha
    | throw <| IO.userError "valid alpha vector was rejected"

  let badAlpha ← parseJson! "[2.0, 0.5]"
  expectRejected "alpha outside [0,1] was accepted" (parseAlphaVec? 2 badAlpha)

  let nonfiniteAlpha ← parseJson! "[1e999, 0.5]"
  expectRejected "non-finite alpha was accepted" (parseAlphaVec? 2 nonfiniteAlpha)

  let nonfiniteBox ← parseJson! "{\"lo\":[0.0,1e999],\"hi\":[1.0,2.0]}"
  expectRejected "non-finite interval certificate was accepted" (parseFlatBox? 2 nonfiniteBox)

  expectFloatPair "ODE parser interpreted u^0 incorrectly" (evalODEAtTwo "u^0") (1.0, 1.0)
  expectFloatPair "ODE parser interpreted u^1 incorrectly" (evalODEAtTwo "u^1") (2.0, 2.0)
  expectFloatPair "ODE parser interpreted u^2 incorrectly" (evalODEAtTwo "u^2") (4.0, 4.0)
  expectFloatPair "ODE exponentiation did not bind more tightly than unary minus"
    (evalODEAtTwo "-u^2") (-4.0, -4.0)
  expectFloatPair "ODE parser interpreted (-u)^2 incorrectly"
    (evalODEAtTwo "(-u)^2") (4.0, 4.0)
  expectFloatPair "ODE parser interpreted -(u^2) incorrectly"
    (evalODEAtTwo "-(u^2)") (-4.0, -4.0)
  expectFloatPair "ODE parser interpreted double negation incorrectly"
    (evalODEAtTwo "--u") (2.0, 2.0)

  expectFloatPair "PDE parser interpreted u^0 incorrectly" (evalPDEAtTwo "u^0") (1.0, 1.0)
  expectFloatPair "PDE parser interpreted u^1 incorrectly" (evalPDEAtTwo "u^1") (2.0, 2.0)
  expectFloatPair "PDE parser interpreted u^2 incorrectly" (evalPDEAtTwo "u^2") (4.0, 4.0)
  expectFloatPair "PDE exponentiation did not bind more tightly than unary minus"
    (evalPDEAtTwo "-u^2") (-4.0, -4.0)
  expectFloatPair "PDE parser interpreted (-u)^2 incorrectly"
    (evalPDEAtTwo "(-u)^2") (4.0, 4.0)
  expectFloatPair "PDE parser interpreted -(u^2) incorrectly"
    (evalPDEAtTwo "-(u^2)") (-4.0, -4.0)
  expectFloatPair "PDE parser interpreted double negation incorrectly"
    (evalPDEAtTwo "--u") (2.0, 2.0)

  let floatNaN := Float.ofBits 0x7ff8000000000000
  expect "Float NaN was accepted as <= a finite value"
    (!(NN.Verification.ODE.Ival.leBool floatNaN 0.0))
  expect "a finite Float was accepted as <= NaN"
    (!(NN.Verification.ODE.Ival.leBool 0.0 floatNaN))
  let floatInf := Float.ofBits 0x7ff0000000000000
  expect "Float infinity was accepted as an ordered finite endpoint"
    (!(NN.Verification.ODE.Ival.leBool floatInf 0.0))
  expect "interval minimum hid a non-finite Float endpoint"
    (!(NN.Verification.ODE.Ival.min2 floatInf 0.0).isFinite)
  expect "interval maximum hid a non-finite Float endpoint"
    (!(NN.Verification.ODE.Ival.max2 0.0 floatInf).isFinite)

  expect "IEEE32Exec NaN was accepted as <= a finite value"
    (!(NN.Verification.ODE.Ival.leBool IEEE32Exec.canonicalNaN IEEE32Exec.posZero))
  expect "a finite IEEE32Exec value was accepted as <= NaN"
    (!(NN.Verification.ODE.Ival.leBool IEEE32Exec.posZero IEEE32Exec.canonicalNaN))
  expect "IEEE32Exec infinity was accepted as an ordered finite endpoint"
    (!(NN.Verification.ODE.Ival.leBool IEEE32Exec.posInf IEEE32Exec.posZero))
  expect "interval minimum hid a non-finite IEEE32Exec endpoint"
    (!(IEEE32Exec.isFinite <| NN.Verification.ODE.Ival.min2 IEEE32Exec.posInf IEEE32Exec.posZero))
  expect "interval maximum hid a non-finite IEEE32Exec endpoint"
    (!(IEEE32Exec.isFinite <| NN.Verification.ODE.Ival.max2 IEEE32Exec.posZero IEEE32Exec.posInf))

  let b2 := flatBox (fun _ => 0.0) (fun _ => 1.0)
  let mismatchCert : Array (Option (FlatBox IEEE32Exec)) := #[some b2, some flatBox3, some b2]
  expect "binary elementwise dimension mismatch was accepted"
    (!(ibpNodePreconditionsOk addGraph mismatchCert 2))

  let nonPositive := flatBox (fun _ => 0.0) (fun _ => 1.0)
  let positive := flatBox (fun _ => 0.1) (fun _ => 2.0)
  expect "non-positive log input was accepted"
    (!(ibpNodePreconditionsOk logGraph #[some nonPositive, some nonPositive] 1))
  expect "positive log input was rejected"
    (ibpNodePreconditionsOk logGraph #[some positive, some positive] 1)

  let emptyStore : ParamStore IEEE32Exec := {}
  let nonPositiveRun := runIBP logGraph (emptyStore.seedInputBox 0 nonPositive)
  let positiveRun := runIBP logGraph (emptyStore.seedInputBox 0 positive)
  expect "IBP evaluated raw log across its nonpositive domain boundary"
    (nonPositiveRun[1]!.isNone)
  expect "IEEE32Exec log was accepted without a directed transcendental implementation"
    (positiveRun[1]!.isNone)

  let inputGraph : NN.IR.Graph := { nodes := #[inputNode 0 2] }
  let authoritative := flatBox (fun _ => 0.0) (fun _ => 1.0)
  let inward := flatBox (fun _ => 0.0) (fun _ => 0.999999)
  let outward := flatBox (fun _ => -0.000001) (fun _ => 1.000001)
  let inwardAccepted ←
    NN.Verification.IBPNodeCert.checkIBPNode inputGraph #[some authoritative] #[some inward] 0
  expect "an inward-shrunk certificate interval was accepted" (!inwardAccepted)
  let outwardAccepted ←
    NN.Verification.IBPNodeCert.checkIBPNode inputGraph #[some authoritative] #[some outward] 0
  expect "an outward-widened certificate interval was rejected" outwardAccepted

  let ctx : AffineCtx := { inputId := 0, inputDim := 2 }
  let crownCert : CROWNNodeCoreCertificate :=
    { ctx := ctx
      ibp := #[some authoritative]
      crown := #[some (boundsIdentity (α := IEEE32Exec) 2)]
      alpha := #[none] }
  expect "a complete exact alpha-CROWN replay was rejected"
    (NN.Verification.CROWNNodeCert.certificateAccepts crownCert inputGraph emptyStore
      #[some authoritative] true)
  let missingCrown := { crownCert with crown := #[none] }
  expect "an alpha-CROWN certificate with a missing affine entry was accepted"
    (!(NN.Verification.CROWNNodeCert.certificateAccepts missingCrown inputGraph emptyStore
      #[some authoritative] true))

  IO.println "certificate_preconditions: ok"

end CertificatePreconditions
end Floats
end Tests
