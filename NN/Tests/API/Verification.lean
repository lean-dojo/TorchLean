/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Verification
public import NN.Verification.Builtin.CrownOpsWorkflow
public import NN.Verification.Builtin.TransformerIBPWorkflow
public import NN.Verification.ODE.Verify
public import NN.Verification.PINN.PdeParse

/-!
# Verification API Tests

Focused checks for the general request/report boundary and automatic ReLU phase inference.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace NN.Tests.API.Verification

open TorchLean

def fail {α : Type} (message : String) : IO α :=
  throw <| IO.userError s!"verification API check failed: {message}"

def expectError {α : Type} (label : String) : Except String α → IO Unit
  | .error _ => pure ()
  | .ok _ => fail s!"{label}: expected an error"

def expectPhase (label : String)
    (actual expected : NN.MLTheory.CROWN.Cert.ReLUPhase) : IO Unit :=
  unless actual = expected do
    fail s!"{label}: got {repr actual}, expected {repr expected}"

/-- Norms and verification goals are named choices on the same operation. -/
example {inputShape outputShape : Shape}
    (trained : Trainer.Result inputShape outputShape)
    (center : Tensor Float inputShape) :
    IO Verification.Report :=
  trained.verify center
    (radius := 0.1)
    (norm := .inf)
    (property := .topLabel 0)

/--
Check LayerNorm enclosures with the native and IEEE arithmetic backends.

For each input box, evaluate the specification at its corners and center and check that every
sample lies between the returned bounds. The rows include constant inputs and nearly equal
coordinates, where epsilon has the largest effect. Reversed intervals, non-finite endpoints,
and empty normalization rows must be rejected.
-/
def checkLayerNorm {α : Type} [Storage α] [Context α]
    [Runtime.FromFloat α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α] : IO Unit := do
  let row (a b : Float) : Tensor α [2] :=
    Tensor.ofFn fun i => Runtime.ofFloat (if i.val = 0 then a else b)
  let enclose := NN.MLTheory.CROWN.Graph.directedLayerNormLastTensor? (α := α)
  for (a, b, radius) in [(-2.0, -1.0, 0.1), (1.0, 3.0, 0.2),
      (-1.0, 1.0, 0.1), (2.0, 2.0, 0.0), (1.0, 1.000001, 0.0000001)] do
    let some (lo, hi) := enclose (row (a - radius) (b - radius))
        (row (a + radius) (b + radius)) | fail "LayerNorm rejected a finite row"
    for da in [-radius, 0.0, radius] do
      for db in [-radius, 0.0, radius] do
        let input : Tensor α [1, 2] := Tensor.dim fun _ => row (a + da) (b + db)
        let actual := Spec.layerNorm input
          (Tensor.full [2] 1) (Tensor.full [2] 0)
        for i in List.finRange 2 do
          let value := (actual.unstack 0).getScalar i
          unless !(decide (value < lo.getScalar i)) &&
              !(decide (hi.getScalar i < value)) && value == value do
            fail "LayerNorm sampled output escaped the directed enclosure"
  unless (enclose (row 2.0 0.0) (row 1.0 3.0)).isNone do
    fail "LayerNorm accepted a reversed coordinate interval"
  for value in [1.0 / 0.0, 0.0 / 0.0] do
    unless (enclose (row value 0.0) (row value 0.0)).isNone do
      fail "LayerNorm accepted a non-finite endpoint"
    let singleton : Tensor α [1] := Tensor.full [1] (Runtime.ofFloat value)
    unless (NN.MLTheory.CROWN.Graph.ibpLayerNormBox? [1]
        { dim := 1, lo := singleton, hi := singleton }).isNone do
      fail "LayerNorm singleton shortcut accepted a non-finite endpoint"
  let empty : Tensor α [0] := Tensor.ofFn Fin.elim0
  unless (NN.MLTheory.CROWN.Graph.directedLayerNormLastTensor? empty empty).isNone do
    fail "LayerNorm accepted an empty normalization row"

/--
Check that LayerNorm bounds use the supplied epsilon, scale, and bias.

The two scales have opposite signs, so the affine step must reverse one interval. Rows include
constant inputs and nearly equal coordinates, where epsilon controls the denominator. The same
samples are checked with both arithmetic backends by `checkBoundArithmetic`.
-/
def checkAffineLayerNorm {α : Type} [Storage α] [Context α]
    [Runtime.FromFloat α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α] : IO Unit := do
  let row (a b : Float) : Tensor α [2] :=
    Tensor.ofFn fun i => Runtime.ofFloat (if i.val = 0 then a else b)
  let gamma := row 2.0 (-0.5)
  let beta := row 0.25 (-1.0)
  let enclose := NN.MLTheory.CROWN.Graph.directedLayerNormRow? (α := α)
  for epsilon in [0.00001, 0.25] do
    let epsilon : α := Runtime.ofFloat epsilon
    for (a, b, radius) in [(-2.0, -1.0, 0.1), (1.0, 3.0, 0.2),
        (2.0, 2.0, 0.0), (1.0, 1.000001, 0.0000001)] do
      let some (lo, hi) := enclose (row (a - radius) (b - radius))
          (row (a + radius) (b + radius)) gamma beta epsilon
        | fail "affine LayerNorm rejected finite inputs and parameters"
      for da in [-radius, 0.0, radius] do
        for db in [-radius, 0.0, radius] do
          let input : Tensor α [1, 2] := Tensor.dim fun _ => row (a + da) (b + db)
          let actual := Spec.layerNorm input gamma beta (epsilon := epsilon)
          for i in List.finRange 2 do
            let value := (actual.unstack 0).getScalar i
            unless !(decide (value < lo.getScalar i)) &&
                !(decide (hi.getScalar i < value)) && value == value do
              fail "affine LayerNorm sampled output escaped its enclosure"
  for epsilon in [0.0, -0.25, 1.0 / 0.0, 0.0 / 0.0] do
    unless (enclose (row 1.0 2.0) (row 1.0 2.0) gamma beta
        (Runtime.ofFloat epsilon)).isNone do
      fail "affine LayerNorm accepted an invalid epsilon"
  for value in [1.0 / 0.0, 0.0 / 0.0] do
    let invalid := row value 1.0
    let epsilon : α := Runtime.ofFloat 0.25
    unless (enclose (row 1.0 2.0) (row 1.0 2.0) invalid beta epsilon).isNone &&
        (enclose (row 1.0 2.0) (row 1.0 2.0) gamma invalid epsilon).isNone do
      fail "affine LayerNorm accepted non-finite scale or bias"
  let inputBox : NN.MLTheory.CROWN.FlatBox α :=
    { dim := 2, lo := row 1.0 2.0, hi := row 1.0 2.0 }
  let parameters : NN.IR.LayerNormParams α :=
    { normalizedShape := [2], gamma, beta, eps := Runtime.ofFloat 0.25 }
  let payloadBox := NN.MLTheory.CROWN.Graph.ibpLayerNormPayloadBox? (α := α)
  unless (payloadBox [1, 2] 1 parameters inputBox).isSome do
    fail "LayerNorm rejected a matching last-axis payload"
  unless (payloadBox [1, 2] 0 parameters inputBox).isNone do
    fail "LayerNorm accepted an unsupported normalization axis"
  let wrongShape : NN.IR.LayerNormParams α :=
    { normalizedShape := [1, 2]
      gamma := Tensor.dim fun _ => gamma
      beta := Tensor.dim fun _ => beta
      eps := parameters.eps }
  unless (payloadBox [1, 2] 1 wrongShape inputBox).isNone do
    fail "LayerNorm accepted a payload with a different normalized shape"

/--
The arithmetic dispatcher must carry nonlinear enclosures through generic callers.
These canonical lowered programs need division for MSE and a LayerNorm enclosure for attention.
-/
def checkBoundArithmetic : IO Unit := do
  for arithmetic in [Runtime.Arithmetic.native, Runtime.Arithmetic.ieee] do
    NN.Verification.Builtin.withBoundArithmetic arithmetic
      (fun {α} _ _ _ _ _ _ => do
        let one : α := Runtime.ofFloat 1.0
        unless (NN.MLTheory.CROWN.NonlinearBoundOps.divBounds one one one one).isSome do
          fail "arithmetic dispatcher lost nonlinear division bounds"
        checkLayerNorm (α := α)
        checkAffineLayerNorm (α := α)
        NN.Verification.Builtin.CrownOpsWorkflow.runMSE (α := α)
        NN.Verification.Builtin.TransformerIBPWorkflow.runMainDefault (α := α))

/-- Shared numeric parsing rejects malformed/nonfinite literals and accepts CRLF whitespace. -/
def checkTextParsing : IO Unit := do
  for text in ["", "-1", "12x"] do
    expectError "invalid decimal natural" (NN.Verification.Util.TextCursor.decimalNat text)
  let cursor : NN.Verification.Util.TextCursor.Cursor := { source := "\r\n -12.5 rest" }
  let .ok (value, rest) := NN.Verification.Util.TextCursor.parseFloat
      (NN.Verification.Util.TextCursor.remainingFuel cursor) cursor
    | fail "decimal parser rejected CRLF whitespace"
  unless value == -12.5 && NN.Verification.Util.TextCursor.peek rest == some ' ' do
    fail "decimal parser changed the value or consumed trailing input"
  let huge := String.ofList (List.replicate 400 '9')
  expectError "overflowing ODE literal" (NN.Verification.ODE.Parse.parseExpr huge)
  expectError "overflowing PDE literal"
    (NN.Verification.PINN.PdeParse.parseExpr (fun _ => none) huge)
  unless (NN.Verification.ODE.Parse.parseExpr "\r\n u + 1\r\n").isOk &&
      (NN.Verification.PINN.PdeParse.parseExpr (fun _ => none) "\r\n u + 1\r\n").isOk do
    fail "expression parsers rejected CRLF whitespace"

/-- General constant corridors exercise finite endpoint seeding near binary64 overflow.
A rejected adjacent-time interval must stop without exhausting an enormous depth budget. -/
def checkODETimePartition : IO Unit := do
  let common :=
    [ "--model=direct", "--arithmetic=native", "--init=0.5"
    , "--lower=NN/Examples/Verification/ODE/zero_mlp.json"
    , "--upper=NN/Examples/Verification/ODE/one_mlp.json"
    , "--slack=0", "--minWidth=0" ]
  for (lo, hi) in [("1e308", "1.7e308"), ("-1.7e308", "-1e308"),
      ("-1.7e308", "1.7e308")] do
    NN.Verification.ODE.Verify.main
      (common ++ ["--rhs=0.5-u", s!"--t0={lo}", s!"--t1={hi}", "--maxDepth=0"])
  let rejected ← try
    NN.Verification.ODE.Verify.main
      (common ++ ["--rhs=2", "--t0=1", "--t1=1.0000000000000002", "--maxDepth=100000"])
    pure false
  catch _ => pure true
  unless rejected do
    fail "ODE accepted a corridor that violates the upper differential inequality"

def run : IO Unit := do
  checkODETimePartition
  checkTextParsing
  checkBoundArithmetic
  expectPhase "Float inactive"
    (NN.MLTheory.CROWN.Cert.inferredPhase (α := Float) (-2.0) 0.0) .inactive
  expectPhase "Float unstable"
    (NN.MLTheory.CROWN.Cert.inferredPhase (α := Float) (-1.0) 1.0) .unstable
  expectPhase "Float active"
    (NN.MLTheory.CROWN.Cert.inferredPhase (α := Float) 0.0 2.0) .active

  let ieee := Runtime.ofFloat (α := (Binary 8 23))
  expectPhase "IEEE32 inactive"
    (NN.MLTheory.CROWN.Cert.inferredPhase (α := (Binary 8 23))
      (ieee (-2.0)) (ieee 0.0))
    .inactive
  expectPhase "IEEE32 unstable"
    (NN.MLTheory.CROWN.Cert.inferredPhase (α := (Binary 8 23))
      (ieee (-1.0)) (ieee 1.0))
    .unstable
  expectPhase "IEEE32 active"
    (NN.MLTheory.CROWN.Cert.inferredPhase (α := (Binary 8 23))
      (ieee 0.0) (ieee 2.0))
    .active

  let bounds : Verification.Bounds :=
    { size := 3, lower := [2.0, -1.0, 0.0]
      upper := [3.0, 0.5, 1.0] }
  let checked ←
    match Verification.Report.fromBounds 0.1 bounds (property := .topLabel 0) with
    | .ok report => pure report
    | .error message => fail message
  unless checked.radius = 0.1 &&
      checked.norm = .inf &&
      checked.property = .topLabel 0 &&
      checked.algorithm = .alphaBetaCrown do
    fail "report did not retain the named verification choices"
  match checked.result with
  | .topLabel 0 margin certified =>
      unless margin = 1.0 && certified do
        fail s!"unexpected positive margin report: {checked}"
  | result => fail s!"unexpected top-label result: {result}"

  let notCertified ←
    match Verification.Report.fromBounds 0.1 bounds (property := .topLabel 2) with
    | .ok report => pure report
    | .error message => fail message
  match notCertified.result with
  | .topLabel 2 margin certified =>
      unless margin = -3.0 && !certified do
        fail s!"unexpected negative margin report: {notCertified}"
  | result => fail s!"unexpected top-label result: {result}"

  let enclosure ←
    match Verification.Report.fromBounds 0.1 bounds with
    | .ok report => pure report
    | .error message => fail message
  match enclosure.result with
  | .bounds => pure ()
  | result => fail s!"unexpected bounds result: {result}"

  expectError "out-of-range label"
    (Verification.Report.fromBounds 0.1 bounds (property := .topLabel 3))
  expectError "one-class output"
    (Verification.Report.fromBounds 0.1 { size := 1, lower := [1.0], upper := [2.0] }
      (property := .topLabel 0))
  expectError "empty output"
    (Verification.Report.fromBounds 0.1
      { size := 0, lower := Tensor.zeros [0], upper := Tensor.zeros [0] }
      (property := .topLabel 0))
  expectError "reversed interval"
    (Verification.Report.fromBounds 0.1 { size := 2, lower := [2.0, 0.0], upper := [1.0, 1.0] }
      (property := .topLabel 0))
  expectError "non-finite interval"
    (Verification.Report.fromBounds 0.1
      { size := 2, lower := [0.0, 0.0 / 0.0], upper := [1.0, 1.0] }
      (property := .topLabel 0))
  expectError "negative radius"
    (Verification.Report.fromBounds (-0.1) bounds (property := .topLabel 0))
  expectError "non-finite radius"
    (Verification.Report.fromBounds (0.0 / 0.0) bounds (property := .topLabel 0))

  IO.println "  verification API: passed"

end NN.Tests.API.Verification
