/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
-- Load the checker implementation for execution; its data contracts are imported normally below.
public meta import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate
public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Certificate

/-!
# A graph numerical certificate

This example certifies a four-node scalar graph:

```text
x in [1, 2]       c in [0.5, 1]
       \             /
        y = x + c
             |
        z = y * c
```

The source intervals are executable binary32 endpoints. The checker propagates them with directed
rounding, rejects non-finite intermediate intervals, and records the backend capsules selected when
the portable CPU profile replans the graph. The resulting range trace is an executable check; a
`ProvedRealEnclosure` supplies the separate proof that an exact-real execution is enclosed. Larger
graphs use the same artifact-generation and replay path.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open Proofs.RuntimeApprox.NumericalCertificate
open Spec TorchLean
open TorchLean
open TorchLean.Floats.IEEE754

namespace NN.Examples.DeepDives.Floats.GraphNumericalCertificate

/-- The example uses scalar nodes so the interval endpoints remain easy to inspect. The certificate
machinery itself stores only one scalar hull per tensor and is independent of tensor rank. -/
def graph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [] },
      { id := 1, parents := #[], kind := .const [], outShape := [] },
      { id := 2, parents := #[0, 1], kind := .add, outShape := [] },
      { id := 3, parents := #[2, 1], kind := .mulElem, outShape := [] }
    ] }

/--
Build a binary32 interval from two bit patterns.

Every source range in this file is written in hexadecimal rather than as a decimal literal. That
keeps
the certificate an exact artifact: no decimal-to-binary conversion sits between what is written here
and what the checker sees.
-/
def interval (lo hi : UInt32) : IEEE32Exec.Interval32 :=
  { lo := ofBits32 lo, hi := ofBits32 hi }

/-- Did an executable certificate operation return a checked value? -/
def accepted {α : Type} : Except String α -> Bool
  | .ok _ => true
  | .error _ => false

/-- Input and constant assumptions. Hexadecimal endpoints preserve the exact binary32 artifact. -/
def sources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0x3f800000 0x40000000 },
  { nodeId := 1, enclosure := interval 0x3f000000 0x3f800000 }
]

/-- Generate and replay the range trace and the selected kernel plan. -/
def checked : Except String RegistryCheckedCertificate :=
  generateChecked NN.Backend.BackendProfile.checkedCpu graph sources

/-- Concrete payload used for bit-level replay. The constant is `0.75`, which lies in the declared
constant range $[0.5,1]$. -/
def payload : NN.IR.Payload (Binary 8 23) where
  const? := fun nodeId =>
    if nodeId = 1 then
      some
        { n := 1
          v := [ofBits32 0x3f400000] }
    else
      none

/-- A concrete input (`1.25`) inside the declared input interval. -/
def input : Spec.SomeTensor (Binary 8 23) :=
  Spec.SomeTensor.ofTensor (Tensor.full [] (ofBits32 0x3fa00000))

/-- Replay the same graph using the bit-level IEEE32 interpreter and check every intermediate. -/
def replay : Except String RangeCheckedExecution := do
  let certificate <- checked
  executeIEEE32 payload input certificate

/-- Deliberately replace the addition range with $[0,0]$. This models a corrupted or optimistic
external artifact; replay must not accept it merely because $[0,0]$ is itself a valid interval. -/
def tampered : Except String GraphNumericalCertificate := do
  let raw <- generate NN.Backend.BackendProfile.checkedCpu graph sources
  let addition <- match raw.ranges[2]? with
    | some row => pure row
    | none => throw "example certificate is missing its addition row"
  let claimed := { addition with enclosure := interval 0x00000000 0x00000000 }
  pure { raw with ranges := raw.ranges.set! 2 claimed }

/-- Check the deliberately corrupted artifact against the canonical graph transfers. -/
def tamperedCheck : Except String RegistryCheckedCertificate := do
  let raw <- tampered
  check NN.Backend.BackendProfile.checkedCpu graph raw

/-!
## A complete model pass

The small graph above makes each range easy to inspect. This graph runs the same machinery
over a two-layer MLP with matrix weights and explicit bias tensors:

```text
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

Nothing in certificate generation is told that this is an MLP. The checker sees ten ordinary IR
nodes and obtains each transfer from `GraphRangeRegistry`. Kernel selection independently chooses
a capsule for every operation. The final replay executes the stored graph with bit-level binary32
semantics and checks all ten intermediate tensors against the regenerated ranges.
-/

def mlpInputShape : Spec.Shape := [1, 2]
def mlpHiddenShape : Spec.Shape := [1, 3]
def mlpFirstWeightShape : Spec.Shape := [2, 3]
def mlpSecondWeightShape : Spec.Shape := [3, 1]
def mlpOutputShape : Spec.Shape := [1, 1]

/-- A two-layer matrix MLP expressed only in the canonical operation IR. -/
def mlpGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := mlpInputShape },
      { id := 1, parents := #[], kind := .const mlpFirstWeightShape,
        outShape := mlpFirstWeightShape },
      { id := 2, parents := #[0, 1], kind := .matmul, outShape := mlpHiddenShape },
      { id := 3, parents := #[], kind := .const mlpHiddenShape, outShape := mlpHiddenShape },
      { id := 4, parents := #[2, 3], kind := .add, outShape := mlpHiddenShape },
      { id := 5, parents := #[4], kind := .relu, outShape := mlpHiddenShape },
      { id := 6, parents := #[], kind := .const mlpSecondWeightShape,
        outShape := mlpSecondWeightShape },
      { id := 7, parents := #[5, 6], kind := .matmul, outShape := mlpOutputShape },
      { id := 8, parents := #[], kind := .const mlpOutputShape, outShape := mlpOutputShape },
      { id := 9, parents := #[7, 8], kind := .add, outShape := mlpOutputShape }
    ] }

/-- Source ranges cover inputs, both weight matrices, and both bias tensors. A single enclosure per
tensor is sufficient for this certificate format; the graph walk remains independent of rank. -/
def mlpSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 1, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 3, enclosure := interval 0xbe800000 0x3e800000 },
  { nodeId := 6, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 8, enclosure := interval 0xbe800000 0x3e800000 }
]

/-! Constant payloads use the IR's canonical flat storage ABI; node shapes recover the typed matrix
view during evaluation. The explicit order below is row-major. -/

def mlpFirstWeightFlat : Tensor (Binary 8 23) [6] :=
  [ ofBits32 0x3f000000
  , ofBits32 0xbe800000
  , ofBits32 0x3f400000
  , ofBits32 0xbf000000
  , (1 : Binary 8 23)
  , ofBits32 0x3e800000 ]

/-- First bias, flat. -/
def mlpHiddenBiasFlat : Tensor (Binary 8 23) [3] :=
  [ofBits32 0x3e000000, ofBits32 0xbe000000, (Binary.zero false : Binary 8 23)]

/-- Second weight matrix `[3, 1]`, flat. -/
def mlpSecondWeightFlat : Tensor (Binary 8 23) [3] :=
  [ofBits32 0x3f000000, ofBits32 0xbf400000, (1 : Binary 8 23)]

/-- Output bias, a single value. -/
def mlpOutputBiasFlat : Tensor (Binary 8 23) [1] :=
  [ofBits32 0x3d800000]

/-- Concrete parameters are payloads of the constant nodes, not special fields in the checker. -/
def mlpPayload : NN.IR.Payload (Binary 8 23) where
  const? := fun nodeId =>
    match nodeId with
    | 1 => some { n := 6, v := mlpFirstWeightFlat }
    | 3 => some { n := 3, v := mlpHiddenBiasFlat }
    | 6 => some { n := 3, v := mlpSecondWeightFlat }
    | 8 => some { n := 1, v := mlpOutputBiasFlat }
    | _ => none

/-- The concrete `[1, 2]` input the full-model replay runs on. -/
def mlpInput : Spec.SomeTensor (Binary 8 23) :=
  let value : Tensor (Binary 8 23) [1, 2] :=
    [[ofBits32 0x3f000000, (-1 : Binary 8 23)]]
  Spec.SomeTensor.ofTensor value

/-- Generate the operation-local range trace and bind it to the checked CPU capsule plan. -/
def mlpCertificate : Except String RegistryCheckedCertificate :=
  generateChecked NN.Backend.BackendProfile.checkedCpu mlpGraph mlpSources

/-- Execute the stored graph in the bit-level binary32 interpreter and check every node. -/
def mlpReplay : Except String RangeCheckedExecution := do
  let certificate <- mlpCertificate
  executeIEEE32 mlpPayload mlpInput certificate

/-- Generate and replay both graphs, and reject the deliberately tampered base artifact. -/
def exampleChecks : Array (String × Bool) :=
  #[ ("base certificate", accepted checked)
  , ("base IEEE replay", accepted replay)
  , ("tampered range rejected", !accepted tamperedCheck)
  , ("two-layer MLP certificate", accepted mlpCertificate)
  , ("two-layer MLP IEEE replay", accepted mlpReplay)
  ]

/-- Help text for the certificate example. -/
def usage : String :=
  String.intercalate "\n"
    [ "Numerical runtime certificate example"
    , ""
    , "Usage:"
    , "  lake exe torchlean numerical_certificate"
    , ""
    , "Runs range-certificate, backend-audit, tamper-rejection, and bit-level"
    , "binary32 replay checks. The final two checks run a complete two-layer MLP."
    ]

/-- Public runner for the certificate examples. A failed positive check or an accepted negative
check produces a nonzero exit code, so this command is also suitable for regression testing. -/
def main (args : List String) : IO UInt32 := do
  if args.any fun arg => arg = "-h" || arg = "--help" then
    IO.println usage
    return 0
  if !args.isEmpty then
    IO.eprintln s!"unexpected arguments: {String.intercalate " " args}"
    IO.eprintln usage
    return 2
  IO.println "TorchLean numerical runtime certificate"
  let mut failed := false
  for (name, ok) in exampleChecks do
    IO.println s!"  {if ok then "ok" else "FAIL"}  {name}"
    if !ok then
      failed := true
  if failed then
    IO.eprintln "Numerical certificate checks failed."
    return 1
  IO.println "All numerical certificate checks passed."
  return 0

end NN.Examples.DeepDives.Floats.GraphNumericalCertificate
