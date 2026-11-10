/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
-- The checker implementation is loaded for execution, and its data contracts come through the
-- ordinary import below.
public meta import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Tensor.Literal

/-!
# A graph numerical certificate

The example certifies a four-node scalar graph:

```text
x in [1, 2]       c in [0.5, 1]
       \             /
        y = x + c
             |
        z = y * c
```

The source intervals have executable binary32 endpoints. The checker propagates them with
directed rounding, rejects non-finite intermediate intervals and records the backend capsules
selected when the portable CPU profile replans the graph. The resulting range trace is an
executable check, and a `RealEnclosure` supplies the separate proof that an exact-real
execution is enclosed. Larger graphs use the same artifact-generation and replay path.
-/

@[expose] public section

open FloatLib.Floats (Binary32)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open TorchLean.NumericalCertificate

namespace TorchLean.Examples.GraphNumericalCertificate

/-- The four-node scalar graph `z = (x + c) * c`, whose interval endpoints are easy to read. The
certificate stores one scalar hull per tensor, so its machinery does not depend on tensor rank. -/
def graph : IR.Graph :=
  { nodes := #[.input 0 (.value []), .const (.value []),
      .op ⟨_, _, Op.add (s := [])⟩ #[⟨0, 0⟩, ⟨1, 0⟩],
      .op ⟨_, _, Op.mul (s := [])⟩ #[⟨2, 0⟩, ⟨1, 0⟩]]
    outputs := #[⟨3, 0⟩] }

/--
The binary32 interval whose endpoints have the bit patterns `lo` and `hi`.

Every source range in this file is written in hexadecimal, so no decimal-to-binary conversion sits
between the written endpoints and what the checker sees.
-/
def interval (lo hi : UInt32) : Interval Binary32 :=
  { lo := ofBits32 lo, hi := ofBits32 hi }

/-- The assumed ranges: the input `x` in $[1, 2]$ and the constant `c` in $[0.5, 1]$. -/
def sources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0x3f800000 0x40000000 },
  { nodeId := 1, enclosure := interval 0x3f000000 0x3f800000 }
]

/-- Generate the range trace and kernel plan for the portable CPU profile, then check them. -/
def checked : Except String Checked :=
  generateChecked Backend.Profile.checkedCPU graph sources

/-- The constant for the bit-level replay, `c = 0.75`, inside its declared range $[0.5,1]$. -/
def payload : IR.Payload Binary32 where
  const? := fun nodeId =>
    if nodeId = 1 then
      some ⟨.value [], Tensor.full [] (ofBits32 0x3f400000)⟩
    else
      none

/-- A concrete input (`1.25`) inside the declared input interval. -/
def inputs : List (Op.Value Binary32) :=
  [⟨.value [], Tensor.full [] (ofBits32 0x3fa00000)⟩]

/-- Replay the same graph using the bit-level IEEE32 interpreter and check every intermediate. -/
def replay : Except String CheckedExecution := do
  let certificate <- checked
  executeIEEE32 payload inputs certificate

/-- The generated certificate with the addition's range replaced by $[0,0]$, a stand-in for a
corrupted or over-optimistic external artifact. $[0,0]$ is a well-formed interval, so the checker
has to recompute the range to reject it. -/
def tampered : Except String NumericalCertificate := do
  let raw <- generate Backend.Profile.checkedCPU graph sources
  let addition <- match raw.ranges[2]? with
    | some row => pure row
    | none => throw "example certificate is missing its addition row"
  let claimed := { addition with enclosure := interval 0x00000000 0x00000000 }
  pure { raw with ranges := raw.ranges.set! 2 claimed }

/-- Check the deliberately corrupted artifact against the canonical graph transfers. -/
def tamperedCheck : Except String Checked := do
  let raw <- tampered
  check Backend.Profile.checkedCPU graph raw

/-!
## A complete model pass

The small graph above keeps each range easy to read. The same machinery runs on a two-layer MLP
with matrix weights and bias tensors:

```text
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

Certificate generation is never told that this is an MLP. The checker sees ten ordinary
operator-graph nodes and takes each transfer from `rangeTransfer?`, and kernel selection
chooses a capsule for every operation on its own. The final replay executes the stored graph with
bit-level binary32 semantics and checks the tensors at all ten nodes against the regenerated
ranges.
-/

namespace MLP

/-- The product of an `[m, n]` matrix and an `[n, p]` matrix. -/
def matmul (m n p : Nat) : Op [.value [m, n], .value [n, p]] [.value [m, p]] :=
  .matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
    (fun h => nomatch h) (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])

/-- A two-layer matrix MLP expressed only with registered operators. -/
def graph : IR.Graph :=
  { nodes := #[.input 0 (.value [1, 2]), .const (.value [2, 3]),
      .op ⟨_, _, matmul 1 2 3⟩ #[⟨0, 0⟩, ⟨1, 0⟩], .const (.value [1, 3]),
      .op ⟨_, _, Op.add (s := [1, 3])⟩ #[⟨2, 0⟩, ⟨3, 0⟩],
      .op ⟨_, _, Op.relu (s := [1, 3])⟩ #[⟨4, 0⟩], .const (.value [3, 1]),
      .op ⟨_, _, matmul 1 3 1⟩ #[⟨5, 0⟩, ⟨6, 0⟩], .const (.value [1, 1]),
      .op ⟨_, _, Op.add (s := [1, 1])⟩ #[⟨7, 0⟩, ⟨8, 0⟩]]
    outputs := #[⟨9, 0⟩] }

/-- Ranges for the input and both weight matrices, $[-1, 1]$, and for both biases,
$[-0.25, 0.25]$. One interval covers every entry of a tensor. -/
def sources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 1, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 3, enclosure := interval 0xbe800000 0x3e800000 },
  { nodeId := 6, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 8, enclosure := interval 0xbe800000 0x3e800000 }
]

/-! Constant payloads are tensors at their nodes' ports, written row by row. -/

/-- First weight matrix `[2, 3]`. -/
def w1 : Tensor Binary32 [2, 3] :=
  [ [ofBits32 0x3f000000, ofBits32 0xbe800000, ofBits32 0x3f400000]
  , [ofBits32 0xbf000000, (1 : Binary32), ofBits32 0x3e800000] ]

/-- First bias, `[1, 3]`. -/
def b1 : Tensor Binary32 [1, 3] :=
  [[ofBits32 0x3e000000, ofBits32 0xbe000000, (Binary.zero false : Binary32)]]

/-- Second weight matrix `[3, 1]`. -/
def w2 : Tensor Binary32 [3, 1] :=
  [[ofBits32 0x3f000000], [ofBits32 0xbf400000], [(1 : Binary32)]]

/-- Output bias, `[1, 1]`. -/
def b2 : Tensor Binary32 [1, 1] :=
  [[ofBits32 0x3d800000]]

/-- The parameters enter as payloads of the constant nodes `1`, `3`, `6` and `8`, so the checker
treats them like any other constant. -/
def payload : IR.Payload Binary32 where
  const? := fun nodeId =>
    match nodeId with
    | 1 => some ⟨.value [2, 3], w1⟩
    | 3 => some ⟨.value [1, 3], b1⟩
    | 6 => some ⟨.value [3, 1], w2⟩
    | 8 => some ⟨.value [1, 1], b2⟩
    | _ => none

/-- The input `[[0.5, -1]]` of shape `[1, 2]` that the full-model replay runs on. -/
def inputs : List (Op.Value Binary32) :=
  let value : Tensor Binary32 [1, 2] :=
    [[ofBits32 0x3f000000, (-1 : Binary32)]]
  [⟨.value [1, 2], value⟩]

/-- Generate the operation-local range trace and bind it to the checked CPU capsule plan. -/
def certificate : Except String Checked :=
  generateChecked Backend.Profile.checkedCPU graph sources

/-- Execute the stored graph in the bit-level binary32 interpreter and check every node. -/
def replay : Except String CheckedExecution := do
  let certificate <- certificate
  executeIEEE32 payload inputs certificate

end MLP

end TorchLean.Examples.GraphNumericalCertificate
