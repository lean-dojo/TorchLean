/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.NodeReplay

/-!
# IBPNodeCert

Per-node IBP certificate checking.

Lean first computes the complete interval trace from the trusted input boxes and parameters. The
untrusted artifact is then checked against that trace. In particular, no node is ever recomputed
from certificate-supplied parent boxes.

Intended certificate JSON format:

```json
{
  "ibp": [
    null,
    { "lo": [...], "hi": [...] },
    ...
  ]
}
```

The array length must equal `g.nodes.size`. Every entry must be an object whose `lo` and `hi`
arrays have length equal to that node's flattened output dimension `g.nodes[i]!.outShape.size`.
The parser reads `null` as a missing entry, and `checkNode` rejects it.

Trust boundary note:
- The certificate is untrusted; we accept it only if Lean recomputation matches.
- A certificate interval must contain the Lean-recomputed interval componentwise. Each decimal
  endpoint is first rounded outward into binary32 (see `NodeReplay.parseFlatBox?`), so an endpoint
  inward of Lean's by less than one binary32 ulp is read as Lean's value and accepted.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.Verification.Cert.IBPNodeCert

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open NN.Verification.Json
open NN.Verification.Cert.NodeReplay
open Spec TorchLean
open TorchLean.Tensor
open Lean Json

/-- Read an IBP node certificate from JSON on disk. -/
def read (g : Graph) (path : String) :
    IO (Array (Option (FlatBox (ExecFloat.Binary 8 23)))) := do
  let topObj ← readJsonObjectFile path
  let arr ← expectFieldArray topObj "ibp" "top-level"
  parsePerNode g "ibp" arr fun node entry => parseFlatBox? node.outShape.size entry

/-- Check one artifact entry against the authoritative Lean IBP trace. -/
def checkNode (g : Graph)
    (authoritative cert : Array (Option (FlatBox (ExecFloat.Binary 8 23)))) (id : Nat) :
    IO Bool := do
  let some node := g.nodes[id]?
    | IO.eprintln s!"[IBPNodeCert] node {id}: out of bounds for graph with {g.nodes.size} nodes"
      pure false
  if !(ibpNodePreconditionsOk g authoritative id) then
    IO.eprintln
      (s!"[IBPNodeCert] node {id}: authoritative trace violates shape/domain preconditions " ++
        s!"for {repr node.kind}")
    return false
  match getFlatBox? cert id, getFlatBox? authoritative id with
  | none, _ =>
      IO.eprintln s!"[IBPNodeCert] node {id}: certificate missing (null)"
      pure false
  | _, none =>
      IO.eprintln s!"[IBPNodeCert] node {id}: authoritative Lean trace has no box"
      pure false
  | some certBox, some leanBox =>
      if certBox.dim ≠ node.outShape.size then
        IO.eprintln
          s!"[IBPNodeCert] node {id}: cert dim {certBox.dim} ≠ outShape.size {node.outShape.size}"
        pure false
      else if leanBox.dim ≠ node.outShape.size then
        IO.eprintln
          s!"[IBPNodeCert] node {id}: Lean dim {leanBox.dim} ≠ outShape.size {node.outShape.size}"
        pure false
      else if flatBoxContains certBox leanBox then
        pure true
      else
        IO.eprintln s!"[IBPNodeCert] inward or mismatched bound at node {id} ({repr node.kind})"
        IO.eprintln s!"  cert: {prettyFlatBox certBox}"
        IO.eprintln s!"  lean: {prettyFlatBox leanBox}"
        pure false

/--
Check a per-node IBP certificate against Lean's graph IBP propagation rules.

Returns `true` iff every node's certificate interval contains the interval recomputed from trusted
inputs and parameters.
-/
def check (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (path : String) : IO Bool := do
  let cert ← read g path
  let authoritative := runIBP (α := (ExecFloat.Binary 8 23)) g ps
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okNode ← checkNode g authoritative cert id
    ok := ok && okNode
  if ok then
    IO.println "[IBPNodeCert] every artifact interval encloses the authoritative Lean trace."
  pure ok

end NN.Verification.Cert.IBPNodeCert
