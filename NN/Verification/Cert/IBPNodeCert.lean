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

The array length must equal `g.nodes.size`. Each non-null entry must have `lo` and `hi` arrays of
length equal to that node's flattened output dimension `g.nodes[i]!.outShape.size`.

Trust boundary note:
- The certificate is untrusted; we accept it only if Lean recomputation matches.
- A certificate interval must contain the Lean-recomputed interval componentwise. Decimal
  serialization may widen an endpoint, but it may not move an endpoint inward.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.Verification.IBPNodeCert

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open NN.Verification.Json
open NN.Verification.Cert.NodeReplay
open Import.PyTorch
open Spec TorchLean
open TorchLean.Tensor
open Lean Data Json
/-- Read an IBP node certificate from JSON on disk. -/
def readIBPNodeCertificate (g : Graph) (path : String) :
    IO (Array (Option (FlatBox (ExecFloat.Binary 8 23)))) := do
  let topObj ← readJsonObjectFile path
  let arr ← expectFieldArray topObj "ibp" "top-level"
  if hSize : arr.size = g.nodes.size then
    let mut out : Array (Option (FlatBox (ExecFloat.Binary 8 23))) := Array.mkEmpty g.nodes.size
    for i in List.finRange g.nodes.size do
      let node := g.nodes[i.val]'i.isLt
      let hArr : i.val < arr.size := by
        rw [hSize]
        exact i.isLt
      let entryJson := arr[i.val]'hArr
      let entry ← parseFlatBox? node.outShape.size entryJson
      out := out.push entry
    pure out
  else
    throw <| IO.userError s!"ibp array length {arr.size} ≠ g.nodes.size {g.nodes.size}"

/-- Check one artifact entry against the authoritative Lean IBP trace. -/
def checkIBPNode (g : Graph)
    (authoritative cert : Array (Option (FlatBox (ExecFloat.Binary 8 23)))) (id : Nat) : IO Bool :=
      do
  let some node := g.nodes[id]?
    | IO.eprintln s!"[IBPNodeCert] node {id}: out of bounds for graph with {g.nodes.size} nodes"
      pure false
  if !(ibpNodePreconditionsOk g authoritative id) then
    IO.eprintln
      (s!"[IBPNodeCert] node {id}: authoritative trace violates shape/domain preconditions " ++
        s!"for {repr node.kind}")
    return false
  let certBox? :=
    match cert[id]? with
    | some entry => entry
    | none => none
  let authoritativeBox? :=
    match authoritative[id]? with
    | some entry => entry
    | none => none
  match certBox?, authoritativeBox? with
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
def checkIBPNodeCertificate (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23)) (path : String) :
  IO Bool :=
  do
  let cert ← readIBPNodeCertificate g path
  let authoritative := runIBP (α := (ExecFloat.Binary 8 23)) g ps
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okNode ← checkIBPNode g authoritative cert id
    ok := ok && okNode
  if ok then
    IO.println "[IBPNodeCert] every artifact interval encloses the authoritative Lean trace."
  pure ok

end NN.Verification.IBPNodeCert
