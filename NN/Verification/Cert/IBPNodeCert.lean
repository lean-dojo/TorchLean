/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Spec.Core.Utils
public import NN.Verification.Cert.Common
public import Lean.Data.Json

/-!
# IBPNodeCert

Per-node IBP certificate checking.

This is stronger than `NN.Verification.IBPCert`: instead of only comparing the final output bounds,
we check that *every node*'s bounds match what Lean's `propagateIBPNode` computes from the
certificate bounds of its parents (plus the `ParamStore` parameters).

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
- We allow a small tolerance due to decimal serialization in JSON.
-/

@[expose] public section


namespace NN.Verification.IBPNodeCert

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open NN.Verification.Json
open NN.Verification.Cert.Common
open Import.PyTorch
open _root_.Spec
open _root_.Spec.Tensor
open Lean Data Json

/-- Read an IBP node certificate from JSON on disk. -/
def readIBPNodeCertificate (g : Graph) (path : String) : IO (Array (Option (FlatBox Float))) := do
  let topObj ← readJsonObjectFile path
  let arr ← expectFieldArray topObj "ibp" "top-level"
  if arr.size ≠ g.nodes.size then
    throw <| IO.userError s!"ibp array length {arr.size} ≠ g.nodes.size {g.nodes.size}"
  let mut out : Array (Option (FlatBox Float)) := Array.mkEmpty g.nodes.size
  for i in [0:g.nodes.size] do
    let dim := g.nodes[i]!.outShape.size
    let entry ← parseFlatBox? dim arr[i]!
    out := out.push entry
  pure out

/-- Check the local IBP enclosure condition for one node against a certificate entry. -/
def checkIBPNode (g : Graph) (ps : ParamStore Float) (cert : Array (Option (FlatBox Float)))
    (id : Nat) (tol : Float) : IO Bool := do
  let node := g.nodes[id]!
  let needsParents :=
    match node.kind with
    | .input | .const _ => false
    | _ => true
  if needsParents && !(parentsOk g cert id) then
    IO.eprintln s!"[IBPNodeCert] node {id}: parent boxes missing or not topo"
    return false
  let certBox? := cert[id]!
  let computed := propagateIBPNode (α := Float) g.nodes ps cert id
  let computedBox? := computed[id]!
  match certBox?, computedBox? with
  | none, _ =>
      IO.eprintln s!"[IBPNodeCert] node {id}: certificate missing (null)"
      pure false
  | _, none =>
      IO.eprintln s!"[IBPNodeCert] node {id}: Lean propagation produced no box"
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
      else if approxEqFlatBox certBox leanBox tol then
        pure true
      else
        IO.eprintln s!"[IBPNodeCert] mismatch at node {id} ({repr node.kind})"
        IO.eprintln s!"  cert: {prettyFlatBox certBox}"
        IO.eprintln s!"  lean: {prettyFlatBox leanBox}"
        pure false

/--
Check a per-node IBP certificate against Lean's graph IBP propagation rules.

Returns `true` iff every node's certificate bounds agree (within `tol`) with what Lean computes
from the certificate parents + `ParamStore`.
-/
def checkIBPNodeCertificate (g : Graph) (ps : ParamStore Float) (path : String) (tol : Float := 1e-5) : IO Bool :=
  do
  let cert ← readIBPNodeCertificate g path
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okNode ← checkIBPNode g ps cert id tol
    ok := ok && okNode
  if ok then
    IO.println "[IBPNodeCert] certificate verified: all nodes match Lean propagation."
  pure ok

end NN.Verification.IBPNodeCert
