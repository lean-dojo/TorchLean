/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.NodeReplay
public import TorchLean.Verify.Certificate.Report
public import TorchLean.Verify.CROWN.Op.Exec
public import TorchLean.Verify.CROWN.BoundOps.Binary32

/-!
# Per-node IBP certificates

Per-node IBP certificate checking on graphs of registered operators (`TorchLean.IR.Graph`).

`check` first rejects a graph that fails `Graph.check`, then computes the complete interval trace
from the trusted input boxes, one per graph input, and the graph's binary32 constants
(`TorchLean.CROWN.Op.runIBP`). `checkNode` compares the untrusted artifact with that trace entry
by entry and takes the trace as given. No node is ever recomputed from certificate-supplied parent
boxes. A node has a box in the trace only where its operator's rule accepted the boxes of its
inputs: equal dimensions for the binary pointwise operators, a positive box for `log`, and for
`reciprocal` at binary32 a box without zero (the backend's interval division refuses a divisor
interval containing zero). The rules do not compare a box with the declared shape, so an `add` at
shape `[2]` of two boxes of 3 entries has a box of 3 entries. The size checks of `checkNode` refuse
such a box, and `Graph.check` refuses a graph whose references do not match the ports they read.

A node whose output `0` is not at a scalar port (a Boolean, index or integer constant such as an
attention mask, or an operator returning one) carries no interval: its entry must be `null`. Every
constant node must have a payload constant readable at its port.

Certificate JSON format:

```json
{
  "ibp": [
    null,
    { "lo": [...], "hi": [...] },
    ...
  ]
}
```

The array has one entry per operator-graph node, indexed by node id, so its length is
`G.nodes.size`. At a node whose output `0` is at a scalar port the entry must be an object whose
`lo` and `hi` arrays have the flattened size of that output (`NodeReplay.outputSize`), and at any
other node it must be `null`. The parser reads `null` as a missing entry, which `checkNode`
rejects at a node with a scalar output.

Trust boundary note:
- The certificate is untrusted and is accepted only if it agrees with Lean's recomputation.
- A certificate interval must contain the Lean-recomputed interval componentwise. Each decimal
  endpoint is first rounded outward into binary32 (see `NodeReplay.parseFlatBox?`), so an endpoint
  inward of Lean's by less than one binary32 ulp is read as Lean's value and accepted. The rounding
  is the decimal conversion of IEEE 754-2019 section 5.12, so an endpoint stands for a binary32
  value. Past the binary32 range, `1e39` is `+∞` as an upper endpoint and the largest finite value
  as a lower one, as `TorchLean.Test.CertificatePreconditions.boundary` checks.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Certificate.IBPNode

open CROWN
open NodeReplay
open TorchLean.Tensor

/-- Read an IBP node certificate for graph `G` from JSON on disk. -/
def read (G : IR.Graph) (path : String) :
    IO (Array (Option (FlatBox (ExecFloat.Binary 8 23)))) := do
  let top ← DecimalJson.readObjectFile path
  let arr ← IO.ofExcept <| top.field "ibp" "top-level" >>= (·.array "top-level.ibp")
  parsePerNode (G.nodes.map outputSize) "ibp" arr parseFlatBox?

/-- Compare the artifact entry of node `id` with the entry of the trace `authoritative`. At a node
whose output `0` is at a scalar port, both boxes must be present with that output's size and the
artifact box must contain the trace's box. At any other node the artifact entry must be `null`.
The result is the reason the entry fails, `none` when it passes. -/
def checkNode (G : IR.Graph)
    (authoritative cert : Array (Option (FlatBox (ExecFloat.Binary 8 23)))) (id : Nat) :
    Option String :=
  match G.nodes[id]? with
  | none => some s!"[IBPNode] node {id}: out of bounds for graph with {G.nodes.size} nodes"
  | some node =>
    if !(node.outs.head?.map (·.kind.isScalar)).getD false then
      if (cert[id]?.join).isSome then
        some s!"[IBPNode] node {id}: an interval for an output without scalar values"
      else
        none
    else
      let dim := outputSize node
      match cert[id]?.join, authoritative[id]?.join with
      | none, _ => some s!"[IBPNode] node {id}: certificate missing (null)"
      | _, none => some s!"[IBPNode] node {id}: authoritative Lean trace has no box"
      | some certBox, some leanBox =>
        if certBox.dim ≠ dim then
          some s!"[IBPNode] node {id}: cert dim {certBox.dim} ≠ output size {dim}"
        else if leanBox.dim ≠ dim then
          some s!"[IBPNode] node {id}: Lean dim {leanBox.dim} ≠ output size {dim}"
        else if flatBoxContains certBox leanBox then
          none
        else
          some (s!"[IBPNode] inward or mismatched bound at node {id}\n" ++
            s!"  cert: {prettyFlatBox certBox}\n  lean: {prettyFlatBox leanBox}")

/-- Whether every constant node of `G` has a payload constant readable at its port. -/
def constsOk {α : Type} [Storage α] (G : IR.Graph) (P : IR.Payload α) : Bool :=
  (List.range G.nodes.size).all fun id =>
    match G.nodes[id]? with
    | some (.const _) => (CROWN.Op.constValue? G P id).isSome
    | _ => true

/--
Check a per-node IBP certificate against Lean's interval rules on graph `G` with payload `P`, one
input box per graph input index in `inputs`.

The report accepts the certificate iff `G` passes `Graph.check`, `inputs` passes
`Op.validInputs`, every constant node has its payload constant, and every node's certificate entry
passes `checkNode` against the trace recomputed from trusted inputs and constants. A rejected
certificate's report has a line for each failing entry.
-/
def check (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (inputs : List (FlatBox (ExecFloat.Binary 8 23))) (path : String) : IO Report := do
  if let .error e := G.check then
    return { rejection := some s!"[IBPNode] graph rejected: {e}" }
  unless CROWN.Op.validInputs G inputs do
    let reason := "[IBPNode] an input box is missing, has the wrong size or has unordered bounds"
    return { rejection := some reason }
  unless constsOk G P do
    let reason := "[IBPNode] a constant node has no payload constant at its port"
    return { rejection := some reason }
  let cert ← read G path
  let authoritative := CROWN.Op.runIBP G P (CROWN.Op.inputSeeds G inputs)
  let failures := (Array.range G.nodes.size).filterMap (checkNode G authoritative cert)
  if failures.isEmpty then
    let line := "[IBPNode] every artifact interval encloses the authoritative Lean trace."
    return { lines := #[line] }
  return {
    lines := failures
    rejection := some s!"[IBPNode] {failures.size} of {G.nodes.size} node entries fail" }

end TorchLean.Certificate.IBPNode
