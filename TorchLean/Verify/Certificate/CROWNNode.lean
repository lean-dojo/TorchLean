/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.ExecCROWN
public import TorchLean.Verify.Certificate.IBPNode
public import TorchLean.Verify.Certificate.FiniteArtifact.Semantics

/-!
# Per-node α-CROWN certificates

Per-node α-CROWN certificates on graphs of registered operators (`TorchLean.IR.Graph`), with
optional ReLU phases: the safety verdict `check` and the replay diagnostic `replayDiagnostic`.

`check` reads a certificate and decides it with `TorchLean.Certificate.FiniteArtifact.accepts`
(`verdict_ok`). The graph must be one vector input followed by linear layers and ReLU nodes. Each
affine entry is read as its exact rational value and must dominate the exact transfer of its
parent's entry over the input box, and the requested output inequalities must hold on the output's
bounds. `FiniteArtifact.accepts_graph_sound` then proves the inequalities for every real input in
the box, on the graph evaluated in real arithmetic on the exact values of its binary32 constants.
`check` rejects every other graph.

`replayDiagnostic` mirrors `TorchLean.Certificate.IBPNode` for the affine bounds of a CROWN/DeepPoly
pass, with optional α-parameters for the ReLU lower relaxation (α-CROWN). Lean replays each node
with the forward affine pass `TorchLean.CROWN.Op.alphaStep?` on its authoritative IBP trace and
the certificate's α vectors and phases. The pass has its own rule for the input (the identity), a
constant (its constant forms), detach, a linear map with constant weight and bias, ReLU with its α
and phases, a sum over every axis and reshape. Every other operator, the scaled matrix product
`.scaledMatmul` among them, gets the constant forms of its IBP box, except the matrix product
`.matmul` and custom operators, which have no affine bound, so `replayDiagnostic` reports every
certificate of a graph with such a node whose output `0` is at a scalar port. A node whose output
`0` is not at a scalar port has neither an interval nor an affine bound.

The ReLU rule takes its lines from Eq. (5) of α-CROWN (Sec. 2.3), so an unstable neuron gets
`y = α z` below and the triangle's chord above. Lean never optimizes α. It uses the certificate's
vector, after `parseAlphaVec?` has refused any entry outside `[0, 1]`, and `Cert.defaultAlphaVec`
at a ReLU whose `alpha` entry is `null`.

Certificate JSON format:

```json
{
  "ctx": { "inputId": 0, "inputDim": 2 },
  "ibp": [ null | { "lo": [...], "hi": [...] }, ... ],
  "crown": [
    null |
      {
        "loA": [[...], ...], "loC": [...],
        "hiA": [[...], ...], "hiC": [...]
      },
    ...
  ],
  "alpha": [ null | [...], ... ],          // optional per-node ReLU α vector
  "phases": [ null | [-1,0,1,...], ... ]  // optional per-node ReLU phases
}
```

A phase is `-1` (inactive, $z\leq0$), `0` (unconstrained) or `1` (active, $0\leq z$), and the
replay uses it only when the IBP pre-activation interval already proves it. Phases therefore carry
no information beyond IBP: with none given the replay is plain α-CROWN, and with phases it is
α-CROWN with those ReLUs fixed. No β multipliers or branch splits run.

Trust boundary notes:
- The certificate is untrusted. `check` accepts it only if `FiniteArtifact.accepts` does, which
  refuses every infinite or NaN coefficient.
- `replayDiagnostic` reports where the binary32 affine transcript differs from Lean's replay. The
  replay rounds each sum, product and quotient to nearest, so a transcript equal to it need not
  enclose the graph, and agreement is not a safety verdict. After the layers `h = x + 1` and
  `y = h + 2 ^ -24` at `x = -1`, where `y = 2 ^ -24`, the replayed upper form of `y` is `x + 1`,
  which is `0` there. On `y = relu x` over `[-M, M]`, with `M` the largest finite value,
  `M - (-M)` overflows and the replayed upper form of `y` is `0`, below `relu M = M`.
  `TorchLeanTest.Verification.CertificateVerdict` gives a linear/ReLU chain whose replay table
  agrees with the replay, while `check` rejects it: an upper form lies below the real output.
- Each coefficient stands for its binary32 value, the decimal rounded to nearest as in IEEE 754-2019
  section 5.12 (see `NodeReplay`), so `1e39` reads as `+∞`. An offset may be the infinity that
  bounds nothing on its side, `-∞` in `loC` and `+∞` in `hiC` (written `-1e39` and `1e39`), and
  like every coefficient it must equal Lean's replay bit for bit in `replayDiagnostic`. Infinite
  matrix entries and an offset at the other infinity are refused when read.
  `TorchLean.Test.CertificatePreconditions.boundary` checks each of these cases with `±1e39`.
- A transcendental operator has the constant forms of its IBP box, a rational enclosure of its
  image rounded outward to binary32, so the replay uses no libm result.

References:
- α-CROWN: Xu et al., "Fast and Complete: Enabling Complete Neural Network Verification with
  Rapid and Massively Parallel Incomplete Verifiers", ICLR 2021, arXiv:2011.13824, Sec. 2.3,
  Eq. (5).
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Certificate.CROWNNode

open CROWN
open Graph
open NodeReplay
open TorchLean.Tensor

/-!
## The replay diagnostic

The helpers of the replay diagnostic parse the artifact, compare the affine entries bit for bit
with Lean's binary32 replay and test the parent and shape requirements. Agreement gives local
replay consistency (`replayAgrees_spec`), which gives no enclosure for real inputs because the
replay rounds to nearest.
-/

/-- Read a CROWN node certificate for graph `G` from JSON on disk. -/
def read (G : IR.Graph) (path : String) : IO CROWNNode := do
  parse (G.nodes.map outputSize)
    (← DecimalJson.readObjectFile path)

/-- Whether node `id` of `G` reads only outputs of earlier nodes, with an entry in `cert` at every
node that a reference to scalar values names. An input or constant node reads nothing and passes. -/
def parentsOk {β : Type} (G : IR.Graph) (cert : Array (Option β)) (id : Nat) : Bool :=
  match G.nodes[id]? with
  | none => false
  | some (.op _ refs) => refs.all fun r =>
      match G.portBefore id r with
      | some p => !p.kind.isScalar || (cert[r.node]?).join.isSome
      | none => false
  | some _ => true

/--
Compare one node of a CROWN affine certificate with the bound `computed?` that Lean replayed
(parent availability, dimensions and bitwise binary32 equality of the transcript). A node whose
output `0` is not at a scalar port has no affine bound: its entry must be `null`. At a scalar port
the entry and `computed?` must both be present, so the node differs wherever Lean replays no bound,
as at `.matmul` and custom operators. The result is how the entry differs, `none` when it is the
replay.
-/
def nodeMismatch
    (G : IR.Graph)
    (authoritativeCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))))
    (certCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))))
    (ctx : AffineCtx)
    (id : Nat)
    (computed? : Option (FlatAffineBounds (ExecFloat.Binary 8 23))) : Option String :=
  match G.nodes[id]? with
  | none => some s!"[CROWNNode] node {id}: out of bounds for graph with {G.nodes.size} nodes"
  | some node =>
    if !(parentsOk G authoritativeCrown id) then
      some s!"[CROWNNode] node {id}: parent affine bounds missing or not topo"
    else if !(node.outs.head?.map (·.kind.isScalar)).getD false then
      if ((certCrown[id]?).join).isSome then
        some s!"[CROWNNode] node {id}: an affine bound for an output without scalar values"
      else
        none
    else
      let dim := outputSize node
      match (certCrown[id]?).join, computed? with
      | none, _ => some s!"[CROWNNode] node {id}: certificate missing (null)"
      | _, none => some s!"[CROWNNode] node {id}: Lean propagation produced no affine bound"
      | some certB, some leanB =>
        if certB.inDim ≠ ctx.inputDim then
          some (s!"[CROWNNode] node {id}: cert inDim {certB.inDim} ≠ " ++
            s!"ctx.inputDim {ctx.inputDim}")
        else if certB.outDim ≠ dim then
          some s!"[CROWNNode] node {id}: cert outDim {certB.outDim} ≠ output size {dim}"
        else if leanB.outDim ≠ dim then
          some s!"[CROWNNode] node {id}: Lean outDim {leanB.outDim} ≠ output size {dim}"
        else if decide (certB = leanB) then
          none
        else
          some (s!"[CROWNNode] mismatch at node {id}\n" ++
            s!"  cert: {prettyAffineBounds certB}\n  lean: {prettyAffineBounds leanB}")

/--
The node diagnostics of the replay: how the node entries differ from it, in node order. At every
node the IBP entry must pass `IBPNode.checkNode` (a box containing Lean's authoritative interval at
a scalar output), and `nodeMismatch` compares the affine entry with the bound that `step` replays
from Lean's bounds of the earlier nodes. Its emptiness is the `diagnosticsOk` input of
`replayAgrees`.

`replayDiagnostic` passes `replayStep`, the step of `replayAgrees_spec`, so this loop and the pure
agreement test replay the same rule.
-/
def mismatches (G : IR.Graph)
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (cert : CROWNNode)
    (step : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) → Nat →
      Option (FlatAffineBounds (ExecFloat.Binary 8 23))) : Array String := Id.run do
  let mut authoritativeCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) :=
    Array.replicate G.nodes.size none
  let mut found : Array String := #[]
  for id in [0:G.nodes.size] do
    let computed? := step authoritativeCrown id
    found := found ++
      (IBPNode.checkNode G authoritativeIbp cert.ibp id).toArray ++
      (nodeMismatch G authoritativeCrown cert.crown cert.ctx id computed?).toArray
    authoritativeCrown := authoritativeCrown.set! id computed?
  return found

/-- The replay of a parsed certificate: α-CROWN, with the certificate's ReLU phases if any. -/
def replayStep (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (cert : CROWNNode) :
    Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) → Nat →
      Option (FlatAffineBounds (ExecFloat.Binary 8 23)) :=
  fun replay id =>
    CROWN.Op.alphaStep? G P authoritativeIbp cert.alpha cert.phases replay cert.ctx id

/--
Whether the certificate agrees with the replay: the node diagnostics passed (`diagnosticsOk`) and
every affine entry is the replay of the entries (`replayAgrees_spec`). This is not a safety
verdict. The replay rounds to nearest, so an agreeing upper form can lie below its node's value.
-/
def replayAgrees
    (cert : CROWNNode) (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool) : Bool :=
  agrees G.nodes.size (replayStep G P authoritativeIbp cert) cert.crown diagnosticsOk

/-- Agreement with the replay supplies local consistency of the replay: one entry per node, each
the replay of the entries. The equality is bitwise, so it includes an offset that is the infinity
bounding nothing on its side. -/
theorem replayAgrees_spec
    (cert : CROWNNode) (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool)
    (haccept : replayAgrees cert G P authoritativeIbp diagnosticsOk = true) :
    cert.crown.size = G.nodes.size ∧ ∀ id : Nat, id < G.nodes.size →
      cert.crown[id]! = replayStep G P authoritativeIbp cert cert.crown id :=
  agrees_spec G.nodes.size (replayStep G P authoritativeIbp cert) cert.crown
    diagnosticsOk haccept

/--
The differences between the certificate at `path` and Lean's binary32 α-CROWN replay on graph `G`
with payload `P`, one input box per graph input index in `inputs`, one line per difference. This
is a diagnostic, not a safety verdict: `check` is the verdict.

The result is empty only when `G` passes `Graph.check`, `inputs` passes `Op.validInputs`, every
constant node has its payload constant, the node diagnostics `mismatches` are empty (the IBP box
of every scalar output contains Lean's authoritative recomputation, and parents and dimensions
fit) and every affine entry is the replay (`replayAgrees`). The replay reads Lean's own IBP boxes,
so the certificate's boxes are only compared by containment. A file that does not parse raises
the reader's error.
-/
def replayDiagnostic (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (inputs : List (FlatBox (ExecFloat.Binary 8 23))) (path : String) : IO (Array String) := do
  if let .error e := G.check then
    return #[s!"[CROWNNode] graph rejected: {e}"]
  unless CROWN.Op.validInputs G inputs do
    return #["[CROWNNode] an input box is missing, has the wrong size or has unordered bounds"]
  unless IBPNode.constsOk G P do
    return #["[CROWNNode] a constant node has no payload constant at its port"]
  let cert ← read G path
  let authoritativeIbp := CROWN.Op.runIBP G P (CROWN.Op.inputSeeds G inputs)
  let found := mismatches G authoritativeIbp cert (replayStep G P authoritativeIbp cert)
  if replayAgrees cert G P authoritativeIbp found.isEmpty then
    return #[]
  if found.isEmpty then
    return #["[CROWNNode] the affine entries are not the replay of the entries"]
  return found

/-!
## The safety verdict

`check` decides a certificate by `FiniteArtifact.accepts` (`verdict_ok`), and
`FiniteArtifact.accepts_graph_sound` turns acceptance into the requested inequalities for every
real input in the box. The verdict does not run the replay and does not read the `ibp` boxes or
the phases.
-/

/--
The safety verdict on `cert` for graph `G` with payload `P`, the box `input` of graph input `0`
and the output inequalities `query`. The report accepts exactly when `FiniteArtifact.accepts`
does (`verdict_ok`), and then `FiniteArtifact.accepts_graph_sound` proves the inequalities for
every real input in the box. A rejection says whether the artifact does not decode or fails
`FiniteArtifact.Decoded.check`.
-/
def verdict (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) (cert : CROWNNode)
    (query : FiniteArtifact.Query) : Report :=
  match FiniteArtifact.decode G P input cert query with
  | none =>
      { rejection := some ("[CROWNNode] not a finite vector linear/ReLU artifact: the graph, " ++
          "input box, certificate or query does not decode") }
  | some decoded =>
      if decoded.check then
        { lines := #["[CROWNNode] accepted: for every real input in the box, the exact-real " ++
            "graph meets every requested inequality (FiniteArtifact.accepts_graph_sound)"] }
      else
        { rejection := some ("[CROWNNode] the decoded artifact fails " ++
            "FiniteArtifact.Decoded.check: a zero dimension, an unordered input box, an entry " ++
            "that does not dominate its exact transfer, or a failed margin") }

/-- The report of `verdict` accepts exactly when `FiniteArtifact.accepts` accepts. -/
theorem verdict_ok (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) (cert : CROWNNode)
    (query : FiniteArtifact.Query) :
    (verdict G P input cert query).ok = true ↔
      FiniteArtifact.accepts G P input cert query = true := by
  unfold verdict FiniteArtifact.accepts
  cases FiniteArtifact.decode G P input cert query with
  | none => exact Iff.intro (fun h => by cases h) (fun h => by cases h)
  | some decoded =>
      dsimp only
      cases decoded.check
      · exact Iff.intro (fun h => by cases h) (fun h => by cases h)
      · exact Iff.intro (fun _ => rfl) (fun _ => rfl)

/--
Read the certificate at `path` for graph `G` and return `verdict` on it, with payload `P`, the box
`input` of graph input `0` and the output inequalities `query`. A file that does not parse raises
the reader's error.
-/
def check (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) (query : FiniteArtifact.Query) (path : String) :
    IO Report := do
  return verdict G P input (← read G path) query

end TorchLean.Certificate.CROWNNode
