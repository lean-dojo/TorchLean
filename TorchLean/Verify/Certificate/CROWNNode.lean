/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.ExecCROWN
public import TorchLean.Verify.Certificate.IBPNode

/-!
# Per-node α-CROWN certificates

Per-node α-CROWN certificate checking on graphs of registered operators (`TorchLean.IR.Graph`), with
optional ReLU phases.

The checker mirrors `TorchLean.Certificate.IBPNode` for the affine bounds of a CROWN/DeepPoly
pass, with optional α-parameters for the ReLU lower relaxation (α-CROWN). Lean replays each node
with the forward affine pass `TorchLean.CROWN.Op.alphaStep?` on its authoritative IBP trace and
the certificate's α vectors and phases. The pass has its own rule for the input (the identity), a
constant (its constant forms), detach, a linear map with constant weight and bias, ReLU with its α
and phases, a sum over every axis and reshape. Every other operator, the scaled matrix product
`.scaledMatmul` among them, gets the constant forms of its IBP box, except the matrix product
`.matmul` and custom operators, which have no affine bound, so `check` rejects every certificate of
a graph with such a node whose output `0` is at a scalar port. A node whose output `0` is not at a
scalar port has neither an interval nor an affine bound.

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

A phase is `-1` (inactive, $z\leq0$), `0` (unconstrained) or `1` (active, $0\leq z$), and it is
accepted only when the IBP pre-activation interval already proves it. Phases therefore carry no
information beyond IBP: with none given the replay is plain α-CROWN, and with phases it is α-CROWN
with those ReLUs fixed. No β multipliers or branch splits run.

Trust boundary notes:
- The certificate is untrusted. We accept it only if its binary32 affine transcript exactly
  matches Lean recomputation.
- Lean's replay rounds each sum, product and quotient to nearest, so an accepted transcript need
  not enclose the graph. After the layers `h = x + 1` and `y = h + 2 ^ -24` at `x = -1`, where
  `y = 2 ^ -24`, the replayed upper form of `y` is `x + 1`, which is `0` there. On `y = relu x`
  over `[-M, M]`, with `M` the largest finite value, `M - (-M)` overflows and the replayed upper
  form of `y` is `0`, below `relu M = M`. For linear and ReLU chains,
  `TorchLean.Certificate.FiniteArtifact.accepts` checks the bounds against exact rational
  transfers, and `FiniteArtifact.accepts_graph_sound` proves the requested output inequalities on
  the whole input box.
- Each coefficient stands for its binary32 value, the decimal rounded to nearest as in IEEE 754-2019
  section 5.12 (see `NodeReplay`), so `1e39` reads as `+∞`. An offset may be the infinity that
  bounds nothing on its side, `-∞` in `loC` and `+∞` in `hiC` (written `-1e39` and `1e39`), and
  like every coefficient it must equal Lean's replay bit for bit. Infinite matrix entries and an
  offset at the other infinity are refused. `TorchLean.Test.CertificatePreconditions.boundary`
  checks each of these cases with `±1e39`.
- A transcendental operator has the constant forms of its IBP box, a rational enclosure of its
  image rounded outward to binary32, so the check uses no libm result.

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
The JSON-facing helpers of the CROWN certificate checkers parse the artifact, require bitwise
binary32 agreement of the affine replay data and check the parent and shape requirements.
Acceptance proves local replay consistency (`accepts_sound`), which gives no enclosure for real
inputs because the replay rounds to nearest.
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
Check one node of a CROWN affine certificate against the bound `computed?` that Lean replayed
(parent availability, dimensions and bitwise binary32 equality of the transcript). A node whose
output `0` is not at a scalar port has no affine bound: its entry must be `null`. At a scalar port
the entry and `computed?` must both be present, so the node fails wherever Lean replays no bound,
as at `.matmul` and custom operators. The result is the reason the entry fails, `none` when it
passes.
-/
def checkNode
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
The per-node part of the check: the reasons the node entries fail, in node order. At every node the
IBP entry must pass `IBPNode.checkNode` (a box containing Lean's authoritative interval at a scalar
output), and the affine entry must pass `checkNode` against the bound that `step` replays from
Lean's bounds of the earlier nodes. Its emptiness is the `diagnosticsOk` input of `accepts`.

`check` passes `replayStep`, the step of `accepts_sound`, so this loop and the pure acceptance
decision replay the same rule.
-/
def checkNodeEntries (G : IR.Graph)
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (cert : CROWNNode)
    (step : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) → Nat →
      Option (FlatAffineBounds (ExecFloat.Binary 8 23))) : Array String := Id.run do
  let mut authoritativeCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) :=
    Array.replicate G.nodes.size none
  let mut failures : Array String := #[]
  for id in [0:G.nodes.size] do
    let computed? := step authoritativeCrown id
    failures := failures ++
      (IBPNode.checkNode G authoritativeIbp cert.ibp id).toArray ++
      (checkNode G authoritativeCrown cert.crown cert.ctx id computed?).toArray
    authoritativeCrown := authoritativeCrown.set! id computed?
  return failures

/-- The replay of a parsed certificate: α-CROWN, with the certificate's ReLU phases if any. -/
def replayStep (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (cert : CROWNNode) :
    Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) → Nat →
      Option (FlatAffineBounds (ExecFloat.Binary 8 23)) :=
  fun replay id =>
    CROWN.Op.alphaStep? G P authoritativeIbp cert.alpha cert.phases replay cert.ctx id

/--
The final in-memory acceptance decision.

`diagnosticsOk` records parsing-independent IBP, shape, and incremental-replay checks. The second
conjunct establishes local replay consistency (`accepts_sound`). The replay rounds to nearest, so
an accepted upper form can lie below its node's value.
-/
def accepts
    (cert : CROWNNode) (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool) : Bool :=
  crownAccepts G.nodes.size (replayStep G P authoritativeIbp cert) cert.crown
    diagnosticsOk

/-- Acceptance supplies local consistency of the replay: one entry per node, each the replay of
the entries. The equality is bitwise, so it includes an offset that is the infinity bounding nothing
on its side. -/
theorem accepts_sound
    (cert : CROWNNode) (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool)
    (haccept : accepts cert G P authoritativeIbp diagnosticsOk = true) :
    cert.crown.size = G.nodes.size ∧ ∀ id : Nat, id < G.nodes.size →
      cert.crown[id]! = replayStep G P authoritativeIbp cert cert.crown id :=
  crownAccepts_sound G.nodes.size (replayStep G P authoritativeIbp cert) cert.crown
    diagnosticsOk haccept

/--
Check a per-node CROWN certificate against Lean's propagation rules on graph `G` with payload `P`,
one input box per graph input index in `inputs`.

The report accepts the certificate only if `G` passes `Graph.check`, `inputs` passes
`Op.validInputs`, every constant node has its payload constant, the IBP box of every scalar output
contains Lean's authoritative recomputation and every node's affine replay data agrees exactly
with Lean's step. A rejected certificate's report has a line for each failing node check.

The per-node loop is part of the verdict: it checks IBP containment, parent coverage and
dimensions, which the pure replay in `accepts` does not, and the pure replay runs only when the
loop passes. The certificate's `ibp` boxes are only checked for containment. Replay uses Lean's
own IBP pass, so they add no information to the affine check.
-/
def check (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (inputs : List (FlatBox (ExecFloat.Binary 8 23))) (path : String) : IO Report := do
  if let .error e := G.check then
    return { rejection := some s!"[CROWNNode] graph rejected: {e}" }
  unless CROWN.Op.validInputs G inputs do
    let reason := "[CROWNNode] an input box is missing, has the wrong size or has unordered bounds"
    return { rejection := some reason }
  unless IBPNode.constsOk G P do
    let reason := "[CROWNNode] a constant node has no payload constant at its port"
    return { rejection := some reason }
  let cert ← read G path
  let authoritativeIbp :=
    CROWN.Op.runIBP G P (CROWN.Op.inputSeeds G inputs)
  let failures := checkNodeEntries G authoritativeIbp cert (replayStep G P authoritativeIbp cert)
  if accepts cert G P authoritativeIbp failures.isEmpty then
    let line := "[CROWNNode] artifact matched an authoritative Lean IBP and alpha-CROWN replay."
    return { lines := #[line] }
  let reason :=
    if failures.isEmpty then "[CROWNNode] the affine entries are not the replay of the entries"
    else s!"[CROWNNode] {failures.size} node checks fail"
  return { lines := failures, rejection := some reason }

end TorchLean.Certificate.CROWNNode
