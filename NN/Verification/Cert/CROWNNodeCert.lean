/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.MLTheory.CROWN.Graph
public import NN.Spec.Core.Utils
public import NN.Verification.Cert.Common
public import Lean.Data.Json

/-!
# CROWNNodeCert

Per-node α-CROWN certificate checking (graph dialect).

This mirrors `NN.Verification.IBPNodeCert`, but for affine bounds produced by a CROWN/DeepPoly pass
with optional α-parameters for the ReLU lower relaxation (α-CROWN).

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
  "alpha": [ null | [...], ... ] // optional per-node ReLU α vector
}
```

Trust boundary notes:
- The certificate is untrusted; we accept it only if Lean recomputation matches (within a float
  tolerance due to JSON decimal serialization).
- Transcendental relaxations are checked only via structural recomputation, not via a formal
  "libm is correct" guarantee.
-/

@[expose] public section


namespace NN.Verification.CROWNNodeCert

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Cert
open NN.Verification.Json
open NN.Verification.Cert.Common
open Import.PyTorch
open _root_.Spec
open _root_.Spec.Tensor
open Lean Data Json

/-!
The helpers below are the JSON-facing boundary for the CROWN certificate checkers.  They parse
the artifact, compare decimal-serialized floats with an explicit tolerance, and check parent/shape
requirements before invoking the semantic checker.
-/

/-!
`CROWNNodeCertificate` is the in-memory representation of a node-wise CROWN certificate read from JSON.

This is part of the public surface of the checker because `readCROWNNodeCertificate` returns it,
and because the blueprint points to it as the “shape of the artifact” being checked.
-/
structure CROWNNodeCertificate where
  /-- Affine-propagation context, including the chosen input node and flattened input dimension. -/
  ctx : AffineCtx
  /-- Optional per-node interval bounds used by nonlinear CROWN steps. -/
  ibp : Array (Option (FlatBox Float))
  /-- Optional per-node affine lower/upper bounds. -/
  crown : Array (Option (FlatAffineBounds Float))
  /-- Optional per-node α values for ReLU lower relaxations. -/
  alpha : Array (Option (FlatVec Float))

/-- Read a CROWN node certificate from JSON on disk. -/
def readCROWNNodeCertificate (g : Graph) (path : String) : IO CROWNNodeCertificate := do
  let topObj ← readJsonObjectFile path
  let ctxObj ← expectFieldObj topObj "ctx" "top-level"
  let inputId ← expectFieldNat ctxObj "inputId" "ctx"
  let inputDim ← expectFieldNat ctxObj "inputDim" "ctx"
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }

  let ibpArr ← expectFieldArray topObj "ibp" "top-level"
  let crownArr ← expectFieldArray topObj "crown" "top-level"
  let alphaArr ←
    match ← optionalField? topObj "alpha" "top-level" with
    | none => pure (Array.replicate g.nodes.size Json.null)
    | some alphaJ => expectArray alphaJ "top-level.alpha"

  if ibpArr.size ≠ g.nodes.size then
    throw <| IO.userError s!"ibp length {ibpArr.size} ≠ g.nodes.size {g.nodes.size}"
  if crownArr.size ≠ g.nodes.size then
    throw <| IO.userError s!"crown length {crownArr.size} ≠ g.nodes.size {g.nodes.size}"
  if alphaArr.size ≠ g.nodes.size then
    throw <| IO.userError s!"alpha length {alphaArr.size} ≠ g.nodes.size {g.nodes.size}"

  let mut ibp : Array (Option (FlatBox Float)) := Array.mkEmpty g.nodes.size
  let mut crown : Array (Option (FlatAffineBounds Float)) := Array.mkEmpty g.nodes.size
  let mut alpha : Array (Option (FlatVec Float)) := Array.mkEmpty g.nodes.size

  for i in [0:g.nodes.size] do
    let outDim := g.nodes[i]!.outShape.size
    let ibpEntry ← parseFlatBox? outDim ibpArr[i]!
    ibp := ibp.push ibpEntry
    let crownEntry ← parseAffineBounds? ctx.inputDim outDim crownArr[i]!
    crown := crown.push crownEntry
    let alphaEntry ← parseFlatVec? outDim alphaArr[i]!
    alpha := alpha.push alphaEntry

  pure { ctx := ctx, ibp := ibp, crown := crown, alpha := alpha }

/-- Check the local CROWN enclosure condition for one node against a certificate entry. -/
def checkCROWNNode (g : Graph) (ps : ParamStore Float)
    (certIbp : Array (Option (FlatBox Float)))
    (certAlpha : Array (Option (FlatVec Float)))
    (certCrown : Array (Option (FlatAffineBounds Float)))
    (ctx : AffineCtx)
    (id : Nat) (tol : Float) : IO Bool := do
  let node := g.nodes[id]!
  let needsParents :=
    match node.kind with
    | .input | .const _ => false
    | _ => true
  if needsParents && !(parentsOk g certCrown id) then
    IO.eprintln s!"[CROWNNodeCert] node {id}: parent affine bounds missing or not topo"
    return false

  let certB? := certCrown[id]!
  let computed? :=
    alphaCrownStepNode? (α := Float) g.nodes ps certIbp certAlpha certCrown ctx id

  match certB?, computed? with
  | none, _ =>
      IO.eprintln s!"[CROWNNodeCert] node {id}: certificate missing (null)"
      pure false
  | _, none =>
      IO.eprintln s!"[CROWNNodeCert] node {id}: Lean propagation produced no affine bound"
      pure false
  | some certB, some leanB =>
      if certB.inDim ≠ ctx.inputDim then
        IO.eprintln
          s!"[CROWNNodeCert] node {id}: cert inDim {certB.inDim} ≠ ctx.inputDim {ctx.inputDim}"
        pure false
      else if certB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[CROWNNodeCert] node {id}: cert outDim {certB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if leanB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[CROWNNodeCert] node {id}: Lean outDim {leanB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if approxEqFlatAffineBounds certB leanB tol then
        pure true
      else
        IO.eprintln s!"[CROWNNodeCert] mismatch at node {id} ({repr node.kind})"
        IO.eprintln s!"  cert: {prettyAffineBounds certB}"
        IO.eprintln s!"  lean: {prettyAffineBounds leanB}"
        pure false

/--
Check a per-node α-CROWN certificate against Lean's propagation rules.

Returns `true` iff every node's certificate affine bounds agree (within `tol`) with what Lean
computes from the certificate parents + `ParamStore`.
-/
def checkCROWNNodeCertificate (g : Graph) (ps : ParamStore Float) (path : String) (tol : Float := 1e-4) : IO Bool :=
  do
  let cert ← readCROWNNodeCertificate g path
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okNode ← checkCROWNNode g ps cert.ibp cert.alpha cert.crown cert.ctx id tol
    ok := ok && okNode
  if ok then
    IO.println "[CROWNNodeCert] certificate verified: all nodes match Lean α-CROWN step."
  pure ok

end NN.Verification.CROWNNodeCert
