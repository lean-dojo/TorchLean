/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
public import NN.MLTheory.CROWN.Graph
public import NN.Spec.Core.Utils
public import NN.Verification.Cert.Common
public import Lean.Data.Json

/-!
# CROWNNodeCertAlphaBeta

Per-node α/β-CROWN certificate checking (graph dialect).

This extends `NN.Verification.CROWNNodeCert` with an optional β phase vector for ReLU nodes.

Certificate JSON format:

```json
{
  "ctx": { "inputId": 0, "inputDim": 2 },
  "ibp": [ null | { "lo": [...], "hi": [...] }, ... ],
  "crown": [
    null |
      { "loA": [[...], ...], "loC": [...],
        "hiA": [[...], ...], "hiC": [...] },
    ...
  ],
  "alpha": [ null | [...], ... ],
  "beta":  [ null | [-1,0,1,...], ... ]   // optional per-node ReLU phase vector
}
```

β encoding (per neuron):
- `-1` = forced inactive (`z ≤ 0`)
- `0`  = unconstrained / unstable
- `1`  = forced active (`0 ≤ z`)

As with the α-CROWN checker, the certificate is accepted only if Lean recomputation matches the
provided affine bounds up to a small float tolerance.
-/

@[expose] public section


namespace NN.Verification.CROWNNodeCertAlphaBeta

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
Helpers for the alpha/beta-CROWN style node certificate checker.

These are the JSON-facing utilities for the checker: they parse imported bounds, compare
decimal-serialized floats with an explicit tolerance, and keep shape mismatches from reaching the
semantic checker.
-/

/-- Parse a JSON integer (used for beta vectors). -/
def parseInt? (j : Json) : Option Int :=
  match j with
  | .num n => n.toString.toInt?
  | .str s => s.toInt?
  | _ => none

/-- Parse a beta vector from JSON. -/
def parseBetaVec? (dim : Nat) (j : Json) : IO (Option (Array Int)) := do
  match j with
  | .null => pure none
  | .arr xs =>
      if xs.size ≠ dim then
        throw <| IO.userError s!"Invalid beta[i]: expected int array length {dim}"
      let mut out : Array Int := Array.mkEmpty dim
      for k in [0:xs.size] do
        let some i := parseInt? xs[k]!
          | throw <| IO.userError s!"Invalid beta[i][{k}]: expected int"
        if i = (-1) || i = 0 || i = 1 then
          out := out.push i
        else
          throw <| IO.userError s!"Invalid beta[i][{k}]: expected -1/0/1"
      pure (some out)
  | _ => throw <| IO.userError "Invalid beta[i]: expected null or int array"

/-!
`AlphaBetaCROWNNodeCertificate` is the in-memory representation of an alpha/beta-CROWN node
certificate read from JSON.

This is part of the public surface of the checker because `readAlphaBetaCROWNNodeCertificate`
returns it, and because the blueprint points to it as the “shape of the artifact” being checked.
-/
structure AlphaBetaCROWNNodeCertificate where
  /-- Affine-propagation context, including the chosen input node and flattened input dimension. -/
  ctx : AffineCtx
  /-- Optional per-node interval bounds used by nonlinear CROWN steps. -/
  ibp : Array (Option (FlatBox Float))
  /-- Optional per-node affine lower/upper bounds. -/
  crown : Array (Option (FlatAffineBounds Float))
  /-- Optional per-node α values for ReLU lower relaxations. -/
  alpha : Array (Option (FlatVec Float))
  /-- Optional per-node β phase annotations for ReLU nodes. -/
  beta : Array (Option (Array Int))

/-- Read an alpha/beta-CROWN node certificate from JSON on disk. -/
def readAlphaBetaCROWNNodeCertificate (g : Graph) (path : String) : IO AlphaBetaCROWNNodeCertificate := do
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
  let betaArr ←
    match ← optionalField? topObj "beta" "top-level" with
    | none => pure (Array.replicate g.nodes.size Json.null)
    | some betaJ => expectArray betaJ "top-level.beta"

  if ibpArr.size ≠ g.nodes.size then
    throw <| IO.userError s!"ibp length {ibpArr.size} ≠ g.nodes.size {g.nodes.size}"
  if crownArr.size ≠ g.nodes.size then
    throw <| IO.userError s!"crown length {crownArr.size} ≠ g.nodes.size {g.nodes.size}"
  if alphaArr.size ≠ g.nodes.size then
    throw <| IO.userError s!"alpha length {alphaArr.size} ≠ g.nodes.size {g.nodes.size}"
  if betaArr.size ≠ g.nodes.size then
    throw <| IO.userError s!"beta length {betaArr.size} ≠ g.nodes.size {g.nodes.size}"

  let mut ibp : Array (Option (FlatBox Float)) := Array.mkEmpty g.nodes.size
  let mut crown : Array (Option (FlatAffineBounds Float)) := Array.mkEmpty g.nodes.size
  let mut alpha : Array (Option (FlatVec Float)) := Array.mkEmpty g.nodes.size
  let mut beta : Array (Option (Array Int)) := Array.mkEmpty g.nodes.size

  for i in [0:g.nodes.size] do
    let outDim := g.nodes[i]!.outShape.size
    let ibpEntry ← parseFlatBox? outDim ibpArr[i]!
    ibp := ibp.push ibpEntry
    let crownEntry ← parseAffineBounds? ctx.inputDim outDim crownArr[i]!
    crown := crown.push crownEntry
    let alphaEntry ← parseFlatVec? outDim alphaArr[i]!
    alpha := alpha.push alphaEntry
    let betaEntry ← parseBetaVec? outDim betaArr[i]!
    beta := beta.push betaEntry

  pure { ctx := ctx, ibp := ibp, crown := crown, alpha := alpha, beta := beta }

/-- Check the local alpha/beta-CROWN enclosure condition for one node against a certificate entry.
  -/
def checkAlphaBetaCROWNNode (g : Graph) (ps : ParamStore Float)
    (certIbp : Array (Option (FlatBox Float)))
    (certAlpha : Array (Option (FlatVec Float)))
    (certBeta : Array (Option (Array Int)))
    (certCrown : Array (Option (FlatAffineBounds Float)))
    (ctx : AffineCtx)
    (id : Nat) (tol : Float) : IO Bool := do
  let node := g.nodes[id]!
  let needsParents :=
    match node.kind with
    | .input | .const _ => false
    | _ => true
  if needsParents && !(parentsOk g certCrown id) then
    IO.eprintln s!"[CROWNNodeCertAlphaBeta] node {id}: parent affine bounds missing or not topo"
    return false

  let certB? := certCrown[id]!
  let computed? :=
    alphaBetaCrownStepNode? (α := Float) g.nodes ps certIbp certAlpha certBeta certCrown ctx id

  match certB?, computed? with
  | none, _ =>
      IO.eprintln s!"[CROWNNodeCertAlphaBeta] node {id}: certificate missing (null)"
      pure false
  | _, none =>
      IO.eprintln s!"[CROWNNodeCertAlphaBeta] node {id}: Lean propagation produced no affine bound"
      pure false
  | some certB, some leanB =>
      if certB.inDim ≠ ctx.inputDim then
        IO.eprintln
          (s!"[CROWNNodeCertAlphaBeta] node {id}: cert inDim {certB.inDim} ≠ " ++
            s!"ctx.inputDim {ctx.inputDim}")
        pure false
      else if certB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[CROWNNodeCertAlphaBeta] node {id}: cert outDim {certB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if leanB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[CROWNNodeCertAlphaBeta] node {id}: Lean outDim {leanB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if approxEqFlatAffineBounds certB leanB tol then
        pure true
      else
        IO.eprintln s!"[CROWNNodeCertAlphaBeta] mismatch at node {id} ({repr node.kind})"
        IO.eprintln s!"  cert: {prettyAffineBounds certB}"
        IO.eprintln s!"  lean: {prettyAffineBounds leanB}"
        pure false

/--
Check a per-node α/β-CROWN certificate against Lean's propagation rules.

Returns `true` iff every node's certificate affine bounds agree (within `tol`) with what Lean
computes from the certificate parents + `ParamStore`.
-/
def checkAlphaBetaCROWNNodeCertificate (g : Graph) (ps : ParamStore Float) (path : String) (tol : Float := 1e-4) :
    IO Bool :=
  do
  let cert ← readAlphaBetaCROWNNodeCertificate g path
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okNode ← checkAlphaBetaCROWNNode g ps cert.ibp cert.alpha cert.beta cert.crown cert.ctx id tol
    ok := ok && okNode
  if ok then
    IO.println "[CROWNNodeCertAlphaBeta] certificate verified: all nodes match Lean α/β-CROWN step."
  pure ok

end NN.Verification.CROWNNodeCertAlphaBeta
