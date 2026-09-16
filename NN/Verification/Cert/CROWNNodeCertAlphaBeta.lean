/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
public import NN.Verification.Cert.IBPNodeCert
public import NN.Spec.Core.Tensor -- shake: keep

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
- `-1` = forced inactive ($z\leq0$)
- `0`  = unconstrained / unstable
- `1`  = forced active ($0\leq z$)

As with the α-CROWN checker, the certificate is accepted only if the provided binary32 affine
bounds exactly match Lean recomputation.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.Verification.CROWNNodeCertAlphaBeta

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Cert
open NN.Verification.Json
open NN.Verification.Cert.NodeReplay
open Import.PyTorch
open Spec TorchLean
open TorchLean.Tensor
open Lean Data Json
/-!
Helpers for the alpha/beta-CROWN style node certificate checker.

These are the JSON-facing utilities for the checker: they parse imported bounds, require exact
binary32 agreement for affine replay data, and keep shape mismatches from reaching the semantic
checker.
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
      if hSize : xs.size = dim then
        let mut out : Array Int := Array.mkEmpty dim
        for k in List.finRange dim do
          let h : k.val < xs.size := by
            rw [hSize]
            exact k.isLt
          let some i := parseInt? (xs[k.val]'h)
            | throw <| IO.userError s!"Invalid beta[i][{k.val}]: expected int"
          if i = (-1) || i = 0 || i = 1 then
            out := out.push i
          else
            throw <| IO.userError s!"Invalid beta[i][{k.val}]: expected -1/0/1"
        pure (some out)
      else
        throw <| IO.userError s!"Invalid beta[i]: expected int array length {dim}"
  | _ => throw <| IO.userError "Invalid beta[i]: expected null or int array"

/--
`AlphaBetaCROWNNodeCertificate` is the in-memory representation of an alpha/beta-CROWN node
certificate read from JSON.

The checker returns this structure from `readAlphaBetaCROWNNodeCertificate`, and the blueprint uses
it as the documented shape of the artifact being checked.
-/
structure AlphaBetaCROWNNodeCertificate where
  /-- Affine-propagation context, including the chosen input node and flattened input dimension. -/
  ctx : AffineCtx
  /-- Optional per-node interval bounds used by nonlinear CROWN steps. -/
  ibp : Array (Option (FlatBox (ExecFloat.Binary 8 23)))
  /-- Optional per-node affine lower/upper bounds. -/
  crown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23)))
  /-- Optional per-node α values for ReLU lower relaxations. -/
  alpha : Array (Option (FlatTensor (ExecFloat.Binary 8 23)))
  /-- Optional per-node β phase annotations for ReLU nodes. -/
  beta : Array (Option (Array Int))

/-- Read an alpha/beta-CROWN node certificate from JSON on disk. -/
def readAlphaBetaCROWNNodeCertificate (g : Graph) (path : String) :
    IO AlphaBetaCROWNNodeCertificate := do
  let topObj ← readJsonObjectFile path
  let core ← parseCROWNNodeCoreCertificate g topObj
  let betaArr ←
    match ← optionalField? topObj "beta" "top-level" with
    | none => pure (Array.replicate g.nodes.size Json.null)
    | some betaJ => expectArray betaJ "top-level.beta"

  if hSize : betaArr.size = g.nodes.size then
    let mut beta : Array (Option (Array Int)) := Array.mkEmpty g.nodes.size
    for i in List.finRange g.nodes.size do
      let node := g.nodes[i.val]'i.isLt
      let hBeta : i.val < betaArr.size := by
        rw [hSize]
        exact i.isLt
      let betaJson := betaArr[i.val]'hBeta
      let betaEntry ← parseBetaVec? node.outShape.size betaJson
      beta := beta.push betaEntry
    pure { ctx := core.ctx, ibp := core.ibp, crown := core.crown, alpha := core.alpha,
           beta := beta }
  else
    throw <| IO.userError s!"beta length {betaArr.size} ≠ g.nodes.size {g.nodes.size}"

/-- Check the local α/β-CROWN enclosure condition for one node against a certificate entry. -/
def checkAlphaBetaCROWNNode (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (certAlpha : Array (Option (FlatTensor (ExecFloat.Binary 8 23))))
    (certBeta : Array (Option (Array Int)))
    (authoritativeCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))))
    (certCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))))
    (ctx : AffineCtx)
    (id : Nat) : IO (Bool × Option (FlatAffineBounds (ExecFloat.Binary 8 23))) := do
  let computed? :=
    alphaBetaCrownStepNode? (α := (ExecFloat.Binary 8 23)) g.nodes ps authoritativeIbp certAlpha
      certBeta
      authoritativeCrown ctx id
  let ok ←
    checkCROWNLikeNode "CROWNNodeCertAlphaBeta" g authoritativeIbp authoritativeCrown certCrown ctx
      id computed?
  pure (ok, computed?)

/-- The α/β-CROWN replay function associated with a parsed certificate. -/
def replayStep (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (cert : AlphaBetaCROWNNodeCertificate) :
    Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) → Nat →
      Option (FlatAffineBounds (ExecFloat.Binary 8 23)) :=
  fun replay id =>
    alphaBetaCrownStepNode? (α := (ExecFloat.Binary 8 23)) g.nodes ps authoritativeIbp cert.alpha
      cert.beta
      replay cert.ctx id

/--
The final in-memory acceptance decision for an α/β-CROWN artifact. It combines all diagnostic
checks with a complete pure replay whose proposition-level meaning is proved below.
-/
def AlphaBetaCROWNNodeCertificate.accepts
    (cert : AlphaBetaCROWNNodeCertificate) (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool) : Bool :=
  crownCertificateAccepts g (replayStep g ps authoritativeIbp cert) cert.crown diagnosticsOk

/-- Acceptance of the concrete α/β-CROWN decision supplies graph-level local consistency. -/
theorem AlphaBetaCROWNNodeCertificate.accepts_eq_true
    (cert : AlphaBetaCROWNNodeCertificate) (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool)
    (haccept : cert.accepts g ps authoritativeIbp diagnosticsOk = true) :
    NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownCertLocalOK
      (g := g) (step := replayStep g ps authoritativeIbp cert) cert.crown := by
  exact crownCertificateAccepts_eq_true g (replayStep g ps authoritativeIbp cert) cert.crown
    diagnosticsOk haccept

/--
Check a per-node α/β-CROWN certificate against Lean's propagation rules.

Returns `true` iff every supplied IBP box contains Lean's authoritative recomputation and every
node's affine replay data agrees exactly with Lean's α/β-CROWN step.
-/
def checkAlphaBetaCROWNNodeCertificate (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23)) (path :
  String) :
    IO Bool := do
  let cert ← readAlphaBetaCROWNNodeCertificate g path
  let authoritativeIbp := runIBP (α := (ExecFloat.Binary 8 23)) g ps
  let mut authoritativeCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) :=
    Array.replicate g.nodes.size none
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okIbp ← NN.Verification.IBPNodeCert.checkIBPNode g authoritativeIbp cert.ibp id
    let (okCrown, computed?) ←
      checkAlphaBetaCROWNNode g ps authoritativeIbp cert.alpha cert.beta authoritativeCrown
        cert.crown cert.ctx id
    authoritativeCrown := authoritativeCrown.set! id computed?
    ok := ok && okIbp && okCrown
  let accepted := cert.accepts g ps authoritativeIbp ok
  if accepted then
    IO.println
      ("[CROWNNodeCertAlphaBeta] artifact matched an authoritative Lean IBP " ++
        "and alpha/beta-CROWN replay.")
  pure accepted

end NN.Verification.CROWNNodeCertAlphaBeta
