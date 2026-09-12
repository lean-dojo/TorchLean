/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.Verification.Cert.IBPNodeCert
public import NN.Spec.Core.Tensor -- shake: keep

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
- The certificate is untrusted; we accept it only if its binary32 affine transcript exactly
  matches Lean recomputation.
- Transcendental relaxations are checked only via structural recomputation, not via a formal
  "libm is correct" guarantee.
-/

@[expose] public section

namespace NN.Verification.CROWNNodeCert

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Cert
open NN.Verification.Json
open NN.Verification.Cert.NodeReplay
open Import.PyTorch
open Spec TorchLean
open TorchLean.Tensor
open Lean Data Json
open TorchLean.Floats.IEEE754

/-!
The helpers below are the JSON-facing boundary for the CROWN certificate checkers. They parse the
artifact, require exact binary32 agreement for affine replay data, and check parent and shape
requirements before invoking the semantic checker.
-/

/-- Read a CROWN node certificate from JSON on disk. -/
def readCROWNNodeCertificate (g : Graph) (path : String) : IO CROWNNodeCoreCertificate := do
  let topObj ← readJsonObjectFile path
  parseCROWNNodeCoreCertificate g topObj

/-- Check the local CROWN enclosure condition for one node against a certificate entry. -/
def checkCROWNNode (g : Graph) (ps : ParamStore IEEE32Exec)
    (authoritativeIbp : Array (Option (FlatBox IEEE32Exec)))
    (certAlpha : Array (Option (FlatTensor IEEE32Exec)))
    (authoritativeCrown : Array (Option (FlatAffineBounds IEEE32Exec)))
    (certCrown : Array (Option (FlatAffineBounds IEEE32Exec)))
    (ctx : AffineCtx)
    (id : Nat) : IO (Bool × Option (FlatAffineBounds IEEE32Exec)) := do
  let computed? :=
    alphaCrownStepNode? (α := IEEE32Exec) g.nodes ps authoritativeIbp certAlpha
      authoritativeCrown ctx id
  let ok ←
    checkCROWNLikeNode "CROWNNodeCert" g authoritativeIbp authoritativeCrown certCrown ctx id
      computed?
  pure (ok, computed?)

/-- The α-CROWN replay function associated with a parsed certificate. -/
def replayStep (g : Graph) (ps : ParamStore IEEE32Exec)
    (authoritativeIbp : Array (Option (FlatBox IEEE32Exec)))
    (cert : CROWNNodeCoreCertificate) :
    Array (Option (FlatAffineBounds IEEE32Exec)) → Nat →
      Option (FlatAffineBounds IEEE32Exec) :=
  fun replay id =>
    alphaCrownStepNode? (α := IEEE32Exec) g.nodes ps authoritativeIbp cert.alpha replay cert.ctx id

/--
The final in-memory acceptance decision for an α-CROWN artifact.

`diagnosticsOk` records parsing-independent IBP, shape, domain, and incremental-replay checks. The
second conjunct establishes local replay consistency. Semantic enclosure additionally requires
coverage of the relevant graph nodes and proved transfer rules for their operations.
-/
def certificateAccepts
    (cert : CROWNNodeCoreCertificate) (g : Graph) (ps : ParamStore IEEE32Exec)
    (authoritativeIbp : Array (Option (FlatBox IEEE32Exec)))
    (diagnosticsOk : Bool) : Bool :=
  crownCertificateAccepts g (replayStep g ps authoritativeIbp cert) cert.crown diagnosticsOk

/-- Acceptance of the concrete α-CROWN decision supplies graph-level local consistency. -/
theorem certificateAccepts_eq_true
    (cert : CROWNNodeCoreCertificate) (g : Graph) (ps : ParamStore IEEE32Exec)
    (authoritativeIbp : Array (Option (FlatBox IEEE32Exec)))
    (diagnosticsOk : Bool)
    (haccept : certificateAccepts cert g ps authoritativeIbp diagnosticsOk = true) :
    NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownCertLocalOK
      (g := g) (step := replayStep g ps authoritativeIbp cert) cert.crown := by
  exact crownCertificateAccepts_eq_true g (replayStep g ps authoritativeIbp cert) cert.crown
    diagnosticsOk haccept

/--
Check a per-node α-CROWN certificate against Lean's propagation rules.

Returns `true` iff every supplied IBP box contains Lean's authoritative recomputation and every
node's affine replay data agrees exactly with Lean's CROWN step.
-/
def checkCROWNNodeCertificate (g : Graph) (ps : ParamStore IEEE32Exec) (path : String) :
    IO Bool := do
  let cert ← readCROWNNodeCertificate g path
  let authoritativeIbp := runIBP (α := IEEE32Exec) g ps
  let mut authoritativeCrown : Array (Option (FlatAffineBounds IEEE32Exec)) :=
    Array.replicate g.nodes.size none
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okIbp ← NN.Verification.IBPNodeCert.checkIBPNode g authoritativeIbp cert.ibp id
    let (okCrown, computed?) ←
      checkCROWNNode g ps authoritativeIbp cert.alpha authoritativeCrown cert.crown cert.ctx id
    authoritativeCrown := authoritativeCrown.set! id computed?
    ok := ok && okIbp && okCrown
  let accepted := certificateAccepts cert g ps authoritativeIbp ok
  if accepted then
    IO.println "[CROWNNodeCert] artifact matched an authoritative Lean IBP and alpha-CROWN replay."
  pure accepted

end NN.Verification.CROWNNodeCert
