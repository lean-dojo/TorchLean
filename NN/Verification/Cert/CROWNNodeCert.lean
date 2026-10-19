/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.PhasedCROWN
public import NN.Verification.Cert.IBPNodeCert

/-!
# CROWNNodeCert

Per-node α-CROWN certificate checking (graph dialect), with optional ReLU phases.

This mirrors `NN.Verification.Cert.IBPNodeCert`, but for affine bounds produced by a CROWN/DeepPoly
pass with optional α-parameters for the ReLU lower relaxation (α-CROWN).

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
- The certificate is untrusted; we accept it only if its binary32 affine transcript exactly
  matches Lean recomputation.
- Transcendental relaxations are checked only via structural recomputation, not via a formal
  "libm is correct" guarantee.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.Verification.Cert.CROWNNodeCert

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Cert
open NN.Verification.Json
open NN.Verification.Cert.NodeReplay
open Spec TorchLean
open TorchLean.Tensor
open Lean Json

/-!
The helpers below are the JSON-facing boundary for the CROWN certificate checkers. They parse the
artifact, require exact binary32 agreement for affine replay data, and check parent and shape
requirements. Acceptance proves local replay consistency, not a real-input enclosure on its own.
-/

/-- Read a CROWN node certificate from JSON on disk. -/
def read (g : Graph) (path : String) : IO CROWNCertificate := do
  let topObj ← readJsonObjectFile path
  parseCROWNCertificate g topObj

/--
The per-node part of the check. Every supplied IBP box must contain Lean's authoritative interval,
and every node's affine entry must equal the bound `step` replays from the nodes before it; `label`
names the checker in its messages. The result is the `diagnosticsOk` input of `accepts`.

`check` passes `replayStep`, the function whose acceptance theorem is proved below, so this loop
and the pure acceptance decision replay the same rule.
-/
def checkNodeEntries (label : String) (g : Graph)
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (cert : CROWNCertificate)
    (step : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) → Nat →
      Option (FlatAffineBounds (ExecFloat.Binary 8 23))) : IO Bool := do
  let mut authoritativeCrown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) :=
    Array.replicate g.nodes.size none
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okIbp ← NN.Verification.Cert.IBPNodeCert.checkNode g authoritativeIbp cert.ibp id
    let computed? := step authoritativeCrown id
    let okCrown ← checkCROWNNode label g authoritativeIbp authoritativeCrown cert.crown
      cert.ctx id computed?
    authoritativeCrown := authoritativeCrown.set! id computed?
    ok := ok && okIbp && okCrown
  pure ok

/-- The replay of a parsed certificate: α-CROWN, with the certificate's ReLU phases if any. -/
def replayStep (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (cert : CROWNCertificate) :
    Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) → Nat →
      Option (FlatAffineBounds (ExecFloat.Binary 8 23)) :=
  fun replay id =>
    phasedCrownStepNode? (α := (ExecFloat.Binary 8 23)) g.nodes ps authoritativeIbp cert.alpha
      cert.phases replay cert.ctx id

/--
The final in-memory acceptance decision.

`diagnosticsOk` records parsing-independent IBP, shape, domain, and incremental-replay checks. The
second conjunct establishes local replay consistency. Semantic enclosure additionally requires
coverage of the relevant graph nodes and proved transfer rules for their operations.
-/
def accepts
    (cert : CROWNCertificate) (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool) : Bool :=
  crownCertificateAccepts g (replayStep g ps authoritativeIbp cert) cert.crown diagnosticsOk

/-- Acceptance supplies graph-level local consistency of the replay. -/
theorem accepts_eq_true
    (cert : CROWNCertificate) (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (authoritativeIbp : Array (Option (FlatBox (ExecFloat.Binary 8 23))))
    (diagnosticsOk : Bool)
    (haccept : accepts cert g ps authoritativeIbp diagnosticsOk = true) :
    NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownCertLocalOK
      (g := g) (step := replayStep g ps authoritativeIbp cert) cert.crown :=
  crownCertificateAccepts_eq_true g (replayStep g ps authoritativeIbp cert) cert.crown
    diagnosticsOk haccept

/--
Check a per-node CROWN certificate against Lean's propagation rules.

Returns `true` iff every supplied IBP box contains Lean's authoritative recomputation and every
node's affine replay data agrees exactly with Lean's step.

The per-node loop is part of the verdict: it checks IBP containment, parent coverage, shape and
domain preconditions, and dimensions, which the pure replay in `accepts` does not. The pure replay
runs only when the loop passes. The certificate's `ibp` boxes are only checked for containment;
replay uses Lean's own IBP pass, so they add no information to the affine check.
-/
def check (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23)) (path : String) : IO Bool := do
  let cert ← read g path
  let authoritativeIbp := runIBP (α := (ExecFloat.Binary 8 23)) g ps
  let ok ← checkNodeEntries "CROWNNodeCert" g authoritativeIbp cert
    (replayStep g ps authoritativeIbp cert)
  let accepted := accepts cert g ps authoritativeIbp ok
  if accepted then
    IO.println "[CROWNNodeCert] artifact matched an authoritative Lean IBP and alpha-CROWN replay."
  pure accepted

end NN.Verification.Cert.CROWNNodeCert
