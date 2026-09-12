/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.NodeReplay
public import NN.Verification.Cert.IBPCert
public import NN.Verification.Cert.IBPNodeCert
public import NN.Verification.Cert.CROWNNodeCert
public import NN.Verification.Cert.CROWNQuery.Json
public import NN.Verification.Cert.CROWNNodeCertAlphaBeta
public import NN.Verification.Cert.AbCrownLeafCert

/-!
# Certificate Verification

Public umbrella import for TorchLean's executable certificate checkers.

These modules define:
- artifact parsers (JSON → typed structures),
- recomputation checkers that replay bound propagation inside Lean, and
- directional enclosure checks for interval claims and exact binary32 checks for replay
  transcripts.

Artifacts are treated as untrusted inputs: they only receive credit after passing these checkers.
-/

@[expose] public section
