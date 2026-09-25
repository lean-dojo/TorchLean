/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardEvaluation
public import NN.Proofs.Verification.ODE
public import NN.Verification.Cert.FiniteArtifactSemantics

/-!
# Verification Proofs

Stable umbrella for proof-backed verification results.

This namespace is for theorems that turn verification hypotheses into mathematical guarantees.
Executable certificate checkers and parsers live under `NN.Verification`; the proof layer here is
where the accepted hypotheses are connected to real-analysis or model-level soundness statements.
-/

@[expose] public section
