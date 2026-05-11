/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation
public import NN.MLTheory.Proofs.Hopfield
public import NN.MLTheory.Proofs.ReLU
public import NN.MLTheory.Proofs.StateSpace
public import NN.MLTheory.Proofs.Verification

/-!
# MLTheory proof chapter

This is the curated entrypoint for theorem-heavy MLTheory developments. It groups the proof files
by mathematical theme:

- approximation and finite-precision universal approximation;
- Hopfield energy descent and convergence;
- ReLU algebra and compact-set approximation;
- state-space / Mamba scan and causality laws; and
- verification-facing robustness theorems.

We keep this as a proof chapter rather than mixing it into model definitions. The model/spec layer
defines semantics; this chapter proves reusable mathematical properties about those semantics.
-/

@[expose] public section
