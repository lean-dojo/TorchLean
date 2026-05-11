/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.StateSpace.MambaCausality
public import NN.MLTheory.Proofs.StateSpace.Scan

/-!
# State-space model proofs

This entrypoint collects the proof layer for S4/Mamba-style state-space sequence models. We prove
two complementary facts:

- affine selective scan summaries compose associatively and agree with left-to-right recurrence;
- recurrent S4/Mamba runners are prefix-causal, so appending future tokens cannot change outputs
  already emitted for a prefix.

The CUDA and runtime implementations are free to use efficient scan schedules, but their semantic
target is the spec-level recurrence captured here.

References:
- Gu, Goel, and Ré, "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022.
- Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", COLM 2024.
- Dao and Gu, "Transformers are SSMs", ICML 2024.
-/

@[expose] public section
