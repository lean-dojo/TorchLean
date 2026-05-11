/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Probability.DiffusionForward

/-!
# Probability Proofs

This module collects probability-theory facts used by TorchLean model proofs.

This bundle contains the Mathlib-backed forward diffusion noising kernel. The model
specification remains in the generative/diffusion spec layer; this proof layer records measure- and
kernel-level facts such as Gaussianity and Markov-kernel structure.
-/

@[expose] public section
