/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Probability.DiffusionForward
public import NN.Proofs.Probability.NormalCdf
public import NN.Proofs.Probability.RandomSupport

/-!
# Probability Proofs

This module collects probability-theory facts used by TorchLean model proofs.

This bundle contains the Mathlib-backed forward diffusion noising kernel. The model
specification remains in the generative/diffusion spec layer; this proof layer records measure- and
kernel-level facts such as Gaussianity and Markov-kernel structure.
It also proves coordinate support and probability-endpoint behavior for the seeded uniform and
mask generators, and identifies the factor of the exact GELU with the distribution function of the
standard normal law.
-/

@[expose] public section
