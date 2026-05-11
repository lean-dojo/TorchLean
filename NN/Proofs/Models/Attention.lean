/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Models.Attention.CausalMask
public import NN.Proofs.Models.Attention.PermutationEquivariance
public import NN.Proofs.Models.Attention.Weights

/-!
# Attention Model Proofs

This is the proof-facing umbrella for attention model facts.

The executable/spec definitions live under `NN.Spec.Layers.Attention`. This module collects theorems
about those definitions: exact causal-mask semantics, attention-weight normalization, and
permutation equivariance of self-attention without positional encodings.

Import this file when downstream proofs need the attention theorem bundle. Import the spec module
directly when you only need the definitions.
-/

@[expose] public section

