/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.ReLU.Approx.ReLUMulApprox
public import NN.MLTheory.Proofs.ReLU.Approximation.CompactSet
public import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge

/-!
# ReLU network algebra and approximation

This entrypoint collects the ReLU-specific proof layer. It includes:

- exact algebraic bridges between TorchLean's two-layer MLP spec and affine/ReLU expressions;
- multiplication approximation on boxes by ReLU networks; and
- compact-set approximation by coordinate-polynomial density plus ReLU realizations.

This module stays separate from the architecture files: the model files define how an MLP computes,
while this chapter proves what ReLU networks can represent and approximate.
-/

@[expose] public section
