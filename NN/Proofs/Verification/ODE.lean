/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Verification.ODE.Enclosure
public import NN.Proofs.Verification.ODE.EnclosureBackends

/-!
# ODE Verification Proofs

This module is the stable umbrella for proof-backed ODE enclosure results.

`Enclosure` contains the real-analysis statements from the learn-and-verify PINN enclosure
argument: clamped solutions remain inside certified lower/upper corridors and therefore satisfy
the original ODE while enclosed. It also contains the constant-extension theorem used to turn a
finite verified corridor into a global-time one under the paper's inward-pointing hypotheses.

`EnclosureBackends` then restates those theorems for backend-valued trajectories by applying their
`toReal` views. That file is only a view adapter; numeric soundness for FP32/IEEE evaluation must
come from the floating-point or certificate checker layer.

The numerical producers themselves (IBP/CROWN/certificate generation, CUDA execution, or external
tools) are intentionally not proved here. This layer states the mathematical endpoint those
producers must justify.
-/

@[expose] public section
