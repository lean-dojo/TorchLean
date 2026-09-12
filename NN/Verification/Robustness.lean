/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Robustness.Digits
public import NN.Verification.Robustness.MarginCert
public import NN.Verification.Robustness.MarginCertCLI
public import NN.Verification.Robustness.TopLabel

/-!
# Robustness verification

Reusable robustness-verification workflows and data-backed certified-accuracy utilities.

The artifacts under `NN/Examples/Verification/Robustness` stay thin: they choose default asset
paths and CLI names, while reusable loaders, typed model shapes, and certified-accuracy logic live
here.
-/

@[expose] public section
