/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Optimization.MuonCertificates

/-!
# Optimization Examples

A concrete Muon direction and parameter update, followed by conditional backend proof examples.

Reusable optimizer statements live under `NN.MLTheory.Optimization`. This folder gives short
worked examples that use the `Optim.Muon` certificate API. Runtime optimizer configuration
uses `TorchLean.optim`.
-/

@[expose] public section
