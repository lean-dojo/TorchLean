/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon.Certificates
public import NN.MLTheory.Optimization.Muon.Core
public import NN.MLTheory.Optimization.Muon.NewtonSchulz
public import NN.MLTheory.Optimization.Muon.QR

/-!
# Muon

Muon updates a momentum buffer and passes that buffer through an explicit matrix orthogonalizer
before applying the parameter step. The modules collected here separate the reusable contracts,
Newton-Schulz backend, step certificates, and exact QR backend.
-/

@[expose] public section
