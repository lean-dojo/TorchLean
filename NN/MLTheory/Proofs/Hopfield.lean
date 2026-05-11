/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Hopfield.Basic
public import NN.MLTheory.Proofs.Hopfield.Convergence
public import NN.MLTheory.Proofs.Hopfield.Dynamics
public import NN.MLTheory.Proofs.Hopfield.Energy
public import NN.MLTheory.Proofs.Hopfield.Progress

/-!
# Hopfield-network proofs

This entrypoint collects the discrete Hopfield proof development:

- Boolean state bookkeeping;
- the classical symmetric-zero-diagonal energy function;
- one-coordinate update energy monotonicity;
- full-sweep progress; and
- convergence to a fixed point within a finite bound.

The proofs follow the standard Lyapunov-energy analysis of Hopfield networks, stated over
TorchLean's spec-level model so the theorem layer and model layer share one semantics.

References:
- Hopfield, "Neural networks and physical systems with emergent collective computational
  abilities", PNAS 1982.
- Hopfield, "Neurons with graded response have collective computational properties like those of
  two-state neurons", PNAS 1984.
-/

@[expose] public section
