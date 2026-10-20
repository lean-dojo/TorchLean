/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Stability.Dynamics.Runtime

/-!
# Stability of dynamical systems

This entrypoint collects the discrete-time stability layer for maps of the form
$x_{t+1}=f(x_t)$ and input-driven systems $x_{t+1}=f(x_t,u_t)$.

The stability predicates themselves (contractivity, Lyapunov, asymptotic and global stability)
are `Spec.Dynamics` in `NN.Spec.Dynamics.System`. The runtime file provides `Float` diagnostics
for concrete systems. As with robustness, the diagnostic layer is empirical unless a
separate theorem connects it to a certified bound.

References:
- Khalil, *Nonlinear Systems*, 3rd edition.
- Sontag, "Input to State Stability: Basic Concepts and Results", 2008.
-/

@[expose] public section
