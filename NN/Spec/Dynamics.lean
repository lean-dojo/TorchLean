/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Dynamics.System
public import NN.Spec.Dynamics.StateSpace

/-!
# Spec dynamics

Umbrella import for deterministic and driven dynamical-system specifications.

These definitions are used by recurrent models, state-space models, generative samplers, and the
theory layer whenever an architecture is best understood as repeated state transition.
-/

@[expose] public section
