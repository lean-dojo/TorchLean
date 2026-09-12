/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.RL.Core -- shake: keep
public import NN.Spec.RL.MarkovMDP -- shake: keep

/-!
# The measure-theoretic MDP under `TorchLean.rl`

`NN.Spec.RL.MarkovMDP` states the Bellman theory for a general state space, where transitions are
probability kernels and the expected next value is a Bochner integral. This file gives that theory
the same `TorchLean.rl.markov` naming as the rest of the RL surface.

It is kept out of `NN.API.RL` on purpose. The kernels and the integral cost about 1400 mathlib
modules, none of which any executable RL algorithm touches, so a training run should not have to
load them. Import this module directly when you want to state or prove something about a Markov
decision process on a measurable space.
-/

@[expose] public section

namespace TorchLean
namespace rl

namespace markov
export Spec.RL.Markov
  (ValueFunction Policy MDP Valid
   transitionMeasure
   expectedNextValue actionValue
   bellmanPolicy bellmanOptimality)
end markov

end rl
end TorchLean
