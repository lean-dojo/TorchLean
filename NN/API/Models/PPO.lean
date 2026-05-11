/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# PPO Model Helpers (API)

Reusable actor/critic MLP constructors for PPO examples.

These helpers intentionally cover only the neural-network shape. Environment collection, trust
boundary checks, advantage computation, and optimizer loops stay in the examples/runtime modules.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a simple PPO actor/critic pair over vector observations. -/
structure PPOActorCriticConfig where
  obsDim : Nat
  hiddenDim : Nat
  nActions : Nat
deriving Repr

abbrev ppoActorInShape (cfg : PPOActorCriticConfig) (pfx : Shape) : Shape :=
  pfx.appendDim cfg.obsDim

abbrev ppoActorOutShape (cfg : PPOActorCriticConfig) (pfx : Shape) : Shape :=
  pfx.appendDim cfg.nActions

abbrev ppoCriticOutShape (_cfg : PPOActorCriticConfig) (pfx : Shape) : Shape :=
  pfx.appendDim 1

/-- Actor MLP mapping observations to action logits. -/
def ppoActor (cfg : PPOActorCriticConfig) (pfx : Shape) :
    nn.M (nn.Sequential (ppoActorInShape cfg pfx) (ppoActorOutShape cfg pfx)) :=
  nn.sequential![
    nn.linear cfg.obsDim cfg.hiddenDim (pfx := pfx),
    nn.tanh,
    nn.linear cfg.hiddenDim cfg.nActions (pfx := pfx)
  ]

/-- Critic MLP mapping observations to a scalar value estimate. -/
def ppoCritic (cfg : PPOActorCriticConfig) (pfx : Shape) :
    nn.M (nn.Sequential (ppoActorInShape cfg pfx) (ppoCriticOutShape cfg pfx)) :=
  nn.sequential![
    nn.linear cfg.obsDim cfg.hiddenDim (pfx := pfx),
    nn.tanh,
    nn.linear cfg.hiddenDim 1 (pfx := pfx)
  ]

end models
end nn

end API
end NN

