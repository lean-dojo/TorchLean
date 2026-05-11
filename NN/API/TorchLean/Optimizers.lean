/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime

/-!
# Optimizer Convenience Constructors

This module provides a compact PyTorch-shaped optimizer surface for the TorchLean trainer API.

## PyTorch Mapping

The names and default hyperparameters mirror common PyTorch optimizers:
- SGD: `https://pytorch.org/docs/stable/generated/torch.optim.SGD.html`
- Adam: `https://pytorch.org/docs/stable/generated/torch.optim.Adam.html`
- AdamW: `https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html`

General optimizer docs:
`https://pytorch.org/docs/stable/optim.html`
-/

@[expose] public section


namespace NN
namespace API
namespace TorchLean
namespace Optimizers

/-- Public optimizer config alias for the high-level trainer surface. -/
abbrev Config := API.TorchLean.Trainer.Optimizer

-- Re-export constructors from `API.TorchLean.Trainer` (canonical).
/-- Construct an SGD optimizer configuration. -/
abbrev sgd := API.TorchLean.Trainer.sgd

/-- Construct a momentum-SGD optimizer configuration. -/
abbrev momentumSGD := API.TorchLean.Trainer.momentumSGD

/-- Construct an Adam optimizer configuration. -/
abbrev adam := API.TorchLean.Trainer.adam

/-- Construct an AdamW optimizer configuration. -/
abbrev adamw := API.TorchLean.Trainer.adamw

end Optimizers
end TorchLean
end API
end NN
