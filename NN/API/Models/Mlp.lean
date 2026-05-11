/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# MLP Model Helpers (API)

Compact, reusable MLP-shaped model constructors for TorchLean examples.

These live in the API layer so runnable examples can stay focused on:
data loading, training loops, and CLI flags, rather than repeating the same
`linear → activation → linear` boilerplate.

Scope note: these are building blocks, not pretrained checkpoints.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a single-hidden-layer MLP over batched row vectors. -/
structure Mlp1Config where
  batch : Nat
  inDim : Nat
  hidDim : Nat
  outDim : Nat
deriving Repr

/-- Input shape `(batch × inDim)` for an `Mlp1Config`. -/
abbrev mlp1InShape (cfg : Mlp1Config) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.inDim

/-- Output shape `(batch × outDim)` for an `Mlp1Config`. -/
abbrev mlp1OutShape (cfg : Mlp1Config) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.outDim

/--
Build a single-hidden-layer MLP with ReLU activation:

`linear(inDim → hidDim) → relu → linear(hidDim → outDim)`.
-/
def mlp1Relu (cfg : Mlp1Config) :
    nn.M (nn.Sequential (mlp1InShape cfg) (mlp1OutShape cfg)) :=
  nn.sequential![
    nn.linear cfg.inDim cfg.hidDim (pfx := NN.Tensor.Shape.Vec cfg.batch),
    nn.relu,
    nn.linear cfg.hidDim cfg.outDim (pfx := NN.Tensor.Shape.Vec cfg.batch)
  ]

end models
end nn

end API
end NN
