/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Common
public import NN.Runtime.Autograd.Torch.Utils

/-!
# `NN.API.Init`

Deterministic tensor initialization helpers (Xavier/Kaiming, etc.) that return TorchLean
`Spec.Tensor`s with shape tracked in the type.
-/

@[expose] public section

namespace NN
namespace API

namespace Init

open Spec

/-!
## Tensor Initialization Helpers

This module is a thin wrapper around `Runtime.Autograd.Torch.Init` that returns TorchLean
`Spec.Tensor`s with shapes tracked in the type.

All initializers are deterministic given an explicit `seed : Nat`, which is convenient for:
- reproducible demos
- stable tests
- proofs that want to fix concrete initial values

### PyTorch Mapping

The names intentionally mirror common PyTorch initializers:
- Xavier/Glorot: `torch.nn.init.xavier_uniform_`, `torch.nn.init.xavier_normal_`
- Kaiming/He: `torch.nn.init.kaiming_uniform_`, `torch.nn.init.kaiming_normal_`

See the PyTorch init docs:
`https://pytorch.org/docs/stable/nn.init.html`
-/

/-- Initialization scheme (zeros/ones/uniform/normal, etc.). -/
abbrev Scheme := _root_.Runtime.Autograd.Torch.Init.Scheme

/--
Initialize a `Float` tensor using the given scheme.

This is the "raw" initializer; most user code should prefer `tensor` which casts into `α`.
-/
def tensorFloat (sch : Scheme) (seed : Nat := 0) {s : Spec.Shape} : Spec.Tensor Float s :=
  _root_.Runtime.Autograd.Torch.Init.tensor (sch := sch) (seed := seed) (s := s)

/--
Initialize a tensor under an arbitrary scalar semantics `α`, by first generating `Float`s and
then casting elementwise via `cast : Float → α`.
-/
def tensor {α : Type} [Context α] (cast : Float → α) (sch : Scheme) (seed : Nat := 0) :
    {s : Spec.Shape} → Spec.Tensor α s
  | .scalar =>
      Common.castTensor cast (tensorFloat (sch := sch) (seed := seed) (s := Spec.Shape.scalar))
  | .dim n s =>
      Common.castTensor cast (tensorFloat (sch := sch) (seed := seed) (s := Spec.Shape.dim n s))

/--
Xavier/Glorot initializer for a linear weight matrix of shape `(outDim, inDim)`.

PyTorch analogue: `torch.nn.init.xavier_uniform_` (for example).
-/
def xavierW {α : Type} [Context α] (cast : Float → α) (outDim inDim : Nat) (seed : Nat := 0) :
    Spec.Tensor α (.dim outDim (.dim inDim .scalar)) :=
  Common.castTensor cast (_root_.Runtime.Autograd.Torch.Init.xavierW outDim inDim seed)

/--
Kaiming/He initializer for a linear weight matrix of shape `(outDim, inDim)`.

PyTorch analogue: `torch.nn.init.kaiming_uniform_` (for example).
-/
def kaimingW {α : Type} [Context α] (cast : Float → α) (outDim inDim : Nat) (seed : Nat := 0) :
    Spec.Tensor α (.dim outDim (.dim inDim .scalar)) :=
  Common.castTensor cast (_root_.Runtime.Autograd.Torch.Init.kaimingW outDim inDim seed)

end Init

end API
end NN
