/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Arithmetic -- shake: keep
public import NN.Runtime.Autograd.Torch.Initialization -- shake: keep
public import NN.Tensor -- shake: keep

/-!
# Tensor Initialization

Deterministic tensor initialization helpers (Xavier/Kaiming, etc.) that return TorchLean
`TorchLean.Tensor`s with shape tracked in the type.
-/

@[expose] public section

namespace TorchLean
namespace Init

open Spec TorchLean

/-!
## Tensor Initialization Helpers

This module exposes `Runtime.Autograd.Torch.Init` through TorchLean
`TorchLean.Tensor`s with shapes tracked in the type.

All initializers are deterministic given an explicit `seed : Nat`, which is convenient for:
- reproducible examples
- stable tests
- proofs that want to fix concrete initial values

### PyTorch Mapping

The names mirror common PyTorch initializers:
- Xavier/Glorot: `torch.nn.init.xavier_uniform_`, `torch.nn.init.xavier_normal_`
- Kaiming/He: `torch.nn.init.kaiming_uniform_`, `torch.nn.init.kaiming_normal_`

See the PyTorch init docs:
`https://pytorch.org/docs/stable/nn.init.html`
-/

export Runtime.Autograd.Torch.Init (Scheme)

/--
Initialize a tensor under any executable arithmetic.

The element type and dimensions are normally inferred from the expected tensor type. Initialization
is generated once in `Float`, then converted through the element type's canonical
`Runtime.FromFloat` instance.
-/
def tensor {α : Type} [TorchLean.Storage α] [Runtime.FromFloat α] {shape : Shape}
    (scheme : Scheme) (seed : Nat := 0) : Tensor α shape :=
  TorchLean.Tensor.map (Runtime.ofFloat (α := α))
    (Runtime.Autograd.Torch.Init.tensor
      (sch := scheme) (seed := seed) (s := shape))

/--
Xavier/Glorot uniform initialization for a linear weight matrix.

Both widths are inferred from the expected `Tensor α [outputWidth, inputWidth]` type.
-/
def xavierUniform {α : Type} [TorchLean.Storage α] [Runtime.FromFloat α]
    {outputWidth inputWidth : Nat} (seed : Nat := 0) :
    Tensor α [outputWidth, inputWidth] :=
  tensor (.xavierUniform inputWidth outputWidth) seed

/--
Kaiming/He uniform initialization for a linear weight matrix used before a ReLU.

Both widths are inferred from the expected `Tensor α [outputWidth, inputWidth]` type.
-/
def kaimingUniform {α : Type} [TorchLean.Storage α] [Runtime.FromFloat α]
    {outputWidth inputWidth : Nat} (seed : Nat := 0) :
    Tensor α [outputWidth, inputWidth] :=
  tensor (.kaimingUniform inputWidth) seed

end Init
end TorchLean
