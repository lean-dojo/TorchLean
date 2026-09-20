/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Core
public import NN.Runtime.Autograd.Model.Functional.Einsum
public import NN.Runtime.Autograd.Model.Functional.Fourier
public import NN.Runtime.Autograd.Model.Functional.SelectiveScan
public import NN.Runtime.Autograd.Model.Functional.ShapeOps
public import NN.Runtime.Autograd.Model.Functional.Spectral

/-!
# Functional

TorchLean functional helpers in the style of `torch.*` building blocks.

These are derived operations built from `Runtime.Autograd.Torch.Ops`, so eager execution and typed
graph construction share the same model and loss definitions.

For background, see the PyTorch documentation for `torch.nn.functional`, `torch.autograd`, and
checkpointing, together with the standard reverse-mode AD references by Linnainmaa and by Griewank
and Walther.

`Functional.EinsumDynamic` is deliberately not re-exported here. It holds the runtime-checked,
string-driven `einsum?`, which is by far the most expensive declaration in this subtree and which
nothing in the runtime or the API layer calls. `Runtime.Autograd.Model` re-exports it, so it stays
part of the public surface without sitting in front of the layer and trainer modules that every
build has to get through.
-/
