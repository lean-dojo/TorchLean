/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph
public import NN.Runtime.Autograd.IRExec
public import NN.Runtime.Autograd.Engine
public import NN.Runtime.Autograd.Torch
public import NN.Runtime.Autograd.Model
public import NN.Runtime.Autograd.Train
public import NN.Runtime.External
public import NN.Runtime.PyTorch
public import NN.Runtime.Optim
public import NN.Runtime.RL

/-!
# Runtime

Import this file when you need TorchLean's executable layer. It collects the runtime pieces used for
building, training, importing, exporting, or checking runnable models:

- the eager and typed graph autograd engines;
- the lower-level `Runtime.Autograd.Torch` session operations;
- the higher-level `Runtime.Autograd.Model` front-end used by `NN.API.Runtime`;
- deterministic dataset/training utilities;
- optional external-process helpers for untrusted producer / trusted checker workflows;
- pure optimizer and scheduler equations;
- PyTorch import/export bridge infrastructure; and
- typed reinforcement-learning runtime helpers.

For ordinary application code, prefer `import NN.API`. Import this file when you need the full
executable subsystem. If you only need pure tensor semantics and theorems, prefer
`NN.Spec` or `NN.Proofs`; those imports keep runtime bridge dependencies out
of the build.

The runtime import includes reusable bridge infrastructure under
`NN.Runtime.PyTorch.*`. Example-only MLP/CNN/Transformer round-trip code lives under
`NN.Examples.Interop.PyTorch.*`, so ordinary runtime imports do not pull example modules into the
library API.

References / context:
- PyTorch autograd overview:
  https://pytorch.org/docs/stable/autograd.html
- PyTorch `nn.Module` / tensor ops API:
  https://pytorch.org/docs/stable/nn.html
  https://pytorch.org/docs/stable/torch.html
- TorchLean’s import/export bridge details live in `NN.Runtime.PyTorch.Export.Core` and
  `NN.Runtime.PyTorch.Import.Core`.
-/

@[expose] public section
