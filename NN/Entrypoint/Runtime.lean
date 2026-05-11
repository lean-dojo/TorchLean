/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Context
public import NN.Runtime.Scalar
public import NN.Runtime.Variable

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Autograd.Ops

public import NN.Runtime.Autograd.Overview
public import NN.Runtime.Autograd.Compiled
public import NN.Runtime.Autograd.Engine
public import NN.Runtime.Autograd.Torch
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train
public import NN.Runtime.Autograd.Utils

public import NN.Runtime.External
public import NN.Runtime.PyTorch

public import NN.Runtime.Optim
public import NN.Runtime.RL

/-!
# Runtime Entrypoint

This is the broad import for TorchLean's executable layer. It collects the runtime pieces that are
useful when building, training, importing, exporting, or checking runnable models:

- the eager and compiled autograd engines;
- the lower-level `Runtime.Autograd.Torch` session operations;
- the higher-level `Runtime.Autograd.TorchLean` front-end used by `NN.API.Runtime`;
- deterministic dataset/training utilities;
- optional external-process helpers for untrusted producer / trusted checker workflows;
- pure optimizer and scheduler equations;
- PyTorch import/export bridge infrastructure; and
- typed reinforcement-learning runtime helpers.

For ordinary user code, prefer `import NN` or `import NN.API.Runtime`. Import this file when you
intentionally want the full executable subsystem. If you only need pure tensor semantics and
theorems, prefer `NN.Entrypoint.Spec` or `NN.Entrypoint.Proofs`; those imports keep runtime bridge
dependencies out of the build.

The runtime entrypoint deliberately imports only reusable bridge infrastructure under
`NN.Runtime.PyTorch.*`. Demo-only MLP/CNN/Transformer round-trip code lives under
`NN.Examples.Interop.PyTorch.*`, so ordinary runtime imports do not pull example modules into the
library surface.

References / context:
- PyTorch autograd overview:
  https://pytorch.org/docs/stable/autograd.html
- PyTorch `nn.Module` / tensor ops surface:
  https://pytorch.org/docs/stable/nn.html
  https://pytorch.org/docs/stable/torch.html
- TorchLean’s import/export bridge details live in `NN.Runtime.PyTorch.Export.Core` and
  `NN.Runtime.PyTorch.Import.Core`.
-/

@[expose] public section
