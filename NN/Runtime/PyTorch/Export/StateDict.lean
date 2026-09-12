/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core

/-!
# PyTorch `state_dict` Bridge

This module is the general weight-interchange path for PyTorch users.

The important split is:

- **Weights** move through PyTorch `state_dict`s. PyTorch’s own documentation recommends saving a
  module’s learned parameters with `torch.save(model.state_dict(), path)` because that is the most
  flexible restoration format.
- **Graphs** move through graph capture (`torch.export`, FX, ONNX, or TorchLean `NN.IR.Graph`).
  A `state_dict` alone does not describe the model architecture; it only names tensors.

Lean should not try to parse PyTorch pickle/zip checkpoints directly. Instead, we emit a small
Python adapter that loads a checkpoint with PyTorch, normalizes common wrappers such as
`{"state_dict": ...}`, and writes shape-checkable JSON:

```json
{
  "params": { "layer.weight": [[...]], "layer.bias": [...] },
  "meta": { "layer.weight": { "shape": [out, in], "dtype": "torch.float32" } }
}
```

`NN.Runtime.PyTorch.Import.Core` then parses the `"params"` object into typed TorchLean tensors.
Architecture-specific loaders are still useful, but only for mapping names and shapes. The transport
format itself is model-agnostic.

References:
- PyTorch documentation, "Saving and Loading Models":
  `https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`
- PyTorch `torch.export` user guide:
  `https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html`
- PyTorch FX overview:
  `https://docs.pytorch.org/docs/stable/fx.html`
-/

@[expose] public section

namespace Export
namespace PyTorch
namespace StateDict

open Export.PyTorch

end StateDict
end PyTorch
end Export
