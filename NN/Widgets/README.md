# Widgets

Optional Lean infoview panels for inspecting library objects. Import `NN.Widgets` for all panels
or a specific module such as `NN.Widgets.Core.Tensor`. Rendering an object does not validate it;
the panel identifies any checker it runs.

| Directory | Views |
| --- | --- |
| `Core` | Tensors, docstrings, shared UI helpers |
| `IR` | Graph structure, shape inference, rewrites, execution traces |
| `Runtime` | Autograd tapes, training logs, confusion matrices |
| `Numerics` | Binary32 values, bits, and rounding |
| `Verification` | CROWN/IBP diagnostic state and bounds |
| `RL` | GridWorld policies/paths, PPO rollouts, transition contracts |
| `Interop` | PyTorch constructor recognition and model sketches |

Start with `NN/Examples/Quickstart/Widgets.lean` or `NN/Examples/DeepDives/Widgets.lean`.
File-backed views show an error panel for missing or malformed artifacts.

Training and text models share `#train_log_view` and `#train_log_file_view`; multiline notes retain
prompts and generated samples. Run training through the model CLI and inspect the saved log.
Confusion-matrix clipping affects displayed cells only; statistics include every class.

## PyTorch sketches

```lean
#pytorch_translate_file "NN/Examples/Interop/PyTorch/MLP/train_mlp.py"
```

The analyzer recognizes constructor lines with literal scalar dimensions and common elementwise
calls. Unsupported options, symbolic sizes, or tuple-valued kernels are reported. It produces a
starting sketch, not an analysis of Python control flow or a proof of equivalent execution.
CNN rows describe shapes; they do not automatically produce an executable CNN.
Use the `torch.export` JSON bridge for model capture and structural validation.
`#pytorch_translate_view text` accepts a snippet already held in memory.
