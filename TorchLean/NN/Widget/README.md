# Widgets

Optional Lean infoview panels for inspecting library objects. They are not part of TorchLean's
trusted semantics. A file imports the panel modules it uses, such as
`TorchLean.NN.Widget.Tensor`. Rendering an object doesn't validate it, and a panel that runs a
checker names it.

| Files | Views |
| --- | --- |
| `Tensor.lean`, `Docs.lean`, `UI.lean` | Tensors, docstrings, shared UI helpers |
| `IR/` | Graph structure, shape inference, rewrites, execution traces |
| `Autograd.lean`, `Training.lean` | Autograd tapes, training logs, confusion matrices |
| `Float32.lean` | Binary floating-point values, bits and rounding |
| `RL/` | PPO rollouts, transition contracts |
| `PyTorchTranslator.lean` | PyTorch constructor recognition and model sketches |

The CROWN and IBP panels are in [`TorchLean/Verify/CROWN`](../../Verify/CROWN/README.md).

Start with the [widget gallery](../../../TorchLeanExamples/DeepDives/WidgetGallery.lean).
File-backed views show an error panel for missing or malformed artifacts.

`#train_log_view` and `#train_log_file_view` show the logs of all training commands, text models
included. A text model's log notes hold its prompt and generated text, and the panel keeps the line
breaks of multiline notes. Training commands write a JSON log by default, and `--log PATH` picks
another path (`--log off` disables it). For example, after
`python3 scripts/datasets/download_example_data.py --auto-mpg`,
`scripts/lake.sh exe torchlean mlp --device cpu --steps 50` writes
`data/examples/mlp_trainlog.json`, which `#train_log_file_view` opens. Confusion-matrix clipping
affects displayed cells only, and statistics include every class.

## PyTorch sketches

```lean
#pytorch_translate_file "TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py"
```

The analyzer recognizes constructor lines with literal scalar dimensions and common elementwise
calls, and it reports unsupported options, symbolic sizes and tuple-valued kernels. The output is a
starting sketch: the analyzer doesn't follow Python control flow, and nothing proves the sketch runs
like the original. A convolution row becomes a comment in the sketch and gives no executable layer.
Use the `torch.export` JSON bridge for model capture and structural validation.
`#pytorch_translate_view text` takes a snippet already held in memory.
