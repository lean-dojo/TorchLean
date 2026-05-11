# TorchLean Widgets

TorchLean widgets are optional Lean infoview panels for inspecting tensors, IR graphs, runtime
state, training logs, verification artifacts, and small reinforcement-learning objects.

They are not part of the trusted mathematical kernel and they are not required for ordinary training
or verification. They are developer tools: small, inspectable views that make the library easier to
learn and debug from inside an editor.

Widgets matter because formal artifacts are hard to debug when they are invisible. A graph node,
tensor shape, affine bound, Float32 bit pattern, tape cotangent, or training log is much easier to
understand when it appears next to the Lean code that produced it.

## Import Surface

Use the umbrella entrypoint for normal work:

```lean
import NN.Entrypoint.Widgets
```

Use direct imports when you only want one family:

```lean
import NN.Widgets.Core.Tensor
import NN.Widgets.IR.Graph
import NN.Widgets.Interop.PyTorchTranslator
import NN.Widgets.Runtime.Training
import NN.Widgets.Verification.CROWN
```

## Widget Families

| Widget family | What it helps inspect |
| --- | --- |
| Tensor | small typed tensors and summaries |
| IR | node ids, parents, op tags, shape inference, execution traces |
| Runtime | autograd tapes, gradients, runtime contexts |
| Numerics | IEEE32 bit layouts and rounding views |
| Verification | IBP/CROWN states and bound tightness |
| Training | loss curves, metrics, confusion matrices |
| RL | GridWorld, PPO, and rollout artifacts |

## Layout

- `Core`: shared UI helpers, tensor rendering, and docstring panels.
- `IR`: graph structure, shape inference, graph rewrite diffs, and execution traces.
- `Interop`: PyTorch-to-TorchLean assistant panels and import/export diagnostics.
- `Runtime`: autograd tape, runtime context, and training log panels.
- `Numerics`: floating point inspection widgets such as binary32 bit views.
- `Verification`: CROWN/IBP certificate and bound propagation panels.
- `RL`: GridWorld and PPO rollout visualizers.

## Examples

Open these files in an editor with the Lean infoview enabled:

- `NN/Examples/Advanced/Widgets.lean`
- `NN/Examples/RL/PPOGridWorldView.lean`
- `NN/Examples/RL/PPOCartPoleView.lean`
- `NN/Examples/RL/PPOPongRamView.lean`
- `NN/Examples/Quickstart/Widgets.lean`, including `#pytorch_translate_file` for the lightweight
  PyTorch-to-TorchLean assistant.

The examples are intentionally small and editor friendly. File backed viewers render an error panel
when an artifact is missing instead of making the Lean build fail.

## PyTorch Translator Widget

`NN.Widgets.Interop.PyTorchTranslator` is the file-based translator widget for the workflow:

```text
PyTorch file -> recognized layer report -> TorchLean skeleton -> trust-boundary notes
```

It is a lightweight supported-subset assistant for common `nn.Sequential`-style models such as MLPs
and simple CNN blocks. It does not parse arbitrary Python and does not prove PyTorch execution
correct. For real modules, use the existing `torch.export` JSON bridge and Lean graph importer; the
widget helps users see whether their model is close to TorchLean's supported subset before they run
the full capture/import path.

In Lean files, the practical command is:

```lean
#pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"
```

The lower-level `#pytorch_translate_view someString` command exists for tests and future editor
integrations that already have selected text in memory.

## Design Inspiration

The widgets follow the style of Lean's ProofWidgets ecosystem: render structured mathematical or
runtime objects close to the code that produced them. Graph panels include GraphViz DOT snippets
because DOT remains a common language for graph debugging, while training log panels use
TensorBoard-like curves without requiring a separate web server.
