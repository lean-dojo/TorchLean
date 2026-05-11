/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.Widgets
public import NN.Floats.IEEEExec.Exec32
public import NN.IR.Graph
public import NN.Runtime.Training.Log
public import NN.Tensor.API

/-!
# Quickstart: Widgets

TorchLean widgets are editor-side inspection tools. They are not part of the runtime semantics and
they do not change proofs; they simply render the values you already have in Lean.

Use this file in the editor:

- put the cursor on a `#tensor_view`, `#float32_view`, `#ir_view`, or `#train_log_view` command;
- put the cursor on `#pytorch_translate_file` to preview a PyTorch-to-TorchLean skeleton;
- Lean's infoview renders an interactive panel;
- if you want the full gallery, open `NN.Examples.Advanced.Widgets`.

This quickstart keeps only the smallest useful examples so the widget path does not feel like a
full model zoo.
-/

@[expose] public section

namespace NN.Examples.Quickstart.Widgets

open Spec
open TorchLean.Floats.IEEE754

/-- A small vector, built with the same typed tensor constructor used in ordinary code. -/
def vector : Spec.Tensor Float (shape![4]) :=
  tensor! [1.0, 2.0, 3.0, 4.0]

/-- A small matrix where the shape is visible both in the type and in the widget. -/
def matrix : Spec.Tensor Int (shape![2, 3]) :=
  tensor! [
    [1, 2, 3],
    [4, 5, 6]
  ]

/-- A binary32 value; the widget shows sign/exponent/fraction fields and classification flags. -/
def one32 : IEEE32Exec :=
  IEEE32Exec.ofFloat 1.0

/-- A small IR graph: input plus constant, then an add node. -/
def addGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := shape![2] },
      { id := 1, parents := [], kind := .const shape![2], outShape := shape![2] },
      { id := 2, parents := [0, 1], kind := .add, outShape := shape![2] }
    ] }

/-- A minimal train log; real training examples can write the same structure as JSON. -/
def tinyTrainLog : Runtime.Training.TrainLog :=
  { title := "Quickstart loss"
    steps := #[0, 1, 2, 3]
    series := #[
      { name := "loss", values := #[1.0, 0.45, 0.22, 0.12], color := "#c44" }
    ]
    notes := #["editor-only visualization; runtime training logs use the same schema"] }

/-!
The commands below are ordinary Lean commands that render editor panels through ProofWidgets.
They do not alter the definitions above and they are not part of any proof. This is why widgets are
safe to keep in quickstarts: deleting the commands leaves the same tensors, graphs, logs, and Python
source files behind.
-/

#tensor_view vector
#tensor_view matrix
#tensor_stats_view vector

#float32_view one32
#float32_round_view (0.1 : Float)

#ir_view addGraph
#shape_infer_view addGraph

/-!
The translator widget is intentionally placed next to the IR/shape widgets. It is a lightweight
preview for "what would this PyTorch layer stack look like in TorchLean?" The checked graph-capture
path is still the `torch.export` importer, which parses and validates explicit IR JSON.
-/
#pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"

#train_log_view tinyTrainLog

end NN.Examples.Quickstart.Widgets
