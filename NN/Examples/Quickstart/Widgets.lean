/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Widgets
public import NN.API.Trainer.Reporting

/-!
# Quickstart: Widgets

TorchLean widgets are editor-side inspection tools. They render values already present in Lean
without changing runtime semantics or proofs.

Try these commands in the editor:

- put the cursor on a `#tensor_view`, `#float32_view`, or `#train_log_view` command;
- Lean's infoview renders an interactive panel;
- graph, rewrite, translator, verification, and RL widgets live in
  `NN.Examples.DeepDives.Widgets`.

This quickstart keeps only the smallest useful examples; the full widget gallery lives in
`NN.Examples.DeepDives.Widgets`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace NN.Examples.Quickstart.Widgets

open TorchLean

/-- A small vector, built with the same typed tensor constructor used in ordinary code. -/
def vector : Tensor Float [4] :=
  [1.0, 2.0, 3.0, 4.0]

/-- A small matrix where the shape is visible both in the type and in the widget. -/
def matrix : Tensor Int [2, 3] :=
  [
    [1, 2, 3],
    [4, 5, 6]
  ]

/-- A binary32 value; the widget shows sign/exponent/fraction fields and classification flags. -/
def one32 : Binary 8 23 :=
  (fun x => (ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat x))) : Binary 8 23))
    1.0

/-- A minimal training log; runtime examples can write the same structure as JSON. -/
def tinyTrainLog : Training.TrainLog :=
  { title := "Quickstart loss"
    steps := #[0, 1, 2, 3]
    series := #[
      { name := "loss", values := #[1.0, 0.45, 0.22, 0.12], color := "#c44" }
    ]
    notes := #["editor-only visualization; runtime training logs use the same schema"] }

/-!
The commands below render editor panels through ProofWidgets. They inspect existing values and do
not change runtime behavior or proof status.
-/

#tensor_view vector
#tensor_view matrix
#tensor_stats_view vector

#float32_view one32
#float32_round_view (0.1 : Float)

#train_log_view tinyTrainLog

end NN.Examples.Quickstart.Widgets
