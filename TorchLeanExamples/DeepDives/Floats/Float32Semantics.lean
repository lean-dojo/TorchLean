/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Batteries.Data.Float.Basic
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Widget.Float32
public import TorchLean.NN.Autograd.Model
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting
public import TorchLean.Tensor.Reduce.Fold
/-!
# Float32 Semantics

The same compact MLP runs under two executable binary32 semantics:

- `Float32`: Lean's native binary32 type and runtime implementation.
- `FloatLib.Floats.Binary32`: FloatLib's configured executable binary32 model.

The example runs one forward pass and one reverse-mode VJP (seeded with `1.0`) under each. Both
results are converted to host `Float` for display, and `max_abs_diff` reports the largest difference
between them.

PyTorch comparison:
```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)).to(torch.float32)

# PyTorch runs the model through its selected float32 kernels.
```

The TorchLean example adds an independent second execution. Lean's native `Float32` supplies the
ordinary runtime path, while `Binary32` computes from the binary32 fields and
rounding rules written out in Lean. The model and the autograd program are shared because their
element type is generic.

Run:
  `scripts/lake.sh exe torchlean float32_semantics`

In the editor, put the cursor on the `#float32_*` commands below to see their widgets. The widgets
only display values, and the tutorial itself is ordinary `def` and `IO` code.
-/

@[expose] public section

open FloatLib.Floats (Binary32)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open TorchLean.Activation.Kind

namespace TorchLean.Examples.Float32Semantics

/-
The tutorial stays on the public TorchLean API:

- models: `nn.Sequential` built from `nn.*`
- execution: `nn.Sequential.lowerToTypedGraph model` and `nn.TypedGraphModel.forward`
- autodiff: `autograd.model.vjp`

The file uses only these public entry points, never the autograd tape or session directly.
-/

/-! ## Float32 widget probes -/

/--
The host `Float` literal `0.1`, the standard decimal with no exact binary representation. The
`#float32_round_view` widget shows the binary32 value that `Binary32` gets by
rounding it.

PyTorch analogue:

```python
torch.tensor(0.1, dtype=torch.float32)
```
-/
def decimalTenth : Float := 0.1

#float32_round_view decimalTenth
#float32_view (1 : Binary32)
#float32_view (Binary.canonicalNaN : Binary32)
#float32_compare_view (1 : Binary32), (Binary.canonicalNaN : Binary32)

/-- The network run with native `Float32` and with FloatLib's `Binary32`. -/
abbrev model : nn.Sequential [2] [1] :=
  -- Linear(2 -> 3) -> ReLU -> Linear(3 -> 1), built with the public `nn` layer API.
  -- `run` supplies every parameter tensor, so the init seed `0` has no effect on the results.
  nn.build 0 <| nn.mlp 2 1 { hiddenWidths := [3], activation := relu }

/-- The state layout of `model`: the weight and bias shapes of its two affine layers, in order. -/
abbrev layout : List Shape :=
  [[3, 2], [3], [1, 3], [1]]

/-- The model builder produces exactly `layout`. -/
theorem model_stateShapes :
    nn.Sequential.stateShapes model = layout := by
  rfl

/-- The output of one forward pass, with the parameter and input gradients of the reverse pass. -/
structure Result (α : Type) [Storage α] where
  /-- Output of the forward pass. -/
  output : Tensor α [1]
  /-- Gradient of the hidden layer's weight. -/
  hiddenWeightGrad : Tensor α [3, 2]
  /-- Gradient of the hidden layer's bias. -/
  hiddenBiasGrad : Tensor α [3]
  /-- Gradient of the output layer's weight. -/
  outputWeightGrad : Tensor α [1, 3]
  /-- Gradient of the output layer's bias. -/
  outputBiasGrad : Tensor α [1]
  /-- Gradient of the input. -/
  inputGrad : Tensor α [2]

namespace Result

/-- Convert every tensor in a run result while preserving its statically known shape. -/
def map {α β : Type} [Storage α] [Storage β]
    (result : Result α)
    (f : ∀ {shape}, Tensor α shape → Tensor β shape) :
    Result β where
  output := f result.output
  hiddenWeightGrad := f result.hiddenWeightGrad
  hiddenBiasGrad := f result.hiddenBiasGrad
  outputWeightGrad := f result.outputWeightGrad
  outputBiasGrad := f result.outputBiasGrad
  inputGrad := f result.inputGrad

end Result

/--
Run one forward and one backward pass over the fixed parameters at scalar type `α`.

The scalar parameter `α` lets native `Float32` and FloatLib's `Binary32` use the same code, source
literals, and graph. Both have binary32 precision, so the comparison exercises their scalar
conversions and arithmetic implementations.
-/
def run {α : Type}
    [Storage α] [Context α] [ToString α]
    [FromFloat α]
    (tag : String) : IO (Result α) := do
  -- Public data boundaries use `Float` literals and convert them into the selected scalar `α`.
  let cast : Float → α := ofFloat

  /-
  ### 1. Explicit parameters

  PyTorch analogue:

  ```python
  with torch.no_grad():
      model[0].weight.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
      model[0].bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
      model[2].weight.copy_(torch.tensor([[0.7, 0.8, 0.9]]))
      model[2].bias.copy_(torch.tensor([0.4]))
  ```

  `autograd.model.State model α` is shape-indexed: the model fixes the shape of every entry, so a
  parameter tensor of the wrong shape is rejected during elaboration.
  -/
  let hiddenWeight : Tensor α [3, 2] :=
    Tensor.map cast ([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] : Tensor Float [3, 2])
  let hiddenBias : Tensor α [3] :=
    Tensor.map cast ([0.1, 0.2, 0.3] : Tensor Float [3])
  let outputWeight : Tensor α [1, 3] :=
    Tensor.map cast ([[0.7, 0.8, 0.9]] : Tensor Float [1, 3])
  let outputBias : Tensor α [1] :=
    Tensor.map cast ([0.4] : Tensor Float [1])
  let paramsByLayer : TensorPack α layout :=
    Pack.nil
      |>.snoc hiddenWeight
      |>.snoc hiddenBias
      |>.snoc outputWeight
      |>.snoc outputBias
  let params : autograd.model.State model α :=
    paramsByLayer.cast model_stateShapes.symm

  let x : Tensor α [2] :=
    Tensor.map cast ([0.5, 0.8] : Tensor Float [2])

  /-
  ### 2. Forward pass

  PyTorch analogue:

  ```python
  y = model(x)
  ```

  `nn.Sequential.lowerToTypedGraph` records the model's forward program as a reusable typed SSA
  graph. TorchLean models are scalar-generic, so this step chooses the scalar `α` used by that
  executable graph.
  -/
  let graph ← nn.Sequential.lowerToTypedGraph model (α := α)
  -- `y = model(params, x)` for one example, with no batch dimension.
  let y := nn.TypedGraphModel.forward graph params x

  /-
  ### 3. Reverse-mode VJP

  PyTorch analogue:

  ```python
  y.sum().backward()
  dState = [p.grad for p in model.parameters()]
  inputGrad = x.grad
  ```

  A VJP needs an upstream gradient for the output. The output has shape `[1]`, so the seed `[1.0]`
  gives the gradient of `sum(y)`, as `y.sum().backward()` does.
  -/
  let outputGradient : Tensor α [1] :=
    Tensor.full [1] (cast 1.0)

  -- State and input gradients come from the same reverse pass.
  let (dState, inputGrad) ← autograd.model.vjp (α := α) model params x outputGradient

  /- TorchLean keeps each parameter-gradient shape in the type of the gradient pack. -/
  let gradientsByLayer := dState.cast model_stateShapes
  let hiddenWeightGrad := gradientsByLayer.get 0
  let hiddenBiasGrad := gradientsByLayer.get 1
  let outputWeightGrad := gradientsByLayer.get 2
  let outputBiasGrad := gradientsByLayer.get 3

  IO.println s!"== {tag} =="
  IO.println s!"y   = {Tensor.pretty y}"
  IO.println s!"hiddenWeightGrad = {Tensor.pretty hiddenWeightGrad}"
  IO.println s!"hiddenBiasGrad = {Tensor.pretty hiddenBiasGrad}"
  IO.println s!"outputWeightGrad = {Tensor.pretty outputWeightGrad}"
  IO.println s!"outputBiasGrad = {Tensor.pretty outputBiasGrad}"
  IO.println s!"inputGrad  = {Tensor.pretty inputGrad}"

  pure { output := y, hiddenWeightGrad, hiddenBiasGrad, outputWeightGrad, outputBiasGrad,
         inputGrad }

/--
The largest absolute entrywise difference between two runs, over the output, the four parameter
gradients and the input gradient. Both runs are converted to host `Float` before the comparison.
-/
def maxAbsDiff (a b : Result Float) : Float :=
  let differences : Tensor Float [6] :=
    [ Tensor.maxAbsDiff a.output b.output
    , Tensor.maxAbsDiff a.hiddenWeightGrad b.hiddenWeightGrad
    , Tensor.maxAbsDiff a.hiddenBiasGrad b.hiddenBiasGrad
    , Tensor.maxAbsDiff a.outputWeightGrad b.outputWeightGrad
    , Tensor.maxAbsDiff a.outputBiasGrad b.outputBiasGrad
    , Tensor.maxAbsDiff a.inputGrad b.inputGrad ]
  Tensor.maxAbs differences

/-- Command-line help for the Float32 semantics tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean Float32 semantics tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean float32_semantics"
    , ""
    , "This demo has no tutorial-specific flags."
    ]

/-- Run both semantics and print the largest difference between their results. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "float32_semantics" args
  IO.println "== Float32 semantics tutorial =="
  IO.println
    "Note: rounded-real binary32 is proof-only and is selected directly in theorem statements."
  IO.println "[TorchLean] FP32: finite rounded-real proof model"
  IO.println "[TorchLean] configured binary32: bit-level binary32 reference"

  let rNative ← run (α := Float32) "Float32 (native runtime)"
  let r32 ← run (α := Binary32) "ExecFloat.Binary 8 23"

  let rNativeF : Result Float :=
    rNative.map fun {_shape} t => Tensor.map Float32.toFloat t
  let r32F : Result Float :=
    r32.map fun {_shape} t =>
      Tensor.map (α := Binary32)
        (fun x => Binary.toFloat x) t

  let diff := maxAbsDiff rNativeF r32F
  IO.println s!"max_abs_diff(Float32 vs configured binary32) = {diff.toStringFull}"

end TorchLean.Examples.Float32Semantics
