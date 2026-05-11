/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.GraphSpec.Models
public import NN.GraphSpec.ToTorchLean

/-!
# GraphSpec tutorial

GraphSpec is TorchLean's architecture-facing graph language. It is useful when you want a model to
exist first as a typed graph that can later be interpreted in several ways:

- as a pure specification (`Interp.spec`) for theorem statements,
- as an executable TorchLean program (`Compile.torchProgram`) for direct execution,
- as a sequential `nn.Sequential` view when the graph is just a layer stack, and
- as a DAG model when the architecture has sharing, skip connections, or multi-input nodes.

That is why this folder is not a duplicate of `NN.Spec.Models` or `NN.Examples.Models`.

- `NN.Spec.Models` describes mathematical/reference model semantics.
- `NN.GraphSpec.Models` describes architecture graphs and their parameter ABI.
- `NN.Examples.Models` contains runnable training scripts.

This tutorial does two things:

1. It runs the smallest complete lowering path: `GraphSpec.Models.mlp → ToTorchLean.toSeq →
   train.run`.
2. It typechecks and explains the broader GraphSpec model ladder: MLP, CNN, residual linear block,
   and ResNet18.

Only the MLP is trained here because it is the compact check path for `Seq` lowering. The larger
models are still GraphSpec models; they are deliberately kept as architecture terms so graph passes,
exporters, and proofs can consume them without turning this tutorial into a slow vision benchmark.

Run:

```bash
lake exe torchlean graphspec --backend eager
lake exe torchlean graphspec --backend compiled
```

You can also pass the standard scalar/runtime flags accepted by `train.run`, such as
`--dtype ieee32`.
-/

@[expose] public section


namespace NN.Examples.Advanced.GraphSpec.Tutorial

open Spec
open Tensor
open NN.Tensor
open NN.API

/-! ## Small architecture terms that should typecheck -/

/--
The smallest sequential GraphSpec model.

Parameter ABI:
`[Mat 3 2, Vec 3, Mat 1 3, Vec 1]`.
-/
def tutorialMlp :
    NN.GraphSpec.Graph
      [ Shape.Mat 3 2, Shape.Vec 3
      , Shape.Mat 1 3, Shape.Vec 1 ]
      (Shape.Vec 2) (Shape.Vec 1) :=
  NN.GraphSpec.Models.mlp (inDim := 2) (hidDim := 3) (outDim := 1)

/--
A small CNN graph, included here so the tutorial is visibly not “just MLP”.

This is still a sequential graph: convolution, ReLU, pooling, convolution, ReLU, pooling, flatten,
linear head. The ugly-looking type is the point: the parameter shapes and intermediate spatial
arithmetic are checked before the model can be used.
-/
def tutorialCnn :=
  NN.GraphSpec.Models.cnn2
    (inC := 1) (c1 := 2) (c2 := 3) (outDim := 4)
    (inH := 8) (inW := 8) (kH := 3) (kW := 3)
    (stride1 := 1) (padding1 := 1) (stride2 := 1) (padding2 := 1)
    (poolKH := 2) (poolKW := 2) (poolStride1 := 2) (poolStride2 := 2)
    (h_inC := by decide) (h_c1 := by decide) (_h_c2 := by decide)
    (h_kH := by decide) (h_kW := by decide)
    (h_poolKH := by decide) (h_poolKW := by decide)
    (h_poolStride1 := by decide) (h_poolStride2 := by decide)

/--
The minimal DAG-native skip-connection example:

`x ↦ relu((W x + b) + x)`.

This cannot be honestly represented as a plain chain without either duplicating the input path or
hiding sharing in a special layer. That is the pedagogical reason `GraphSpec.DAG` exists.
-/
def tutorialResidual :=
  NN.GraphSpec.Models.residualLinear (d := 4)

/--
The larger residual-family architecture available from the same model catalog.

We bind it here as a compile-time smoke check: users can inspect its type in the editor, while the
runtime tutorial below stays small enough to run instantly.
-/
def tutorialResNet18 :=
  NN.GraphSpec.Models.ResNet18.model

/-- Print the architecture ladder this tutorial is checking. -/
def printCatalog : IO Unit := do
  IO.println "GraphSpec architecture ladder:"
  IO.println "  1. MLP: sequential layer stack; lowers to nn.Sequential and trains below."
  IO.println "  2. CNN: sequential vision graph with checked conv/pool shape arithmetic."
  IO.println "  3. residualLinear: minimal DAG-native skip connection."
  IO.println "  4. ResNet18: larger DAG-style residual architecture."
  IO.println ""

/-- Run the compact MLP lowering/training path. -/
def runMlpTrainingPath (args : List String) : IO Unit := do
  let inDim : Nat := 2
  let hidDim : Nat := 3
  let outDim : Nat := 1

  let xShape : Spec.Shape := NN.Tensor.Shape.Vec inDim
  let yShape : Spec.Shape := NN.Tensor.Shape.Vec outDim

  -- GraphSpec is the source architecture. This exact graph also has pure semantics and an
  -- executable program view; here we ask for the additional `nn.Sequential` training view.
  let g := NN.GraphSpec.Models.mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim)

  match NN.GraphSpec.ToTorchLean.toSeq (σ := xShape) (τ := yShape) g with
  | .error msg =>
      throw <| IO.userError s!"GraphSpec.ToTorchLean.toSeq failed: {msg}"
  | .ok seqR =>
      let seq : nn.Sequential xShape yShape := by
        -- `API.nn.Sequential` is a facade alias for the same runtime `Seq` type.
        simpa using seqR
      let task := train.regression seq

      -- `train.run` is the canonical entrypoint: it parses `--dtype ...` and `--backend ...`,
      -- instantiates the runner, and passes leftover args to the callback.
      train.run task
        (CLI.dropDashDash args)
        (fun {α} _ _ _ _ runner rest => do
          Common.orThrow "GraphSpecTutorial" <| CLI.requireNoArgs rest
          let cast : Float → α := Runtime.ofFloat

          -- Build constants from Float literals, then cast into the chosen runtime scalar `α`.
          let xF : Spec.Tensor Float xShape := tensorF! id [inDim] [0.5, 0.8]
          let sampleYF : Spec.Tensor Float yShape := tensorF! id [outDim] [1.0]
          let x : Spec.Tensor α xShape := Spec.mapTensor cast xF

          -- PyTorch analogue:
          --
          -- ```python
          -- dataset = TensorDataset(X, Y)
          -- y = model(x)
          -- loss.backward()
          -- optimizer.step()
          -- ```
          --
          -- TorchLean keeps the shapes in the tensor type. `supervisedDim0F` says the leading
          -- dimension is the dataset axis, so this is a one-example regression dataset.
          let XFloat : Spec.Tensor Float (shape![1, inDim]) := Spec.Tensor.dim (fun _ => xF)
          let YFloat : Spec.Tensor Float (shape![1, outDim]) := Spec.Tensor.dim (fun _ => sampleYF)
          let dataset := Data.supervisedDim0F (α := α) XFloat YFloat

          let _yTorchLean : Spec.Tensor α yShape := ← train.predict runner x
          IO.println "forward: GraphSpec MLP lowered to TorchLean and executed"

          let loss0 ← train.meanLossDataset runner dataset
          IO.println s!"loss(before)={loss0}"
          let _report ← train.fitDataset runner (train.steps 3 (optim.sgd 0.1) (logEvery := 1))
            dataset
          train.evalMode runner
          let loss1 ← train.meanLossDataset runner dataset
          IO.println s!"loss(after)={loss1}")

def main (args : List String) : IO Unit := do
  IO.println "== GraphSpec tutorial =="
  printCatalog
  runMlpTrainingPath args

end NN.Examples.Advanced.GraphSpec.Tutorial
