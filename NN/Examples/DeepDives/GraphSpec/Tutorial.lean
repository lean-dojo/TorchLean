/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.GraphSpec.Models
public import NN.GraphSpec.ToSequential

/-!
# GraphSpec tutorial

GraphSpec is TorchLean's architecture-facing graph language. It is useful when you want a model to
exist first as a typed graph that can later be interpreted in several ways:

- as a pure specification (`Interp.spec`) for theorem statements,
- as an executable TorchLean program (`Chain.toProgram`) for direct execution,
- as a sequential `nn.Sequential` view when the graph is just a layer stack, and
- as a DAG model when the architecture has sharing, skip connections, or multi-input nodes.

That is why this folder is not a duplicate of `NN.Spec.Models` or `NN.Examples.Models`.

- `NN.Spec.Models` describes mathematical/reference model semantics.
- `NN.GraphSpec.Models` describes architecture graphs and their parameter ABI.
- `NN.Examples.Models` contains runnable training scripts.

This tutorial does two things:

1. It runs the smallest complete sequential-model path: `GraphSpec.Models.mlp → ToSequential.toSeq →
  Trainer.new → trainer.train`.
2. It typechecks the broader GraphSpec model ladder: MLP, CNN, and a residual linear block.

Only the MLP is trained here because it is the compact check path for `Seq` lowering. The CNN and
residual block remain architecture terms that graph passes, exporters, and proofs can consume.

Run:

```bash
scripts/lake.sh exe torchlean graphspec --execution eager
scripts/lake.sh exe torchlean graphspec --execution typed-graph
```

You can also pass the standard TorchLean runtime flags such as `--arithmetic ieee`,
`--execution eager`, or `--execution typed-graph`.
-/

@[expose] public section


namespace NN.Examples.DeepDives.GraphSpec.Tutorial

open TorchLean
open TorchLean.Tensor

/-- Command-line help for the GraphSpec tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean GraphSpec tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean graphspec [options]"
    , ""
    , "Options:"
    , "  --arithmetic native|ieee"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|gpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

/-! ## Small architecture terms that should typecheck -/

/--
The smallest sequential GraphSpec model.

Parameter ABI:
`[[3, 2], [3], [1, 3], [1]]`.
-/
def mlp :
    NN.GraphSpec.Chain
      [ [3, 2], [3]
      , [1, 3], [1] ]
      [2] [1] :=
  NN.GraphSpec.Models.mlp (inputWidth := 2) (hiddenWidth := 3) (outputWidth := 1)

/--
A small CNN graph, included here so the tutorial is visibly not “just MLP”.

This is still a sequential graph: convolution, ReLU, pooling, convolution, ReLU, pooling, flatten,
linear head. The type records the parameter shapes and intermediate spatial
arithmetic are checked before the model can be used.
-/
def cnn :=
  let features :=
    NN.GraphSpec.Chain.conv 1 2 [3, 3] [1, 1] [1, 1] [8, 8] >>>
    NN.GraphSpec.Chain.relu [2, 8, 8] >>>
    NN.GraphSpec.Chain.maxPool 2 [2, 2] [2, 2] [0, 0] [8, 8]
      (hKernel := by intro i; fin_cases i <;> decide)
      (hStride := by intro i; fin_cases i <;> decide) >>>
    NN.GraphSpec.Chain.conv 2 3 [3, 3] [1, 1] [1, 1] [4, 4] >>>
    NN.GraphSpec.Chain.relu [3, 4, 4] >>>
    NN.GraphSpec.Chain.maxPool 3 [2, 2] [2, 2] [0, 0] [4, 4]
      (hKernel := by intro i; fin_cases i <;> decide)
      (hStride := by intro i; fin_cases i <;> decide)
  NN.GraphSpec.Models.classifier features 4

/--
The minimal DAG-native skip-connection example:

$$
x\mapsto\operatorname{ReLU}(Wx+b+x).
$$

This is not a plain chain: representing it that way would either duplicate the input path or hide
sharing in a special layer. That is the pedagogical reason `GraphSpec.DAG` exists.
-/
def residual :=
  NN.GraphSpec.Models.residualLinear (d := 4)

/-- Tiny one-sample dataset for the lowered GraphSpec MLP training path. -/
def dataset : Trainer.Dataset [2] [1] :=
  let inputs : Tensor Float [1, 2] := [[0.5, 0.8]]
  let targets : Tensor Float [1, 1] := [[1.0]]
  Data.fromTensors inputs targets

/-- Run the compact MLP lowering/training path. -/
def run (args : List String) : IO Unit := do
  -- Lower the same architecture term shown above; its parameter ABI stays visible in `mlp`.
  -- A sequential model lives in Type 1, so inspect it before entering IO.
  match NN.GraphSpec.ToSequential.toSeq (σ := [2]) (τ := [1]) mlp with
  | .error message => CLI.orThrow "GraphSpec.ToSequential.toSeq failed" (.error message)
  | .ok network =>
    let runConfig ← TorchLean.CLI.Trainer.parseCommandLine "graphspec"
      (CLI.dropDashDash args)
      { optimizer := optim.sgd { learningRate := 0.1 } }
    let trainer :=
      Trainer.new network <|
      Trainer.RunConfig.forObjective runConfig .mse
    trainer.printSummary
    let trained ← trainer.train dataset { steps := 3, logTitle := "GraphSpec tutorial" }
    IO.println "forward: GraphSpec MLP lowered to TorchLean and executed"
    trained.printSummary

/--
Entry point: print the operation catalogue, then lower a GraphSpec MLP and train it for a few steps.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  IO.println "== GraphSpec tutorial =="
  IO.println "GraphSpec architecture ladder:"
  IO.println "  1. MLP: sequential layer stack; lowers to nn.Sequential and trains below."
  IO.println "  2. CNN: sequential vision graph with checked conv/pool shape arithmetic."
  IO.println "  3. residualLinear: minimal DAG-native skip connection."
  IO.println ""
  run args

end NN.Examples.DeepDives.GraphSpec.Tutorial
