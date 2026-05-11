/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Core

/-!
# GraphSpec MLP Example

This file contains the smallest GraphSpec architecture example:

`Linear(in,hid) → ReLU → Linear(hid,out)`.

The goal is not to duplicate TorchLean's executable MLP helper. That constructor lives under
`NN.GraphSpec.Models.TorchLean.Mlp` and is re-exported through `NN.Entrypoint.TorchLeanModels`.
Here the point is narrower and proof-oriented:

- show the sequential `Graph` DSL in its simplest useful form;
- make the parameter ABI visible in the type;
- provide a stable target for GraphSpec equivalence and deterministic-init proofs.

Because this is a pure sequential chain, it is authored with `Graph` and `>>>`. The companion
`mlpDAGModelZeroInit` lowers the same chain to the general DAG model representation so DAG-only
tooling can consume it.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace Models

open Spec
open NN.Tensor

/--
2-layer MLP: `Linear(in,hid) → ReLU → Linear(hid,out)`.

Notice how the parameter interface is explicit in the type:

- the first `Linear(in,hid)` contributes `W₁ : Mat hid in` and `b₁ : Vec hid`,
- the second `Linear(hid,out)` contributes `W₂ : Mat out hid` and `b₂ : Vec out`,
- and `ReLU` contributes no parameters.

So the overall parameter list is exactly:
`[Mat hid in, Vec hid, Mat out hid, Vec out]`.
-/
def mlp (inDim hidDim outDim : Nat) :
    Graph
      [ Shape.Mat hidDim inDim, Shape.Vec hidDim
      , Shape.Mat outDim hidDim, Shape.Vec outDim ]
      (Shape.Vec inDim) (Shape.Vec outDim) :=
  Graph.linear inDim hidDim >>>
  Graph.relu (Shape.Vec hidDim) >>>
  Graph.linear hidDim outDim

/--
The same 2-layer MLP, but exposed as a DAG `Model` via the structural lowering
`LowerToDAG.Graph.toDAGModelZeroInit`.

This is mainly for GraphSpec example ergonomics: downstream tooling that expects DAG terms can
consume this even though it was authored using the sequential `>>>` syntax.

Initialization: all-zero parameters (see `LowerToDAG.Graph.toDAGModelZeroInit`).
-/
def mlpDAGModelZeroInit (inDim hidDim outDim : Nat) :
    DAG.Model
      [ Shape.Mat hidDim inDim, Shape.Vec hidDim
      , Shape.Mat outDim hidDim, Shape.Vec outDim ]
      [Shape.Vec inDim]
      (Shape.Vec outDim) :=
  LowerToDAG.Graph.toDAGModelZeroInit (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim))

/-!
## Example Usage

You can build a simple classifier head by appending a softmax:

```lean
def g (inDim hidDim outDim : Nat) :
    Graph
      [ Shape.Mat hidDim inDim, Shape.Vec hidDim
      , Shape.Mat outDim hidDim, Shape.Vec outDim ]
      (Shape.Vec inDim) (Shape.Vec outDim) :=
  Models.mlp inDim hidDim outDim >>> Graph.softmax (Shape.Vec outDim)
```

Then:

- `Interp.spec (g …)` is a pure function `Params → Tensor → Tensor`;
- `Compile.torchProgram (g …)` is an executable TorchLean `Program` with arguments
  `params ++ [input]`.
-/

end Models
end GraphSpec
end NN
