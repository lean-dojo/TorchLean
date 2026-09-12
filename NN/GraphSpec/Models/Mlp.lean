/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Primitives
public import NN.GraphSpec.Chain.ToDAG.Model
public import NN.Runtime.Autograd.Model.Layers.Seq

/-!
# GraphSpec MLP Example

This file contains the smallest GraphSpec architecture example:

`Linear(input, hidden) → ReLU → Linear(hidden, output)`.

This does not duplicate TorchLean's executable MLP helper. Application code uses
`TorchLean.nn.mlp`; this file keeps only the proof-oriented graph description.
The point here is narrower and proof-oriented:

- show the sequential `Chain` DSL in its simplest useful form;
- make the parameter ABI visible in the type;
- provide a stable target for GraphSpec equivalence and deterministic-init proofs.

Because this is a pure sequential chain, it is authored with `Chain` and `>>>`. The companion
`mlpDAGModelZeroInit` lowers the same chain to the general DAG model representation so DAG-only
tooling can consume it.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace Models

open Spec TorchLean
open TorchLean.Tensor

/--
2-layer MLP: `Linear(input, hidden) → ReLU → Linear(hidden, output)`.

Notice how the parameter interface is explicit in the type:

- the first linear layer contributes tensors `W₁ : Tensor α [hiddenWidth, inputWidth]` and
  `b₁ : Tensor α [hiddenWidth]`,
- the second linear layer contributes tensors `W₂ : Tensor α [outputWidth, hiddenWidth]` and
  `b₂ : Tensor α [outputWidth]`,
- and `ReLU` contributes no parameters.

So the overall parameter list is exactly:
`[[hiddenWidth, inputWidth], [hiddenWidth], [outputWidth, hiddenWidth], [outputWidth]]`.
-/
def mlp (inputWidth hiddenWidth outputWidth : Nat) :
    Chain
      [[hiddenWidth, inputWidth], [hiddenWidth], [outputWidth, hiddenWidth], [outputWidth]]
      [inputWidth] [outputWidth] :=
  Chain.linear inputWidth hiddenWidth >>>
  Chain.relu [hiddenWidth] >>>
  Chain.linear hiddenWidth outputWidth

/--
The same 2-layer MLP, but exposed as a DAG `Model` via the structural lowering
`LowerToDAG.Chain.toDAGModelZeroInit`.

This is mainly for GraphSpec example ergonomics: downstream tooling that expects DAG terms can
consume this even though it was authored using the sequential `>>>` syntax.

Initialization: all-zero parameters (see `LowerToDAG.Chain.toDAGModelZeroInit`).
-/
def mlpDAGModelZeroInit (inputWidth hiddenWidth outputWidth : Nat) :
    DAG.Model
      [[hiddenWidth, inputWidth], [hiddenWidth], [outputWidth, hiddenWidth], [outputWidth]]
      [[inputWidth]]
      [outputWidth] :=
  LowerToDAG.Chain.toDAGModelZeroInit
    (mlp (inputWidth := inputWidth) (hiddenWidth := hiddenWidth) (outputWidth := outputWidth))

/-!
## Example Usage

You can build a simple classifier head by appending a softmax:

```lean
def g (inputWidth hiddenWidth outputWidth : Nat) :
    Chain
      [[hiddenWidth, inputWidth], [hiddenWidth], [outputWidth, hiddenWidth], [outputWidth]]
      [inputWidth] [outputWidth] :=
  Models.mlp inputWidth hiddenWidth outputWidth >>> Chain.softmax [outputWidth] 0
```

Then:

- `Interp.spec (g …)` maps a parameter pack and input tensor to an output tensor;
- `Chain.toProgram (g …)` is an executable TorchLean `Program` with arguments
  `params ++ [input]`.
-/

end Models
end GraphSpec
end NN
