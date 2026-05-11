# GraphSpec

GraphSpec is the layer between friendly model-building syntax and the low-level IR. It is useful
when an architecture's parameter layout, sharing structure, and pure semantics should be explicit
before the model is lowered or executed.

A GraphSpec model has two interpretations:

1. a pure specification semantics in Lean, and
2. a compiled TorchLean program that can run.

Use the subsystem entrypoint:

```lean
import NN.Entrypoint.GraphSpec
```

`NN/IR` is the shared op-tagged graph IR used by runtime compilation and verification. GraphSpec is
an authoring layer that can feed the broader TorchLean pipeline; it is not a replacement for
`NN.IR.Graph`.

## Where GraphSpec Fits

| Layer | Best for |
| --- | --- |
| `nn.Sequential` | ordinary tutorials and training examples |
| `GraphSpec` | typed architecture authoring with explicit parameter layout and sharing |
| `NN.IR.Graph` | op-tagged graph artifacts for verification, widgets, and export |

## Main Files

| File | Role |
| --- | --- |
| `Core.lean` | sequential `Graph` syntax with `>>>` composition |
| `DAG/Core.lean` | A-normal/SSA-style DAG terms with sharing |
| `DAG.lean` | DAG primitive constructors |
| `Primitives.lean` | common primitive packs |
| `Primitives/Vision.lean` | convolution, pooling, flattening, and image helpers |
| `Primitives/Embedding.lean` | embedding primitive and theorem surface |
| `ToTorchLean.lean` | lowering from the supported sequential subset to `TorchLean.NN.Seq` |
| `Models/*` | GraphSpec-authored examples |

## Sequential Graphs

`Graph ps σ τ` represents a chain from input shape `σ` to output shape `τ`, with parameter shapes
`ps : List Shape` tracked at the type level.

```lean
import NN.Entrypoint.GraphSpec

open NN
open NN.GraphSpec

def g (inDim hidDim outDim : Nat) :=
  GraphSpec.Models.mlp inDim hidDim outDim

#check GraphSpec.Interp.spec (g 4 8 2)
#check GraphSpec.Compile.torchProgram (g 4 8 2)
#check GraphSpec.LowerToDAG.Graph.toDAGModelZeroInit (g 4 8 2)
```

Sequential graphs are the right surface for MLPs and simple feed-forward pipelines.

## DAG Models

`DAG.Term Γ τ` is the internal GraphSpec representation for explicit sharing and skip connections.
Its environment `Γ` contains parameters followed by data inputs, so the parameter interface stays
visible in the type.

DAG models provide:

| Object | Meaning |
| --- | --- |
| `Term.eval` | pure specification semantics |
| `Term.compile` | executable TorchLean compilation |
| `DAG.Model` | packaged parameters, inputs, and body term |

Use DAG terms when a model needs residual connections, multiple inputs, or explicit reuse of an
intermediate value.

## Adding A Primitive

A primitive must provide both a pure meaning and an executable TorchLean meaning.

For unary sequential layers, define a `Primitive ps σ τ`:

```lean
namespace NN
namespace GraphSpec
namespace Primitive

open Spec
open Tensor
open NN.Tensor

def myOp (s : Shape) : Primitive [] s s :=
  { name := "myOp"
    specFwd := fun {α} _ctx _params x => x
    torchProgram := fun {α} _ctx _deq =>
      fun {m} _ _ => fun x => pure x
    toLayerDefM? := none
    countsAsLayer := false }

end Primitive
end GraphSpec
end NN
```

If the same unary primitive is needed inside DAG syntax, embed it through:

```lean
LowerToDAG.Primitive.toDAGPrimOp
```

For true multi-input operations, define a `DAG.PrimOp ins τ` directly in `DAG.lean`.

## Proof Pattern

GraphSpec proofs usually compare the interpreter with an existing specification:

1. choose a small model, such as an MLP or residual block;
2. state equality between `Interp.spec` or `DAG.Term.eval` and the reference forward function;
3. unfold the GraphSpec syntax and simplify with a focused simp set;
4. use the model-specific tensor lemmas for the remaining arithmetic.

See `NN/GraphSpec/Models/MlpSpecEquivalence.lean` for the smallest version of this pattern.

## References

- ResNets / skip connections: He et al. (2016), "Deep Residual Learning for Image Recognition".
- SSA form: Cytron et al. (1991), "Efficiently Computing Static Single Assignment Form".
- Automatic differentiation: Baydin et al. (2018), "Automatic Differentiation in Machine Learning:
  a Survey".
- PyTorch architecture references: `torch.nn.Sequential` and `torch.fx`.
