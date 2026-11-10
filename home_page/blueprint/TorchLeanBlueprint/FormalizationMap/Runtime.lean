import Verso
import VersoManual
import VersoBlueprint
import TorchLean.Verify.LibTorch.TreeSum
import TorchLean.NN.Models.FNO.Block
import TorchLean.NN.Functional.SelectiveScan
import TorchLean.NN.Functional.Spectral
import TorchLean.NN.Module.Evaluator
import TorchLean.NN.Module.Instantiation
import TorchLean.NN.Layer.Attention
import TorchLean.NN.Layer.ConvPool
import TorchLean.NN.Layer.Mamba
import TorchLean.NN.Layer.Norm
import TorchLean.NN.Models.Recurrent.Stack
import TorchLean.NN.Sequential
import TorchLean.NN.Models.VQVAE.Autograd
import TorchLean.NN.Tape.Lowering.BackwardGraph
import TorchLean.NN.Tape.Apply.Correctness
import TorchLean.NN.Tape.Lowering.Compiled
import TorchLean.NN.Ops.Tensor
import TorchLean.NN.Session.Random

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Autograd and Execution" =>

TorchLean's runtime has a dynamic tape for eager execution and a typed SSA builder for typed graph
execution. Both sit below the layer API.

:::group "autograd_execution"
Programs, tapes, and typed graphs.
:::

:::definition "runtime_ops_program" (parent := "autograd_execution") (lean := "TorchLean.Program")
The operation interface describes programs over
{uses "shape_indexed_tensors"}[shape-indexed tensors] and
{uses "scalar_context"}[scalar operations] without choosing the eager tape or the typed graph
recorder at the call site.
:::

:::definition "runtime_autograd_tape" (parent := "autograd_execution") (lean := "TorchLean.Tape")
The CPU tape is a dynamic computation DAG. It stores shape-erased values and accumulates
reverse-mode gradients by node index.

Shape erasure lets one array hold a vector input, a matrix intermediate, and a scalar loss.
Typed graph lowering must preserve which index names each value. During reverse accumulation,
multiple uses of an intermediate contribute to the same stored cotangent, so index alignment
matters for both reading forward values and adding backward contributions.
:::

:::definition "runtime_tape_apply" (parent := "autograd_execution") (lean := "TorchLean.Tape.apply")
`Tape.apply` records one registered operator of `TorchLean.Op` on the
{uses "runtime_autograd_tape"}[CPU tape] through the operator's rule in `Op.rule?`. It checks the
operator's attributes and the value domain, prepares the rule with the parents' runtime gradient
flags, appends one node for the value output and returns the index and Boolean outputs. On the CUDA
tape, an operator is recorded through its entry in `LibTorch.Kernel.ofOp` (`LibTorch.Tape.apply`).
:::

:::theorem "tape_apply_backward" (parent := "autograd_execution") (lean := "TorchLean.Tape.apply_backward_eq_filter")
After a successful {uses "runtime_tape_apply"}[`Tape.apply`], the recorded node's backward is the
backward of the node prepared with every input flagged, keeping the contributions to parents that
need a gradient. This statement covers every registered operator at once, including the alias rule
used when two value positions name one node, and `apply_backwardDense` and its companions conclude
that the reverse sweeps are those of the node prepared with every input flagged.
:::

:::proof "tape_apply_backward"
A successful `apply` records the prepared node. Preparing with the runtime flags gives the fully
flagged pullback masked by those flags (`prepared_pullback`, from each rule's `prepare_eq`), and
emitting a masked pullback keeps exactly the contributions to parents that need a gradient
(`emit_mask`).
:::

:::definition "runtime_typed_graph_autograd" (parent := "autograd_execution") (lean := "TorchLean.Tape.GraphData.toTape")
`Tape.GraphData.toTape` takes executable typed graph data, its input context and auxiliary
data, then returns a
{uses "runtime_autograd_tape"}[runtime tape] together with the typed list of inputs and intermediate
values.
:::

:::theorem "typed_graph_backward_agreement" (parent := "autograd_execution") (lean := "TorchLean.Tape.GraphData.backwardDenseFrom_toTape")
For executable typed graph data with `Storage` and `Add`, dense reverse accumulation on the
{uses "runtime_autograd_tape"}[lowered tape] equals proof-level graph backpropagation after both
contexts are converted to the tape's value array. Any seed context is allowed: the seed
`TensorPack.single output seed` differentiates one typed output reference, which may be a graph
input or an intermediate node as well as the last node.

This is agreement with the graph's stored executable backward rules. It shows that lowering and
accumulation preserve those rules, but it doesn't establish their derivative formulas (local
derivative laws are a separate obligation). This operational equality retains scalar addition order
and needs no commutative-semiring laws, so it doesn't assume exact real arithmetic for rounded
carriers.
:::

:::proof "typed_graph_backward_agreement"
Induction over the typed graph keeps the forward context and tape indices aligned while the reverse
loop accumulates each node's contribution.
:::

:::definition "runtime_typed_graph_compiled" (parent := "autograd_execution") (lean := "TorchLean.TypedGraph.compileChecked")
Checked execution validates and evaluates each node while retaining an indexed primal context and
local reverse programs in `Compiled`. Checked and pure VJP calls use this saved execution through
proved compiler simplifications, while the reverse-mode Jacobian transforms call it directly.
:::

:::theorem "typed_graph_compiled_forward_agreement" (parent := "autograd_execution") (lean := "TorchLean.TypedGraph.compileChecked_eq")
Taking the tape and forward context from a {uses "runtime_typed_graph_compiled"}[compiled execution]
gives the complete result of `lowerToTapeChecked`, including the same failures, runtime tape, and
full primal context.
:::

:::proof "typed_graph_compiled_forward_agreement"
Each prepared node preserves its forward, validation, and dense VJP programs. Induction over the
checked lowering keeps these programs and the indexed context aligned with the original pack
interface.
:::

:::theorem "typed_graph_compiled_backward_agreement" (parent := "autograd_execution") (lean := "TorchLean.TypedGraph.Compiled.backwardDenseFrom_eq_tape")
After successful checked compilation, `Compiled.backwardDenseFrom` agrees with dense backward on
the original lowered tape for every seed pack and every gradient entry. The hypotheses require
`Storage` and `Add`, preserving the chosen carrier's scalar addition order.
:::

:::proof "typed_graph_compiled_backward_agreement"
Saved backward agrees with graph backpropagation for the complete seed context. Composing this with
{uses "typed_graph_backward_agreement"}[the dense tape agreement] gives the result. Neither equality
supplies a real derivative law for unrestricted IEEE arithmetic or a GPU kernel theorem.
:::

:::definition "runtime_layer_model" (parent := "autograd_execution") (lean := "TorchLean.nn.Sequential")
Layer definitions carry parameter and buffer initialization, training or evaluation mode, and
shape-checked sequential composition. They build their computations through
{uses "runtime_ops_program"}[the runtime operation interface].
:::
