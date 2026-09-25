import Verso
import VersoManual
import VersoBlueprint
import NN.Runtime.Autograd.Engine
import NN.Runtime.Autograd.TypedGraph
import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence
import NN.Runtime.Autograd.Torch
import NN.Runtime.Autograd.Torch.TypedGraphSession
import NN.Runtime.Autograd.Model
import NN.Proofs.Autograd.Runtime.Link.BackwardGraphData
import NN.Verification.Builtin.Proved.Correctness

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Autograd and Execution" =>

The runtime has a dynamic tape for eager execution and a typed SSA builder for typed graph
execution. Both sit below the layer API. A separate bridge executes `NN.IR.Graph` through a
forward-only graph; its correctness theorem names its single remaining side condition.

:::group "autograd_execution"
Programs, tapes, and typed graph execution.
:::

:::definition "runtime_ops_program" (parent := "autograd_execution") (lean := "Runtime.Autograd.Model.Program")
The operation interface describes programs over
{uses "shape_indexed_tensors"}[shape-indexed tensors] and
{uses "scalar_context"}[scalar operations] without choosing eager or typed graph execution at the
call site.
:::

:::definition "runtime_autograd_tape" (parent := "autograd_execution") (lean := "Runtime.Autograd.Tape")
The CPU tape is a dynamic computation DAG. It stores shape-erased values and accumulates
reverse-mode gradients by node index.

Shape erasure lets one array hold a vector input, a matrix intermediate, and a scalar loss.
Typed graph lowering must preserve which index names each value. During reverse accumulation,
multiple uses of an intermediate contribute to the same stored cotangent, so index alignment
matters for both reading forward values and adding backward contributions.
:::

:::definition "runtime_typed_graph_autograd" (parent := "autograd_execution") (lean := "Runtime.Autograd.TypedGraph.lowerToTape")
`lowerToTape` takes executable typed graph data and its input context, then returns a
{uses "runtime_autograd_tape"}[runtime tape] together with the typed list of inputs and intermediate
values.
:::

:::theorem "typed_graph_backward_agreement" (parent := "autograd_execution") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx")
For executable typed graph data with `Storage` and `Add`, dense reverse accumulation on the
{uses "runtime_autograd_tape"}[lowered tape] equals proof-level graph backpropagation after both
contexts are converted to the tape's value array.

This is agreement with the graph's stored executable backward rules. It shows that lowering and
accumulation preserve those rules; it does not establish their derivative formulas. Local derivative
laws are a separate obligation. This operational equality retains scalar addition order and needs
no commutative-semiring laws, so it does not assume exact real arithmetic for rounded carriers.
:::

:::proof "typed_graph_backward_agreement"
Induction over the typed graph keeps the forward context and tape indices aligned while the reverse
loop accumulates each node's contribution.
:::

:::definition "runtime_typed_graph_compiled" (parent := "autograd_execution") (lean := "Runtime.Autograd.TypedGraph.compileChecked")
Checked execution validates and evaluates each node while retaining an indexed primal context and
certified local reverse programs in `Compiled`. Checked and pure VJP calls use this saved execution
through proved compiler simplifications; sessions and the typed trainer use it directly.
:::

:::theorem "typed_graph_compiled_forward_agreement" (parent := "autograd_execution") (lean := "Runtime.Autograd.TypedGraph.compileChecked_asLegacy")
Mapping a {uses "runtime_typed_graph_compiled"}[compiled execution] through `Compiled.asLegacy`
gives the complete result of `lowerToTapeChecked`, including the same failures, runtime tape, and
full primal context.
:::

:::proof "typed_graph_compiled_forward_agreement"
Each prepared node preserves its forward, validation, and dense VJP programs. The checked lowering
induction keeps these programs and the indexed context aligned with the original pack interface.
:::

:::theorem "typed_graph_compiled_backward_agreement" (parent := "autograd_execution") (lean := "Runtime.Autograd.TypedGraph.compileChecked_backwardDenseFrom_eq_tape")
After successful checked compilation, `Compiled.backwardDenseFrom` agrees with dense backward on
the original lowered tape for every seed pack and every gradient entry. The hypotheses require
`Storage` and `Add`, preserving the chosen carrier's scalar addition order.
:::

:::proof "typed_graph_compiled_backward_agreement"
Saved backward agrees with graph backpropagation for the complete seed context. Composing this with
{uses "typed_graph_backward_agreement"}[the dense tape agreement] gives the result. Neither equality
supplies a real derivative law for unrestricted IEEE arithmetic or a GPU kernel theorem.
:::

:::theorem "typed_graph_output_backward_agreement" (parent := "autograd_execution") (lean := "Runtime.Autograd.TypedGraph.backwardDenseAllFrom_lowerToTape_eq_backpropAllCtx")
For any typed output reference, including an input or intermediate node, lowering to the runtime
tape and running reverse mode agrees with seeding that same output in executable graph
backpropagation.
:::

:::proof "typed_graph_output_backward_agreement"
The general graph-data lowering theorem is instantiated with a seed context containing the supplied
cotangent at exactly the selected output reference.

Selecting an intermediate asks for the sensitivity of that intermediate, rather than of the
last node in the graph. Selecting an input is meaningful too. The seed identifies both the value
being differentiated and the cotangent applied there, so both routes must seed the same reference.
:::

:::theorem "runtime_typed_graph_backprop_link" (parent := "autograd_execution") (lean := "Runtime.Autograd.Torch.Internal.TypedGraphSession.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx")
For the executable graph data in a typed-graph session snapshot, dense reverse accumulation on
{uses "runtime_autograd_tape"}[the lowered tape] agrees with graph backpropagation. Runtime leaf
names and `requiresGrad` masks are attached after this raw graph-data lowering theorem.
:::

:::proof "runtime_typed_graph_backprop_link"
The typed-graph session theorem specializes
{uses "typed_graph_backward_agreement"}[the graph-data backward theorem] to the session's graph,
inputs, and auxiliary index environment.
:::

:::definition "runtime_layer_model" (parent := "autograd_execution") (lean := "Runtime.Autograd.Model.Layers.Seq")
Layer definitions carry parameter and buffer initialization, training or evaluation mode, and
shape-checked sequential composition. They build their computations through
{uses "runtime_ops_program"}[the runtime operation interface].
:::

:::group "shared_ir_runtime"
The executable path from the shared graph IR.
:::

:::definition "shared_ir_execution" (parent := "shared_ir_runtime") (lean := "Runtime.Autograd.IRExec.lowerToForwardGraph")
After {uses "ir_structural_validation"}[structural validation], the supported shared IR operations
are lowered to a shape-indexed `ForwardGraph` for forward evaluation. Its `ForwardData` contains
only forward closures: there are no JVP or VJP fields to call. It is distinct from the reusable
autograd `Torch.TypedGraph`.

The correctness claim below consequently concerns forward values. It compares this lowered
reference execution with the IR evaluator using the same inputs and payload. It does not certify
native CPU or CUDA kernels, and this forward-only bridge provides no backward implementation to
which the tape-agreement theorem could be applied.
:::

:::theorem "shared_ir_execution_correctness" (parent := "shared_ir_runtime") (lean := "Runtime.Autograd.IRExec.denoteAll_eq_of_lowerToForwardGraph")
{uses "shared_ir_execution"}[Successful lowering] agrees with
{uses "ir_denotation"}[IR denotation] on every input, provided `NoRawLog` excludes all raw `log`
nodes. This restriction matters because the IR evaluator rejects nonpositive inputs to raw `log`,
while the lowered closure applies the total spec logarithm. A theorem that admitted raw `log`
would need per-node positivity hypotheses; this theorem instead excludes that operation.
Every other operation kind is covered, including `mseLoss`, `concat` along any axis, matmul with
any shared leading shape, and batched linear layers.
:::

:::proof "shared_ir_execution_correctness"
After unfolding {uses "shared_ir_execution"}[IR lowering], the proof peels off
{uses "ir_structural_validation"}[the structural check] and input node. The recursive lowering
invariant then matches each forward-graph value with {uses "ir_denotation"}[the corresponding
denotational value].
:::
