import Verso
import VersoManual
import VersoBlueprint
import TorchLean.Tensor.Context
import TorchLean.Tensor.Algebra
import TorchLean.Tensor.Euclidean
import TorchLean.NN.Models.Classifier
import TorchLeanExamples.DeepDives.Sequential.MLP
import TorchLean.NN.IR.CustomRule
import TorchLean.NN.IR.Tangent
import TorchLean.NN.IR.Program
import TorchLean.NN.IR.OpGraph
import TorchLean.NN.Session.Random
import TorchLean.Tensor.Op.Pointwise.Real

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Tensors and Graphs" =>

TorchLean starts with shape-indexed tensors and then describes computations at two ends of one
pipeline. A sequential model is typed layer syntax whose layers are programs over one operation
interface, and a graph of registered operators (`TorchLean.IR.Graph`) is used at runtime boundaries.
The typed graph of the training runtime lies between them. The registered operators `TorchLean.Op`
have at most one executable rule each (`TorchLean.Op.rule?`), and the graph applies them. Keeping
those representations distinct makes the claims attached to each one easier to read.

:::group "tensor_foundations"
Scalar operations, tensor shapes, and the algebra used by model specifications.
:::

:::definition "scalar_context" (parent := "tensor_foundations") (lean := "TorchLean.Context")
`Context α` collects the arithmetic, order, constants, natural-number casts, and transcendental
operations needed by scalar-polymorphic model code. It carries no laws.
:::

:::definition "lawful_context" (parent := "tensor_foundations") (lean := "TorchLean.LawfulContext")
`LawfulContext α` records, for a scalar type that also carries a Mathlib linearly ordered field
structure, that the {uses "scalar_context"}[`Context` dictionary] computes the same addition,
multiplication, subtraction, division, negation, order, maximum, minimum, absolute value, and
natural and rational casts as the field. It also identifies Boolean equality with equality and
requires a positive default epsilon. We prove the instance for `ℝ`. Transcendental constants and
the total power operation stay unconstrained.

This separation lets executable scalar types provide operations without claiming field laws that
rounded arithmetic can't satisfy. A theorem using `LawfulContext` can transfer the listed
operations to Mathlib's field structure, but it has to state any further hypotheses about
logarithms, exponentials, or powers itself, because the class doesn't supply those analytic facts.
:::

:::definition "shape_indexed_tensors" (parent := "tensor_foundations") (lean := "TorchLean.Tensor")
`TorchLean.Shape` is `List Nat` (dimensions outermost first), and it indexes `TorchLean.Tensor α s`,
so tensor dimensions are present in the type. Each value stores one row-major buffer with a proof
that its length equals the shape's element count. `Storage α` selects packed `FloatArray` storage
for `Float`, `ByteArray` for `UInt8`, and ordinary arrays for other scalars. The first two store
unboxed entries, but a general array may contain boxed values, so a flat tensor doesn't imply a
contiguous block of unboxed scalars for every element type. Proofs observe this same value through
coordinate lookup, while the storage invariant doesn't prove the correctness of native C or CUDA
implementations.
:::

:::definition "shape_well_formedness" (parent := "tensor_foundations") (lean := "TorchLean.Shape.Positive")
A shape is well formed when every one of its dimensions is positive.

An empty list of dimensions satisfies this condition (it describes one scalar coordinate), whereas
a shape with a zero axis has no elements and fails it. That's why positivity of total size is a
useful hypothesis for a mean or a normalization denominator.
:::

:::definition "broadcast_compatibility" (parent := "tensor_foundations") (lean := "TorchLean.Shape.CanBroadcastTo")
`CanBroadcastTo s₁ s₂` records when values with shape `s₁` can be expanded to shape `s₂`.
:::

:::theorem "well_formed_shape_has_elements" (parent := "tensor_foundations") (lean := "TorchLean.Shape.size_pos_of_positive")
A {uses "shape_well_formedness"}[well-formed shape] has positive total size.
:::

:::proof "well_formed_shape_has_elements"
We recurse over the {uses "shape_well_formedness"}[list of axes], using positivity of every
dimension.
:::

:::theorem "flatten_round_trip" (parent := "tensor_foundations") (lean := "TorchLean.Tensor.reshape_reshape_self")
Rebuilding a flattened {uses "shape_indexed_tensors"}[shape-indexed tensor] at its original shape
returns that tensor.
:::

:::proof "flatten_round_trip"
Flattening and rebuilding preserve the same certified row-major scalar sequence. We reduce the
proof to the inverse laws for the tensor representation's zero-copy reshape.
:::

:::definition "tensor_euclidean_structure" (parent := "tensor_foundations") (lean := "TorchLean.Tensor.toEuclidean")
`toEuclidean` is a linear equivalence between the real tensor representation at a shape and the
Euclidean space indexed by its coordinates. We induce the `NormedAddCommGroup` and
`InnerProductSpace ℝ` instances on real {uses "shape_indexed_tensors"}[tensors] along it, so a real
tensor is a finite-dimensional inner-product space without a second representation.

Coordinates give the connection: each tensor entry becomes the corresponding coordinate of the
Euclidean vector, and the inverse reconstructs those entries. So the induced norm is about the same
values that tensor operations use, and analytic statements can use Mathlib's normed-space API
without introducing a separate tensor conversion into every theorem.
:::

:::theorem "tensor_inner_product_formula" (parent := "tensor_foundations") (lean := "TorchLean.Tensor.inner_eq_sum")
Under the {uses "tensor_euclidean_structure"}[induced structure], the inner product of two real
tensors is the sum of their coordinatewise products, and the tensor library's norm lemmas follow
from this formula.
:::

:::proof "tensor_inner_product_formula"
We transport the inner product along {uses "tensor_euclidean_structure"}[`toEuclidean`], where it's
the Euclidean sum.
:::

:::theorem "tensor_linear_adjointness" (parent := "tensor_foundations") (lean := "TorchLean.Tensor.dot_contract_matVec")
Matrix-vector multiplication satisfies the dot-product adjoint identity used by the linear-layer
gradient rule for {uses "shape_indexed_tensors"}[typed tensors] over a commutative semiring.
:::

:::proof "tensor_linear_adjointness"
We expand {uses "shape_indexed_tensors"}[the typed tensor operations] into finite sums and
rearrange those sums.

For an input perturbation `dx` and output cotangent `dLdy`, this moves the linear map from one side
of the dot product to the other (the resulting input cotangent has the input width). The semiring
hypotheses justify rearranging exact sums, but this identity alone doesn't bound rounded
reductions.
:::

:::definition "registered_operators" (parent := "tensor_foundations") (lean := "TorchLean.Op")
`TorchLean.Op ins outs` names one registered operator together with its attributes, indexed by the
lists of its input and output ports. A port is an element sort and a shape: a tensor of the scalar,
a frozen or detached tensor of the scalar, a tensor of indices, a tensor of Booleans, or a tensor
of integer words.
:::

:::definition "registered_operator_rule" (parent := "tensor_foundations") (lean := "TorchLean.Op.rule?")
`Op.rule?` gives each {uses "registered_operators"}[registered operator] its rule at a scalar type
with a {uses "scalar_context"}[scalar context]: the forward map on
{uses "shape_indexed_tensors"}[typed tensors], the dense pullback, and the node a recorder
prepares from the runtime flags of its inputs. An operator whose scalar lacks a capability it
needs has no rule there.
:::

:::group "graph_representations"
Typed architecture descriptions and the shared runtime graph.
:::

:::definition "sequential_syntax" (parent := "graph_representations") (lean := "TorchLean.nn.Sequential")
A sequential model records its input shape and output shape in its type, and the ordered shapes of
its parameters and buffers as `stateShapes`. Models are built from the identity and from a layer
followed by a model, and each layer carries one forward program over the operation interface.
:::

:::definition "sequential_pure_semantics" (parent := "graph_representations") (lean := "TorchLean.nn.Sequential.eval")
Pure interpretation runs the forward program of {uses "sequential_syntax"}[a sequential model]
under the pure interpreter, on {uses "shape_indexed_tensors"}[typed parameter and input tensors]
passed in as arguments, using {uses "scalar_context"}[scalar-polymorphic operations]. It returns
zeros where an operator has no rule.
:::

:::definition "sequential_runtime_translation" (parent := "graph_representations") (lean := "TorchLean.nn.Sequential.forward")
For {uses "sequential_syntax"}[a sequential model], the forward program is an execution-polymorphic
TorchLean program over {uses "scalar_context"}[the same scalar operations]. Eager execution runs
it, and `nn.Sequential.lowerToTypedGraph` records it as a typed graph.

Values shared inside a layer are bound once in its program and read by later operations, so model
syntax has no separate sharing construct. Typed graphs and the IR below keep their own references
between nodes.
:::

:::definition "sequential_mlp_model" (parent := "graph_representations") (lean := "TorchLean.Examples.Sequential.MLP.model")
This {uses "sequential_syntax"}[sequential] MLP composes two linear layers with an intervening
ReLU, and its parameter shapes fix the order and shapes of both weights and biases.
:::

:::theorem "sequential_mlp_spec_alignment" (parent := "graph_representations") (lean := "TorchLean.Examples.Sequential.MLP.eval_model")
Evaluating the {uses "sequential_mlp_model"}[sequential MLP] on the weights and biases of two
linear layers gives the same tensor as the hand-written two-layer MLP specification.

Both sides receive the same parameter tensors and input. This result identifies two descriptions of
that forward computation, but it doesn't compare separately trained models or establish that an
optimizer reaches particular weights. Initialization alignment below addresses the distinct
question of how the initial parameter pack is assembled.
:::

:::proof "sequential_mlp_spec_alignment"
Under {uses "sequential_pure_semantics"}[the pure interpretation] the linear operator's rule is the
linear specification and the ReLU operator's rule is the tensor ReLU, so both sides for
{uses "sequential_mlp_model"}[the MLP] reduce to the same two linear layers with an intervening
ReLU.
:::

:::theorem "sequential_mlp_initialization_alignment" (parent := "graph_representations") (lean := "TorchLean.Examples.Sequential.MLP.initState_model")
For the {uses "sequential_mlp_model"}[sequential MLP], the initial state is that of a linear layer
with weight seed `0` followed by that of a linear layer with weight seed `1`.
:::

:::proof "sequential_mlp_initialization_alignment"
A composed model's initial state appends its layers' initial states in order (ReLU has none). Both
biases are initialized to zero.
:::

:::theorem "sequential_classifier_alignment" (parent := "graph_representations") (lean := "TorchLean.nn.Sequential.eval_classifier")
For every {uses "sequential_syntax"}[sequential feature model], evaluating the classifier built on
it, with the feature model's state followed by the head's weights and bias, gives the linear
specification of the head applied to the flattened feature output.
:::

:::proof "sequential_classifier_alignment"
Evaluation of a composed model is evaluation of its parts in sequence
(`TorchLean.nn.Sequential.eval_comp`). Flattening is a reshape, and the head's forward program is
the linear specification under {uses "sequential_pure_semantics"}[the pure interpretation].
:::

:::definition "ir_shape_validation" (parent := "graph_representations") (lean := "TorchLean.Port.accepts")
A port is an element sort and a shape, and each registered operator is typed by its input and
output ports, so a graph of registered operators stores no output shape to infer. Shape validation
compares ports: an output can feed an input at the same port, and an output of the model's scalar
can feed an input of the same shape that holds it without a gradient (frozen or detached).
:::

:::definition "ir_structural_predicate" (parent := "graph_representations") (lean := "TorchLean.IR.Graph.WellFormed")
A graph of registered operators is well formed when every operator node has one reference per input
port (in port order), each naming an existing output of an earlier node at a port its input
{uses "ir_shape_validation"}[accepts], and passes its operator's check, and every graph output
names an existing output. Graph inputs and constants are well formed.
:::

:::definition "ir_structural_validation" (parent := "graph_representations") (lean := "TorchLean.IR.Graph.check")
This diagnostic checker tests the conditions of the {bpref "ir_structural_predicate"}[structural
predicate] node by node and returns the first one that fails as an error message.
`TorchLean.IR.Graph.check_ok_iff` proves that the checker succeeds exactly on the well-formed
graphs.
:::

:::definition "ir_denotation" (parent := "graph_representations") (lean := "TorchLean.IR.Graph.run")
IR execution evaluates the nodes in order into a table of each node's outputs at their ports
({uses "shape_indexed_tensors"}[shape-tagged tensor values]): a graph input or a constant from the
supplied payload is read at its port, and an operator is applied to the outputs its references
name using {uses "scalar_context"}[scalar-polymorphic operations]. The first error stops the run.
:::

:::theorem "ir_shape_soundness" (parent := "graph_representations") (lean := "TorchLean.IR.Graph.run_ports")
Whenever {uses "ir_denotation"}[a run] succeeds, entry `j` of its table is a list of values at
exactly the ports of node `j`: every value has its node's declared element sort and shape. No check
is needed first: an operator's output ports are part of its type, and reading a graph input or a
constant at its port fails unless the value has that port's shape.

This theorem is about the runs that succeed. It doesn't say that a run succeeds for every payload
or input.
:::

:::proof "ir_shape_soundness"
Each node's evaluation returns values at the ports the node declares
(`TorchLean.IR.Node.eval_ports`), and an induction over the nodes follows the run as it appends
them.
:::

:::theorem "ir_denotation_shape" (parent := "graph_representations") (lean := "TorchLean.IR.Graph.run_size")
Whenever {uses "ir_denotation"}[a run] succeeds, its table has one entry per node.
:::

:::definition "operator_graph" (parent := "graph_representations") (lean := "TorchLean.IR.Graph")
A node of an operator graph is a graph input at a port, a constant of the payload at a port, or a
{uses "registered_operators"}[registered operator] with one reference per input port. A reference
names a node and one of its outputs, so an operator with several outputs is one node, and index,
Boolean and integer values flow between nodes at their ports.
:::
