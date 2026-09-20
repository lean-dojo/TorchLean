import Verso
import VersoManual
import VersoBlueprint
import NN.Spec.Core.Context
import NN.Spec.Core.Tensor
import NN.Spec.Core.TensorReductionShape
import NN.Proofs.Tensor.Algebra
import NN.Proofs.Tensor.Euclidean
import NN.GraphSpec
import NN.GraphSpec.Models.MlpDeterministicInit
import NN.GraphSpec.Models.MlpSpecEquivalence
import NN.IR

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Tensors and Graphs" =>

TorchLean starts with shape-indexed tensors and then offers three ways to describe a computation.
`GraphSpec.Chain` is typed sequential syntax, `GraphSpec.DAG` adds sharing, and `NN.IR.Graph` is
the ordinary op-tagged graph used at runtime boundaries. Keeping those representations distinct
makes the claims attached to each one easier to read.

:::group "tensor_foundations"
Scalar operations, tensor shapes, and the algebra used by model specifications.
:::

:::definition "scalar_context" (parent := "tensor_foundations") (lean := "Context")
`Context α` collects the arithmetic, order, constants, natural-number casts, and transcendental
operations needed by scalar-polymorphic model code. It carries no laws.
:::

:::definition "lawful_context" (parent := "tensor_foundations") (lean := "LawfulContext")
`LawfulContext α` records, for a scalar type that also carries a Mathlib linearly ordered field
structure, that the {uses "scalar_context"}[`Context` dictionary] computes the same addition,
multiplication, subtraction, division, negation, order, maximum, minimum, absolute value, and
small numeric constants as the field, and that its epsilon is positive. The instance for `ℝ` is
proved; transcendental constants and the total power operation stay unconstrained.

This separation lets executable scalar types provide operations without claiming field laws that
rounded arithmetic cannot satisfy. A theorem using `LawfulContext` can transfer the listed
operations to Mathlib's field structure. It still needs any further hypotheses about logarithms,
exponentials, or powers explicitly; the class does not supply those analytic facts.
:::

:::definition "shape_indexed_tensors" (parent := "tensor_foundations") (lean := "TorchLean.Tensor")
`Spec.Shape` is `List Nat`, the dimensions outermost first, and it indexes `TorchLean.Tensor α s`,
so tensor dimensions are present in the type. Each value stores one contiguous row-major buffer
with a proof that its length equals the shape's element count. `Storage α` selects packed
`FloatArray` storage for `Float`, `ByteArray` for `UInt8`, and ordinary arrays for other scalars.
Proofs observe this same value through coordinate lookup; the storage invariant does not prove
the correctness of native C or CUDA implementations.
:::

:::definition "shape_well_formedness" (parent := "tensor_foundations") (lean := "Spec.Shape.wellFormed")
A shape is well formed when every one of its dimensions is positive.

The empty list of dimensions satisfies this condition and describes one scalar coordinate.
A shape with a zero axis instead has no elements and fails the condition. This distinction is why
positivity of total size is a useful hypothesis for a mean or a normalization denominator.
:::

:::definition "broadcast_compatibility" (parent := "tensor_foundations") (lean := "Spec.Shape.CanBroadcastTo")
`CanBroadcastTo s₁ s₂` records when values with shape `s₁` can be expanded to shape `s₂`.
:::

:::theorem "well_formed_shape_has_elements" (parent := "tensor_foundations") (lean := "Spec.Shape.size_pos_of_well_formed")
A {uses "shape_well_formedness"}[well-formed shape] has positive total size.
:::

:::proof "well_formed_shape_has_elements"
The proof recurses over the {uses "shape_well_formedness"}[list of axes] and uses positivity of
every dimension.
:::

:::theorem "flatten_round_trip" (parent := "tensor_foundations") (lean := "TorchLean.Tensor.unflattenSpec_flattenSpec")
Rebuilding a flattened {uses "shape_indexed_tensors"}[shape-indexed tensor] at its original shape
returns that tensor.
:::

:::proof "flatten_round_trip"
Flattening and rebuilding preserve the same certified row-major scalar sequence; the proof reduces
to the inverse laws for the tensor representation's zero-copy reshape.
:::

:::definition "tensor_euclidean_structure" (parent := "tensor_foundations") (lean := "TorchLean.Tensor.toEuclidean")
`toEuclidean` is a linear equivalence between the real tensor representation at a shape and the
Euclidean space indexed by its coordinates. The `NormedAddCommGroup` and `InnerProductSpace ℝ`
instances on real {uses "shape_indexed_tensors"}[tensors] are induced along it, so a real tensor is
a finite-dimensional inner-product space without a second representation.

Coordinates give the connection: each tensor entry becomes the corresponding coordinate of the
Euclidean vector, and the inverse reconstructs those entries. The induced norm is therefore about
the same values used by tensor operations. Analytic statements can use Mathlib's normed-space API
without introducing a separate tensor conversion into every theorem.
:::

:::theorem "tensor_inner_product_formula" (parent := "tensor_foundations") (lean := "TorchLean.Tensor.inner_eq_sum")
Under the {uses "tensor_euclidean_structure"}[induced structure], the inner product of two real
tensors is the sum of their coordinatewise products; the norm lemmas of the tensor library follow
from this formula.
:::

:::proof "tensor_inner_product_formula"
The inner product is transported along {uses "tensor_euclidean_structure"}[`toEuclidean`], where it
is the Euclidean sum.
:::

:::theorem "tensor_linear_adjointness" (parent := "tensor_foundations") (lean := "Proofs.TensorAlgebra.dot_mat_linear_adjoint")
Matrix-vector multiplication satisfies the dot-product adjoint identity used by the linear-layer
gradient rule for {uses "shape_indexed_tensors"}[typed tensors] over a commutative semiring.
:::

:::proof "tensor_linear_adjointness"
The proof expands {uses "shape_indexed_tensors"}[the typed tensor operations] into finite sums and
rearranges those sums.

For an input perturbation `dx` and output cotangent `dLdy`, this moves the linear map from one side
of the dot product to the other. The resulting input cotangent has the input width. The semiring
hypotheses justify rearranging exact sums; this identity alone does not bound rounded reductions.
:::

:::group "graph_representations"
Typed architecture descriptions and the shared runtime graph.
:::

:::definition "graphspec_syntax" (parent := "graph_representations") (lean := "NN.GraphSpec.Chain")
A sequential `GraphSpec` records its parameter shapes, input shape, and output shape in its type.
Graphs are built from identity, primitive, and sequential-composition nodes.
:::

:::definition "graphspec_pure_semantics" (parent := "graph_representations") (lean := "NN.GraphSpec.Interp.spec")
The pure interpreter evaluates {uses "graphspec_syntax"}[a sequential graph] on
{uses "shape_indexed_tensors"}[typed parameter and input tensors] using
{uses "scalar_context"}[scalar-polymorphic operations].
:::

:::definition "graphspec_runtime_translation" (parent := "graph_representations") (lean := "NN.GraphSpec.Chain.toProgram")
The runtime translation turns {uses "graphspec_syntax"}[a sequential graph] into an
execution-polymorphic TorchLean program over {uses "scalar_context"}[the same scalar operations].
:::

:::definition "typed_dag_syntax" (parent := "graph_representations") (lean := "NN.GraphSpec.DAG.Model")
A typed DAG model pairs initialized parameters with a term whose environment contains the parameter
and input shapes. Terms may name arguments and share intermediate results.

Sharing is represented by a typed reference into that environment. Two later operations can use
one previously computed value without copying its defining term. The reference's shape determines
which operations may consume it, while the environment records where the value came from.
:::

:::definition "typed_dag_pure_semantics" (parent := "graph_representations") (lean := "NN.GraphSpec.DAG.Model.specFwd")
The DAG interpreter evaluates {uses "typed_dag_syntax"}[the model body] from typed parameter and
input lists under {uses "scalar_context"}[the scalar context].
:::

:::definition "typed_dag_runtime_translation" (parent := "graph_representations") (lean := "NN.GraphSpec.DAG.Model.toProgram")
The DAG lowering pass turns {uses "typed_dag_syntax"}[the same model body] into an
execution-polymorphic TorchLean program over {uses "scalar_context"}[the same scalar operations].
:::

:::definition "graphspec_mlp_model" (parent := "graph_representations") (lean := "NN.GraphSpec.Models.mlp")
The {uses "graphspec_syntax"}[sequential GraphSpec] MLP composes two linear maps with an intervening
ReLU. Its type fixes the order and shapes of both weights and biases.
:::

:::theorem "graphspec_mlp_spec_alignment" (parent := "graph_representations") (lean := "NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward")
Interpreting the {uses "graphspec_mlp_model"}[GraphSpec MLP] gives the same tensor as the
hand-written two-layer MLP specification.

Both sides receive the same parameter tensors and input. The result identifies two descriptions
of that forward computation; it does not compare separately trained models or establish that an
optimizer reaches particular weights. Initialization alignment below addresses the distinct question
of how the initial parameter pack is assembled.
:::

:::proof "graphspec_mlp_spec_alignment"
The proof unpacks the four-tensor parameter list, unfolds
{uses "graphspec_pure_semantics"}[the pure interpreter] for
{uses "graphspec_mlp_model"}[the MLP], and reduces both sides to the same two linear maps with an
intervening ReLU.
:::

:::theorem "graphspec_mlp_initialization_alignment" (parent := "graph_representations") (lean := "NN.GraphSpec.Models.mlp_detInitParams_eq_torchlean_linear_inits")
Deterministic initialization for the {uses "graphspec_mlp_model"}[GraphSpec MLP] produces the same
typed parameter list, in the same order, as the two TorchLean linear-layer initializers.
:::

:::proof "graphspec_mlp_initialization_alignment"
For {uses "graphspec_mlp_model"}[the two-layer graph], the occurrence-indexed seed calculation
reduces to seeds `0, 1` for the first layer and `2, 3` for the second.
:::

:::definition "ir_structural_predicate" (parent := "graph_representations") (lean := "NN.IR.Graph.wellFormed")
The Boolean structural predicate checks that node identifiers match their array positions,
operation arities are valid, and every parent points to an earlier node.
:::

:::definition "ir_structural_validation" (parent := "graph_representations") (lean := "NN.IR.Graph.checkWellFormed")
The diagnostic checker enforces the same conditions as the
{bpref "ir_structural_predicate"}[Boolean structural predicate], but returns the first useful error
message instead of a bare `false`. The two implementations are kept separate; no equivalence
theorem currently connects them.
:::

:::definition "ir_shape_validation" (parent := "graph_representations") (lean := "NN.IR.Graph.checkShapes")
After {uses "ir_structural_validation"}[the structural check], shape validation infers every node's
output shape and compares it with the shape stored in the graph.
:::

:::definition "ir_denotation" (parent := "graph_representations") (lean := "NN.IR.Graph.denoteAll")
After {uses "ir_structural_validation"}[the structural check], IR denotation evaluates every node
into a table of {uses "shape_indexed_tensors"}[shape-tagged tensor values] using
{uses "scalar_context"}[scalar-polymorphic operations] and the supplied external payload.
:::

:::theorem "ir_shape_soundness" (parent := "graph_representations") (lean := "NN.IR.Graph.checkShapes_sound")
If a graph passes {uses "ir_shape_validation"}[shape validation] and
{uses "ir_denotation"}[denotation] succeeds, then every value in the resulting table has exactly
the output shape declared by its node. Shape inference and the reference semantics are two matches
over the operation kinds; this theorem says they agree.

The two success hypotheses serve different purposes. Shape validation checks the declared graph
against its operators; successful denotation supplies the actual value table to which the conclusion
applies. The theorem does not turn a passed shape check into a guarantee that evaluation succeeds
for every external payload or scalar input.
:::

:::proof "ir_shape_soundness"
One lemma per operator family shows that the raw node evaluator returns the inferred shape when its
parents have the inferred shapes. An induction over the node array then runs
{uses "ir_shape_validation"}[inference] and {uses "ir_denotation"}[evaluation] in lockstep.
:::

:::theorem "ir_denotation_shape" (parent := "graph_representations") (lean := "NN.IR.Graph.denoteAll_shape")
Whenever {uses "ir_denotation"}[denotation] succeeds, the value table has one entry per node and
each entry carries its node's declared shape. This holds without a prior shape check because the
evaluator normalizes each result against the declared shape and fails otherwise.
:::
