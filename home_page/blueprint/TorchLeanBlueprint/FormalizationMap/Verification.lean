import Verso
import VersoManual
import VersoBlueprint
import NN.Proofs.Autograd.Core.SemiringCorrectness
import NN.Proofs.Autograd.Tape.Algebra.Soundness
import NN.Proofs.Autograd.Tape.Algebra.Nodes
import NN.Proofs.Autograd.Tape.Core.FDeriv
import NN.Proofs.Autograd.Runtime.Link.BackwardGraph
import NN.Proofs.Autograd.Runtime.Link.FDeriv
import NN.Proofs.Autograd.Runtime.Link.BackwardDenseGraph
import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
import NN.Proofs.Models.Attention.HardMask
import NN.Proofs.Autograd.Tape.Ops.Attention.SpecBridge
import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm
import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormAdjoint
import NN.Proofs.Autograd.Tape.Ops.Norm.BatchNormFDeriv
import NN.Verification.Builtin.Proved.Correctness
import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main
import NN.MLTheory.CROWN.Proofs.GraphRunibpEndToEnd
import NN.MLTheory.CROWN.Proofs.GraphRuntimeBridge
import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.AlphaBeta
import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.EndToEnd
import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
import NN.Verification.Cert.CROWNNodeCertAlphaBeta
import NN.MLTheory.CROWN.Lyapunov.Certificate
import NN.MLTheory.CROWN.Lyapunov.Verification
import NN.MLTheory.Optimization.StronglyConvexGD
import NN.MLTheory.LearningTheory.DifferentialPrivacy.Core
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationIEEE32ExecTwoLayerMlp
import NN.Proofs.Verification.ODE.Enclosure
import NN.Verification.Geometry3D.Box3D

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Verification" =>

This part of the map follows proofs from local algebra to executable checkers. A solid dependency
edge means the later statement uses the earlier definition or theorem. Runtime checkers that lack a
proved acceptance-to-semantics bridge are described beside the matching proof work, without adding
a dependency edge.

For autograd, the first step is a local identity between a Jacobian-vector product (JVP) and a
vector-Jacobian product (VJP). Graph induction extends that identity to a whole computation.
Derivative theorems additionally require the local operations to be differentiable, and runtime
theorems identify which executable accumulation agrees with the proved graph. The entries below
keep these steps separate so that the assumptions needed for each conclusion remain visible.

Read a linked declaration as a function from assumptions to a conclusion. A binder such as
`(h : TopoSorted g)` asks the caller for evidence about this particular graph; a result quantified
by `∀ x` applies to every input satisfying its later premises. The proof entries explain where
that evidence comes from. This matters when following an application backward: an enclosure
result may need both a theorem about the propagation rule and a separate theorem identifying the
program whose values it encloses.

:::group "autograd_correctness"
Algebraic, executable, and analytic accounts of reverse-mode differentiation.
:::

:::definition "autograd_local_adjoint_contract" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Node")
Each node in an algebraic proof graph carries forward, JVP, and VJP functions together with the
local dot-product identity relating its JVP and VJP.

Here the tangent describes a perturbation of the node's inputs, and the output cotangent chooses
a scalar combination of its outputs. The identity says that evaluating this combination after a
JVP gives the same scalar as pairing the input perturbation with the VJP. Requiring it for every
tangent and cotangent lets later proofs use the node in any surrounding graph.
:::

:::definition "algebraic_autograd_graph_data" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.GraphData")
`GraphData` stores a typed sequence of executable forward, JVP, and VJP node operations without
local correctness proofs.

The type parameter `Γ` lists the differentiable input shapes; `Δ` holds auxiliary data that is
kept fixed. A trainable weight must belong to the differentiable context if its cotangent is to
be returned. Storing that weight only in `Δ` changes the differentiation problem even when forward
evaluation computes the same numbers.
:::

:::definition "algebraic_autograd_graph" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph")
A proof-carrying algebraic graph is a typed sequence of
{uses "autograd_local_adjoint_contract"}[locally correct nodes]. Its output-shape list records each
intermediate added to the graph.
:::

:::definition "autograd_operation_adjoint_contract" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.OpSpecCorrect")
`OpSpecCorrect` packages a unary tensor operation, its JVP, and the local inner-product identity
relating that JVP to the operation's VJP.
:::

:::definition "autograd_operation_node_adapter" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Node.ofOpSpecCorrect")
The unary-operation adapter places an
{uses "autograd_operation_adjoint_contract"}[`OpSpecCorrect`] value at a typed input index and
produces {uses "autograd_local_adjoint_contract"}[a proof-carrying graph node].
:::

:::theorem "autograd_algebra_adjoint" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backprop_correct")
For an {uses "algebraic_autograd_graph"}[algebraic graph] over a commutative semiring, reverse
accumulation is adjoint to the graph JVP.

The commutative-semiring assumption supplies the laws for rearranging finite sums and products.
No limit or topology enters this statement. It establishes the relationship between the supplied
JVP and VJP implementations; the analytic graph theorem below additionally identifies the JVP
with the derivative of the forward function.
:::

:::proof "autograd_algebra_adjoint"
Graph induction expands the JVP and VJP at each node, then closes the new step with
{uses "autograd_local_adjoint_contract"}[that node's local adjoint law].
:::

:::definition "linear_layer_adjoint_contract" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.linearCorrect")
The linear operation satisfies
{uses "autograd_operation_adjoint_contract"}[the unary operation contract] over any commutative
semiring. Its correctness field applies
{uses "tensor_linear_adjointness"}[matrix-vector adjointness] and commutativity of the tensor dot
product.
:::

:::theorem "autograd_runtime_backward_link" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx")
Lowering an {uses "algebraic_autograd_graph"}[algebraic graph] to
{uses "runtime_autograd_tape"}[the runtime tape] and running dense backward returns the graph's
proved reverse accumulator after its typed tensor context is converted to the runtime array
representation.
:::

:::proof "autograd_runtime_backward_link"
Induction over {uses "algebraic_autograd_graph"}[the graph] maintains the index and
context correspondence of {uses "runtime_autograd_tape"}[the runtime tape] through reverse
accumulation.
:::

:::theorem "autograd_real_tape_inner_product" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Graph.backprop_correct_inner")
For a real proof-carrying tape, the inner product of a JVP with a cotangent seed equals the inner
product of the input tangent with reverse accumulation.
:::

:::proof "autograd_real_tape_inner_product"
Tape induction applies each real node's local vector adjoint law while preserving the Euclidean
inner product across context append and split operations.
:::

:::theorem "autograd_real_graph_adjoint" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Graph.backpropVec_eq_adjoint_fderiv_at")
For differentiable real graph nodes, reverse accumulation equals the adjoint Fréchet derivative at
the chosen point.
:::

:::proof "autograd_real_graph_adjoint"
{uses "autograd_real_tape_inner_product"}[Real tape soundness] supplies the inner-product identity.
The analytic hypotheses identify the graph JVP with a Fréchet derivative, and the adjoint is then
characterized by its inner products.
:::

:::theorem "autograd_lowered_tape_fderiv" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphToTape_adjoint_fderiv_at")
For a real algebraic graph at a differentiable execution point, lowering the graph and running the
exact dense tape returns the full algebraic reverse context. Its input prefix is the adjoint
Fréchet derivative of graph evaluation applied to the output seed.
:::

:::proof "autograd_lowered_tape_fderiv"
The real, environment-free algebraic graph and the analytic graph convert in both directions while
preserving evaluation, JVPs, and reverse accumulation. The tape-lowering correctness theorem gives
the full cotangent context; prefix extraction identifies its input block with analytic backprop,
and {uses "autograd_real_graph_adjoint"}[the analytic graph theorem] identifies that value with the
adjoint Fréchet derivative.
:::

:::theorem "lowered_tape_zero_preserving" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.lowerGraphToTape_zeroPreserving")
Every {uses "runtime_autograd_tape"}[tape] produced by lowering an
{uses "algebraic_autograd_graph"}[algebraic graph] over a commutative semiring is zero preserving:
each lowered node's backward closure sends a zero cotangent to zero contributions of its parents'
shapes. This is the hypothesis under which the executed sweep `backwardDenseAll` agrees with the
proved sweep `backwardDenseFrom`.
:::

:::proof "lowered_tape_zero_preserving"
A node's {uses "autograd_local_adjoint_contract"}[adjointness law] gives
`dot (jvp x dx d) δ = dotList dx (vjp x d δ)`; with `δ = 0` the left side vanishes for every `dx`,
and nondegeneracy of the tensor pairing over a commutative semiring forces `vjp x d 0 = 0`.
:::

:::theorem "autograd_executed_backward_link" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx")
For an {uses "algebraic_autograd_graph"}[algebraic graph] over a commutative semiring, the
executed dense sweep `Tape.backwardDenseAll` on the lowered
{uses "runtime_autograd_tape"}[tape], seeded at any typed output index with any seed of the
output's shape, returns exactly `backpropAllCtx` of the one-hot seed context, converted to the
tape's shape-erased array. This is the sweep the runtime runs, not only the proof-side sweep.
:::

:::proof "autograd_executed_backward_link"
{uses "lowered_tape_zero_preserving"}[Zero preservation] lets the executed sweep be rewritten as
the proved sweep started from the one-hot seed array;
{uses "autograd_runtime_backward_link"}[the proved-sweep theorem] then identifies that sweep with
graph backpropagation.

A one-hot seed context is zero at every graph value except the selected output. The seed in
that output block can be an arbitrary tensor, so this is enough to ask for a particular output
coordinate or a weighted combination. When several later nodes use the same parent, their
cotangent contributions accumulate at the same parent entry.
:::

:::theorem "autograd_executed_backward_fderiv" (parent := "autograd_correctness") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseAll_lowerGraphToTape_adjoint_fderiv")
For a real {uses "algebraic_autograd_graph"}[algebraic graph] whose nodes satisfy the analytic
`GraphFDerivCorrect` hypothesis, the executed dense sweep on the lowered tape returns the full
reverse context, and the input block of that context is the adjoint Fréchet derivative of the
graph's forward map applied to the output seed. The statement is about the exact tape model over
`ℝ`; it says nothing about `Float` rounding or the CUDA path.
:::

:::proof "autograd_executed_backward_fderiv"
The first conjunct is {uses "autograd_executed_backward_link"}[the executed-sweep link]; the second
is {uses "autograd_lowered_tape_fderiv"}[the proved-sweep Fréchet theorem] read through the same
one-hot seed context.
:::

:::group "verified_lowering"
The proved lowering from typed TorchLean programs to verifier IR.
:::

:::definition "verified_program_language" (parent := "verified_lowering") (lean := "NN.Verification.Builtin.Proved.ForwardProgram")
The verified source language is a typed sequence of supported tensor operations with one input and
shape-indexed intermediate values.

The source type already records which shapes each operation consumes and produces. Lowering
changes how those values are named and stored: a typed position becomes a node reference in the
IR. The two theorems below check complementary consequences of that change. Well-formedness makes
the references structurally valid; semantic equality shows that they still name the intended
computations.
:::

:::definition "verified_forward_lowering" (parent := "verified_lowering") (lean := "NN.Verification.Builtin.Proved.lowerForwardProgramToIR")
`lowerForwardProgramToIR` lowers {uses "verified_program_language"}[the verified source program] to
the shared IR while preserving its typed node order.
:::

:::theorem "verified_forward_well_formed" (parent := "verified_lowering") (lean := "NN.Verification.Builtin.Proved.Correctness.lowerForwardProgramToIR_wellFormed")
Every graph produced by {uses "verified_forward_lowering"}[the verified forward lowering] passes
{uses "ir_structural_predicate"}[the IR structural well-formedness predicate].
:::

:::proof "verified_forward_well_formed"
Induction over {uses "verified_program_language"}[the source program] shows that
{uses "verified_forward_lowering"}[the lowering] preserves the node-index invariant at every
append.
:::

:::theorem "verified_forward_correct" (parent := "verified_lowering") (lean := "NN.Verification.Builtin.Proved.Correctness.runForwardIR_eq_evalForward")
Running the output of {uses "verified_forward_lowering"}[the verified forward lowering] with
{uses "ir_denotation"}[the IR semantics] gives the same result as evaluating the source forward
program.
:::

:::proof "verified_forward_correct"
Induction over {uses "verified_program_language"}[the source program] relates each lowered step to
its matching {uses "ir_denotation"}[IR denotation rule].
:::

:::group "bound_propagation"
Interval and affine certificate soundness.
:::

:::theorem "ibp_local_certificate_sound" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CertSoundness.cert_encloses_semantics")
A topologically sorted supported graph over {uses "shape_indexed_tensors"}[shape-indexed tensors]
and {uses "scalar_context"}[ordered real scalars] encloses every computed node value when its local
semantic and box certificates are sound.
:::

:::proof "ibp_local_certificate_sound"
Topological induction follows the graph's {uses "shape_indexed_tensors"}[tensor values] and applies
each local enclosure result using the order from {uses "scalar_context"}[the real scalar setting].
:::

:::theorem "ibp_engine_end_to_end" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CertSoundness.runIBP?_encloses_evalGraphRec")
The proof-side real `runIBP?` pass supplies the certificates required by
{uses "ibp_local_certificate_sound"}[the local soundness theorem], so on a topologically sorted
supported graph with enclosed inputs every box it produces encloses the value computed by the
recursive evaluator `evalGraphRec`.
:::

:::proof "ibp_engine_end_to_end"
The pass is shown to produce the local certificates required by
{uses "ibp_local_certificate_sound"}[the generic soundness theorem].
:::

:::theorem "ibp_engine_matches_proof_pass" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CertSoundness.runIBP_eq_runIBP?")
On graphs whose nodes are all in `EngineCore` (input, constant, detach, addition, subtraction,
elementwise multiplication, ReLU, linear, matrix multiplication, softplus, and safe logarithm),
when the semantic guard accepts the graph and the proof-side pass produced a box at every node
(`IBPCovers`), the
executable engine's `runIBP` computes exactly the proof-side `runIBP?`. Coverage rules out the
engine's default-box path at a missing parent.

The `Option` entries matter here. `some box` supplies an enclosure candidate, while `none` records
an absent result. A fallback value in an executable array access is not evidence that the missing
parent was enclosed. `IBPCovers` ensures that the engine reads boxes actually produced by the
proved pass at every node needed by this correspondence.
:::

:::proof "ibp_engine_matches_proof_pass"
Induction over the node prefix: on each `EngineCore` kind the engine step and the proof-side step
compute the same box once their parents agree.
:::

:::theorem "ibp_executable_engine_sound" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CertSoundness.runIBP_encloses_evalGraphRec")
Under `TopoSorted`, `EngineCore`, `IBPCovers`, and `InputsEnclosed`, every box produced by the
executable `runIBP` over `ℝ` encloses the matching value of `evalGraphRec`. The theorem covers the
executable IBP engine at the real scalar. Transcendental node kinds outside `EngineCore` and
floating-point rounding require separate results.
:::

:::proof "ibp_executable_engine_sound"
{uses "ibp_engine_matches_proof_pass"}[The engine equals the proof-side pass], and
{uses "ibp_engine_end_to_end"}[the proof-side pass encloses the semantics].
:::

:::theorem "ir_crown_node_bridge" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CertSoundness.evalNode_bridge")
For the bridged node kinds, when the IR payload matches the CROWN parameter store and the IR input
is lifted to the CROWN input map, a successful step of {uses "ir_denotation"}[the IR node
evaluator] equals the CROWN node evaluator on the lifted value table. Linear nodes additionally
require vector-shaped parents. This is the per-node link between the shared IR semantics and the
graph semantics that the bound theorems above are stated against.
:::

:::proof "ir_crown_node_bridge"
Case analysis over the bridged operation kinds, unfolding both evaluators and the flattening of
{uses "shape_indexed_tensors"}[shape-tagged tensors] to flat values.
:::

:::definition "crown_transfer_contract" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownTransferSound")
An affine transfer implementation satisfies this contract when each backward transfer preserves
the represented lower and upper bounds.
:::

:::theorem "crown_generic_checker_sound" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.CrownCertSoundness.crown_checker_encloses_semantics")
A locally consistent real affine certificate encloses graph semantics when its transfer step
satisfies {uses "crown_transfer_contract"}[`CrownTransferSound`].
:::

:::proof "crown_generic_checker_sound"
Reverse topological induction composes the certified affine forms and discharges each node with the
{uses "crown_transfer_contract"}[transfer-soundness premise].
:::

:::theorem "alpha_crown_transfer" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaCrown_transfer_sound")
On a topologically sorted graph with locally consistent semantic values, matching inputs,
enclosing IBP boxes, and valid slopes (`AlphaOK`), the concrete real α-CROWN transfer step satisfies
{uses "crown_transfer_contract"}[the generic transfer contract].

For the supplied α vectors, `AlphaOK` requires each component to lie between zero and one. These
are admissible slopes for the relaxation; the theorem does not ask how the slopes were selected.
An optimizer can search for tighter bounds while this local condition remains the same proof
obligation for every candidate.
:::

:::proof "alpha_crown_transfer"
The proof checks the affine relaxation chosen for each supported operation against
{uses "crown_transfer_contract"}[the generic transfer contract].
:::

:::theorem "alpha_beta_crown_transfer" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaBetaCrown_transfer_sound")
Under the same graph, semantic, input, box, and slope hypotheses, the α/β-CROWN transfer step
satisfies {uses "crown_transfer_contract"}[the transfer contract].
Unchanged nodes reduce to {uses "alpha_crown_transfer"}[the α-CROWN transfer theorem].
:::

:::proof "alpha_beta_crown_transfer"
Split constraints are handled directly. The remaining operations reuse
{uses "alpha_crown_transfer"}[the α-CROWN transfer proof].
:::

:::theorem "ibp_boxes_enclose_values" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.ibp_encloses_vals_of_cert_local_ok")
The transfer theorems assume `IBPEnclosesVals`: every IBP box present at a node encloses the
semantic value there. On a topologically sorted supported graph with locally consistent boxes and
values and enclosed inputs, that assumption follows from
{uses "ibp_local_certificate_sound"}[IBP soundness].
:::

:::proof "ibp_boxes_enclose_values"
{uses "ibp_local_certificate_sound"}[The IBP theorem] is repackaged in the shape the transfer
theorems expect.
:::

:::theorem "alpha_crown_end_to_end" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaCrown_cert_encloses_semantics")
On a topologically sorted supported graph, given locally consistent IBP boxes and values, enclosed
inputs, an input matching the affine context, valid slopes (`AlphaOK`), and an affine certificate
that replays the α-CROWN step (`CrownCertLocalOK`), every certificate entry encloses the matching
semantic value at the input. Unlike the transfer theorem, no `IBPEnclosesVals` hypothesis remains.
:::

:::proof "alpha_crown_end_to_end"
{uses "crown_generic_checker_sound"}[The generic checker theorem] is applied with
{uses "alpha_crown_transfer"}[the α-CROWN transfer], whose IBP hypothesis is discharged by
{uses "ibp_boxes_enclose_values"}[the enclosure lemma].
:::

:::theorem "alpha_crown_total_end_to_end" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaCrown_cert_encloses_evalGraphRec")
When the IBP boxes are the proof-side `runIBP? g ps` and the values are `evalGraphRec g ps inputs`,
the α-CROWN certificate encloses the semantics under only `TopoSorted`, `Supported`, the input
conditions, `AlphaOK`, and `CrownCertLocalOK`. The local-consistency hypotheses on boxes and values
are discharged by the definitions of the two passes.

The conclusion quantifies over a node index, a stored affine bound, and a semantic value. The
premises saying that the corresponding entries are `some b` and `some v` identify which records
the enclosure relates. To apply the result to a model output, one supplies its node index and
these lookup equalities, then evaluates the affine bound at the chosen enclosed input.
:::

:::proof "alpha_crown_total_end_to_end"
{uses "alpha_crown_end_to_end"}[The α-CROWN corollary] with the local-consistency lemmas for
`runIBP?` and `evalGraphRec`, the same lemmas that drive
{uses "ibp_engine_end_to_end"}[the IBP end-to-end theorem].
:::

:::theorem "alpha_beta_crown_end_to_end" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaBetaCrown_cert_encloses_semantics'")
The α/β-CROWN analogue of {uses "alpha_crown_end_to_end"}[the α-CROWN corollary]: with a branch
vector `beta` and a certificate replaying the α/β step, every certificate entry encloses the
semantic value, with the IBP enclosure hypothesis discharged rather than assumed.
:::

:::proof "alpha_beta_crown_end_to_end"
{uses "alpha_beta_crown_transfer"}[The α/β transfer theorem] composed with
{uses "ibp_boxes_enclose_values"}[the enclosure lemma] through the generic checker theorem.
:::

:::definition "alpha_beta_crown_runner" (parent := "bound_propagation") (lean := "NN.MLTheory.CROWN.Cert.runAlphaBetaCROWN")
The native fixed-relaxation runner computes IBP bounds, infers stable ReLU phases, and replays the
α/β-CROWN affine pass. Unstable phases remain unsplit; this runner does not implement the external
optimizer's branch-and-bound search.
:::

:::definition "alpha_beta_node_checker" (parent := "bound_propagation") (lean := "NN.Verification.CROWNNodeCertAlphaBeta.checkAlphaBetaCROWNNodeCertificate")
The executable checker parses and replays an `IEEE32Exec` node certificate. Its final acceptance
decision has a proved bridge to the proposition-level local replay condition. Connecting that
binary32 condition to the real enclosure in {bpref "crown_generic_checker_sound"}[] still requires
the refinement assumptions for the operations in the graph.
:::

:::theorem "alpha_beta_node_acceptance" (parent := "bound_propagation") (lean := "NN.Verification.CROWNNodeCertAlphaBeta.AlphaBetaCROWNNodeCertificate.accepts_eq_true")
Acceptance of the in-memory α/β-CROWN decision implies `CrownCertLocalOK` for the exact
`IEEE32Exec` replay step used by the checker.
:::

:::proof "alpha_beta_node_acceptance"
The checker compares every dependent affine record bit-for-bit. Soundness of the tensor, matrix,
affine-vector, and optional-record comparisons turns the successful Boolean replay into equality at
every graph node.

The type of `accepts_eq_true` fixes the graph, parameter store, authoritative IBP array, and
certificate before asking for acceptance. Its result refers to the replay step built from those
same arguments. This prevents a successful check for one parameter store from being reused as a
local-consistency proof for a different checkpoint.
:::

:::group "spec_bridges"
Specification-layer kernels identified with their analytic presentations.
:::

:::theorem "softmax_spec_fderiv" (parent := "spec_bridges") (lean := "Proofs.Autograd.hasFDerivAt_softmaxSpec_vec")
The specification softmax `Activation.softmaxSpec 0` on a real vector, which uses the numerically
stable max-shifted form, is Fréchet differentiable after vectorization, with the derivative of the
analytic `softmaxVec`. The log-softmax kernel has the same theorem.
:::

:::proof "softmax_spec_fderiv"
The spec kernel is identified coordinatewise with the analytic softmax; the max shift cancels in
the quotient, and the analytic derivative transfers.

This avoids differentiating the maximum itself at tied coordinates. Multiplying every
exponential by the same positive factor leaves the normalized quotient unchanged, so the smooth
analytic function describes the entire shifted implementation, including ties. `HasFDerivAt`
then records that function, its continuous linear derivative, and the input point where the
claim holds.
:::

:::theorem "softmax_backward_is_vjp" (parent := "spec_bridges") (lean := "Proofs.Autograd.softmaxBackwardSpec_eq_vjp")
The specification backward rule `Activation.softmaxBackwardSpec 0` on a real vector is the
vector-Jacobian product of the specification softmax at that point. The same holds for
log-softmax.
:::

:::proof "softmax_backward_is_vjp"
`softmaxFDerivCorrect` packages the kernel as an `OpSpecFDerivCorrect` using
{uses "softmax_spec_fderiv"}[the derivative theorem]; its `backward_eq_adjoint_fderiv` field gives
the identity.
:::

:::theorem "hard_mask_all_true" (parent := "spec_bridges") (lean := "NN.Proofs.Models.Attention.hardMaskedSoftmaxSpec_allTrueMask")
Over `ℝ`, hard-masked softmax with the all-true Boolean mask equals plain axis-one softmax of the
scores. The two code paths of `Spec.scaledDotProductAttention` therefore agree wherever both apply.
:::

:::proof "hard_mask_all_true"
The row scan of the hard mask computes the fold of `max` over the row, which is the shift used by
the stable softmax; with every position allowed the numerators and denominators coincide.
:::

:::theorem "attention_tape_is_spec_vjp" (parent := "spec_bridges") (lean := "Proofs.Autograd.Attention.backpropVec_eq_adjoint_fderiv_scaledDotProductAttention")
For a nonempty sequence, the tape reverse pass on the proof-carrying scaled dot-product attention
graph, seeded on the output block, is the adjoint Fréchet derivative of
`Spec.scaledDotProductAttention` without a mask and with the canonical inverse square-root scale.
A companion theorem covers the masked code path with the all-true mask through
{uses "hard_mask_all_true"}[the hard-mask identity].
:::

:::proof "attention_tape_is_spec_vjp"
The output block of the graph evaluation is shown to be the vectorized specification forward pass,
node by node, and {uses "autograd_real_graph_adjoint"}[the analytic graph theorem] then identifies
the tape reverse pass with the adjoint derivative; {uses "softmax_spec_fderiv"}[the softmax
bridge] supplies the softmax node.
:::

:::theorem "layernorm_graph_is_spec" (parent := "spec_bridges") (lean := "Proofs.Autograd.LayerNorm.outputCLM_evalVec_layerNormGraph")
For positive row and feature counts, the output block of the proof-carrying LayerNorm graph
evaluated on packed inputs equals the vectorized `Spec.layerNorm` of those inputs.
:::

:::proof "layernorm_graph_is_spec"
Both sides are reduced to the same closed form for each matrix entry: centered value, inverse
stabilized standard deviation, scale, and shift.

Packing puts the input matrix, scale vector, and bias vector in one differentiation context.
The output projection selects the normalized matrix from a context that also contains
intermediates. Thus the equality aligns both the numerical formula and the placement of its
arguments before a derivative theorem is applied to the composed graph.
:::

:::theorem "layernorm_backward_adjoint" (parent := "spec_bridges") (lean := "Proofs.Autograd.LayerNorm.layerNormJvp_layerNormBackward_adjoint")
For positive row and feature counts, the specification LayerNorm JVP paired with an output
cotangent equals the sum of the three
pairings of the input, scale, and bias tangents with the corresponding components of
`Spec.layerNormBackward`. The backward rule is therefore the adjoint of the JVP.
:::

:::proof "layernorm_backward_adjoint"
The two pairings are expanded into finite sums over matrix entries and matched term by term; the
row-statistics terms are regrouped so that each input tangent coordinate meets the corresponding
entry of the specification backward rule.

Scale and bias are shared across rows, so their cotangents sum contributions from every row;
input cotangents retain the full matrix shape. The three pairings in the conclusion express
these different shapes in one scalar equality. This adjoint identity permits arbitrary epsilon
as written; identifying the formula with an analytic derivative needs the separate conditions
that make the stabilized denominator differentiable.
:::

:::theorem "batchnorm_fderiv" (parent := "spec_bridges") (lean := "Proofs.Autograd.BatchNorm.hasFDerivAt_batchNorm")
For a well-formed channel-first shape and $`0 < \varepsilon`, the map sending input, scale, and
shift to `Spec.batchNorm` is Fréchet differentiable on the flattened vectors.
:::

:::proof "batchnorm_fderiv"
BatchNorm flattens the spatial axes to a channel-by-position matrix and normalizes each row; the
derivative of one normalized row entry is composed with the reshaping and affine stages, and the
clamps in the specification are shown inactive because the variance is a mean of squares.

Positive epsilon makes the stabilized variance strictly positive even when every value in a
channel is identical. That is where the hypothesis enters the calculus: square root and
reciprocal are differentiated away from their singular points. The channel-first shape
hypothesis separately justifies which coordinates are collected into each normalization row.
:::

:::theorem "batchnorm_fderiv_is_jvp" (parent := "spec_bridges") (lean := "Proofs.Autograd.BatchNorm.fderiv_batchNorm_eq_batchNormJvp")
Under the hypotheses of {uses "batchnorm_fderiv"}[the differentiability theorem], the Fréchet
derivative of BatchNorm applied to a tangent equals the specification `Spec.batchNormJvp`.
:::

:::proof "batchnorm_fderiv_is_jvp"
{uses "batchnorm_fderiv"}[The derivative] is read off entrywise and compared with the closed-form
row differential that defines the JVP.
:::

:::group "proof_applications"
Selected end-to-end mathematical results and explicit assumptions.
:::

:::definition "lyapunov_certificate_valid" (parent := "proof_applications") (lean := "NN.MLTheory.CROWN.Lyapunov.LyapunovCert.ValidFor")
A Lyapunov certificate is valid for a pair of functions when its two intervals enclose the function
and orbital-derivative values throughout the stated region.
:::

:::theorem "lyapunov_conditions" (parent := "proof_applications") (lean := "NN.MLTheory.CROWN.Lyapunov.Real.lyapunov_conditions")
Certificate thresholds imply positive Lyapunov values and negative derivatives on the certified
region, conditional on {uses "lyapunov_certificate_valid"}[the certificate validity predicate].
:::

:::proof "lyapunov_conditions"
The enclosures in {uses "lyapunov_certificate_valid"}[the validity hypothesis] are compared with
the certificate thresholds and strengthened to strict sign conditions.

The conclusion gives two pointwise statements for every state inside `cert.region`. To use
them in a stability argument, the region and dynamics must be the ones of interest, and the
validity proof must enclose the stated orbital derivative there. Strict positivity of the lower
value bound also means that a zero-valued equilibrium cannot lie in this certified region.
:::

:::theorem "gradient_descent_linear_convergence" (parent := "proof_applications") (lean := "Optim.GD.dist_sq_iterate_le_of_q_nonneg")
Let $`x^\star` be a root of the update map $`g`, let $`\eta` be the step size, and set
$`q(\eta,\mu,L)=1-2\eta\mu+\eta^2L^2`. Under the stated strong-monotonicity and Lipschitz
hypotheses, with $`0\le\eta` and $`0\le q(\eta,\mu,L)`, the iterates satisfy

$$`\left\lVert \operatorname{step}_{\eta}(g)^{\,k}(x)-x^\star\right\rVert^2
\leq q(\eta,\mu,L)^k\left\lVert x-x^\star\right\rVert^2.`

Here $`k` counts update steps. The bound gives geometric decay when $`q<1`; the next theorem
provides step-size conditions that establish that inequality.
:::

:::proof "gradient_descent_linear_convergence"
The one-step contraction is iterated, and nonnegativity of $`q(\eta,\mu,L)` controls
multiplication by the geometric factor.
:::

:::theorem "gradient_descent_step_size" (parent := "proof_applications") (lean := "Optim.GD.dist_sq_iterate_le_of_step_size")
For a $`\mu`-strongly monotone, $`L`-Lipschitz update map with root $`x^\star`, if
$`0\le\mu\le L`, $`0<\eta`, and $`\eta L^2<2\mu`, then the geometric bound of
{uses "gradient_descent_linear_convergence"}[the iterate theorem] holds and, in addition,
$`q(\eta,\mu,L)<1`, so the squared distance to the root decreases geometrically. This packages the
step-size conditions a caller must check instead of assuming the contraction factor is below one.
:::

:::proof "gradient_descent_step_size"
$`q-1=\eta(\eta L^2-2\mu)` gives $`q<1` from the step-size condition; $`\mu\le L` gives
$`q\ge 0`; {uses "gradient_descent_linear_convergence"}[the iterate theorem] supplies the bound.
:::

:::theorem "differential_privacy_postprocessing" (parent := "proof_applications") (lean := "NN.MLTheory.LearningTheory.differentialPrivacy_postprocess")
Measurable post-processing preserves $`(\varepsilon,\delta)`-differential privacy.
:::

:::proof "differential_privacy_postprocessing"
The proof rewrites measurable preimages through the post-processing map and reuses the original
privacy inequality.

An observable event after post-processing corresponds to its preimage before post-processing.
Measurability makes that preimage a legal event for the original privacy bound. The deterministic
map receives only the mechanism's output in this theorem; a function that also consults the
private dataset would require a different argument.
:::

:::theorem "fp32_relu_approximation_budget" (parent := "proof_applications") (lean := "NN.MLTheory.Proofs.UniversalApproximation.IEEE32ExecTwoLayerMLP.relu_twoLayerMlp_ieee32exec_threeTerm")
Assuming real approximation, parameter quantization, and
{uses "executable_binary32"}[IEEE32 execution] budgets for a two-layer ReLU network, the total
error is bounded by their sum.
:::

:::proof "fp32_relu_approximation_budget"
After interpreting {uses "executable_binary32"}[the IEEE32 result] as a real value, two triangle
inequalities split the target error into the three assumed budgets.
:::

:::theorem "ode_extended_enclosure" (parent := "proof_applications") (lean := "NN.Proofs.Verification.ODE.Enclosure.extendedSolutionEnclosed_fromClampedDynamics")
A comparison argument encloses a clamped scalar ODE solution with constant extension outside the
integration interval.
:::

:::proof "ode_extended_enclosure"
The proof combines the in-interval differential inequality with the two constant-extension cases.

The hypotheses include the initial value between the walls, continuity and one-sided derivative
bounds up to the horizon, and inward-pointing inequalities at the frozen walls afterward. The
result both encloses the supplied clamped solution and shows that it satisfies the original ODE.
Once the solution lies between the walls, clamping its value has no effect on the vector field.
:::

:::theorem "camera_box_checker_sound" (parent := "proof_applications") (lean := "NN.Verification.Geometry3D.Box3D.checkCert_sound")
Acceptance by the boolean 3D box and camera checker over
{uses "shape_indexed_tensors"}[shape-indexed inputs] yields a `Verified3DBox` certificate.
:::

:::proof "camera_box_checker_sound"
Each boolean guard over {uses "shape_indexed_tensors"}[the tensor inputs] is reflected into its
proposition and assembled into the certificate structure.

The resulting fields record positive image dimensions and corner depths, an ordered box inside
the image, projected corners inside the image, and their enclosure by the box with its stated
tolerance. These are properties of the supplied camera artifact. They can be inspected separately
when a downstream geometric argument needs, for example, the positive-depth premise for a
projection.
:::
