import Verso
import VersoManual
import VersoBlueprint
import TorchLean.NN.Tape.AdjointOp
import TorchLean.NN.Tape.Graph
import TorchLean.NN.Tape.LawfulNode
import TorchLean.NN.Tape.RealGraph.FDeriv
import TorchLean.NN.Tape.Lowering.BackwardGraph
import TorchLean.NN.Tape.Lowering.FDeriv
import TorchLean.NN.Tape.Lowering.BackwardDenseGraph
import TorchLean.NN.Op.Certificates
import TorchLean.NN.Op.Softmax
import TorchLean.NN.Softmax.Correctness
import TorchLean.NN.Attention.HardMask
import TorchLean.NN.Attention.ScaledDotProduct.FDeriv
import TorchLean.NN.Norm.LayerNorm
import TorchLean.NN.Norm.LayerNorm.Adjoint
import TorchLean.NN.Norm.BatchNorm.FDeriv
import TorchLean.Verify.CROWN.Arithmetic
import TorchLean.Verify.Certificate.CROWNNode
import TorchLean.Verify.CROWN.Op.Alpha
import TorchLean.Verify.CROWN.Op.BackwardRun
import TorchLean.Verify.CROWN.Op.ExecCert
import TorchLeanExamples.Verification.Lyapunov.Certificate
import TorchLeanExamples.Verification.Lyapunov.SignConditions
import TorchLean.NN.Optimizer.GD.Convergence
import TorchLean.NN.LearningTheory.DifferentialPrivacy
import TorchLean.Verify.UniversalApproximation.Rounded
import TorchLean.Verify.UniversalApproximation.Binary
import TorchLean.Verify.UniversalApproximation.BinaryMLP.Error
import TorchLeanExamples.Verification.ODE.Enclosure
import TorchLeanExamples.Verification.Geometry3D.Box3D

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
Derivative theorems also require the local operations to be differentiable, and runtime theorems
identify which executable accumulation agrees with the proved graph. We keep these steps in
separate entries so that the assumptions needed for each conclusion remain visible.

Read a linked declaration as a function from assumptions to a conclusion. A binder such as
`(h : TopoSorted g)` asks the caller for evidence about this particular graph, while a result
quantified by `∀ x` applies to every input satisfying its later premises. Proof entries explain
where that evidence comes from. This matters when following an application backward: an enclosure
result may need both a theorem about the propagation rule and a separate theorem identifying the
program whose values it encloses.

:::group "autograd_correctness"
Algebraic, executable, and analytic accounts of reverse-mode differentiation.
:::

:::definition "autograd_local_adjoint_contract" (parent := "autograd_correctness") (lean := "TorchLean.Tape.LawfulNode")
Each node in an algebraic proof graph carries forward, JVP, and VJP functions together with the
local dot-product identity relating its JVP and VJP.

Here the tangent describes a perturbation of the node's inputs, and the output cotangent chooses a
scalar combination of its outputs. Read this way, the identity says that evaluating this
combination after a JVP gives the same scalar as pairing the input perturbation with the VJP.
Requiring it for every tangent and cotangent lets later proofs use the node in any surrounding
graph.
:::

:::definition "algebraic_autograd_graph_data" (parent := "autograd_correctness") (lean := "TorchLean.Tape.GraphData")
`GraphData` stores a typed sequence of executable forward, JVP, and VJP node operations without
local correctness proofs.

The type parameter `Γ` lists the differentiable input shapes, while `Δ` holds auxiliary data that
is kept fixed. A trainable weight must belong to the differentiable context if its cotangent is to
be returned. Storing that weight only in `Δ` changes the differentiation problem even when forward
evaluation computes the same numbers.
:::

:::definition "algebraic_autograd_graph" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph")
A proof-carrying algebraic graph is a typed sequence of
{uses "autograd_local_adjoint_contract"}[locally correct nodes]. Its output-shape list records each
intermediate added to the graph.
:::

:::definition "autograd_operation_adjoint_contract" (parent := "autograd_correctness") (lean := "TorchLean.Tape.AdjointOp")
`Tape.AdjointOp` packages a unary tensor operation, its JVP, and the local inner-product identity
relating that JVP to the operation's VJP.
:::

:::definition "autograd_operation_node_adapter" (parent := "autograd_correctness") (lean := "TorchLean.Tape.LawfulNode.ofAdjointOp")
The unary-operation adapter places an
{uses "autograd_operation_adjoint_contract"}[`Tape.AdjointOp`] value at a typed input index and
produces {uses "autograd_local_adjoint_contract"}[a proof-carrying graph node].
:::

:::theorem "autograd_algebra_adjoint" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph.jvpCtx_backpropCtx_adjoint")
For an {uses "algebraic_autograd_graph"}[algebraic graph] over a commutative semiring, reverse
accumulation is adjoint to the graph JVP.

The commutative-semiring assumption supplies the laws for rearranging finite sums and products. No
limit or topology enters this statement. It establishes the relationship between the supplied JVP
and VJP implementations, and the analytic graph theorem below also identifies the JVP with the
derivative of the forward function.
:::

:::proof "autograd_algebra_adjoint"
Graph induction expands the JVP and VJP at each node, then closes the inductive step with
{uses "autograd_local_adjoint_contract"}[that node's local adjoint law].
:::

:::definition "linear_layer_adjoint_contract" (parent := "autograd_correctness") (lean := "TorchLean.Tape.AdjointOp.linear")
The linear operation satisfies
{uses "autograd_operation_adjoint_contract"}[the unary operation contract] over any commutative
semiring. Its correctness field applies
{uses "tensor_linear_adjointness"}[matrix-vector adjointness] and commutativity of the tensor dot
product.
:::

:::theorem "autograd_real_tape_inner_product" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph.jvpVec_backpropVec_adjoint")
For an {uses "algebraic_autograd_graph"}[algebraic graph] over `ℝ` at a fixed environment, the
inner product of the vectorized JVP with a cotangent seed equals the inner product of the input
tangent with vectorized reverse accumulation.
:::

:::proof "autograd_real_tape_inner_product"
Flattening is a bijection that carries the tensor-list pairing to the Euclidean inner product, so
the statement is {uses "autograd_algebra_adjoint"}[the algebraic adjointness theorem] at `ℝ`.
:::

:::theorem "autograd_real_graph_adjoint" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph.backpropVec_eq_fderiv")
For an {uses "algebraic_autograd_graph"}[algebraic graph] over `ℝ` at a fixed environment whose
nodes are differentiable at the values they receive, reverse accumulation equals the adjoint
Fréchet derivative at the chosen point.
:::

:::proof "autograd_real_graph_adjoint"
{uses "autograd_real_tape_inner_product"}[Tape soundness over `ℝ`] supplies the inner-product
identity. Then the analytic hypotheses identify the graph JVP with a Fréchet derivative, and the
adjoint is characterized by its inner products.
:::

:::theorem "autograd_lowered_tape_fderiv" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph.backwardDenseFrom_eq_fderiv")
For a real algebraic graph at a differentiable execution point, lowering the graph and running the
exact dense tape returns the full algebraic reverse context. Its input prefix is the adjoint
Fréchet derivative of graph evaluation applied to the output seed.
:::

:::proof "autograd_lowered_tape_fderiv"
{uses "typed_graph_backward_agreement"}[The tape-lowering correctness theorem] gives the full
cotangent context. Prefix extraction identifies its input block with the inputs-only reverse pass,
and {uses "autograd_real_graph_adjoint"}[the analytic graph theorem] for the same graph at the same
environment identifies that value with the adjoint Fréchet derivative.
:::

:::theorem "lowered_tape_zero_preserving" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph.zeroPreserving_toTape")
Every {uses "runtime_autograd_tape"}[tape] produced by lowering an
{uses "algebraic_autograd_graph"}[algebraic graph] over a commutative semiring is zero preserving:
each lowered node's backward closure sends a zero cotangent to zero contributions of its parents'
shapes. Over a scalar whose zero is an additive identity, this is a sufficient condition for the
executed sweep `backwardDenseAll` to agree with the proved sweep `backwardDenseFrom`. IEEE
floating point is not such a scalar, since $`(+0) + (-0) = +0` under round-to-nearest
(IEEE 754-2019 6.3).
:::

:::proof "lowered_tape_zero_preserving"
A node's {uses "autograd_local_adjoint_contract"}[adjointness law] gives
`dot (jvp x dx d) δ = dotList dx (vjp x d δ)`. With `δ = 0` the left side vanishes for every `dx`,
and nondegeneracy of the tensor pairing over a commutative semiring forces `vjp x d 0 = 0`.
:::

:::theorem "autograd_executed_backward_link" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph.backwardDenseAll_toTape")
For an {uses "algebraic_autograd_graph"}[algebraic graph] over a commutative semiring, the
executed dense sweep `Tape.backwardDenseAll` on the lowered
{uses "runtime_autograd_tape"}[tape], seeded at any typed output index with any seed of the
output's shape, returns exactly `backpropAllCtx` of the one-hot seed context, converted to the
tape's shape-erased array. The eager runtime runs the same function on its session tape, not on a
lowered tape, so this theorem does not describe that run.
:::

:::proof "autograd_executed_backward_link"
{uses "lowered_tape_zero_preserving"}[Zero preservation] lets the executed sweep be rewritten as
the proved sweep started from the one-hot seed array, and
{uses "typed_graph_backward_agreement"}[the proved-sweep theorem] then identifies that sweep with
graph backpropagation.

A one-hot seed context is zero at every graph value except the selected output. In that output
block the seed can be an arbitrary tensor, so this is enough to ask for a particular output
coordinate or a weighted combination. When several later nodes use the same parent, their cotangent
contributions accumulate at the same parent entry.
:::

:::theorem "autograd_executed_backward_fderiv" (parent := "autograd_correctness") (lean := "TorchLean.Tape.Graph.backwardDenseAll_eq_fderiv")
For a real {uses "algebraic_autograd_graph"}[algebraic graph] whose nodes satisfy the analytic
`RealGraph.FDerivCorrectAt` hypothesis at the input, the executed dense sweep on the lowered tape
returns the full reverse context, and the input block of that context is the adjoint Fréchet
derivative of the graph's forward map applied to the output seed. This statement is about the exact
tape model over `ℝ`, so it says nothing about `Float` rounding or the CUDA path.
:::

:::proof "autograd_executed_backward_fderiv"
Of the two conjuncts, the first is {uses "autograd_executed_backward_link"}[the executed-sweep
link] and the second is {uses "autograd_lowered_tape_fderiv"}[the proved-sweep Fréchet theorem]
read through the same one-hot seed context.
:::

:::definition "registered_operator_certificate" (parent := "autograd_correctness") (lean := "TorchLean.Op.softmax_backward_eq_adjoint")
A certificate of a registered operator is a theorem registered with `op_cert` on the operator's arm
of `Op.rule?`, with a kind that says what it claims: an adjoint pair, a Fréchet derivative, an AD
convention, a prepared pullback or a rounding bound. For softmax, the derivative certificate
states, at every shape and axis, that the arm's forward map has the Fréchet derivative that applies
the softmax derivative to each fiber along the axis, and that the arm's pullback is its adjoint.
:::

:::group "bound_propagation"
Interval and affine certificate soundness on graphs of registered operators.
:::

:::theorem "ibp_local_certificate_sound" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.certLocalOK_encloses")
Over {uses "shape_indexed_tensors"}[shape-indexed tensors] and
{uses "scalar_context"}[ordered real scalars], under local certificate consistency, a table of node
outputs that is the fixpoint of the graph's node evaluation, and seed boxes that enclose the graph
inputs, every certificate box encloses the flat value of its node in that table. There is no
operator, parent shape, or ordering premise. The checker boxes input and constant nodes, the
operators of kind `shared` in the table `Op.ruleKind`, and tanh, sigmoid, sine, and cosine. Every
other operator gets no box.
:::

:::proof "ibp_local_certificate_sound"
The checker's rule is sound at every node whose references' boxes enclose the values they read: at
tanh and the sigmoid by their monotone endpoint bounds, at sine and cosine by the clipped midpoint
bounds, and at every other operator because it is the rule of the rounded pass at `ℝ`, which
{uses "ibp_rounded_step_sound"}[the step theorem] covers. Strong induction over node ids finishes
it, since a successful node evaluation reads only the nodes before it.
:::

:::theorem "ibp_certificate_computed" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.certify_localOK")
On a graph whose references name earlier nodes, the certificate `certify` computes node by node is
locally consistent, so it supplies the certificate premise of
{uses "ibp_local_certificate_sound"}[the local soundness theorem].
:::

:::proof "ibp_certificate_computed"
Each step of the certificate reads only the entries of the nodes its references name, all of which
come earlier, so the computed table is the step's fixpoint.
:::

:::theorem "ibp_certificate_run" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.semLocalOK_iff_run")
A table of node outputs is the fixpoint of the graph's node evaluation exactly when it is the
outputs of a successful run of the graph. With
{uses "ibp_local_certificate_sound"}[the certificate theorem], every box of a certificate that the
local IBP checker accepts encloses the flat value of its node in every successful real run whose
graph inputs the seed boxes contain.
:::

:::proof "ibp_certificate_run"
A successful run evaluates each node on the outputs of the nodes before it, which is the node
evaluation there. Conversely, induction over the node prefix shows that the run of a fixpoint
table's first nodes returns that table's first entries.
:::

:::theorem "ibp_engine_matches_certificate" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.runIBP_eq_certify")
On a graph whose every operator has the same interval rule in the executable pass `runIBP` and in
the certificate checker (`SharedOnly`: kind `shared` or `none` in `Op.ruleKind`), the pass at `ℝ`
computes exactly the certificate. Elsewhere the pass's table is in general not a certificate: it
has boxes where the checker has none (at the operators of kind `directed` such as reshape, softmax,
or layer norm and at their consumers), and at tanh, sigmoid, sine, and cosine it takes the
backend's enclosures.
:::

:::proof "ibp_engine_matches_certificate"
Both tables are computed node by node, with the same step at every shared operator, so they agree
by induction over the node prefix.
:::

:::theorem "ibp_executable_engine_sound" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.runIBP_encloses_of_run")
At `ℝ`, on a graph without tanh, sigmoid, sine, and cosine (`NoBackendEnclosure`), the executable
pass gives every node that the certificate boxes the same box, and that box encloses the node's
flat value in every successful real run whose inputs the seed boxes enclose. There is no ordering
premise.
:::

:::proof "ibp_executable_engine_sound"
Away from the four backend enclosures the pass keeps every box of the certificate that `certify`
computes. Each entry of that table is the certificate step on the entries before it, so
{uses "ibp_rounded_step_sound"}[the step theorem] under the checker's rule, sound by
{uses "ibp_local_certificate_sound"}[the certificate theorem's proof], encloses every one of them by
strong induction over node ids.
:::

:::theorem "ibp_rounded_step_sound" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.Directed.step?_encloses")
At a scalar with `LawfulBoundOps`, on any table of boxes, the box the interval step under a sound
rule (`RuleEncloses`) gives a node encloses the node's flat value in a successful real run on a
payload holding the reals of the rounded constants, when every seed box encloses its graph input
and the table's box at each reference to an earlier node encloses the value there. At a scalar that
also has `LawfulNonlinearBoundOps` the rule of the rounded pass is sound (`Op.ibp?_encloses`), and
every operator it boxes has a kind other than `none` in `Op.ruleKind` (`supported_of_ibp?`).
:::

:::proof "ibp_rounded_step_sound"
An input node's box is its seed, a constant node's box is the point box of its rounded value, and
an operator node's box is the rule's on the boxes of its references. The rounded pass's rule is
sound by case analysis over the operator rules: copies, pointwise arithmetic and the backend
enclosures, sums and means, the matrix products, pooling, normalization, convolution,
concatenation, and the re-indexings, each from the directed endpoint laws and the scalar enclosure
laws.
:::

:::theorem "ibp_rounded_engine_sound" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.Directed.encloses_of_run")
At a scalar with `LawfulBoundOps` and `LawfulNonlinearBoundOps`, every box returned by the
executable `runIBP` encloses the flat value of its node in a successful real run on a payload
holding the reals of the rounded constants, when every seed box encloses the graph input its input
node reads. There is no operator, parent shape, or ordering premise and no intermediate enclosure
hypothesis. Both the reals and the rounded-real `FP32` model have the required scalar instances.

The theorem constrains successful entries. Operators without an interval rule, invalid shapes,
missing constants, and unavailable arithmetic transfers can still return `none`.
:::

:::proof "ibp_rounded_engine_sound"
Strong induction along the node order supplies each reference's enclosure before applying
{uses "ibp_rounded_step_sound"}[the step theorem] to the prefix of the executable table.
:::

:::theorem "directed_crown_rounded_end_to_end" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.backwardObjectiveBox?_of_run")
At a scalar with lawful endpoint operations, a successful backward CROWN objective query over an
interval table whose boxes enclose a successful real run returns an interval containing the real
objective at the output, when the payload holds the reals of the rounded constants, the objective
has the output's width, and the input box encloses graph input `0`. `runIBP` produces such a table
(`runIBP_encloses`).

Backward propagation, interval fallbacks, McCormick product planes, the transposed convolution, and
final affine-bound evaluation are composed in this result. There is no node equation to assume. But
the theorem concerns a successful `.ok` result, and it doesn't guarantee that every query succeeds
or yields a tighter bound than IBP. `outputBoxCROWN?_of_run` is the same statement for the CROWN
output box.
:::

:::proof "directed_crown_rounded_end_to_end"
We take the real point of the graph and the real equation of every operator node from the run
(`graphPoint_of_run`, `equation_of_run`). Directed backward soundness and the outward-rounded
affine evaluator then enclose the objective.
:::

:::theorem "full_crown_rounded_sound" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.crownBoxes_encloses")
Full CROWN meets each node's interval box with its affine bounds evaluated on the input box. At a
scalar with lawful endpoint operations, every box it returns encloses its node's flat value in a
successful real run, when the seed boxes, the supplied table, and the input box enclose the run.
`runCROWN_encloses` gives the same for the affine bounds of every node.
:::

:::proof "full_crown_rounded_sound"
Node by node: {uses "ibp_rounded_step_sound"}[the interval step] encloses the run, and so do its
meets with the supplied box and with the node's affine bounds evaluated on the input box
(`evalCROWNOutputBox?_encloses`).
:::

:::theorem "alpha_crown_end_to_end" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.Alpha.encloses_of_run")
At `ℝ`, every affine bound of a table whose present entries the forward α-CROWN step replays
encloses, at the input `x`, the flat value of its node in a successful real run, when the interval
table encloses the run's flat values, the designated input node has flat value `x`, and the α
vectors lie in `[0,1]` (`AlphaOK`). The step covers detach, linear layers with constant weight and
bias, ReLU with slopes or with phases the intervals justify, sums over every axis, and reshape.
Every other operator but the matrix product and custom operators takes the constant forms of its
own box.

For the supplied α vectors, `AlphaOK` requires each component to lie between zero and one. These
are admissible slopes for the relaxation, and the theorem doesn't ask how the slopes were selected.
An optimizer can search for tighter bounds while this local condition remains the same proof
obligation for every candidate.
:::

:::proof "alpha_crown_end_to_end"
Strong induction over node ids applies the one-node rule `alpha?_encloses`, whose ReLU case is
the α-CROWN and phase relaxation `reluBounds?_encloses`, with the interval enclosure supplied by
the premise.
:::

:::theorem "phased_crown_end_to_end" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.runAlphaCROWN_encloses")
At `ℝ`, every affine bound of {uses "phased_crown_runner"}[the phased runner] encloses, at the
input `x`, the flat value of its node in a successful real run, when the interval table encloses
the run and the designated input node has flat value `x`. For the runner's default slopes and
inferred phases, no `AlphaOK` premise is needed.
:::

:::proof "phased_crown_end_to_end"
We apply the one-node step of {uses "alpha_crown_end_to_end"}[the α-CROWN theorem] node by node
along the runner's table, whose default slopes lie in `[0,1]`.
:::

:::definition "phased_crown_runner" (parent := "bound_propagation") (lean := "TorchLean.CROWN.Op.runAlphaCROWN")
Given IBP bounds, the fixed-relaxation runner infers stable ReLU phases and runs the phased α-CROWN
forward pass. Its `outputBoxAlphaCROWN?` caller computes the IBP bounds first. Unstable phases
remain unsplit, no β multipliers are optimized, and this runner doesn't implement the
branch-and-bound search of the external alpha-beta-CROWN verifier.
:::

:::definition "crown_node_checker" (parent := "bound_propagation") (lean := "TorchLean.Certificate.CROWNNode.check")
The executable safety verdict parses a FloatLib binary32 node certificate on a graph of registered
operators and decides it with `TorchLean.Certificate.FiniteArtifact.accepts`, which checks every
affine entry against the exact rational transfer of its parent over the input box. For linear and
ReLU chains, `TorchLean.Certificate.FiniteArtifact.accepts_graph_sound` proves the requested output
inequalities of an accepted artifact for every real input in the box, and the verdict rejects every
other graph. Agreement with the operator graph's forward α-CROWN pass replayed in binary32, with or
without ReLU phases, is the separate diagnostic `TorchLean.Certificate.CROWNNode.replayDiagnostic`,
which is not a safety verdict. That replay gives no real enclosure, because it rounds to nearest:
after the layers $`h = x + 1` and $`y = h + 2^{-24}` at $`x = -1`, where $`y = 2^{-24}`, the
replayed upper form of $`y` is $`x + 1`, which is $`0` there.
:::

:::theorem "crown_node_replay_agreement" (parent := "bound_propagation") (lean := "TorchLean.Certificate.CROWNNode.replayAgrees_spec")
Agreement of an in-memory CROWN node certificate with the replay implies local consistency for the
exact FloatLib binary32 replay step of the replay diagnostic: one optional table entry per node of
the operator graph, each equal to the replay of the table. It does not show that any entry is
present, and agreement is not a safety verdict.
:::

:::proof "crown_node_replay_agreement"
The diagnostic compares every dependent affine record bit-for-bit. Soundness of the tensor, matrix,
affine-vector, and optional-record comparisons turns the successful Boolean replay test into
equality at every graph node.

The type of `CROWNNode.replayAgrees_spec` fixes the graph, parameter store, authoritative IBP
array, and certificate before asking for agreement. Its result refers to the replay step built from
those same arguments. This prevents a successful comparison for one parameter store from being
reused as a local-consistency proof for a different checkpoint.
:::

:::group "spec_bridges"
Specification-layer kernels identified with their analytic presentations.
:::

:::theorem "softmax_spec_fderiv" (parent := "spec_bridges") (lean := "TorchLean.Tensor.hasFDerivAt_softmax")
On a real vector, the specification softmax `Tensor.softmax x 0` (which uses the numerically stable
max-shifted form) is Fréchet differentiable after vectorization, with the derivative of the
analytic `Vec.softmax`. `Tensor.hasFDerivAt_logSoftmax` gives the corresponding result for
log-softmax.
:::

:::proof "softmax_spec_fderiv"
We identify the spec kernel coordinatewise with the analytic softmax. Its max shift cancels in the
quotient, and the analytic derivative transfers.

This avoids differentiating the maximum itself at tied coordinates. Multiplying every
exponential by the same positive factor leaves the normalized quotient unchanged, so the smooth
analytic function describes the entire shifted implementation, including ties. `HasFDerivAt`
then records that function, its continuous linear derivative, and the input point where the
claim holds.
:::

:::theorem "softmax_backward_is_vjp" (parent := "spec_bridges") (lean := "TorchLean.Tensor.softmaxBackward_eq_fderiv")
At a real vector, the specification backward rule `Tensor.softmaxBackward x dY 0` is the
vector-Jacobian product of the specification softmax at that point. `logSoftmaxBackward_eq_fderiv`
gives the corresponding result for log-softmax.
:::

:::proof "softmax_backward_is_vjp"
`FDerivOp.softmax` packages the kernel as an `FDerivOp` using
{uses "softmax_spec_fderiv"}[the derivative theorem], and
`FDerivOp.backward_eq_fderiv` applied to it gives the identity.
:::

:::theorem "hard_mask_all_true" (parent := "spec_bridges") (lean := "TorchLean.Softmax.hardMasked_allTrueMask")
Over `ℝ`, hard-masked softmax with the all-true Boolean mask equals plain axis-one softmax of the
scores. So the two code paths of `TorchLean.Attention.scaledDotProduct` agree wherever both apply.
:::

:::proof "hard_mask_all_true"
In the hard mask, the row scan computes the fold of `max` over the row, which is the shift used by
the stable softmax, so with every position allowed the numerators and denominators coincide.
:::

:::theorem "attention_tape_is_spec_vjp" (parent := "spec_bridges") (lean := "TorchLean.Attention.backprop_eq_fderiv")
For a nonempty sequence, the tape reverse pass on the proof-carrying scaled dot-product attention
graph, seeded on the output block, is the adjoint Fréchet derivative of
`TorchLean.Attention.scaledDotProduct` without a mask and with the canonical inverse square-root
scale. A companion theorem covers the masked code path with the all-true mask through
{uses "hard_mask_all_true"}[the hard-mask identity].
:::

:::proof "attention_tape_is_spec_vjp"
Node by node, we show that the output block of the graph evaluation is the vectorized specification
forward pass, and {uses "autograd_real_graph_adjoint"}[the analytic graph theorem] then identifies
the tape reverse pass with the adjoint derivative. {uses "softmax_spec_fderiv"}[The softmax bridge]
supplies the softmax node.
:::

:::theorem "layernorm_graph_is_spec" (parent := "spec_bridges") (lean := "TorchLean.LayerNorm.outputCLM_evalVec_graph")
For positive row and feature counts, the output block of the proof-carrying LayerNorm graph
evaluated on packed inputs equals the vectorized `Tensor.layerNorm` of those inputs.
:::

:::proof "layernorm_graph_is_spec"
Both sides are reduced to the same closed form for each matrix entry: centered value, inverse
stabilized standard deviation, scale, and shift.

Packing puts the input matrix, scale vector, and bias vector in one differentiation context, and
the output projection selects the normalized matrix from a context that also contains
intermediates. Thus the equality aligns both the numerical formula and the placement of its
arguments before a derivative theorem is applied to the composed graph.
:::

:::theorem "layernorm_backward_adjoint" (parent := "spec_bridges") (lean := "TorchLean.LayerNorm.jvp_backward_adjoint")
For positive row and feature counts, the specification LayerNorm JVP paired with an output
cotangent equals the sum of the three pairings of the input, scale, and bias tangents with the
corresponding components of `TorchLean.LayerNorm.backward`. So the backward rule is the adjoint of
the JVP.
:::

:::proof "layernorm_backward_adjoint"
We expand the two pairings into finite sums over matrix entries and match them term by term,
regrouping the row-statistics terms so that each input tangent coordinate meets the corresponding
entry of the specification backward rule.

Scale and bias are shared across rows, so their cotangents sum contributions from every row, while
input cotangents retain the full matrix shape. In the conclusion, the three pairings express these
different shapes in one scalar equality. This adjoint identity permits arbitrary epsilon as
written, but identifying the formula with an analytic derivative needs the separate conditions that
make the stabilized denominator differentiable.
:::

:::theorem "batchnorm_fderiv" (parent := "spec_bridges") (lean := "TorchLean.BatchNorm.hasFDerivAt_forwardVec")
For a well-formed channel-first shape and $`0 < \varepsilon`, the map sending input, scale, and
shift to `Tensor.batchNorm` along the leading channel axis with batch statistics is Fréchet
differentiable on the flattened vectors.
:::

:::proof "batchnorm_fderiv"
BatchNorm flattens the spatial axes to a channel-by-position matrix and normalizes each row. We
compose the derivative of one normalized row entry with the reshaping and affine stages, and show
that the clamps in the specification are inactive because the variance is a mean of squares.

Positive epsilon makes the stabilized variance strictly positive even when every value in a channel
is identical. That is where the hypothesis enters the calculus: square root and reciprocal are
differentiated away from their singular points. Separately, the channel-first shape hypothesis
justifies which coordinates are collected into each normalization row.
:::

:::theorem "batchnorm_fderiv_is_jvp" (parent := "spec_bridges") (lean := "TorchLean.BatchNorm.fderiv_forwardVec_eq_jvp")
Under the hypotheses of {uses "batchnorm_fderiv"}[the differentiability theorem], the Fréchet
derivative of BatchNorm applied to a tangent equals the specification `TorchLean.BatchNorm.jvp`.
:::

:::proof "batchnorm_fderiv_is_jvp"
We read {uses "batchnorm_fderiv"}[the derivative] off entrywise and compare it with the closed-form
row differential that defines the JVP.
:::

:::group "proof_applications"
Selected end-to-end mathematical results and their assumptions.
:::

:::definition "lyapunov_certificate_valid" (parent := "proof_applications") (lean := "TorchLean.Examples.Lyapunov.Cert.ValidFor")
A Lyapunov certificate is valid for a pair of functions when its two intervals enclose the function
and orbital-derivative values throughout the stated region.
:::

:::theorem "lyapunov_conditions" (parent := "proof_applications") (lean := "TorchLean.Examples.Lyapunov.Cert.sign_conditions")
Certificate thresholds imply positive Lyapunov values and negative derivatives on the certified
region, conditional on {uses "lyapunov_certificate_valid"}[the certificate validity predicate].
:::

:::proof "lyapunov_conditions"
We compare the enclosures in {uses "lyapunov_certificate_valid"}[the validity hypothesis] with the
certificate thresholds and strengthen them to strict sign conditions.

For every state inside `cert.region`, the conclusion gives two pointwise statements. To use them in
a stability argument, the region and dynamics must be the ones of interest, and the validity proof
must enclose the stated orbital derivative there. Strict positivity of the lower value bound also
means that a zero-valued equilibrium can't lie in this certified region.
:::

:::theorem "gradient_descent_linear_convergence" (parent := "proof_applications") (lean := "TorchLean.GD.dist_sq_iterate_le_of_q_nonneg")
Let $`x^\star` be a root of the update map $`g`, let $`\eta` be the step size, and set
$`q(\eta,\mu,L)=1-2\eta\mu+\eta^2L^2`. Under the stated strong-monotonicity and Lipschitz
hypotheses, with $`0\le\eta` and $`0\le q(\eta,\mu,L)`, the iterates satisfy

$$`\left\lVert \operatorname{step}_{\eta}(g)^{\,k}(x)-x^\star\right\rVert^2
\leq q(\eta,\mu,L)^k\left\lVert x-x^\star\right\rVert^2.`

Here $`k` counts update steps. When $`q<1` the bound gives geometric decay, and the next theorem
provides step-size conditions that establish that inequality.
:::

:::proof "gradient_descent_linear_convergence"
We iterate the one-step contraction, and nonnegativity of $`q(\eta,\mu,L)` controls multiplication
by the geometric factor.
:::

:::theorem "gradient_descent_step_size" (parent := "proof_applications") (lean := "TorchLean.GD.dist_sq_iterate_le_of_step_size")
For a $`\mu`-strongly monotone, $`L`-Lipschitz update map with root $`x^\star`, if
$`0\le\mu\le L`, $`0<\eta`, and $`\eta L^2<2\mu`, then the geometric bound of
{uses "gradient_descent_linear_convergence"}[the iterate theorem] holds and, in addition,
$`q(\eta,\mu,L)<1`, so the squared distance to the root decreases geometrically. This packages the
step-size conditions a caller must check instead of assuming the contraction factor is below one.
:::

:::proof "gradient_descent_step_size"
$`q-1=\eta(\eta L^2-2\mu)` gives $`q<1` from the step-size condition, $`\mu\le L` gives $`q\ge 0`,
and {uses "gradient_descent_linear_convergence"}[the iterate theorem] supplies the bound.
:::

:::theorem "differential_privacy_postprocessing" (parent := "proof_applications") (lean := "TorchLean.LearningTheory.DifferentialPrivacy.postprocess")
Measurable post-processing preserves $`(\varepsilon,\delta)`-differential privacy.
:::

:::proof "differential_privacy_postprocessing"
The proof rewrites measurable preimages through the post-processing map and reuses the original
privacy inequality.

An observable event after post-processing corresponds to its preimage before post-processing, and
measurability makes that preimage a legal event for the original privacy bound. In this theorem the
deterministic map receives only the mechanism's output. A function that also consults the private
dataset would require a different argument.
:::

:::theorem "rounded_relu_approximation" (parent := "proof_applications") (lean := "TorchLean.UniversalApproximation.Rounded.relu_Icc")
For each FloatLib format, the constructive hinge network approximates a Lipschitz target with error
below the requested real approximation tolerance plus its pointwise evaluation-rounding bound. Its
parameters inhabit FloatLib's rounded-real carrier, and their stored real values need not be
representable in the selected format.
:::

:::proof "rounded_relu_approximation"
Parameters come from the real hinge construction. A fold invariant bounds the accumulated rounding
error using FloatLib's half-ULP theorem at each subtraction, product, and addition. A triangle
inequality adds this evaluation bound to the approximation tolerance.
:::

:::theorem "binary_hinge_approximation" (parent := "proof_applications") (lean := "TorchLean.UniversalApproximation.relu_Icc_binary_three_term")
For a configured IEEE format and its storage codec, executable hinge evaluation has error below
the supplied approximation and parameter-quantization budgets plus the proved rounding bound.
Every intermediate computation and the output must satisfy the theorem's finiteness hypotheses.
:::

:::proof "binary_hinge_approximation"
FloatLib's arithmetic refinement identifies each finite executable operation with its rounded-real
counterpart. After a fold proof extends this correspondence to the hinge sum, the rounded-real
error bound and two triangle inequalities give the total budget.
:::

:::theorem "fp32_relu_approximation_budget" (parent := "proof_applications") (lean := "TorchLean.UniversalApproximation.relu_mlp_binary_three_term")
Assuming real approximation, parameter quantization, and
execution budgets for a two-layer ReLU network over a configured binary format, the total
error is bounded by their sum. This includes {uses "executable_binary32"}[binary32 execution].
:::

:::proof "fp32_relu_approximation_budget"
After interpreting the configured binary result as a real value, two triangle
inequalities split the target error into the three assumed budgets.
:::

:::theorem "ode_extended_enclosure" (parent := "proof_applications") (lean := "TorchLean.Examples.ODE.extended_solution_of_clamped")
A comparison argument encloses a clamped scalar ODE solution with constant extension outside the
integration interval.
:::

:::proof "ode_extended_enclosure"
The proof combines the in-interval differential inequality with the two constant-extension cases.

Among the hypotheses are the initial value between the walls, continuity and one-sided derivative
bounds up to the horizon, and inward-pointing inequalities at the frozen walls afterward. Besides
enclosing the supplied clamped solution, the result shows that it satisfies the original ODE. Once
the solution lies between the walls, clamping its value has no effect on the vector field.
:::

:::theorem "camera_box_checker_sound" (parent := "proof_applications") (lean := "TorchLean.Examples.Geometry3D.checkCert_sound")
Acceptance by the boolean 3D box and camera checker over
{uses "shape_indexed_tensors"}[shape-indexed inputs] proves `Geometry3D.Verified` for the
certificate.
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
