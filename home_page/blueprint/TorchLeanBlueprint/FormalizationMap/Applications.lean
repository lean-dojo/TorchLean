import Verso
import VersoManual
import VersoBlueprint
import NN.API
import NN.API.Models
import NN.Spec.Models
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
import NN.Proofs.Probability.DiffusionForward
import NN.MLTheory.Proofs.Hopfield
import NN.Proofs.RL.MarkovMDP
import NN.Proofs.Models.Attention.CausalMask
import NN.MLTheory.Proofs.StateSpace.MambaCausality
import NN.MLTheory.SelfSupervised.VICReg

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Models and Mathematical Examples" =>

The model API provides seeded architecture builders, while the proof library also studies compact
mathematical models directly. The examples include an implemented MLP, reference
specifications, and theorem families for approximation, diffusion, associative memory,
reinforcement learning, attention, state-space models, and self-supervised objectives.

The links below identify the model semantics used by each result. An approximation theorem
constructs parameters, a transition theorem describes a probability law, and a causality theorem
compares executions with shared prefixes. Their conclusions answer different questions even when
the surrounding application uses all three ideas.

:::group "model_api"
Seeded construction of executable models.
:::

:::definition "seeded_builder_state" (parent := "model_api") (lean := "TorchLean.nn.Builder")
`nn.Builder` is the state monad used by seeded builders. It threads an explicit `SeedStream` through
a pure construction and yields the same result whenever `nn.build` starts from the same seed.
:::

:::definition "seeded_relu_mlp" (parent := "model_api") (lean := "TorchLean.nn.mlp")
This {uses "seeded_builder_state"}[seeded builder] constructs one batched, single-hidden-layer
MLP: a linear layer, ReLU, and a second linear layer.
:::

:::group "model_specs"
Reference definitions used by the mathematical examples below.
:::

:::definition "knn_model_container" (parent := "model_specs") (lean := "Spec.KNN")
A k-nearest-neighbor model stores $`k` together with a list of
{uses "shape_indexed_tensors"}[fixed-length feature vectors] and their labels or regression targets.
The structure itself does not assert any statistical property.
:::

:::definition "hopfield_asynchronous_update" (parent := "model_specs") (lean := "Spec.Hopfield.updateAt")
A Hopfield state is a Boolean vector. Given a weight matrix and one threshold per coordinate, this
update applies the threshold rule at one index, with ties sent to the active state.
:::

:::definition "hopfield_energy" (parent := "model_specs") (lean := "Spec.Hopfield.energy")
Over a field, Hopfield energy is the usual quadratic weight term plus the linear threshold term on
the Boolean state's $`\{-1,1\}` encoding.
:::

:::group "model_theorems"
Representative mathematical results attached to model semantics.
:::

:::theorem "real_relu_universal_approximation" (parent := "model_theorems") (lean := "NN.MLTheory.Proofs.UniversalApproximation.relu_universal_approximation_Icc")
Let $`a<b` and $`L>0`. If a real function is $`L`-Lipschitz on $`[a,b]`, then for every
positive error tolerance there is a two-layer ReLU MLP that stays within that tolerance throughout
the interval. This uses the real instance of {uses "scalar_context"}[the scalar context].
:::

:::proof "real_relu_universal_approximation"
A uniform grid gives a piecewise-linear approximation. Its slope changes become shifted ReLU
hinges, and the {uses "shape_indexed_tensors"}[shape-indexed tensor semantics] show that the
explicit two-layer MLP computes the resulting hinge sum exactly. The hidden units encode changes
of slope, so their number depends on the grid needed for the requested tolerance. This constructs
suitable parameters; it does not say that a particular training run finds them.
:::

:::definition "forward_diffusion_kernel" (parent := "model_theorems") (lean := "NN.Proofs.Probability.forwardKernel")
A forward diffusion step scales the current value and adds independent Gaussian noise according to
the schedule.
:::

:::theorem "forward_noising_is_gaussian" (parent := "model_theorems") (lean := "NN.Proofs.Probability.isGaussian_forwardKernel")
On a finite-dimensional real inner-product space with its Borel structure, each measure returned
by the {uses "forward_diffusion_kernel"}[forward diffusion kernel] is Gaussian.
:::

:::proof "forward_noising_is_gaussian"
The {uses "forward_diffusion_kernel"}[forward kernel] is identified with the affine image of a
standard Gaussian, and affine images preserve Gaussianity. The state is held fixed when describing
this transition law. The result explains the forward corruption distribution used in diffusion;
it does not identify the law produced by a learned reverse denoiser.
:::

:::theorem "hopfield_cycle_progress" (parent := "model_theorems") (lean := "NN.MLTheory.Proofs.Hopfield.cycleUpdate_progress")
For symmetric Hopfield weights with a zero diagonal, a state-changing sweep of
{uses "hopfield_asynchronous_update"}[asynchronous coordinate updates] either lowers
{uses "hopfield_energy"}[the energy] or leaves it unchanged while strictly increasing the number of
active coordinates.
:::

:::proof "hopfield_cycle_progress"
The proof folds the one-coordinate {uses "hopfield_energy"}[energy] inequalities over
{uses "hopfield_asynchronous_update"}[the updates] in one sweep. When the state changes without
lowering energy, the tie rule forces the active-coordinate count to increase. Energy alone would
allow a state-changing step on a flat level set. Counting active coordinates supplies the extra
progress needed to rule out returning around such a level set.
:::

:::theorem "hopfield_has_no_nontrivial_cycles" (parent := "model_theorems") (lean := "NN.MLTheory.Proofs.Hopfield.cycleUpdate_no_nontrivial_cycles")
Under the same symmetry and zero-diagonal assumptions, the
{uses "hopfield_cycle_progress"}[cycle progress theorem] shows that any state lying on a
positive-period cycle is already fixed by one full sweep.
:::

:::proof "hopfield_has_no_nontrivial_cycles"
By {uses "hopfield_cycle_progress"}[cycle progress], a non-fixed first step would force
lexicographic progress in energy and active-coordinate count. That progress cannot return to its
starting value after finitely many sweeps.
:::

:::definition "bellman_value_distance" (parent := "model_theorems") (lean := "Proofs.RL.Markov.valueSupDist")
The distance between two real value functions is the supremum of their pointwise absolute
differences.
:::

:::theorem "bellman_optimality_contraction" (parent := "model_theorems") (lean := "Proofs.RL.Markov.bellmanOptimality_contraction")
For a valid MDP with a nonempty state space and a finite nonempty action space, the Bellman
optimality operator contracts {uses "bellman_value_distance"}[the sup distance] between bounded
measurable value functions by the discount factor.
:::

:::proof "bellman_optimality_contraction"
After unfolding {uses "bellman_value_distance"}[the sup distance], the maximum over actions is
nonexpansive, while integration scales the remaining pointwise difference by the discount factor.
:::

:::theorem "bellman_fixed_point_unique" (parent := "model_theorems") (lean := "Proofs.RL.Markov.bellmanOptimality_fixedPoint_unique")
For the same class of MDPs, {uses "bellman_optimality_contraction"}[the contraction bound] shows
that bounded measurable fixed points of the Bellman optimality operator coincide.
:::

:::proof "bellman_fixed_point_unique"
Applying {uses "bellman_optimality_contraction"}[the Bellman contraction] to two fixed points gives
$`d\le\gamma d`. Since $`d\ge0` and $`\gamma<1`, their sup distance is zero,
forcing pointwise equality. Boundedness makes the distance a usable finite quantity, and the
strict discount bound excludes the case where the inequality merely says $`d\le d`. The result
identifies a fixed point uniquely if one exists; it does not construct one here.
:::

:::theorem "causal_mask_blocks_future" (parent := "model_theorems") (lean := "NN.Proofs.Models.Attention.causalMask_blocks_future")
For attention over {uses "shape_indexed_tensors"}[shape-indexed tensors], the causal mask marks
every strict-future key as blocked.
:::

:::proof "causal_mask_blocks_future"
The {uses "shape_indexed_tensors"}[typed mask] is read at the two indices, and the result follows
from the comparison used to construct that entry.
:::

:::theorem "masked_attention_future_zero" (parent := "model_theorems") (lean := "NN.Proofs.Models.Attention.hardMaskedSoftmaxSpec_causal_future_zero")
Combined with {uses "causal_mask_blocks_future"}[the causal-mask result], exact hard-masked softmax
assigns zero weight to every strict-future position.
:::

:::proof "masked_attention_future_zero"
The {uses "causal_mask_blocks_future"}[causal-mask theorem] selects the blocked branch, whose output
is definitionally zero. The mask has shape $`[n,n]`, with rows representing queries and columns
representing keys. Zero strict-future weights exclude those values from this attention mixture;
causality of a complete model also depends on its other operations.
:::

:::theorem "mamba_prefix_causality" (parent := "model_theorems") (lean := "NN.MLTheory.StateSpace.compactMamba_runArray_append_outputs_prefix")
For compact Mamba runs over {uses "shape_indexed_tensors"}[shape-indexed tensors], appending later
inputs does not change the earlier output prefix.
:::

:::proof "mamba_prefix_causality"
The generic array-scan prefix theorem shows that appending later inputs leaves every earlier step
and state unchanged. The initial recurrent state is shared by both runs, so the comparison isolates
appending inputs rather than restarting from a different memory. This is the property needed to
reuse an already computed prefix when reasoning about a sequential implementation.
:::

:::definition "vicreg_variance_term" (parent := "model_theorems") (lean := "NN.MLTheory.SelfSupervised.varianceTerm")
This finite analogue of the VICReg variance term sums the natural-number hinge
$`\gamma-v` over already-computed coordinate summaries.
:::

:::theorem "vicreg_variance_penalizes_collapse" (parent := "model_theorems") (lean := "NN.MLTheory.SelfSupervised.varianceTerm_collapsed_positive")
If $`\gamma` is positive, then {uses "vicreg_variance_term"}[the variance term] is positive on any
nonempty list in which every coordinate summary is zero. This finite arithmetic statement does
not claim that an optimizer avoids collapse.
:::

:::proof "vicreg_variance_penalizes_collapse"
The {uses "vicreg_variance_term"}[variance sum] on $`d+1` zeros is rewritten as
$`(d+1)\gamma`, whose factors are both positive. The nonempty condition matters because an empty
summary has zero sum. Positivity assigns a cost to collapse; excluding a collapsed optimum would
also require an attainable objective value below that cost and a link to the actual summaries.
:::
