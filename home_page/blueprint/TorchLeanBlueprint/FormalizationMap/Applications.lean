import Verso
import VersoManual
import VersoBlueprint
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.NN.Models.Autoencoder
import TorchLean.NN.Models.CNN.TwoBlock
import TorchLean.NN.Models.GAN
import TorchLean.NN.Models.GMM
import TorchLean.NN.Models.GCN
import TorchLean.NN.Models.HMM
import TorchLean.NN.Models.KNN
import TorchLean.NN.Models.LinearRegression
import TorchLean.NN.Models.LogisticRegression
import TorchLean.NN.Models.NaiveBayes
import TorchLean.NN.Models.PCA
import TorchLean.NN.Models.RandomForest
import TorchLean.NN.Models.Seq2Seq
import TorchLean.NN.Models.SVM
import TorchLean.NN.Models.Transformer
import TorchLean.NN.Models.VAE
import TorchLean.NN.Models.VQVAE
import TorchLean.NN.UniversalApproximation
import TorchLean.NN.Models.Diffusion.ForwardProcess.Gaussian
import TorchLean.NN.Models.Hopfield.Convergence
import TorchLean.NN.Models.Hopfield.Dynamics
import TorchLean.NN.RL.MDP.Bellman
import TorchLean.NN.Attention.CausalMask
import TorchLean.NN.StateSpace.Causality
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Models and Mathematical Examples" =>

TorchLean's model API provides seeded architecture builders, and the proof library also studies
compact mathematical models in their own right. Among the examples are an implemented MLP,
reference specifications, and theorem families for approximation, diffusion, associative memory,
reinforcement learning, attention, state-space models, and self-supervised objectives.

Each result below links to the model semantics it uses. An approximation theorem constructs
parameters, a transition theorem describes a probability law, and a causality theorem compares
executions with shared prefixes. Their conclusions answer different questions, even when the
surrounding application uses all three ideas.

:::group "model_api"
Seeded construction of executable models.
:::

:::definition "seeded_builder_state" (parent := "model_api") (lean := "TorchLean.nn.Builder")
`nn.Builder` is the state monad that seeded builders run in: it threads a `SeedStream` through a
pure construction and gives the same result whenever `nn.build` starts from the same seed.
:::

:::definition "seeded_relu_mlp" (parent := "model_api") (lean := "TorchLean.nn.mlp")
This {uses "seeded_builder_state"}[seeded builder] constructs an MLP with configured hidden widths,
activation, optional dropout, and batch shape. Each hidden stage runs a linear layer followed by
the activation and optional dropout, and a final linear layer produces the output.
:::

:::group "model_specs"
Reference definitions used by the mathematical examples below.
:::

:::definition "knn_model_container" (parent := "model_specs") (lean := "TorchLean.KNN")
A k-nearest-neighbor model stores $`k` together with an array of
{uses "shape_indexed_tensors"}[fixed-length feature vectors] and their labels or regression
targets. On its own, the structure asserts no statistical property.
:::

:::definition "hopfield_asynchronous_update" (parent := "model_specs") (lean := "TorchLean.Hopfield.updateAt")
A Hopfield state is a Boolean vector. Given a weight matrix and one threshold per coordinate, this
update applies the threshold rule at one index (ties go to the active state).
:::

:::definition "hopfield_energy" (parent := "model_specs") (lean := "TorchLean.Hopfield.energy")
Over a field, Hopfield energy is the usual quadratic weight term plus the linear threshold term on
the Boolean state's $`\{-1,1\}` encoding.
:::

:::group "model_theorems"
Representative mathematical results attached to model semantics.
:::

:::theorem "real_relu_universal_approximation" (parent := "model_theorems") (lean := "TorchLean.UniversalApproximation.relu_Icc")
Let $`a<b` and $`L>0`. If a real function is $`L`-Lipschitz on $`[a,b]`, then for every positive
error tolerance there is a two-layer ReLU MLP that stays within that tolerance throughout the
interval. (This uses the real instance of {uses "scalar_context"}[the scalar context].)
:::

:::proof "real_relu_universal_approximation"
We start from a uniform grid, which gives a piecewise-linear approximation whose slope changes
become shifted ReLU hinges, and the {uses "shape_indexed_tensors"}[shape-indexed tensor semantics]
show that the constructed two-layer MLP computes the resulting hinge sum exactly. Hidden units
encode changes of slope, so their number depends on the grid needed for the requested tolerance.
This constructs suitable parameters, but it doesn't say that a particular training run finds them.
:::

:::definition "forward_diffusion_kernel" (parent := "model_theorems") (lean := "TorchLean.Diffusion.forwardKernel")
A forward diffusion step scales the current value and adds independent Gaussian noise according to
the schedule.
:::

:::theorem "forward_noising_is_gaussian" (parent := "model_theorems") (lean := "TorchLean.Diffusion.isGaussian_forwardKernel")
On a finite-dimensional real inner-product space with its Borel structure, each measure returned
by the {uses "forward_diffusion_kernel"}[forward diffusion kernel] is Gaussian.
:::

:::proof "forward_noising_is_gaussian"
We identify the {uses "forward_diffusion_kernel"}[forward kernel] with the affine image of a
standard Gaussian (the state is held fixed when we describe this transition law), and affine images
preserve Gaussianity. This result explains the forward corruption distribution used in diffusion,
but it doesn't identify the law produced by a learned reverse denoiser.
:::

:::theorem "hopfield_cycle_progress" (parent := "model_theorems") (lean := "TorchLean.Hopfield.cycleUpdate_progress")
For symmetric Hopfield weights with a zero diagonal, a state-changing sweep of
{uses "hopfield_asynchronous_update"}[asynchronous coordinate updates] either lowers
{uses "hopfield_energy"}[the energy] or leaves it unchanged while strictly increasing the number of
active coordinates.
:::

:::proof "hopfield_cycle_progress"
We fold the one-coordinate {uses "hopfield_energy"}[energy] inequalities over
{uses "hopfield_asynchronous_update"}[the updates] in one sweep. When the state changes without
lowering energy, the tie rule forces the active-coordinate count to increase. Energy alone would
allow a state-changing step on a flat level set, but counting active coordinates supplies the extra
progress needed to rule out returning around such a level set.
:::

:::theorem "hopfield_has_no_nontrivial_cycles" (parent := "model_theorems") (lean := "TorchLean.Hopfield.cycleUpdate_eq_self_of_cycle")
Under the same symmetry and zero-diagonal assumptions, the
{uses "hopfield_cycle_progress"}[cycle progress theorem] shows that any state lying on a
positive-period cycle is already fixed by one full sweep.
:::

:::proof "hopfield_has_no_nontrivial_cycles"
By {uses "hopfield_cycle_progress"}[cycle progress], a non-fixed first step would force
lexicographic progress in energy and active-coordinate count, and that progress can't return to its
starting value after finitely many sweeps.
:::

:::definition "bellman_value_distance" (parent := "model_theorems") (lean := "TorchLean.RL.MDP.valueSupDist")
For two real value functions, the distance is the supremum of their pointwise absolute differences.
:::

:::theorem "bellman_optimality_contraction" (parent := "model_theorems") (lean := "TorchLean.RL.MDP.bellmanMax_contraction")
For an MDP with a nonempty state space and a finite nonempty action space whose transition is a
Markov kernel and whose discount is nonnegative, the Bellman optimality operator contracts
{uses "bellman_value_distance"}[the sup distance] between bounded measurable value functions by
the discount factor.
:::

:::proof "bellman_optimality_contraction"
After unfolding {uses "bellman_value_distance"}[the sup distance], the maximum over actions is
nonexpansive, while integration scales the remaining pointwise difference by the discount factor.
:::

:::theorem "bellman_fixed_point_unique" (parent := "model_theorems") (lean := "TorchLean.RL.MDP.bellmanMax_fixedPoint_unique")
For a valid MDP with a nonempty state space and a finite nonempty action space,
{uses "bellman_optimality_contraction"}[the contraction bound] shows that bounded measurable fixed
points of the Bellman optimality operator coincide.
:::

:::proof "bellman_fixed_point_unique"
Applying {uses "bellman_optimality_contraction"}[the Bellman contraction] to two fixed points gives
$`d\le\gamma d`, and since $`d\ge0` and $`\gamma<1`, their sup distance is zero, which forces
pointwise equality. Boundedness makes the distance a usable finite quantity, and the strict
discount bound excludes the case where the inequality merely says $`d\le d`. This result identifies
a fixed point uniquely if one exists, but it doesn't construct one here.
:::

:::theorem "causal_mask_blocks_future" (parent := "model_theorems") (lean := "TorchLean.Attention.get2_causalMask_eq_false_of_lt")
For attention over {uses "shape_indexed_tensors"}[shape-indexed tensors], the causal mask marks
every strict-future key as blocked.
:::

:::proof "causal_mask_blocks_future"
We read the {uses "shape_indexed_tensors"}[typed mask] at the two indices, and the result follows
from the comparison that built that entry.
:::

:::theorem "masked_attention_future_zero" (parent := "model_theorems") (lean := "TorchLean.Softmax.hardMasked_causalMask_eq_zero")
Combined with {uses "causal_mask_blocks_future"}[the causal-mask result], exact hard-masked softmax
assigns zero weight to every strict-future position.
:::

:::proof "masked_attention_future_zero"
By the {uses "causal_mask_blocks_future"}[causal-mask theorem] the blocked branch is selected, and
its output is definitionally zero. The mask has shape $`[n,n]` (rows represent queries and columns
represent keys). Zero strict-future weights exclude those values from this attention mixture, but
causality of a complete model also depends on its other operations.
:::

:::theorem "mamba_prefix_causality" (parent := "model_theorems") (lean := "TorchLean.Mamba.CompactBlock.runArray_causal")
For compact Mamba runs over {uses "shape_indexed_tensors"}[shape-indexed tensors], appending later
inputs does not change the earlier output prefix.
:::

:::proof "mamba_prefix_causality"
We use the generic array-scan prefix theorem: appending later inputs leaves every earlier step and
state unchanged. Both runs share the initial recurrent state, so the comparison isolates appending
inputs (neither run restarts from a different memory). That's the property needed to reuse an
already computed prefix when reasoning about a sequential implementation.
:::
