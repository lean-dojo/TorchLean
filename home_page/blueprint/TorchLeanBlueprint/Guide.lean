import VersoManual
import VersoBlueprint
import VersoBlueprint.Commands.Graph
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.FormalizationMap
import TorchLeanBlueprint.Guide.Ch1_Introduction.Overview
import TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation
import TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour
import TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming
import TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage
import TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch
import TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample
import TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes
import TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels
import TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders
import TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch
import TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI
import TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes
import TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection
import TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough
import TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels
import TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd
import TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip
import TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR
import TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer
import TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec
import TorchLeanBlueprint.Guide.Ch3_Backend.Floats
import TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature
import TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA
import TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI
import TorchLeanBlueprint.Guide.Ch4_Verification.Verification
import TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems
import TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs
import TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation
import TorchLeanBlueprint.Guide.Ch4_Verification.LearningTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.OptimizationTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.SelfSupervisedTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.ApproximationTheory
import TorchLeanBlueprint.Guide.Ch4_Verification.ClassicalMLProofs
import TorchLeanBlueprint.Guide.Ch4_Verification.ProbabilityAndGradients
import TorchLeanBlueprint.Guide.Ch4_Verification.ScientificMLVerification
import TorchLeanBlueprint.Guide.Ch4_Verification.FactorizationsCholeskyQR
import TorchLeanBlueprint.Guide.Ch4_Verification.Certificates
import TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness
import TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows
import TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels
import TorchLeanBlueprint.Guide.Ch5_Applications.ModelExamplesDeepDive
import TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels
import TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning
import TorchLeanBlueprint.Guide.Ch5_Applications.Widgets
import TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog
import TorchLeanBlueprint.Guide.Ch5_Applications.Examples
import TorchLeanBlueprint.Guide.Ch5_Applications.CLI
import TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion

open Verso.Genre Manual
open Informal

#doc (Manual) "TorchLean" =>
%%%
shortTitle := "TorchLean"
tag := "torchlean"
%%%

TorchLean is a Lean 4 library for writing, training, and reasoning about neural networks. Tensor
dimensions appear in types, model definitions are executable, and supported programs can be
recorded as graphs for differentiation, export, and verification.

The running example is a nonlinear regression model with two linear layers and a ReLU. We train it,
inspect its parameters and gradients, run it through different numerical backends, and state exact
claims about the corresponding mathematical definitions. The distinction matters: a theorem about
real-valued matrix multiplication does not by itself verify the floating-point instructions issued
by a GPU kernel.

The same questions recur in larger models. An attention mask changes which tokens can influence
a prediction; a reduction order changes where arithmetic rounds; a certificate needs to identify
the weights and input region it covers. We will follow these connections from small calculations
to transformers, ResNets, Fourier neural operators, diffusion, and reinforcement learning.

All commands are run from the repository root. Readers new to Lean may also use
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/).

# Introduction

An architecture fixes how layers connect, but many different functions share that architecture.
The weight values select one of them. A training run changes those values while retaining the
same layer interfaces; saving a checkpoint records one particular state of that process.
When we later ask Lean to prove something about a model, the statement must identify which of
these objects it concerns. A shape theorem may apply to every parameter state, while an output
bound will usually depend on the weights.

The running example keeps these distinctions visible in a small program. Its input has two
coordinates and its output has one, so the forward map can be written out and compared with the
layer definition. The same program supplies concrete tensors, gradients, and loss values for
reading the API. Lean syntax enters as a way to express those objects and their relationships.

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Overview}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample}


# Building Models

At the held-out input $`(0.25,-0.75)`, the network in our
{ref "running-example"}[running example] initially predicts `-0.088261`. The target is `0.2`.
After 200 Adam updates, its prediction is `0.228325`. The network still has the same two linear
layers, the same eight hidden units, and the same ReLU between them. What changed was the set of
numbers inside its four parameter tensors. Understanding how those numbers determine a function,
and how a training program changes them, gives us a way to read more than the final loss in a log.

Write the parameters as $`\theta=(W_1,b_1,W_2,b_2)`. For an input $`x`, the network computes

$$`
\begin{aligned}
h &= \operatorname{ReLU}(W_1x+b_1),\\
f_\theta(x) &= W_2h+b_2.
\end{aligned}
`

Each of the eight hidden units takes a weighted sum of the two input coordinates, adds a bias, and
replaces a negative result with zero. Over the input plane, a nonzero weight row and its bias define
a line where that unit switches on. Changing the weight row can rotate the line, while changing the
bias shifts it. The unit's output weight controls its contribution on the active side. This is how
a small ReLU network can represent a surface with several slopes. Our regression target has
precisely this structure: its slope changes along $`x_1+x_2=0` and $`x_2-x_1=0`. The training data
gives us sampled values of that surface, and the optimizer uses the errors at those samples to
adjust the network.

The parameter layout follows directly from this calculation. $`W_1` has shape `[8, 2]`, $`b_1`
has shape `[8]`, $`W_2` has shape `[1, 8]`, and $`b_2` has shape `[1]`: 33 scalar parameters in
all. In Lean, these dimensions appear in the tensor types. The first layer produces eight values,
ReLU preserves that shape, and the second layer consumes eight values. Connecting it to a layer
that expects seven is a type error at the model definition. Once those dimensions agree, we can
change the parameter values throughout training while preserving the same interfaces between
layers.

The data has a corresponding structure. The quickstart stores 25 input pairs in a tensor of shape
`[25, 2]` and their targets in a tensor of shape `[25, 1]`. Each training item is one input of shape
`[2]` paired with one target of shape `[1]`. The leading `25` counts the available examples; it does
not mean that every update uses all of them. The recorded run takes one example per update. A
batched version carries a batch dimension through the model as well. Keeping that distinction
visible matters when we compare a per-step loss with a mean over the whole dataset, or compare
runs that process different numbers of examples.

For one input and target, mean squared error reduces to the squared difference between the
network's single output and the target. That scalar connects a prediction to every parameter that
contributed to it. Reverse mode propagates the loss sensitivity through the output layer, the
ReLU, and the input layer, producing four gradient tensors with the same shapes as the parameters.
Adam uses those gradients together with its running first and second moments to compute an
update. The next example is evaluated with the new parameter values, so even a repeated input can
produce a different prediction.

TorchLean's interfaces let us inspect each of these objects separately. The model definition
specifies the layer composition and parameter layout. Initialization supplies a starting state;
the dataset supplies examples; the objective and optimizer specify how learning proceeds.
`Trainer.new` brings these choices together, and `trainer.train` returns a result whose learned
parameters we can inspect, save, and use for prediction. This separation becomes useful as soon as
we change an experiment: trying another optimizer can preserve the model and initialization,
while loading a checkpoint supplies a particular parameter state for evaluation. The tensor and
model definitions below make those relationships explicit in the code.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI}


# Runtime, Autograd, and Interop

Training changes the parameters in the forward map

$$`f_\theta(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2`.

To compute an update, reverse mode needs intermediate values from the forward pass, and the
optimizer may need state from earlier updates. Eager execution and typed graph execution retain
that information differently. CUDA and LibTorch add a choice of implementation for the numerical
operations.

At the ReLU, reverse mode needs to know which pre-activations were positive. At a linear layer,
it needs the input to form the weight gradient and the weight matrix to propagate sensitivity
back toward the input. A forward pass therefore leaves information that a later backward pass
will use. Execution choices affect where that information lives, how long it stays available,
and which operations can consume it.

The scalar arithmetic and the execution strategy are separate choices. Recording a typed graph
does not by itself choose CUDA, and moving tensors to a device does not change the mathematical
definition of the loss. Interoperation adds another concrete question: whether the imported
weights, layouts, and operation conventions describe the same model. Following one forward
and backward calculation through these interfaces makes their roles easier to distinguish.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip}


# Semantics and Graphs

Verification needs a definition of what a model computes for every input covered by the claim.
The specification layer gives that mathematical function. `GraphSpec` records its architecture
with shapes in the types, while the shared IR uses a node array that importers and verification
passes can inspect.

For a linear layer, the specification can say directly that each output is a dot product plus
a bias. A graph must also say where the input came from, which parameter entry supplies the
weights, and which node receives the result. The added structure lets an interpreter execute
the program and lets a verifier inspect intermediate values. It also creates facts that need
checking: a parent index must name an available node, and the chosen weights must fit the
declared dimensions.

These representations support different tasks. Shape-indexed composition makes incompatible
layer interfaces difficult to express. A node array makes imported structure and local
transformations explicit. The useful relationship is a theorem that evaluating the translated
representation preserves the specified function. That is what lets a later bound on a graph
refer back to the model from which it was built.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR}


# Floating Point and Native Boundaries

A dot product looks like a sum of products on paper. To run it, we also choose a number format
and an order of evaluation. Products and partial sums may round, so two implementations of the
same real formula can return different values.

In TorchLean, the scalar type selects a format from FloatLib. We will use binary32 for a first
example, then change the precision through the same interface. Custom exponent and fraction widths
are available on the typed CPU path; CUDA providers support native binary32 and binary64.
The useful question is what a precision change preserves in the actual calculation.

An error bound adds another step. We need to relate the rounded operations to their real-valued
specification, including what happens near zero, at overflow, and on exceptional inputs.
FloatLib supplies the scalar arithmetic and rounding theorems. TorchLean carries the relevant
bounds through tensor operations and records the numerical choices of native providers.
A bound for one accumulation order applies to a kernel only once that kernel's order is accounted
for.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.Floats}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness}


# Verification and Certificates

Training tells us how a model behaves on the examples it sees. A verification claim can ask
about every input in a region, including points never evaluated during training. For a
classifier, a useful first property is that the predicted label stays fixed when each input
coordinate changes by at most a stated amount. We can express this through the output scores:
the chosen class must remain above every competitor.

For a graph $`g`, parameter payload $`\theta`, and input region $`B`, a verification claim has the
form

$$`\forall x\in B,\qquad P(\operatorname{denote}(g,\theta,x))`.

Here $`\operatorname{denote}` is the chosen graph semantics and $`P` is the property required of
each output. Interval and affine bounds establish such properties over whole input regions.
Lowering proofs and numerical error bounds justify transferring a claim between representations
or arithmetic models. The same need to state what is preserved arises for derivatives and
optimizer updates.

The quantifier `∀ x ∈ B` is the demanding part. Sampling the box gives examples of behavior;
an enclosure gives a bound that applies throughout it under the enclosure theorem's hypotheses.
If the chosen score has lower bound `L` and its competitor has upper bound `U`, then `L > U`
settles that comparison. A loose enclosure may fail to separate the scores even when the
classifier is stable, which is why the choice of interval or affine propagation affects what
can be established.

A certificate records evidence used in such an argument. Its contents determine what a checker
can justify: a collection of claimed margins, a replayable trace of intermediate bounds, and a
Lean value carrying enclosure proofs support different conclusions. The classifier chapter
develops the score comparison first, then reads the relevant theorem signatures in terms of
that example. The later certificate chapters examine the exported data and the precise
conditions needed to turn local bounds into a claim about the whole region.

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Verification}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.LearningTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.OptimizationTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.SelfSupervisedTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ApproximationTheory}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ClassicalMLProofs}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ProbabilityAndGradients}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ScientificMLVerification}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FactorizationsCholeskyQR}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Certificates}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows}


# Examples and Applications

A ResNet adds spatial layouts and skip connections to the operations used by the MLP. GPT adds
token streams and causal masks. Fourier neural operators connect learned maps to PDE data, while
diffusion and reinforcement learning introduce probabilistic transitions and evolving state.
We can still trace a prediction through the familiar layer operations, provided we also account
for these additional choices.

The extra structure changes what we need to inspect. For attention, two tensors can have the
expected dimensions while using the wrong causal mask. For a Fourier layer, a transform convention
or selected mode range affects the represented operator. A diffusion step depends on a time
index and noise schedule as well as learned weights. The examples keep these choices close to
the code and outputs so that the familiar layer operations remain connected to the actual
model being run.

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModernModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ModelExamplesDeepDive}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.GenerativeModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.ReinforcementLearning}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Widgets}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.BugZooCatalog}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.Examples}

{include 2 TorchLeanBlueprint.Guide.Ch5_Applications.CLI}

{include 2 TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion}


# Formalization Map

The chapters below trace the main definitions, executable boundaries, and proved results through
TorchLean. This is a declaration-level map: an edge records a mathematical or implementation
dependency between selected entries, and theorem proof dependencies are tracked separately from
statement dependencies. The website's Graphs tab remains the place to inspect Lean module
imports.

The selection favors declarations that explain an entire subsystem. Small helper lemmas stay in the
API reference, where their full statements and source locations are easier to inspect.

Use a map entry after the corresponding explanation has given its declaration a meaning. An
enclosure theorem, for example, depends on local transfer laws and hypotheses about the graph;
following those entries shows where its premises come from. A dependency edge alone does not
say that a runtime path satisfies those premises. The entry's statement and explanatory text
identify the objects being related, while the source link leads to the actual definition or
proof.

{include 2 TorchLeanBlueprint.FormalizationMap.Foundations}

{include 2 TorchLeanBlueprint.FormalizationMap.Numerics}

{include 2 TorchLeanBlueprint.FormalizationMap.Runtime}

{include 2 TorchLeanBlueprint.FormalizationMap.Backends}

{include 2 TorchLeanBlueprint.FormalizationMap.Verification}

{include 2 TorchLeanBlueprint.FormalizationMap.Applications}

{blueprint_graph}

{blueprint_bibliography}
