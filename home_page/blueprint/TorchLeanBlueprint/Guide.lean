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
import TorchLeanBlueprint.Guide.Ch2_Frontend.RunConfiguration
import TorchLeanBlueprint.Guide.Ch2_Frontend.CustomComputations
import TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection
import TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough
import TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels
import TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd
import TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip
import TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR
import TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer
import TorchLeanBlueprint.Guide.Ch3_Backend.Operators
import TorchLeanBlueprint.Guide.Ch3_Backend.Sequential
import TorchLeanBlueprint.Guide.Ch3_Backend.Floats
import TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature
import TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA
import TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI
import TorchLeanBlueprint.Guide.Ch4_Verification.Verification
import TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems
import TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs
import TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation
import TorchLeanBlueprint.Guide.Ch4_Verification.AttentionRoundoff
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

/-! The TorchLean guide: the book that includes every chapter and the theorem index. -/

#doc (Manual) "TorchLean" =>
%%%
shortTitle := "TorchLean"
tag := "torchlean"
%%%

Suppose a model gives you a surprising prediction. Was it the weights, the input normalization,
the attention mask, or the way a reduction rounded on the GPU? A graph with the right shapes can
still compute the wrong function. This guide is about following the calculation far enough to
tell which claim you can actually make about it.

The guide begins with a two-layer network small enough to calculate by hand: run it, train it, and
inspect the tensors that change. That leads to harder questions. Which inputs can affect one
output? Does a property hold over a whole region, rather than the few points we tested? How much
can the answer move when arithmetic rounds? The tensor program and those questions can live
together in Lean. Models that start in Python can enter through the supported
[JAX or PyTorch capture path](https://lean-dojo.github.io/TorchLean/examples/verify/), after which
their graphs can be inspected here.

All commands are run from the repository root. Readers new to Lean may also use
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/).

# Introduction

What does it mean to verify a *trained* network? Before training, we know its layers and tensor
shapes. After training, we also have particular weights. The layer shapes tell us which operations
fit together; an output bound depends on the weights we actually saved. The first network remains
small enough to see both kinds of information on the page.

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Overview}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample}


# Building Models

At the held-out input $`(0.25,-0.75)`, the network in our {ref "running-example"}[running example]
initially predicts `-0.047640` for a target of `0.2`. After 200 Adam updates, we compare the
trained prediction with the same target. Throughout, the model keeps two linear layers, eight
hidden units, and a ReLU between them. What changed was the set of numbers inside its four
parameter tensors. Understanding how those numbers determine a function (and how a training program
changes them) gives us a way to read more than the final loss in a log.

Write the parameters as $`\theta=(W_1,b_1,W_2,b_2)`. For an input $`x`, the network computes

$$`
\begin{aligned}
h &= \operatorname{ReLU}(W_1x+b_1),\\
f_\theta(x) &= W_2h+b_2.
\end{aligned}
`

Each of the eight hidden units takes a weighted sum of the two input coordinates, adds a bias, and
replaces a negative result with zero. Over the input plane, a nonzero weight row and its bias
define a line where that unit switches on. Changing the weight row can rotate the line, while
changing the bias shifts it. Each unit's output weight controls its contribution on the active
side. That's how a small ReLU network can represent a surface with several slopes. Our regression
target has precisely this structure: its slope changes along $`x_1+x_2=0` and $`x_2-x_1=0`. From
the training data we get sampled values of that surface, and the optimizer uses the errors at those
samples to adjust the network.

This calculation determines the parameter layout. $`W_1` has shape `[8, 2]`, $`b_1` has shape
`[8]`, $`W_2` has shape `[1, 8]`, and $`b_2` has shape `[1]`: 33 scalar parameters in all. In Lean
these dimensions appear in the tensor types: the first layer produces eight values, ReLU preserves
that shape, and the second layer consumes eight values. Connecting those eight values to a layer
that expects seven is a type error at the model definition. Once those dimensions agree, we can
change the parameter values throughout training and keep the same interfaces between layers.

Our data has a corresponding structure. In the quickstart, 25 input pairs sit in a tensor of shape
`[25, 2]` and their targets in a tensor of shape `[25, 1]`, and each training item is one input of
shape `[2]` paired with one target of shape `[1]`. That leading `25` counts the available examples,
but it doesn't mean that every update uses all of them. Our recorded run takes one example per
update (a batched version carries a batch dimension through the model as well). Keeping that
distinction visible matters when we compare a per-step loss with a mean over the whole dataset, or
compare runs that process different numbers of examples.

For one input and target, mean squared error reduces to the squared difference between the
network's single output and the target. That scalar connects a prediction to every parameter that
contributed to it. Reverse mode propagates the loss sensitivity through the output layer, the ReLU,
and the input layer, producing four gradient tensors with the same shapes as the parameters. Adam
uses those gradients together with its running first and second moments to compute an update. We
evaluate the next example with the updated parameter values, so even a repeated input can produce a
different prediction.

The model definition specifies the layer composition and parameter layout. Initialization supplies
a starting state, the dataset supplies examples, and the objective and optimizer specify how
learning proceeds.
`Trainer.create` brings these choices together, and `trainer.train` returns a result whose learned
parameters we can inspect, save, and use for prediction. That separation pays off as soon as we
change an experiment: trying another optimizer can keep the model and initialization, while loading
a checkpoint supplies a particular parameter state for evaluation. We write those relationships out
as code in the tensor and model definitions below.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI}


# Runtime, Autograd, and Interop

Take a ReLU that received `-2` on the forward pass. Its backward rule needs that value: the
gradient through this unit is zero. The tape remembers it. Adam remembers different things, such
as its running moments, between training steps. Losing either piece of state changes the next
calculation.

The same calculation can run on a CPU or GPU, use a custom tensor function, or load weights from
PyTorch. The next pages follow where each value and operation goes.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.RunConfiguration}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.CustomComputations}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BackendSelection}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ScientificForwardModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip}


# Semantics and Graphs

On paper, a linear layer computes a dot product and adds a bias. In a graph, we must also say which
earlier node supplies each input. Swapping the weight and bias references, or dropping the bias,
can leave a graph that still passes a shape check. How do we know which function that graph means?

The formula, operator rules, and graph trace describe the same layer at three levels.
`IR.lowerForward_eq_ok` connects the lowered graph to the typed graph that produced it. A captured
Python model brings another question: did capture preserve the Python calculation?

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.Operators}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.Sequential}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR}


# Floating Point and Native Boundaries

Try adding `2²⁵ + 1 - 2²⁵` in binary32. If you add from left to right, the `1` disappears when
it meets `2²⁵`. The result is zero, although the real-number sum is one. A GPU reduction can
change where additions happen, so an error bound has to account for its arithmetic order.

Small FloatLib calculations expose the scalar rounding results before the same tensor operations
reach the CUDA path. Dtype, layout, and reduction order enter at different
points, and we need to know which one a numerical claim uses.

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
each output. Interval and affine bounds establish such properties over whole input regions, while
lowering proofs and numerical error bounds justify transferring a claim between representations or
arithmetic models. Derivatives and optimizer updates raise the same need to state what is
preserved.

What makes the claim demanding is the quantifier `∀ x ∈ B`. Sampling the box gives examples of
behavior, while an enclosure gives a bound that applies throughout it under the enclosure theorem's
hypotheses. If the chosen score has lower bound `L` and its competitor has upper bound `U`, then
`L > U` settles that comparison. But a loose enclosure may fail to separate the scores even when
the classifier is stable, which is why the choice of interval or affine propagation affects what
can be established.

A certificate records evidence used in such an argument, and its contents determine what a checker
can justify: a collection of claimed margins, a replayable trace of intermediate bounds, and a Lean
value carrying enclosure proofs support different conclusions. We develop the score comparison
first in the classifier chapter, then read the relevant theorem signatures in terms of that
example. After that, the certificate chapters examine the exported data and the precise conditions
needed to turn local bounds into a claim about the whole region. For attention, one chapter proves
that FlashAttention's online softmax computes the two-pass value over the reals, bounds its
rounding error in every IEEE format, and bounds the error of a transformer block.

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Verification}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.ProofSystems}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.AutogradProofs}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.RuntimeApproximation}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.AttentionRoundoff}

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

In a language model, token five should not read token six when predicting token six. An attention
mask enforces that rule, and one wrong comparison in the mask can leak the future while every
tensor still has the expected shape. Similar details matter in a diffusion sampler's time step
and a Fourier model's frequency layout.

These examples keep the model code close to the question. Each pairs a training or application run
with a calculation small enough to check by hand, such as which attention entries a mask zeros or
what a sampler does with an exact noise prediction.

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

An enclosure for a graph depends on smaller results. A matrix product has its own
interval rule, and a ReLU has another. Follow those two rules to the theorem that combines
them for the whole graph.

The map follows links at the level of Lean declarations: move from a theorem to the statements it
uses, then open their source. For the imports between
whole Lean files, use the site's Graphs tab.

{include 2 TorchLeanBlueprint.FormalizationMap.Foundations}

{include 2 TorchLeanBlueprint.FormalizationMap.Numerics}

{include 2 TorchLeanBlueprint.FormalizationMap.Runtime}

{include 2 TorchLeanBlueprint.FormalizationMap.Backends}

{include 2 TorchLeanBlueprint.FormalizationMap.Verification}

{include 2 TorchLeanBlueprint.FormalizationMap.Applications}

{blueprint_graph}

{blueprint_bibliography}
