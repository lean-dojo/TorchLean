import VersoManual
import VersoBlueprint
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
import TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough
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
import TorchLeanBlueprint.Guide.Ch4_Verification.Certificates
import TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness
import TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows
import TorchLeanBlueprint.Guide.Ch5_Advanced.ModernModels
import TorchLeanBlueprint.Guide.Ch5_Advanced.ModelZooDeepDive
import TorchLeanBlueprint.Guide.Ch5_Advanced.GenerativeModels
import TorchLeanBlueprint.Guide.Ch5_Advanced.ReinforcementLearning
import TorchLeanBlueprint.Guide.Ch5_Advanced.Widgets
import TorchLeanBlueprint.Guide.Ch5_Advanced.BugZooCatalog
import TorchLeanBlueprint.Guide.Ch5_Advanced.Examples
import TorchLeanBlueprint.Guide.Ch5_Advanced.CLI
import TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion

open Verso.Genre Manual
open Informal

#doc (Manual) "TorchLean Guide" =>
%%%
shortTitle := "TorchLean"
tag := "torchlean"
%%%

TorchLean is a Lean 4 library for neural network code that can be run, inspected, lowered into a
shared graph IR, and connected to explicit mathematical contracts. A model can appear as ordinary
user code, as a graph, as a runtime computation, and as a verification target without changing which
mathematical object the project is talking about.

This guide is the readable path through that stack. It starts with the reason TorchLean exists,
then follows one model through typed tensors, training, runtime execution, graph semantics,
Float32/native boundaries, verification, examples, and the concluding summary.

The generated [API docs](../docs/) are still the best place to look up an exact declaration. The
[module graph](../graphs/) is the best place to explore imports. The guide gives the narrative spine:
why we built the pieces, how they fit, and what kind of guarantee each layer is meant to provide.

![TorchLean guide map](Guide/Assets/torchlean-guide-map.png)

For a first pass, read Introduction → Building Models → Runtime and Interop → Semantics and Graphs.
If you are here for verification, jump from there to Verification and Certificates. If you want
concrete demos, skip ahead to Examples and Applications.

If you are new to Lean, keep the official Lean material nearby:
[*Functional Programming in Lean*](https://lean-lang.org/functional_programming_in_lean/),
[*Theorem Proving in Lean 4*](https://lean-lang.org/theorem_proving_in_lean4/), and
[*The Lean Language Reference*](https://lean-lang.org/doc/reference/latest/). We cite papers and
external tools in the chapters where they matter, instead of front-loading a bibliography before the
reader has context.

# Introduction

These chapters establish the working model for the rest of the guide. We start with the basic
question TorchLean answers: how can one codebase support neural network programs that run, graph
artifacts that tools can inspect, and mathematical claims that Lean can check?

The introduction is meant to be read linearly. It first explains the motivation, then compares the
TorchLean style with familiar PyTorch workflows, and finally follows a small classifier through the
main representations that appear later in the book.

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Overview}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.Motivation}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.API_Tour}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.WhyFunctionalProgramming}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TheLeanLanguage}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.TorchLeanVsPyTorch}

{include 2 TorchLeanBlueprint.Guide.Ch1_Introduction.RunningExample}


# Building Models

These chapters start with the public surface: tensors, shapes, model builders, datasets, and small
training loops. They are intentionally practical. They show how a TorchLean model is written and run
before asking you to care about the IR or proof machinery underneath it.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TensorsAndShapes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.BuildingModels}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.DataAndLoaders}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TrainingFromScratch}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.TorchLeanAPI}


# Runtime, Autograd, and Interop

Once the model exists, the next question is how it runs, how gradients are computed, and how
TorchLean talks to workflows that feel familiar from PyTorch. This chapter starts with the everyday
runtime choices, then opens the autograd and interop layers that support training, inspection, and
verification.

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.ExecutionModes}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.AutogradWalkthrough}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.RuntimeAndAutograd}

{include 2 TorchLeanBlueprint.Guide.Ch2_Frontend.PyTorchRoundtrip}


# Semantics and Graphs

This section explains the objects that TorchLean claims are about.

The spec layer gives mathematical meanings to tensors, layers, losses, masks, modes, and scalar
choices. GraphSpec gives a typed architecture language for models whose parameter layout and
sharing structure should be explicit. `NN.IR.Graph` gives the operation-tagged graph consumed by
widgets, exporters, runtime bridges, and verification passes.

These layers are connected, but they are not interchangeable. A spec definition is a mathematical
reference. A GraphSpec model is an architecture with a typed parameter interface. An IR graph is a
low-level artifact with op tags, parent ids, shapes, and payloads. Keeping those roles separate is
what lets TorchLean say exactly what a runtime, compiler, or verifier has checked.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphsAndIR}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.SpecLayer}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GraphSpec}


# Floating Point and Native Boundaries

Neural network claims often move across numerical worlds. A model may be specified over the reals,
executed in Float32, accelerated by CUDA, bounded by a verifier, and checked through an imported
certificate. These worlds are connected, but they are not identical.

TorchLean separates them deliberately. Real-valued specs are used for clean mathematical
statements. `FP32` gives a rounded-real proof model for error budgets. `IEEE32Exec` gives
executable binary32 behavior inside Lean, including special values. Native execution through Lean
`Float32`, CUDA, cuBLAS, cuFFT, Python, Julia, Arb, or external verifiers enters through explicit
producer/checker interfaces.

The goal is not to avoid practical numerical tools. The goal is to know exactly how their outputs
support the claim being made.

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.Floats}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.FloatingPointLiterature}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.GPUAndCUDA}

{include 2 TorchLeanBlueprint.Guide.Ch3_Backend.ExternalToolsAndFFI}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.FP32Soundness}


# Verification and Certificates

This chapter explains the verification and theorem layers of TorchLean. The word verification is
used broadly here. Sometimes it means a bound propagation certificate for one graph and one input
region. Sometimes it means a compiler theorem, an autograd correctness theorem, a finite precision
approximation theorem, a learning theory predicate, or a scientific ML certificate.

The common pattern is that every claim has an object and a support. The object may be a graph, a
parameter payload, a tensor program, a dataset, a trajectory, a residual, or an external JSON
certificate. The support may be a Lean theorem, a small checker, a replayed artifact, a runtime
diagnostic, or an explicitly named external producer.

This section has three kinds of pages. The first group explains verifier artifacts: IBP/CROWN
bounds, certificates, two-stage workflows, ODE tubes, PINN residuals, and α,β-CROWN leaves. The
second group explains proof infrastructure: compiler correctness, autograd soundness, runtime
approximation, and reusable gradient/probability lemmas. The third group explains mathematical ML
theory: differential privacy, stability, optimization, self-supervised objectives, universal
approximation, Hopfield energy, and state-space causality.

The goal is to make those levels readable. A run is evidence about execution. A certificate is a
finite artifact with a stated checker. A real-valued theorem needs a numerical bridge before it
becomes a Float32 claim. When the right bridges are present, TorchLean can turn concrete artifacts
into precise mathematical statements.

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

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.Certificates}

{include 2 TorchLeanBlueprint.Guide.Ch4_Verification.TwoStageWorkflows}


# Examples and Applications

The final section shows TorchLean in use. Earlier chapters described the semantic pieces: typed
tensors, graph IR, runtime execution, floating point models, CUDA boundaries, autograd proofs, and
verification checkers. The examples show how those pieces behave when they meet real ML shapes.

By the end of the guide, the system should not feel like a collection of isolated formalizations.
The examples range from small MLPs to GPT-style models, Mamba, ResNet, ViT, FNOs, diffusion, and
reinforcement learning because those models bring in the details that matter in practice: masks,
positions, residual branches, scan state, spectral convolutions, sampling schedules, environment
transitions, and GPU execution.

The examples are small enough to inspect, but they touch real sources of complexity. A causal
language model is not only a tensor program; it has token ids, positions, masks, and sometimes a
cache. An FNO is not only a supervised model; it has spectral structure and scientific data. An RL
example is not only a loss function; it has an environment boundary and trajectory data. BugZoo
then takes the same idea one step further by showing how common ML bugs can be turned into explicit
contracts.

The question to ask of each example is not only "did it run?" The better question is: what object
did it produce, what can we inspect, and what claim could this object support?

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.ModernModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.ModelZooDeepDive}

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.GenerativeModels}

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.ReinforcementLearning}

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.Widgets}

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.BugZooCatalog}

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.Examples}

{include 2 TorchLeanBlueprint.Guide.Ch5_Advanced.CLI}

{include 2 TorchLeanBlueprint.Guide.Ch6_Conclusion.Conclusion}
