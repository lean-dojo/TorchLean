/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lean
import TorchLean.CUDA.Kernel.Emit.Source
import TorchLean.NN.Attention.Adjoint
import TorchLean.NN.Attention.BiasedScaledDotProduct
import TorchLean.NN.Attention.PermutationEquivariance
import TorchLean.NN.Attention.ScaledDotProduct.FDeriv
import TorchLean.NN.Backend.Graph.Soundness
import TorchLean.NN.Backend.Session
import TorchLean.NN.Conv.FDeriv
import TorchLean.NN.Dual.Special
import TorchLean.NN.FDeriv.MLP
import TorchLean.NN.IR.CustomRule
import TorchLean.NN.IR.Flow.Reals
import TorchLean.NN.IR.Program.Step
import TorchLean.NN.IR.ReducePrecision
import TorchLean.NN.LearningTheory.DifferentialPrivacy
import TorchLean.NN.LoRA
import TorchLean.NN.MoE
import TorchLean.NN.Models.Autoencoder
import TorchLean.NN.Models.CNN.TwoBlock
import TorchLean.NN.Models.Classifier
import TorchLean.NN.Models.Diffusion.ForwardProcess.Gaussian
import TorchLean.NN.Models.Diffusion.ImageDDIM.Training
import TorchLean.NN.Models.Diffusion.Loss
import TorchLean.NN.Models.Diffusion.Samplers
import TorchLean.NN.Models.GAN.Loss
import TorchLean.NN.Models.GCN
import TorchLean.NN.Models.GMM
import TorchLean.NN.Models.HMM
import TorchLean.NN.Models.Hopfield.Convergence
import TorchLean.NN.Models.Hopfield.Dynamics
import TorchLean.NN.Models.KNN
import TorchLean.NN.Models.LogisticRegression
import TorchLean.NN.Models.NaiveBayes
import TorchLean.NN.Models.PCA
import TorchLean.NN.Models.RandomForest
import TorchLean.NN.Models.Recurrent.Elman
import TorchLean.NN.Models.Recurrent.Stack.Correctness
import TorchLean.NN.Models.SVM
import TorchLean.NN.Models.Seq2Seq
import TorchLean.NN.Models.Transformer
import TorchLean.NN.Models.Transformer.DecoderBlock
import TorchLean.NN.Models.Transformer.EncoderBlock
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.VAE.Elbo
import TorchLean.NN.Models.VQVAE.Loss
import TorchLean.NN.Norm.BatchNorm
import TorchLean.NN.Norm.LayerNorm.Eager
import TorchLean.NN.NormalCDF
import TorchLean.NN.Op.BatchNorm
import TorchLean.NN.Op.Complex
import TorchLean.NN.Op.ComplexRound
import TorchLean.NN.Op.Conv
import TorchLean.NN.Op.Fourier
import TorchLean.NN.Op.Frexp
import TorchLean.NN.Op.LayerNorm
import TorchLean.NN.Op.Piecewise
import TorchLean.NN.Op.Pool
import TorchLean.NN.Op.Power
import TorchLean.NN.Op.Reduce
import TorchLean.NN.Op.Rounding
import TorchLean.NN.Op.Scan
import TorchLean.NN.Op.Special
import TorchLean.NN.Optimizer.AdamW
import TorchLean.NN.Optimizer.GD.StrongConvexity
import TorchLean.NN.Optimizer.TensorOptimizer
import TorchLean.NN.RL.Bandits
import TorchLean.NN.RL.Boundary.Soundness
import TorchLean.NN.RL.DQN.SoftUpdate
import TorchLean.NN.RL.Environment.Safety
import TorchLean.NN.RL.PolicyGradient.Real
import TorchLean.NN.RL.Replay.Capacity
import TorchLean.NN.RL.StochasticMDP.Bellman
import TorchLean.NN.RL.Tabular
import TorchLean.NN.Robustness.Empirical
import TorchLean.NN.StateSpace.Causality
import TorchLean.NN.Tape.Lowering.BackwardDenseGraph
import TorchLean.NN.Tape.RealNode.MSE
import TorchLean.NN.Text.GPT2BPE.Merge
import TorchLean.NN.UniversalApproximation.CompactSet
import TorchLean.NN.UniversalApproximation.Rate
import TorchLean.NN.Widget.Docs
import TorchLean.NN.Widget.RL.PPO
import TorchLean.Tensor.AdjacentSwap.Adjoint
import TorchLean.Tensor.ConfiguredBinary.Order
import TorchLean.Tensor.Einops.Lowering.Einsum.Order
import TorchLean.Tensor.Einops.Lowering.Reduce.Mean
import TorchLean.Tensor.Factorization.RealSchur
import TorchLean.Tensor.Factorization.Solve
import TorchLean.Tensor.Layers.Normalization.GroupNorm
import TorchLean.Tensor.Layers.RealFFT.Complex
import TorchLean.Tensor.LinearAlgebra
import TorchLean.Tensor.Numeric.Frexp
import TorchLean.Tensor.Op.Bitcast
import TorchLean.Tensor.Op.Table.Compare
import TorchLean.Tensor.Op.Vocabulary
import TorchLean.Tensor.Rep.Fiber.Product.DualNumber
import TorchLean.Tensor.Rep.Matrix
import TorchLean.Tensor.Special.Bessel
import TorchLean.Tensor.Special.Fn
import TorchLean.Tensor.Special.Gamma
import TorchLean.Tensor.Special.GammaCDF
import TorchLean.Tensor.Special.Hurwitz
import TorchLean.Tensor.Special.Zeta
import TorchLean.Tensor.Tactic.Verify.Lowering
import TorchLean.Verify.Approximation.FloatInterval.ConstantTarget
import TorchLean.Verify.CROWN.Conv.Derivatives
import TorchLean.Verify.CROWN.FlatBox.Refinement
import TorchLean.Verify.CROWN.LayerNorm.DerivativeEnclosure
import TorchLean.Verify.CROWN.MLP.Distillation
import TorchLean.Verify.CROWN.Op.Alpha
import TorchLean.Verify.CROWN.Op.BackwardRun
import TorchLean.Verify.CROWN.Op.ExecCert
import TorchLean.Verify.CROWN.Op.Linear
import TorchLean.Verify.Capture.Error
import TorchLean.Verify.Certificate.CROWNNode
import TorchLean.Verify.Certificate.DecimalJSONFuel
import TorchLean.Verify.Certificate.FiniteArtifact.Semantics
import TorchLean.Verify.Coverage.Elementwise.Activation
import TorchLean.Verify.Coverage.Elementwise.Arithmetic
import TorchLean.Verify.Coverage.Elementwise.Rounding
import TorchLean.Verify.Coverage.Elementwise.Selection
import TorchLean.Verify.Coverage.Elementwise.Sigmoid
import TorchLean.Verify.Coverage.Elementwise.Transcendental
import TorchLean.Verify.Coverage.Indexing.Masking
import TorchLean.Verify.Coverage.Indexing.Movement
import TorchLean.Verify.Coverage.Indexing.Reduction
import TorchLean.Verify.Coverage.Indexing.Signal
import TorchLean.Verify.Coverage.Layers.Attention
import TorchLean.Verify.Coverage.Layers.BatchNorm
import TorchLean.Verify.Coverage.Layers.Comparison
import TorchLean.Verify.Coverage.Layers.Convolution
import TorchLean.Verify.Coverage.Layers.LayerNorm
import TorchLean.Verify.Coverage.Layers.Linear
import TorchLean.Verify.Coverage.Layers.MultiheadAttention
import TorchLean.Verify.Coverage.Layers.Ordering
import TorchLean.Verify.Coverage.Layers.Pooling
import TorchLean.Verify.Coverage.Layers.Random
import TorchLean.Verify.Coverage.Layers.Softmax
import TorchLean.Verify.Coverage.Layers.Structure
import TorchLean.Verify.Coverage.Losses.Distillation
import TorchLean.Verify.Coverage.Losses.Likelihood
import TorchLean.Verify.Coverage.Losses.Margin
import TorchLean.Verify.Coverage.Losses.Regression
import TorchLean.Verify.Coverage.Ordering
import TorchLean.Verify.Coverage.Pointwise.Arithmetic
import TorchLean.Verify.Coverage.Pointwise.Comparison
import TorchLean.Verify.Coverage.Pointwise.Rounding
import TorchLean.Verify.Coverage.Pointwise.Selection
import TorchLean.Verify.Coverage.Pointwise.Transcendental
import TorchLean.Verify.Coverage.Structure
import TorchLean.Verify.Coverage.TapeOperations
import TorchLean.Verify.Coverage.TensorComposites.Attention
import TorchLean.Verify.Coverage.TensorComposites.Embedding
import TorchLean.Verify.Coverage.TensorComposites.Linear
import TorchLean.Verify.Coverage.TensorComposites.Loss
import TorchLean.Verify.Coverage.TensorComposites.Normalization
import TorchLean.Verify.Coverage.TensorComposites.Positional
import TorchLean.Verify.Coverage.TensorComposites.Routing
import TorchLean.Verify.Coverage.TensorComposites.Softmax
import TorchLean.Verify.Coverage.TensorOperations.Movement
import TorchLean.Verify.Coverage.TensorOperations.Reduction
import TorchLean.Verify.Coverage.TensorOperations.Window
import TorchLean.Verify.Coverage.Training
import TorchLean.Verify.Floats.IEEE754.Order
import TorchLean.Verify.Monotonicity.JSON
import TorchLean.Verify.NumericalCertificate.NumericalPolicy
import TorchLean.Verify.Robustness.LipschitzCertified
import TorchLean.Verify.Robustness.MLP
import TorchLean.Verify.RuntimeApprox.OnlineSoftmax.FDeriv
import TorchLean.Verify.RuntimeApprox.RL.Checked
import TorchLean.Verify.RuntimeApprox.RL.PPORatio
import TorchLean.Verify.RuntimeApprox.Roundoff.Baseline
import TorchLean.Verify.RuntimeApprox.Roundoff.Composites
import TorchLean.Verify.RuntimeApprox.Roundoff.Contractions
import TorchLean.Verify.RuntimeApprox.Roundoff.Convolution
import TorchLean.Verify.RuntimeApprox.Roundoff.CrossEntropy
import TorchLean.Verify.RuntimeApprox.Roundoff.OnlineSoftmax
import TorchLean.Verify.RuntimeApprox.Roundoff.Operators
import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxAxes
import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay.Binary32
import TorchLean.Verify.RuntimeApprox.TransformerBlock.Ops
import TorchLean.Verify.UniversalApproximation.BinaryMLP.Error
import TorchLeanExamples.BugZoo.AttentionMask
import TorchLeanExamples.BugZoo.AutogradDomain
import TorchLeanExamples.BugZoo.Broadcasting
import TorchLeanExamples.BugZoo.ConstantSlice
import TorchLeanExamples.BugZoo.IgnoredLabel
import TorchLeanExamples.BugZoo.KVCache
import TorchLeanExamples.BugZoo.LayerNormAxis
import TorchLeanExamples.BugZoo.RoPEPosition
import TorchLeanExamples.BugZoo.StableLoss
import TorchLeanExamples.BugZoo.Tokenizer
import TorchLeanExamples.DeepDives.Floats.EffectiveRounding
import TorchLeanExamples.DeepDives.Floats.GraphNumericalCertificate
import TorchLeanExamples.DeepDives.Floats.LibraryBugs
import TorchLeanExamples.DeepDives.Floats.LossBugs.FocalPower
import TorchLeanExamples.DeepDives.Floats.ProbBugs.SurvivalComplement
import TorchLeanExamples.DeepDives.Floats.RoundingBugs
import TorchLeanExamples.DeepDives.Floats.Stable.Exp
import TorchLeanExamples.DeepDives.Floats.Stable.K3
import TorchLeanExamples.DeepDives.Floats.Stable.Variance
import TorchLeanExamples.DeepDives.Muon
import TorchLeanExamples.DeepDives.Stability.RidgeRegression1D
import TorchLeanExamples.DeepDives.TensorOperations
import TorchLeanExamples.DeepDives.WidgetGallery
import TorchLeanExamples.Models.RL.GymnasiumRollout
import TorchLeanExamples.Models.RL.PPOGridWorld.View
import TorchLeanExamples.Models.Sequence.GPT2BlockBound
import TorchLeanExamples.Quickstart
import TorchLeanExamples.Quickstart.TypedTraining
import TorchLeanExamples.Runner
import TorchLeanExamples.Transcendentals
import TorchLeanExamples.Verification.CLI
import TorchLeanExamples.Verification.Lyapunov.SignConditions
import TorchLeanExamples.Verification.ODE.ArithParseLanguage
import TorchLeanExamples.Verification.ODE.EnclosureBackends
import TorchLeanExamples.Verification.RobustMLP
import TorchLeanTest.CUDA.KernelGradient
import TorchLeanTest.CUDA.KernelIndex
import TorchLeanTest.CUDA.MoE
import TorchLeanTest.CUDA.Scheduler
import TorchLeanTest.CUDA.Suite
import TorchLeanTest.Data.NPY
import TorchLeanTest.Data.Text
import TorchLeanTest.Floats.NativePrimitives
import TorchLeanTest.IR.Flow
import TorchLeanTest.IR.MoEImport
import TorchLeanTest.IR.Pointwise
import TorchLeanTest.Interop.Suite
import TorchLeanTest.OpTable
import TorchLeanTest.RL.PolicyKL
import TorchLeanTest.Tensor.Complex
import TorchLeanTest.Tensor.EinopsLexer
import TorchLeanTest.Tensor.Elementary
import TorchLeanTest.Tensor.Elementwise
import TorchLeanTest.Tensor.Frexp
import TorchLeanTest.Tensor.Fma
import TorchLeanTest.Tensor.OnlineSoftmax
import TorchLeanTest.Tensor.Operations
import TorchLeanTest.Tensor.Order
import TorchLeanTest.Tensor.Special
import TorchLeanTest.Tensor.Storage
import TorchLeanTest.Tree.Suite
import TorchLeanTest.Verification.BuilderImport
import TorchLeanTest.Verification.CameraCertificates
import TorchLeanTest.Verification.CertificateParsing
import TorchLeanTest.Verification.GradientKernel
import TorchLeanTest.Verification.KernelMap
import TorchLeanTest.Verification.OpCert
import TorchLeanTest.Verification.OpDeriv
import TorchLeanTest.Verification.OpDerivCat
import TorchLeanTest.Verification.OpDerivScatter
import TorchLeanTest.Verification.Recorder
import TorchLeanTest.Verification.RoundoffBaseline
import TorchLeanTest.Verification.RoundoffBinders
import TorchLeanTest.Verification.RoundoffMatcher
import TorchLeanTest.Verification.RoundoffMatcherTwin
import TorchLeanTest.Verification.RoundoffReuse
import TorchLeanTest.Verification.SoftmaxBound

/-!
# Checks on the built libraries

`scripts/lake.sh lint` builds the libraries and then runs this file with `lean`. Its imports reach
every module under `TorchLean/`, `TorchLeanExamples/` and `TorchLeanTest/` but the roots of the
executables, and `scripts/checks/dependency_audit.py` fails when they miss one. A module does
not import the private theorems of the modules it reads, nor the bodies of their definitions that
are not exposed, so this file is not a module.

## The axiom check

It runs `Lean.collectAxioms`, the function behind `#print axioms`, on every theorem of the imported
modules under those directories, including private theorems and the ones Lean generates, such as
equation lemmas and the auxiliary theorems of a definition's proofs. It fails unless each depends on
at most `propext`, `Classical.choice` and `Quot.sound`.

## The coverage check

It reads the declarations of the modules under `TorchLean/`, the libraries other than the examples
and the tests, and the routes that `#roundoff_coverage` analyzed in any imported module. It fails
unless every public operator specification (`TorchLean.VJPOp`) among those declarations and every
public function among them that records a node on a CPU tape and takes no graph is registered with
`#roundoff_coverage`, every constructor of `TorchLean.Op` at which `TorchLean.Op.hasRule` does not
reduce to `false` has an equation lemma among them named after `TorchLean.Op.rule?` and the
constructor, as `TorchLean.Op.rule?_relu`, and a route analyzed for the configured scalar when one
of its ports holds the model's scalar (`TorchLean.Port.Kind.isScalar`), and every `op_cert`
certificate of the imported modules speaks about a route of its operator: an analyzed route, the
gradient of one of its value inputs or its tangent. Every `derivative` certificate must have passed
the check of `op_cert` against `TorchLean.DerivCert`, the operator's forward and pullback at a
point. The gradient of every value input of a constructor with a rule must also have such a
certificate under `TorchLean/` in all the value inputs of the constructor together that leaves each
of its arguments a distinct variable, an element at the scalar and a proof the constructor takes
excepted. A constructor whose input ports are mapped from a list attribute is read at each length
`TorchLean.Roundoff.listLengths` gives, as `Op.special` at one, two and three orders and `Op.cat`
at two sizes. At an attribute of an inductive family with indices, as the special function of
`Op.special`, the check takes each member of the family the attribute's type admits and finds the
certificates that claim nothing there, as one of their premises is `False` after its binders. Such a
premise still holds where the types of its binders are empty, so those certificates state only that
case, as a tensor without entries. Some certificate must claim something at each member, and
those of each gradient at some member. The message counts the `derivative` certificates under
`TorchLean/`, those of them without domain premises, and the gradients. A function that takes a
graph lowers the whole graph to a tape and needs no registration. The routes that record one
generated theorem must describe it alike (status, reference, budget, premises, caveats, primitive
theorems, use of IEEE, assumed contracts and arms), and no two generated theorems may have one
statement. The outcomes of the analyses must also match `scripts/checks/roundoff_baseline.jsonl`, as
`TorchLean.Roundoff.Baseline.check` compares them: the check fails on a lower status, a weaker
budget, a new premise or caveat, a changed reference and a gone analysis, and on a new analysis or
an improvement until the file is updated with `TORCHLEAN_ROUNDOFF_BASELINE=update`. Its message
counts the outcomes for the configured scalar. -/

open Lean Meta Elab Command

run_cmd do
  let env ← getEnv
  if env.header.isModule then
    throwError "the axiom check reads private theorems, which a module does not import: run it \
      in a file that is not a module"
  let standard := #[``propext, ``Classical.choice, ``Quot.sound]
  let roots := [`TorchLean, `TorchLeanExamples, `TorchLeanTest]
  let mut modules : Nat := 0
  let mut theorems : Nat := 0
  let mut offenders := #[]
  for module in env.header.moduleNames, data in env.header.moduleData do
    unless roots.any (·.isPrefixOf module) do continue
    let mut declared := 0
    for name in data.constNames do
      let some (.thmInfo _) := env.find? name | continue
      declared := declared + 1
      let axioms ← collectAxioms name
      unless axioms.all standard.contains do
        offenders := offenders.push m!"{name} ({module}): {axioms}"
    if declared > 0 then
      modules := modules + 1
      theorems := theorems + declared
  unless offenders.isEmpty do
    throwError m!"theorems with axioms other than propext, Classical.choice and Quot.sound:\n\
      {MessageData.joinSep offenders.toList "\n"}"
  logInfo m!"the {theorems} theorems of {modules} modules depend on at most propext, \
    Classical.choice and Quot.sound"

namespace CoverageCheck

/-- The roots of the modules whose declarations the coverage check reads. -/
def coverageRoots : List Name := [`TorchLean]

/-- Whether `name` is declared in an imported module under `coverageRoots`. -/
def inCoverageRoots (env : Environment) (name : Name) : Bool :=
  match env.getModuleIdxFor? name with
  | some index => coverageRoots.any (·.isPrefixOf env.header.moduleNames[index.toNat]!)
  | none => false

/-- The first words of the route part of a route name. -/
def routeWords : List String := ["forward", "backward", "jvp", "prepared", "alias"]

/-- A route name split into the attribute choice it is analyzed at (`sum`, `mean true`, or empty)
and the route (`forward`, `backward 1`, `jvp`, `prepared backward 0`). -/
def splitRoute (route : String) : String × String :=
  let words := (route.splitOn " ").filter (!·.isEmpty)
  let choice := words.takeWhile (!routeWords.contains ·)
  (" ".intercalate choice, " ".intercalate (words.drop choice.length))

/-- The position a route speaks about: the input of a gradient, the output of a value or a
tangent, `0` without a number. -/
def routePosition (route : String) : Nat :=
  ((route.splitOn " ").getLast?.bind String.toNat?).getD 0

/-- Which fields of `TorchLean.Op.Rule` a certificate's statement reads. -/
structure RuleFields where
  forward : Bool
  backward : Bool
  prepare : Bool
  jvp : Bool

/-- The fields of `TorchLean.Op.Rule` that `type` projects, by projection function or by index. -/
def ruleFields (type : Expr) : MetaM RuleFields := do
  let fields := getStructureFields (← getEnv) ``TorchLean.Op.Rule
  let reads (projection : Name) : Bool :=
    (type.find? fun e => e.isAppOf projection ||
      match e with
      | .proj owner position _ => owner == ``TorchLean.Op.Rule &&
        fields[position]? == some (Name.mkSimple projection.getString!)
      | _ => false).isSome
  return { forward := reads ``TorchLean.Op.Rule.forward,
           backward := reads ``TorchLean.Op.Rule.backward,
           prepare := reads ``TorchLean.Op.Rule.prepare, jvp := reads ``TorchLean.Op.Rule.jvp? }

/-- Whether a certificate whose statement reads `fields`, about the inputs `inputs` (every input
when empty), speaks about the route `route`. -/
def speaksAbout (fields : RuleFields) (inputs : List Nat) (route : String) : Bool :=
  let position := routePosition route
  let input := inputs.isEmpty || inputs.contains position
  if route.startsWith "jvp" then fields.jvp
  else if route.startsWith "prepared" then fields.prepare && input
  else if route.startsWith "backward" then
    !fields.jvp && !fields.prepare && fields.backward && input
  else if route.startsWith "forward" then
    !fields.jvp && !fields.prepare && !fields.backward && fields.forward
  else false

/-- The public definitions declared under `coverageRoots` whose type satisfies `test`, sorted by
name. -/
def publicDefinitions (test : Expr → MetaM Bool) : MetaM (Array Name) := do
  let env ← getEnv
  let mut found := #[]
  for (name, info) in env.constants.toList do
    unless info matches .defnInfo _ do continue
    unless inCoverageRoots env name do continue
    if isPrivateName name || name.isInternalDetail || (← isProjectionFn name) then continue
    if ← test info.type then found := found.push name
  return found.qsort (·.toString < ·.toString)

/-- Whether `type`, after its binders, is an operator specification. -/
def isOperatorType (type : Expr) : MetaM Bool :=
  forallTelescope type fun _ result => return result.isAppOf ``TorchLean.VJPOp

/-- Whether `type` takes a graph, as a lowering of a whole graph to a tape does. -/
def takesGraph (type : Expr) : MetaM Bool :=
  forallTelescope type fun arguments _ => arguments.anyM fun argument => do
    let argumentType ← whnfR (← inferType argument)
    return argumentType.isAppOf ``TorchLean.Tape.GraphData ||
      argumentType.isAppOf ``TorchLean.Tape.Graph

/-- Whether `type` is that of a function that records a node on a CPU tape and takes no graph: it
mentions `TorchLean.SomeTensor`, and after its binders it is the extended tape and the node's id,
possibly as a `Result`. -/
def isTapeConstructor (type : Expr) : MetaM Bool := do
  unless (type.find? (·.isConstOf ``TorchLean.SomeTensor)).isSome do return false
  unless ← TorchLean.Roundoff.recordsNode type do return false
  return !(← takesGraph type)

/-- A constructor of the operator table at one length of each list attribute that its ports are
mapped from (`TorchLean.Roundoff.attributeChoices`), with fresh variables for its other attributes
and every element at the scalar, as `#roundoff_coverage` analyzes it. -/
structure TableSample where
  /-- The lengths, as `two sizes`, or empty for a constructor without such an attribute. -/
  label : String
  /-- Whether `TorchLean.Op.hasRule` reduces to `false` at it. -/
  ruleless : Bool
  /-- Whether one of its ports holds the model's scalar (`TorchLean.Port.Kind.isScalar`). A port
  whose sort does not reduce to a constructor, and a list of ports that does not reduce to its
  constructors, hold none. -/
  scalar : Bool
  /-- The gradient routes of its value inputs: `backward` for one input, `backward k` for the
  input at position `k`. -/
  gradients : Array String
  /-- The constructors of the inductive families among its attributes (`familyMembers`), each with
  the certificates that claim something at it (`claimsNothingAt`). -/
  members : Array (Name × Array Name)

/-- The constructors of the inductive family `type` that have that type, each applied to new
metavariables for its parameters and fields, when `type` has indices, as the special functions of
two arguments are the constructors of `SpecialFn 2`. -/
def familyMembers (type : Expr) : MetaM (Array (Name × Expr)) := do
  let type ← whnfR type
  let .const typeName _ := type.getAppFn | return #[]
  let some (.inductInfo info) := (← getEnv).find? typeName | return #[]
  if info.numIndices == 0 then return #[]
  info.ctors.toArray.filterMapM fun constructor => do
    let value ← mkConstWithFreshMVarLevels constructor
    let (fields, _, result) ← forallMetaTelescope (← inferType value)
    return if ← isDefEq result type then some (constructor, mkAppN value fields) else none

/-- Whether the certificate `certificate` of `operator` claims nothing at the operator `op`: its
occurrence of the rule unifies with `op`, and then one of its premises is `False` after its
binders, as the premise that every tuple of entries lies in an empty smooth domain. Such a premise
holds only where the types of its binders are empty, as for inputs without entries. -/
def claimsNothingAt (operator certificate : Name) (op : Expr) : MetaM Bool :=
  withNewMCtxDepth do
    let (locals, _, body) ←
      forallMetaTelescope (← inferType (← mkConstWithFreshMVarLevels certificate))
    let some ruleApp := body.find? fun e =>
        e.isAppOf ``TorchLean.Op.rule? && e.getAppNumArgs > 0 &&
          e.getAppArgs.back!.getAppFn.constName? == some operator
      | return false
    unless ← isDefEq ruleApp.getAppArgs.back! op do return false
    locals.anyM fun premise => do
      let type ← instantiateMVars (← inferType premise)
      unless ← isProp type do return false
      forallTelescope type fun _ conclusion => return (← whnf conclusion).isConstOf ``False

/-- The constructor `operator` of the operator table, applied as `op` to the variables
`attributes` at the lengths `label`, as `TableSample` reads it, with the certificates of
`certificates` that claim something at each member of a family among its attributes. -/
def tableSample (operator : Name) (certificates : Array Name) (label : String) (op : Expr)
    (attributes : Array Expr) : Term.TermElabM TableSample := do
  let rule ← whnf (← mkAppM ``TorchLean.Op.hasRule #[op])
  let #[ins, outs] := (← whnf (← inferType op)).getAppArgs |
    throwError "the coverage check: {operator} is not an operator"
  let sorts (ports : Expr) : MetaM (Array (Option Expr)) := do
    let ports ← try TorchLean.Roundoff.portList ports catch _ => pure #[]
    ports.mapM fun port => do
      try return some (← TorchLean.Roundoff.portParts port).1 catch _ => return none
  let isScalar (dtype : Option Expr) : Term.TermElabM Bool := do
    let some dtype := dtype | return false
    return (← whnf (mkApp (mkConst ``TorchLean.Port.Kind.isScalar) dtype)).isConstOf ``Bool.true
  let inputs ← sorts ins
  let mut gradients := #[]
  for h : position in [:inputs.size] do
    if inputs[position].any (·.isConstOf ``TorchLean.Port.Kind.value) then
      gradients := gradients.push
        (if inputs.size == 1 then "backward" else s!"backward {position}")
  let mut members := #[]
  for binder in attributes do
    for (member, value) in ← familyMembers (← inferType binder) do
      let atMember := op.replaceFVar binder value
      members := members.push (member,
        ← certificates.filterM fun c => return !(← claimsNothingAt operator c atMember))
  return { label, ruleless := rule.isConstOf ``Bool.false,
           scalar := ← (inputs ++ (← sorts outs)).anyM isScalar, gradients, members }

/-- The constructor `operator` of the operator table as `TableSample` reads it, at each choice of
lengths of the list attributes that its ports are mapped from, with the certificates of
`certificates` that claim something at each member of a family among its attributes. -/
def tableConstructor (operator : Name) (certificates : Array Name) :
    Term.TermElabM (Array TableSample) := do
  let constant ← mkConstWithFreshMVarLevels operator
  (← TorchLean.Roundoff.attributeChoices constant (constructors := false)).toArray.mapM
    fun (choice, label) => TorchLean.Roundoff.withAttributes constant choice
      (tableSample operator certificates label)

/-- The scalar elements. An operand whose element is a constructor argument, as those of `add`,
is a value port exactly at the scalar element. -/
def scalarElements : List Name :=
  [``TorchLean.Op.Element.scalar, ``TorchLean.Op.NumericElement.scalar]

/-- Whether the statement `type` leaves every argument of the constructor `operator` a variable at
its occurrence of the rule, no two the same, a proof the constructor takes and an element at the
scalar excepted. -/
def fixesNoArgument (operator : Name) (type : Expr) : MetaM Bool := do
  let proofs ← forallTelescope (← getConstInfo operator).type fun binders _ =>
    binders.mapM fun binder => do isProp (← inferType binder)
  forallTelescope type fun locals body => do
    let some ruleApp := body.find? fun e =>
        e.isAppOf ``TorchLean.Op.rule? && e.getAppNumArgs > 0 &&
          e.getAppArgs.back!.getAppFn.constName? == some operator
      | return false
    let arguments := (ruleApp.getAppArgs.back!.getAppArgs.zip proofs).filterMap
      fun (argument, proof) =>
        if proof || scalarElements.any argument.isConstOf then none else some argument
    return arguments.all (fun a => a.isFVar && locals.contains a) &&
      arguments.toList.eraseDups.length == arguments.size

run_cmd do
  let env ← getEnv
  if env.header.isModule then
    throwError "the coverage check reads the bodies of definitions that are not exposed, which a \
      module does not import: run it in a file that is not a module"
  -- The routes of every imported module, wherever the registrations sit.
  let mut entries : Array TorchLean.Roundoff.CoverageEntry := #[]
  for module in env.header.moduleNames do
    let some index := env.getModuleIdx? module | continue
    entries := entries ++ PersistentEnvExtension.getModuleEntries
      TorchLean.Roundoff.coverageExtension env index
  let registered := entries.foldl (fun names entry => names.insert entry.operator) NameSet.empty
  let mut problems : Array String := #[]
  -- The routes that record one generated theorem describe it alike, and no two generated
  -- theorems have one statement.
  let mut described : Std.HashMap Name TorchLean.Roundoff.CoverageEntry := {}
  let mut statements : Std.HashMap Expr Name := {}
  for entry in entries do
    if entry.evidence.isAnonymous then continue
    match described[entry.evidence]? with
    | some first =>
      unless first.status == entry.status && first.reference == entry.reference &&
          first.budget == entry.budget && first.premises == entry.premises &&
          first.caveats == entry.caveats && first.contracts == entry.contracts &&
          first.usesIEEE == entry.usesIEEE && first.functions == entry.functions &&
          first.arms == entry.arms do
        problems := problems.push s!"the {entry.route} route of {entry.operator} \
          ({entry.backend}, {entry.mode}) describes {entry.evidence} otherwise than the \
          {first.route} route of {first.operator}"
    | none =>
      described := described.insert entry.evidence entry
      let some info := env.find? entry.evidence | continue
      if let some other := statements[info.type]? then
        problems := problems.push s!"the generated theorems {other} and {entry.evidence} have \
          one statement"
      statements := statements.insert info.type entry.evidence
  let specifications ← liftTermElabM <| publicDefinitions isOperatorType
  let recorders ← liftTermElabM <| publicDefinitions isTapeConstructor
  for name in specifications ++ recorders do
    unless registered.contains name do
      problems := problems.push s!"{name} is not registered with #roundoff_coverage"
  let some (.inductInfo table) := env.find? ``TorchLean.Op |
    throwError "the coverage check: no operator table"
  -- The equation lemmas are named after the rule function, as `TorchLean.Op.rule?_relu` after
  -- `TorchLean.Op.rule?`.
  let rule := ``TorchLean.Op.rule?
  let stem := rule.getString! ++ "_"
  let lemmas := env.constants.toList.filterMap fun (name, info) =>
    match name, info with
    | .str space last, .thmInfo _ =>
      if space == rule.getPrefix && last.startsWith stem && inCoverageRoots env name then some last
      else none
    | _, _ => none
  let configured := entries.filter fun entry =>
    entry.backend == "configured" && !entry.route.isEmpty
  let mut certified : Nat := 0
  let mut positions : Nat := 0
  let mut canonical : Nat := 0
  let mut unconditional : Nat := 0
  for operator in table.ctors do
    -- Every derivative certificate passed the check of `op_cert`. Those declared under
    -- `coverageRoots` count, and the ones that fix no argument may cover the gradients.
    let mut candidates : Array (Name × RuleFields × List Nat) := #[]
    for entry in TorchLean.Op.certificates env operator do
      unless entry.kind == `derivative do continue
      let some checked := TorchLean.Op.derivEntry? env entry.certificate |
        problems := problems.push s!"{operator}: the derivative certificate \
          {entry.certificate} did not pass the check of op_cert"
      unless inCoverageRoots env entry.certificate do continue
      canonical := canonical + 1
      if checked.premises.isEmpty then unconditional := unconditional + 1
      let type := (← getConstInfo entry.certificate).type
      if ← liftTermElabM <| fixesNoArgument operator type then
        candidates := candidates.push
          (entry.certificate, ← liftTermElabM <| ruleFields type, checked.varying)
    let samples ← liftTermElabM <| tableConstructor operator (candidates.map (·.1))
    let gradients := samples.foldl (fun all sample => sample.gradients.foldl
      (fun all route => if all.contains route then all else all.push route) all) #[]
    let short := operator.getString!
    let rows := configured.filter (·.operator == operator)
    let named (name : String) := name == stem ++ short || name.startsWith (stem ++ short ++ "_")
    unless samples.all (·.ruleless) do
      unless lemmas.any named do
        problems := problems.push s!"{operator} has no equation lemma {rule}_{short}"
      if samples.any (·.scalar) && rows.isEmpty then
        problems := problems.push s!"{operator} has a port of scalars and no route analyzed for \
          the configured scalar"
    let routes := rows.foldl (fun routes entry =>
      let route := (splitRoute entry.route).2
      if routes.contains route then routes else routes.push route) gradients
    for entry in TorchLean.Op.certificates env operator do
      certified := certified + 1
      let fields ← liftTermElabM <| ruleFields (← getConstInfo entry.certificate).type
      unless fields.jvp || routes.any (speaksAbout fields entry.inputs) do
        problems := problems.push s!"{operator}: the certificate {entry.certificate} speaks \
          about no route"
    -- At each sample with a rule, the certificates in all the value inputs together (their
    -- varying positions, or a mapped list of inputs) cover the gradients, and claim something at
    -- each member of a family among the attributes.
    for sample in samples do
      if sample.ruleless then continue
      let values := sample.gradients.map routePosition
      let joint := candidates.filter fun (_, _, varying) =>
        varying.isEmpty || values.all varying.contains
      for (member, claiming) in sample.members do
        unless joint.any (claiming.contains ·.1) do
          problems := problems.push s!"{operator}: no derivative certificate claims anything at \
            {member}"
      for route in sample.gradients do
        positions := positions + 1
        let speaking := joint.filter fun (_, fields, inputs) => speaksAbout fields inputs route
        let label := " ".intercalate ([sample.label, route].filter (!·.isEmpty))
        if speaking.isEmpty then
          problems := problems.push s!"{operator}: its gradient {label} has no derivative \
            certificate of the canonical form in all its value inputs together that leaves its \
            arguments distinct variables"
        else if !sample.members.isEmpty && sample.members.all fun (_, claiming) =>
            !speaking.any (claiming.contains ·.1) then
          problems := problems.push s!"{operator}: the derivative certificates of its gradient \
            {label} claim nothing at any member of a family among its attributes"
  -- The outcomes of the analyses against the baseline next to this file.
  problems := problems ++ (← TorchLean.Roundoff.Baseline.check entries
    ((System.FilePath.mk (← getFileName)).withFileName "roundoff_baseline.jsonl"))
  unless problems.isEmpty do
    throwError m!"the coverage check:\n{"\n".intercalate problems.toList}"
  let count (status : TorchLean.Roundoff.CoverageStatus) : Nat :=
    (configured.filter (·.status == status)).size
  logInfo m!"{specifications.size} operator specifications and {recorders.size} tape \
    constructors are registered with #roundoff_coverage. The {table.ctors.length} constructors \
    of the operator table have the equation lemmas and routes they need, and their {certified} \
    certificates speak about routes. {canonical} derivative certificates have the canonical \
    form, {unconditional} of them without domain premises, and each of the {positions} gradients \
    of their value inputs has one in all the value inputs together that leaves the operator's \
    arguments distinct variables and claims something at a member of each family among its \
    attributes. Of the {configured.size} analyses for the \
    configured scalar, {count .proved} are proved, {count .conditional} conditional, \
    {count .unsupported} unsupported and {count .failed} failed."

end CoverageCheck
