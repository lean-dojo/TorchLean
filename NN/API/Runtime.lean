/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

/-!
# TorchLean Runtime Facade

This module defines the `NN.API.TorchLean` namespace: the main re-export layer over the executable
TorchLean runtime:
- tensor primitives (`add`, `matmul`, `reshape`, ...)
- derived functional ops (`F.*`)
- losses / norms
- autograd entry points (grad/vjp/jacobian, etc.)
- neural-network layer combinators (`NN.Seq`, `NN.LayerDef`, ...)

The goal is to expose a stable, well-namespaced "public surface" without forcing users to import
internal runtime modules directly.

## How This Relates To `NN.API.Public`

`NN.API.Public` is the higher-level, PyTorch-like facade intended for most examples and tutorial
code. It builds on this runtime facade and adds more ergonomic constructors (record-based configs,
fewer proof arguments, etc.).

## PyTorch Mapping

- `API.TorchLean.NN.*` corresponds to pieces of `torch.nn` and `torch.nn.functional`
  (`https://pytorch.org/docs/stable/nn.html`, `https://pytorch.org/docs/stable/nn.functional.html`)
- `API.TorchLean.Optim.*` / `API.TorchLean.Trainer.*` correspond to `torch.optim`
  (`https://pytorch.org/docs/stable/optim.html`)

Most layer helpers keep the PyTorch name but append a shape/layout suffix such as `CHW` or `NCHW`
so the expected tensor layout stays visible at the call site.

TorchLean differs in two crucial ways:
- Shapes are part of types, so "wrong shape" often becomes a type error.
- Some backends are *proof-only*; use `NN.API.DType` selection when writing executables.

## Recommended Import

This is the canonical module for the runtime facade: `import NN.API.Runtime`.

For most user-facing code, prefer `NN.API.Public` instead. It keeps the same runtime machinery
behind a smaller, more PyTorch-shaped surface, while this module stays closer to the executable
TorchLean internals and lower-level compatibility re-exports.
-/

@[expose] public section


namespace NN
namespace API

namespace TorchLean

/-!
### Core Exports

Most of this namespace is a curated re-export of `_root_.Runtime.Autograd.TorchLean.*`, so users can
`import NN.API.Runtime` and get a stable API surface without importing internal runtime modules.

Rough grouping:
- execution control: `Backend`, `Options`
- program interface: `Ops`, `RefTy`, `Program`, `CompiledOut`, `CompiledScalar`, ...
- primitive tensor ops: `add`, `matmul`, `reshape`, elementwise activations, pooling, ...
- training utilities: `trainCycle*`, `meanLoss`
-/

export _root_.Runtime.Autograd.TorchLean (Backend Options TensorRef Param AnyParam)
export _root_.Runtime.Autograd.FastKernels (GpuMatmulPrecision)
export _root_.Runtime.Autograd.TorchLean (CompiledScalar compileScalar)
export _root_.Runtime.Autograd.TorchLean (CompiledOut compileOut)
export _root_.Runtime.Autograd.TorchLean (ParamList ScalarTrainer scalarTrainer)
export _root_.Runtime.Autograd.TorchLean (TList Ops Ref RefList CurriedRef RefTy Program)
export _root_.Runtime.Autograd.TorchLean.Curried (Fn curry uncurry)
export _root_.Runtime.Autograd.TorchLean.CurriedRef (uncurry applyVarList)
export _root_.Runtime.Autograd.TorchLean.RefList (append)
export _root_.Runtime.Autograd.TorchLean
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape transpose2d reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec
     scatterAddRow
   matmul concatVectors
   maxPool2d maxPool2dPad smoothMaxPool2d avgPool2d avgPool2dPad
   relu silu gelu sigmoid tanh softmax softplus exp log inv safeLog logSoftmax
   globalAvgPool2dChw globalAvgPool2dNchw
   sum flatten
   linear mseLoss layerNorm batchnormChannelFirst multiHeadAttention conv2d)
export _root_.Runtime.Autograd.TorchLean
  (scalarOf tlist1 tlist2 tlist3 tlist4 trainCycleSGD trainCycleOptim meanLoss)

/-
`TList` is a *typed list of tensors* whose shape list lives in the type.

It is great for safety (the compiler tracks parameter order/shapes), but raw destructuring
with `.cons ... .nil` is noisy in examples.

For tuple-like constructors/accessors (`tlist.unpack2`, `tlist.get1`, etc.), see:
`NN/API/TList.lean` (`namespace NN.API.tlist`).
-/

namespace RefList

/-- Unpack a 2-element `RefList` into a pair. -/
def unpack2 {Ref : Spec.Shape → Type} {s₁ s₂ : Spec.Shape} :
    TorchLean.RefList Ref [s₁, s₂] → (Ref s₁ × Ref s₂)
  | .cons x₁ (.cons x₂ .nil) => (x₁, x₂)

end RefList

namespace F
/- Functional tensor helpers mirroring `torch.nn.functional`-style building blocks. -/
export _root_.Runtime.Autograd.TorchLean.F
  (square checkpoint
   addB mulB
   embedding mean
   detach stopGrad
   dropoutSeeded)
end F

namespace Loss
/- Loss helpers mirroring the usual `torch.nn.functional` loss family. -/
export _root_.Runtime.Autograd.TorchLean.Loss
  (Reduction mse nllOneHot crossEntropyOneHot nllIndex nllNat crossEntropyIndex crossEntropyNat
    bceWithLogits bce)
end Loss

namespace Norm
/- Normalization helpers exposed at the runtime facade. -/
export _root_.Runtime.Autograd.TorchLean.Norm
  (rmsNormLast instanceNorm2dNchw groupNorm2dNchw
   batchNorm2dNchwTrain batchNorm2dNchwTrainStats batchNormRunningUpdate
   batchNorm2dNchwEval batchNorm2dChwEval)
end Norm

namespace Autodiff
/- Autodiff entrypoints for compiled and eager runtime programs. -/
export _root_.Runtime.Autograd.TorchLean.Autodiff
  (compileLoss compileOut
   gradParams gradInputs
   vjpOutParams vjpOutInputs
   jacrevOutParams jacrevOutInputs
   jacfwd1
   hessian1
   jvpLossParams jvpLossInputs
   hvpParams hvpInputs)
end Autodiff

namespace Metrics
/- Small post-processing metrics such as argmax and classification correctness. -/
export _root_.Runtime.Autograd.TorchLean.Metrics
  (argmax? classOfOneHot? correctOneHot?)
end Metrics

namespace Optim
/- Optimizer constructors used by the runtime trainer facade. -/
export _root_.Runtime.Autograd.TorchLean.Optim (StateList Optimizer)
export _root_.Runtime.Autograd.TorchLean.Optim
  (sgd momentumSGD adagrad rmsprop adam adamw adadelta projectedSGD muon)

/-!
### Optimizer Handles (PyTorch-Like)

TorchLean optimizers are purely functional in their state: `opt.step` returns a new state.

This small wrapper stores the optimizer state in an `IO.Ref` so users can write:

```
let h ← API.TorchLean.Optim.handle m (TorchLean.Optim.sgd lr)
h.step sample
```

without manually threading the optimizer state through the training loop.
-/

/--
A mutable optimizer handle bound to a concrete TorchLean `ScalarModule`.

The internal optimizer state is stored in an `IO.Ref` and updated when you call `h.step sample`.
-/
structure Handle (α : Type) [Context α] [DecidableEq Spec.Shape]
    (paramShapes inputShapes : List Spec.Shape) (State : Type) where
  /-- The module whose parameters will be updated in-place. -/
  module : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule α paramShapes inputShapes
  /-- Mutable optimizer state. -/
  state : IO.Ref State
  /-- One training step on a single sample, updating the internal optimizer state. -/
  step : TList α inputShapes → IO Unit

/--
Create an optimizer handle for a module by initializing optimizer state from the module's current
parameters.
-/
def handle {α : Type} [Context α] [DecidableEq Spec.Shape]
    {paramShapes inputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule α paramShapes inputShapes)
    (opt : Optimizer α paramShapes) :
    IO (Handle α paramShapes inputShapes opt.State) := do
  let st0 ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.initOptim (m := m) opt
  let stRef ← IO.mkRef st0
  let step (sample : TList α inputShapes) : IO Unit := do
    let st ← stRef.get
    let st' ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.stepWith (m := m) opt st sample
    stRef.set st'
  pure { module := m, state := stRef, step := step }
end Optim

namespace RL
/- Reinforcement-learning helpers spanning bandits, tabular control, value learning, and policy
objectives. -/
  export _root_.Spec.RL
    (AdvantageStep
     continueMask discountedBackup tdTarget tdResidual
     discountedReturns discountedReturnsFrom discountedReturnsDone
     generalizedAdvantageEstimation returnsFromAdvantages
     ValueFunction Policy FiniteMDP
     valueAt stateActionValue actionValues
     bellmanPolicy bellmanOptimality)
  export _root_.Spec.RL.FiniteMDP (toEnv)
  export _root_.Spec.RL.Markov
    (ValueFunction Policy MDP Valid
     transitionMeasure
     expectedNextValue actionValue
     bellmanPolicy bellmanOptimality)
  export _root_.Spec.RL.FiniteStochastic
    (MDP Valid
     expectedNextValue actionValue actionValues
     bellmanPolicy bellmanOptimality)
  export _root_.Runtime.RL.Core
    (Transition IndexedTransition
     oneHotAction
     discountedReturnsVecFrom discountedReturnsVec discountedReturnsVecDone
     generalizedAdvantageEstimationVec returnsFromAdvantagesVec
     squaredError huberLoss)
  export _root_.Runtime.RL.Bandits
    (ValueState PreferenceState
     greedyAction? epsilonGreedyAction?
     sampleAverageStep totalPulls
     ucb1Bonus ucb1Scores ucb1Action?
     gradientPolicy gradientBanditStep)
  export _root_.Runtime.RL.Bandits.ValueState (init)
  export _root_.Runtime.RL.Bandits.PreferenceState (init)
export _root_.Runtime.RL.Tabular
  (actionRow maxActionValue greedyAction? expectedActionValue
   td0Update
   sarsaTarget expectedSarsaTarget qLearningTarget doubleQTarget
   sarsaUpdate expectedSarsaUpdate qLearningUpdate
   doubleQUpdateLeft doubleQUpdateRight)
  export _root_.Runtime.RL.ValueLearning
    (chosenActionValue maxQValue
     dqnTarget doubleDqnTarget
     dqnResidual dqnMSELoss dqnHuberLoss doubleDqnResidual
     ddpgActorObjective ddpgCriticTarget td3Target
     sacTarget sacActorObjective)
  export _root_.Runtime.RL.PolicyGradient
    (actionPolicy actionProbability actionLogProbability entropyBonus
     reinforceLoss actorLoss criticLoss actorCriticLoss
     importanceRatio ppoClippedObjective ppoLoss)

  namespace Autograd
  /- Differentiable (autograd-capable) policy-gradient objectives over TorchLean `Ops`. -/
  export _root_.Runtime.RL.PolicyGradient.Autograd
    (actionLogProbOneHotBatch
     entropyMean
     ppoClippedObjectiveBatch
     ppoLossBatch
     ppoActorCriticScalarModuleDef)
  end Autograd
  end RL

namespace NN
/- Neural-network layer constructors and sequential-model helpers. -/
export _root_.Runtime.Autograd.TorchLean.NN
  (Mode LayerDef Seq
   linear rnn gru mamba lstm
   relu silu gelu sigmoid tanh softmax square sum flatten dropout
   layerNorm rmsNorm
   batchnormChannelFirst batchnormChannelFirstEval batchnormChannelFirstMode
   instanceNorm2dNchw groupNorm2dNchw batchNorm2dNchw batchNorm2dNchwMode
   multiHeadAttention conv2d
   maxPool2d maxPool2dPad avgPool2d avgPool2dPad
   globalAvgPool2dChw globalAvgPool2dNchw
   seq1)

/-
To keep example code "PyTorch-like", the `seq!` macro supports stacking either:
- a single layer (`LayerDef σ τ`), or
- an already-sequential model (`Seq σ τ`)
in the same `seq! ...` expression.

Lean's coercion insertion is not always reliable in partially-applied situations, so we provide an
explicit, typeclass-driven adapter that `seq!` can use.
-/
universe u v

/--
Adapter typeclass used by the `seq!` macro to treat both layers and already-sequential models as
composable building blocks.

This exists purely for ergonomics: it lets examples mix `LayerDef` and `Seq` in the same `seq!`
expression without relying on Lean's coercion insertion heuristics.
-/
class AsSeqK (F : Spec.Shape → Spec.Shape → Sort u) where
  /-- Convert a layer-like thing into a `Seq` so `seq!` can compose it. -/
  asSeq : {σ τ : Spec.Shape} → F σ τ → Seq σ τ

/-- A single `LayerDef` can always be viewed as a 1-layer sequential model (`seq1`). -/
instance : AsSeqK LayerDef where
  asSeq := fun {_σ _τ} layer => seq1 layer

/-- A sequential model is already a sequential model (identity). -/
instance : AsSeqK Seq where
  asSeq := fun {_σ _τ} s => s

/--
Compose either layers or sequential models without relying on coercions.

This is the helper used by the `seq! ...` macro so examples can write
`seq! layer1, model2, layer3` while still mirroring PyTorch's "stack layers" style.
-/
def compAny {σ τ υ : Spec.Shape}
    {F : Spec.Shape → Spec.Shape → Sort u} {G : Spec.Shape → Spec.Shape → Sort v}
    [AsSeqK F] [AsSeqK G] (f : F σ τ) (g : G τ υ) : Seq σ υ :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq.comp (AsSeqK.asSeq f) (AsSeqK.asSeq g)

namespace Seq
export _root_.Runtime.Autograd.TorchLean.NN.Seq
  (paramShapes paramRequiresGrad initParams comp updateBuffers
   programWithMode program
   scalarModuleDefWithMode scalarModuleDef
   mseScalarModuleDefWithMode mseScalarModuleDef
   crossEntropyOneHotScalarModuleDefWithMode crossEntropyOneHotScalarModuleDef
   compileOutWithMode compileOut
   predict1WithMode predict1 eval1 eval1NoGrad eval1CompiledNoGrad predict1NoGrad)
end Seq
end NN

namespace Random
/-!
Deterministic RNG helpers re-exported for the runtime facade.

These are small utilities used by demos and training loops (`keyOf`, `nextSeed`, `uniform`, `mask`).
-/
export _root_.Runtime.Autograd.TorchLean.Random (keyOf nextSeed uniform mask)
end Random

namespace Layers

/-!
### Sequential Layer Helpers

`Runtime.Autograd.TorchLean.NN` exposes *layers* (`LayerDef σ τ`) and *sequential models*
(`Seq σ τ`). For demos we often want to "just stack layers", so this namespace provides small
helpers that return `Seq` directly (and compute a few common derived shapes like `flattenLinear`).

For the more fully-documented public surface (named-field configs, blocks, etc.), see
`NN.API.Public` under `API.nn`.
-/

/-- Lift a single layer into a sequential model. -/
def of {σ τ : Spec.Shape} (layer : API.TorchLean.NN.LayerDef σ τ) :
    API.TorchLean.NN.Seq σ τ :=
  API.TorchLean.NN.seq1 layer

/-- Linear layer over vectors (returns a 1-layer `Seq`). -/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  of <| API.TorchLean.NN.linear inDim outDim seedW seedB

/-- Elementwise ReLU. -/
def relu {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.relu (s := s)

/-- Elementwise SiLU/Swish. -/
def silu {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.silu (s := s)

/-- Elementwise GELU. -/
def gelu {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.gelu (s := s)

/-- Elementwise sigmoid. -/
def sigmoid {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.sigmoid (s := s)

/-- Elementwise tanh. -/
def tanh {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.tanh (s := s)

/-- Softmax layer. -/
def softmax {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.softmax (s := s)

/-- Elementwise square. -/
def square {s : Spec.Shape} : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.square (s := s)

/-- Reduce-sum to a scalar. -/
def sum {s : Spec.Shape} : API.TorchLean.NN.Seq s Spec.Shape.scalar :=
  of <| API.TorchLean.NN.sum (s := s)

/-- Flatten any input shape into a 1D vector of length `Spec.Shape.size s`. -/
def flatten {s : Spec.Shape} : API.TorchLean.NN.Seq s (.dim (Spec.Shape.size s) .scalar) :=
  of <| API.TorchLean.NN.flatten (s := s)

/-- Dropout layer that is active in training mode and identity in eval mode. -/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : API.TorchLean.NN.Seq s s :=
  of <| API.TorchLean.NN.dropout (s := s) p seed

/-- `Flatten -> Linear` head, with the input dimension computed from the input shape. -/
def flattenLinear {s : Spec.Shape} (outDim : Nat) (seedW seedB : Nat := 0) :
    API.TorchLean.NN.Seq s (NN.Tensor.Shape.Vec outDim) :=
  (flatten (s := s)) >>> (linear (Spec.Shape.size s) outDim seedW seedB)

/-- Sequential 2D convolution layer for CHW inputs. -/
def conv2d (inC outC kH kW stride padding inH inW : Nat)
    {hInC : inC ≠ 0} {hKH : kH ≠ 0} {hKW : kW ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  of <| API.TorchLean.NN.conv2d
    (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) (h1 := hInC) (h2 := hKH) (h3 := hKW)
    (seedK := seedK) (seedB := seedB) (kInit := kInit)

/-- Sequential max-pooling layer for CHW inputs. -/
def maxPool2d (kH kW inH inW inC stride : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  of <| API.TorchLean.NN.maxPool2d
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := hKH) (h2 := hKW)

/-- Sequential padded max-pooling layer for CHW inputs. -/
def maxPool2dPad (kH kW inH inW inC stride padding : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  of <| API.TorchLean.NN.maxPool2dPad
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
      padding)
    (h1 := hKH) (h2 := hKW)

/-- Sequential average-pooling layer for CHW inputs. -/
def avgPool2d (kH kW inH inW inC stride : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  of <| API.TorchLean.NN.avgPool2d
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    hKH hKW

/-- Sequential padded average-pooling layer for CHW inputs. -/
def avgPool2dPad (kH kW inH inW inC stride padding : Nat)
    {hKH : kH ≠ 0} {hKW : kW ≠ 0} :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  of <| API.TorchLean.NN.avgPool2dPad
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
      padding)
    hKH hKW

/--
Global average-pooling over `C×H×W` inputs.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` followed by
flattening the spatial axes.
-/
def globalAvgPoolCHW (c h w : Nat)
    {hC : c > 0} {hH : h > 0} {hW : w > 0} :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.CHW c h w) (NN.Tensor.Shape.Vec c) :=
  of <| API.TorchLean.NN.globalAvgPool2dChw (c := c) (h := h) (w := w)
    (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)

/--
Global average-pooling over `N×C×H×W` inputs.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` and then reshaping
to `(N, C)`.
-/
def globalAvgPoolNCHW (n c h w : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0} :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (.dim n (.dim c .scalar)) :=
  of <| API.TorchLean.NN.globalAvgPool2dNchw (n := n) (c := c) (h := h) (w := w)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)

/--
Sequence-wise layer normalization.

PyTorch analogy: `torch.nn.LayerNorm(embedDim)` applied to each position in a sequence.
-/
def layerNorm (batch seqLen embedDim : Nat)
    {hSeq : seqLen > 0} {hEmbed : embedDim > 0}
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.NN.Seq (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| API.TorchLean.NN.layerNorm (batch := batch)
    (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Sequence-wise RMS normalization.

PyTorch analogy: an `RMSNorm`-style layer over `(seqLen × embedDim)` tensors.
-/
def rmsNorm (batch seqLen embedDim : Nat)
    {hSeq : seqLen > 0} {hEmbed : embedDim > 0}
    (seedGamma : Nat := 0) :
    API.TorchLean.NN.Seq (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| API.TorchLean.NN.rmsNorm (batch := batch)
    (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := seedGamma)

/--
Mode-aware batch norm on a single `C×H×W` image tensor.

PyTorch analogy: `torch.nn.BatchNorm2d(channels)` on a single sample, with the layer's mode
controlling whether running statistics are updated or reused.
-/
def batchNormCHW (channels height width : Nat)
    {hC : channels > 0} {hH : height > 0} {hW : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW channels height width)
      (NN.Tensor.Shape.CHW channels height width) :=
  of <| API.TorchLean.NN.batchnormChannelFirstMode
    (channels := channels) (height := height) (width := width)
    (h_c := hC) (h_h := hH) (h_w := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)
    (seedMean := seedMean) (seedVar := seedVar)

/--
Eval-mode batch norm on a single `C×H×W` image tensor with explicit running statistics.

PyTorch analogy: `torch.nn.BatchNorm2d(...).eval()` with `running_mean` and `running_var`.
-/
def batchNormEvalCHW (channels height width : Nat)
    {hC : channels > 0} {hH : height > 0} {hW : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    API.TorchLean.NN.Seq
      (NN.Tensor.Shape.CHW channels height width)
      (NN.Tensor.Shape.CHW channels height width) :=
  of <| API.TorchLean.NN.batchnormChannelFirstEval
    (channels := channels) (height := height) (width := width)
    (h_c := hC) (h_h := hH) (h_w := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)
    (seedMean := seedMean) (seedVar := seedVar)

/--
Instance normalization over `N×C×H×W` tensors.

PyTorch analogy: `torch.nn.InstanceNorm2d(c, affine=True)` with `NCHW` layout.
-/
def instanceNorm2dNCHW (n c h w : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0}
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  of <| API.TorchLean.NN.instanceNorm2dNchw
    (n := n) (c := c) (h := h) (w := w)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Group normalization over `N×C×H×W` tensors.

PyTorch analogy: `torch.nn.GroupNorm(groups, c)` with `NCHW` layout.
-/
def groupNorm2dNCHW (n c h w groups : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0} {hG : groups > 0}
    (hGE : c ≥ groups) (hDiv : c % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  of <| API.TorchLean.NN.groupNorm2dNchw
    (n := n) (c := c) (h := h) (w := w) (groups := groups)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW) (h_g_pos := hG)
    hGE hDiv
    (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Batch norm over `N×C×H×W` tensors in training mode.

PyTorch analogy: `torch.nn.BatchNorm2d(c)` during training, where batch statistics are used.
-/
def batchNorm2dNCHW (n c h w : Nat)
    {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    API.TorchLean.NN.Seq (NN.Tensor.Shape.NCHW n c h w) (NN.Tensor.Shape.NCHW n c h w) :=
  of <| API.TorchLean.NN.batchNorm2dNchwMode
    (n := n) (c := c) (h := h) (w := w)
    (h_n_pos := hN) (h_c_pos := hC) (h_h_pos := hH) (h_w_pos := hW)
    (seedGamma := seedGamma) (seedBeta := seedBeta)
    (seedMean := seedMean) (seedVar := seedVar)

/--
Multi-head self-attention over sequence embeddings.

PyTorch analogy: `torch.nn.MultiheadAttention(embed_dim=dModel, num_heads=numHeads)` in self-
attention mode, with explicit `n × dModel` shapes.
-/
def attention (batch n dModel numHeads headDim : Nat)
    {hN : n ≠ 0}
    (seedW : Nat := 0)
    (mask : Option (_root_.Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    API.TorchLean.NN.Seq (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  of <| API.TorchLean.NN.multiHeadAttention (batch := batch)
    (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
    (h1 := hN) (seedW := seedW) (mask := mask)

end Layers

namespace Autodiff

namespace Model

/-
This section provides "model-shaped" autodiff helpers:
- a `Seq σ τ` model,
- an `OutputLoss τ υ` (loss built from model output + target),
- and convenience wrappers for VJP/Jacobian/HVP/JVP and gradient extraction.

These are thin wrappers over `Runtime.Autograd.TorchLean.Autodiff` that hide the program/argument
packing boilerplate and keep call sites readable.
-/

/-- Parameter list type for a given model (a `TList` over `Seq.paramShapes`). -/
abbrev Params {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (α : Type) :=
  API.TorchLean.TList α (API.TorchLean.NN.Seq.paramShapes model)

/--
Loss function over a model output and a target.

This is expressed in terms of `RefTy` so it works uniformly for eager execution and compiled
execution.
-/
abbrev OutputLoss (τ υ : Spec.Shape) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
    {m : Type → Type} → [Monad m] → [API.TorchLean.Ops (m := m) (α := α)] →
      API.TorchLean.RefTy (m := m) (α := α) τ →
      API.TorchLean.RefTy (m := m) (α := α) υ →
      m (API.TorchLean.RefTy (m := m) (α := α) Spec.Shape.scalar)

/--
Initialize model parameters by casting the model's `Float` initializers elementwise using `cast`.
-/
def initParamsWith {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} (cast : Float → α) :
    Params model α :=
  _root_.Runtime.Autograd.TorchLean.Module.castTList cast (API.TorchLean.NN.Seq.initParams model)

/-- Initialize model parameters using the runtime literal injection `API.Runtime.ofFloat`. -/
def initParams {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [API.Runtime.Scalar α] :
    Params model α :=
  Model.initParamsWith (model := model) API.Runtime.ofFloat

/-- Pack explicit weight and bias tensors for a single `Layers.linear` model. -/
def linearParams {α : Type} {inDim outDim : Nat} {seedW seedB : Nat}
    (w : _root_.Spec.Tensor α (NN.Tensor.Shape.Mat outDim inDim))
    (b : _root_.Spec.Tensor α (NN.Tensor.Shape.Vec outDim)) :
    Params (API.TorchLean.Layers.linear inDim outDim seedW seedB) α :=
  API.TorchLean.tlist2 w b

namespace OutputLoss

/-- Mean-squared error loss (`mse`) between `yhat` and `y`. -/
def mse {τ : Spec.Shape} (reduction : API.TorchLean.Loss.Reduction := .mean) :
    OutputLoss τ τ :=
  fun {α} _ _ =>
    fun {m} _ _ yhat y =>
      API.TorchLean.Loss.mse (m := m) (α := α) (s := τ) yhat y (reduction := reduction)

/-- Cross-entropy loss between logits and one-hot targets. PyTorch analogue: `nn.CrossEntropyLoss`.
  -/
def crossEntropyOneHot {τ : Spec.Shape} (reduction : API.TorchLean.Loss.Reduction := .mean) :
    OutputLoss τ τ :=
  fun {α} _ _ =>
    fun {m} _ _ logits targetOneHot =>
      API.TorchLean.Loss.crossEntropyOneHot (m := m) (α := α) (s := τ) logits targetOneHot
        (reduction := reduction)

/--
Detach the model output before feeding it into a loss.

This is useful when you want to compute a metric loss without backpropagating through it.
-/
def detach {τ υ : Spec.Shape} (loss : OutputLoss τ υ) : OutputLoss τ υ :=
  fun {α} _ _ =>
    fun {m} _ _ yhat y => do
      let yhat' ← API.TorchLean.F.detach (m := m) (α := α) (s := τ) yhat
      loss (α := α) (m := m) yhat' y

end OutputLoss

/--
Build a TorchLean `Program` that computes a scalar loss from `(params, x, target)`.

This is the bridge between `Seq.program` (which produces model outputs) and the autograd entry
points (which expect a scalar-valued program).
-/
def lossProgram {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ) :
    ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
      API.TorchLean.Program α (API.TorchLean.NN.Seq.paramShapes model ++ [σ, υ]) Spec.Shape.scalar
        :=
  fun {α} _ _ =>
    fun {m} _ _ =>
      _root_.Runtime.Autograd.Torch.CurriedRef.curry
        (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
        (ss := API.TorchLean.NN.Seq.paramShapes model ++ [σ, υ])
        (β := m (API.TorchLean.RefTy (m := m) (α := α) Spec.Shape.scalar))
        (fun args => do
          let (ps, xy) :=
            _root_.Runtime.Autograd.Torch.RefList.split
              (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
              (ss₁ := API.TorchLean.NN.Seq.paramShapes model) (ss₂ := [σ, υ]) args
          let (x, y) := RefList.unpack2 xy
          let yhat ←
            _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
              (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
              (ss := API.TorchLean.NN.Seq.paramShapes model ++ [σ])
              (β := m (API.TorchLean.RefTy (m := m) (α := α) τ))
              (API.TorchLean.NN.Seq.program (model := model) (α := α))
              (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons x .nil))
          loss (α := α) (m := m) yhat y)

/-- VJP of the model output w.r.t. parameters. -/
def vjpParams {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Params model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ => API.TorchLean.NN.Seq.program (model := model) (α := β))
    params (API.TorchLean.tlist1 x) seedOut

/-- VJP of the model output w.r.t. inputs. -/
def vjpInputs {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (API.TorchLean.TList α [σ]) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutInputs
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ => API.TorchLean.NN.Seq.program (model := model) (α := β))
    params (API.TorchLean.tlist1 x) seedOut

/-- Jacobian (reverse-mode) of the model output w.r.t. parameters, returned as rows. -/
def jacrevParams {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) :
    IO (Array (Params model α)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacrevOutParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ => API.TorchLean.NN.Seq.program (model := model) (α := β))
    params (API.TorchLean.tlist1 x)

/-- Gradient of `loss(model(params, x), target)` w.r.t. parameters. -/
def gradParams {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Params model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tlist2 x target)

/-- Gradient of `loss(model(params, x), target)` w.r.t. inputs (`x` and `target`). -/
def gradInputs {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (API.TorchLean.TList α [σ, υ]) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradInputs
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tlist2 x target)

/-- JVP of a scalar loss w.r.t. parameters in direction `vparams`. -/
def jvpParams {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : Params model α) :
    IO α :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jvpLossParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tlist2 x target) vparams

/-- HVP (Hessian-vector product) of a scalar loss w.r.t. parameters in direction `vparams`. -/
def hvpParams {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : Params model α) :
    IO (Params model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hvpParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tlist2 x target) vparams

end Model

namespace Function1

/-
Function-1 autodiff helpers.

This is the "no parameters" case: treat a pure tensor function `f : Tensor σ -> Tensor τ` as the
thing we differentiate, rather than a model with an explicit parameter list.
-/

/--
Type of a pure tensor function expressed in `RefTy` form.

This matches the calling convention expected by `TorchLean.Program`/autodiff compilation.
-/
abbrev Fn (σ τ : Spec.Shape) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
    {m : Type → Type} → [Monad m] → [API.TorchLean.Ops (m := m) (α := α)] →
      API.TorchLean.RefTy (m := m) (α := α) σ →
      m (API.TorchLean.RefTy (m := m) (α := α) τ)

/-- Turn an `Fn` into a single-input TorchLean `Program`. -/
def program {σ τ : Spec.Shape} (f : Fn σ τ) :
    ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → API.TorchLean.Program α [σ] τ :=
  fun {α} _ _ =>
    fun {m} _ _ =>
      _root_.Runtime.Autograd.Torch.CurriedRef.curry
        (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
        (ss := [σ]) (β := m (API.TorchLean.RefTy (m := m) (α := α) τ))
        (fun args =>
          match args with
          | .cons x .nil => f (α := α) (m := m) x)

/-- Forward-mode Jacobian (rows) of a pure function. -/
def jacfwd {σ τ : Spec.Shape} (f : Fn σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α τ)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacfwd1
    (α := α) (σ := σ) (τ := τ) (program f) x

/-- Hessian for a scalar-valued pure function. -/
def hessian {σ : Spec.Shape} (f : Fn σ Spec.Shape.scalar)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hessian1
    (α := α) (σ := σ) (program f) x

end Function1

end Autodiff

namespace Models

/-!
## Model constructors (re-export)

This namespace re-exports a small set of ready-made model constructors (MLP/CNN/ResNet18/etc.),
primarily for runnable demos and smoke tests.

For compositional building blocks, prefer `API.TorchLean.NN` and `API.TorchLean.Layers`.
-/

export _root_.NN.GraphSpec.Models.TorchLean
  (mlp autoencoder cnn2 softmaxRegression mlpClassifier transformerBlock
   resnet18Model resnet18Program resnet18InitParams)
end Models

namespace Module

/-!
### ScalarModule API (Session-Like Interface)

The `ScalarModule` interface is the TorchLean equivalent of "instantiate a model, then do forward,
backward, and optimizer steps" in an imperative runtime.

This section mostly re-exports `Runtime.Autograd.TorchLean.Module.*` and adds small CLI-friendly
helpers (`Module.withModule` / `Module.withModuleRuntime`) that select dtype/backend from flags.
-/

export _root_.Runtime.Autograd.TorchLean.Module (ScalarModuleDef ScalarModule)
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModule
  (create forward backward step initOptim stepWith params setParams trainSGD trainWith meanLoss)
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef (instantiate)

/--
Instantiate a `ScalarModuleDef` under explicit Torch options (`backend`, `fastKernels`, `useGpu`,
etc.).

This is the most direct "runtime" entrypoint (used by the CPU/CUDA example binaries), since it
threads the same options record all the way down to the eager tape / CUDA tape selection.

Note: `instantiate` (without options) is a convenience wrapper that only selects the backend and
uses default runtime options.
-/
def instantiateWithOptions
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (cast : Float → α) (opts : Options) :
    IO (ScalarModule α paramShapes inputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
    (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) defn cast opts

/--
Execution configuration parsed from CLI flags.

Supported flags (parsed by `ExecConfig.parseAndStrip`):
- `--dtype ...` / `--float32-mode ...` (see `NN.API.DType`)
- `--backend eager|compiled`
- `--cpu` / `--cuda` (eager device selection)
- `--fast-kernels` (eager-only performance hooks, no effect on compiled backend)
- `--fast-gpu-matmul-precision fp32|fp64` (fast-kernel CUDA matmul precision)
-/
structure ExecConfig where
  /-- Scalar dtype selection. -/
  dtype : DType := .float
  /-- Execution backend selection. -/
  backend : Backend := .eager
  /--
  Eager execution device selector.

  When `true` and `backend = .eager`, TorchLean uses the CUDA tape backend.
  -/
  useGpu : Bool := false
  /-- Enable runtime-only eager fast kernels (tight-loop implementations for a few hot ops). -/
  fastKernels : Bool := false
  /-- GPU precision for fast-kernel matmul over Lean `Float` tensors. -/
  fastGpuMatmulPrecision : GpuMatmulPrecision := .fp32
  deriving Repr, DecidableEq

namespace ExecConfig

/-- Parse a backend selector string into a runtime `Backend`. -/
def parseBackend (v : String) : Except String Backend := do
  if v == "eager" then
    pure .eager
  else if v == "compiled" then
    pure .compiled
  else
    throw s!"unknown --backend {v} (supported: eager | compiled)"

/-- Parse a fast-kernel CUDA matmul precision selector. -/
def parseFastGpuMatmulPrecision (v : String) : Except String GpuMatmulPrecision := do
  if v == "fp32" || v == "float32" then
    pure .fp32
  else if v == "fp64" || v == "float64" || v == "double" then
    pure .fp64
  else
    throw s!"unknown --fast-gpu-matmul-precision {v} (supported: fp32 | fp64)"

/--
Parse CLI flags handled by `ExecConfig` and return `(cfg, rest)`.

Consumed flags:
- `--backend eager|compiled` (at most once),
- `--cpu` / `--cuda` (boolean flags; last one wins; removed from `rest`),
- `--fast-kernels` (boolean flag; removed from `rest`).
- `--fast-gpu-matmul-precision fp32|fp64` (at most once).

All dtype/Float32 selection flags are delegated to `DType.parseAndStripWithDefault`.

Default dtype policy:
- If the user does not specify `--dtype` / `--float32-mode` and `--cuda` is present, default to
  `dtype=float` (CUDA eager supports `Float` upload/download).
- Otherwise default to `dtype=float32` (executable IEEE-754 float32 semantics).
-/
def parseAndStripWithDefaultDType (args : List String) (defaultDType : DType) :
    Except String (ExecConfig × List String) := do
  let (dtype, args1) ← DType.parseAndStripWithDefault args defaultDType
  let (backendV?, args2) ← CLI.takeFlagValueOnce args1 "backend"
  let backend ←
    match backendV? with
    | none => pure .eager
    | some v => parseBackend v
  let (fastPrecisionV?, args3) ← CLI.takeFlagValueOnce args2 "fast-gpu-matmul-precision"
  let fastGpuMatmulPrecision ←
    match fastPrecisionV? with
    | none => pure .fp32
    | some v => parseFastGpuMatmulPrecision v
  let rec go (useGpu fastKernels : Bool) (acc : List String) :
      List String → (Bool × Bool × List String)
    | [] => (useGpu, fastKernels, acc.reverse)
    | a :: as =>
        if a == "--cuda" then
          go true fastKernels acc as
        else if a == "--cpu" then
          go false fastKernels acc as
        else if a == "--fast-kernels" then
          go useGpu true acc as
        else
          go useGpu fastKernels (a :: acc) as
  let (useGpu, fastKernels, rest) := go false false [] args3
  pure ({
    dtype := dtype,
    backend := backend,
    useGpu := useGpu,
    fastKernels := fastKernels,
    fastGpuMatmulPrecision := fastGpuMatmulPrecision
  }, rest)

/-- Parse CLI flags with the standard TorchLean default dtype policy. -/
def parseAndStrip (args : List String) : Except String (ExecConfig × List String) := do
  let defaultDType : DType := if args.contains "--cuda" then .float else .float32 {}
  parseAndStripWithDefaultDType args defaultDType

/-- Log the chosen execution config to stdout (for reproducible demos). -/
def log (cfg : ExecConfig) : IO Unit := do
  DType.log cfg.dtype
  IO.println s!"[TorchLean] backend: {reprStr cfg.backend}"
  IO.println s!"[TorchLean] device: {if cfg.useGpu then "cuda" else "cpu"}"
  IO.println s!"[TorchLean] fastKernels: {cfg.fastKernels}"
  IO.println s!"[TorchLean] fastGpuMatmulPrecision: {reprStr cfg.fastGpuMatmulPrecision}"

end ExecConfig

/--
Parse runtime flags (`--dtype`, `--backend`, `--cpu|--cuda`, `--fast-kernels`,
`--fast-gpu-matmul-precision`) and choose an executable scalar `α`, then call `k` with:
- `cast : Float → α` for building inputs from literals
- `opts : Options` selecting the backend/kernel mode
- `rest : List String` containing the remaining CLI arguments

This is useful for scripts that need to build a dataset/loader (and maybe determine shapes/batch
sizes) before instantiating a concrete `ScalarModuleDef`.
-/
def withRuntime
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] → (cast : Float → α) → (opts : Options) → (rest : List String) → IO
          Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts : Options :=
    { backend := cfg.backend
      useGpu := cfg.useGpu
      fastKernels := cfg.fastKernels
      fastGpuMatmulPrecision := cfg.fastGpuMatmulPrecision }
  match (← DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
        k (α := α) (API.Runtime.ofFloat (α := α)) opts rest
      )) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/--
Instantiate a `ScalarModuleDef` under CLI runtime flags (`--dtype`, `--backend`, `--cpu|--cuda`,
  `--fast-kernels`, `--fast-gpu-matmul-precision`), then call a continuation.

This provides the cast function `Float → α` so call sites can build inputs from float literals.
-/
def withModule
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        (cast : Float → α) → ScalarModule α paramShapes inputShapes → (rest : List String) → IO
          Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  match (← DType.withExec cfg.dtype (fun {α} _ _ _ cast => do
        let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
          (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
          defn cast
            { backend := cfg.backend
              useGpu := cfg.useGpu
              fastKernels := cfg.fastKernels
              fastGpuMatmulPrecision := cfg.fastGpuMatmulPrecision }
        k (α := α) cast m rest
      )) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/--
Like `withModule`, but also provides an `API.Runtime.Scalar α` instance (for numeric literals).
-/
def withModuleRuntime
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] → ScalarModule α paramShapes inputShapes → (rest : List String) → IO
          Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  match (← DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
        let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
          (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
          defn (API.Runtime.ofFloat (α := α))
            { backend := cfg.backend
              useGpu := cfg.useGpu
              fastKernels := cfg.fastKernels
              fastGpuMatmulPrecision := cfg.fastGpuMatmulPrecision }
        k (α := α) m rest
      )) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/-!
## Executable `main` Helpers

TorchLean has a lot of pure, type-indexed code (models live in `Type 2`), but runnable scripts still
want a "single entrypoint" that handles:
- parsing `--seed`,
- selecting an executable dtype/backend/device from flags,
- seeding TorchLean's global RNG stream (`API.rand`) so `nn.freshSeed`/`nn.withModel` are deterministic.
-/

/-- Options for `TorchLean.Module.run` (banner printing, trailing ok, etc.). -/
structure RunOptions where
  /-- Optional banner to print before executing the program. -/
  banner? : Option (Options → String) := none
  /-- Flush stdout after printing the banner (if present). -/
  flush : Bool := true
  /-- Print `"{exeName}: ok"` on success. -/
  printOk : Bool := false
deriving Inhabited

namespace RunOptions

def printBanner (o : RunOptions) (opts : Options) : IO Unit := do
  match o.banner? with
  | none => pure ()
  | some banner =>
      IO.println (banner opts)
      if o.flush then
        (← IO.getStdout).flush

end RunOptions

/-- How `run` should select the scalar backend for an executable. -/
inductive RunAction where
  /--
  Allow `--dtype` selection (the continuation must work for all executable scalar backends).
  -/
  | any
      (k :
        ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
          [API.Runtime.Scalar α] →
          (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit)
  /--
  Force the scalar backend to builtin `Float` (useful for Float-only IO bridges / CUDA upload paths).
  -/
  | float (k : (opts : Options) → (rest : List String) → IO Unit)

/--
CLI entrypoint helper for executable `main` functions.

This parses:
- `--seed N` (via `API.CLI.takeSeed`), and
- runtime execution flags (`--dtype`, `--float32-mode`, `--backend`, `--cpu|--cuda`,
  `--fast-kernels`, `--fast-gpu-matmul-precision`),
then executes the chosen `RunAction`.

It also seeds TorchLean's global RNG stream (`API.rand`) so code that draws init seeds via
`API.nn.freshSeed`/`API.nn.withModel` is deterministic by default, matching the PyTorch pattern of
calling `torch.manual_seed` once in `main`.
-/
def run
    (exeName : String)
    (args : List String)
    (action : RunAction)
    (runOpts : RunOptions := {}) :
    IO UInt32 := do
  let args := API.CLI.dropDashDash args
  let (seed, args) ←
    match API.CLI.takeSeed args 0 with
    | .ok v => pure v
    | .error msg => throw <| IO.userError s!"{exeName}: {msg}"

  _root_.NN.API.rand.manualSeed seed

  let printOk : IO Unit := do
    if runOpts.printOk then
      IO.println s!"{exeName}: ok"

  match action with
  | .any k =>
      withRuntime args (fun {α} _ _ _ _ cast opts rest => do
        -- Keep seed in the same `Options` record used by the Torch eager/compiled sessions so scripts
        -- can still follow the familiar pattern `nn.manualSeed opts.seed` when desired.
        let opts : Options := { opts with seed := seed }
        runOpts.printBanner opts
        k (α := α) cast opts rest
        printOk
      )
      pure 0
  | .float k =>
      let (cfg, rest) ←
        match ExecConfig.parseAndStripWithDefaultDType args .float with
        | .ok v => pure v
        | .error msg => throw <| IO.userError msg
      if cfg.dtype != .float then
        throw <| IO.userError s!"{exeName}: this program only supports `--dtype float`"
      ExecConfig.log cfg
      let opts : Options :=
        { backend := cfg.backend
          seed := seed
          fastKernels := cfg.fastKernels
          fastGpuMatmulPrecision := cfg.fastGpuMatmulPrecision
          useGpu := cfg.useGpu }
      runOpts.printBanner opts
      k opts rest
      printOk
      pure 0

end Module

namespace Supervised

/-
Supervised training helpers built directly on `ScalarModule`.

This is a slightly lower-level layer than `NN.API.Public.train`: it is designed around a
`SeqTask σ τ` (model + loss) and produces a `Runner` + `Stepper` that can be used in scripts.
-/

/-- Built-in loss choices for `SeqTask`. -/
inductive SeqLoss where
  | mse (reduction : API.TorchLean.Loss.Reduction := .mean)
  | crossEntropyOneHot (reduction : API.TorchLean.Loss.Reduction := .mean)

/-- A supervised task is just a model plus a choice of loss. -/
structure SeqTask (σ τ : Spec.Shape) where
  /-- Model to run. -/
  model : API.TorchLean.NN.Seq σ τ
  /-- Loss function. -/
  loss : SeqLoss

/--
Build a `ScalarModuleDef` for a task, choosing an explicit model mode (train/eval).

This is the underlying "instantiate me as a runnable module" step for training.
-/
def SeqTask.moduleDefWithMode {σ τ : Spec.Shape} (task : SeqTask σ τ)
    (mode : API.TorchLean.NN.Mode) :
    API.TorchLean.Module.ScalarModuleDef (API.TorchLean.NN.Seq.paramShapes task.model) [σ, τ] :=
  match task.loss with
  | .mse reduction =>
      API.TorchLean.NN.Seq.mseScalarModuleDefWithMode mode (model := task.model) (reduction :=
        reduction)
  | .crossEntropyOneHot reduction =>
      API.TorchLean.NN.Seq.crossEntropyOneHotScalarModuleDefWithMode mode
        (model := task.model) (reduction := reduction)

/-- Default module definition for a task (training mode). -/
def SeqTask.moduleDef {σ τ : Spec.Shape} (task : SeqTask σ τ) :
    API.TorchLean.Module.ScalarModuleDef (API.TorchLean.NN.Seq.paramShapes task.model) [σ, τ] :=
  task.moduleDefWithMode .train

namespace SeqTask

/-- Constructor: regression task (MSE loss). -/
def mse {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .mse reduction }

/-- Constructor: one-hot classification task (cross-entropy loss). -/
def crossEntropyOneHot {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .crossEntropyOneHot reduction }

end SeqTask

/-- Parameter shapes for a task (delegates to `Seq.paramShapes`). -/
abbrev paramShapes {σ τ : Spec.Shape} (task : SeqTask σ τ) : List Spec.Shape :=
  API.TorchLean.NN.Seq.paramShapes task.model

/--
Optimizer hyperparameter configuration for the supervised training helpers.

We keep this small for examples and lightweight trainers. It mirrors a few common PyTorch
optimizers by name/defaults, but it does not try to cover the full option surface of
  `torch.optim.*`.
-/
inductive OptimizerConfig where
  /--
  SGD optimizer config.

  PyTorch analogy: `torch.optim.SGD(..., lr=..., momentum=...)` when `momentum > 0`,
  and plain SGD when `momentum = 0`.
  -/
  | sgd (lr : Float) (momentum : Float := 0.0)
  /-- Adam optimizer config. -/
  | adam (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8)
  /-- AdamW optimizer config (decoupled weight decay). -/
  | adamw (lr : Float) (weightDecay : Float := 0.01)
      (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8)
  deriving Repr

/--
Step-based training configuration for `fit` / `fitDataset`.

Fields:
- `steps`: number of parameter updates,
- `optimizer`: optimizer hyperparameters,
- `scheduler`: optional learning-rate schedule (applied per step),
- `logEvery`: progress printing frequency (`0` disables logging).
-/
structure FitConfig where
  /-- Number of training steps. -/
  steps : Nat
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option API.TorchLean.Schedulers.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  deriving Repr

/--
Small summary returned by `fit*` helpers.

By default, `before` and `after` are mean loss values, but the type is polymorphic so callers can
report other scalars in the same shape.
-/
structure FitReport (α : Type) where
  /-- Metrics before training. -/
  before : α
  /-- Metrics after training. -/
  after : α

/--
Epoch-based training configuration for `fitLoader` (data-loader training).

Fields:
- `epochs`: number of epochs (each epoch iterates once over the loader),
- `optimizer`: optimizer hyperparameters,
- `scheduler`: optional learning-rate schedule (applied per step/epoch depending on helper),
- `logEvery`: progress printing frequency (`0` disables logging).
-/
structure LoaderFitConfig where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option API.TorchLean.Schedulers.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  deriving Repr

/-- Extract the base learning rate encoded in an optimizer configuration. -/
def optimizerLR : OptimizerConfig → Float
  | .sgd lr _ => lr
  | .adam lr _ _ _ => lr
  | .adamw lr _ _ _ _ => lr

/--
Resolve the learning rate to use at a given training step.

If a scheduler is present, it takes precedence over the optimizer's baked-in base learning rate.
Otherwise this simply returns `optimizerLR cfg`.
-/
def stepLR (scheduler : Option API.TorchLean.Schedulers.Config) (cfg : OptimizerConfig)
    (step : Nat) : Float :=
  match scheduler with
  | some sched => API.TorchLean.Schedulers.lrAt sched step
  | none => optimizerLR cfg

/-- Map a state update over every optimizer-state entry in a shape-indexed parameter list. -/
def mapStateList {State : Type → Spec.Shape → Type} {α : Type} :
    {ss : List Spec.Shape} →
    ({s : Spec.Shape} → State α s → State α s) →
    API.TorchLean.Optim.StateList State α ss →
    API.TorchLean.Optim.StateList State α ss
  | [], _, .nil => .nil
  | _ :: ss, f, .cons st rest => .cons (f st) (mapStateList (ss := ss) f rest)

/-- Set the learning rate field of every Adam optimizer state entry to `lr`. -/
def adamStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every momentum-SGD optimizer state entry to `lr`. -/
def momentumSGDStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every AdamW optimizer state entry to `lr`. -/
def adamwStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/--
A fully instantiated supervised task runner.

This bundles:
- the imperative `ScalarModule` (parameters/buffers stored in refs),
- compiled predictors and loss functions for both `.train` and `.eval` modes (so switching mode is
  cheap),
- and the current mode stored in an `IO.Ref`.

The mode influences both operator behavior (e.g. dropout/batchnorm) and whether buffers are updated
during training.
-/
structure Runner (α : Type) [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Instantiated scalar module storing parameters/buffers in mutable refs. -/
  module : API.TorchLean.Module.ScalarModule α (paramShapes task) [σ, τ]
  /-- Compiled forward predictor specialized to training-mode behavior. -/
  predictorTrain : _root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes task ++ [σ]) τ
  /-- Compiled forward predictor specialized to eval-mode behavior. -/
  predictorEval : _root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes task ++ [σ]) τ
  /-- Compiled loss function for training-mode behavior. -/
  lossTrain : _root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])
  /-- Compiled loss function for eval-mode behavior. -/
  lossEval : _root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])
  /-- Mutable mode flag (`.train` / `.eval`) used by stateful layers (e.g. dropout/batchnorm). -/
  mode : IO.Ref API.TorchLean.NN.Mode

/--
Instantiate a `Runner` by explicitly providing a `Float → α` cast and backend.

Use this when you want to run the same task over different numeric backends (e.g. `Float` vs
`IEEE32Exec`) or when you want custom literal injection.
-/
def instantiateWithOptions {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) := do
  let module ← API.TorchLean.Module.instantiateWithOptions (α := α) task.moduleDef cast opts
  let predictorTrain ← API.TorchLean.NN.Seq.compileOutWithMode .train (α := α) task.model
  let predictorEval ← API.TorchLean.NN.Seq.compileOutWithMode .eval (α := α) task.model
  let lossTrain ← API.TorchLean.Autodiff.compileLoss (α := α)
    (paramShapes := paramShapes task) (inputShapes := [σ, τ])
    (task.moduleDefWithMode .train).loss
  let lossEval ← API.TorchLean.Autodiff.compileLoss (α := α)
    (paramShapes := paramShapes task) (inputShapes := [σ, τ])
    (task.moduleDefWithMode .eval).loss
  let mode : IO.Ref API.TorchLean.NN.Mode ← IO.mkRef .eval
  pure {
    module := module
    predictorTrain := predictorTrain
    predictorEval := predictorEval
    lossTrain := lossTrain
    lossEval := lossEval
    mode := mode
  }

/--
Instantiate a `Runner` by explicitly providing a `Float → α` cast and a backend selector.

This is a backward-compatible wrapper over `instantiateWithOptions`.
-/
def instantiateWith {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) := do
  instantiateWithOptions (task := task) (α := α) cast { backend := backend }

/--
Instantiate a `Runner` using the standard runtime literal injection `API.Runtime.ofFloat`.

This is the common entrypoint for executable examples.
-/
def instantiateWithRuntimeOptions {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) :=
  instantiateWithOptions (task := task) (α := α) API.Runtime.ofFloat opts

/--
Instantiate a `Runner` using the standard runtime literal injection `API.Runtime.ofFloat` and a
backend selector.

This is a backward-compatible wrapper over `instantiateWithRuntimeOptions`.
-/
def instantiate {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) :=
  instantiateWithRuntimeOptions (task := task) (α := α) { backend := backend }

/--
Run a TorchLean task with CLI-style dtype/backend selection, then call `k` with a fully constructed
  runner.

This is used by `lake exe` entrypoints: `run` takes care of parsing dtype flags and instantiating
the underlying module/compiled programs.
-/
def run {σ τ : Spec.Shape} (task : SeqTask σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] →
        Runner α task → List String → IO Unit) :
    IO Unit := do
  API.TorchLean.Module.withModuleRuntime task.moduleDef args (fun {α} _ _ _ _ module rest => do
    let predictorTrain ← API.TorchLean.NN.Seq.compileOutWithMode .train (α := α) task.model
    let predictorEval ← API.TorchLean.NN.Seq.compileOutWithMode .eval (α := α) task.model
    let lossTrain ← API.TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes task) (inputShapes := [σ, τ])
      (task.moduleDefWithMode .train).loss
    let lossEval ← API.TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes task) (inputShapes := [σ, τ])
      (task.moduleDefWithMode .eval).loss
    let mode : IO.Ref API.TorchLean.NN.Mode ← IO.mkRef .eval
    k (α := α)
      { module := module
        predictorTrain := predictorTrain
        predictorEval := predictorEval
        lossTrain := lossTrain
        lossEval := lossEval
        mode := mode }
      rest
  )

/-- Read the current parameter list from a runner. -/
def params {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (API.TorchLean.TList α (paramShapes task)) :=
  API.TorchLean.Module.params runner.module

/-- Read the runner's current mode (`.train` or `.eval`). -/
def mode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO API.TorchLean.NN.Mode :=
  runner.mode.get

/-- Set the runner mode (`.train` or `.eval`). -/
def setMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (value : API.TorchLean.NN.Mode) : IO Unit :=
  runner.mode.set value

/-- Convenience: `setMode runner .train`. -/
def trainMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .train

/-- Convenience: `setMode runner .eval`. -/
def evalMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .eval

/-- Predicate: are we in training mode? -/
def isTraining {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Bool :=
  do
    pure ((← mode runner) == .train)

/-- Pick the predictor compiled for the runner's current mode (`.train` or `.eval`). -/
def activePredictor {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (_root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes task ++ [σ]) τ) := do
  match (← mode runner) with
  | .train => pure runner.predictorTrain
  | .eval => pure runner.predictorEval

/-- Pick the loss program compiled for the runner's current mode (`.train` or `.eval`). -/
def activeLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (_root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])) := do
  match (← mode runner) with
  | .train => pure runner.lossTrain
  | .eval => pure runner.lossEval

/--
Refresh mode-dependent runner buffers using one supervised sample.

This mutates the module parameters only in `.train` mode, mirroring PyTorch-style buffer updates
for layers such as normalization. In `.eval` mode it is a no-op.
-/
def updateRunnerBuffers {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TList α [σ, τ]) : IO Unit := do
  let currentMode ← mode runner
  if currentMode == .train then
    match sample with
    | .cons x (.cons _y .nil) => do
        let ps ← params runner
        let ps' ← API.TorchLean.NN.Seq.updateBuffers currentMode task.model ps x
        API.TorchLean.Module.setParams runner.module ps'
  else
    pure ()

/--
Run one forward/backward pass on a single supervised sample and return gradients for all parameters.

This is the TorchLean analogue of the `loss.backward()` payload in PyTorch, except TorchLean returns
the gradients explicitly instead of storing them in `.grad` fields.
-/
def backward {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TList α [σ, τ]) :
    IO (API.TorchLean.TList α (paramShapes task)) := do
  -- The instantiated scalar module always uses the training-mode program; keep the runner mode
  -- aligned so `updateRunnerBuffers` is not accidentally skipped.
  trainMode runner
  updateRunnerBuffers runner sample
  API.TorchLean.Module.backward runner.module sample

/-- Predict on one input tensor using the runner's active mode (`.train` or `.eval`). -/
def predict {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) := do
  let ps ← params runner
  let predictor ← activePredictor runner
  pure (API.TorchLean.NN.Seq.predict1 task.model predictor ps x)

/-- Predict on a list of inputs by repeatedly calling `predict`. -/
def predictBatch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) := do
  let ps ← params runner
  let predictor ← activePredictor runner
  pure <| xs.map (API.TorchLean.NN.Seq.predict1 task.model predictor ps)

/-- For classification heads: run `predict`, then take `argmax` over the logits (if defined). -/
def predictClass? {σ : Spec.Shape} {n : Nat} {task : SeqTask σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Option (Fin n)) := do
  let logits ← predict runner x
  pure <| API.TorchLean.Metrics.argmax? (α := α) (n := n) logits

/-- Compute `(correct, total)` for a one-hot classification dataset. -/
def accuracyOneHot {σ : Spec.Shape} {n : Nat} {task : SeqTask σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task) (samples : List (API.TorchLean.TList α [σ, .dim n .scalar])) :
    IO (Nat × Nat) := do
  let rec go (correct total : Nat) :
      List (API.TorchLean.TList α [σ, .dim n .scalar]) → IO (Nat × Nat)
    | [] => pure (correct, total)
    | sample :: rest =>
        do
          let (x, y) :=
            match sample with
            | .cons x (.cons y .nil) => (x, y)
          let logits ← predict runner x
          let ok := API.TorchLean.Metrics.correctOneHot? (α := α) (n := n) logits y
          go (if ok = some true then correct + 1 else correct) (total + 1) rest
  go 0 0 samples

/-- Mean scalar loss over a list of supervised samples (uses the runner's active mode). -/
def meanLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Add α] [Div α] [Zero
      α] [Coe Nat α]
    (runner : Runner α task) (samples : List (API.TorchLean.TList α [σ, τ])) :
    IO α := do
  let compiled ← activeLoss runner
  let ps ← params runner
  let values ← samples.mapM (fun sample => do
    let args : API.TorchLean.TList α (paramShapes task ++ [σ, τ]) :=
      _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append
        (α := α) (ss₁ := paramShapes task) (ss₂ := [σ, τ]) ps sample
    pure (Spec.Tensor.toScalar <| _root_.Runtime.Autograd.Torch.CompiledScalar.forward compiled
      args))
  match values with
  | [] => pure 0
  | xs => pure (xs.foldl (· + ·) 0 / (xs.length : α))

/-- Mean scalar loss over a dataset (materialized via `dataset.toList`). -/
def meanLossDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Add α] [Div α] [Zero
      α] [Coe Nat α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TList α [σ, τ])) :
    IO α :=
  meanLoss runner dataset.toList

/-- Scalar loss for one sample through the instantiated runtime module. -/
def moduleLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TList α [σ, τ]) : IO α := do
  let loss ← API.TorchLean.Module.forward runner.module sample
  pure (Spec.Tensor.toScalar loss)

/--
Fit on a small in-memory list of supervised samples for a fixed number of steps.

This is the simplest training-loop helper: it is intended for examples and small synthetic datasets.
For loader-based training, see `fitLoader`.
-/
def fit {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Add α] [Div α] [Zero
      α] [Coe Nat α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig) (samples : List (API.TorchLean.TList α [σ, τ])) :
    IO (FitReport α) := do
  trainMode runner
  let before ← meanLoss runner samples
  let _ ←
    match cfg.optimizer with
    | .sgd lr momentum =>
        if momentum == 0.0 then
          if samples.isEmpty then
            pure ()
          else
            let mut rest := samples
            for stepIdx in [0:cfg.steps] do
              if rest.isEmpty then
                rest := samples
              let sample ←
                match rest with
                | [] => throw <| IO.userError "Supervised.fit: empty sample cycle"
                | sample :: rest' =>
                    rest := rest'
                    pure sample
              updateRunnerBuffers runner sample
              let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
              API.TorchLean.Module.step runner.module lrα sample
              if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
                let loss ← API.TorchLean.Module.forward runner.module sample
                IO.println s!"step {stepIdx}: loss={Spec.Tensor.toScalar loss}"
        else
          match cfg.scheduler with
          | none =>
              let opt := API.TorchLean.Optim.momentumSGD
                (α := α) (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
              let st0 ← API.TorchLean.Module.initOptim runner.module opt
              let _ ← API.TorchLean.Module.trainWith runner.module opt st0 cfg.steps samples
                (logEvery := cfg.logEvery)
              pure ()
          | some _ =>
              let opt := API.TorchLean.Optim.momentumSGD
                (α := α) (paramShapes := paramShapes task)
                (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
              let st0 : API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α (paramShapes
                task) ←
                API.TorchLean.Module.initOptim runner.module opt
              let mut st := st0
              if samples.isEmpty then
                pure ()
              else
                let mut rest := samples
                for stepIdx in [0:cfg.steps] do
                  if rest.isEmpty then
                    rest := samples
                  let sample ←
                    match rest with
                  | [] => throw <| IO.userError "Supervised.fit: empty sample cycle"
                  | sample :: rest' =>
                      rest := rest'
                      pure sample
                  updateRunnerBuffers runner sample
                  let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
                  st := momentumSGDStateWithLR (paramShapes := paramShapes task) lrα st
                  st ← API.TorchLean.Module.stepWith runner.module opt st sample
                  if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
                    let loss ← API.TorchLean.Module.forward runner.module sample
                    IO.println s!"step {stepIdx}: loss={Spec.Tensor.toScalar loss}"
    | .adam lr beta1 beta2 epsilon =>
        match cfg.scheduler with
        | none =>
            let opt := API.TorchLean.Optim.adam
              (α := α) (lr := API.Runtime.ofFloat lr)
              (beta1 := API.Runtime.ofFloat beta1)
              (beta2 := API.Runtime.ofFloat beta2)
              (epsilon := API.Runtime.ofFloat epsilon)
            let st0 ← API.TorchLean.Module.initOptim runner.module opt
            let _ ← API.TorchLean.Module.trainWith runner.module opt st0 cfg.steps samples
              (logEvery := cfg.logEvery)
            pure ()
        | some _ =>
            let opt := API.TorchLean.Optim.adam
              (α := α) (paramShapes := paramShapes task)
              (lr := API.Runtime.ofFloat lr)
              (beta1 := API.Runtime.ofFloat beta1)
              (beta2 := API.Runtime.ofFloat beta2)
              (epsilon := API.Runtime.ofFloat epsilon)
            let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adam.State α (paramShapes task) ←
              API.TorchLean.Module.initOptim runner.module opt
            let mut st := st0
            if samples.isEmpty then
              pure ()
            else
              let mut rest := samples
              for stepIdx in [0:cfg.steps] do
                if rest.isEmpty then
                  rest := samples
                let sample ←
                  match rest with
                | [] => throw <| IO.userError "Supervised.fit: empty sample cycle"
                | sample :: rest' =>
                    rest := rest'
                    pure sample
                updateRunnerBuffers runner sample
                let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
                st := adamStateWithLR (paramShapes := paramShapes task) lrα st
                st ← API.TorchLean.Module.stepWith runner.module opt st sample
                if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
                  let loss ← API.TorchLean.Module.forward runner.module sample
                  IO.println s!"step {stepIdx}: loss={Spec.Tensor.toScalar loss}"
    | .adamw lr weightDecay beta1 beta2 epsilon =>
        match cfg.scheduler with
        | none =>
            let opt := API.TorchLean.Optim.adamw
              (α := α) (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat
                weightDecay)
              (beta1 := API.Runtime.ofFloat beta1)
              (beta2 := API.Runtime.ofFloat beta2)
              (epsilon := API.Runtime.ofFloat epsilon)
            let st0 ← API.TorchLean.Module.initOptim runner.module opt
            let _ ← API.TorchLean.Module.trainWith runner.module opt st0 cfg.steps samples
              (logEvery := cfg.logEvery)
            pure ()
        | some _ =>
            let opt := API.TorchLean.Optim.adamw
              (α := α) (paramShapes := paramShapes task)
              (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
              (beta1 := API.Runtime.ofFloat beta1)
              (beta2 := API.Runtime.ofFloat beta2)
              (epsilon := API.Runtime.ofFloat epsilon)
            let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α (paramShapes task) ←
              API.TorchLean.Module.initOptim runner.module opt
            let mut st := st0
            if samples.isEmpty then
              pure ()
            else
              let mut rest := samples
              for stepIdx in [0:cfg.steps] do
                if rest.isEmpty then
                  rest := samples
                let sample ←
                  match rest with
                | [] => throw <| IO.userError "Supervised.fit: empty sample cycle"
                | sample :: rest' =>
                    rest := rest'
                    pure sample
                updateRunnerBuffers runner sample
                let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
                st := adamwStateWithLR (paramShapes := paramShapes task) lrα st
                st ← API.TorchLean.Module.stepWith runner.module opt st sample
                if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
                  let loss ← API.TorchLean.Module.forward runner.module sample
                  IO.println s!"step {stepIdx}: loss={Spec.Tensor.toScalar loss}"
  let after ← meanLoss runner samples
  pure { before := before, after := after }

/-- `fit` over a dataset (materialized as a list). -/
def fitDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Add α] [Div α] [Zero
      α] [Coe Nat α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TList α [σ, τ])) :
    IO (FitReport α) :=
  fit runner cfg dataset.toList

/--
Fit over a `DataLoader` for `cfg.epochs` epochs, returning the final report and the updated loader.

This corresponds to the common PyTorch pattern:
`for epoch in ...: for batch in loader: step(batch)`.
-/
def fitLoader {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Add α] [Div α] [Zero
      α] [Coe Nat α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (dl : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) := do
  trainMode runner
  let nextEpoch
      (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) :
      IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]) ×
        List (List (API.TorchLean.TList α [σ, τ]))) :=
    match _root_.Runtime.Autograd.Train.DataLoader.epoch "Supervised.fitLoader" loader with
    | .ok out => pure out
    | .error msg => throw <| IO.userError s!"Supervised.fitLoader: {msg}"

  let before ← meanLossDataset runner dl.dataset

  match cfg.optimizer with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let rec trainSgdBatches
            (epoch : Nat)
            (batches : List (List (API.TorchLean.TList α [σ, τ]))) :
            IO Unit := do
          match batches with
          | [] => pure ()
          | batch :: rest =>
              let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch)
              for sample in batch do
                updateRunnerBuffers runner sample
                API.TorchLean.Module.step runner.module lrα sample
              trainSgdBatches epoch rest

        let rec runSgdEpochs (remaining : Nat)
            (epoch : Nat)
            (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) :
            IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) := do
          match remaining with
          | 0 => pure loader
          | n + 1 =>
              let (loader', batches) ← nextEpoch loader
              trainSgdBatches epoch batches
              runSgdEpochs n (epoch + 1) loader'

        let dl' ← runSgdEpochs cfg.epochs 0 dl
        let after ← meanLossDataset runner dl'.dataset
        pure ({ before := before, after := after }, dl')
      else
        let opt := API.TorchLean.Optim.momentumSGD
          (α := α) (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
        let st0 ← API.TorchLean.Module.initOptim runner.module opt

        let rec trainMomSamples (epoch stepIdx : Nat) (state : opt.State)
            (samples : List (API.TorchLean.TList α [σ, τ])) :
            IO (opt.State × Nat) := do
          match samples with
          | [] => pure (state, stepIdx)
          | sample :: rest =>
              updateRunnerBuffers runner sample
              let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
              if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
                let loss ← API.TorchLean.Module.forward runner.module sample
                IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
              trainMomSamples epoch (stepIdx + 1) state' rest

        let rec trainMomBatches (epoch stepIdx : Nat) (state : opt.State)
            (batches : List (List (API.TorchLean.TList α [σ, τ]))) :
            IO (opt.State × Nat) := do
          match batches with
          | [] => pure (state, stepIdx)
          | batch :: rest =>
              let (state', stepIdx') ← trainMomSamples epoch stepIdx state batch
              trainMomBatches epoch stepIdx' state' rest

        let rec runMomEpochs (epoch remaining : Nat)
            (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]))
            (st : opt.State) :
            IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]) × opt.State)
              := do
          match remaining with
          | 0 => pure (loader, st)
          | n + 1 =>
              let (loader', batches) ← nextEpoch loader
              let stSched : opt.State :=
                match cfg.scheduler with
                | none => st
                | some _ =>
                    momentumSGDStateWithLR
                      (paramShapes := paramShapes task)
                      (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                      st
              let (st', _) ← trainMomBatches epoch 0 stSched batches
              runMomEpochs (epoch + 1) n loader' st'

        let (dl', _) ← runMomEpochs 0 cfg.epochs dl st0
        let after ← meanLossDataset runner dl'.dataset
        pure ({ before := before, after := after }, dl')

  | .adam lr beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adam
        (α := α) (lr := API.Runtime.ofFloat lr)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdamSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TList α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdamSamples epoch (stepIdx + 1) state' rest

      let rec trainAdamBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TList α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdamSamples epoch stepIdx state batch
            trainAdamBatches epoch stepIdx' state' rest

      let rec runAdamEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adamStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdamBatches epoch 0 stSched batches
            runAdamEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdamEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adamw
        (α := α) (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdamWSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TList α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdamWSamples epoch (stepIdx + 1) state' rest

      let rec trainAdamWBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TList α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdamWSamples epoch stepIdx state batch
            trainAdamWBatches epoch stepIdx' state' rest

      let rec runAdamWEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adamwStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdamWBatches epoch 0 stSched batches
            runAdamWEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdamWEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

/--
Stateful training loop object: a `Runner` plus an optimizer state and a step counter.

This is the TorchLean analogue of holding a PyTorch `optimizer` object plus the model, ready to
`step()` on batches.
-/
structure Stepper (α : Type) [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Underlying task runner (module + compiled predictors/losses). -/
  runner : Runner α task
  /-- Run a single optimization step on one supervised sample, returning the loss value. -/
  stepSample : API.TorchLean.TList α [σ, τ] → IO α
  /-- Run an epoch over an explicit list of samples, returning the per-step loss values. -/
  epochSamples : List (API.TorchLean.TList α [σ, τ]) → IO (List α)
  /-- Read the total number of `stepSample` calls performed so far. -/
  stepCount : IO Nat

/--
Construct a `Stepper` for a runner, optimizer config, and optional scheduler.

This is the recommended way to build custom training loops without reimplementing the optimizer
logic: call `stepper`, then choose `stepSample` for single batches or `epochSamples` for explicit
sample lists.
-/
def stepper {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (runner : Runner α task) (optimizer : OptimizerConfig)
    (scheduler : Option API.TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) := do
  trainMode runner
  let stepRef ← IO.mkRef 0
  match optimizer with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let runStep := fun (sample : API.TorchLean.TList α [σ, τ]) => do
          trainMode runner
          let stepIdx ← stepRef.get
          let loss ← moduleLoss runner sample
          updateRunnerBuffers runner sample
          let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          API.TorchLean.Module.step runner.module lrα sample
          stepRef.set (stepIdx + 1)
          pure loss
        pure {
          runner := runner
          stepSample := runStep
          epochSamples := fun samples => samples.mapM runStep
          stepCount := stepRef.get
        }
      else
        let opt := API.TorchLean.Optim.momentumSGD
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α (paramShapes task)
          ←
          API.TorchLean.Module.initOptim runner.module opt
        let stRef ← IO.mkRef st0
        let runStep := fun (sample : API.TorchLean.TList α [σ, τ]) => do
          trainMode runner
          let stepIdx ← stepRef.get
          let loss ← moduleLoss runner sample
          updateRunnerBuffers runner sample
          let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          let st0 ← stRef.get
          let st := momentumSGDStateWithLR (paramShapes := paramShapes task) lrα st0
          let st' ← API.TorchLean.Module.stepWith runner.module opt st sample
          stRef.set st'
          stepRef.set (stepIdx + 1)
          pure loss
        pure {
          runner := runner
          stepSample := runStep
          epochSamples := fun samples => samples.mapM runStep
          stepCount := stepRef.get
        }
  | .adam lr beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adam
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adam.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TList α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        let loss ← moduleLoss runner sample
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamStateWithLR (paramShapes := paramShapes task) lrα st0
        let st' ← API.TorchLean.Module.stepWith runner.module opt st sample
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure loss
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adamw
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TList α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        let loss ← moduleLoss runner sample
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamwStateWithLR (paramShapes := paramShapes task) lrα st0
        let st' ← API.TorchLean.Module.stepWith runner.module opt st sample
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure loss
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }

/-- Run one optimization step on a single supervised sample. -/
def step {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : API.TorchLean.TList α [σ, τ]) : IO α :=
  loop.stepSample sample

/-- Run one epoch over a list of supervised samples, returning the per-step losses. -/
def epoch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (API.TorchLean.TList α [σ, τ])) : IO (List α) :=
  loop.epochSamples samples

end Supervised

namespace Trainer

/-!
`NN.API.TorchLean.Trainer` is a stable facade over the internal `Supervised` training machinery.

Usage note:

If you're writing tutorials or user code, prefer the PyTorch-shaped entrypoint `NN.API.train`
(available after `import NN`). This `API.TorchLean.Trainer` namespace is the underlying runtime
layer that `API.train` delegates to, and it exposes more knobs than most users need.

The intended workflow is:
- pick a `Task` (regression / classification),
- call `instantiate` to get a `Runner` (parameters + buffers + backend state),
- call `fit` / `fitDataset` / `fitLoader`, or build a `Stepper` for custom loops.

This API is backend-agnostic: the same code can run in `.eager` mode or via a compiled backend,
depending on the `backend` argument passed to `instantiate`.
-/

@[inherit_doc Supervised.SeqTask]
abbrev Task := Supervised.SeqTask
@[inherit_doc Supervised.Runner]
abbrev Runner := Supervised.Runner
@[inherit_doc Supervised.OptimizerConfig]
abbrev Optimizer := Supervised.OptimizerConfig
@[inherit_doc Supervised.FitConfig]
abbrev FitConfig := Supervised.FitConfig
@[inherit_doc Supervised.LoaderFitConfig]
abbrev LoaderFitConfig := Supervised.LoaderFitConfig
@[inherit_doc Supervised.FitReport]
abbrev FitReport := Supervised.FitReport
@[inherit_doc Supervised.Stepper]
abbrev Stepper := Supervised.Stepper

/-- Regression task with mean-squared error loss by default. -/
def regression {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    Task σ τ :=
  Supervised.SeqTask.mse model reduction

/-- One-hot classification task with cross-entropy loss by default. -/
def classificationOneHot {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    Task σ τ :=
  Supervised.SeqTask.crossEntropyOneHot model reduction

/--
SGD optimizer config.

PyTorch analogue: `torch.optim.SGD`
  (`https://pytorch.org/docs/stable/generated/torch.optim.SGD.html`).
-/
def sgd (lr : Float) (momentum : Float := 0.0) : Optimizer := .sgd lr momentum

/--
Momentum SGD optimizer config (PyTorch-style default `momentum = 0.9`).

This is just `sgd lr momentum` with a different default.
-/
def momentumSGD (lr : Float) (momentum : Float := 0.9) : Optimizer := .sgd lr momentum

/--
Adam optimizer config with standard defaults.

PyTorch analogue: `torch.optim.Adam`
  (`https://pytorch.org/docs/stable/generated/torch.optim.Adam.html`).
-/
def adam (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8) :
    Optimizer :=
  .adam lr beta1 beta2 epsilon

/--
AdamW optimizer config with standard defaults (PyTorch-style `weightDecay = 0.01`).

PyTorch analogue: `torch.optim.AdamW`
  (`https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html`).
-/
def adamw (lr : Float) (weightDecay : Float := 0.01)
    (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8) :
    Optimizer :=
  .adamw lr weightDecay beta1 beta2 epsilon

/-- Fixed-step training config over an in-memory sample list or dataset. -/
def steps (count : Nat) (optimizer : Optimizer := sgd 0.01) (logEvery : Nat := 1) : FitConfig :=
  { steps := count, optimizer := optimizer, logEvery := logEvery }

/-- Epoch-based training config over a data loader. -/
def epochs (count : Nat) (optimizer : Optimizer := sgd 0.01) (logEvery : Nat := 1) :
    LoaderFitConfig :=
  { epochs := count, optimizer := optimizer, logEvery := logEvery }

/-- Attach a scheduler to a step-based training config. -/
def withScheduler (cfg : FitConfig) (scheduler : API.TorchLean.Schedulers.Config) : FitConfig :=
  { cfg with scheduler := some scheduler }

/-- Attach a scheduler to an epoch-based loader training config. -/
def withEpochScheduler (cfg : LoaderFitConfig) (scheduler : API.TorchLean.Schedulers.Config) :
    LoaderFitConfig :=
  { cfg with scheduler := some scheduler }

/-- Step-based constant learning-rate schedule. -/
def constantLR (cfg : FitConfig) (lr : Float) : FitConfig :=
  withScheduler cfg (.constant lr)

/-- Step-based step-decay schedule. -/
def stepLR (cfg : FitConfig) (base : Float) (stepSize : Nat) (gamma : Float := 0.1) : FitConfig :=
  withScheduler cfg (.step base stepSize gamma)

/-- Step-based exponential-decay schedule. -/
def exponentialLR (cfg : FitConfig) (base : Float) (gamma : Float) : FitConfig :=
  withScheduler cfg (.exponential base gamma)

/-- Epoch-based constant learning-rate schedule. -/
def constantEpochLR (cfg : LoaderFitConfig) (lr : Float) : LoaderFitConfig :=
  withEpochScheduler cfg (.constant lr)

/-- Epoch-based step-decay schedule. -/
def stepEpochLR (cfg : LoaderFitConfig) (base : Float) (stepSize : Nat) (gamma : Float := 0.1) :
    LoaderFitConfig :=
  withEpochScheduler cfg (.step base stepSize gamma)

/-- Epoch-based exponential-decay schedule. -/
def exponentialEpochLR (cfg : LoaderFitConfig) (base : Float) (gamma : Float) : LoaderFitConfig :=
  withEpochScheduler cfg (.exponential base gamma)

/--
Instantiate a runner under explicit Torch options (`backend`, `useGpu`, `fastKernels`, ...).

This is the recommended entrypoint when you want CUDA eager execution from the training helpers
without dropping down to `TorchLean.Module` directly.
-/
def instantiateWithOptions {σ τ : Spec.Shape} (task : Task σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) :=
  Supervised.instantiateWithRuntimeOptions (task := task) (α := α) opts

/--
Instantiate a runner (parameters + buffers + backend state) for the given task.

This allocates and initializes model parameters (via `Seq.initParams`) and sets up the chosen
execution backend (`.eager` vs `.compiled`).
-/
def instantiate {σ τ : Spec.Shape} (task : Task σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) :=
  instantiateWithOptions (task := task) (α := α) { backend := backend }

/--
CLI-oriented runner entry point.

This parses dtype/backend flags (via `NN.API.DType` / `Module.ExecConfig`) and then calls the
continuation `k` under the selected scalar backend.
-/
def run {σ τ : Spec.Shape} (task : Task σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] → Runner α task → List String → IO Unit) :
    IO Unit :=
  Supervised.run task args k

/-- Get the current model parameters from a runner. -/
def params {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (API.TorchLean.TList α (Supervised.paramShapes task)) :=
  Supervised.params runner

/-- Read the current mode (train vs eval). -/
def mode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO API.TorchLean.NN.Mode :=
  Supervised.mode runner

/-- Set the mode (train vs eval). -/
def setMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (value : API.TorchLean.NN.Mode) : IO Unit :=
  Supervised.setMode runner value

/-- Switch to training mode. -/
def trainMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  Supervised.trainMode runner

/-- Switch to eval mode. -/
def evalMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  Supervised.evalMode runner

/-- Check whether the runner is in training mode. -/
def isTraining {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Bool :=
  Supervised.isTraining runner

/-- Run forward+backward on one supervised sample and return gradients for all parameters. -/
def backward {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TList α [σ, τ]) :
    IO (API.TorchLean.TList α (Supervised.paramShapes task)) :=
  Supervised.backward runner sample

/--
Predict on a single input tensor.

This runs the forward pass under the runner's current mode.
-/
def predict {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) :=
  Supervised.predict runner x

/-- Predict on a list of inputs (runs the forward pass repeatedly). -/
def predictBatch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) :=
  Supervised.predictBatch runner xs

/--
Predict the argmax class for a classification task, if `argmax` is well-defined for `α`.

This is a convenience wrapper over `predict` + `Metrics.argmax?`.
-/
def predictClass? {σ : Spec.Shape} {n : Nat} {task : Task σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Option (Fin n)) :=
  Supervised.predictClass? (task := task) runner x

/-- Count correct predictions in a one-hot labeled sample list (returns `(correct, total)`). -/
def accuracyOneHot {σ : Spec.Shape} {n : Nat} {task : Task σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task) (samples : List (API.TorchLean.TList α [σ, .dim n .scalar])) :
    IO (Nat × Nat) :=
  Supervised.accuracyOneHot (task := task) runner samples

/-- Mean loss over an explicit list of samples. -/
def meanLoss {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task) (samples : List (API.TorchLean.TList α [σ, τ])) :
    IO α :=
  Supervised.meanLoss runner samples

/-- Mean loss over an entire `Dataset`. -/
def meanLossDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TList α [σ, τ])) :
    IO α :=
  Supervised.meanLossDataset runner dataset

/--
Fit on an explicit list of samples for a fixed number of steps.

Returns a small report with mean loss before/after.
-/
def fit {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig) (samples : List (API.TorchLean.TList α [σ, τ])) :
    IO (FitReport α) :=
  Supervised.fit runner cfg samples

/-- Fit on a `Dataset` for a fixed number of steps. -/
def fitDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TList α [σ, τ])) :
    IO (FitReport α) :=
  Supervised.fitDataset runner cfg dataset

/-- Fit using a `DataLoader` for a fixed number of epochs. -/
def fitLoader {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) :=
  Supervised.fitLoader runner cfg loader

/--
Construct a stateful stepper for custom loops.

This is useful if you want to control:
- evaluation cadence,
- logging,
- validation, early stopping, etc.

The returned `Stepper` still uses TorchLean's optimizer/scheduler implementations.
-/
def stepper {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (runner : Runner α task) (optimizer : Optimizer)
    (scheduler : Option API.TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) :=
  Supervised.stepper runner optimizer scheduler

/-- Run a single training step on one sample using a `Stepper`. -/
def step {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : API.TorchLean.TList α [σ, τ]) : IO α :=
  Supervised.step loop sample

/-- Run an epoch over a list of samples using a `Stepper` (returns the per-step losses). -/
def epoch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (API.TorchLean.TList α [σ, τ])) : IO (List α) :=
  Supervised.epoch loop samples

/--
Convenience: instantiate + fit on a list of samples.

Returns both the `Runner` (so you can keep using the trained parameters) and the fit report.
-/
def train {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : FitConfig)
    (samples : List (API.TorchLean.TList α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × FitReport α) := do
  let runner ← instantiate (task := task) (α := α) backend
  let report ← fit runner cfg samples
  pure (runner, report)

/--
Convenience: instantiate + fit on a `Dataset`.

Returns both the `Runner` and the fit report.
-/
def trainDataset {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : FitConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TList α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × FitReport α) := do
  let runner ← instantiate (task := task) (α := α) backend
  let report ← fitDataset runner cfg dataset
  pure (runner, report)

/--
Convenience: instantiate + fit using a `DataLoader`.

Returns the `Runner`, the fit report, and the updated loader state (shuffled epoch cursor).
-/
def trainLoader {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × FitReport α ×
      _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TList α [σ, τ])) := do
  let runner ← instantiate (task := task) (α := α) backend
  let (report, loader') ← fitLoader runner cfg loader
  pure (runner, report, loader')

end Trainer

/-
The remaining exports expose the imperative "session" interface.

Most users should start from `NN.API.Public` (or the runtime facade above). These exports are for:
- interactive/debug workflows that want mutable tape control, or
- advanced tooling that needs the low-level session primitives.
-/

/-- Execution config parsed from CLI flags (dtype/backend/fast-kernels). -/
abbrev ExecConfig := Module.ExecConfig

namespace ExecConfig
/-- Parse and strip execution flags, returning `(config, remainingArgs)`. -/
abbrev parseAndStrip := Module.ExecConfig.parseAndStrip
/-- Log the chosen execution config to stdout (useful for reproducible demos). -/
abbrev log := Module.ExecConfig.log
end ExecConfig

namespace ScalarTrainer
/-!
Re-export of the low-level imperative scalar trainer interface.

This exposes `forwardT`/`backwardT`/`stepT` from `Runtime.Autograd.TorchLean.ScalarTrainer`.
Most users should prefer the higher-level `NN.API.train` / `API.TorchLean.Trainer` APIs.
-/
export _root_.Runtime.Autograd.TorchLean.ScalarTrainer (forwardT backwardT stepT)
end ScalarTrainer

namespace Session
/-!
Imperative session API: a tape-backed interface that can run in eager or compiled mode.

This is roughly analogous to using PyTorch "eager tensors", except TorchLean makes the tape/session
explicit. The `Session` surface is useful for:
- interactive experiments in `IO`,
- debugging (inspect intermediate values),
- building higher-level runners.
-/
export _root_.Runtime.Autograd.TorchLean.Session
  (new resetTape param use input inputNat getNat setNat inputNatVec getNatVec setNatVec const
    getValue
   withFreshTape sgdStepScalarGraph
   add sub mul scale abs sqrt clamp max min
   broadcastTo reshape transpose2d transpose3dFirstToLast transpose3dLastToFirst
     transpose3dLastTwo
   swapAdjacentAtDepth
   reduceSum reduceMean
   gatherScalar gatherRow gatherScalarRef gatherRowRef gatherVecRef gatherRowsRef
   gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec scatterAddRow
   matmul bmm concatVectors concatDim0 sliceRange0 maxPool2d smoothMaxPool2d avgPool2d
   relu sigmoid tanh softmax softplus exp log safeLog sum flatten
   linear mseLoss layerNorm conv2d multiHeadAttention
   backwardDenseAll backwardScalarDenseAll grad sgdStepAll)
end Session

end TorchLean

end API
end NN
