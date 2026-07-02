/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Runtime
public import NN.API.Models
public import NN.API.Public.NN.Transformer
public import NN.API.RL
public import NN.API.Rand
public import NN.API.Samples.Bands
public import NN.API.Text.Bpe
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean Public Names

Short root names and training-log types used by `import NN`.
-/

@[expose] public section

namespace TorchLean

@[inherit_doc Spec.Shape]
abbrev Shape := Spec.Shape

/-- Runtime options such as backend, dtype, and CUDA fast-kernel settings. -/
abbrev Options := NN.API.TorchLean.Options

/-- One supervised training example with an input tensor and target tensor. -/
abbrev SupervisedSample := NN.API.SupervisedSample

/-- Shape-indexed sequential model. Most examples use the shorter `nn.Sequential` name. -/
abbrev SequentialModel := NN.API.nn.Sequential

/-- Randomized model builder used by `nn.run` and built-in examples. -/
abbrev ModelBuilder := NN.API.nn.M

/-- Parameter shapes required by a sequential model. -/
abbrev modelParamShapes {σ τ : Shape} (model : SequentialModel σ τ) : List Shape :=
  NN.API.nn.paramShapes model

/--
TorchLean's typed tuple of tensors.

A `TensorPack α [s₁, s₂, ...]` is a fixed tuple whose tensor shapes are tracked by the type-level
list.
-/
abbrev TensorPack (α : Type) (shapes : List Shape) :=
  NN.API.TensorPack α shapes

/--
Concrete parameter tensors for a model or model slice. The `nn.ParamTensors` spelling points back to
this same type.
-/
abbrev ParamTensors (α : Type) (shapes : List Shape) :=
  TensorPack α shapes

/-- Shape-indexed module definition used by the executable TorchLean training runtime. -/
abbrev ScalarModuleDef := NN.API.TorchLean.Module.ScalarModuleDef

/-- How a vector of per-example losses is reduced to one scalar loss. -/
abbrev LossReduction := NN.API.TorchLean.Loss.Reduction

/-- CSV parsing options used by the public data loaders. -/
abbrev CsvOptions := NN.API.Data.CsvOptions

namespace Training

/-- A scalar metric curve, usually a loss or accuracy series over training steps. -/
abbrev Curve := _root_.Runtime.Training.Curve

/-- JSON-serializable training log with metrics and run metadata. -/
abbrev TrainLog := _root_.Runtime.Training.TrainLog

/-- Mutable experiment log used by longer-running examples. -/
abbrev ExperimentLog := _root_.Runtime.Training.ExperimentLog

/-- Output destination for training logs. -/
abbrev LogDestination := _root_.Runtime.Training.LogDestination

/-- In-memory history for named training metrics. -/
abbrev MetricHistory := _root_.Runtime.Training.MetricHistory

/-- Finite in-memory dataset used by TorchLean trainers. -/
abbrev Dataset := _root_.Runtime.Autograd.Train.Dataset

/-- Stateful minibatch loader for finite datasets. -/
abbrev DataLoader := _root_.Runtime.Autograd.Train.DataLoader

namespace MetricHistory

export _root_.Runtime.Training.MetricHistory (empty)

end MetricHistory

end Training

end TorchLean
