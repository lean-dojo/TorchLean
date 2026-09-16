/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Module.Execution -- shake: keep
public import NN.Runtime.Autograd.Model.Layers.Seq -- shake: keep
import Mathlib.Algebra.Order.Algebra -- shake: keep
public import NN.API.Init -- shake: keep
public import NN.API.Runtime -- shake: keep
import NN.Spec.Core.Tensor -- shake: keep
public import NN.API.Neural.State -- shake: keep
public import NN.Tensor -- shake: keep
public import NN.Runtime.Autograd.Model.Functional.Core -- shake: keep
public import NN.Runtime.Autograd.Model.Functional.Einsum -- shake: keep
public import NN.Runtime.Autograd.Model.Functional.ShapeOps -- shake: keep
public import NN.Runtime.Autograd.Model.Module.RuntimeInit -- shake: keep
public import NN.Runtime.Autograd.Model.Layers.Activations -- shake: keep
public import NN.Runtime.Autograd.Model.Layers.Attention -- shake: keep
public import NN.Runtime.Autograd.Model.Layers.ConvPool -- shake: keep
public import NN.Runtime.Autograd.Model.Layers.Core -- shake: keep
public import NN.Runtime.Autograd.Model.Layers.Normalization -- shake: keep
public import NN.Runtime.Autograd.Model.Layers.Recurrent -- shake: keep
public import NN.Spec.Core.Shape -- shake: keep
import NN.Spec.Core.TensorReductionShape.Reductions -- shake: keep

@[expose] public section

namespace TorchLean

/-!
# Layer Construction

This module defines the shape-typed `Sequential` model type, the affine configuration record, and
the composition helpers shared by every layer. The public layer constructors live in
`NN.API.Seeded` and allocate initialization seeds from a deterministic stream.
-/

namespace nn

/-- Training-sensitive layer behavior. -/
abbrev Mode := Runtime.Autograd.Model.Layers.Mode

namespace Mode

export Runtime.Autograd.Model.Layers.Mode (train eval)

end Mode

/-- Parameter initialization for an affine layer. `none` selects Xavier-uniform weights. -/
structure Linear.Config where
  /--
  Weight initializer. `none` selects TorchLean's Xavier-uniform default
  (Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural
  networks", AISTATS 2010). Set an explicit scheme when reproducing a model whose
  initialization differs, or load the same parameter values when comparing runtimes.
  -/
  weightInitialization? : Option Init.Scheme := none
  /-- Bias initializer. The default starts every output coordinate at zero. -/
  biasInitialization : Init.Scheme := .zeros

namespace Linear.Config

/-- Validate affine dimensions and both parameter initializers. -/
def validate (config : Linear.Config) (inputWidth outputWidth : Nat) : Except String Unit := do
  if inputWidth = 0 then
    throw "Linear: input width must be positive"
  if outputWidth = 0 then
    throw "Linear: output width must be positive"
  match config.weightInitialization? with
  | none => pure ()
  | some initialization => initialization.validate
  config.biasInitialization.validate

end Linear.Config

/-- Sequential model type (TorchLean `Seq`), analogous to PyTorch `nn.Sequential`. -/
abbrev Sequential := Runtime.Autograd.Model.Layers.Seq

export Runtime.Autograd.Model.Layers (Layer)

/-!
Expose common `Seq` helpers under `TorchLean.nn`.

The names mirror the TorchLean runtime layer so users can move between the public API and
runtime layer code without learning a second vocabulary.
-/
export Runtime.Autograd.Model.Layers.Seq
  (stateShapes requiresGrad validate runtimeInit? hasBufferUpdates updateBuffers)

/-- Semantic initial values for every parameter and persistent buffer in a sequential model. -/
def initialState {σ τ : Spec.Shape} (model : Sequential σ τ) :
    State Float (stateShapes model) :=
  State.Internal.fromTensorPack
    (Runtime.Autograd.Model.Layers.Seq.initState model)

/-! Constructors that pair an immutable model with a scalar training loss. -/
namespace Objective

/--
Pair an immutable sequential model with a scalar loss.

The result is a public objective definition, not an executable module. Instantiate it to allocate
runtime state.
-/
def fromLoss {σ τ : Spec.Shape} (model : Sequential σ τ)
    (loss : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      Runtime.Autograd.Model.Program α [τ, τ] [])
    (mode : Mode := .train) :
    TorchLean.Module.ObjectiveDefinition Unit (stateShapes model) [σ, τ] :=
  Runtime.Autograd.Model.Layers.Seq.Objective.fromLoss model loss mode

/-- Pair an immutable sequential model with mean-squared error. -/
def meanSquaredError {σ τ : Spec.Shape} (model : Sequential σ τ)
    (reduction : TorchLean.Loss.Reduction := .mean)
    (mode : Mode := .train) :
    TorchLean.Module.ObjectiveDefinition Unit (stateShapes model) [σ, τ] :=
  Runtime.Autograd.Model.Layers.Seq.Objective.meanSquaredError model reduction mode

/-- Pair an immutable sequential model with one-hot cross entropy. -/
def oneHotCrossEntropy {σ τ : Spec.Shape} (model : Sequential σ τ)
    (axis : Nat)
    (reduction : TorchLean.Loss.Reduction := .mean)
    (mode : Mode := .train) :
    TorchLean.Module.ObjectiveDefinition Unit (stateShapes model) [σ, τ] :=
  Runtime.Autograd.Model.Layers.Seq.Objective.oneHotCrossEntropy model axis reduction mode

end Objective

namespace Sequential

/-- Sequential model that returns its input unchanged. -/
def identity (shape : Spec.Shape) : Sequential shape shape :=
  Runtime.Autograd.Model.Layers.Seq.id shape

/-- Lift one layer into a sequential model. -/
def fromLayer {σ τ : Spec.Shape} (layer : Layer σ τ) : Sequential σ τ :=
  Runtime.Autograd.Model.Layers.Seq.fromLayer layer

end Sequential

universe u v

/-!
### Internal helpers

Everything in `nn.Internal` is machinery the layer and model constructors share. It is not part of
the user-facing surface: if you are writing a model you should never need to name it. The sibling
namespace `nn.IndexedModel.Internal` plays the same role for indexed models, and keeping both under
`Internal` is what stops `nn.` autocompletion from filling up with plumbing.
-/

namespace Internal

/--
Build a model placeholder whose configuration is known to be invalid.

Layer constructors use this when a value-level configuration check is needed to construct a
shape-indexed model. Normal model validation rejects the placeholder before state allocation or
execution, so callers receive the same error path as every other invalid layer.

Why a placeholder instead of an `Except`? A `Sequential σ τ` is indexed by its shapes, so a
constructor that has already committed to `σ` and `τ` cannot back out and return an error value
without changing every caller's type. PyTorch has the same problem and solves it by raising at
construction time; we cannot raise inside a pure definition, so we return a model that is
guaranteed to fail `validate` with `message` before it ever touches storage.
-/
def invalidConfiguration (input output : Spec.Shape) (kind message : String) :
    Sequential input output :=
  Sequential.fromLayer
    { kind
      stateShapes := []
      initState := .nil
      runtimeInit := some .nil
      requiresGrad := #[]
      validateConfig := .error message
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun _input =>
            Runtime.Autograd.Torch.const (m := m) (α := α)
              (TorchLean.Tensor.zeros output) }

end Internal

/-- Convert a layer-like value to a sequential model for `nn.compose!` composition. -/
class AsSequential (F : Spec.Shape → Spec.Shape → Sort u) where
  asSequential : {σ τ : Spec.Shape} → F σ τ → Sequential σ τ

instance : AsSequential Layer where
  asSequential := Runtime.Autograd.Model.Layers.Seq.fromLayer

instance : AsSequential Sequential where
  asSequential := id

/-- Compose layers and sequential models accepted by the `nn.compose!` syntax. -/
def compose {σ τ υ : Spec.Shape}
    {F : Spec.Shape → Spec.Shape → Sort u} {G : Spec.Shape → Spec.Shape → Sort v}
    [AsSequential F] [AsSequential G] (f : F σ τ) (g : G τ υ) : Sequential σ υ :=
  Runtime.Autograd.Model.Layers.Seq.comp
    (AsSequential.asSequential f) (AsSequential.asSequential g)

end nn

end TorchLean
