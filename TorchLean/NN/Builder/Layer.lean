/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.Module.Execution -- shake: keep
public import TorchLean.NN.Sequential -- shake: keep
public import TorchLean.NN.Init.Scheme -- shake: keep
public import TorchLean.NN.Layer.Activation -- shake: keep
public import TorchLean.NN.Layer.Attention -- shake: keep
public import TorchLean.NN.Layer.ConvPool -- shake: keep
public import TorchLean.NN.Layer.Norm -- shake: keep
public import TorchLean.NN.Layer.Recurrent -- shake: keep

@[expose] public section

namespace TorchLean

/-!
# Layer Construction

The affine configuration record `Linear.Config` and the helpers every layer shares, such as
`compose`, `initialState` and `Sequential.invalid`. The model type `nn.Sequential` is declared
with the runtime layers in `TorchLean.NN.Sequential`. The public layer constructors live in
`TorchLean.NN.Builder` and draw their initialization seeds from a deterministic stream.
-/

namespace nn

/-- Initializers for the weight and the bias of an affine layer. -/
structure Linear.Config where
  /--
  Weight initializer. `none` selects TorchLean's Xavier-uniform default (Glorot and Bengio,
  "Understanding the difficulty of training deep feedforward neural networks", AISTATS 2010).
  Set a scheme here to reproduce a model with a different initialization, or load the same
  parameter values into both runtimes when comparing them.
  -/
  weightInit? : Option Init.Scheme := none
  /-- Bias initializer. The default starts every output coordinate at zero. -/
  biasInit : Init.Scheme := .zeros

namespace Linear.Config

/-- Check that both widths are positive and that both initializers pass `Init.Scheme.validate`. -/
def validate (config : Linear.Config) (inputWidth outputWidth : Nat) : Except String Unit := do
  if inputWidth = 0 then
    throw "Linear: input width must be positive"
  if outputWidth = 0 then
    throw "Linear: output width must be positive"
  match config.weightInit? with
  | none => pure ()
  | some initialization => initialization.validate
  config.biasInit.validate

end Linear.Config

/--
Initial parameters and buffers converted to the scalar type `α`, which defaults to `Float`.

Model builders store their seeded initial values in `Float` and the conversion maps each one with
`ofFloat`, so a wider `α` gets no extra random digits. For initial values with more precision than
`Float`, build a `TensorPack α` directly and pass it as `initialState?` when instantiating the
model or opening a trainer.
-/
def initialState {σ τ : Shape} (model : Sequential σ τ) (α : Type := Float)
    [TorchLean.Storage α] [FromFloat α] : TensorPack α (Sequential.stateShapes model) :=
  (Sequential.initState model).map
    (fun tensor => TorchLean.Tensor.map (ofFloat (α := α)) tensor)

universe u v

/-!
### Constructor helpers

The layer and model constructors share these helpers. A model never needs to name them.
-/

/--
Placeholder model for a configuration that failed a value-level check. Its `validateConfig` is
`.error message`, so model validation rejects it before state allocation or execution, on the same
error path as every other invalid layer.

Reporting the failure through validation lets pure constructors keep their `Sequential σ τ` type
and compose with valid layers. The forward body returns zeros of the output shape, which supplies
the required result type and nothing more.
-/
def Sequential.invalid (input output : Shape) (kind message : String) :
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
            Ops.const (m := m) (α := α)
              (TorchLean.Tensor.zeros output) }

/--
Layer-like types that convert to a `Sequential` model. `Layer` and `Sequential` have instances, so
`nn.compose!` accepts both.
-/
class AsSequential (F : Shape → Shape → Sort u) where
  /-- Convert a layer-like value to a `Sequential` model with the same shapes. -/
  asSequential : {σ τ : Shape} → F σ τ → Sequential σ τ

instance : AsSequential Layer where
  asSequential := Sequential.fromLayer

instance : AsSequential Sequential where
  asSequential := id

/--
Run `f`, then `g`, for two layer-like values (a `Layer` or a `Sequential`). The `nn.compose!` and
`nn.Sequential!` literals expand to nested calls of `compose`.
-/
def compose {σ τ υ : Shape}
    {F : Shape → Shape → Sort u} {G : Shape → Shape → Sort v}
    [AsSequential F] [AsSequential G] (f : F σ τ) (g : G τ υ) : Sequential σ υ :=
  Sequential.comp
    (AsSequential.asSequential f) (AsSequential.asSequential g)

end nn

end TorchLean
