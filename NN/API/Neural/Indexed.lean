/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders

@[expose] public section

namespace TorchLean

/-!
# Indexed Models and Embeddings

This module defines models with non-differentiable tensor inputs, their scalar objectives, and
embedding-table builders. Import `NN.API.Neural.Indexed` when constructing or executing an indexed
model; ordinary sequential builders remain in `NN.API.Neural.Builders`.
-/

namespace nn

/-- A shape-typed model with one non-differentiable tensor input. -/
structure IndexedModel (σ τ : Spec.Shape) (β : Type) [TorchLean.Storage β] where
  private mk ::
  /-- Shapes of trainable parameters and persistent buffers. -/
  stateShapes : List Spec.Shape
  private kindValue : String
  private initialStateValue : State Float stateShapes
  private initializationPlanValue : Option
    (Runtime.Autograd.Model.Module.RuntimeInit.Plan stateShapes)
  private trainableMaskValue : Array Bool
  private modelValidatorValue : Except String Unit
  private inputValidatorValue : Tensor β σ → Except String Unit
  private programValue : ∀ (_mode : Mode)
      {α : Type}, [TorchLean.Storage α] → [Context α] →
        Runtime.Autograd.Model.ProgramWithDataInputs α β stateShapes [σ] τ

namespace IndexedModel

namespace Internal

/-- Construct an indexed model from its complete runtime definition. -/
opaque create {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (stateShapes : List Spec.Shape)
    (initialState : State Float stateShapes)
    (program : ∀ (_mode : Mode)
      {α : Type}, [TorchLean.Storage α] → [Context α] →
        Runtime.Autograd.Model.ProgramWithDataInputs α β stateShapes [σ] τ)
    (kind : String := "IndexedModel")
    (initializationPlan : Option
      (Runtime.Autograd.Model.Module.RuntimeInit.Plan stateShapes) := none)
    (trainableMask : Array Bool := Array.replicate stateShapes.length true)
    (validateModel : Except String Unit := pure ())
    (validateInput : Tensor β σ → Except String Unit := fun _ => pure ()) :
    IndexedModel σ τ β :=
  ⟨stateShapes, kind, initialState, initializationPlan, trainableMask,
    validateModel, validateInput, program⟩

/--
Build an indexed-model placeholder whose static configuration is known to be invalid.

The placeholder owns no parameters or buffers and is rejected by `IndexedModel.validate` before
allocation or execution.
-/
def invalidConfiguration {β : Type} [TorchLean.Storage β]
    (input output : Spec.Shape) (kind message : String) :
    IndexedModel input output β :=
  create
    []
    State.empty
    (fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun _input =>
          Runtime.Autograd.Torch.const (m := m) (α := α)
            (TorchLean.Tensor.zeros output))
    (kind := kind)
    (initializationPlan := some .nil)
    (trainableMask := #[])
    (validateModel := .error message)

/-!
The readers below are `opaque` on purpose. `IndexedModel` keeps every field but `stateShapes`
private, so the only way to look inside is through this namespace, and an `opaque` reader is one
that `simp`, `decide` and `rfl` cannot unfold back into the field. Builders therefore stay free to
change how a model is represented without any downstream proof noticing, which is the whole reason
the fields were made private in the first place.
-/

/-- Label the builder gave this model, used in summaries and error messages. -/
opaque kind {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : String :=
  match model with
  | ⟨_, kind, _, _, _, _, _, _⟩ => kind

/-- Parameters and buffers the model starts from, in `stateShapes` order. -/
opaque initialState {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : State Float model.stateShapes :=
  match model with
  | ⟨_, _, initialState, _, _, _, _, _⟩ => initialState

/-- How the runtime should fill the state at allocation time, or `none` for an unplanned model. -/
opaque initializationPlan {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) :
    Option (Runtime.Autograd.Model.Module.RuntimeInit.Plan model.stateShapes) :=
  match model with
  | ⟨_, _, _, initializationPlan, _, _, _, _⟩ => initializationPlan

/-- One flag per state entry saying whether the optimizer may update it. -/
opaque trainableMask {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : Array Bool :=
  match model with
  | ⟨_, _, _, _, trainableMask, _, _, _⟩ => trainableMask

/-- Static verdict on the configuration, checked before anything is allocated. -/
opaque validateModel {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : Except String Unit :=
  match model with
  | ⟨_, _, _, _, _, validateModel, _, _⟩ => validateModel

/-- Verdict on one concrete input, for conditions the shape type cannot express. -/
opaque validateInput {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) (input : Tensor β σ) : Except String Unit :=
  match model with
  | ⟨_, _, _, _, _, _, validateInput, _⟩ => validateInput input

/-- The runtime program for a given `Mode`, scalar-polymorphic in its execution type. -/
opaque program {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) (mode : Mode)
    {α : Type} [TorchLean.Storage α] [Context α] :
    Runtime.Autograd.Model.ProgramWithDataInputs α β model.stateShapes [σ] τ :=
  match model with
  | ⟨_, _, _, _, _, _, _, program⟩ => program mode

end Internal

/-- Model label used in summaries and diagnostics. -/
def kind {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : String :=
  Internal.kind model

/-- Semantic initial values for the complete model state. -/
def initialState {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : State Float model.stateShapes :=
  Internal.initialState model

/-- Gradient flags aligned with `stateShapes`. -/
def requiresGrad {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : Array Bool :=
  Internal.trainableMask model

/-- Validate static model configuration before allocation or graph execution. -/
def validate {σ τ : Spec.Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : Except String Unit :=
  do
    unless model.requiresGrad.size = model.stateShapes.length do
      throw (s!"{model.kind}: expected {model.stateShapes.length} requiresGrad flags, "
        ++ s!"got {model.requiresGrad.size}")
    Internal.validateModel model
    match Internal.initializationPlan model with
    | some plan => plan.validate
    | none => pure ()

/-- Append an ordinary sequential model after an indexed-input model. -/
def andThen {β : Type} [TorchLean.Storage β]
    {σ τ υ : Spec.Shape} (first : IndexedModel σ τ β)
    (rest : Sequential τ υ) : IndexedModel σ υ β :=
  Internal.create
    (first.stateShapes ++ Runtime.Autograd.Model.Layers.Seq.stateShapes rest)
    (first.initialState.append (nn.initialState rest))
    (fun mode {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun s => TorchLean.Runtime.ValueRef (m := m) (α := α) s)
          (ss := first.stateShapes ++ Runtime.Autograd.Model.Layers.Seq.stateShapes rest)
          (β := Runtime.Autograd.Torch.CurriedRef
            (fun s => Runtime.Autograd.Torch.DataRef (m := m) (α := α) β s)
            [σ] (m (TorchLean.Runtime.ValueRef (m := m) (α := α) υ)))
          (fun state =>
            let (firstState, restState) := Runtime.Autograd.Torch.RefList.split
              (Ref := fun s => TorchLean.Runtime.ValueRef (m := m) (α := α) s)
              (ss₁ := first.stateShapes)
              (ss₂ := Runtime.Autograd.Model.Layers.Seq.stateShapes rest) state
            Runtime.Autograd.Torch.CurriedRef.curry
              (Ref := fun s => Runtime.Autograd.Torch.DataRef
                (m := m) (α := α) β s)
              (ss := [σ])
              (fun dataInputs =>
                match dataInputs with
                | .cons indices .nil => do
                    let embedded ←
                      (Runtime.Autograd.Torch.CurriedRef.uncurry
                        (Ref := fun s =>
                          TorchLean.Runtime.ValueRef (m := m) (α := α) s)
                        (ss := first.stateShapes)
                        (β := Runtime.Autograd.Torch.CurriedRef
                          (fun s => Runtime.Autograd.Torch.DataRef
                            (m := m) (α := α) β s)
                          [σ]
                          (m (TorchLean.Runtime.ValueRef (m := m) (α := α) τ)))
                        (Internal.program first mode (α := α)) firstState) indices
                    Runtime.Autograd.Model.Layers.Seq.forwardState
                      (model := rest) (α := α) (m := m) mode restState embedded)))
    (kind := s!"{first.kind} → Sequential")
    (initializationPlan :=
      match Internal.initializationPlan first,
          Runtime.Autograd.Model.Layers.Seq.runtimeInit? rest with
      | some firstPlan, some restPlan => some (firstPlan.append restPlan)
      | _, _ => none)
    (trainableMask := Internal.trainableMask first ++
      Runtime.Autograd.Model.Layers.Seq.requiresGrad rest)
    (validateModel := do
      first.validate
      Runtime.Autograd.Model.Layers.Seq.validate rest)
    (validateInput := Internal.validateInput first)

/-! ### Scalar objectives -/

namespace Objective

/--
Pair an indexed-input model with a scalar loss.

The resulting training module accepts one ordinary target tensor followed by the model's
non-differentiable input tensor. Keeping those packs separate ensures that indices cannot receive
gradients or be reinterpreted through the model's floating-point element type.

The target shape is independent of the model output shape, so this constructor
also supports losses whose labels use a different representation from the
prediction. Training mode is the default.
-/
def fromLoss {β : Type} [TorchLean.Storage β] {σ τ υ : Spec.Shape}
    (model : IndexedModel σ τ β)
    (loss : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      Runtime.Autograd.Model.Program α [τ, υ] [])
    (mode : Mode := .train) :
    TorchLean.Module.ObjectiveDefinition β
      model.stateShapes [υ] [σ] :=
  { initState := State.Internal.toTensorPack model.initialState
    runtimeInit := Internal.initializationPlan model
    requiresGrad := Internal.trainableMask model
    validate := model.validate
    validateDataInputs := fun
      | .cons input .nil => Internal.validateInput model input
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun s => TorchLean.Runtime.ValueRef (m := m) (α := α) s)
          (ss := model.stateShapes ++ [υ])
          (β := Runtime.Autograd.Torch.CurriedRef
            (fun s => Runtime.Autograd.Torch.DataRef (m := m) (α := α) β s)
            [σ]
            (m (TorchLean.Runtime.ValueRef
              (m := m) (α := α) [])))
          (fun arguments =>
            let (state, targets) := Runtime.Autograd.Torch.RefList.split
              (Ref := fun s => TorchLean.Runtime.ValueRef (m := m) (α := α) s)
              (ss₁ := model.stateShapes) (ss₂ := [υ]) arguments
            let .cons target .nil := targets
            Runtime.Autograd.Torch.CurriedRef.curry
              (Ref := fun s =>
                Runtime.Autograd.Torch.DataRef (m := m) (α := α) β s)
              (ss := [σ])
              (β := m (TorchLean.Runtime.ValueRef
                (m := m) (α := α) []))
              (fun dataInputs =>
                match dataInputs with
                | .cons indices .nil => do
                    let withInput := Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        TorchLean.Runtime.ValueRef (m := m) (α := α) s)
                      (ss := model.stateShapes)
                      (β := Runtime.Autograd.Torch.CurriedRef
                        (fun s => Runtime.Autograd.Torch.DataRef
                          (m := m) (α := α) β s)
                        [σ]
                        (m (TorchLean.Runtime.ValueRef (m := m) (α := α) τ)))
                      (Internal.program model mode (α := α)) state
                    let prediction ← withInput indices
                    Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        TorchLean.Runtime.ValueRef (m := m) (α := α) s)
                      (ss := [τ, υ]) (loss (α := α) (m := m))
                      (.cons prediction (.cons target .nil)))) }

/-- Pair an indexed-input model with mean-squared error. -/
def mse {β : Type} [TorchLean.Storage β]
    {σ τ : Spec.Shape} (model : IndexedModel σ τ β)
    (reduction : TorchLean.Loss.Reduction := .mean)
    (mode : Mode := .train) :
    TorchLean.Module.ObjectiveDefinition β
      model.stateShapes [τ] [σ] :=
  fromLoss model
    (fun {α} _ _ => fun {m} _ _ => fun prediction target =>
      TorchLean.Loss.mse (m := m) (α := α) (s := τ) prediction target
        (reduction := reduction))
    (mode := mode)

end Objective

end IndexedModel

/--
A reusable trainable embedding table with `vocabularySize` rows and vectors of length
`embeddingWidth`.

Unlike `IndexedModel`, this definition is independent of the eventual token-tensor shape. Calling
`table.model indices` specializes it to an input shape with bounded `Fin vocabularySize` indices.
Instantiating that model produces mutable weight storage; the definition itself remains immutable
so it can be lowered and used in proofs.
-/
structure Embedding (vocabularySize embeddingWidth : Nat) where
  private mk ::
  private initialWeightValue : Tensor Float [vocabularySize, embeddingWidth]
  private initializationPlanValue : Runtime.Autograd.Model.Module.RuntimeInit.Plan
    [[vocabularySize, embeddingWidth]]
  private trainableValue : Bool
  private validationValue : Except String Unit

namespace Embedding

namespace Internal

/-- Construct an embedding table at the runtime-builder boundary. -/
opaque create {vocabularySize embeddingWidth : Nat}
    (initialWeight : Tensor Float [vocabularySize, embeddingWidth])
    (initializationPlan : Runtime.Autograd.Model.Module.RuntimeInit.Plan
      [[vocabularySize, embeddingWidth]])
    (trainable : Bool := true)
    (validation : Except String Unit := pure ()) :
    Embedding vocabularySize embeddingWidth :=
  ⟨initialWeight, initializationPlan, trainable, validation⟩

/-- The table's starting weight matrix, one row per vocabulary entry. -/
opaque initialWeight {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) :
    Tensor Float [vocabularySize, embeddingWidth] :=
  match table with
  | ⟨weight, _, _, _⟩ => weight

/-- How the runtime should fill that matrix at allocation time. -/
opaque initializationPlan {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) :
    Runtime.Autograd.Model.Module.RuntimeInit.Plan
      [[vocabularySize, embeddingWidth]] :=
  match table with
  | ⟨_, plan, _, _⟩ => plan

/-- Whether the optimizer may update the table, false for a frozen lookup. -/
opaque isTrainable {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) : Bool :=
  match table with
  | ⟨_, _, trainable, _⟩ => trainable

/-- Static verdict on the table's configuration; `Embedding.invalid` is how it becomes an error. -/
opaque validation {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) : Except String Unit :=
  match table with
  | ⟨_, _, _, validation⟩ => validation

/-- Construct a rejected table value without exposing parameter state through `Embedding.model`. -/
def invalid (vocabularySize embeddingWidth : Nat) (message : String) :
    Embedding vocabularySize embeddingWidth :=
  create
    (Tensor.zeros [vocabularySize, embeddingWidth])
    (.cons .zeros .nil)
    (trainable := false)
    (validation := .error message)

end Internal

/-- Initial table values, with one row per vocabulary item. -/
def initialWeight {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) :
    Tensor Float [vocabularySize, embeddingWidth] :=
  Internal.initialWeight table

/-- Construction options for a freshly initialized embedding table. -/
structure Config where
  /--
  Initialization scheme for the table.

  The default agrees with `torch.nn.Embedding.reset_parameters`: independent samples from the
  standard normal distribution. Language-model constructors normally override this with their
  architecture-specific initialization, such as GPT-2's standard deviation `0.02`.
  -/
  weightInitialization : Init.Scheme := .normal 0.0 1.0
  /-- Freeze the table by excluding it from reverse-mode parameter gradients. -/
  freeze : Bool := false

namespace Config

/-- Validate dimensions shared by configured and exact-weight embeddings. -/
def validateDimensions (vocabularySize embeddingWidth : Nat) : Except String Unit := do
  if vocabularySize = 0 then
    throw "Embedding: vocabulary size must be positive"
  if embeddingWidth = 0 then
    throw "Embedding: embedding width must be positive"

/-- Validate embedding dimensions and initialization. -/
def validate (config : Config) (vocabularySize embeddingWidth : Nat) : Except String Unit := do
  validateDimensions vocabularySize embeddingWidth
  config.weightInitialization.validate

end Config

/--
Construct an embedding from an exact initial weight table.

This is the typed counterpart of passing `_weight` to `torch.nn.Embedding`. The supplied tensor
fixes both dimensions at compile time and is also recorded as an exact row-major runtime
initializer, so CPU and CUDA module construction start from the same payload.
-/
def fromWeight {vocabularySize embeddingWidth : Nat}
    (weight : Tensor Float [vocabularySize, embeddingWidth])
    (freeze : Bool := false) : Embedding vocabularySize embeddingWidth :=
  Internal.create weight
    (.cons (.flat (weight.to FloatArray)) .nil)
    (trainable := !freeze)
    (validation := Config.validateDimensions vocabularySize embeddingWidth)

/-- Specialize an embedding table to a concrete index-tensor shape. -/
def model {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) (input : Spec.Shape) :
    IndexedModel input
      (input.appendDim embeddingWidth) (Fin vocabularySize) :=
  match Internal.validation table with
  | .error message =>
      IndexedModel.Internal.invalidConfiguration
        input (input.appendDim embeddingWidth)
        s!"Embedding({vocabularySize}, {embeddingWidth})" message
  | .ok () =>
      let weightShape : Spec.Shape := [vocabularySize, embeddingWidth]
      IndexedModel.Internal.create
        [weightShape]
        (State.empty.push table.initialWeight)
        (fun _ {α} _ _ =>
          fun {m} _ _ => fun weight =>
            Runtime.Autograd.Torch.CurriedRef.curry
              (Ref := fun s => Runtime.Autograd.Torch.DataRef
                (m := m) (α := α) (Fin vocabularySize) s)
              (ss := [input])
              (fun dataInputs =>
                match dataInputs with
                | .cons tokenIds .nil =>
                    Runtime.Autograd.Model.F.embedding
                      (m := m) (α := α) (vocabularySize := vocabularySize)
                      (embeddingWidth := embeddingWidth) weight tokenIds))
        (kind := s!"Embedding({vocabularySize}, {embeddingWidth})")
        (initializationPlan := some (Internal.initializationPlan table))
        (trainableMask := #[Internal.isTrainable table])
        (validateModel := pure ())

end Embedding

end nn
end TorchLean
