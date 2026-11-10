/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Layer

@[expose] public section

namespace TorchLean

/-!
# Indexed Models and Embeddings

An `IndexedModel σ τ β` takes one non-differentiable input tensor with elements of type `β`, such
as token ids in `Fin vocabularySize`. The module also holds its scalar objectives
(`IndexedModel.Objective.fromLoss`, `IndexedModel.Objective.mse`) and the `Embedding` tables that
specialize to indexed models. Ordinary sequential builders live in `TorchLean.NN.Builder`.
-/

namespace nn

/-- A shape-typed model with one non-differentiable tensor input. -/
structure IndexedModel (σ τ : Shape) (β : Type) [TorchLean.Storage β] where
  private mk ::
  /-- Shapes of trainable parameters and persistent buffers. -/
  stateShapes : List Shape
  private kindValue : String
  private initialStateValue : TensorPack Float stateShapes
  private initializationPlanValue : Option
    (Init.Plan stateShapes)
  private trainableMaskValue : Array Bool
  private modelValidatorValue : Except String Unit
  private inputValidatorValue : Tensor β σ → Except String Unit
  private programValue : ∀ (_mode : Mode)
      {α : Type}, [TorchLean.Storage α] → [Context α] →
        Program.Data α β stateShapes [σ] τ

namespace IndexedModel

/-- Construct an indexed model from its complete runtime definition. -/
opaque create {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (stateShapes : List Shape)
    (initialState : TensorPack Float stateShapes)
    (program : ∀ (_mode : Mode)
      {α : Type}, [TorchLean.Storage α] → [Context α] →
        Program.Data α β stateShapes [σ] τ)
    (kind : String := "IndexedModel")
    (initializationPlan : Option
      (Init.Plan stateShapes) := none)
    (trainableMask : Array Bool := Array.replicate stateShapes.length true)
    (validateModel : Except String Unit := pure ())
    (validateInput : Tensor β σ → Except String Unit := fun _ => pure ()) :
    IndexedModel σ τ β :=
  ⟨stateShapes, kind, initialState, initializationPlan, trainableMask,
    validateModel, validateInput, program⟩

/--
Indexed-model placeholder for a configuration that failed a static check.

It owns no parameters or buffers, and `IndexedModel.validate` rejects it with `message` before
allocation or execution.
-/
def invalid {β : Type} [TorchLean.Storage β]
    (input output : Shape) (kind message : String) :
    IndexedModel input output β :=
  create
    []
    .nil
    (fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun _input =>
          Ops.const (m := m) (α := α)
            (TorchLean.Tensor.zeros output))
    (kind := kind)
    (initializationPlan := some .nil)
    (trainableMask := #[])
    (validateModel := .error message)

/-!
The model readers are `opaque`, so proofs cannot unfold them into the private fields and builders
can change the representation without any downstream proof seeing it. The public readers (`kind`,
`initialState`, `requiresGrad`) expose model metadata, and this namespace holds the runtime-only
readers for the initialization plan, the two validators and the program.
-/

/-- How the runtime should fill the state at allocation time, or `none` for an unplanned model. -/
opaque initializationPlan {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) :
    Option (Init.Plan model.stateShapes) :=
  match model with
  | ⟨_, _, _, initializationPlan, _, _, _, _⟩ => initializationPlan

/-- Static verdict on the configuration, checked before anything is allocated. -/
opaque validateModel {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : Except String Unit :=
  match model with
  | ⟨_, _, _, _, _, validateModel, _, _⟩ => validateModel

/-- Verdict on one concrete input, for conditions the shape type cannot express. -/
opaque validateInput {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) (input : Tensor β σ) : Except String Unit :=
  match model with
  | ⟨_, _, _, _, _, _, validateInput, _⟩ => validateInput input

/-- The runtime program for a given `Mode`, scalar-polymorphic in its execution type. -/
opaque program {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) (mode : Mode)
    {α : Type} [TorchLean.Storage α] [Context α] :
    Program.Data α β model.stateShapes [σ] τ :=
  match model with
  | ⟨_, _, _, _, _, _, _, program⟩ => program mode

/-- Model label used in summaries and diagnostics. -/
opaque kind {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : String :=
  match model with
  | ⟨_, kind, _, _, _, _, _, _⟩ => kind

/-- Initial values of the whole model state, stored in `Float`. -/
opaque initialState {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : TensorPack Float model.stateShapes :=
  match model with
  | ⟨_, _, initialState, _, _, _, _, _⟩ => initialState

/-- Gradient flags, one per entry of `stateShapes` once `validate` has passed. -/
opaque requiresGrad {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : Array Bool :=
  match model with
  | ⟨_, _, _, _, trainableMask, _, _, _⟩ => trainableMask

/--
Static check before allocation or graph execution: one gradient flag per state tensor, then the
model's own validator, then the initialization plan if the model has one.
-/
def validate {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : IndexedModel σ τ β) : Except String Unit :=
  validateStatic model.kind model.requiresGrad
    (validateModel model) (initializationPlan model)

/-- Append an ordinary sequential model after an indexed-input model. -/
def andThen {β : Type} [TorchLean.Storage β]
    {σ τ υ : Shape} (first : IndexedModel σ τ β)
    (rest : Sequential τ υ) : IndexedModel σ υ β :=
  create
    (first.stateShapes ++ Sequential.stateShapes rest)
    (first.initialState.append (nn.initialState rest))
    (fun mode {α} _ _ =>
      fun {m} _ _ =>
        CurriedRef.curry
          (Ref := fun s => Ops.Ref (m := m) (α := α) s)
          (ss := first.stateShapes ++ Sequential.stateShapes rest)
          (β := CurriedRef
            (fun s => Ops.DataRef (m := m) (α := α) β s)
            [σ] (m (Ops.Ref (m := m) (α := α) υ)))
          (fun state =>
            let (firstState, restState) := TorchLean.Pack.split
              (F := fun s => Ops.Ref (m := m) (α := α) s)
              (ss₁ := first.stateShapes)
              (ss₂ := Sequential.stateShapes rest) state
            CurriedRef.curry
              (Ref := fun s => Ops.DataRef
                (m := m) (α := α) β s)
              (ss := [σ])
              (fun dataInputs =>
                match dataInputs with
                | .cons indices .nil => do
                    let embedded ←
                      (CurriedRef.uncurry
                        (Ref := fun s =>
                          Ops.Ref (m := m) (α := α) s)
                        (ss := first.stateShapes)
                        (β := CurriedRef
                          (fun s => Ops.DataRef
                            (m := m) (α := α) β s)
                          [σ]
                          (m (Ops.Ref (m := m) (α := α) τ)))
                        (program first mode (α := α)) firstState) indices
                    Sequential.forwardState
                      (model := rest) (α := α) (m := m) mode restState embedded)))
    (kind := s!"{first.kind} → Sequential")
    (initializationPlan :=
      match IndexedModel.initializationPlan first,
          Sequential.runtimeInit? rest with
      | some firstPlan, some restPlan => some (firstPlan.append restPlan)
      | _, _ => none)
    (trainableMask := first.requiresGrad ++
      Sequential.requiresGrad rest)
    (validateModel := do
      first.validate
      Sequential.validate rest)
    (validateInput := IndexedModel.validateInput first)

/-! ### Scalar objectives -/

namespace Objective

/--
Pair an indexed-input model with a scalar loss.

The objective definition takes one ordinary target tensor and, in a separate pack, the
non-differentiable model input. Because the packs are separate, the indices never receive
gradients and are never reinterpreted in the model's floating-point element type.

The target shape `υ` is independent of the model output shape `τ`, so the loss may compare the
prediction with labels in another representation. The mode defaults to `train`.
-/
def fromLoss {β : Type} [TorchLean.Storage β] {σ τ υ : Shape}
    (model : IndexedModel σ τ β)
    (loss : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      Program α [τ, υ] [])
    (mode : Mode := .train) :
    ObjectiveDef β
      model.stateShapes [υ] [σ] :=
  { initState := model.initialState
    runtimeInit := initializationPlan model
    requiresGrad := model.requiresGrad
    validate := model.validate
    validateDataInputs := fun
      | .cons input .nil => validateInput model input
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry
          (Ref := fun s => Ops.Ref (m := m) (α := α) s)
          (ss := model.stateShapes ++ [υ])
          (β := CurriedRef
            (fun s => Ops.DataRef (m := m) (α := α) β s)
            [σ]
            (m (Ops.Ref
              (m := m) (α := α) [])))
          (fun arguments =>
            let (state, targets) := TorchLean.Pack.split
              (F := fun s => Ops.Ref (m := m) (α := α) s)
              (ss₁ := model.stateShapes) (ss₂ := [υ]) arguments
            let .cons target .nil := targets
            CurriedRef.curry
              (Ref := fun s =>
                Ops.DataRef (m := m) (α := α) β s)
              (ss := [σ])
              (β := m (Ops.Ref
                (m := m) (α := α) []))
              (fun dataInputs =>
                match dataInputs with
                | .cons indices .nil => do
                    let withInput := CurriedRef.uncurry
                      (Ref := fun s =>
                        Ops.Ref (m := m) (α := α) s)
                      (ss := model.stateShapes)
                      (β := CurriedRef
                        (fun s => Ops.DataRef
                          (m := m) (α := α) β s)
                        [σ]
                        (m (Ops.Ref (m := m) (α := α) τ)))
                      (program model mode (α := α)) state
                    let prediction ← withInput indices
                    loss (α := α) (m := m) prediction target)) }

/-- Pair an indexed-input model with mean-squared error. -/
def mse {β : Type} [TorchLean.Storage β]
    {σ τ : Shape} (model : IndexedModel σ τ β)
    (reduction : TorchLean.Loss.Reduction := .mean)
    (mode : Mode := .train) :
    ObjectiveDef β
      model.stateShapes [τ] [σ] :=
  fromLoss model
    (fun {α} _ _ => fun {m} _ _ => fun prediction target =>
      TorchLean.Loss.mse (m := m) (α := α) (s := τ) prediction target
        (reduction := reduction))
    (mode := mode)

end Objective

end IndexedModel

/--
A reusable embedding table with `vocabularySize` rows of length `embeddingWidth`.

The table does not fix the shape of the token tensor: `table.model input` specializes it to index
tensors of shape `input` with entries in `Fin vocabularySize`. Instantiating that model allocates
mutable weight storage, while the definition stays immutable, so it can be lowered and used in
proofs.
-/
structure Embedding (vocabularySize embeddingWidth : Nat) where
  private mk ::
  private initialWeightValue : Tensor Float [vocabularySize, embeddingWidth]
  private initializationPlanValue : Init.Plan
    [[vocabularySize, embeddingWidth]]
  private trainableValue : Bool
  private validationValue : Except String Unit

namespace Embedding

/--
Construct an embedding table from its initial weight, initialization plan, trainability flag and
validation verdict.
-/
opaque create {vocabularySize embeddingWidth : Nat}
    (initialWeight : Tensor Float [vocabularySize, embeddingWidth])
    (initializationPlan : Init.Plan
      [[vocabularySize, embeddingWidth]])
    (trainable : Bool := true)
    (validation : Except String Unit := pure ()) :
    Embedding vocabularySize embeddingWidth :=
  ⟨initialWeight, initializationPlan, trainable, validation⟩

/-- How the runtime fills the weight matrix at allocation time. -/
opaque initializationPlan {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) :
    Init.Plan
      [[vocabularySize, embeddingWidth]] :=
  match table with
  | ⟨_, plan, _, _⟩ => plan

/-- Whether the optimizer may update the table, false for a frozen lookup. -/
opaque isTrainable {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) : Bool :=
  match table with
  | ⟨_, _, trainable, _⟩ => trainable

/-- Static configuration verdict, including errors from `Embedding.invalid`. -/
opaque validation {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) : Except String Unit :=
  match table with
  | ⟨_, _, _, validation⟩ => validation

/--
A table whose validation fails with `message`. `Embedding.model` turns it into a placeholder with
no parameters.
-/
def invalid (vocabularySize embeddingWidth : Nat) (message : String) :
    Embedding vocabularySize embeddingWidth :=
  create
    (Tensor.zeros [vocabularySize, embeddingWidth])
    (.cons (.scheme .zeros) .nil)
    (trainable := false)
    (validation := .error message)

/-- Initial table values, with one row per vocabulary item. -/
opaque initialWeight {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) :
    Tensor Float [vocabularySize, embeddingWidth] :=
  match table with
  | ⟨weight, _, _, _⟩ => weight

/-- Construction options for a freshly initialized embedding table. -/
structure Config where
  /--
  Initialization scheme for the table.

  The default agrees with `torch.nn.Embedding.reset_parameters`: independent samples from the
  standard normal distribution. Language-model constructors normally override this with their
  architecture-specific initialization, such as GPT-2's standard deviation `0.02`.
  -/
  weightInit : Init.Scheme := .normal 0.0 1.0
  /-- Freeze the table by excluding it from reverse-mode parameter gradients. -/
  freeze : Bool := false

namespace Config

/--
Check that the vocabulary size and the embedding width are positive, for configured and
exact-weight tables alike.
-/
def validateDimensions (vocabularySize embeddingWidth : Nat) : Except String Unit := do
  if vocabularySize = 0 then
    throw "Embedding: vocabulary size must be positive"
  if embeddingWidth = 0 then
    throw "Embedding: embedding width must be positive"

/-- Validate embedding dimensions and initialization. -/
def validate (config : Config) (vocabularySize embeddingWidth : Nat) : Except String Unit := do
  validateDimensions vocabularySize embeddingWidth
  config.weightInit.validate

end Config

/--
Embedding table with the given initial weights, the typed counterpart of passing `_weight` to
`torch.nn.Embedding`.

The tensor type fixes both dimensions at compile time, and the values are recorded as a row-major
runtime initializer, so CPU and CUDA module construction start from the same payload.
-/
def fromWeight {vocabularySize embeddingWidth : Nat}
    (weight : Tensor Float [vocabularySize, embeddingWidth])
    (freeze : Bool := false) : Embedding vocabularySize embeddingWidth :=
  create weight
    (.cons (.flat (weight.to FloatArray)) .nil)
    (trainable := !freeze)
    (validation := Config.validateDimensions vocabularySize embeddingWidth)

/-- Specialize an embedding table to a concrete index-tensor shape. -/
def model {vocabularySize embeddingWidth : Nat}
    (table : Embedding vocabularySize embeddingWidth) (input : Shape) :
    IndexedModel input
      (input ++ [embeddingWidth]) (Fin vocabularySize) :=
  match validation table with
  | .error message =>
      IndexedModel.invalid
        input (input ++ [embeddingWidth])
        s!"Embedding({vocabularySize}, {embeddingWidth})" message
  | .ok () =>
      let weightShape : Shape := [vocabularySize, embeddingWidth]
      IndexedModel.create
        [weightShape]
        (.singleton table.initialWeight)
        (fun _ {α} _ _ =>
          fun {m} _ _ => fun weight tokenIds =>
            functional.embedding
              (m := m) (α := α) (vocabularySize := vocabularySize)
              (embeddingWidth := embeddingWidth) weight tokenIds)
        (kind := s!"Embedding({vocabularySize}, {embeddingWidth})")
        (initializationPlan := some (Embedding.initializationPlan table))
        (trainableMask := #[isTrainable table])
        (validateModel := pure ())

end Embedding

end nn
end TorchLean
