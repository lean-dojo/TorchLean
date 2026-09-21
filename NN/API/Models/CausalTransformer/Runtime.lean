/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.CausalTransformer.Architecture

/-!
# Causal Transformer Runtime Support

Complete indexed-token and tied-weight causal Transformer models, together with their
causal-language-model objective.
-/

@[expose] public section

namespace TorchLean

open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models
namespace CausalTransformer

namespace Internal

/-- State layout for a model whose token lookup and vocabulary projection share one matrix. -/
abbrev tiedStateShapes (config : Config)
    {batchShape : Shape}
    (body : nn.Sequential
      (config.embeddingShape batchShape)
      (config.embeddingShape batchShape)) :
    List Shape :=
  [config.vocabularySize, config.modelWidth] :: nn.stateShapes body

/-- Execute a tied token lookup, hidden Transformer, and vocabulary projection. -/
def tiedForward
    {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m]
    [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (mode : nn.Mode)
    (config : Config)
    {batchShape : Shape}
    (body : nn.Sequential
      (config.embeddingShape batchShape)
      (config.embeddingShape batchShape))
    (state : Runtime.Autograd.Torch.RefList
      (TorchLean.Runtime.ValueRef (m := m) (α := α))
      (tiedStateShapes config body))
    (tokens : Runtime.Autograd.Torch.DataRef
      (m := m) (α := α) (Fin config.vocabularySize) (config.tokenShape batchShape)) :
    m (TorchLean.Runtime.ValueRef
      (m := m) (α := α) (config.vocabularyShape batchShape)) := do
  let .cons tokenEmbedding bodyState := state
  let computedEmbeddings ← Runtime.Autograd.Model.F.embedding
    (m := m) (α := α) (vocabularySize := config.vocabularySize)
    (embeddingWidth := config.modelWidth) tokenEmbedding tokens
  let embeddings : TorchLean.Runtime.ValueRef
      (m := m) (α := α) (config.embeddingShape batchShape) := by
    simpa [Config.tokenShape, Config.embeddingShape, Shape.appendDim_appendDim_eq_concat] using
      computedEmbeddings
  let hidden ← Runtime.Autograd.Model.Layers.Seq.forwardState
    (model := body) (α := α) (m := m) mode bodyState embeddings
  let hiddenRows ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := config.embeddingShape batchShape)
    (s₂ := [(config.tokenShape batchShape).size, config.modelWidth])
    hidden (by
      simp [Config.embeddingShape, Config.tokenShape, Shape.size_concat, Shape.size_appendDim,
        Shape.size, Nat.mul_assoc])
  let projection ← Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
    (s := [config.vocabularySize, config.modelWidth]) 0 tokenEmbedding
  let logitsRows ← Runtime.Autograd.Torch.matmul (m := m) (α := α)
    (batchA := []) (batchB := []) (batch := [])
    (mDim := (config.tokenShape batchShape).size) (nDim := config.modelWidth)
    (pDim := config.vocabularySize)
    hiddenRows projection
  Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := [(config.tokenShape batchShape).size, config.vocabularySize])
    (s₂ := config.vocabularyShape batchShape) logitsRows
    (by
      simp [Config.vocabularyShape, Config.tokenShape, Shape.size_concat, Shape.size_appendDim,
        Shape.size, Nat.mul_assoc])

/-- Assemble a complete tied-weight indexed model from one table and one hidden body. -/
def tiedModel
    (config : Config)
    {batchShape : Shape}
    (table : nn.Embedding config.vocabularySize config.modelWidth)
    (body : nn.Sequential
      (config.embeddingShape batchShape)
      (config.embeddingShape batchShape)) :
    nn.IndexedModel (config.tokenShape batchShape) (config.vocabularyShape batchShape)
      (Fin config.vocabularySize) :=
  let stateShapes := tiedStateShapes config body
  let initialState : nn.State Float stateShapes := by
    simpa [stateShapes, tiedStateShapes] using
      (nn.State.empty.push table.initialWeight).append (nn.initialState body)
  let initializationPlan :
      Option (Runtime.Autograd.Model.Module.RuntimeInit.Plan stateShapes) :=
    match Runtime.Autograd.Model.Layers.Seq.runtimeInit? body with
    | some bodyPlan =>
        some ((nn.Embedding.Internal.initializationPlan table).append bodyPlan)
    | none => none
  nn.IndexedModel.Internal.create
    stateShapes
    initialState
    (fun mode {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := TorchLean.Runtime.ValueRef (m := m) (α := α))
          (ss := stateShapes)
          (β := Runtime.Autograd.Torch.CurriedRef
            (fun s => Runtime.Autograd.Torch.DataRef
              (m := m) (α := α) (Fin config.vocabularySize) s)
            [config.tokenShape batchShape]
            (m (TorchLean.Runtime.ValueRef
              (m := m) (α := α) (config.vocabularyShape batchShape))))
          (fun state => fun tokens =>
            tiedForward (m := m) (α := α) mode config body state tokens))
    (kind := "CausalTransformer.tied")
    (initializationPlan := initializationPlan)
    (trainableMask :=
      #[nn.Embedding.Internal.isTrainable table] ++ nn.requiresGrad body)
    (validateModel := do
      config.validate
      nn.validate body)

end Internal

/--
Build a complete indexed-token causal Transformer with an independent vocabulary head.

The token table, hidden Transformer, and output projection draw from one builder seed stream.
The returned model is the single value used by training, checkpointing, and prediction.
-/
def indexed (config : Config)
    (batchShape : Shape := []) :
    nn.Builder
      (nn.IndexedModel (config.tokenShape batchShape) (config.vocabularyShape batchShape)
        (Fin config.vocabularySize)) :=
  fun stream =>
    match config.validate with
    | .error message =>
        (nn.IndexedModel.Internal.invalidConfiguration
          (config.tokenShape batchShape) (config.vocabularyShape batchShape)
          "CausalTransformer.indexed" message, stream)
    | .ok () =>
        let embeddingInitialization :=
          config.parameterInitialization?.getD (.uniform (-0.02) 0.02)
        let (table, afterTable) :=
          nn.embedding config.vocabularySize config.modelWidth
            { weightInitialization := embeddingInitialization } stream
        let lookup : nn.IndexedModel
            (config.tokenShape batchShape) (config.embeddingShape batchShape)
            (Fin config.vocabularySize) := by
          simpa [Config.tokenShape, Config.embeddingShape,
            Shape.appendDim_eq_concat, Shape.concat_assoc] using
            nn.Embedding.model table (config.tokenShape batchShape)
        let (body, afterBody) :=
          fromEmbeddings config batchShape afterTable
        (lookup.andThen body, afterBody)

/--
Build a complete causal Transformer with one matrix shared by token lookup and output projection.
-/
def tied (config : Config)
    (batchShape : Shape := []) :
    nn.Builder
      (nn.IndexedModel (config.tokenShape batchShape) (config.vocabularyShape batchShape)
        (Fin config.vocabularySize)) :=
  fun stream =>
    match config.validate with
    | .error message =>
        (nn.IndexedModel.Internal.invalidConfiguration
          (config.tokenShape batchShape) (config.vocabularyShape batchShape)
          "CausalTransformer.tied" message, stream)
    | .ok () =>
        let embeddingInitialization :=
          config.parameterInitialization?.getD (.uniform (-0.02) 0.02)
        let (table, afterTable) :=
          nn.embedding config.vocabularySize config.modelWidth
            { weightInitialization := embeddingInitialization } stream
        let (body, afterBody) :=
          hidden config batchShape afterTable
        (Internal.tiedModel config table body, afterBody)

/--
Pair a complete indexed-token causal Transformer with next-token cross entropy.

Inputs and targets are bounded token tensors with `batchShape` followed by the sequence axis.
-/
def objective
    (config : Config)
    {batchShape : Shape}
    (model : nn.IndexedModel
      (config.tokenShape batchShape)
      (config.vocabularyShape batchShape)
      (Fin config.vocabularySize))
    (reduction : TorchLean.Loss.Reduction := .mean)
    (mode : nn.Mode := .train) :
    TorchLean.Module.ObjectiveDefinition (Fin config.vocabularySize)
      model.stateShapes [] [config.tokenShape batchShape, config.tokenShape batchShape] :=
  { initState := nn.State.Internal.toTensorPack model.initialState
    runtimeInit := nn.IndexedModel.Internal.initializationPlan model
    requiresGrad := model.requiresGrad
    validate := model.validate
    validateDataInputs := fun
      | .cons tokens (.cons _targets .nil) =>
          nn.IndexedModel.Internal.validateInput model tokens
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := TorchLean.Runtime.ValueRef (m := m) (α := α))
          (ss := model.stateShapes ++ ([] : List Shape))
          (β := Runtime.Autograd.Torch.CurriedRef
            (fun s => Runtime.Autograd.Torch.DataRef
              (m := m) (α := α) (Fin config.vocabularySize) s)
            [config.tokenShape batchShape, config.tokenShape batchShape]
            (m (TorchLean.Runtime.ValueRef (m := m) (α := α) [])))
          (fun arguments => fun tokens => fun targets => (do
            let (state, empty) :=
              Runtime.Autograd.Torch.RefList.split
                (Ref := TorchLean.Runtime.ValueRef (m := m) (α := α))
                (ss₁ := model.stateShapes)
                (ss₂ := ([] : List Shape)) arguments
            let .nil := empty
            let withInput := Runtime.Autograd.Torch.CurriedRef.uncurry
              (Ref := TorchLean.Runtime.ValueRef (m := m) (α := α))
              (ss := model.stateShapes)
              (β := Runtime.Autograd.Torch.CurriedRef
                (fun s => Runtime.Autograd.Torch.DataRef
                  (m := m) (α := α) (Fin config.vocabularySize) s)
                [config.tokenShape batchShape]
                (m (TorchLean.Runtime.ValueRef
                  (m := m) (α := α) (config.vocabularyShape batchShape))))
              (nn.IndexedModel.Internal.program model mode (α := α)) state
            let logits ← withInput tokens
            let logitsIndexed : TorchLean.Runtime.ValueRef (m := m) (α := α)
                ((config.tokenShape batchShape).concat [config.vocabularySize]) := by
              simpa [Config.vocabularyShape, Config.tokenShape,
                Shape.appendDim_eq_concat, Shape.concat_assoc] using logits
            let targetsIndexed : Runtime.Autograd.Torch.DataRef
                (m := m) (α := α) (Fin config.vocabularySize)
                ((config.tokenShape batchShape).concat []) := by
              simpa using targets
            TorchLean.Loss.crossEntropy (m := m) (α := α)
              (leading := config.tokenShape batchShape) (trailing := [])
              (classes := config.vocabularySize)
              (config.tokenShape batchShape).rank rfl
              logitsIndexed targetsIndexed (reduction := reduction) :
              m (TorchLean.Runtime.ValueRef (m := m) (α := α) []))) }

end CausalTransformer
end models
end nn

end TorchLean
