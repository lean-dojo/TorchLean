/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.CausalTransformer

/-!
# Causal Transformer Runtime Support

`indexed` builds a complete causal Transformer over bounded token indices with an independent
vocabulary head, and `tied` builds one whose token lookup and output projection share a matrix.
`objective` pairs either model with next-token cross entropy.

`tied` computes the logits as `h Eᵀ` from the token table `E`, with no output bias. That's the head
of GPT-2's released code (`src/model.py` multiplies by `wte` transposed), and the weight sharing
of Press and Wolf (2017). `objective` scores the logits at position `i` against `targets` at
position `i`, so the caller shifts the sequence by one token before passing it. With the default
`.mean` the loss averages over every position of every sequence, the value
`torch.nn.CrossEntropyLoss` (PyTorch 2.8 docs) returns for its default `reduction='mean'` when no
class `weight` is given.

References:
- Ofir Press and Lior Wolf (2017), "Using the Output Embedding to Improve Language Models",
  EACL 2017, arXiv:1608.05859.
-/

@[expose] public section

open TorchLean.Shape (size_append)

namespace TorchLean

open TorchLean.Tensor

namespace nn
namespace CausalTransformer

/-- State layout for a model whose token lookup and vocabulary projection share one matrix. -/
abbrev tiedStateShapes (config : Config)
    {batchShape : Shape}
    (body : nn.Sequential
      (config.embeddingShape batchShape)
      (config.embeddingShape batchShape)) :
    List Shape :=
  [config.vocabularySize, config.modelWidth] :: nn.Sequential.stateShapes body

/-- Look up tokens, run the hidden body and project with the transposed embedding table. -/
def tiedForward
    {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m]
    [Ops (m := m) (α := α)]
    (mode : nn.Mode)
    (config : Config)
    {batchShape : Shape}
    (body : nn.Sequential
      (config.embeddingShape batchShape)
      (config.embeddingShape batchShape))
    (state : Pack (Ops.Ref (m := m) (α := α))
      (tiedStateShapes config body))
    (tokens : Ops.DataRef
      (m := m) (α := α) (Fin config.vocabularySize) (config.tokenShape batchShape)) :
    m (Ops.Ref
      (m := m) (α := α) (config.vocabularyShape batchShape)) := do
  let .cons tokenEmbedding bodyState := state
  let computedEmbeddings ← functional.embedding
    (m := m) (α := α) (vocabularySize := config.vocabularySize)
    (embeddingWidth := config.modelWidth) tokenEmbedding tokens
  let embeddings : Ops.Ref
      (m := m) (α := α) (config.embeddingShape batchShape) := by
    simpa [Config.tokenShape, Config.embeddingShape] using
      computedEmbeddings
  let hidden ← Sequential.forwardState
    (model := body) (α := α) (m := m) mode bodyState embeddings
  let hiddenRows ← Ops.apply (m := m) (α := α)
      (.reshape (s := config.embeddingShape batchShape)
        (t := [(config.tokenShape batchShape).size, config.modelWidth]) (by
      simp [Config.embeddingShape, Config.tokenShape, Shape.size_append,
          Nat.mul_assoc])) (hidden, ())
  let projection ← Ops.apply (m := m) (α := α)
      (.permute (s := [config.vocabularySize, config.modelWidth])
        (Shape.adjacentSwap [config.vocabularySize, config.modelWidth] 0)
        (Shape.swapAdjacentAtDepth [config.vocabularySize, config.modelWidth] 0)
            (by tensor_shape))
      (tokenEmbedding, ())
  let logitsRows ← Ops.apply (m := m) (α := α)
      (.matmul (batchA := []) (batchB := []) (batch := [])
        (m := (config.tokenShape batchShape).size) (n := config.modelWidth)
        (p := config.vocabularySize) false false (fun h => nomatch h) (fun h => nomatch h)
        Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (hiddenRows, projection, ())
  Ops.apply (m := m) (α := α)
      (.reshape (s := [(config.tokenShape batchShape).size, config.vocabularySize])
        (t := config.vocabularyShape batchShape) (by
      simp [Config.vocabularyShape, Config.tokenShape, Shape.size_append,
          Nat.mul_assoc])) (logitsRows, ())

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
  let initialState : TensorPack Float stateShapes := by
    simpa [stateShapes, tiedStateShapes] using
      (Pack.singleton table.initialWeight).append (nn.initialState body)
  let initializationPlan :
      Option (Init.Plan stateShapes) :=
    match Sequential.runtimeInit? body with
    | some bodyPlan =>
        some ((nn.Embedding.initializationPlan table).append bodyPlan)
    | none => none
  nn.IndexedModel.create
    stateShapes
    initialState
    (fun mode {α} _ _ =>
      fun {m} _ _ =>
        CurriedRef.curry
          (Ref := Ops.Ref (m := m) (α := α))
          (ss := stateShapes)
          (β := CurriedRef
            (fun s => Ops.DataRef
              (m := m) (α := α) (Fin config.vocabularySize) s)
            [config.tokenShape batchShape]
            (m (Ops.Ref
              (m := m) (α := α) (config.vocabularyShape batchShape))))
          (fun state => fun tokens =>
            tiedForward (m := m) (α := α) mode config body state tokens))
    (kind := "CausalTransformer.tied")
    (initializationPlan := initializationPlan)
    (trainableMask :=
      #[nn.Embedding.isTrainable table] ++ nn.Sequential.requiresGrad body)
    (validateModel := do
      config.validate
      nn.Sequential.validate body)

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
        (nn.IndexedModel.invalid
          (config.tokenShape batchShape) (config.vocabularyShape batchShape)
          "CausalTransformer.indexed" message, stream)
    | .ok () =>
        let (table, afterTable) :=
          nn.embedding config.vocabularySize config.modelWidth
            { weightInit := config.embeddingInit } stream
        let lookup : nn.IndexedModel
            (config.tokenShape batchShape) (config.embeddingShape batchShape)
            (Fin config.vocabularySize) := by
          simpa [Config.tokenShape, Config.embeddingShape, List.append_assoc] using
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
        (nn.IndexedModel.invalid
          (config.tokenShape batchShape) (config.vocabularyShape batchShape)
          "CausalTransformer.tied" message, stream)
    | .ok () =>
        let (table, afterTable) :=
          nn.embedding config.vocabularySize config.modelWidth
            { weightInit := config.embeddingInit } stream
        let (body, afterBody) :=
          hidden config batchShape afterTable
        (tiedModel config table body, afterBody)

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
    ObjectiveDef (Fin config.vocabularySize)
      model.stateShapes [] [config.tokenShape batchShape, config.tokenShape batchShape] :=
  { initState := model.initialState
    runtimeInit := nn.IndexedModel.initializationPlan model
    requiresGrad := model.requiresGrad
    validate := model.validate
    validateDataInputs := fun
      | .cons tokens (.cons _targets .nil) =>
          nn.IndexedModel.validateInput model tokens
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry
          (Ref := Ops.Ref (m := m) (α := α))
          (ss := model.stateShapes ++ ([] : List Shape))
          (β := CurriedRef
            (fun s => Ops.DataRef
              (m := m) (α := α) (Fin config.vocabularySize) s)
            [config.tokenShape batchShape, config.tokenShape batchShape]
            (m (Ops.Ref (m := m) (α := α) [])))
          (fun arguments => fun tokens => fun targets => (do
            let (state, empty) :=
              Pack.split
                (F := Ops.Ref (m := m) (α := α))
                (ss₁ := model.stateShapes)
                (ss₂ := ([] : List Shape)) arguments
            let .nil := empty
            let withInput := CurriedRef.uncurry
              (Ref := Ops.Ref (m := m) (α := α))
              (ss := model.stateShapes)
              (β := CurriedRef
                (fun s => Ops.DataRef
                  (m := m) (α := α) (Fin config.vocabularySize) s)
                [config.tokenShape batchShape]
                (m (Ops.Ref
                  (m := m) (α := α) (config.vocabularyShape batchShape))))
              (nn.IndexedModel.program model mode (α := α)) state
            let logits ← withInput tokens
            let logitsIndexed : Ops.Ref (m := m) (α := α)
                ((config.tokenShape batchShape) ++ [config.vocabularySize]) := by
              simpa [Config.vocabularyShape, Config.tokenShape, List.append_assoc] using logits
            let targetsIndexed : Ops.DataRef
                (m := m) (α := α) (Fin config.vocabularySize)
                ((config.tokenShape batchShape) ++ []) := by
              simpa using targets
            TorchLean.Loss.crossEntropy (m := m) (α := α)
              (leading := config.tokenShape batchShape) (trailing := [])
              (classes := config.vocabularySize)
              (config.tokenShape batchShape).rank rfl
              logitsIndexed targetsIndexed (reduction := reduction) :
              m (Ops.Ref (m := m) (α := α) []))) }

end CausalTransformer
end nn

end TorchLean
