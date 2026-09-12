/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.RealPaths
public import NN.Examples.Support.Training

/-!
# Char-GPT (minGPT-style) Example

This example follows the character-level Transformer from Andrej Karpathy's
"Let's build GPT: from scratch, in code, spelled out" lecture:

- build an alphabet (`itos`) from the training text,
- build a `stoi` tokenizer from that alphabet,
- train a compact causal Transformer to predict the next character,
- sample text continuations from a prompt.

The `karpathy` preset follows the lecture configuration: batch size 64, context length 256, width
384, six attention heads, six pre-normalized Transformer blocks, ReLU feed-forward layers, dropout
0.2, AdamW, and 5,000 updates. The CUDA command executes the numerical path in float32. TorchLean
applies dropout to the attention and feed-forward sublayer outputs; unlike the lecture code, it does
not yet apply a second dropout to the attention weights themselves. All dimensions remain
command-line choices.

Training draws a fresh deterministic batch of corpus windows at every step.  The windows are built
on demand, so a long run does not retain thousands of large one-hot tensors in host memory.

Unlike the introductory model examples, this file intentionally uses the low-level `Module` API.
The indexed objective receives token IDs while the model parameters and loss use floating-point
storage, so it crosses a mixed-dtype boundary that the homogeneous `Trainer` interface does not
express. Ordinary floating-point models should use `Trainer.new` and `trainer.train`.

Quick check:

```bash
lake -R -K cuda=true build torchlean:exe
lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare --preset smoke
```

Full lecture experiment:

```bash
lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare --preset karpathy
```

Reference: <https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py>.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.CharGpt

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "chargpt"

/-- Parse corpus flags and return the UTF-8 training text plus remaining CLI arguments. -/
def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName NN.Examples.Data.RealPaths.tinyShakespeare
    [("--tiny-shakespeare", NN.Examples.Data.RealPaths.tinyShakespeare)]
    NN.Examples.Data.RealPaths.missingTinyShakespeareHint args

/-- Build a deterministic character alphabet from the corpus. -/
def buildAlphabet (s : String) : Array Char :=
  let chars : List Char := s.toList.eraseDups
  -- Deterministic order: sort by codepoint.
  let sorted := List.mergeSort chars (fun a b => decide (a.toNat ≤ b.toNat))
  sorted.toArray

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "chargpt"

/-- Architecture and evaluation controls independent of the corpus and runtime device. -/
structure Preset where
  modelWidth : Nat
  attentionHeads : Nat
  transformerLayers : Nat
  dropoutProbability : Float
  trainingSteps : Nat
  batchSize : Nat
  contextLength : Nat
  learningRate : Float
  evalEvery : Nat
  evaluationBatches : Nat
  generationLength : Nat
deriving Repr

namespace Preset

/-- Fast configuration used to validate the complete training and generation path. -/
def smoke : Preset :=
  { modelWidth := 32
    attentionHeads := 4
    transformerLayers := 2
    dropoutProbability := 0.0
    trainingSteps := 2
    batchSize := 2
    contextLength := 16
    learningRate := 3e-4
    evalEvery := 1
    evaluationBatches := 1
    generationLength := 32 }

/-- Hyperparameters from Karpathy's final Tiny Shakespeare lecture model. -/
def karpathy : Preset :=
  { modelWidth := 384
    attentionHeads := 6
    transformerLayers := 6
    dropoutProbability := 0.2
    trainingSteps := 5000
    batchSize := 64
    contextLength := 256
    learningRate := 3e-4
    evalEvery := 500
    evaluationBatches := 200
    generationLength := 500 }

/--
Resolve a `--preset` name. `smoke` is the fast configuration used in CI; `karpathy` reproduces the
hyperparameters from the nanoGPT character-level Shakespeare run.
-/
def parse (name : String) : Except String Preset :=
  match name.trimAscii.toString.toLower with
  | "smoke" => .ok smoke
  | "karpathy" => .ok karpathy
  | other => .error s!"unknown --preset '{other}'; expected smoke or karpathy"

end Preset

/-- Help text for character-level GPT training. -/
def usage : String :=
  String.intercalate "\n"
    [ "torchlean chargpt: character-level GPT training"
    , ""
    , "Usage:"
    , "  lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare "
        ++ "--preset PRESET [flags]"
    , ""
    , "Presets:"
    , "  smoke       two-update end-to-end CUDA check"
    , "  karpathy    full Tiny Shakespeare lecture experiment"
    , ""
    , "Architecture:"
    , "  --width N       embedding width"
    , "  --heads N       attention heads; must divide width"
    , "  --layers N      Transformer blocks"
    , "  --dropout P     dropout probability in [0, 1)"
    , "  --batch-size N  training windows per update"
    , "  --seq-len N     context length"
    , "  --steps N       optimizer updates"
    , "  --lr FLOAT      AdamW learning rate"
    , "  --eval-every N  validation cadence; 0 disables intermediate evaluation"
    , "  --eval-iters N  batches averaged at each validation point"
    , "  --load-checkpoint P restore model state before training"
    , "  --save-checkpoint P save final model state"
    , ""
    , "Notes:"
    , "  - Training and validation use disjoint 90/10 corpus splits."
    , "  - CPU and CUDA runs both use native Float32 arithmetic."
    ]

/-- Command-local controls for CharGPT training, checkpointing, and generation. -/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Prompt and token-sampling policy. -/
  generation : text.GenerationOptions
  /-- Optional checkpoint input and output paths. -/
  checkpoint : text.CheckpointOptions
  /-- Context length in characters. -/
  contextLength : Nat
deriving Repr

namespace Options

/-- Parse training controls after the experiment preset has supplied defaults. -/
def parse (args : List String) (defaults : Preset)
    (generationDefaults : text.GenerationOptions) : Except String (Options × List String) := do
  let (contextLength, args) ←
    CLI.takePositiveNatFlag args exeName "seq-len" (default := defaults.contextLength)
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := defaults.trainingSteps)
      (defaultLearningRate := defaults.learningRate)
      (defaultBatchSize := defaults.batchSize)
      (allowZeroSteps := true)
  let (generation, args) ← text.GenerationOptions.parse exeName args generationDefaults
  let (checkpoint, args) ← text.CheckpointOptions.parse args
  pure ({ training
          generation
          checkpoint
          contextLength }, args)

end Options

/-- Decode token ids for terminal output with control characters escaped. -/
def escapeCharIdsForDisplay (t : text.Tokenizer) (ids : Array Nat) : String :=
  text.escape (t.decode ids)

/-- Printable-ASCII generation filter used by `--ascii-only`. -/
def asciiAllowed (c : Char) : Bool :=
  c = '\n' || (32 ≤ c.toNat && c.toNat ≤ 126)

/-- Fitted predictor for a runtime-sized character GPT model. -/
abbrev Predictor (α : Type) (batchSize contextLength vocabularySize : Nat) :=
  Tensor (Fin vocabularySize) [batchSize, contextLength] →
    IO (Tensor α [batchSize, contextLength, vocabularySize])

/-- Autoregressively extend character token ids using a trained CharGPT model. -/
def generateSampledFromIds {α : Type} {promptLength : Nat}
    (toFloat : α → Float)
    (batchSize contextLength vocabularySize : Nat) [NeZero vocabularySize]
    (predict : Predictor α batchSize contextLength vocabularySize)
    (promptTokens : Tensor (Fin vocabularySize) [promptLength])
    (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float)
    (allowToken : Fin vocabularySize → Bool := fun _ => true)
    (paddingTokenId : Fin vocabularySize := 0) :
    IO (Tensor (Fin vocabularySize) [promptLength + steps]) := do
  let gen : text.GenerationOptions :=
    { prompt := ""
      newTokenCount := steps
      temperature := temperature
      topK := topK
      repeatPenalty := repeatPenalty
      repeatWindow := repeatWindow
      seed := seed
      asciiOnly := false }
  if hBatchSize : batchSize = 0 then
    throw (IO.userError "generation requires a nonempty model batch")
  else
    let firstBatchRow : Fin batchSize := ⟨0, Nat.pos_of_ne_zero hBatchSize⟩
    let ids ← text.autoregressiveTokenIds contextLength paddingTokenId.val
      (promptTokens.map Fin.val) gen
      (fun padded position => do
        let bounded ← IO.ofExcept (Tensor.checkIndices vocabularySize padded)
        let logits ← predict (Tensor.repeatAxis 0 batchSize bounded)
        pure ((text.batchLogitScoresAt logits firstBatchRow position).map toFloat))
      allowToken
    let bounded ← IO.ofExcept (Tensor.checkIndices vocabularySize ids)
    pure bounded

/-- CLI entrypoint for character-level GPT training and sampling. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  Module.Command.run
    (config := {
      banner? := some fun _ => s!"{exeName}: char-level GPT training"
      printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (corpus, rest) ← takeInputText rest
      let (presetName?, rest) ← CLI.orThrow exeName <|
        TorchLean.CLI.takeFlagValue? rest "preset"
      let defaults ← CLI.orThrow exeName <|
        Preset.parse (presetName?.getD "smoke")
      let presetName := presetName?.getD "smoke" |>.trimAscii.toString.toLower
      let defaultPrompt :=
        if presetName == "karpathy" then "\n" else "First Citizen:"
      let generationDefaults : text.GenerationOptions :=
        if presetName == "karpathy" then
          { prompt := defaultPrompt
            newTokenCount := defaults.generationLength
            temperature := 1.0
            topK := 0
            repeatPenalty := 1.0
            repeatWindow := 0
            seed := 1337
            asciiOnly := false }
        else
          { prompt := defaultPrompt
            newTokenCount := defaults.generationLength
            temperature := 0.9
            topK := 12
            repeatPenalty := 1.15
            repeatWindow := 64
            seed := 7
            asciiOnly := false }
      let (train, rest) ← CLI.orThrow exeName <|
        Options.parse rest defaults generationDefaults
      let (modelWidth, rest) ← CLI.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlag rest exeName "width" (default := defaults.modelWidth)
      let (attentionHeads, rest) ← CLI.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlag rest exeName "heads" (default := defaults.attentionHeads)
      let (transformerLayers, rest) ← CLI.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlag rest exeName "layers"
          (default := defaults.transformerLayers)
      let (dropoutProbability, rest) ← CLI.orThrow exeName <|
        TorchLean.CLI.takeNonnegativeFloatFlag rest exeName "dropout"
          (default := defaults.dropoutProbability)
      let (evalEvery, rest) ← CLI.orThrow exeName <|
        TorchLean.CLI.takeNatFlag rest "eval-every" (default := defaults.evalEvery)
      let (evaluationBatches, rest) ← CLI.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlag rest exeName "eval-iters"
          (default := defaults.evaluationBatches)
      CLI.requireNoArgs exeName rest
      if dropoutProbability >= 1.0 then
        throw <| IO.userError s!"{exeName}: --dropout must be smaller than 1"
      if modelWidth % attentionHeads != 0 then
        throw <| IO.userError s!"{exeName}: --heads must divide --width"
      let alphabetFull := buildAlphabet corpus
      let unknownTokenId : Fin alphabetFull.size ←
        if h : 0 < alphabetFull.size then
          pure ⟨0, h⟩
        else
          throw <| IO.userError s!"{exeName}: corpus contains no characters"
      let tok := text.Tokenizer.fromAlphabet alphabetFull unknownTokenId (unknownCharacter := '?')
      let vocabularySize := tok.vocabularySize
      have hVocabularySize : vocabularySize ≠ 0 := by
        intro h
        have hAlphabet : alphabetFull.size = 0 := by
          simpa [vocabularySize, tok, text.Tokenizer.fromAlphabet] using h
        rw [hAlphabet] at unknownTokenId
        exact Fin.elim0 unknownTokenId
      letI : NeZero vocabularySize := ⟨hVocabularySize⟩
      do
        let batchSize := train.training.batchSize
        let contextLength := train.contextLength
        let headWidthValue := modelWidth / attentionHeads
        let config : nn.models.CausalTransformer.Config :=
          { sequenceLength := contextLength
            vocabularySize := vocabularySize
            headCount := attentionHeads
            headWidth := headWidthValue
            feedForwardWidth := 4 * modelWidth
            layerCount := transformerLayers
            activation := .relu
            dropout? := if dropoutProbability == 0.0 then none else some dropoutProbability
            normalizeFirst := true
            attentionOutputBias := true
            parameterInitialization? := some (.normal 0.0 0.02) }
        let model : nn.IndexedModel
            (config.tokens [batchSize])
            (config.vocabulary [batchSize])
            (Fin config.vocabularySize) :=
          nn.build runtime.seed <|
            nn.models.CausalTransformer.indexed config [batchSize]

        let encoded := tok.encode corpus
        let allTokens ← CLI.orThrow exeName <|
          Tensor.checkIndices vocabularySize (Tensor.from encoded)
        let trainCount := encoded.size * 9 / 10
        let validationCount := encoded.size - trainCount
        let trainTokens := Tensor.window allTokens trainCount 0 unknownTokenId
        let valTokens := Tensor.window allTokens validationCount trainCount unknownTokenId
        if trainCount <= contextLength || validationCount <= contextLength then
          throw <| IO.userError s!"{exeName}: corpus split is too short for context length {
            contextLength}"

        let trainingBatchAt (step : Nat) :=
          Data.CausalLM.tokenSample [batchSize] contextLength <|
            text.Corpus.randomTokenBatch trainTokens batchSize contextLength runtime.seed step
              unknownTokenId
        let validationBatchAt (step : Nat) :=
          Data.CausalLM.tokenSample [batchSize] contextLength <|
            text.Corpus.randomTokenBatch valTokens batchSize contextLength
              (runtime.seed + 1000003) step unknownTokenId
        let trainDef := nn.models.CausalTransformer.objective config model
        let evalDef := nn.models.CausalTransformer.objective config model (mode := .eval)
        let module ← TorchLean.Module.instantiate trainDef runtime
        match train.checkpoint.loadCheckpoint? with
        | none => pure ()
        | some path =>
            Checkpoint.load module path
            IO.println s!"  loaded checkpoint: {path}"
        let trainStep ← module.dataStep <|
          optim.adamW { learningRate := train.training.learningRate }
        let storedScalarCount :=
          model.stateShapes.foldl
            (fun total shape => total + Shape.size shape) 0
        let trainableParameterCount :=
          (model.stateShapes.toArray.zip model.requiresGrad).foldl
            (fun total entry =>
              let (shape, trainable) := entry
              if trainable then total + Shape.size shape else total) 0
        IO.println s!"  trainable_parameters={trainableParameterCount}"
        if storedScalarCount != trainableParameterCount then
          IO.println s!"  non_trainable_state_scalars={storedScalarCount - trainableParameterCount}"
        let evaluateLoss ← module.dataLossEvaluator evalDef
        let evalLoss : IO Float := do
          let losses ← Tensor.generateFlatM [evaluationBatches] fun i => do
            let sample := validationBatchAt i.val
            let loss ← evaluateLoss sample.input sample.target
            pure (Float32.toFloat (Tensor.item loss))
          pure losses.mean
        let lossBefore ← evalLoss
        IO.println s!"  step 0: val loss={lossBefore}"
        for step in [0:train.training.steps] do
          let sample := trainingBatchAt step
          trainStep sample.input sample.target
          let done := step + 1
          if evalEvery != 0 && (done % evalEvery == 0 || done == train.training.steps) then
            let loss ← evalLoss
            IO.println s!"  step {done}: val loss={loss}"
        let lossAfter ← evalLoss
        let predict ← module.indexedPredictor model
        let promptTokens ← CLI.orThrow exeName <|
          Tensor.checkIndices vocabularySize (Tensor.from (tok.encode train.generation.prompt))
        let allowToken : Fin vocabularySize → Bool :=
          if train.generation.asciiOnly then
            fun i => alphabetFull[i.val]?.any asciiAllowed
          else
            fun _ => true
        let outIds ←
          generateSampledFromIds Float32.toFloat batchSize contextLength vocabularySize predict
            promptTokens train.generation.newTokenCount train.generation.temperature
            train.generation.topK train.generation.seed train.generation.repeatWindow
            train.generation.repeatPenalty
            (allowToken := allowToken) (paddingTokenId := unknownTokenId)
        let sampled := escapeCharIdsForDisplay tok ((outIds.map Fin.val).to (Array Nat))
        match train.checkpoint.saveCheckpoint? with
        | none => pure ()
        | some path =>
            Checkpoint.save module path
            IO.println s!"  wrote checkpoint: {path}"
        IO.println s!"  vocabularySize={vocabularySize} (unique chars)"
        IO.println s!"  architecture=modelWidth {modelWidth}, attentionHeads {attentionHeads}, \
transformerLayers {transformerLayers}, dropoutProbability {dropoutProbability}"
        IO.println s!"  sampled={sampled}"
        text.Log.writeGeneration
          train.training.logDestination
            "CharGPT (minGPT-style)" train.training.steps lossBefore lossAfter
          train.generation sampled
          #[Support.deviceNote runtime,
            s!"vocabularySize={vocabularySize}",
            s!"train_tokens={trainCount}",
            s!"validation_tokens={validationCount}",
            Support.cudaMemoryNote runtime train.training.steps
              train.training.cudaMemorySampleEvery]
        pure ())

end NN.Examples.Models.Sequence.CharGpt
