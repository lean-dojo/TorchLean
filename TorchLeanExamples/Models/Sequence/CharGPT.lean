/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Data.CausalLM
public import TorchLean.NN.Models.CausalTransformer.LanguageModel
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Text.Generation
public import TorchLeanExamples.Data.RealPaths
public import TorchLeanExamples.Models.Training
public import TorchLean.NN.Checkpoint

/-!
# Char-GPT (minGPT-style) Example

The example follows the character-level Transformer from Andrej Karpathy's
"Let's build GPT: from scratch, in code, spelled out" lecture:

- build an alphabet (`itos`) from the training text,
- build a `stoi` tokenizer from that alphabet,
- train a compact causal Transformer to predict the next character,
- sample text continuations from a prompt.

The `karpathy` preset follows the lecture configuration: batch size 64, context length 256, width
384, six attention heads, six pre-normalized Transformer blocks, ReLU feed-forward layers, dropout
0.2, AdamW, and 5,000 updates. CPU and CUDA runs both compute in float32. The example applies
dropout to the attention and feed-forward sublayer outputs. The lecture code also applies it to the
attention weights, which the `attentionDropout?` field controls and the example leaves unset. The
width, head count, depth, dropout, batch size and context length all have command-line flags.

Every training step draws a fresh deterministic batch of token-ID windows from the corpus. The
windows are built on demand, so a long run keeps no precomputed batches in host memory.

The introductory model examples use `Trainer`, and this file uses the low-level `Module` API. Its
indexed objective takes token IDs, while the parameters and the loss use floating-point storage.
The homogeneous `Trainer` interface does not express that mixed-dtype boundary. Ordinary
floating-point models should use `Trainer.create` and `trainer.train`.

Quick check:

```bash
scripts/lake.sh -Kcuda=true build torchlean:exe
scripts/lake.sh -Kcuda=true exe torchlean chargpt --device cuda --tiny-shakespeare --preset smoke
```

Full lecture experiment:

```bash
scripts/lake.sh -Kcuda=true exe torchlean chargpt --device cuda --tiny-shakespeare --preset karpathy
```

Reference: <https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py>.
-/

@[expose] public section

open TorchLean.Activation.Kind
open TorchLean.Shape (size)

namespace TorchLean.Examples.CharGPT

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "chargpt"

/-- Parse corpus flags and return the UTF-8 training text plus remaining CLI arguments. -/
def takeInputText (args : List String) : IO (String × List String) :=
  Text.Corpus.takeUTF8Input exeName Examples.RealData.tinyShakespeare
    [("--tiny-shakespeare", Examples.RealData.tinyShakespeare)]
    Examples.RealData.missingTinyShakespeareHint args

/-- Collect the distinct characters of the corpus in code-point order. -/
def buildAlphabet (s : String) : Array Char :=
  let chars : List Char := s.toList.eraseDups
  -- Deterministic order: sort by codepoint.
  let sorted := List.mergeSort chars (fun a b => decide (a.toNat ≤ b.toNat))
  sorted.toArray

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "chargpt"

/--
Architecture, training, evaluation and generation defaults that do not depend on the corpus or the
runtime device.
-/
structure Preset where
  /-- Default embedding width (`--width`). -/
  modelWidth : Nat
  /-- Default number of attention heads (`--heads`), which must divide the width. -/
  attentionHeads : Nat
  /-- Default number of Transformer blocks (`--layers`). -/
  transformerLayers : Nat
  /-- Default dropout probability (`--dropout`), smaller than `1`. -/
  dropoutProbability : Float
  /-- Default number of optimizer updates (`--steps`). -/
  trainingSteps : Nat
  /-- Default number of training windows per update (`--batch-size`). -/
  batchSize : Nat
  /-- Default context length in characters (`--seq-len`). -/
  contextLength : Nat
  /-- Default AdamW learning rate (`--lr`). -/
  lr : Float
  /-- Default number of updates between validation points (`--eval-every`), where `0` disables
  intermediate evaluation. -/
  evalEvery : Nat
  /-- Default number of batches averaged at each validation point (`--eval-iters`). -/
  evaluationBatches : Nat
  /-- Default number of generated characters (`--generate`). -/
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
    lr := 3e-4
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
    lr := 3e-4
    evalEvery := 500
    evaluationBatches := 200
    generationLength := 500 }

/--
Resolve a `--preset` name after trimming it and converting it to lower case. `smoke` is a short
local run, and `karpathy` uses the hyperparameters of the final model in Karpathy's lecture code
(`gpt.py`).
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
    , "  scripts/lake.sh -Kcuda=true exe torchlean chargpt --device cuda --tiny-shakespeare "
        ++ "--preset PRESET [flags]"
    , ""
    , "Presets:"
    , "  smoke       two-update training run"
    , "  karpathy    full Tiny Shakespeare lecture experiment"
    , ""
    , "Corpus:"
    , "  --data-file PATH | --tiny-shakespeare"
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
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , "  --cuda-mem-watch N sample CUDA allocator state every N updates"
    , ""
    , "Generation:"
    , "  --prompt TEXT --generate N --temperature X --top-k N"
    , "  --repeat-penalty X --repeat-window N --sample-seed N"
    , "  --ascii-only [true|false]"
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
  generation : Text.GenerationOptions
  /-- Optional checkpoint input and output paths. -/
  checkpoint : Text.CheckpointOptions
  /-- Context length in characters. -/
  contextLength : Nat
deriving Repr

namespace Options

/-- Parse training controls after the experiment preset has supplied defaults. -/
def parse (args : List String) (defaults : Preset)
    (generationDefaults : Text.GenerationOptions) : Except String (Options × List String) := do
  let (contextLength, args) ←
    CLI.takePositiveNatFlag args exeName "seq-len" (default := defaults.contextLength)
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := defaults.trainingSteps)
      (defaultLr := defaults.lr)
      (defaultBatchSize := defaults.batchSize)
      (allowZeroSteps := true)
  let (generation, args) ← Text.GenerationOptions.parse exeName args generationDefaults
  let (checkpoint, args) ← Text.CheckpointOptions.parse args
  pure ({ training
          generation
          checkpoint
          contextLength }, args)

end Options

/--
Prediction function of a trained character GPT whose sizes are chosen at run time. It maps a batch
of token-ID windows to one row of vocabulary logits per position.
-/
abbrev Predictor (α : Type) [Storage α] (batchSize contextLength vocabularySize : Nat) :=
  Tensor (Fin vocabularySize) [batchSize, contextLength] →
    IO (Tensor α [batchSize, contextLength, vocabularySize])

/--
Autoregressively extend a prompt of character token IDs with a trained CharGPT model. Each step
repeats the padded context window across the batch and samples the next token from the first row's
logits at the current position.
-/
def generate {α : Type} [Storage α] {promptLength : Nat}
    (toFloat : α → Float)
    (batchSize contextLength vocabularySize : Nat) [NeZero vocabularySize]
    (predict : Predictor α batchSize contextLength vocabularySize)
    (promptTokens : Tensor (Fin vocabularySize) [promptLength])
    (options : Text.GenerationOptions)
    (allowToken : Fin vocabularySize → Bool := fun _ => true)
    (paddingTokenId : Fin vocabularySize := 0) :
    IO (Tensor (Fin vocabularySize) [promptLength + options.continuationLength]) := do
  if hBatchSize : batchSize = 0 then
    throw (IO.userError "generation requires a nonempty model batch")
  else
    let firstBatchRow : Fin batchSize := ⟨0, Nat.pos_of_ne_zero hBatchSize⟩
    let ids ← Text.generate contextLength paddingTokenId.val
      (promptTokens.map Fin.val) options
      (fun padded position => do
        let bounded ← IO.ofExcept (Tensor.checkIndices vocabularySize padded)
        let logits ← predict (Tensor.repeatAxis 0 batchSize bounded)
        pure (((logits.unstack firstBatchRow).unstack position).map toFloat))
      allowToken
    IO.ofExcept (Tensor.checkIndices vocabularySize ids)

/-- CLI entrypoint for character-level GPT training and sampling. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
    return 0
  Module.Command.run
    (config := {
      banner? := some fun _ => s!"{exeName}: char-level GPT training"
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (corpus, rest) ← takeInputText rest
      let (presetName?, rest) ← CLI.orThrow exeName <|
        CLI.takeFlagValue? rest "preset"
      let defaults ← CLI.orThrow exeName <|
        Preset.parse (presetName?.getD "smoke")
      let presetName := presetName?.getD "smoke" |>.trimAscii.toString.toLower
      let defaultPrompt :=
        if presetName == "karpathy" then "\n" else "First Citizen:"
      let generationDefaults : Text.GenerationOptions :=
        if presetName == "karpathy" then
          { prompt := defaultPrompt
            continuationLength := defaults.generationLength
            temperature := 1.0
            topK := 0
            repeatPenalty := 1.0
            repeatWindow := 0
            seed := 1337
            asciiOnly := false }
        else
          { prompt := defaultPrompt
            continuationLength := defaults.generationLength
            temperature := 0.9
            topK := 12
            repeatPenalty := 1.15
            repeatWindow := 64
            seed := 7
            asciiOnly := false }
      let (train, rest) ← CLI.orThrow exeName <|
        Options.parse rest defaults generationDefaults
      let (modelWidth, rest) ← CLI.orThrow exeName <|
        CLI.takePositiveNatFlag rest exeName "width" (default := defaults.modelWidth)
      let (attentionHeads, rest) ← CLI.orThrow exeName <|
        CLI.takePositiveNatFlag rest exeName "heads" (default := defaults.attentionHeads)
      let (transformerLayers, rest) ← CLI.orThrow exeName <|
        CLI.takePositiveNatFlag rest exeName "layers"
          (default := defaults.transformerLayers)
      let (dropoutProbability, rest) ← CLI.orThrow exeName <|
        CLI.takeNonnegativeFloatFlag rest exeName "dropout"
          (default := defaults.dropoutProbability)
      let (evalEvery, rest) ← CLI.orThrow exeName <|
        CLI.takeNatFlag rest "eval-every" (default := defaults.evalEvery)
      let (evaluationBatches, rest) ← CLI.orThrow exeName <|
        CLI.takePositiveNatFlag rest exeName "eval-iters"
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
      let tok := Text.Tokenizer.fromAlphabet alphabetFull unknownTokenId (unknownCharacter := '?')
      let vocabularySize := tok.vocabularySize
      have hVocabularySize : vocabularySize ≠ 0 := by
        intro h
        have hAlphabet : alphabetFull.size = 0 := by
          simpa [vocabularySize, tok, Text.Tokenizer.fromAlphabet] using h
        rw [hAlphabet] at unknownTokenId
        exact Fin.elim0 unknownTokenId
      letI : NeZero vocabularySize := ⟨hVocabularySize⟩
      do
        let batchSize := train.training.batchSize
        let contextLength := train.contextLength
        let headWidthValue := modelWidth / attentionHeads
        let config : nn.CausalTransformer.Config :=
          { sequenceLength := contextLength
            vocabularySize := vocabularySize
            headCount := attentionHeads
            headWidth := headWidthValue
            feedForwardWidth := 4 * modelWidth
            layerCount := transformerLayers
            activation := relu
            dropout? := if dropoutProbability == 0.0 then none else some dropoutProbability
            normalizeFirst := true
            attentionOutputBias := true
            parameterInit? := some (.normal 0.0 0.02) }
        let model : nn.IndexedModel
            (config.tokenShape [batchSize])
            (config.vocabularyShape [batchSize])
            (Fin config.vocabularySize) :=
          nn.build runtime.seed <|
            nn.CausalTransformer.indexed config [batchSize]

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
            Text.Corpus.randomTokenBatch trainTokens batchSize contextLength runtime.seed step
              unknownTokenId
        let validationBatchAt (step : Nat) :=
          Data.CausalLM.tokenSample [batchSize] contextLength <|
            Text.Corpus.randomTokenBatch valTokens batchSize contextLength
              (runtime.seed + 1000003) step unknownTokenId
        let trainDef := nn.CausalTransformer.objective config model
        let evalDef := nn.CausalTransformer.objective config model (mode := nn.Mode.eval)
        let module ← Module.instantiate trainDef runtime
        match train.checkpoint.loadCheckpoint? with
        | none => pure ()
        | some path =>
            module.runtime.loadState (← Checkpoint.State.load model.stateShapes path)
            IO.println s!"  loaded checkpoint: {path}"
        let trainStep ← module.dataStep <|
          optim.adamW { lr := train.training.lr }
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
          pure (losses.sum / (Tensor.meanDenominator [evaluationBatches] : Float))
        let lossBefore ← evalLoss
        IO.println s!"  step 0: val loss={lossBefore}"
        let watchEvery :=
          Trainer.Memory.cadence runtime train.training.steps train.training.cudaMemorySampleEvery
        let mut memorySample? ← Trainer.Memory.sample runtime watchEvery train.training.steps 0 none
        let mut lastEval? : Option Float := none
        for step in [0:train.training.steps] do
          let sample := trainingBatchAt step
          trainStep sample.input sample.target
          let done := step + 1
          memorySample? ←
            Trainer.Memory.sample runtime watchEvery train.training.steps done memorySample?
          if evalEvery != 0 && (done % evalEvery == 0 || done == train.training.steps) then
            let loss ← evalLoss
            IO.println s!"  step {done}: val loss={loss}"
            if done == train.training.steps then
              lastEval? := some loss
        let lossAfter ←
          match lastEval? with
          | some loss => pure loss
          | none => evalLoss
        let predict ← module.indexedPredictor model
        let promptTokens ← CLI.orThrow exeName <|
          Tensor.checkIndices vocabularySize (Tensor.from (tok.encode train.generation.prompt))
        let allowToken : Fin vocabularySize → Bool :=
          if train.generation.asciiOnly then
            fun i => alphabetFull[i.val]?.any (fun c => Text.isPrintableAscii c.toNat)
          else
            fun _ => true
        let outIds ←
          generate Float32.toFloat batchSize contextLength vocabularySize predict
            promptTokens train.generation
            (allowToken := allowToken) (paddingTokenId := unknownTokenId)
        let sampled := Text.escape (tok.decode ((outIds.map Fin.val).to (Array Nat)))
        match train.checkpoint.saveCheckpoint? with
        | none => pure ()
        | some path =>
            Checkpoint.State.save (← module.runtime.state) path
            IO.println s!"  wrote checkpoint: {path}"
        IO.println s!"  vocabularySize={vocabularySize} (unique chars)"
        IO.println s!"  architecture=modelWidth {modelWidth}, attentionHeads {attentionHeads}, \
transformerLayers {transformerLayers}, dropoutProbability {dropoutProbability}"
        IO.println s!"  sampled={sampled}"
        Text.Log.writeGeneration
          train.training.logDestination
            "CharGPT (minGPT-style)" train.training.steps lossBefore lossAfter
          train.generation sampled
          #[Training.deviceNote runtime,
            s!"vocabularySize={vocabularySize}",
            s!"train_tokens={trainCount}",
            s!"validation_tokens={validationCount}",
            Training.cudaMemoryNote runtime train.training.steps
              train.training.cudaMemorySampleEvery]
        pure ())

end TorchLean.Examples.CharGPT
