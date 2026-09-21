/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# Mamba Text Training

Runnable byte-level language-model training with the public Mamba API constructor.

The model is trainable end-to-end:

`mamba(contextLength, vocabularySize, stateWidth) → linear(stateWidth → vocabularySize)`

and the same code runs on CPU or CUDA through TorchLean autograd.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare --steps 1 --windows 1 \
  --generate 0
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Mamba

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "mamba"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "mamba"

/-- Complete command help, including the text and training flags parsed after runtime selection. -/
def usage : String :=
  Module.Command.usage s!"lake exe torchlean {exeName}" ++ "\n" ++ String.intercalate "\n"
    [ "Text data:"
    , "  --data-file PATH | --tiny-shakespeare | --tinystories-valid"
    , ""
    , "Training:"
    , "  --steps N          optimizer updates (default: 1)"
    , "  --batch-size N     corpus windows accumulated per update (default: 1)"
    , "  --windows N        corpus windows available to training (default: 1)"
    , "  --lr X             Adam learning rate (default: 0.002)"
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , "  --cuda-mem-watch N sample CUDA allocator state every N updates"
    , ""
    , "Generation:"
    , "  --prompt TEXT --generate N --temperature X --top-k N --sample-seed N"
    ]

/-- Training and generation context length for the Mamba text example. -/
def contextLength : Nat := 4

/-- Byte tokenizer used by this sequence model. -/
def tokenizer : text.Tokenizer := text.Tokenizer.byte

/-- Number of byte-token classes. -/
def vocabularySize : Nat := 256

/-- Width of each token embedding and of the Mamba block's output. -/
def modelWidth : Nat := 4

/-- Byte value used to pad short text windows. -/
def paddingByte : Nat := 32

/-- Mamba text-model configuration shared by shapes and the constructor. -/
abbrev modelConfig : nn.models.Mamba.Config :=
  { vocabularySize := vocabularySize
    modelWidth := modelWidth }

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- Embed a byte id in the complete 256-entry byte vocabulary. -/
def byteIndex (id : Nat) : Fin vocabularySize :=
  Fin.ofNat vocabularySize id

/-- Input shape: one sequence of one-hot byte tokens. -/
abbrev input : Shape := modelConfig.inputShape contextLength

/-- Output shape: one vocabulary-logit row per input position. -/
abbrev output : Shape := modelConfig.outputShape contextLength

/-- Public Mamba language-model constructor specialized to the example config. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.Mamba.languageModel modelConfig contextLength

/-- Command-local training, sampling, and corpus-window controls. -/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Prompt and token-sampling policy. -/
  generation : text.GenerationOptions
  /-- Number of corpus windows exposed to training. -/
  window : text.WindowOptions
deriving Repr

namespace Options

/-- Parse the Mamba command's training and sampling flags. -/
def parse (args : List String) : Except String (Options × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 1) (defaultLearningRate := 0.002)
  let (window, args) ← text.WindowOptions.parse exeName args 1
  let (generation, args) ← text.GenerationOptions.parse exeName args
    { prompt := "First Citizen:"
      newTokenCount := 0
      temperature := 0.9
      topK := 16
      repeatPenalty := 1.0
      repeatWindow := 0
      seed := 0
      asciiOnly := false }
  pure ({ training, generation, window }, args)

end Options

/-- Convert a token window into the one-hot next-token sample consumed by the Mamba model. -/
def sampleFromTokenIds (ids : Tensor Nat [contextLength + 1]) :
    Sample.Supervised Float input output :=
  Data.CausalLM.oneHotSample (α := Float) []
    (sequenceLength := contextLength)
    (vocabularySize := vocabularySize)
    (ids.map byteIndex)

/-- Build a finite training set from approximately evenly spaced corpus windows. -/
def samplesFromCorpus (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  Data.CausalLM.byteSamples contextLength vocabularySize byteIndex windows corpus paddingByte

/-- Print the current argmax prediction beside the prompt and shifted target text. -/
def printPredictionReport (label prompt : String) (logits : Tensor Float output) : IO Unit := do
  IO.println
    s!"  {label} pred={text.escape (text.decodeArgmaxLogits tokenizer logits)}"
  IO.println s!"  prompt={
    text.escape (text.decodeWindow tokenizer contextLength prompt
      (paddingTokenId := paddingByte))}"
  IO.println s!"  target={
    text.escape (text.decodeWindow tokenizer contextLength prompt
      (offset := 1) (paddingTokenId := paddingByte))}"

/-- Convert a prompt window into the typed one-hot input tensor used during generation. -/
def inputTensorFromIds (ids : Tensor Nat [contextLength]) : Tensor Float input :=
  Tensor.oneHotIndices (α := Float) vocabularySize (ids.map byteIndex)

/-- Autoregressively extend a prompt using the trained Mamba parameters. -/
partial def generateSampled
    (predict : Tensor Float input → IO (Tensor Float output))
    (generation : text.GenerationOptions) : IO String := do
  let allowToken :=
    if generation.asciiOnly then text.isPrintableAscii else fun _ => true
  let ids ←
    text.autoregressiveTokenIds contextLength paddingByte
      (Tensor.from (tokenizer.encode generation.prompt)) generation
      (fun padded predPos => do
        let logits ← predict (inputTensorFromIds padded)
        pure (text.logitScoresAt logits predPos))
      (allowToken := fun i => allowToken i.val)
  pure (tokenizer.decode (ids.to (Array Nat)))

/-- Train the Mamba language model and print before/after prediction and generation reports. -/
def trainOnText (runtime : Runtime.Config) (corpus : String)
    (options : Options) :
    IO (Float × Float) := do
  let samples := samplesFromCorpus corpus options.window.windowCount
  let reportSample := sampleFromTokenIds <|
    text.tokenWindow tokenizer (contextLength + 1) options.generation.prompt
      (paddingTokenId := paddingByte)
  let run := Trainer.RunConfig.fromRuntime runtime
    { optimizer := optim.adam { learningRate := options.training.learningRate } }
  let trainer := Trainer.new model <|
    Trainer.RunConfig.forObjective run (.oneHotCrossEntropy 1) (seed := runtime.seed)
  let cudaMemorySampleEvery :=
    Trainer.Memory.cadence runtime options.training.steps options.training.cudaMemorySampleEvery
  let beforeLogits ← trainer.predict reportSample.input
  printPredictionReport "before" options.generation.prompt beforeLogits
  let trained ← trainer.train
    (Data.fromStream samples)
    (options.training.trainOptions
      (enableLog := false)
      (logTitle := "Mamba text training")
      (logNotes := #[Support.deviceNote runtime, s!"windows={options.window.windowCount}",
        s!"cuda_mem_watch={cudaMemorySampleEvery}"]))
  let afterLogits ← trained.predict reportSample.input
  printPredictionReport "after " options.generation.prompt afterLogits
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after
  let generated ← generateSampled trained.predict options.generation
  IO.println s!"  generated={text.escape generated}"
  IO.println s!"  corpus_bytes={corpus.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({options.generation.topK}), temperature={
    options.generation.temperature}, seed={options.generation.seed}"
  pure (lossBefore, lossAfter)

/-- CLI entrypoint for the Mamba text command. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDevice exeName "Mamba text training"
      usage? := some usage
      printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (corpus, rest) ← CLI.orThrow exeName <| RealData.TextCorpusFlags.parse rest
      let (train, rest) ← CLI.orThrow exeName <|
        Options.parse rest
      CLI.requireNoArgs exeName rest
      let corpusText ← RealData.TextCorpusFlags.read exeName corpus
      let (lossBefore, lossAfter) ← trainOnText runtime corpusText train
      let extraNotes :=
        #[s!"data={corpus.path}", Support.deviceNote runtime,
          s!"windows={train.window.windowCount}", s!"lr={train.training.learningRate}",
          Support.cudaMemoryNote runtime train.training.steps train.training.cudaMemorySampleEvery]
      text.Log.writeGeneration
        train.training.logDestination
          "Mamba text training" train.training.steps lossBefore lossAfter
        train.generation none extraNotes
    )

end NN.Examples.Models.Sequence.Mamba
