/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Data.CausalLM
public import TorchLean.NN.Models.Mamba
public import TorchLean.NN.Text.Generation
public import TorchLeanExamples.Models.Training

/-!
# Mamba Text Training

Runnable byte-level language-model training with the public Mamba constructor
`nn.Mamba.languageModel`.

The model is trainable end-to-end:

`mamba(contextLength, vocabularySize, modelWidth) → linear(modelWidth → vocabularySize)`

and the same code runs on CPU or CUDA through TorchLean autograd.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
scripts/lake.sh -Kcuda=true exe torchlean mamba --device cuda \
    --tiny-shakespeare --steps 1 --windows 1 \
  --generate 0
```
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.Mamba

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "mamba"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "mamba"

/-- Complete command help, including the text and training flags parsed after runtime selection. -/
def usage : String :=
  Module.Command.usage s!"scripts/lake.sh exe torchlean {exeName}" ++ "\n" ++
    String.intercalate "\n"
    [ "Text data:"
    , "  --data-file PATH | --tiny-shakespeare"
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
    , "  --repeat-penalty X --repeat-window N --ascii-only [true|false]"
    ]

/-- Training and generation context length for the Mamba text example. -/
def contextLength : Nat := 4

/-- Byte tokenizer used by this sequence model. -/
def tokenizer : Text.Tokenizer := Text.Tokenizer.byte

/-- Number of byte-token classes. -/
def vocabularySize : Nat := 256

/-- Output width of the single Mamba layer, which is also the input width of the vocabulary head. -/
def modelWidth : Nat := 4

/-- Byte value used to pad short text windows, the ASCII space. -/
def paddingByte : Nat := 32

/-- Mamba text-model configuration shared by shapes and the constructor. -/
abbrev modelConfig : nn.Mamba.Config :=
  { vocabularySize := vocabularySize
    modelWidths := [modelWidth] }

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- Map a byte id into the 256-entry byte vocabulary, reducing it modulo 256. -/
def byteIndex (id : Nat) : Fin vocabularySize :=
  Fin.ofNat vocabularySize id

/-- Input shape: one sequence of one-hot byte tokens. -/
abbrev input : Shape := modelConfig.inputShape contextLength

/-- Output shape: one vocabulary-logit row per input position. -/
abbrev output : Shape := modelConfig.outputShape contextLength

/-- Public Mamba language-model constructor specialized to the example config. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.Mamba.languageModel modelConfig contextLength

/-- Command-local training, sampling, and corpus-window controls. -/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Prompt and token-sampling policy. -/
  generation : Text.GenerationOptions
  /-- Number of corpus windows exposed to training. -/
  windows : Nat
deriving Repr

namespace Options

/--
Parse the Mamba command's training and sampling flags. The defaults are one step at an Adam
learning rate of `0.002`, one corpus window and no generated tokens.
-/
def parse (args : List String) : Except String (Options × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 1) (defaultLr := 0.002)
  let (windows, args) ← CLI.takePositiveNatFlag args exeName "windows" (default := 1)
  let (generation, args) ← Text.GenerationOptions.parse exeName args
    { prompt := "First Citizen:"
      continuationLength := 0
      temperature := 0.9
      topK := 16
      repeatPenalty := 1.0
      repeatWindow := 0
      seed := 0
      asciiOnly := false }
  pure ({ training, generation, windows }, args)

end Options

/-- Convert a token window into the one-hot next-token sample consumed by the Mamba model. -/
def sample (ids : Tensor Nat [contextLength + 1]) :
    Sample.Supervised Float input output :=
  Data.CausalLM.oneHotSample (α := Float) []
    (sequenceLength := contextLength)
    (vocabularySize := vocabularySize)
    (ids.map byteIndex)

/-- Build a finite training set from approximately evenly spaced corpus windows. -/
def samples (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  Data.CausalLM.byteSamples contextLength vocabularySize byteIndex windows corpus paddingByte

/-- Print the current argmax prediction beside the prompt and shifted target text. -/
def printPrediction (label prompt : String) (logits : Tensor Float output) : IO Unit := do
  IO.println
    s!"  {label} pred={Text.escape (Text.decodeLogits tokenizer logits)}"
  IO.println s!"  prompt={
    Text.escape (Text.decodeWindow tokenizer contextLength prompt
      (paddingTokenId := paddingByte))}"
  IO.println s!"  target={
    Text.escape (Text.decodeWindow tokenizer contextLength prompt
      (offset := 1) (paddingTokenId := paddingByte))}"

/--
Autoregressively extend the prompt with `predict`, restricted to printable ASCII when `asciiOnly`
is set, and decode the result as text.
-/
def generate
    (predict : Tensor Float input → IO (Tensor Float output))
    (generation : Text.GenerationOptions) : IO String := do
  let allowToken :=
    if generation.asciiOnly then Text.isPrintableAscii else fun _ => true
  let ids ←
    Text.generate contextLength paddingByte
      (Tensor.from (tokenizer.encode generation.prompt)) generation
      (fun padded predPos => do
        let logits ← predict (Tensor.oneHotIndices vocabularySize (padded.map byteIndex))
        pure (logits.unstack predPos))
      (allowToken := fun i => allowToken i.val)
  pure (tokenizer.decode (ids.to (Array Nat)))

/--
Train the model and print its argmax predictions for the prompt before and after training. The
result holds the losses before and after training and the generated text, for the log.
-/
def train (runtime : Session.Config) (corpus : String)
    (options : Options) :
    IO (Float × Float × String) := do
  let samples := samples corpus options.windows
  let reportSample := sample <|
    Text.tokenWindow tokenizer (contextLength + 1) options.generation.prompt
      (paddingTokenId := paddingByte)
  let trainer := Trainer.create model <|
    { runtime with
        optimizer := optim.adam { lr := options.training.lr }
        objective := oneHotCrossEntropy 1 }
  let beforeLogits ← trainer.predict reportSample.input
  printPrediction "before" options.generation.prompt beforeLogits
  let trained ← trainer.train
    (Data.fromStream samples)
    (options.training.trainOptions
      (enableLog := false)
      (logTitle := "Mamba text training")
      (logNotes := #[Training.deviceNote runtime, s!"windows={options.windows}",
        Training.cudaMemoryNote runtime options.training.steps
          options.training.cudaMemorySampleEvery]))
  let afterLogits ← trained.predict reportSample.input
  printPrediction "after " options.generation.prompt afterLogits
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after
  let generated ← generate trained.predict options.generation
  IO.println s!"  generated={Text.escape generated}"
  IO.println s!"  corpus_bytes={corpus.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({options.generation.topK}), temperature={
    options.generation.temperature}, seed={options.generation.seed}"
  pure (lossBefore, lossAfter, generated)

/--
CLI entrypoint for the Mamba text command. The trainer's own log is disabled, and the command writes
one generation log with the losses and the sampled text.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner exeName "Mamba text training"
      usage? := some usage
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (corpus, rest) ← CLI.orThrow exeName <| Corpus.parse rest
      let (options, rest) ← CLI.orThrow exeName <|
        Options.parse rest
      CLI.requireNoArgs exeName rest
      let corpusText ← Corpus.read exeName corpus
      let (lossBefore, lossAfter, generated) ← train runtime corpusText options
      let extraNotes :=
        #[s!"data={corpus}", Training.deviceNote runtime,
          s!"windows={options.windows}", s!"lr={options.training.lr}",
          Training.cudaMemoryNote runtime options.training.steps
            options.training.cudaMemorySampleEvery]
      Text.Log.writeGeneration
        options.training.logDestination
          "Mamba text training" options.training.steps lossBefore lossAfter
        options.generation (some (Text.escape generated)) extraNotes
    )

end TorchLean.Examples.Mamba
