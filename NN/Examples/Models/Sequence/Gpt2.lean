/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA text example:
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1 --windows 1 --generate 0
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
    --prompt "First Citizen:" --steps 1 \
    --windows 1 --generate 0 --temperature 0.85 --top-k 12 --sample-seed 7
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 \
    --save-checkpoint data/examples/gpt2_shakespeare.state.json
  lake -R -K cuda=true exe torchlean gpt2_saved --device cuda \
    --checkpoint data/examples/gpt2_shakespeare.state.json \
    --prompt "First Citizen:" --generate 0

Dataset example:
  python3 scripts/datasets/download_example_data.py --tiny-shakespeare
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 \
    --generate 0

This is a GPT-2-style *causal* language-model command (byte-level tokens).

Performance note: use CUDA for this example. The pure Lean CPU path exists for debugging tiny model
states, but Transformer workloads are too slow there for a useful run. The default command uses one
training window so it finishes quickly; pass larger `--steps`, `--windows`, and `--generate` values
when you want a real text experiment.
The command exercises masked self-attention, LayerNorm, and feed-forward blocks through the public
`TorchLean.nn` model constructors and `TorchLean.text` token tools.

After a run that writes `--log <path>`, you can view the prompt and sampled continuation in the
infoview via:

`#train_log_file_view "<path>"`
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# GPT-2-Style Causal Language Model Example

Runnable `torchlean gpt2` example. It builds a GPT-2-style causal transformer over
byte-level tokens, with optional real text input from tiny-shakespeare or `--data-file PATH`.

For the simplest "Karpathy-style single text file" path, use `torchlean chargpt`
(character-level tokenizer). This `gpt2` command is byte-level and shows the Transformer block
wiring and save/reload loop.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 \
  --generate 0
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Gpt2

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "gpt2"

/-- Complete command help, including the text and training flags parsed after runtime selection. -/
def usage : String :=
  Module.Command.usage s!"lake exe torchlean {exeName}" ++ "\n" ++ String.intercalate "\n"
    [ "Text data:"
    , "  --data-file PATH | --tiny-shakespeare | --tinystories-valid"
    , ""
    , "Training:"
    , "  --steps N          optimizer updates"
    , "  --batch-size N     corpus windows accumulated per update"
    , "  --windows N        corpus windows available to training"
    , "  --lr X             Adam learning rate"
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , "  --cuda-mem-watch N sample CUDA allocator state every N updates"
    , "  --load-checkpoint PATH | --save-checkpoint PATH"
    , ""
    , "Generation:"
    , "  --prompt TEXT --generate N --temperature X --top-k N"
    , "  --repeat-penalty X --repeat-window N --sample-seed N"
    , "  --ascii-only [true|false] --interactive"
    ]

/-- Batch size for the byte-level causal Transformer. -/
def batchSize : Nat := 1

local instance : NeZero batchSize := ⟨by decide⟩

/-- Prompt/target window length for the runnable GPT example. -/
def contextLength : Nat := 4

/-- Byte vocabulary width used by the one-hot tokenizer. -/
def vocabularySize : Nat := 256

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- Embed a byte id in the complete 256-entry byte vocabulary. -/
def byteIndex (id : Nat) : Fin vocabularySize :=
  Fin.ofNat vocabularySize id

/-- Number of attention heads in the miniature Transformer block. -/
def attentionHeads : Nat := 2

/-- Per-head embedding width. -/
def attentionHeadWidth : Nat := 2

/-- Transformer embedding width. -/
def modelWidth : Nat := attentionHeads * attentionHeadWidth

/-- Hidden width of the feed-forward sublayer. -/
def feedForwardWidth : Nat := 8

/-- Number of Transformer encoder blocks. -/
def transformerLayers : Nat := 1

local instance : NeZero contextLength := ⟨by decide⟩
local instance : NeZero modelWidth := ⟨by decide⟩

/-- Input shape: batched byte-level one-hot token windows. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/-- Output shape: one vocabulary-logit row for every input token position. -/
abbrev output : Shape :=
  input

/-- Public GPT-style causal Transformer constructor specialized to the byte-level config. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.CausalTransformer.oneHot
    { sequenceLength := contextLength
      vocabularySize := vocabularySize
      headCount := attentionHeads
      headWidth := attentionHeadWidth
      feedForwardWidth := feedForwardWidth
      layerCount := transformerLayers }
    (batchShape := [batchSize])

/-- Command-local controls for GPT training, checkpointing, generation, and the prompt loop. -/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Prompt and token-sampling policy. -/
  generation : text.GenerationOptions
  /-- Number of corpus windows exposed to training. -/
  window : text.WindowOptions
  /-- Optional checkpoint input and output paths. -/
  checkpoint : text.CheckpointOptions
  /-- Terminal prompt-loop policy. -/
  interaction : text.InteractiveOptions
deriving Repr

namespace Options

/-- Parse the byte-level GPT command's training and generation flags. -/
def parse (args : List String) (defaultSteps : Nat) :
    Except String (Options × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := defaultSteps) (defaultLearningRate := 0.001)
      (allowZeroSteps := true)
  let (window, args) ← text.WindowOptions.parse exeName args 1
  let (generation, args) ← text.GenerationOptions.parse exeName args
    { prompt := "First Citizen:"
      newTokenCount := 0
      temperature := 0.85
      topK := 12
      repeatPenalty := 1.25
      repeatWindow := 24
      seed := 0
      asciiOnly := false }
  let (checkpoint, args) ← text.CheckpointOptions.parse args
  let (interactive, args) ← text.InteractiveOptions.parse args
  pure ({ training, generation, window, checkpoint, interaction := interactive }, args)

end Options

/-- Build a batch sample from exactly one token window per batch row. -/
def batchSampleFromTokenIds (idsByBatch : Tensor Nat [batchSize, contextLength + 1]) :
    Sample.Supervised Float input output :=
  Data.CausalLM.oneHotSample (α := Float) [batchSize] contextLength vocabularySize
    (idsByBatch.map byteIndex)

/--
Parse GPT-2-specific data flags and return the training corpus plus remaining runtime flags.
-/
def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName NN.Examples.Data.RealPaths.tinyShakespeare
    [("--tiny-shakespeare", NN.Examples.Data.RealPaths.tinyShakespeare),
      ("--tinystories-valid", NN.Examples.Data.RealPaths.tinyStoriesValid)]
    NN.Examples.Data.RealPaths.missingTinyShakespeareOrTinyStoriesHint args

/-- Byte-token window used for reporting prompt/target text. -/
def tokenWindowIds (prompt : String) (offset : Nat) : Tensor Nat [contextLength] :=
  text.tokenWindow text.Tokenizer.byte contextLength prompt
    (offset := offset) (paddingTokenId := 32)

/-- Print a compact before/after language-model report for the first batch row. -/
def printPredictionReport (label prompt : String) (logits : Tensor Float output) :
    IO Unit := do
  let predIds := text.batchArgmaxTokens (α := Float) logits 0
  IO.println s!"  {label} pred={text.formatByteTokens predIds}"
  IO.println s!"  prompt={text.formatByteTokens (tokenWindowIds prompt 0)}"
  IO.println s!"  target={text.formatByteTokens (tokenWindowIds prompt 1)}"

/-- Convert byte ids into the typed batched one-hot input tensor used for generation. -/
def inputTensorFromIds (ids : Tensor Nat [contextLength]) : Tensor Float input :=
  Tensor.repeatAxis 0 batchSize <|
    Data.CausalLM.oneHotInputs (α := Float) vocabularySize (ids.map byteIndex)

/--
Fitted byte-level GPT predictor.

Training, saved-checkpoint inference, and future optimized runners all provide this one closure.
Generation only needs a logit-producing function; it does not depend on where the logits came from.
-/
abbrev Predictor :=
  Tensor Float input → IO (Tensor Float output)


/-- Autoregressively extend byte token ids using a trained byte-level GPT model. -/
def generateSampledFromIds {promptLength : Nat}
    (predict : Predictor)
    (promptTokens : Tensor Nat [promptLength]) (steps : Nat) (temperature : Float)
    (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (Tensor Nat [promptLength + steps]) := do
  let gen : text.GenerationOptions :=
    { prompt := ""
      newTokenCount := steps
      temperature := temperature
      topK := topK
      repeatPenalty := repeatPenalty
      repeatWindow := repeatWindow
      seed := seed
      asciiOnly := asciiOnly }
  let allowToken := if asciiOnly then text.isPrintableAscii else fun _ => true
  let ids ←
    text.autoregressiveTokenIds contextLength 32 promptTokens gen
      (fun padded predPos => do
        let logits ← predict (inputTensorFromIds padded)
        pure (text.batchLogitScoresAt logits 0 predPos))
      (allowToken := fun i => allowToken i.val)
  pure ids

/-- Encode a string prompt and autoregressively extend it. -/
def generateSampled
    (predict : Predictor)
    (prompt : String) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) :
    IO (Tensor Nat [(text.Tokenizer.byte.encode prompt).size + steps]) := do
  let init := text.Tokenizer.byte.encode prompt
  generateSampledFromIds predict (Tensor.from init) steps temperature topK seed
    repeatWindow repeatPenalty asciiOnly

/--
Tokenize the corpus once and build a finite training set from approximately evenly spaced windows.
The sample stream retains the token tensor across batch requests.
-/
def samplesFromCorpus (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input output) :=
  let toks := text.Tokenizer.byte.encode corpus
  let tokens := Tensor.from toks
  let offs := text.Corpus.evenlySpacedOffsets toks.size contextLength windows
  Data.SampleStream.fromFunction windows (fun index =>
    let off := offs[index]
    let idsByBatch : Tensor Nat [batchSize, contextLength + 1] :=
      Tensor.stack 0 fun i =>
        let off' :=
          (off + i.val * (contextLength / 2 + 1)) %
            text.Corpus.usableTokenStarts toks.size contextLength
        Tensor.window tokens (contextLength + 1) off' 32
    batchSampleFromTokenIds idsByBatch)

/--
Interactive prompt loop for the in-memory Float model.

Each line is appended to the current byte context, decoded through the trained local model, and then
kept as context for the next prompt unless the user clears it.
-/
partial def interactiveLoopFloat
    (predict : Predictor)
    (options : Options) :
    IO Unit := do
  IO.println ("  interactive: enter text; :q exits, :clear resets, :show prints context "
    ++ s!"(window={contextLength} bytes)")
  let stdin ← IO.getStdin
  let rec loop {count : Nat} (ctx : Tensor Nat [count]) : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else if prompt = ":clear" then
      IO.println "  interactive: cleared context"
      loop (Tensor.full [0] 0)
    else if prompt = ":show" then
      IO.println s!"  context={text.formatByteTokens ctx}"
      loop ctx
    else
      let encoded := text.Tokenizer.byte.encode prompt
      let inputIds :=
        Tensor.concat (Tensor.concat ctx (Tensor.from encoded)) ([10] : Tensor Nat [1])
      let outIds ←
        generateSampledFromIds predict inputIds options.generation.newTokenCount
          options.generation.temperature options.generation.topK options.generation.seed
          options.generation.repeatWindow options.generation.repeatPenalty
          options.generation.asciiOnly
      let genOnly := Tensor.window outIds options.generation.newTokenCount
        (count + encoded.size + 1) 0
      IO.println s!"  generated={text.formatByteTokens genOnly}"
      loop outIds
  loop (Tensor.full [0] 0)

/--
Train the byte-level model and decode a prediction report.

Inputs and reports use host `Float`, while `RunConfig.arithmetic` selects the arithmetic used by
the trainer. The command fixes that runtime choice to native execution.
-/
def trainAndDecode (runtime : Runtime.Config) (corpus : String)
    (options : Options) :
    IO (Float × Float × String) := do
  let samples := samplesFromCorpus corpus options.window.windowCount
  let reportSample :=
    Data.CausalLM.byteBatch
      (α := Float) batchSize contextLength vocabularySize byteIndex options.generation.prompt
  let run := Trainer.RunConfig.fromRuntime runtime
    { optimizer := optim.adam { learningRate := options.training.learningRate } }
  let objective : Trainer.Objective output := .oneHotCrossEntropy 2
  let trainer := Trainer.new model <|
    Trainer.RunConfig.forObjective run objective (seed := runtime.seed)
  trainer.printSummary

  /-
  The GPT-2 command trains on a bounded, prompt-aware window table.  That makes the training
  schedule explicit and reproducible, and it lets the public trainer own checkpointing and optimizer
  state.  The example stays focused on text windows, decoding, and generation instead of runtime
  module bookkeeping.
  -/
  let trained ← trainer.train
    (Data.fromStream samples)
    { steps := options.training.steps
      samplesPerStep := options.training.batchSize
      cudaMemorySampleEvery := options.training.cudaMemorySampleEvery
      logDestination := .disabled
      loadCheckpoint? := options.checkpoint.loadCheckpoint?
      saveCheckpoint? := options.checkpoint.saveCheckpoint? }
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after

  let afterLogits ← trained.predict reportSample.input
  printPredictionReport "after " options.generation.prompt afterLogits
  let generatedIds ←
    generateSampled trained.predict options.generation.prompt options.generation.newTokenCount
      options.generation.temperature options.generation.topK options.generation.seed
      options.generation.repeatWindow options.generation.repeatPenalty
      options.generation.asciiOnly
  let generated := text.formatByteTokens generatedIds
  IO.println s!"  generated={generated}"
  IO.println s!"  corpus_bytes={corpus.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({options.generation.topK}), temperature={
    options.generation.temperature}, seed={options.generation.seed}"
  IO.println s!"  repetition_penalty={options.generation.repeatPenalty} repeat_window={
    options.generation.repeatWindow}"
  if options.interaction.interactive then
    interactiveLoopFloat trained.predict options
  let cudaMemorySampleEvery :=
    Trainer.Memory.cadence runtime options.training.steps options.training.cudaMemorySampleEvery
  text.Log.writeGeneration
    options.training.logDestination
      "GPT-2 byte prompt training" options.training.steps lossBefore lossAfter
    options.generation generated
    #[Support.deviceNote runtime,
      s!"windows={options.window.windowCount}",
      s!"cuda_mem_watch={cudaMemorySampleEvery}"]
  pure (lossBefore, lossAfter, generated)

/-- CLI entrypoint for byte-level GPT training, sampling, logging, and checkpointing. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDevice exeName "causal LM training"
      usage? := some usage
      printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (corpus, rest) ← takeInputText rest
      let defaultSteps : Nat := if runtime.usesCuda then 1 else 0
      let (train, rest) ← CLI.orThrow exeName <|
        Options.parse rest defaultSteps
      CLI.requireNoArgs exeName rest
      let _ ← trainAndDecode runtime corpus train)

end NN.Examples.Models.Sequence.Gpt2
