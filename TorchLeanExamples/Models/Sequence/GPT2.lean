/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA text example:
  scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda \
    --steps 1 --windows 1 --generate 0
  scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare \
    --prompt "First Citizen:" --steps 1 \
    --windows 1 --generate 0 --temperature 0.85 --top-k 12 --sample-seed 7
  scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda \
    --tiny-shakespeare --steps 1 --windows 1 \
    --save-checkpoint data/examples/gpt2_shakespeare.state
  scripts/lake.sh -Kcuda=true exe torchlean gpt2_saved --device cuda \
    --checkpoint data/examples/gpt2_shakespeare.state \
    --prompt "First Citizen:" --generate 0

Use CUDA for this example. The pure Lean CPU path is there for debugging tiny model states, and
Transformer workloads run too slowly on it for a useful run. These commands train on one window so
they finish quickly, and larger `--steps`, `--windows` and `--generate` values give a real text
experiment. The command exercises masked self-attention, LayerNorm and feed-forward blocks through
the public `TorchLean.nn` model constructors and the `TorchLean.Text` token tools.

A run writes a `TrainLog` JSON (`--log PATH` picks the file and `--log false` turns it off), and
`#train_log_file_view "<path>"` shows its prompt and any sampled continuation in the infoview.
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Data.CausalLM
public import TorchLean.NN.Models.CausalTransformer
public import TorchLean.NN.Text.Generation
public import TorchLeanExamples.Models.Training

/-!
# GPT-2-Style Causal Language Model Example

Runnable `torchlean gpt2` example. It builds a GPT-2-style causal transformer over byte-level
tokens and trains it on real text: tiny-shakespeare by default, the TinyStories validation file
with `--tinystories-valid`, or any UTF-8 file given by `--data-file PATH`.

For the simplest "Karpathy-style single text file" path, use `torchlean chargpt`
(character-level tokenizer). The `gpt2` command is byte-level and shows the Transformer block
wiring and checkpoint saving and reloading.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda \
    --tiny-shakespeare --steps 1 --windows 1 \
  --generate 0
```
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.GPT2

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "gpt2"

/-- Complete command help, including the text and training flags parsed after runtime selection. -/
def usage : String :=
  Module.Command.usage s!"scripts/lake.sh exe torchlean {exeName}" ++ "\n" ++
    String.intercalate "\n"
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

/-- Size of the byte vocabulary, which is also the width of each one-hot input row. -/
def vocabularySize : Nat := 256

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- Map a byte id into the 256-entry byte vocabulary, reducing it modulo 256. -/
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

/-- Number of causal Transformer blocks. -/
def transformerLayers : Nat := 1

local instance : NeZero contextLength := ⟨by decide⟩
local instance : NeZero modelWidth := ⟨by decide⟩

/-- Shape of the batched byte-level one-hot token windows, and of the model's vocabulary logits,
one row per token position. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/-- Public GPT-style causal Transformer constructor specialized to the byte-level config. -/
def model : nn.Builder (nn.Sequential input input) :=
  nn.CausalTransformer.oneHot
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
  generation : Text.GenerationOptions
  /-- Number of corpus windows exposed to training. -/
  windows : Nat
  /-- Optional checkpoint input and output paths. -/
  checkpoint : Text.CheckpointOptions
  /-- Terminal prompt-loop policy. -/
  interactive : Bool
deriving Repr

namespace Options

/--
Parse the byte-level GPT command's training, generation, checkpoint and `--interactive` flags.
Zero steps are allowed, and the defaults are an Adam learning rate of `0.001`, one corpus window
and no generated tokens.
-/
def parse (args : List String) (defaultSteps : Nat) :
    Except String (Options × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := defaultSteps) (defaultLr := 0.001)
      (allowZeroSteps := true)
  let (windows, args) ← CLI.takePositiveNatFlag args exeName "windows" (default := 1)
  let (generation, args) ← Text.GenerationOptions.parse exeName args
    { prompt := "First Citizen:"
      continuationLength := 0
      temperature := 0.85
      topK := 12
      repeatPenalty := 1.25
      repeatWindow := 24
      seed := 0
      asciiOnly := false }
  let (checkpoint, args) ← Text.CheckpointOptions.parse args
  let (interactive, args) ← CLI.takeBoolFlag args "interactive"
  pure ({ training, generation, windows, checkpoint, interactive }, args)

end Options

/--
Build a one-hot batch sample from one window of `contextLength + 1` byte ids per batch row. The
input holds the first `contextLength` ids of each window and the target the same window shifted by
one.
-/
def batchSampleFromTokenIds (idsByBatch : Tensor Nat [batchSize, contextLength + 1]) :
    Sample.Supervised Float input input :=
  Data.CausalLM.oneHotSample (α := Float) [batchSize] contextLength vocabularySize
    (idsByBatch.map byteIndex)

/--
Parse GPT-2-specific data flags and return the training corpus plus remaining runtime flags.
-/
def takeInputText (args : List String) : IO (String × List String) :=
  Text.Corpus.takeUTF8Input exeName Examples.RealData.tinyShakespeare
    [("--tiny-shakespeare", Examples.RealData.tinyShakespeare),
      ("--tinystories-valid", Examples.RealData.tinyStoriesValid)]
    Examples.RealData.missingTinyCorporaHint args

/--
The `contextLength` byte ids of the prompt from `offset` on, padded with spaces (byte 32), for the
prompt and target lines of the report.
-/
def tokenWindowIds (prompt : String) (offset : Nat) : Tensor Nat [contextLength] :=
  Text.tokenWindow Text.Tokenizer.byte contextLength prompt
    (offset := offset) (paddingTokenId := 32)

/-- Print a compact before/after language-model report for the first batch row. -/
def printPredictionReport (label prompt : String) (logits : Tensor Float input) :
    IO Unit := do
  let predIds := Text.argmaxTokens (α := Float) (logits.unstack 0)
  IO.println s!"  {label} pred={Text.formatBytes predIds}"
  IO.println s!"  prompt={Text.formatBytes (tokenWindowIds prompt 0)}"
  IO.println s!"  target={Text.formatBytes (tokenWindowIds prompt 1)}"

/-- Convert byte ids into the typed batched one-hot input tensor used for generation. -/
def inputTensorFromIds (ids : Tensor Nat [contextLength]) : Tensor Float input :=
  Tensor.repeatAxis 0 batchSize <|
    Tensor.oneHotIndices (α := Float) vocabularySize (ids.map byteIndex)

/--
Fitted byte-level GPT predictor.

Training and saved-checkpoint inference both provide this one closure.
Generation only needs a logit-producing function. It does not depend on where the logits come from.
-/
abbrev Predictor :=
  Tensor Float input → IO (Tensor Float input)


/-- Extend a token context using the selected sampling policy. -/
def generate {promptLength : Nat}
    (predict : Predictor) (tokens : Tensor Nat [promptLength])
    (options : Text.GenerationOptions) :
    IO (Tensor Nat [promptLength + options.continuationLength]) :=
  Text.generate contextLength 32 tokens options
    (fun padded position => do
      let logits ← predict (inputTensorFromIds padded)
      pure ((logits.unstack 0).unstack position))
    (allowToken := fun i => !options.asciiOnly || Text.isPrintableAscii i.val)

/--
Tokenize the corpus once and build a finite training set from approximately evenly spaced windows.
The sample stream retains the token tensor across batch requests.
-/
def samplesFromCorpus (corpus : String) (windows : Nat) :
    Data.SampleStream (Sample.Supervised Float input input) :=
  let toks := Text.Tokenizer.byte.encode corpus
  let tokens := Tensor.from toks
  let offs := Text.Corpus.evenlySpacedOffsets toks.size contextLength windows
  Data.SampleStream.mk windows (fun index =>
    let off := offs[index]
    let idsByBatch : Tensor Nat [batchSize, contextLength + 1] :=
      Tensor.stack 0 fun i =>
        let off' :=
          (off + i.val * (contextLength / 2 + 1)) %
            Text.Corpus.usableTokenStarts toks.size contextLength
        Tensor.window tokens (contextLength + 1) off' 32
    batchSampleFromTokenIds idsByBatch)

/--
Interactive prompt loop for the in-memory Float model.

Each line, followed by a newline byte, is appended to the current byte context and extended by the
trained model. The extended context carries over to the next prompt. `:clear` resets it, `:show`
prints it, and an empty line, `:q` or `:quit` ends the loop.
-/
partial def interactiveLoop
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
      IO.println s!"  context={Text.formatBytes ctx}"
      loop ctx
    else
      let encoded := Text.Tokenizer.byte.encode prompt
      let inputIds :=
        Tensor.concat (Tensor.concat ctx (Tensor.from encoded)) ([10] : Tensor Nat [1])
      let outIds ←
        generate predict inputIds options.generation
      let genOnly := Tensor.window outIds options.generation.continuationLength
        (count + encoded.size + 1) 0
      IO.println s!"  generated={Text.formatBytes genOnly}"
      loop outIds
  loop (Tensor.full [0] 0)

/--
Train the byte-level model and decode a prediction report.

Inputs and reports use host `Float`, while `RunConfig.arithmetic` selects the arithmetic used by
the trainer. The command fixes that runtime choice to native execution.
-/
def trainAndDecode (runtime : Session.Config) (corpus : String)
    (options : Options) :
    IO (Float × Float × String) := do
  let samples := samplesFromCorpus corpus options.windows
  let reportSample :=
    Data.CausalLM.byteBatch
      (α := Float) batchSize contextLength vocabularySize byteIndex options.generation.prompt
  let trainer := Trainer.create model <|
    { runtime with
        optimizer := optim.adam { lr := options.training.lr }
        objective := oneHotCrossEntropy 2 }
  trainer.printSummary

  /-
  The GPT-2 command trains on a bounded table of evenly spaced corpus windows. The fixed table makes
  the training schedule reproducible and lets the public trainer own checkpointing and optimizer
  state, so the example can focus on text windows, decoding and generation.
  -/
  let trained ← trainer.train
    (Data.fromStream samples)
    { steps := options.training.steps
      samplesPerStep := options.training.batchSize
      cudaMemorySampleEvery := options.training.cudaMemorySampleEvery
      loadCheckpoint? := options.checkpoint.loadCheckpoint?
      saveCheckpoint? := options.checkpoint.saveCheckpoint? }
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after

  let afterLogits ← trained.predict reportSample.input
  printPredictionReport "after " options.generation.prompt afterLogits
  let generatedIds ←
    generate trained.predict
      (Tensor.from (Text.Tokenizer.byte.encode options.generation.prompt)) options.generation
  let generated := Text.formatBytes generatedIds
  IO.println s!"  generated={generated}"
  IO.println s!"  corpus_bytes={corpus.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({options.generation.topK}), temperature={
    options.generation.temperature}, seed={options.generation.seed}"
  IO.println s!"  repetition_penalty={options.generation.repeatPenalty} repeat_window={
    options.generation.repeatWindow}"
  if options.interactive then
    interactiveLoop trained.predict options
  Text.Log.writeGeneration
    options.training.logDestination
      "GPT-2 byte prompt training" options.training.steps lossBefore lossAfter
    options.generation generated
    #[Training.deviceNote runtime,
      s!"windows={options.windows}",
      Training.cudaMemoryNote runtime options.training.steps
        options.training.cudaMemorySampleEvery]
  pure (lossBefore, lossAfter, generated)

/--
CLI entrypoint for byte-level GPT training, sampling, logging, and checkpointing. Without
`--steps`, a CUDA run takes one update and a CPU run takes none.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner exeName "causal LM training"
      usage? := some usage
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (corpus, rest) ← takeInputText rest
      let defaultSteps : Nat := if runtime.usesCUDA then 1 else 0
      let (train, rest) ← CLI.orThrow exeName <|
        Options.parse rest defaultSteps
      CLI.requireNoArgs exeName rest
      let _ ← trainAndDecode runtime corpus train)

end TorchLean.Examples.GPT2
