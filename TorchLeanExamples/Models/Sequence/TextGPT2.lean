/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

GPU-only corpus-training example:
  scripts/lake.sh -Kcuda=true build
  scripts/lake.sh -Kcuda=true exe torchlean text_gpt2 --device cuda \
    --data-file data/real/text/tinystories_valid.txt \
    --allow-small-data --steps 1 --generate 0

Prepare that file with:
  python3 scripts/datasets/download_example_data.py --tinystories-valid

GPT-2 BPE tokenizer run:
  scripts/lake.sh -Kcuda=true exe torchlean text_gpt2 --device cuda \
    --data-file data/real/text/tiny_shakespeare.txt \
    --bpe-vocab data/real/gpt2/vocab.json \
    --bpe-merges data/real/gpt2/merges.txt \
    --allow-small-data --max-chars 20000 --steps 10 \
    --prompt "First Citizen:" --generate 8

Local file run:
  scripts/lake.sh -Kcuda=true exe torchlean text_gpt2 --device cuda \
    --data-file /tmp/tiny.txt --allow-small-data --steps 1 --generate 0
-/

module

public import TorchLean.NN.Data.CausalLM
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Text.Vocabulary
public import TorchLean.NN.Text.GPT2BPE
public import TorchLean.NN.Text.Generation
public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train
public import TorchLeanExamples.Models.Training
public import TorchLean.NN.Models.CausalTransformer

/-!
# GPU GPT-2 Corpus Trainer

The `text_gpt2` command trains a small GPT-2-style model on a text corpus. TorchLean initializes
the weights at random and its runtime trains them, with no pretrained PyTorch or Hugging Face
checkpoint involved. Tokenization comes from `TorchLean.Text` and the causal Transformer from
`TorchLean.nn`. Training needs CUDA: the command supplies `--device cuda` when no device
is given and rejects any other device.

The byte path has one output class for each of the 256 byte values. Passing `--bpe-vocab` and
`--bpe-merges` loads the GPT-2 tokenizer files and projects the observed token ids into a local
vocabulary of at most 512 entries, so the output head has 512 classes where GPT-2 has 50,257.
Ids outside the local vocabulary map to entry zero. Both paths use four-token contexts, and the
generated text is a runtime demonstration.
-/

@[expose] public section

open TorchLean.Trainer.Objective TorchLean.Backend.Device

namespace TorchLean.Examples.TextGPT2

/-- Runner subcommand name of the corpus trainer. -/
def exeName : String := "text_gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "text_gpt2"

/-- Smallest corpus accepted without `--allow-small-data`: 100 MiB. -/
def minTrainingBytes : Nat :=
  100 * 1024 * 1024

/--
Context window of both corpus trainers.

It sits at the top of the file so that corpus validation and both model namespaces share one
value. Four positions are enough to exercise causal attention and keep the command a quick runtime
check.
-/
def contextLength : Nat := 4

local instance : NeZero contextLength := ⟨by decide⟩

/-! ## Corpus, tokenizer and prompt options -/

/-- Required text-corpus path plus the explicit small-data option used by local corpus trainers. -/
structure CorpusFileOptions where
  /-- UTF-8 or raw-byte corpus path selected by `--data-file`. -/
  dataFile : System.FilePath
  /-- Allow local runs below the normal corpus-size floor. -/
  allowSmallData : Bool
deriving Repr

namespace CorpusFileOptions

/-- Parse the required `--data-file` corpus flag and optional `--allow-small-data` switch. -/
def parse
    (exeName : String)
    (arguments : List String) :
    Except String (CorpusFileOptions × List String) := do
  let (dataFile, arguments) ←
    CLI.requirePathFlag arguments "data-file" (exeName := exeName)
  let (allowSmallData, arguments) ← CLI.takeBoolFlag arguments "allow-small-data"
  pure ({ dataFile := dataFile, allowSmallData := allowSmallData }, arguments)

end CorpusFileOptions

/-- Optional second corpus pass after the main training run. -/
structure FinetuneOptions where
  /-- Optional corpus used for a second fine-tuning pass. -/
  file? : Option System.FilePath
  /-- Number of optimizer steps used on that second corpus when present. -/
  steps : Nat
deriving Repr

namespace FinetuneOptions

/--
Parse the optional `--finetune-file` / `--finetune-steps` pair, rejecting `--finetune-steps` without
`--finetune-file`.

The caller supplies the default step count so commands can reuse their main training-step default.
-/
def parse
    (arguments : List String)
    (defaultSteps : Nat) :
    Except String (FinetuneOptions × List String) := do
  let (file?, arguments) ← CLI.takePathFlag? arguments "finetune-file"
  let (steps?, arguments) ← CLI.takeNatFlag? arguments "finetune-steps"
  if file?.isNone && steps?.isSome then
    throw "--finetune-steps requires --finetune-file"
  pure ({ file?, steps := steps?.getD defaultSteps }, arguments)

end FinetuneOptions

/-- Optional GPT-2 BPE tokenizer bundle plus an optional bounded-text cap. -/
structure BPECorpusOptions where
  /-- Optional GPT-2 `vocab.json` path. Must be paired with `mergesFile?`. -/
  vocabularyFile? : Option System.FilePath
  /-- Optional GPT-2 `merges.txt` path. Must be paired with `vocabularyFile?`. -/
  mergesFile? : Option System.FilePath
  /-- Optional text-character cap for bounded local BPE runs. -/
  maxCharacters? : Option Nat
deriving Repr

namespace BPECorpusOptions

/--
Parse the optional GPT-2 BPE tokenizer bundle.

`--bpe-vocab` and `--bpe-merges` must appear together. `--max-chars` is independent.
-/
def parse
    (arguments : List String) :
    Except String (BPECorpusOptions × List String) := do
  let ((vocabularyFile?, mergesFile?), arguments) ←
    CLI.takePairedPathFlags arguments "bpe-vocab" "bpe-merges"
  let (maximumCharacters?, arguments) ←
    CLI.takeNatFlag? arguments "max-chars"
  pure ({ vocabularyFile?, mergesFile?, maxCharacters? := maximumCharacters? }, arguments)

end BPECorpusOptions

/-- Command-local corpus, training, tokenization, generation, and prompt-loop controls. -/
structure Options where
  /-- Step, batching, logging, and allocator controls. -/
  training : CLI.Training.RunOptions
  /-- Prompt (`--prompt`) and continuation length (`--generate`), decoded greedily (`topK = 1`)
  without a repeat penalty. -/
  generation : Text.GenerationOptions
  /-- Whether to read prompts from the terminal after training (`--interactive`). -/
  interactive : Bool
  /-- Corpus path (`--data-file`) and the `--allow-small-data` override. -/
  corpus : CorpusFileOptions
  /-- Optional fine-tuning corpus and its step count. -/
  finetune : FinetuneOptions
  /-- Optional GPT-2 BPE files and the `--max-chars` cap. -/
  bpe : BPECorpusOptions
deriving Repr

namespace Options

/-- Help text for corpus training, optional GPT-2 tokenization, and generation. -/
def usage : String :=
  String.intercalate "\n" [
    "Usage: scripts/lake.sh exe torchlean text_gpt2 --data-file PATH [options]",
    "",
    "Corpus and tokenizer:",
    "  --data-file PATH       training corpus",
    "  --allow-small-data     permit corpora below the normal size floor",
    "  --bpe-vocab PATH       GPT-2 vocab.json (requires --bpe-merges)",
    "  --bpe-merges PATH      GPT-2 merges.txt (requires --bpe-vocab)",
    "  --max-chars N          cap corpus characters before BPE tokenization",
    "  --finetune-file PATH   optional second corpus pass",
    "  --finetune-steps N     optimizer updates on the second corpus (requires --finetune-file)",
    "",
    "Training and generation:",
    "  --steps N --batch-size N --log PATH|false --cuda-mem-watch N",
    "  --prompt TEXT --generate N --interactive",
    "",
    "Runtime:",
    "  --device cuda --seed N --show-backend"
  ]

/-- Parse the `text_gpt2` options, returning them with the unconsumed arguments. -/
def parse (args : List String) : Except String (Options × List String) := do
  let (corpus, args) ← CorpusFileOptions.parse exeName args
  let (training, args) ←
    CLI.Training.RunOptions.parse exeName args defaultLogPath (defaultSteps := 1)
  let (prompt, args) ← CLI.takeFlagValue args "prompt" (default := "First Citizen:")
  let (continuationLength, args) ← CLI.takeNatFlag args "generate" (default := 0)
  let (interactive, args) ← CLI.takeBoolFlag args "interactive"
  let (finetune, args) ← FinetuneOptions.parse args training.steps
  let (bpe, args) ← BPECorpusOptions.parse args
  pure ({ training
          generation :=
            { prompt, continuationLength, temperature := 1.0, topK := 1, repeatPenalty := 0.0,
              repeatWindow := 0, seed := 0, asciiOnly := false }
          interactive
          corpus
          finetune
          bpe }, args)

end Options

/--
Read the primary corpus as raw bytes. It must hold more than `contextLength` bytes, and at least
`minTrainingBytes` unless `--allow-small-data` is passed.
-/
def readCorpusBytes (corpusOptions : Options) : IO ByteArray :=
  Text.Corpus.readByteFile exeName corpusOptions.corpus.dataFile
    corpusOptions.corpus.allowSmallData minTrainingBytes contextLength

namespace Byte

/-- Byte vocabulary: one class for each of the 256 byte values. -/
def vocabularySize : Nat := 256

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- A byte id as a vocabulary index, reduced modulo 256 by `Fin.ofNat`. -/
def index (id : Nat) : Fin vocabularySize :=
  Fin.ofNat vocabularySize id

/-- Single-sequence batch for the byte-level corpus path. -/
def batchSize : Nat := 1

instance : NeZero batchSize := ⟨by decide⟩

/-- Number of attention heads in the compact byte-level Transformer. -/
def attentionHeads : Nat := 2

/-- Per-head width. -/
def attentionHeadWidth : Nat := 2

/-- Transformer embedding width. -/
def modelWidth : Nat := attentionHeads * attentionHeadWidth

/-- Feed-forward hidden width. -/
def feedForwardWidth : Nat := 8

/-- Number of Transformer blocks. -/
def transformerLayers : Nat := 1

local instance : NeZero modelWidth := ⟨by decide⟩

/-- Causal Transformer configuration of the byte model, built from the constants above. -/
abbrev modelConfig : nn.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := vocabularySize
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := transformerLayers }

/-- Shape `[1, 4, 256]` of one sequence of four one-hot bytes, and of the model's byte logits, one
row per position. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/--
Byte-level causal Transformer for corpus pretraining and fine-tuning.

It has one block with two heads of width 2 and a feed-forward width of 8, which is small enough
for the eager CUDA path and big enough to exercise multi-head causal attention.
-/
def model : nn.Builder (nn.Sequential input input) :=
  nn.CausalTransformer.oneHot modelConfig (batchShape := [batchSize])

end Byte

/--
Training sample `i` of the byte model: the `contextLength + 1` bytes at offset `i` (modulo the
number of window starts), one-hot encoded as inputs and next-byte targets.
-/
def byteCorpusSampleAt (bytes : ByteArray) (i : Nat) :
    Sample.Supervised Float Byte.input Byte.input :=
  let toks := (Text.byteTokenWindow bytes (contextLength + 1)
    (offset := Text.Corpus.byteOffset bytes i contextLength)
    ).map Byte.index
  Data.CausalLM.oneHotSample
    (α := Float) [Byte.batchSize] contextLength Byte.vocabularySize
      (Tensor.repeatAxis 0 Byte.batchSize toks)

/-- Extend `options.prompt` by `options.continuationLength` bytes chosen by `Text.generate` under
`options`, and decode the whole sequence. -/
def generateByte
    (predict : Tensor Float Byte.input → IO (Tensor Float Byte.input))
    (options : Text.GenerationOptions) : IO String := do
  let ids ←
    Text.generate contextLength 0
      (Tensor.from (Text.Tokenizer.byte.encode options.prompt)) options
      (fun padded predPos => do
        let x := Tensor.repeatAxis 0 Byte.batchSize <|
          Tensor.oneHotIndices (α := Float) Byte.vocabularySize
            (padded.map Byte.index)
        let logits ← predict x
        pure ((logits.unstack 0).unstack predPos))
  pure (Text.Tokenizer.byte.decode (ids.to (Array Nat)))

/-- Read prompts until an empty line, `:q` or `:quit`, printing each generated response. -/
partial def interactive
    (generate : String → IO String) (window : String) (newTokenCount : Nat) : IO Unit := do
  IO.println ("  interactive: enter a prompt; empty line or :q exits "
    ++ s!"(window={window}, generate={newTokenCount})")
  let stdin ← IO.getStdin
  let rec loop : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else
      let response ← generate prompt
      IO.println s!"  response={Text.escape response}"
      loop
  loop

namespace BPE

/--
Local vocabulary size of the BPE path.

The tokenizer reads GPT-2's 50,257-token BPE files, and the example maps the observed token ids, in
order of first occurrence, into a local vocabulary of 512 entries whose entry zero is reserved for
unknown tokens. A full 50,257-way output head would make a much larger training run, and this
example is about the tokenizer and data path.
-/
def vocabularySize : Nat := 512

instance : NeZero vocabularySize := ⟨by decide⟩

/-- Batch size for the BPE corpus path. -/
def batchSize : Nat := 2

instance : NeZero batchSize := ⟨by decide⟩

/-- Number of attention heads in the miniature BPE Transformer. -/
def attentionHeads : Nat := 1

/-- Per-head width for the BPE Transformer. -/
def attentionHeadWidth : Nat := 8

/-- Transformer embedding width. -/
def modelWidth : Nat := attentionHeads * attentionHeadWidth

/-- Feed-forward hidden width. -/
def feedForwardWidth : Nat := 32

/-- Number of Transformer blocks. -/
def transformerLayers : Nat := 1

local instance : NeZero modelWidth := ⟨by decide⟩

/-- Causal Transformer configuration of the BPE model, built from the constants above. -/
abbrev modelConfig : nn.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := vocabularySize
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := transformerLayers }

/-- Shape `[2, 4, 512]` of two sequences of four one-hot local BPE ids, and of the model's
local-BPE logits, one row per position. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/--
Causal Transformer for the GPT-2 BPE path, with outputs over the 512-entry local vocabulary.

It has one block with one head of width 8 and a feed-forward width of 32, so its architecture and
scale differ from OpenAI's GPT-2 small.
-/
def model : nn.Builder (nn.Sequential input input) :=
  nn.CausalTransformer.oneHot modelConfig (batchShape := [batchSize])

end BPE

/-! ## Example-local compact vocabulary -/

/--
Training sample `i` of the BPE model: `batchSize` corpus windows at deterministic pseudo-random
offsets (seed 0, step `i`), one-hot encoded as inputs and next-token targets.
-/
def bpeCorpusSampleAt {tokenCount : Nat}
    (tokens : Tensor (Fin BPE.vocabularySize) [tokenCount]) (i : Nat) :
    Sample.Supervised Float BPE.input BPE.input :=
  Data.CausalLM.oneHotSample (α := Float) [BPE.batchSize]
    contextLength BPE.vocabularySize <|
      Text.Corpus.randomTokenBatch tokens BPE.batchSize contextLength 0 i 0

/--
Encode a prompt as one BPE sample: its last `contextLength + 1` local ids, right-padded with zero
when the prompt is shorter, repeated across the batch.
-/
def bpePromptSample
    (tok : Text.GPT2BPE.Tokenizer) (lv : Text.VocabularyProjection BPE.vocabularySize)
    (prompt : String) :
    Except String (Sample.Supervised Float BPE.input BPE.input) := do
  let encoded ← Text.GPT2BPE.encode tok prompt
  let ids := lv.encode (Tensor.from encoded)
  let start :=
    if encoded.size > contextLength + 1 then
      encoded.size - (contextLength + 1)
    else
      0
  let window : Tensor Nat [contextLength + 1] :=
    Tensor.window ids (contextLength + 1) start 0
  let bounded ← Tensor.checkIndices BPE.vocabularySize window
  pure <| Data.CausalLM.oneHotSample (α := Float)
    [BPE.batchSize] contextLength BPE.vocabularySize
      (Tensor.repeatAxis 0 BPE.batchSize bounded)

/-- Map local ids back to GPT-2 token ids and decode them to text. -/
def decodeLocalBPE {count : Nat} (tok : Text.GPT2BPE.Tokenizer)
    (lv : Text.VocabularyProjection BPE.vocabularySize) (ids : Tensor Nat [count]) :
    Except String String :=
  Text.GPT2BPE.decode tok ((lv.decode ids).to (Array Nat))

/-- Whether output slot `token` is assigned in this run's local vocabulary (below `lv.size`). -/
def isLocalBPEId (lv : Text.VocabularyProjection BPE.vocabularySize)
    (token : Fin BPE.vocabularySize) : Bool :=
  decide (token.val < lv.size)

/--
Greedy local ids at each position of batch row 0, skipping unassigned vocabulary slots. Fails when
a position has no assigned slot with a non-NaN score.
-/
def argmaxLocalBPETokens
    (lv : Text.VocabularyProjection BPE.vocabularySize)
    (logits : Tensor Float BPE.input) :
    Except String (Tensor Nat [contextLength]) :=
  Tensor.generateFlatM [contextLength] fun index =>
    let position : Fin contextLength := ⟨index.val, by simpa using index.isLt⟩
    match Text.greedy? ((logits.unstack 0).unstack position) (isLocalBPEId lv) with
    | some token => pure token.val
    | none => throw "no assigned compact-vocabulary token has a non-NaN score"

/-- Print the decoded argmax prediction of the BPE model for `prompt`, tagged with `label`. -/
def printBPEPredictionProbe
    (tok : Text.GPT2BPE.Tokenizer)
    (lv : Text.VocabularyProjection BPE.vocabularySize)
    (predict : Tensor Float BPE.input → IO (Tensor Float BPE.input))
    (label prompt : String) : IO Unit := do
  let sample ← CLI.orThrow exeName <| bpePromptSample tok lv prompt
  let logits ← predict sample.input
  let ids ← CLI.orThrow exeName <| argmaxLocalBPETokens lv logits
  let decoded ← CLI.orThrow exeName <| decodeLocalBPE tok lv ids
  IO.println s!"  {label} pred={Text.escape decoded}"
  IO.println s!"  prompt={Text.escape prompt}"

/--
BPE generation under `options` using up to the last `contextLength` tokens. The next-token logits
come from the current context position, before any right padding, and unassigned local vocabulary
slots are excluded from selection.
-/
def generateBPE
    (tok : Text.GPT2BPE.Tokenizer)
    (lv : Text.VocabularyProjection BPE.vocabularySize)
    (predict : Tensor Float BPE.input → IO (Tensor Float BPE.input))
    (options : Text.GenerationOptions) : IO String := do
  let initOrigIds ← CLI.orThrow exeName <| Text.GPT2BPE.encode tok options.prompt
  let initIds := lv.encode (Tensor.from initOrigIds)
  let ids ←
    Text.generate contextLength 0 initIds options
      (fun padded predPos => do
        let bounded ← CLI.orThrow exeName <|
          Tensor.checkIndices BPE.vocabularySize padded
        let x := Tensor.repeatAxis 0 BPE.batchSize <|
          Tensor.oneHotIndices (α := Float) BPE.vocabularySize bounded
        let logits ← predict x
        pure ((logits.unstack 0).unstack predPos))
      (allowToken := isLocalBPEId lv)
  CLI.orThrow exeName <| decodeLocalBPE tok lv ids

/--
Train the byte-level model on a corpus with CUDA.

The sample stream is lazy and holds only the scheduled windows: `--steps` times `--batch-size`
from the corpus, then `--finetune-steps` times `--batch-size` from the optional fine-tuning file.
The model is small by GPT-2 standards, but the data path is the real one: file bytes → token
windows → one-hot tensors → TorchLean CUDA training.
-/
def trainCorpus (runtime : Session.Config)
    (trainOpts : Options)
    (bytes : ByteArray) : IO Unit := do
  let first := Text.byteTokenWindow bytes (contextLength + 1)
  let firstIds := Tensor.to first (Array Nat)
  IO.println s!"  mode=byte bytes={bytes.size} steps={trainOpts.training.steps} window={
    contextLength}"
  IO.println s!"  first prompt={Text.escape (Text.Tokenizer.byte.decode
    (firstIds.take contextLength))}"
  IO.println
    s!"  first target={Text.escape (Text.Tokenizer.byte.decode (firstIds.drop 1))}"
  let ftBytes? ←
    match trainOpts.finetune.file? with
    | none => pure none
    | some path => do
        let ftBytes ←
          Text.Corpus.readByteFile exeName path trainOpts.corpus.allowSmallData minTrainingBytes
            contextLength
        pure (some ftBytes)
  let pretrainSamples := Data.SampleStream.mk
    (trainOpts.training.steps * trainOpts.training.batchSize) fun index =>
      byteCorpusSampleAt bytes index.val
  let finetuneSamples := match ftBytes? with
    | none => Data.SampleStream.mk 0 (fun index => nomatch index)
    | some ftBytes => Data.SampleStream.mk
        (trainOpts.finetune.steps * trainOpts.training.batchSize) fun index =>
          byteCorpusSampleAt ftBytes index.val
  let allSamples := pretrainSamples.append finetuneSamples
  let totalSteps := trainOpts.training.steps +
    match ftBytes? with
    | none => 0
    | some _ => trainOpts.finetune.steps
  let trainer := Trainer.create Byte.model <|
    { runtime with
        optimizer := optim.adam { lr := 1e-3 }
        objective := oneHotCrossEntropy 2 }
  trainer.printSummary
  /-
  The trainer reads the stream in order, `--batch-size` samples per optimizer step, so the first
  `--steps` updates see the main corpus and the later `--finetune-steps` updates see the
  fine-tuning file. Gradient accumulation and optimizer state belong to the trainer.
  -/
  let trained ← trainer.train
    (Data.fromStream allSamples)
    { steps := totalSteps
      samplesPerStep := trainOpts.training.batchSize
      logEvery := Nat.max 1 (totalSteps / 10)
      cudaMemorySampleEvery := trainOpts.training.cudaMemorySampleEvery }
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after
  let generated ← generateByte trained.predict trainOpts.generation
  IO.println s!"  greedy generated={Text.escape generated}"
  Text.Log.writeGeneration
    trainOpts.training.logDestination
      "GPT-2 byte corpus training" totalSteps lossBefore lossAfter
    trainOpts.generation (some generated)
    #[s!"data={trainOpts.corpus.dataFile}", Training.deviceNote runtime,
      s!"bytes={bytes.size}"]
  if trainOpts.interactive then
    interactive
      (fun prompt => generateByte trained.predict { trainOpts.generation with prompt })
      s!"{contextLength} bytes" trainOpts.generation.continuationLength

/-- Validate, optionally cap, and tokenize one UTF-8 corpus file with GPT-2 BPE. -/
def loadBPEFileTokens
    (trainOpts : Options)
    (tok : Text.GPT2BPE.Tokenizer)
    (path : System.FilePath) :
    IO (Array Nat) := do
  let fullCorpusText ← IO.FS.readFile path
  let byteCount := fullCorpusText.toUTF8.size
  if byteCount <= contextLength then
    throw <| IO.userError s!"{exeName}: corpus {path} has {byteCount} bytes; need more than {
      contextLength}"
  if !trainOpts.corpus.allowSmallData && byteCount < minTrainingBytes then
    throw <| IO.userError s!"{exeName}: corpus {path} has {byteCount} bytes; expected at least {
      minTrainingBytes} (pass --allow-small-data for an intentional small run)"
  let corpusText :=
    match trainOpts.bpe.maxCharacters? with
    | some n => (fullCorpusText.take n).toString
    | none => fullCorpusText
  let ids ← CLI.orThrow exeName <| Text.GPT2BPE.encode tok corpusText
  if ids.size <= contextLength then
    throw <| IO.userError s!"{exeName}: BPE corpus {path} has {ids.size} tokens after capping; \
need more than {contextLength}"
  pure ids

/-- Print the first BPE training window for inspecting tokenization and windowing. -/
def printBPECorpusPreview {tokenCount : Nat} (tok : Text.GPT2BPE.Tokenizer)
    (lv : Text.VocabularyProjection BPE.vocabularySize)
    (tokens : Tensor Nat [tokenCount]) : IO Unit := do
  let first := Tensor.window tokens (contextLength + 1) 0 0
  let firstIds := Tensor.to first (Array Nat)
  let prompt ← CLI.orThrow exeName <|
    decodeLocalBPE tok lv (Tensor.window first contextLength 0 0)
  let target ← CLI.orThrow exeName <|
    decodeLocalBPE tok lv (Tensor.window first contextLength 1 0)
  IO.println s!"  first local BPE ids={firstIds.toList}"
  IO.println s!"  first prompt={Text.escape prompt}"
  IO.println s!"  first target={Text.escape target}"

/--
Train the BPE model on GPT-2 tokens projected to the local vocabulary.

The run exercises the GPT-2 tokenizer and vocabulary path and can overfit local windows. The
weights start from a random TorchLean initialization, with no pretrained GPT-2 checkpoint.
-/
def trainBPECorpus {tokenCount : Nat} (runtime : Session.Config)
    (trainOpts : Options)
    (tok : Text.GPT2BPE.Tokenizer)
    (lv : Text.VocabularyProjection BPE.vocabularySize)
    (tokens : Tensor Nat [tokenCount])
    (finetuneTokens? : Option ((count : Nat) × Tensor Nat [count])) : IO Unit := do
  let totalSteps := trainOpts.training.steps +
    match finetuneTokens? with
    | none => 0
    | some _ => trainOpts.finetune.steps
  IO.println s!"  mode=bpe local-vocab={lv.size}/{BPE.vocabularySize} tokens={
    tokenCount} steps={totalSteps}"
  printBPECorpusPreview tok lv tokens
  let bounded ← CLI.orThrow exeName <| Tensor.checkIndices BPE.vocabularySize tokens
  let pretrainSamples := Data.SampleStream.mk
    (trainOpts.training.steps * trainOpts.training.batchSize) fun index =>
      bpeCorpusSampleAt bounded index.val
  let finetuneSamples ←
    match finetuneTokens? with
    | none => pure (Data.SampleStream.mk 0 fun index => nomatch index)
    | some ⟨_, finetuneTokens⟩ => do
        let bounded ← CLI.orThrow exeName <|
          Tensor.checkIndices BPE.vocabularySize finetuneTokens
        pure <| Data.SampleStream.mk
          (trainOpts.finetune.steps * trainOpts.training.batchSize) fun index =>
            bpeCorpusSampleAt bounded index.val
  let scheduledSamples := pretrainSamples.append finetuneSamples
  let trainer := Trainer.create BPE.model <|
    { runtime with
        optimizer := optim.adam { lr := 1e-3 }
        objective := oneHotCrossEntropy 2 }
  trainer.printSummary
  /-
  The trainer sees only one-hot windows over the local vocabulary. Mapping ids back to GPT-2 tokens
  for printing happens in this file, because it is presentation and plays no part in training.
  -/
  let trained ← trainer.train
    (Data.fromStream scheduledSamples)
    { steps := totalSteps
      samplesPerStep := trainOpts.training.batchSize
      logEvery := Nat.max 1 (totalSteps / 10)
      cudaMemorySampleEvery := trainOpts.training.cudaMemorySampleEvery }
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after
  printBPEPredictionProbe tok lv trained.predict "after " trainOpts.generation.prompt
  let generated ← generateBPE tok lv trained.predict trainOpts.generation
  IO.println s!"  greedy generated={Text.escape generated}"
  Text.Log.writeGeneration
    trainOpts.training.logDestination
      "GPT-2 BPE corpus training" totalSteps lossBefore lossAfter
    trainOpts.generation (some generated)
    #[s!"data={trainOpts.corpus.dataFile}", Training.deviceNote runtime,
      s!"localVocab={lv.size}/{BPE.vocabularySize}", s!"tokens={tokenCount}"]
  if trainOpts.interactive then
    interactive
      (fun prompt => generateBPE tok lv trained.predict { trainOpts.generation with prompt })
      s!"{contextLength} tokens" trainOpts.generation.continuationLength

/--
CLI entry point. It trains on the BPE path when `--bpe-vocab` and `--bpe-merges` are both given
and on the byte path otherwise.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner exeName "GPU corpus trainer"
      usage? := some Options.usage
      printSuccess := true
      runtime := { device? := some cuda } })
    exeName args
    (fun runtime rest => do
      let (trainOpts, rest) ← CLI.orThrow exeName <|
        Options.parse rest
      CLI.requireNoArgs exeName rest
      match trainOpts.bpe.vocabularyFile?, trainOpts.bpe.mergesFile? with
      | some vocabPath, some mergesPath =>
          let tok ← Text.GPT2BPE.load vocabPath mergesPath
            (progress := true) (label := exeName)
          let capMsg :=
            match trainOpts.bpe.maxCharacters? with
            | some n => s!" (max chars={n})"
            | none => ""
          IO.eprintln s!"{exeName}: encoding BPE corpus{capMsg}"
          let tokens ← loadBPEFileTokens trainOpts tok trainOpts.corpus.dataFile
          IO.eprintln s!"{exeName}: encoded BPE corpus original-tokens={tokens.size}"
          let finetuneTokens? ←
            match trainOpts.finetune.file? with
            | none => pure none
            | some path =>
                IO.eprintln s!"{exeName}: encoding BPE fine-tuning corpus {path}"
                some <$> loadBPEFileTokens trainOpts tok path
          let promptTokens ← CLI.orThrow exeName <|
            Text.GPT2BPE.encode tok trainOpts.generation.prompt
          let projection := Text.VocabularyProjection.fromTokens BPE.vocabularySize 50256
            (Tensor.from tokens)
          let projection := match finetuneTokens? with
            | none => projection
            | some ids => projection.extend (Tensor.from ids)
          let lv := projection.extend (Tensor.from promptTokens)
          let localTokens := lv.encode (Tensor.from tokens)
          let localFinetuneTokens? := finetuneTokens?.map fun ids =>
            (⟨ids.size, lv.encode (Tensor.from ids)⟩ : (count : Nat) × Tensor Nat [count])
          IO.eprintln s!"{exeName}: projected BPE ids to local vocabulary {lv.size}/{
            BPE.vocabularySize}"
          trainBPECorpus runtime trainOpts tok lv localTokens localFinetuneTokens?
      | _, _ =>
          let bytes ← readCorpusBytes trainOpts
          trainCorpus runtime trainOpts bytes)

end TorchLean.Examples.TextGPT2
