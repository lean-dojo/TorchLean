/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

GPU-only corpus-training example:
  lake -R -K cuda=true build
  lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
    --data-file data/real/text/tinystories_valid.txt \
    --allow-small-data --steps 1 --generate 0

Prepare that file with:
  python3 scripts/datasets/download_example_data.py --tinystories-valid

GPT-2 BPE tokenizer run:
  lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
    --data-file data/real/text/tiny_shakespeare.txt \
    --bpe-vocab data/real/gpt2/vocab.json \
    --bpe-merges data/real/gpt2/merges.txt \
    --allow-small-data --max-chars 20000 --steps 10 \
    --prompt "First Citizen:" --generate 8

Local file run:
  lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
    --data-file /tmp/tiny.txt --allow-small-data --steps 1 --generate 0
-/

module

public import NN.API.Text.Vocabulary
public import NN.API
public import NN.Examples.Support

/-!
# GPU GPT-2 Corpus Trainer

This command trains GPT-2-style models from text in TorchLean.

The model is initialized inside TorchLean and trained by the TorchLean runtime. It does not load a
pretrained PyTorch/Hugging Face checkpoint:

* reusable tokenization lives under `TorchLean.text`,
* the compact GPT-2-style architecture lives under `TorchLean.nn.models`,
* the runnable corpus trainer enforces CUDA by default.

The byte path uses one output class for each of the 256 UTF-8 byte values.
Passing `--bpe-vocab` and `--bpe-merges` loads the GPT-2 tokenizer files, then projects
observed token IDs into a local vocabulary of at most 512 entries. IDs outside that local
vocabulary map to entry zero. The model therefore does not have a 50,257-way GPT-2 output
head. Both paths train randomly initialized models with four-token contexts; generated text is
a runtime demonstration.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.TextGpt2

/-- Runner subcommand name. This subcommand trains a randomly initialized GPT-2-style model. -/
def exeName : String := "text_gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "text_gpt2"

/-- Minimum corpus size for the default public training path: 100 MiB. -/
def minTrainingBytes : Nat :=
  100 * 1024 * 1024

/--
Default context window for both corpus trainers.

Keeping this near the file top lets corpus validation and the model architecture agree without
depending on declaration order. Four positions are enough to exercise causal attention while
keeping this command a compact runtime check.
-/
def contextLength : Nat := 4

/-- Command-local corpus, training, tokenization, generation, and prompt-loop controls. -/
structure Options where
  /-- Step, batching, logging, and allocator controls. -/
  training : CLI.Training.RunOptions
  /-- Prompt and continuation-length settings. -/
  generation : text.PromptGenerationOptions
  /-- Terminal prompt-loop policy. -/
  interaction : text.InteractiveOptions
  /-- Primary corpus and explicit small-data override. -/
  corpus : text.CorpusFileOptions
  /-- Optional second corpus pass. -/
  finetune : text.FinetuneOptions
  /-- Optional GPT-2 BPE tokenizer bundle. -/
  bpe : text.BpeCorpusOptions
deriving Repr

namespace Options

/-- Help text for corpus training, optional GPT-2 tokenization, and generation. -/
def usage : String :=
  String.intercalate "\n" [
    "Usage: lake exe torchlean text_gpt2 --data-file PATH [options]",
    "",
    "Corpus and tokenizer:",
    "  --data-file PATH       training corpus",
    "  --allow-small-data     permit corpora below the normal size floor",
    "  --bpe-vocab PATH       GPT-2 vocab.json (requires --bpe-merges)",
    "  --bpe-merges PATH      GPT-2 merges.txt (requires --bpe-vocab)",
    "  --max-chars N          cap corpus characters before BPE tokenization",
    "  --finetune-file PATH   optional second corpus pass",
    "  --finetune-steps N     optimizer updates on the second corpus",
    "",
    "Training and generation:",
    "  --steps N --batch-size N --log PATH|false --cuda-mem-watch N",
    "  --prompt TEXT --generate N --interactive",
    "",
    "Runtime:",
    "  --device cuda --execution eager|typed-graph --seed N --show-backend"
  ]

/-- Parse the complete option surface owned by the `text_gpt2` executable. -/
def parse (args : List String) : Except String (Options × List String) := do
  let (corpus, args) ← text.CorpusFileOptions.parse exeName args
  let (training, args) ←
    CLI.Training.RunOptions.parse exeName args defaultLogPath (defaultSteps := 1)
  let (prompt, args) ← text.PromptGenerationOptions.parse args
    { prompt := "First Citizen:", newTokenCount := 0 }
  let (interactive, args) ← text.InteractiveOptions.parse args
  let (finetune, args) ← text.FinetuneOptions.parse args training.steps
  let (bpe, args) ← text.BpeCorpusOptions.parse args
  pure ({ training
          generation := prompt
          interaction := interactive
          corpus
          finetune
          bpe }, args)

end Options

/-- Read the primary raw text corpus. -/
def readCorpusBytes (corpusOptions : Options) : IO ByteArray :=
  text.Corpus.readByteFile exeName corpusOptions.corpus.dataFile
    corpusOptions.corpus.allowSmallData minTrainingBytes contextLength

namespace ByteModel

/-- Compact byte-level vocabulary for the default corpus path. -/
def vocabularySize : Nat := 256

local instance : NeZero vocabularySize := ⟨by decide⟩

/-- Embed a byte id in the complete 256-entry byte vocabulary. -/
def byteIndex (id : Nat) : Fin vocabularySize :=
  Fin.ofNat vocabularySize id

/-- Single-sequence batch for the byte-level corpus path. -/
def batchSize : Nat := 1

instance : NeZero batchSize := ⟨by decide⟩

/--
Context window shared by corpus validation, byte training, BPE training, and generation.
-/
def contextLength : Nat := TextGpt2.contextLength

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

local instance : NeZero contextLength := ⟨by decide⟩
local instance : NeZero modelWidth := ⟨by decide⟩

/-- Byte-level GPT configuration shared by shapes and the model constructor. -/
abbrev modelConfig : nn.models.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := vocabularySize
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := transformerLayers }

/-- Input shape: byte-level one-hot token sequence. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/-- Output shape: one byte-logit row per input position. -/
abbrev output : Shape :=
  input

/--
Runnable byte-level GPT-style model for corpus pretraining/fine-tuning.

The model is compact enough for the eager CUDA path while exercising nontrivial causal attention,
feed-forward layers, byte tokenization, and the interactive prompt loop.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.CausalTransformer.oneHot modelConfig (batchShape := [batchSize])

end ByteModel

/-- Build one byte-level training sample from a corpus byte offset. -/
def byteCorpusSampleAt (bytes : ByteArray) (i : Nat) :
    Sample.Supervised Float ByteModel.input ByteModel.output :=
  let toks := (text.byteTokenWindow bytes (ByteModel.contextLength + 1)
    (offset := text.Corpus.byteOffset bytes i ByteModel.contextLength)
    ).map ByteModel.byteIndex
  Data.CausalLM.oneHotSample
    (α := Float) [ByteModel.batchSize] ByteModel.contextLength ByteModel.vocabularySize
      (Tensor.repeatAxis 0 ByteModel.batchSize toks)

/-- Greedy byte-level generation from the trained model. -/
def generateByteGreedy
    (predict : Tensor Float ByteModel.input → IO (Tensor Float ByteModel.output))
    (prompt : String) (steps : Nat) : IO String := do
  let gen : text.GenerationOptions :=
    { prompt := prompt
      newTokenCount := steps
      temperature := 1.0
      topK := 1
      repeatPenalty := 0.0
      repeatWindow := 0
      seed := 0
      asciiOnly := false }
  let ids ←
    text.autoregressiveTokenIds ByteModel.contextLength 0
      (Tensor.from (text.Tokenizer.byte.encode prompt)) gen
      (fun padded predPos => do
        let x := Tensor.repeatAxis 0 ByteModel.batchSize <|
          Data.CausalLM.oneHotInputs (α := Float) ByteModel.vocabularySize
            (padded.map ByteModel.byteIndex)
        let logits ← predict x
        pure (text.batchLogitScoresAt logits 0 predPos))
  pure (text.Tokenizer.byte.decode (ids.to (Array Nat)))

/-- Terminal prompt loop for the trained byte-level model. -/
partial def interactiveByteLoop
    (predict : Tensor Float ByteModel.input → IO (Tensor Float ByteModel.output))
    (newTokenCount : Nat) : IO Unit := do
  IO.println ("  interactive: enter a prompt; empty line or :q exits "
    ++ s!"(window={ByteModel.contextLength} bytes, generate={newTokenCount})")
  let stdin ← IO.getStdin
  let rec loop : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else
      let out ← generateByteGreedy predict prompt newTokenCount
      IO.println s!"  response={text.escape out}"
      loop
  loop

namespace BpeModel

/--
Compact vocabulary used by the runnable BPE training path.

The tokenizer still uses GPT-2's real 50,257-token BPE files. For this Lean/CUDA model
we project the corpus tokens into a local vocabulary of the first observed BPE ids. A full 50k-way
output head is a much larger training run; this example focuses on the tokenizer/data path.
-/
def vocabularySize : Nat := 512

instance : NeZero vocabularySize := ⟨by decide⟩

/-- Batch size for the BPE corpus path. -/
def batchSize : Nat := 2

instance : NeZero batchSize := ⟨by decide⟩

/-- Short context window used by the trainer. -/
def contextLength : Nat := TextGpt2.contextLength

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

local instance : NeZero contextLength := ⟨by decide⟩
local instance : NeZero modelWidth := ⟨by decide⟩

/-- BPE GPT configuration shared by shapes and the model constructor. -/
abbrev modelConfig : nn.models.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := vocabularySize
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := transformerLayers }

/-- Input shape: local-BPE one-hot token batch. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/-- Output shape: one local-BPE logit row per input position. -/
abbrev output : Shape :=
  input

/--
Compact GPT-2-style model with the real GPT-2 BPE tokenizer path.

This TorchLean-native Transformer reads GPT-2 BPE tokenizer files and uses a local output
projection over the corpus ids it observes. Its architecture and scale differ from OpenAI
GPT-2-small.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.CausalTransformer.oneHot modelConfig (batchShape := [batchSize])

end BpeModel

/-! ## Example-local compact vocabulary -/

/-- Build one BPE training sample from a tokenized corpus. -/
def bpeCorpusSampleAt {tokenCount : Nat}
    (tokens : Tensor (Fin BpeModel.vocabularySize) [tokenCount]) (i : Nat) :
    Sample.Supervised Float BpeModel.input BpeModel.output :=
  Data.CausalLM.oneHotSample (α := Float) [BpeModel.batchSize]
    BpeModel.contextLength BpeModel.vocabularySize <|
      text.Corpus.randomTokenBatch tokens BpeModel.batchSize BpeModel.contextLength 0 i 0

/-- Turn a BPE prompt into one model input window. -/
def bpePromptSample
    (tok : text.GPT2BPE.Tokenizer) (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (prompt : String) :
    Except String (Sample.Supervised Float BpeModel.input BpeModel.output) := do
  let encoded ← text.GPT2BPE.encode tok prompt
  let ids := lv.encode (Tensor.from encoded)
  let start :=
    if encoded.size > BpeModel.contextLength + 1 then
      encoded.size - (BpeModel.contextLength + 1)
    else
      0
  let window : Tensor Nat [BpeModel.contextLength + 1] :=
    Tensor.window ids (BpeModel.contextLength + 1) start 0
  let bounded ← Tensor.checkIndices BpeModel.vocabularySize window
  pure <| Data.CausalLM.oneHotSample (α := Float)
    [BpeModel.batchSize] BpeModel.contextLength BpeModel.vocabularySize
      (Tensor.repeatAxis 0 BpeModel.batchSize bounded)

/-- Decode projected BPE ids at the tokenizer's serialization boundary. -/
def decodeLocalBPE {count : Nat} (tok : text.GPT2BPE.Tokenizer)
    (lv : text.VocabularyProjection BpeModel.vocabularySize) (ids : Tensor Nat [count]) :
    Except String String :=
  text.GPT2BPE.decode tok ((lv.decode ids).to (Array Nat))

/-- Whether a model output id belongs to the compact vocabulary built for this run. -/
def isLocalBPEId (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (token : Fin BpeModel.vocabularySize) : Bool :=
  decide (token.val < lv.size)

/-- Decode one batch row while excluding unassigned compact-vocabulary output slots. -/
def argmaxLocalBPETokens
    (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (logits : Tensor Float BpeModel.output) :
    Except String (Tensor Nat [BpeModel.contextLength]) :=
  Tensor.generateFlatM [BpeModel.contextLength] fun index =>
    let position : Fin BpeModel.contextLength := ⟨index.val, by simpa [Shape.size] using index.isLt⟩
    match text.greedyToken? (text.batchLogitScoresAt logits 0 position) (isLocalBPEId lv) with
    | some token => pure token.val
    | none => throw "no assigned compact-vocabulary token has a non-NaN score"

/-- Print an argmax prediction report for a prompt under the BPE model. -/
def printBpePredictionProbe
    (tok : text.GPT2BPE.Tokenizer)
    (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (predict : Tensor Float BpeModel.input → IO (Tensor Float BpeModel.output))
    (label prompt : String) : IO Unit := do
  let sample ← CLI.orThrow exeName <| bpePromptSample tok lv prompt
  let logits ← predict sample.input
  let ids ← CLI.orThrow exeName <| argmaxLocalBPETokens lv logits
  let decoded ← CLI.orThrow exeName <| decodeLocalBPE tok lv ids
  IO.println s!"  {label} pred={text.escape decoded}"
  IO.println s!"  prompt={text.escape prompt}"

/--
Greedy BPE generation by repeatedly feeding the last `contextLength` tokens and appending the
final-position argmax. This is a deterministic sampling path for inspecting the trained next-token
model.
-/
def generateBpeGreedy
    (tok : text.GPT2BPE.Tokenizer)
    (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (predict : Tensor Float BpeModel.input → IO (Tensor Float BpeModel.output))
    (prompt : String) (steps : Nat) : IO String := do
  let initOrigIds ← CLI.orThrow exeName <| text.GPT2BPE.encode tok prompt
  let initIds := lv.encode (Tensor.from initOrigIds)
  let gen : text.GenerationOptions :=
    { prompt := prompt
      newTokenCount := steps
      temperature := 1.0
      topK := 1
      repeatPenalty := 0.0
      repeatWindow := 0
      seed := 0
      asciiOnly := false }
  let ids ←
    text.autoregressiveTokenIds BpeModel.contextLength 0 initIds gen
      (fun padded predPos => do
        let bounded ← CLI.orThrow exeName <|
          Tensor.checkIndices BpeModel.vocabularySize padded
        let x := Tensor.repeatAxis 0 BpeModel.batchSize <|
          Data.CausalLM.oneHotInputs (α := Float) BpeModel.vocabularySize bounded
        let logits ← predict x
        pure (text.batchLogitScoresAt logits 0 predPos))
      (allowToken := isLocalBPEId lv)
  CLI.orThrow exeName <| decodeLocalBPE tok lv ids

/-- Terminal prompt loop for the trained BPE model. -/
partial def interactiveBpeLoop
    (tok : text.GPT2BPE.Tokenizer)
    (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (predict : Tensor Float BpeModel.input → IO (Tensor Float BpeModel.output))
    (newTokenCount : Nat) : IO Unit := do
  IO.println s!"  interactive: enter a prompt; empty line or :q exits (window={
    BpeModel.contextLength} tokens, generate={newTokenCount})"
  let stdin ← IO.getStdin
  let rec loop : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else
      let out ← generateBpeGreedy tok lv predict prompt newTokenCount
      IO.println s!"  response={text.escape out}"
      loop
  loop

/--
Train the GPT-2-style model over a text corpus using CUDA.

This materializes only the requested deterministic training schedule, rather than every possible
corpus window. The example is compact by GPT-2 standards, but the data path is real:
file bytes → token windows → one-hot tensors → TorchLean CUDA training.
-/
def trainCorpus (runtime : Runtime.Config)
    (trainOpts : Options)
    (bytes : ByteArray) : IO Unit := do
  let sample0 := byteCorpusSampleAt bytes 0
  let first := text.byteTokenWindow bytes (ByteModel.contextLength + 1)
  let firstIds := Tensor.to first (Array Nat)
  IO.println s!"  mode=byte bytes={bytes.size} steps={trainOpts.training.steps} window={
    ByteModel.contextLength}"
  IO.println s!"  first prompt={text.escape (text.Tokenizer.byte.decode
    (firstIds.take ByteModel.contextLength))}"
  IO.println
    s!"  first target={text.escape (text.Tokenizer.byte.decode (firstIds.drop 1))}"
  let ftBytes? ←
    match trainOpts.finetune.finetuneFile? with
    | none => pure none
    | some path => do
        let ftBytes ←
          text.Corpus.readByteFile exeName path trainOpts.corpus.allowSmallData minTrainingBytes
            ByteModel.contextLength
        pure (some ftBytes)
  let pretrainSamples := Data.SampleStream.fromFunction
    (trainOpts.training.steps * trainOpts.training.batchSize) fun index =>
      byteCorpusSampleAt bytes index.val
  let finetuneSamples := match ftBytes? with
    | none => Data.SampleStream.fromFunction 0 (fun index => nomatch index)
    | some ftBytes => Data.SampleStream.fromFunction
        (trainOpts.finetune.finetuneSteps * trainOpts.training.batchSize) fun index =>
          byteCorpusSampleAt ftBytes index.val
  let allSamples := pretrainSamples.append finetuneSamples
  let totalSteps := trainOpts.training.steps +
    match ftBytes? with
    | none => 0
    | some _ => trainOpts.finetune.finetuneSteps
  let trainSamples := if allSamples.size == 0 then
    Data.SampleStream.fromFunction 1 (fun _ => sample0) else allSamples
  let run :=
    Trainer.RunConfig.fromRuntime runtime { optimizer := optim.adam { learningRate := 1e-3 } }
  let trainer := Trainer.new ByteModel.model <|
    Trainer.RunConfig.forObjective run (.oneHotCrossEntropy 2) (seed := runtime.seed)
  trainer.printSummary
  /-
  Each optimizer step consumes `batchSize` deterministic corpus windows. Indexing that
  schedule keeps the pretraining and optional fine-tuning boundaries inspectable while the public
  trainer owns gradient accumulation and optimizer state.
  -/
  let trained ← trainer.train
    (Data.fromStream trainSamples)
    { steps := totalSteps
      samplesPerStep := trainOpts.training.batchSize
      logDestination := .disabled
      logEvery := Nat.max 1 (totalSteps / 10)
      cudaMemorySampleEvery := trainOpts.training.cudaMemorySampleEvery }
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after
  let generated ←
    generateByteGreedy
      trained.predict trainOpts.generation.prompt trainOpts.generation.newTokenCount
  IO.println s!"  greedy generated={text.escape generated}"
  text.Log.writePrompt
    trainOpts.training.logDestination
      "GPT-2 byte corpus training" totalSteps lossBefore lossAfter
    trainOpts.generation (some generated)
    #[s!"data={trainOpts.corpus.dataFile}", Support.deviceNote runtime,
      s!"bytes={bytes.size}"]
  if trainOpts.interaction.interactive then
    interactiveByteLoop trained.predict trainOpts.generation.newTokenCount

/-- Validate, optionally cap, and tokenize one UTF-8 corpus file with GPT-2 BPE. -/
def loadBpeFileTokens
    (trainOpts : Options)
    (tok : text.GPT2BPE.Tokenizer)
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
    match trainOpts.bpe.maximumCharacters? with
    | some n => (fullCorpusText.take n).toString
    | none => fullCorpusText
  let ids ← CLI.orThrow exeName <| text.GPT2BPE.encode tok corpusText
  if ids.size <= BpeModel.contextLength then
    throw <| IO.userError s!"{exeName}: BPE corpus {path} has {ids.size} tokens after capping; \
need more than {BpeModel.contextLength}"
  pure ids

/-- Print the first BPE training window for inspecting tokenization and windowing. -/
def printBpeCorpusPreview {tokenCount : Nat} (tok : text.GPT2BPE.Tokenizer)
    (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (tokens : Tensor Nat [tokenCount]) : IO Unit := do
  let first := Tensor.window tokens (BpeModel.contextLength + 1) 0 0
  let firstIds := Tensor.to first (Array Nat)
  let prompt ← CLI.orThrow exeName <|
    decodeLocalBPE tok lv (Tensor.window first BpeModel.contextLength 0 0)
  let target ← CLI.orThrow exeName <|
    decodeLocalBPE tok lv (Tensor.window first BpeModel.contextLength 1 0)
  IO.println s!"  first local BPE ids={firstIds.toList}"
  IO.println s!"  first prompt={text.escape prompt}"
  IO.println s!"  first target={text.escape target}"

/--
Train the compact GPT-2-style model with the real GPT-2 BPE tokenizer.

This exercises the GPT-2 tokenizer/vocabulary path and can overfit local windows. It is not a
pretrained GPT-2 checkpoint; it is a randomly initialized TorchLean model trained by this command.
-/
def trainBpeCorpus {tokenCount : Nat} (runtime : Runtime.Config)
    (trainOpts : Options)
    (tok : text.GPT2BPE.Tokenizer)
    (lv : text.VocabularyProjection BpeModel.vocabularySize)
    (tokens : Tensor Nat [tokenCount])
    (finetuneTokens? : Option ((count : Nat) × Tensor Nat [count])) : IO Unit := do
  let totalSteps := trainOpts.training.steps +
    match finetuneTokens? with
    | none => 0
    | some _ => trainOpts.finetune.finetuneSteps
  IO.println s!"  mode=bpe local-vocab={lv.size}/{BpeModel.vocabularySize} tokens={
    tokenCount} steps={totalSteps}"
  printBpeCorpusPreview tok lv tokens
  let bounded ← CLI.orThrow exeName <| Tensor.checkIndices BpeModel.vocabularySize tokens
  let sample0 := bpeCorpusSampleAt bounded 0
  let pretrainSamples := Data.SampleStream.fromFunction
    (trainOpts.training.steps * trainOpts.training.batchSize) fun index =>
      bpeCorpusSampleAt bounded index.val
  let finetuneSamples ←
    match finetuneTokens? with
    | none => pure (Data.SampleStream.fromFunction 0 fun index => nomatch index)
    | some ⟨_, finetuneTokens⟩ => do
        let bounded ← CLI.orThrow exeName <|
          Tensor.checkIndices BpeModel.vocabularySize finetuneTokens
        pure <| Data.SampleStream.fromFunction
          (trainOpts.finetune.finetuneSteps * trainOpts.training.batchSize) fun index =>
            bpeCorpusSampleAt bounded index.val
  let scheduledSamples := pretrainSamples.append finetuneSamples
  let samples := if scheduledSamples.size == 0 then
    Data.SampleStream.fromFunction 1 (fun _ => sample0) else scheduledSamples
  let run :=
    Trainer.RunConfig.fromRuntime runtime { optimizer := optim.adam { learningRate := 1e-3 } }
  let trainer := Trainer.new BpeModel.model <|
    Trainer.RunConfig.forObjective run (.oneHotCrossEntropy 2) (seed := runtime.seed)
  trainer.printSummary
  /-
  BPE mode uses the real GPT-2 tokenizer but a compact local output vocabulary. The public trainer
  sees the already-projected one-hot windows; decoding remains here because it is presentation
  logic, not training machinery.
  -/
  let trained ← trainer.train
    (Data.fromStream samples)
    { steps := totalSteps
      samplesPerStep := trainOpts.training.batchSize
      logDestination := .disabled
      logEvery := Nat.max 1 (totalSteps / 10)
      cudaMemorySampleEvery := trainOpts.training.cudaMemorySampleEvery }
  trained.printSummary
  let lossBefore := trained.report.loss.before
  let lossAfter := trained.report.loss.after
  printBpePredictionProbe tok lv trained.predict "after " trainOpts.generation.prompt
  let generated ←
    generateBpeGreedy
      tok lv trained.predict trainOpts.generation.prompt trainOpts.generation.newTokenCount
  IO.println s!"  greedy generated={text.escape generated}"
  text.Log.writePrompt
    trainOpts.training.logDestination
      "GPT-2 BPE corpus training" totalSteps lossBefore lossAfter
    trainOpts.generation (some generated)
    #[s!"data={trainOpts.corpus.dataFile}", Support.deviceNote runtime,
      s!"localVocab={lv.size}/{BpeModel.vocabularySize}", s!"tokens={tokenCount}"]
  if trainOpts.interaction.interactive then
    interactiveBpeLoop tok lv trained.predict trainOpts.generation.newTokenCount

/-- CLI entrypoint for CUDA byte/BPE corpus training. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDevice exeName "GPU corpus trainer"
      usage? := some Options.usage
      printSuccess := true
      runtime := { device? := some .cuda } })
    exeName args
    (.native fun runtime rest => do
      let (trainOpts, rest) ← CLI.orThrow exeName <|
        Options.parse rest
      CLI.requireNoArgs exeName rest
      match trainOpts.bpe.vocabularyFile?, trainOpts.bpe.mergesFile? with
      | some vocabPath, some mergesPath =>
          let tok ← text.GPT2BPE.load vocabPath mergesPath
            (progress := true) (label := exeName)
          let capMsg :=
            match trainOpts.bpe.maximumCharacters? with
            | some n => s!" (max chars={n})"
            | none => ""
          IO.eprintln s!"{exeName}: encoding BPE corpus{capMsg}"
          let tokens ← loadBpeFileTokens trainOpts tok trainOpts.corpus.dataFile
          IO.eprintln s!"{exeName}: encoded BPE corpus original-tokens={tokens.size}"
          let finetuneTokens? ←
            match trainOpts.finetune.finetuneFile? with
            | none => pure none
            | some path =>
                IO.eprintln s!"{exeName}: encoding BPE fine-tuning corpus {path}"
                some <$> loadBpeFileTokens trainOpts tok path
          let promptTokens ← CLI.orThrow exeName <|
            text.GPT2BPE.encode tok trainOpts.generation.prompt
          let projection := text.VocabularyProjection.fromTokens BpeModel.vocabularySize 0
            (Tensor.from tokens)
          let projection := match finetuneTokens? with
            | none => projection
            | some ids => projection.extend (Tensor.from ids)
          let lv := projection.extend (Tensor.from promptTokens)
          let localTokens := lv.encode (Tensor.from tokens)
          let localFinetuneTokens? := finetuneTokens?.map fun ids =>
            (⟨ids.size, lv.encode (Tensor.from ids)⟩ : (count : Nat) × Tensor Nat [count])
          IO.eprintln s!"{exeName}: projected BPE ids to local vocabulary {lv.size}/{
            BpeModel.vocabularySize}"
          trainBpeCorpus runtime trainOpts tok lv localTokens localFinetuneTokens?
      | _, _ =>
          let bytes ← readCorpusBytes trainOpts
          trainCorpus runtime trainOpts bytes)

end NN.Examples.Models.Sequence.TextGpt2
