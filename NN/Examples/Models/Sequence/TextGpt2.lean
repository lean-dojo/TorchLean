/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

GPU-only corpus-training example:
  lake build -R -K cuda=true
  lake exe torchlean text_gpt2 \
    --data-file data/real/text/tinystories_valid.txt \
    --allow-small-data --steps 1000 --log-every 100

Prepare that file with:
  python3 scripts/datasets/download_example_data.py --tinystories-valid

GPT-2 BPE smoke run:
  lake exe torchlean text_gpt2 \
    --data-file data/real/text/tiny_shakespeare.txt \
    --bpe-vocab data/real/gpt2/vocab.json \
    --bpe-merges data/real/gpt2/merges.txt \
    --allow-small-data --max-chars 20000 --steps 10 --log-every 1 \
    --prompt "First Citizen:" --generate 8

Smoke test on a small local file:
  lake exe torchlean text_gpt2 \
    --data-file /tmp/tiny.txt --allow-small-data --steps 2 --log-every 1
-/

module

public import NN
public import NN.API.Text.Bpe
public import NN.API.Models.Gpt2
public import NN.Runtime.Autograd.TorchLean.NN

/-!
# GPU GPT-2 Corpus Trainer

This file trains GPT-2-style models from text in TorchLean.

The model is initialized inside TorchLean and trained by the TorchLean runtime. It does not load a
pretrained PyTorch/Hugging Face checkpoint:

* reusable tokenization lives in `NN.API.Text` / `NN.API.Text.Bpe`,
* the compact GPT-2-style architecture lives in `NN.API.nn.models` (see `NN.API.Models.Gpt2`),
* this file is the runnable corpus trainer and enforces CUDA by default.

The default path is byte-level because it is compact and fast.  Passing `--bpe-vocab` and
`--bpe-merges` switches to the Lean-native GPT-2 BPE tokenizer, using the standard 50,257-way GPT-2
token vocabulary.  That BPE path is still **training from scratch** in TorchLean; it does not load a
pretrained checkpoint.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.TextGpt2

/-- Runner subcommand name. This subcommand trains a GPT-2-style model from scratch. -/
def exeName : String := "torchlean text_gpt2"
def defaultLogJson : System.FilePath := "data/model_zoo/text_gpt2_trainlog.json"

/-- Minimum corpus size for the default public training path: 100 MiB. -/
def minTrainingBytes : Nat :=
  100 * 1024 * 1024

/--
Default byte-level context window for the CUDA corpus trainer.

Keeping this near the file top lets corpus validation and the model architecture agree without
depending on declaration order.
-/
def byteSeqLen : Nat := 8

/-- Parsed local options for the corpus trainer. -/
structure TrainOptions where
  /-- UTF-8 or raw-byte text corpus. -/
  dataFile : System.FilePath
  /-- Shared step count and TrainLog destination. -/
  train : Common.LoggedTrainFlags
  /-- Optional second corpus for fine-tuning after the main corpus pass. -/
  finetuneFile? : Option System.FilePath
  /-- Number of optimizer steps on the fine-tuning corpus. -/
  finetuneSteps : Nat
  /-- Print loss every `logEvery` steps.  `0` disables progress logging. -/
  logEvery : Nat
  /-- Allow small files for bounded local checks. -/
  allowSmallData : Bool
  /-- Optional GPT-2 `vocab.json` path.  Supplying this plus `bpeMerges?` enables BPE mode. -/
  bpeVocab? : Option System.FilePath
  /-- Optional GPT-2 `merges.txt` path.  Supplying this plus `bpeVocab?` enables BPE mode. -/
  bpeMerges? : Option System.FilePath
  /-- Prompt used for the post-training generation probe. -/
  prompt : String
  /-- Number of autoregressive BPE tokens to generate in the post-training probe. -/
  generate : Nat
  /-- Keep the trained CUDA model alive and read prompts from stdin. -/
  interactive : Bool
  /-- Optional text-character cap for bounded BPE runs. -/
  maxChars? : Option Nat
deriving Repr

namespace TrainOptions

def steps (trainOpts : TrainOptions) : Nat :=
  trainOpts.train.steps

def logPath (trainOpts : TrainOptions) : System.FilePath :=
  trainOpts.train.logPath

def log (trainOpts : TrainOptions) : _root_.Runtime.Training.LogDestination :=
  trainOpts.train.log

end TrainOptions

/-- Parse options owned by this example; runtime flags are parsed by `TorchLean.Module.run`. -/
def parseTrainOptions (args : List String) : Except String (TrainOptions × List String) := do
  let (dataFile?, args) ← CLI.takePathFlagOnce args "data-file"
  let dataFile ←
    match dataFile? with
    | some p => pure p
    | none => throw "--data-file is required for text_gpt2"
  let (train, args) ← Common.parseLoggedTrainFlags exeName args defaultLogJson 1000
  let (finetuneFile?, args) ← CLI.takePathFlagOnce args "finetune-file"
  let (finetuneSteps?, args) ← CLI.takeNatFlagOnce args "finetune-steps"
  let (logEvery?, args) ← CLI.takeNatFlagOnce args "log-every"
  let (allowSmallData, args) ← CLI.takeBoolFlagOnce args "allow-small-data"
  let (bpeVocab?, args) ← CLI.takePathFlagOnce args "bpe-vocab"
  let (bpeMerges?, args) ← CLI.takePathFlagOnce args "bpe-merges"
  let (prompt?, args) ← CLI.takeFlagValueOnce args "prompt"
  let (generate?, args) ← CLI.takeNatFlagOnce args "generate"
  let (interactive, args) ← CLI.takeBoolFlagOnce args "interactive"
  let (maxChars?, args) ← CLI.takeNatFlagOnce args "max-chars"
  match bpeVocab?, bpeMerges? with
  | some _, none => throw "--bpe-vocab requires --bpe-merges"
  | none, some _ => throw "--bpe-merges requires --bpe-vocab"
  | _, _ => pure ()
  pure ({ dataFile := dataFile
          train := train
          finetuneFile? := finetuneFile?
          finetuneSteps := finetuneSteps?.getD train.steps
          logEvery := logEvery?.getD 100
          allowSmallData := allowSmallData
          bpeVocab? := bpeVocab?
          bpeMerges? := bpeMerges?
          prompt := prompt?.getD "First Citizen:"
          generate := generate?.getD 8
          interactive := interactive
          maxChars? := maxChars? }, args)

/--
Force the runner into the intended CUDA configuration.

Users should not have to remember `--cuda --fast-kernels` for this example.  We still reject
`--cpu` explicitly because silently switching to CPU would make a large text-training run look
hung rather than correctly configured.
-/
def forceCudaArgs (args : List String) : Except String (List String) := do
  if args.contains "--cpu" then
    throw "text_gpt2 is GPU-only; remove --cpu"
  let args := if args.contains "--cuda" then args else "--cuda" :: args
  let args := if args.contains "--fast-kernels" then args else "--fast-kernels" :: args
  pure args

/-- Read the primary raw text corpus. -/
def readCorpusBytes (opts : TrainOptions) : IO ByteArray :=
  text.Corpus.readByteFile exeName opts.dataFile opts.allowSmallData minTrainingBytes byteSeqLen

/-- Build a supervised next-token sample from already-tokenized ids. -/
def mkSampleFromTokensWith {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    {batch seqLen vocab : Nat} (tokens : List Nat) :
    API.sample.Supervised α (shape![batch, seqLen, vocab])
      (shape![batch, seqLen, vocab]) :=
  text.causalLmSampleOneHotBatch (α := α) batch seqLen vocab tokens

def mkSampleFromTokenRowsWith {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    {batch seqLen vocab : Nat} (tokensAt : Fin batch → List Nat) :
    API.sample.Supervised α (shape![batch, seqLen, vocab])
      (shape![batch, seqLen, vocab]) :=
  text.causalLmSampleOneHotBatchRows (α := α) batch seqLen vocab tokensAt

namespace ByteGpt2

/-- Byte-level vocabulary: one token per byte. -/
def vocab : Nat := text.Tokenizer.byte.vocabSize

/-- Single-sequence batches keep the example small and fully interactive. -/
def batch : Nat := 1

/--
Interactive context window.

This shares the folder-level byte context constant so corpus validation, byte training, and BPE
training use the same tensor layout. Larger windows require more allocator headroom, not
something we should quietly make the default before allocator pressure is solved.
-/
def seqLen : Nat := byteSeqLen

/-- Small two-head Transformer width. -/
def numHeads : Nat := 2

/-- Per-head width. -/
def headDim : Nat := 4

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Feed-forward hidden width. -/
def ffnHidden : Nat := 32

/-- Number of Transformer blocks. -/
def layers : Nat := 1

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

def cfg : nn.models.CausalOneHotConfig :=
  { batch := batch
    seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers }

abbrev σ : Shape :=
  nn.models.causalOneHotShape cfg

abbrev τ : Shape :=
  σ

/--
Runnable byte-level GPT-style model for corpus pretraining/fine-tuning.

This is deliberately compact, but it has enough context to make the interactive prompt loop
useful for quick local experiments.
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot cfg

end ByteGpt2

def mkByteCorpusSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (bytes : ByteArray) (i : Nat) : API.sample.Supervised α ByteGpt2.σ ByteGpt2.τ :=
  let toks := text.byteTokenWindow bytes (ByteGpt2.seqLen + 1)
    (offset := text.Corpus.byteOffset bytes i ByteGpt2.seqLen)
  mkSampleFromTokensWith (α := α) (batch := ByteGpt2.batch) (seqLen := ByteGpt2.seqLen)
    (vocab := ByteGpt2.vocab) toks

def mkBytePromptSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (prompt : String) : API.sample.Supervised α ByteGpt2.σ ByteGpt2.τ :=
  let ids := text.Tokenizer.byte.encode prompt
  let start := if ids.length > ByteGpt2.seqLen then ids.length - ByteGpt2.seqLen else 0
  let window := (ids.drop start).take ByteGpt2.seqLen
  mkSampleFromTokensWith (α := α) (batch := ByteGpt2.batch) (seqLen := ByteGpt2.seqLen)
    (vocab := ByteGpt2.vocab) window

def lastPredictedByteId (logits : Tensor Float ByteGpt2.τ) : Nat :=
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float)
    (batch := ByteGpt2.batch) (seqLen := ByteGpt2.seqLen) (vocab := ByteGpt2.vocab)
    (batchIdx := ⟨0, by decide⟩) logits
  ids.getD (ByteGpt2.seqLen - 1) 0

def generateByteGreedy
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential ByteGpt2.σ ByteGpt2.τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model)
      [ByteGpt2.σ, ByteGpt2.τ])
    (prompt : String) (steps : Nat) : IO String := do
  let mut ids := text.Tokenizer.byte.encode prompt
  for _ in [0:steps] do
    let start := if ids.length > ByteGpt2.seqLen then ids.length - ByteGpt2.seqLen else 0
    let window := (ids.drop start).take ByteGpt2.seqLen
    let sample := mkSampleFromTokensWith (α := Float) (batch := ByteGpt2.batch)
      (seqLen := ByteGpt2.seqLen) (vocab := ByteGpt2.vocab) window
    let logits ← nn.eval1 (α := Float) opts model
      m.trainer.params
      (NN.API.sample.x sample)
    ids := ids ++ [lastPredictedByteId logits]
  pure (text.Tokenizer.byte.decode ids)

partial def interactiveByteLoop
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential ByteGpt2.σ ByteGpt2.τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model)
      [ByteGpt2.σ, ByteGpt2.τ])
    (generate : Nat) : IO Unit := do
  IO.println s!"  interactive: enter a prompt; empty line or :q exits (window={ByteGpt2.seqLen} bytes, generate={generate})"
  let stdin ← IO.getStdin
  let rec loop : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else
      let out ← generateByteGreedy opts model m prompt generate
      IO.println s!"  response={text.escapeForDisplay out}"
      loop
  loop

namespace BpeGpt2

/--
Compact vocabulary used by the runnable BPE training path.

The tokenizer still uses GPT-2's real 50,257-token BPE files.  For the small Lean/CUDA smoke model
we project the corpus tokens into a local vocabulary of the first observed BPE ids.  This keeps the
example interactive while preserving the tokenizer/data path; a full 50k-way output head is a much
larger training run.
-/
def vocab : Nat := 512

/-- Keep the BPE smoke model aligned with the small byte-level GPT-2 path. -/
def batch : Nat := 2

/-- Short context window used by the trainer. -/
def seqLen : Nat := byteSeqLen

/-- Number of attention heads in the miniature BPE Transformer. -/
def numHeads : Nat := 1

/-- Per-head width.  The model is intentionally compact even though the vocabulary is real GPT-2. -/
def headDim : Nat := 8

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Feed-forward hidden width. -/
def ffnHidden : Nat := 32

/-- Number of Transformer blocks. -/
def layers : Nat := 1

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

def cfg : nn.models.CausalOneHotConfig :=
  { batch := batch
    seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers }

abbrev σ : Shape :=
  nn.models.causalOneHotShape cfg

abbrev τ : Shape :=
  σ

/--
Compact GPT-2-style model with the real GPT-2 BPE vocabulary.

This is not OpenAI GPT-2-small.  It is a TorchLean-native miniature Transformer whose input/output
vocabulary matches GPT-2 BPE, so tokenizer/probing behavior is realistic while the model remains
small enough for a local smoke run.
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot cfg

end BpeGpt2

/-- Local projection from original GPT-2 BPE ids to the compact working vocabulary. -/
structure LocalBpeVocab where
  /-- Original GPT-2 id for each local id. -/
  originals : Array Nat
  /-- Reverse lookup from original GPT-2 id to local id. -/
  toLocalMap : Std.HashMap Nat Nat

/-- Number of live entries in a local BPE projection. -/
def LocalBpeVocab.size (lv : LocalBpeVocab) : Nat :=
  lv.originals.size

/-- Map an original GPT-2 BPE id into the compact local vocabulary, using local id `0` as OOV. -/
def LocalBpeVocab.toLocal (lv : LocalBpeVocab) (id : Nat) : Nat :=
  (lv.toLocalMap[id]?).getD 0

/-- Map a compact local id back to its original GPT-2 BPE id. -/
def LocalBpeVocab.toOriginal (lv : LocalBpeVocab) (localId : Nat) : Nat :=
  lv.originals.getD localId (lv.originals.getD 0 0)

/-- Build the compact working vocabulary from corpus ids and prompt ids. -/
def buildLocalBpeVocab (maxVocab : Nat) (corpusIds promptIds : Array Nat) : LocalBpeVocab :=
  Id.run do
    let mut originals : Array Nat := #[0]
    let mut map : Std.HashMap Nat Nat := (Std.HashMap.emptyWithCapacity).insert 0 0
    let addId (originals : Array Nat) (map : Std.HashMap Nat Nat) (id : Nat) :
        Array Nat × Std.HashMap Nat Nat :=
      if map.contains id || originals.size ≥ maxVocab then
        (originals, map)
      else
        let localId := originals.size
        (originals.push id, map.insert id localId)
    for id in corpusIds do
      let p := addId originals map id
      originals := p.1
      map := p.2
    for id in promptIds do
      let p := addId originals map id
      originals := p.1
      map := p.2
    return { originals := originals, toLocalMap := map }

/-- Apply a local BPE projection to an array of original GPT-2 ids. -/
def localizeBpeTokens (lv : LocalBpeVocab) (tokens : Array Nat) : Array Nat :=
  tokens.map (fun id => lv.toLocal id)

/-- Build one BPE training sample from a tokenized corpus. -/
def mkBpeCorpusSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (tokens : Array Nat) (i : Nat) : API.sample.Supervised α BpeGpt2.σ BpeGpt2.τ :=
  -- The BPE smoke model uses a real batch as well: each batch row gets a different deterministic
  -- corpus window.  This keeps the tokenizer/vocabulary path realistic without needing a huge
  -- 50k-way model head for this runnable example.
  let toksAt := text.Corpus.randomBatchTokenWindows tokens BpeGpt2.batch BpeGpt2.seqLen 0 i
  mkSampleFromTokenRowsWith (α := α) (batch := BpeGpt2.batch) (seqLen := BpeGpt2.seqLen)
    (vocab := BpeGpt2.vocab) toksAt

/-- Turn a BPE prompt into one model input window. -/
def mkBpePromptSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (tok : text.Gpt2Bpe.Tokenizer) (lv : LocalBpeVocab) (prompt : String) :
    Except String (API.sample.Supervised α BpeGpt2.σ BpeGpt2.τ) := do
  let ids ← (text.Gpt2Bpe.encode tok prompt).map (fun ids => ids.map lv.toLocal)
  let start := if ids.length > BpeGpt2.seqLen then ids.length - BpeGpt2.seqLen else 0
  let window := (ids.drop start).take BpeGpt2.seqLen
  pure <| mkSampleFromTokensWith (α := α) (batch := BpeGpt2.batch) (seqLen := BpeGpt2.seqLen)
    (vocab := BpeGpt2.vocab) window

/-- Argmax token id at the final context position. -/
def lastPredictedTokenId (logits : Tensor Float BpeGpt2.τ) : Nat :=
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float)
    (batch := BpeGpt2.batch) (seqLen := BpeGpt2.seqLen) (vocab := BpeGpt2.vocab)
    (batchIdx := ⟨0, by decide⟩) logits
  ids.getD (BpeGpt2.seqLen - 1) 0

def decodeBpeD (tok : text.Gpt2Bpe.Tokenizer) (ids : List Nat) : String :=
  text.Gpt2Bpe.decodeD tok ids

def decodeLocalBpeD (tok : text.Gpt2Bpe.Tokenizer) (lv : LocalBpeVocab) (ids : List Nat) :
    String :=
  decodeBpeD tok (ids.map lv.toOriginal)

/-- Print an argmax probe for a prompt under the BPE model. -/
def printBpePredictionProbe
    (opts : Runtime.Autograd.Torch.Options)
    (tok : text.Gpt2Bpe.Tokenizer)
    (lv : LocalBpeVocab)
    (model : nn.Sequential BpeGpt2.σ BpeGpt2.τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model)
      [BpeGpt2.σ, BpeGpt2.τ])
    (label prompt : String) : IO Unit := do
  let sample ← Common.orThrow exeName <| mkBpePromptSample (α := Float) tok lv prompt
  let logits ← nn.eval1 (α := Float) opts model
    m.trainer.params
    (NN.API.sample.x sample)
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float)
    (batch := BpeGpt2.batch) (seqLen := BpeGpt2.seqLen) (vocab := BpeGpt2.vocab)
    (batchIdx := ⟨0, by decide⟩) logits
  IO.println s!"  {label} pred={text.escapeForDisplay (decodeLocalBpeD tok lv ids)}"
  IO.println s!"  prompt={text.escapeForDisplay prompt}"

/--
Greedy BPE generation by repeatedly feeding the last `seqLen` tokens and appending the final-position
argmax. This is a compact diagnostic loop, not a high-quality sampler.
-/
def generateBpeGreedy
    (opts : Runtime.Autograd.Torch.Options)
    (tok : text.Gpt2Bpe.Tokenizer)
    (lv : LocalBpeVocab)
    (model : nn.Sequential BpeGpt2.σ BpeGpt2.τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model)
      [BpeGpt2.σ, BpeGpt2.τ])
    (prompt : String) (steps : Nat) : IO String := do
  let initOrigIds ← Common.orThrow exeName <| text.Gpt2Bpe.encode tok prompt
  let initIds := initOrigIds.map lv.toLocal
  let mut ids := initIds
  for _ in [0:steps] do
    let start := if ids.length > BpeGpt2.seqLen then ids.length - BpeGpt2.seqLen else 0
    let window := (ids.drop start).take BpeGpt2.seqLen
    let sample := mkSampleFromTokensWith (α := Float) (batch := BpeGpt2.batch)
      (seqLen := BpeGpt2.seqLen) (vocab := BpeGpt2.vocab) window
    let logits ← nn.eval1 (α := Float) opts model
      m.trainer.params
      (NN.API.sample.x sample)
    ids := ids ++ [lastPredictedTokenId logits]
  pure (decodeLocalBpeD tok lv ids)

/--
Train the GPT-2-style model over a text corpus using CUDA.

This intentionally performs one optimizer step per corpus window, rather than materializing the
entire dataset in memory.  The example is still compact by GPT-2 standards, but the data path is real:
file bytes → token windows → one-hot tensors → TorchLean CUDA training.
-/
def trainCorpusFloat (opts : Runtime.Autograd.Torch.Options) (trainOpts : TrainOptions)
    (bytes : ByteArray) : IO Unit := do
  nn.withModel ByteGpt2.mkModel fun model => do
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let sample0 := mkByteCorpusSample (α := Float) bytes 0
    let loss0 ← TorchLean.Module.forward (α := Float) m sample0
    let L0 := Tensor.toScalar loss0
    IO.println s!"  mode=byte bytes={bytes.size} steps={trainOpts.steps} window={ByteGpt2.seqLen}"
    IO.println s!"  initial loss={L0}"
    let first := text.byteTokenWindow bytes (ByteGpt2.seqLen + 1)
    IO.println s!"  first prompt={text.escapeForDisplay (text.Tokenizer.byte.decode (first.take ByteGpt2.seqLen))}"
    IO.println s!"  first target={text.escapeForDisplay (text.Tokenizer.byte.decode (first.drop 1))}"

    let opt := TorchLean.Optim.adam (α := Float)
      (paramShapes := nn.paramShapes model)
      (lr := 1e-3)
      (beta1 := 0.9)
      (beta2 := 0.999)
      (epsilon := 1e-8)
    let optH ← TorchLean.Optim.handle (α := Float) m opt
    let trainPhase (label : String) (phaseBytes : ByteArray) (steps : Nat) : IO Unit := do
      IO.println s!"  phase={label} bytes={phaseBytes.size} steps={steps}"
      for step in [0:steps] do
        let sample := mkByteCorpusSample (α := Float) phaseBytes step
        optH.step sample
        let done := step + 1
        if trainOpts.logEvery != 0 && done % trainOpts.logEvery == 0 then
          let probe := mkByteCorpusSample (α := Float) phaseBytes done
          let loss ← TorchLean.Module.forward (α := Float) m probe
          IO.println s!"  {label} step={done} loss={Tensor.toScalar loss}"

    trainPhase "pretrain" bytes trainOpts.steps
    match trainOpts.finetuneFile? with
    | none => pure ()
    | some path =>
        let ftBytes ←
          text.Corpus.readByteFile exeName path trainOpts.allowSmallData minTrainingBytes ByteGpt2.seqLen
        trainPhase "finetune" ftBytes trainOpts.finetuneSteps

    let loss1 ← TorchLean.Module.forward (α := Float) m sample0
    let L1 := Tensor.toScalar loss1
    let generated ← generateByteGreedy opts model m trainOpts.prompt trainOpts.generate
    IO.println s!"  greedy generated={text.escapeForDisplay generated}"
    IO.println s!"  final first-window loss={L1}"
    let totalSteps := trainOpts.steps +
      match trainOpts.finetuneFile? with
      | none => 0
      | some _ => trainOpts.finetuneSteps
    Common.writeBeforeAfterLossLogTo trainOpts.log "GPT-2 byte corpus training" totalSteps L0 L1
      #[s!"data={trainOpts.dataFile}", s!"device={if opts.useGpu then "cuda" else "cpu"}",
        s!"bytes={bytes.size}"]
    if trainOpts.interactive then
      interactiveByteLoop opts model m trainOpts.generate

/-- Load and tokenize the text corpus with GPT-2 BPE. -/
def loadBpeCorpusTokens (trainOpts : TrainOptions) (tok : text.Gpt2Bpe.Tokenizer) :
    IO (Array Nat) := do
  let corpusText0 ← IO.FS.readFile trainOpts.dataFile
  let corpusText :=
    match trainOpts.maxChars? with
    | some n => (corpusText0.take n).toString
    | none => corpusText0
  let ids ← Common.orThrow exeName <| text.Gpt2Bpe.encode tok corpusText
  let arr := ids.toArray
  if arr.size <= BpeGpt2.seqLen then
    throw <| IO.userError s!"{exeName}: BPE corpus is too small for a {BpeGpt2.seqLen}-token window"
  pure arr

/-- Verbose BPE loader used by this example so long startup work is visible. -/
def loadBpeTokenizerForDemo (vocabPath mergesPath : System.FilePath) :
    IO text.Gpt2Bpe.Tokenizer := do
  IO.eprintln s!"{exeName}: reading BPE vocab.json"
  let vocabText ← IO.FS.readFile vocabPath
  IO.eprintln s!"{exeName}: parsing BPE vocab.json chars={vocabText.length}"
  let vocab ←
    match text.Gpt2Bpe.parseVocabText vocabText with
    | .ok v => pure v
    | .error e => throw <| IO.userError e
  IO.eprintln s!"{exeName}: parsed BPE vocab entries={vocab.size}"
  IO.eprintln s!"{exeName}: reading BPE merges.txt"
  let mergesText ← IO.FS.readFile mergesPath
  let merges := text.Gpt2Bpe.parseMerges mergesText
  IO.eprintln s!"{exeName}: parsed BPE merges={merges.size}"
  IO.eprintln s!"{exeName}: building BPE lookup maps"
  pure { vocab := vocab
         merges := merges
         vocabMap := text.Gpt2Bpe.vocabMapOf vocab
         idMap := text.Gpt2Bpe.idMapOf vocab
         mergeMap := text.Gpt2Bpe.mergeMapOf merges }

/-- Print the first BPE training window for sanity. -/
def printBpeCorpusPreview (tok : text.Gpt2Bpe.Tokenizer) (lv : LocalBpeVocab)
    (tokens : Array Nat) : IO Unit := do
  let first := text.Corpus.tokenArrayWindow tokens (BpeGpt2.seqLen + 1) 0
  IO.println s!"  first local BPE ids={first}"
  IO.println s!"  first prompt={text.escapeForDisplay (decodeLocalBpeD tok lv (first.take BpeGpt2.seqLen))}"
  IO.println s!"  first target={text.escapeForDisplay (decodeLocalBpeD tok lv (first.drop 1))}"

/--
Train the compact GPT-2-style model with the real GPT-2 BPE tokenizer.

This is deliberately a smoke-scale model: it exercises the 50,257-way tokenizer/vocabulary path and
can overfit local windows, but it is far too small and short-trained to behave like pretrained GPT-2.
-/
def trainBpeCorpusFloat (opts : Runtime.Autograd.Torch.Options) (trainOpts : TrainOptions)
    (tok : text.Gpt2Bpe.Tokenizer) (lv : LocalBpeVocab) (tokens : Array Nat) : IO Unit := do
  nn.withModel BpeGpt2.mkModel fun model => do
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let sample0 := mkBpeCorpusSample (α := Float) tokens 0
    let loss0 ← TorchLean.Module.forward (α := Float) m sample0
    let L0 := Tensor.toScalar loss0
    IO.println s!"  mode=bpe local-vocab={lv.size}/{BpeGpt2.vocab} tokens={tokens.size} steps={trainOpts.steps}"
    IO.println s!"  initial loss={L0}"
    printBpeCorpusPreview tok lv tokens
    printBpePredictionProbe opts tok lv model m "before" trainOpts.prompt

    let opt := TorchLean.Optim.adam (α := Float)
      (paramShapes := nn.paramShapes model)
      (lr := 1e-3)
      (beta1 := 0.9)
      (beta2 := 0.999)
      (epsilon := 1e-8)
    let optH ← TorchLean.Optim.handle (α := Float) m opt
    for step in [0:trainOpts.steps] do
      let sample := mkBpeCorpusSample (α := Float) tokens step
      optH.step sample
      let done := step + 1
      if trainOpts.logEvery != 0 && done % trainOpts.logEvery == 0 then
        let probe := mkBpeCorpusSample (α := Float) tokens done
        let loss ← TorchLean.Module.forward (α := Float) m probe
        IO.println s!"  step={done} loss={Tensor.toScalar loss}"

    let loss1 ← TorchLean.Module.forward (α := Float) m sample0
    let L1 := Tensor.toScalar loss1
    printBpePredictionProbe opts tok lv model m "after " trainOpts.prompt
    let generated ← generateBpeGreedy opts tok lv model m trainOpts.prompt trainOpts.generate
    IO.println s!"  greedy generated={text.escapeForDisplay generated}"
    IO.println s!"  final first-window loss={L1}"
    Common.writeBeforeAfterLossLogTo trainOpts.log "GPT-2 BPE corpus training" trainOpts.steps L0 L1
      #[s!"data={trainOpts.dataFile}", s!"device={if opts.useGpu then "cuda" else "cpu"}",
        s!"localVocab={lv.size}/{BpeGpt2.vocab}", s!"tokens={tokens.size}"]

def main (args : List String) : IO UInt32 := do
  match forceCudaArgs args with
  | .error e =>
      IO.eprintln s!"{exeName}: {e}"
      pure 1
  | .ok args =>
      TorchLean.Module.run exeName args
        (.float (fun opts rest => do
          if !opts.useGpu then
            throw <| IO.userError s!"{exeName}: CUDA runtime was not selected"
          let (trainOpts, rest) ← Common.orThrow exeName <| parseTrainOptions rest
          Common.orThrow exeName <| CLI.requireNoArgs rest
          let bytes ← readCorpusBytes trainOpts
          match trainOpts.bpeVocab?, trainOpts.bpeMerges? with
          | some vocabPath, some mergesPath =>
              IO.eprintln s!"{exeName}: loading BPE tokenizer vocab={vocabPath} merges={mergesPath}"
              let tok ← loadBpeTokenizerForDemo vocabPath mergesPath
              IO.eprintln s!"{exeName}: loaded BPE tokenizer vocab={tok.vocab.size} merges={tok.merges.size}"
              let capMsg :=
                match trainOpts.maxChars? with
                | some n => s!" (max chars={n})"
                | none => ""
              IO.eprintln s!"{exeName}: encoding BPE corpus{capMsg}"
              let tokens ← loadBpeCorpusTokens trainOpts tok
              IO.eprintln s!"{exeName}: encoded BPE corpus original-tokens={tokens.size}"
              let promptIds ← Common.orThrow exeName <| text.Gpt2Bpe.encode tok trainOpts.prompt
              let lv := buildLocalBpeVocab BpeGpt2.vocab tokens promptIds.toArray
              let localTokens := localizeBpeTokens lv tokens
              IO.eprintln s!"{exeName}: projected BPE ids to local vocabulary {lv.size}/{BpeGpt2.vocab}"
              trainBpeCorpusFloat opts trainOpts tok lv localTokens
          | _, _ =>
              trainCorpusFloat opts trainOpts bytes))
        { banner? := some (fun opts =>
            s!"{exeName}: GPU corpus trainer (device={if opts.useGpu then "cuda" else "cpu"})")
          printOk := true }

end NN.Examples.Models.Sequence.TextGpt2
