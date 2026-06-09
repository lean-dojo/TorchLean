/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  lake exe torchlean gpt2 --cpu
  lake build -R -K cuda=true && lake exe torchlean gpt2 --cuda
  lake exe torchlean gpt2 --cuda --tiny-shakespeare --prompt "First Citizen:" --steps 200 \
    --generate 80 --temperature 0.85 --top-k 12 --sample-seed 7
  lake exe torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare --steps 200 \
    --save-params data/model_zoo/gpt2_shakespeare.params.json
  lake exe torchlean gpt2_saved --cuda --fast-kernels --params data/model_zoo/gpt2_shakespeare.params.json \
    --prompt "First Citizen:" --generate 160

Dataset example:
  python3 scripts/datasets/download_example_data.py --tiny-shakespeare
  lake exe torchlean gpt2 --cuda --tiny-shakespeare --steps 100
  lake exe torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare --steps 100

This is a compact GPT-2-style *causal* language-model command (byte-level tokens).

Performance note: the CPU path is a pure Lean eager runtime and is much slower than CUDA for
Transformer workloads. By default we run a forward pass on CPU (`--steps` defaults to `0`) and a
one-step training update on CUDA (`--steps` defaults to `1`). You can always force training on CPU
by passing `--steps <n>`.
The goal is to exercise masked self-attention + LayerNorm + FFN on both CPU and CUDA eager
backends, using reusable `API.nn` layers and `API.text` helpers (tokenizer + one-hot samples).

After a run that writes `--log <path>`, you can view the prompt and sampled continuation in the
infoview via:

`#gpt2_train_log_file_view "<path>"`
-/

module


public import NN
public import NN.API.Models.Gpt2
public import NN.Runtime.Autograd.TorchLean.NN
public import NN.API.Runtime
public import NN.Examples.Models.Common.RealData
public import LeanProfiler

/-!
GPT-2 style sequence model example.

This file defines the runnable causal language-model architecture used by the example CLI,
including training, generation, and infoview helpers that exercise the same runtime path as the
command-line model.
-/

/-!
# GPT-2-Style Causal Language Model Example

Runnable `torchlean gpt2` example. It builds a compact GPT-2-style causal transformer over
byte-level tokens, with optional real text input from tiny-shakespeare or `--data-file PATH`.

If you are looking for the simplest "Karpathy-style single text file" path, start with
`torchlean chargpt` (character-level tokenizer). This `gpt2` command is byte-level and is meant to
show the Transformer block wiring and save/reload loop.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake build -R -K cuda=true && lake exe torchlean gpt2 --cuda --tiny-shakespeare --steps 100
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.Gpt2

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := Common.modelZooTrainLog "gpt2"

/--
Batch size for the byte-level causal Transformer.

The executable trains on a compact real-text slice rather than presenting itself as a full
pretraining recipe. The point is to exercise the full TorchLean stack for a causal Transformer: build
typed batches, run attention and feed-forward layers, update parameters, and decode logits back to
text.
-/
def batch : Nat := 2

/--
Prompt/target window length.

Sixty-four byte tokens stays practical for local eager-CUDA runs, but it gives the compact
Transformer enough local context to learn short names, line breaks, speaker prefixes, and a little
phrase structure in Tiny Shakespeare. Shorter windows are useful for parser/kernel checks but
underrepresent the model stack during text generation.
-/
def seqLen : Nat := 64

/-- Byte-level vocabulary size. Each UTF-8 byte is one token. -/
def vocab : Nat := text.Tokenizer.byte.vocabSize

/-- Number of attention heads in the miniature Transformer block. -/
def numHeads : Nat := 2

/--
Per-head embedding width. The model dimension is `numHeads * headDim`.

The default is sized for a local byte-level language-model run. A wider `dModel = 64` variant runs,
but in the current eager-CUDA training loop it is slower and did not improve the 2k-step Shakespeare
sample enough to justify making it the default. Use this file to inspect Transformer/autograd
behavior; use the Mamba example when the goal is a compact text model with longer sequence context.
-/
def headDim : Nat := 16

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Hidden width of the feed-forward sublayer. -/
def ffnHidden : Nat := 128

/-- Number of Transformer encoder blocks. -/
def layers : Nat := 2

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

/-- Input shape: batched byte-level one-hot token windows. -/
abbrev σ : Shape :=
  shape![batch, seqLen, vocab]

/-- Output shape: one vocabulary-logit row for every input token position. -/
abbrev τ : Shape :=
  σ

/-- Public GPT-style causal Transformer constructor specialized to the byte-level config. -/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot
    { batch := batch
      seqLen := seqLen
      vocab := vocab
      numHeads := numHeads
      headDim := headDim
      ffnHidden := ffnHidden
      layers := layers }

/-- Build a batched causal-LM sample by repeating one token window across all rows. -/
def mkSampleFromTokenIds {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (toks : List Nat) : API.sample.Supervised α σ τ :=
  let (x2DF, y2DF) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab) toks
    (padId := 32)
  let x2D : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat x2DF
  let y2D : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat y2DF
  let x : Tensor α σ := Tensor.dim (fun _bi => x2D)
  let y : Tensor α τ := Tensor.dim (fun _bi => y2D)
  API.sample.mk x y

/--
Build a batch sample from per-row token windows.

`idsByBatch[i]` is the `(seqLen + 1)`-token window for batch row `i`. If fewer than `batch` windows
are provided we repeat the last one; callers should normally pass exactly `batch` windows.
-/
def mkSampleBatchFromTokenIds (idsByBatch : Array (List Nat)) :
    API.sample.Supervised Float σ τ :=
  let fallback : List Nat := idsByBatch.getD 0 (List.replicate (seqLen + 1) 32)
  let idsAt (i : Fin batch) : List Nat :=
    idsByBatch.getD i.val fallback
  text.causalLmSampleOneHotBatchRows (α := Float) batch seqLen vocab idsAt (padId := 32)

/-- Build one next-token-prediction sample from text. -/
def mkSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (input : String := "First Citizen:") : API.sample.Supervised α σ τ :=
  mkSampleFromTokenIds (α := α) (text.Tokenizer.byte.encode input)

/--
Parse GPT-2-specific data flags and return the training corpus plus remaining runtime flags.
-/
def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName RealData.tinyShakespearePath
    [("--tiny-shakespeare", RealData.tinyShakespearePath),
      ("--tinystories-valid", RealData.tinyStoriesValidPath)]
    RealData.missingTinyShakespeareOrTinyStoriesHint args

/-- Parsed training, sampling, and checkpoint options for the byte-level GPT command. -/
structure TrainOptions where
  /-- Common model-training flags: steps, log path, CUDA memory watch, and learning rate. -/
  base : Common.ModelTrainFlags
  /-- Number of corpus windows cycled through during training. -/
  windows : Nat
  /-- Prompt used for before/after reports and generation. -/
  prompt : String
  /-- Number of generated byte tokens after training. -/
  generate : Nat
  /-- Sampling temperature for generation. -/
  temperature : Float
  /-- Top-k cutoff for generation; `1` gives greedy decoding. -/
  topK : Nat
  /-- Penalty applied to recently generated byte ids. -/
  repeatPenalty : Float
  /-- Number of recent bytes considered by the repetition penalty. -/
  repeatWindow : Nat
  /-- Seed for reproducible top-k sampling. -/
  seed : Nat
  /-- Restrict generated bytes to printable ASCII plus newline. -/
  asciiOnly : Bool
  /-- Enable the terminal prompt loop after training. -/
  interactive : Bool
  /-- Optional checkpoint path loaded before training/generation. -/
  loadParams? : Option System.FilePath
  /-- Optional checkpoint path written after training. -/
  saveParams? : Option System.FilePath
deriving Repr

namespace TrainOptions

/-- Number of optimizer steps. -/
def steps (train : TrainOptions) : Nat :=
  train.base.train.steps

/-- Adam learning rate used by the GPT-style training command. -/
def lr (train : TrainOptions) : Float :=
  train.base.lr

/-- Training-log destination. -/
def log (train : TrainOptions) : _root_.Runtime.Training.LogDestination :=
  train.base.train.log

/-- Concrete JSON log path when the destination is file-backed. -/
def logPath (train : TrainOptions) : System.FilePath :=
  train.base.train.logPath

/-- CUDA allocator telemetry cadence. -/
def cudaMemWatch (train : TrainOptions) : Nat :=
  train.base.cudaMemWatch

end TrainOptions

/-- Parse GPT-specific flags after runtime/device flags. -/
def parseTrainOptions (opts : Runtime.Autograd.Torch.Options) (args : List String) :
    Except String (TrainOptions × List String) := do
  let defaultSteps : Nat := if opts.useGpu then 100 else 0
  let (base, args) ← Common.parseModelTrainFlags exeName args defaultLogJson defaultSteps 0.001
    (allowZeroSteps := true)
  let (windows?, args) ← CLI.takeNatFlagOnce args "windows"
  let (gen, args) ← text.parseGenerationOptions exeName args
    { prompt := "First Citizen:"
      generate := 64
      temperature := 0.85
      topK := 12
      repeatPenalty := 1.25
      repeatWindow := 24
      seed := 0
      asciiOnly := false }
  let (interactive, args) ← CLI.takeBoolFlagOnce args "interactive"
  let (loadParamsRaw?, args) ← CLI.takeFlagValueOnce args "load-params"
  let (saveParamsRaw?, args) ← CLI.takeFlagValueOnce args "save-params"
  let windows := windows?.getD 128
  Common.requirePositiveNatFlag exeName "windows" windows
  pure ({ base := base
          windows := windows
          prompt := gen.prompt
          generate := gen.generate
          temperature := gen.temperature
          topK := gen.topK
          repeatPenalty := gen.repeatPenalty
          repeatWindow := gen.repeatWindow
          seed := gen.seed
          asciiOnly := gen.asciiOnly
          interactive := interactive
          loadParams? := loadParamsRaw?.map (fun p => (p : System.FilePath))
          saveParams? := saveParamsRaw?.map (fun p => (p : System.FilePath)) }, args)

/-- Byte-token window used for reporting prompt/target text. -/
def tokenWindowIds (input : String) (offset : Nat) : List Nat :=
  text.tokenWindow text.Tokenizer.byte seqLen input (offset := offset) (padId := 32)

/-- Print a compact before/after language-model report for the first batch row. -/
def printPredictionReport (label : String) (input : String) (logits : Tensor Float σ) :
    IO Unit := do
  let predIds := text.argmaxTokenIdsFromBatchLogits (α := Float) logits ⟨0, by decide⟩
  IO.println s!"  {label} pred={text.escapeByteIdsForDisplay predIds}"
  IO.println s!"  prompt={text.escapeByteIdsForDisplay (tokenWindowIds input 0)}"
  IO.println s!"  target={text.escapeByteIdsForDisplay (tokenWindowIds input 1)}"

/-- Convert byte ids into the typed batched one-hot input tensor used for generation. -/
def inputTensorFromIds (ids : List Nat) : Tensor Float σ :=
  let (x2DF, _) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab) ids
    (padId := 32)
  let x : Tensor Float σ := Tensor.dim (fun _bi => x2DF)
  x

/-- Extract the first batch row's vocabulary score vector at one sequence position. -/
def logitsArrayAt (logits : Tensor Float σ) (pos : Nat) : Array Float :=
  let pos : Fin seqLen :=
    ⟨Nat.min pos (seqLen - 1),
      Nat.lt_of_le_of_lt (Nat.min_le_right pos (seqLen - 1)) (by decide)⟩
  match logits with
  | Tensor.dim batches =>
      match batches ⟨0, by decide⟩ with
      | Tensor.dim rows =>
          match rows pos with
          | Tensor.dim cols =>
              Array.ofFn (fun j : Fin vocab =>
                match cols j with
                | Tensor.scalar x => x)

mutual

/-- Autoregressively extend byte token ids using a trained byte-level GPT model. -/
partial def generateSampledFromIds
    (opts : Runtime.Autograd.Torch.Options) (model : nn.Sequential σ τ)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (promptIds : List Nat) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (List Nat) := do
  let gen : text.GenerationOptions :=
    { prompt := ""
      generate := steps
      temperature := temperature
      topK := topK
      repeatPenalty := repeatPenalty
      repeatWindow := repeatWindow
      seed := seed
      asciiOnly := asciiOnly }
  let allowId := if asciiOnly then text.printableAsciiByte else fun _ => true
  let rec loop (ids : List Nat) : Nat → IO (List Nat)
    | 0 => pure ids
    | n + 1 => do
        let generatedSoFar := steps - (n + 1)
        let start := if ids.length > seqLen then ids.length - seqLen else 0
        let window := (ids.drop start).take seqLen
        let predPos := if window.isEmpty then 0 else window.length - 1
        let padded := window ++ List.replicate (seqLen - window.length) 32
        let logits ← nn.eval1NoGrad (α := Float) opts model params (inputTensorFromIds padded)
        let recent :=
          if repeatWindow = 0 then
            []
          else
            ids.drop (ids.length - Nat.min ids.length repeatWindow)
        let tok := text.chooseNextToken (logitsArrayAt logits predPos) gen generatedSoFar recent allowId
        -- Keep the same "space" fallback convention across byte-level text commands.
        let nextTok := if tok < vocab then tok else 32
        loop (ids ++ [nextTok]) n
  loop promptIds steps

/-- Encode a string prompt and autoregressively extend it. -/
partial def generateSampled
    (opts : Runtime.Autograd.Torch.Options) (model : nn.Sequential σ τ)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (prompt : String) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (List Nat) := do
  let init := text.Tokenizer.byte.encode prompt
  generateSampledFromIds opts model params init steps temperature topK seed repeatWindow repeatPenalty asciiOnly

end

/-- Build a finite cyclic training set from corpus text, biased toward the prompt when present. -/
def samplesFromCorpus (input prompt : String) (windows : Nat) :
    Array (API.sample.Supervised Float σ τ) :=
  let toks := text.Tokenizer.byte.encode input
  let promptToks := text.Tokenizer.byte.encode prompt
  let promptOffset? := text.Corpus.findWindow? toks.toArray promptToks.toArray
  let offs := (text.Corpus.promptAwareOffsets toks.length seqLen windows promptOffset?).toArray
  offs.map (fun off =>
    let idsByBatch : Array (List Nat) :=
      Array.ofFn (fun i : Fin batch =>
        let off' := (off + i.val * (seqLen / 2 + 1)) % Nat.max 1 (toks.length - (seqLen + 1))
        text.tokenWindow text.Tokenizer.byte (seqLen + 1) input (offset := off') (padId := 32))
    mkSampleBatchFromTokenIds idsByBatch)

/-- Fallback sample used when a caller passes an empty training-window array. -/
def firstSample (samples : Array (API.sample.Supervised Float σ τ)) :
    API.sample.Supervised Float σ τ :=
  samples.getD 0 (mkSampleFromTokenIds (α := Float) (List.replicate (seqLen + 1) 32))

/-- Mean loss over a bounded deterministic prefix of the training windows. -/
def meanLossOnSamples
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (nn.paramShapes model) [σ, τ])
    (samples : Array (API.sample.Supervised Float σ τ)) : IO Float := do
  -- Reporting loss over every training window would run a separate scalar forward for each window.
  -- A fixed evaluation prefix keeps logs stable without making evaluation dominate training.
  let evalCount := Nat.min samples.size 32
  let mut total := 0.0
  for i in [0:evalCount] do
    let sample := samples.getD i (firstSample samples)
    let loss ← TorchLean.Module.forward (α := Float) m sample
    total := total + Tensor.toScalar loss
  pure (total / Float.ofNat (Nat.max 1 evalCount))

/--
Interactive prompt loop for the in-memory Float model.

Each line is appended to the current byte context, decoded through the trained local model, and then
kept as context for the next prompt unless the user clears it.
-/
partial def interactiveLoopFloat
    (opts : Runtime.Autograd.Torch.Options) (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ])
    (train : TrainOptions) :
    IO Unit := do
  IO.println s!"  interactive: enter text; :q exits, :clear resets, :show prints context (window={seqLen} bytes)"
  let stdin ← IO.getStdin
  let rec loop (ctx : List Nat) : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else if prompt = ":clear" then
      IO.println "  interactive: cleared context"
      loop []
    else if prompt = ":show" then
      IO.println s!"  context={text.escapeByteIdsForDisplay ctx}"
      loop ctx
    else
      let inputIds := ctx ++ text.Tokenizer.byte.encode prompt ++ [10]
      let outIds ←
        generateSampledFromIds opts model m.trainer.params inputIds train.generate
          train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let genOnly := outIds.drop inputIds.length
      IO.println s!"  generated={text.escapeByteIdsForDisplay genOnly}"
      loop outIds
  loop []

/-- Metadata stored beside the before/after loss for byte-level GPT runs. -/
def trainLogNotes (opts : Runtime.Autograd.Torch.Options) (train : TrainOptions)
    (generated : String) (cudaMemWatch? : Option Nat := none) : Array String :=
  #[s!"device={if opts.useGpu then "cuda" else "cpu"}",
    s!"prompt={text.escapeForDisplay train.prompt}",
    s!"generated={generated}",
    s!"windows={train.windows}",
    s!"temperature={train.temperature}", s!"top_k={train.topK}", s!"sample_seed={train.seed}",
    s!"repeat_penalty={train.repeatPenalty}", s!"repeat_window={train.repeatWindow}",
    s!"ascii_only={train.asciiOnly}"] ++
  match cudaMemWatch? with
  | none => #[]
  | some cudaMemWatch => #[s!"cuda_mem_watch={cudaMemWatch}"]

/-- Write the standard byte-level GPT before/after loss log. -/
def writeGpt2TrainLog (opts : Runtime.Autograd.Torch.Options) (train : TrainOptions)
    (loss0 loss1 : Float) (generated : String) (cudaMemWatch? : Option Nat := none) : IO Unit :=
  Common.writeBeforeAfterLossLogTo train.log "GPT-2 byte prompt training" train.steps loss0 loss1
    (trainLogNotes opts train generated cudaMemWatch?)

/-- Save a checkpoint when `--save-params` is present. -/
def saveParamsIfRequested
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ])
    (train : TrainOptions) : IO Unit := do
  match train.saveParams? with
  | none => pure ()
  | some path =>
      TorchLean.ParamIO.saveModuleParamsBits
        (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ]) m path
      IO.println s!"  wrote params: {path}"

/-- Scalar-polymorphic training path for one fixed byte-level sample. -/
def unitTrainSteps {α : Type} [Semantics.Scalar α] [DecidableEq Shape] [ToString α]
    [Runtime.Scalar α] [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (opts : Runtime.Autograd.Torch.Options) (input : String) (steps : Nat) :
    IO (α × α) := do
  nn.withModel mkModel fun model => do
  let sample := mkSample (α := α) (input := input)
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let m ← TorchLean.Module.instantiateWithOptions (α := α) modDef cast opts
    let loss0 ← TorchLean.Module.forward (α := α) m sample
    let L0 := Tensor.toScalar loss0

    if steps = 0 then
      IO.println s!"  steps=0 loss0={L0}"
      pure (L0, L0)
    else
      let opt := TorchLean.Optim.adam (α := α)
        (paramShapes := nn.paramShapes model)
        (lr := Runtime.ofFloat (α := α) 1e-4)
        (beta1 := Runtime.ofFloat (α := α) 0.9)
        (beta2 := Runtime.ofFloat (α := α) 0.999)
        (epsilon := Runtime.ofFloat (α := α) 1e-8)
      let optH ← TorchLean.Optim.handle (α := α) m opt
      for _ in [0:steps] do
        optH.step sample

      let loss1 ← TorchLean.Module.forward (α := α) m sample
      let L1 := Tensor.toScalar loss1
      Common.printFitReport steps
        ({ before := L0, after := L1 } : TorchLean.Trainer.FitReport α)
      pure (L0, L1)

/--
Float-specialized training path with decoded prediction reports.

The CUDA executable uses Lean `Float` tensors, so this branch can show actual prompt,
target, and predicted text before and after training. The polymorphic path above remains useful for
checking the same training loop over other scalar backends.
-/
def unitTrainStepsFloat (opts : Runtime.Autograd.Torch.Options) (input : String)
    (train : TrainOptions) : IO (Float × Float × String) := do
  nn.withModel mkModel fun model => do
    let samples := samplesFromCorpus input train.prompt train.windows
    let reportSample := mkSample (α := Float) (input := train.prompt)
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    match train.loadParams? with
    | none => pure ()
    | some path =>
        TorchLean.ParamIO.loadModuleParamsBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ])
          m path
    let logits0 ← nn.eval1 (α := Float) opts model
      m.trainer.params
      (NN.API.sample.x reportSample)
    printPredictionReport "before" train.prompt logits0
    let L0 ← meanLossOnSamples model m samples

    if train.steps = 0 then
      IO.println s!"  steps=0 loss0={L0}"
      if train.interactive then
        interactiveLoopFloat opts model m train
      let generatedIds ← generateSampled opts model m.trainer.params train.prompt train.generate
        train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let generated := text.escapeByteIdsForDisplay generatedIds
      writeGpt2TrainLog opts train L0 L0 generated
      saveParamsIfRequested model m train
      pure (L0, L0, generated)
    else
      let opt := TorchLean.Optim.adam (α := Float)
        (paramShapes := nn.paramShapes model)
        (lr := train.lr)
        (beta1 := 0.9)
        (beta2 := 0.999)
        (epsilon := 1e-8)
      let optH ← TorchLean.Optim.handle (α := Float) m opt
      let cudaMemWatch := Common.effectiveCudaMemWatch opts train.steps train.cudaMemWatch
      let mut memWatch? ← Common.reportCudaMemWatch opts cudaMemWatch train.steps 0 none
      for step in [0:train.steps] do
        let sample := samples.getD (step % Nat.max 1 samples.size) (firstSample samples)
        optH.step sample
        memWatch? ← Common.reportCudaMemWatch opts cudaMemWatch train.steps (step + 1) memWatch?

      let L1 ← meanLossOnSamples model m samples
      let logits1 ← nn.eval1 (α := Float) opts model
        m.trainer.params
        (NN.API.sample.x reportSample)
      printPredictionReport "after " train.prompt logits1
      let generatedIds ← generateSampled opts model m.trainer.params train.prompt train.generate
        train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let generated := text.escapeByteIdsForDisplay generatedIds
      IO.println s!"  generated={generated}"
      IO.println s!"  corpus_bytes={input.toByteArray.size} windows={samples.size}"
      IO.println s!"  steps={train.steps} lr={train.lr} loss0={L0} loss1={L1}"
      IO.println s!"  sampling=top_k({train.topK}), temperature={train.temperature}, seed={train.seed}"
      IO.println s!"  repetition_penalty={train.repeatPenalty} repeat_window={train.repeatWindow}"
      if train.interactive then
        interactiveLoopFloat opts model m train
      writeGpt2TrainLog opts train L0 L1 generated (some cudaMemWatch)
      saveParamsIfRequested model m train
      pure (L0, L1, generated)

/-- CLI entrypoint for byte-level GPT training, sampling, logging, and checkpointing. -/
def main (args : List String) : IO UInt32 := do
  Common.runAnyOrFloat exeName args
    (preferFloat := fun args => args.contains "--cuda" || CLI.hasFlagValue args "log")
    (banner := fun opts =>
      s!"{exeName}: causal LM training (device={if opts.useGpu then "cuda" else "cpu"})")
    (anyK := fun {α} _ _ _ _ cast opts rest => do
      let (input, rest) ← takeInputText rest
      let (train, rest) ← Common.orThrow exeName <| parseTrainOptions opts rest
      Common.orThrow exeName <| CLI.requireNoArgs rest
      if train.interactive then
        throw <| IO.userError s!"{exeName}: --interactive is supported only by the Float/CUDA path"
      let _ ← unitTrainSteps (α := α) cast opts input (steps := train.steps)
      pure ())
    (floatK := fun opts rest => do
      let (input, rest) ← takeInputText rest
      let (train, rest) ← Common.orThrow exeName <| parseTrainOptions opts rest
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let (_L0, _L1, _generated) ← unitTrainStepsFloat opts input train)

end NN.Examples.Models.Sequence.Gpt2
