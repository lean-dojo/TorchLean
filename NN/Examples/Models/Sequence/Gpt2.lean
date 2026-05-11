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
  mkdir -p data/real/text
  curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -o data/real/text/tiny_shakespeare.txt
  lake exe torchlean gpt2 --cuda --data-file data/real/text/tiny_shakespeare.txt --steps 100
  lake exe torchlean gpt2 --cuda --fast-kernels --data-file data/real/text/tiny_shakespeare.txt \
    --steps 100

This is a small GPT2-style *causal* language-model walkthrough (byte-level tokens).

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

/-!
# GPT-2-Style Causal Language Model Example

Runnable `torchlean gpt2` example. It builds a small GPT-2-style causal transformer over
byte-level tokens, with optional real text input from tiny-shakespeare or `--data-file PATH`.

If you are looking for the simplest "Karpathy-style single text file" path, start with
`torchlean chargpt` (character-level tokenizer). This `gpt2` example is byte-level and is meant to
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

def exeName : String := "torchlean gpt2"
def defaultLogJson : System.FilePath := "data/model_zoo/gpt2_trainlog.json"

/--
Small batch size.

The executable intentionally overfits a small real-text slice rather than presenting it as a full pretraining run: it shows the
full TorchLean stack can run a causal Transformer, update parameters, and decode logits back to
text.
-/
def batch : Nat := 2

/--
Prompt/target window length.

Sixty-four byte tokens is still small enough for local eager-CUDA runs, but it gives the miniature
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

We keep the default small so the tutorial finishes locally. A wider `dModel = 64` variant runs, but
in the current eager-CUDA training loop it is slower and did not improve the 2k-step Shakespeare
sample enough to justify making it the default. Use this file to inspect Transformer/autograd
behavior; use the Mamba example when the goal is the cleanest compact text sample.
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

/-- Conventional local path for the Tiny Shakespeare text corpus. -/
def tinyShakespearePath : System.FilePath :=
  "data/real/text/tiny_shakespeare.txt"

/-- Conventional local path for the TinyStories validation slice. -/
def tinyStoriesValidPath : System.FilePath :=
  "data/real/text/tinystories_valid.txt"

/-- Shared data-preparation hint for the GPT text examples. -/
def missingTextHint : String :=
  "Download Tiny Shakespeare with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --tiny-shakespeare\n" ++
  "For TinyStories (valid split):\n" ++
  "  python3 scripts/datasets/download_example_data.py --tinystories-valid"

abbrev σ : Shape :=
  shape![batch, seqLen, vocab]

abbrev τ : Shape :=
  σ

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot
    { batch := batch
      seqLen := seqLen
      vocab := vocab
      numHeads := numHeads
      headDim := headDim
      ffnHidden := ffnHidden
      layers := layers }

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
  text.Corpus.takeUtf8Input exeName tinyShakespearePath
    [("--tiny-shakespeare", tinyShakespearePath), ("--tinystories-valid", tinyStoriesValidPath)]
    missingTextHint args

structure TrainOptions where
  base : Common.ModelTrainFlags
  windows : Nat
  prompt : String
  generate : Nat
  temperature : Float
  topK : Nat
  repeatPenalty : Float
  repeatWindow : Nat
  seed : Nat
  asciiOnly : Bool
  interactive : Bool
  loadParams? : Option System.FilePath
  saveParams? : Option System.FilePath
deriving Repr

namespace TrainOptions

def steps (train : TrainOptions) : Nat :=
  train.base.train.steps

def lr (train : TrainOptions) : Float :=
  train.base.lr

def log (train : TrainOptions) : _root_.Runtime.Training.LogDestination :=
  train.base.train.log

def logPath (train : TrainOptions) : System.FilePath :=
  train.base.train.logPath

end TrainOptions

def parseTrainOptions (opts : Runtime.Autograd.Torch.Options) (args : List String) :
    Except String (TrainOptions × List String) := do
  let defaultSteps : Nat := if opts.useGpu then 100 else 0
  let (base, args) ← Common.parseModelTrainFlags exeName args defaultLogJson defaultSteps 0.001
    (allowZeroSteps := true)
  let (windows?, args) ← CLI.takeNatFlagOnce args "windows"
  let (prompt?, args) ← CLI.takeFlagValueOnce args "prompt"
  let (generate?, args) ← CLI.takeNatFlagOnce args "generate"
  let (temperature?, args) ← CLI.takeFloatFlagOnce args "temperature"
  let (topK?, args) ← CLI.takeNatFlagOnce args "top-k"
  let (repeatPenalty?, args) ← CLI.takeFloatFlagOnce args "repeat-penalty"
  let (repeatWindow?, args) ← CLI.takeNatFlagOnce args "repeat-window"
  let (seed?, args) ← CLI.takeNatFlagOnce args "sample-seed"
  let (asciiOnlyRaw?, args) ← CLI.takeFlagValueOnce args "ascii-only"
  let (asciiOnlyFlag, args) ← CLI.takeBoolFlagOnce args "ascii-only"
  let asciiOnly :=
    match asciiOnlyRaw? with
    | none => asciiOnlyFlag
    | some v =>
        if v = "true" || v = "1" then true
        else if v = "false" || v = "0" then false
        else
          -- Keep the parser total; validation happens below.
          asciiOnlyFlag
  let (interactive, args) ← CLI.takeBoolFlagOnce args "interactive"
  let (loadParamsRaw?, args) ← CLI.takeFlagValueOnce args "load-params"
  let (saveParamsRaw?, args) ← CLI.takeFlagValueOnce args "save-params"
  let temperature := temperature?.getD 0.85
  if temperature <= 0.0 then
    throw s!"{exeName}: --temperature must be > 0"
  let repeatPenalty := repeatPenalty?.getD 1.25
  if repeatPenalty < 0.0 then
    throw s!"{exeName}: --repeat-penalty must be >= 0"
  let windows := windows?.getD 128
  if windows = 0 then
    throw s!"{exeName}: --windows must be > 0"
  match asciiOnlyRaw? with
  | none => pure ()
  | some v =>
      if v = "true" || v = "1" || v = "false" || v = "0" then
        pure ()
      else
        throw s!"{exeName}: --ascii-only expects true/false (or 1/0), got {v}"
  pure ({ base := base
          windows := windows
          prompt := prompt?.getD "First Citizen:"
          generate := generate?.getD 64
          temperature := temperature
          topK := topK?.getD 12
          repeatPenalty := repeatPenalty
          repeatWindow := repeatWindow?.getD 24
          seed := seed?.getD 0
          asciiOnly := asciiOnly
          interactive := interactive
          loadParams? := loadParamsRaw?.map (fun p => (p : System.FilePath))
          saveParams? := saveParamsRaw?.map (fun p => (p : System.FilePath)) }, args)

def escapeByte (b : Nat) : String :=
  if b = 10 then "\\n"
  else if b = 9 then "\\t"
  else if b = 34 then "\\\""
  else if b = 92 then "\\\\"
  else if 32 ≤ b && b ≤ 126 then
    String.singleton (Char.ofNat b)
  else
    let hex : Array Char := #['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    let hi := (b / 16) % 16
    let lo := b % 16
    "\\x" ++ String.singleton (hex.getD hi '0') ++ String.singleton (hex.getD lo '0')

def escapeByteIdsForDisplay (ids : List Nat) : String :=
  "\"" ++ String.join (ids.map escapeByte) ++ "\""

def tokenWindowIds (input : String) (offset : Nat) : List Nat :=
  text.tokenWindow text.Tokenizer.byte seqLen input (offset := offset) (padId := 32)

/-- Print a compact before/after language-model probe for the first batch row. -/
def printPredictionProbe (label : String) (input : String) (logits : Tensor Float σ) :
    IO Unit := do
  let predIds := text.argmaxTokenIdsFromBatchLogits (α := Float) logits ⟨0, by decide⟩
  IO.println s!"  {label} pred={escapeByteIdsForDisplay predIds}"
  IO.println s!"  prompt={escapeByteIdsForDisplay (tokenWindowIds input 0)}"
  IO.println s!"  target={escapeByteIdsForDisplay (tokenWindowIds input 1)}"

def inputTensorFromIds (ids : List Nat) : Tensor Float σ :=
  let (x2DF, _) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab) ids
    (padId := 32)
  let x : Tensor Float σ := Tensor.dim (fun _bi => x2DF)
  x

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

/--
Apply a lightweight repetition penalty during decoding.

This is intentionally a generation-side control, not a training shortcut. This compact GPT-2-style example can
learn the local next-token objective but still fall into byte-level loops such as `oooooo`; reducing
the logits of recently emitted bytes makes the example's sampled text reflect more of the learned
distribution instead of the first local attractor it finds.
-/
def penalizedScores (scores : Array Float) (recent : List Nat) (repeatPenalty : Float) : Array Float :=
  text.penalizeRepeats scores recent repeatPenalty

def asciiAllowed (i : Nat) : Bool :=
  i = 10 || (32 ≤ i && i ≤ 126)

def restrictToAscii (scores : Array Float) : Array Float :=
  text.restrictScores scores asciiAllowed

def greedyTokenAt (logits : Tensor Float σ) (pos : Nat)
    (recent : List Nat := []) (repeatPenalty : Float := 0.0) (asciiOnly : Bool := false) : Nat :=
  let base := penalizedScores (logitsArrayAt logits pos) recent repeatPenalty
  let scores := if asciiOnly then restrictToAscii base else base
  (text.topKIndices scores 1).head?.getD 32

def sampleFromLogitsAt (logits : Tensor Float σ) (pos : Nat)
    (temperature : Float) (topK seed counter : Nat)
    (recent : List Nat := []) (repeatPenalty : Float := 0.0) (asciiOnly : Bool := false) : Nat := Id.run do
  let base := penalizedScores (logitsArrayAt logits pos) recent repeatPenalty
  let scores := if asciiOnly then restrictToAscii base else base
  let tok := text.sampleTopKIndex scores temperature topK seed counter
  -- Keep the same "space" fallback convention as the byte-level demos.
  if tok < vocab then tok else 32

mutual

partial def generateSampledFromIds
    (opts : Runtime.Autograd.Torch.Options) (model : nn.Sequential σ τ)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (promptIds : List Nat) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (List Nat) := do
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
        let nextTok :=
          if topK = 1 then
            greedyTokenAt logits predPos recent repeatPenalty (asciiOnly := asciiOnly)
          else
            sampleFromLogitsAt logits predPos temperature topK seed generatedSoFar recent repeatPenalty
              (asciiOnly := asciiOnly)
        loop (ids ++ [nextTok]) n
  loop promptIds steps

partial def generateSampled
    (opts : Runtime.Autograd.Torch.Options) (model : nn.Sequential σ τ)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (prompt : String) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (List Nat) := do
  let init := text.Tokenizer.byte.encode prompt
  generateSampledFromIds opts model params init steps temperature topK seed repeatWindow repeatPenalty asciiOnly

end

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

def firstSample (samples : Array (API.sample.Supervised Float σ τ)) :
    API.sample.Supervised Float σ τ :=
  samples.getD 0 (mkSampleFromTokenIds (α := Float) (List.replicate (seqLen + 1) 32))

def meanLossOnSamples
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (nn.paramShapes model) [σ, τ])
    (samples : Array (API.sample.Supervised Float σ τ)) : IO Float := do
  -- Reporting loss over every training window is surprisingly expensive on eager CUDA because each
  -- scalar forward builds temporary tape state. A fixed probe subset keeps the metric stable enough
  -- for example logs without turning evaluation into the memory bottleneck.
  let evalCount := Nat.min samples.size 32
  let mut total := 0.0
  for i in [0:evalCount] do
    let sample := samples.getD i (firstSample samples)
    let loss ← TorchLean.Module.forward (α := Float) m sample
    total := total + Tensor.toScalar loss
  pure (total / Float.ofNat (Nat.max 1 evalCount))

/--
Compact interactive prompt loop for the in-memory Float model.

This is a diagnostic REPL, not pretrained text generation. Each line is interpreted as one causal LM
window, and the model prints the per-position argmax prediction for that window.
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
      IO.println s!"  context={escapeByteIdsForDisplay ctx}"
      loop ctx
    else
      let inputIds := ctx ++ text.Tokenizer.byte.encode prompt ++ [10]
      let outIds ←
        generateSampledFromIds opts model m.trainer.params inputIds train.generate
          train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let genOnly := outIds.drop inputIds.length
      IO.println s!"  generated={escapeByteIdsForDisplay genOnly}"
      loop outIds
  loop []

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
      IO.println s!"  steps={steps} loss0={L0} loss1={L1}"
      pure (L0, L1)

/--
Float-specialized training path with decoded prediction probes.

The CUDA executable uses Lean `Float` tensors, so this branch can show actual prompt,
target, and predicted text before and after training. The polymorphic path above is still used for
non-Float dtype smoke runs.
-/
def unitTrainStepsFloat (opts : Runtime.Autograd.Torch.Options) (input : String)
    (train : TrainOptions) : IO (Float × Float × String) := do
  nn.withModel mkModel fun model => do
    let samples := samplesFromCorpus input train.prompt train.windows
    let probeSample := mkSample (α := Float) (input := train.prompt)
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    match train.loadParams? with
    | none => pure ()
    | some path =>
        TorchLean.ParamIO.loadModuleParamsBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ])
          m path
    let logits0 ← nn.eval1 (α := Float) opts model
      m.trainer.params
      (NN.API.sample.x probeSample)
    printPredictionProbe "before" train.prompt logits0
    let L0 ← meanLossOnSamples model m samples

    if train.steps = 0 then
      IO.println s!"  steps=0 loss0={L0}"
      if train.interactive then
        interactiveLoopFloat opts model m train
      let generatedIds ← generateSampled opts model m.trainer.params train.prompt train.generate
        train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let generated := escapeByteIdsForDisplay generatedIds
      Common.writeBeforeAfterLossLogTo train.log "GPT-2 byte prompt training" train.steps L0 L0
        #[s!"device={if opts.useGpu then "cuda" else "cpu"}",
          s!"prompt={text.escapeForDisplay train.prompt}",
          s!"generated={generated}",
          s!"windows={train.windows}",
          s!"temperature={train.temperature}", s!"top_k={train.topK}", s!"sample_seed={train.seed}",
          s!"repeat_penalty={train.repeatPenalty}", s!"repeat_window={train.repeatWindow}",
          s!"ascii_only={train.asciiOnly}"]
      match train.saveParams? with
      | none => pure ()
      | some path =>
          TorchLean.ParamIO.saveModuleParamsBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ])
            m path
          IO.println s!"  wrote params: {path}"
      pure (L0, L0, generated)
    else
      let opt := TorchLean.Optim.adam (α := Float)
        (paramShapes := nn.paramShapes model)
        (lr := train.lr)
        (beta1 := 0.9)
        (beta2 := 0.999)
        (epsilon := 1e-8)
      let optH ← TorchLean.Optim.handle (α := Float) m opt
      for step in [0:train.steps] do
        let sample := samples.getD (step % Nat.max 1 samples.size) (firstSample samples)
        optH.step sample

      let L1 ← meanLossOnSamples model m samples
      let logits1 ← nn.eval1 (α := Float) opts model
        m.trainer.params
        (NN.API.sample.x probeSample)
      printPredictionProbe "after " train.prompt logits1
      let generatedIds ← generateSampled opts model m.trainer.params train.prompt train.generate
        train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let generated := escapeByteIdsForDisplay generatedIds
      IO.println s!"  generated={generated}"
      IO.println s!"  corpus_bytes={input.toByteArray.size} windows={samples.size}"
      IO.println s!"  steps={train.steps} lr={train.lr} loss0={L0} loss1={L1}"
      IO.println s!"  sampling=top_k({train.topK}), temperature={train.temperature}, seed={train.seed}"
      IO.println s!"  repetition_penalty={train.repeatPenalty} repeat_window={train.repeatWindow}"
      if train.interactive then
        interactiveLoopFloat opts model m train
      Common.writeBeforeAfterLossLogTo train.log "GPT-2 byte prompt training" train.steps L0 L1
        #[s!"device={if opts.useGpu then "cuda" else "cpu"}",
          s!"prompt={text.escapeForDisplay train.prompt}",
          s!"generated={generated}",
          s!"windows={train.windows}",
          s!"temperature={train.temperature}", s!"top_k={train.topK}", s!"sample_seed={train.seed}",
          s!"repeat_penalty={train.repeatPenalty}", s!"repeat_window={train.repeatWindow}",
          s!"ascii_only={train.asciiOnly}"]
      match train.saveParams? with
      | none => pure ()
      | some path =>
          TorchLean.ParamIO.saveModuleParamsBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ])
            m path
          IO.println s!"  wrote params: {path}"
      pure (L0, L1, generated)

def main (args : List String) : IO UInt32 := do
  if args.contains "--cuda" || CLI.hasFlagValue args "log" then
    TorchLean.Module.run exeName args
      (.float (fun opts rest => do
        let (input, rest) ← takeInputText rest
        let (train, rest) ← Common.orThrow exeName <| parseTrainOptions opts rest
        Common.orThrow exeName <| CLI.requireNoArgs rest
        let (_L0, _L1, _generated) ← unitTrainStepsFloat opts input train
      ))
      { banner? := some (fun opts =>
          s!"{exeName}: causal LM training (device={if opts.useGpu then "cuda" else "cpu"})")
        printOk := true }
  else
    TorchLean.Module.run exeName args
      (.any (fun {α} _ _ _ _ cast opts rest => do
        let (input, rest) ← takeInputText rest
        let (train, rest) ← Common.orThrow exeName <| parseTrainOptions opts rest
        Common.orThrow exeName <| CLI.requireNoArgs rest
        if train.interactive then
          throw <| IO.userError s!"{exeName}: --interactive is supported only by the Float/CUDA path"
        let _ ← unitTrainSteps (α := α) cast opts input (steps := train.steps)
        pure ()
      ))
      { banner? := some (fun opts =>
          s!"{exeName}: causal LM training (device={if opts.useGpu then "cuda" else "cpu"})")
        printOk := true }

end NN.Examples.Models.Sequence.Gpt2
