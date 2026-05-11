/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Models.Mamba
public import NN.Examples.Models.Common.RealData

/-!
# Mamba Text Training

Runnable byte-level language-model training with the public Mamba API constructor.

The model is trainable end-to-end:

`mamba(seqLen, vocab, stateDim) → linear(stateDim → vocab)`

and the same code runs on CPU or CUDA through TorchLean autograd.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake exe -K cuda=true torchlean mamba --cuda --tiny-shakespeare --steps 300 --windows 128 \
  --temperature 0.85 --top-k 12 --sample-seed 7
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.Mamba

def exeName : String := "torchlean mamba"
def defaultLogJson : System.FilePath := "data/model_zoo/mamba_trainlog.json"

/--
Training/generation context length in byte tokens.

Mamba scales more gently with sequence length than attention, so the tutorial uses a 64-byte
window. That is long enough to carry speaker tags and short phrases from Tiny Shakespeare while
remaining fast in eager CUDA.
-/
def seqLen : Nat := 64
def tokenizer : text.Tokenizer := text.Tokenizer.byte

def cfg : nn.models.MambaTextConfig :=
  { vocab := tokenizer.vocabSize
    -- A wider `128/12` variant learns too, but eager CUDA tape memory can dominate post-training
    -- generation. The default state size balances memory use with representative generation.
    stateDim := 96
    ssmStateDim := 8
    convWidth := 3 }

abbrev σ : Shape := nn.models.mambaTokenMat cfg seqLen
abbrev τ : Shape := nn.models.mambaLogitMat cfg seqLen

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.mambaTextLm cfg seqLen

structure TrainOptions where
  base : Common.ModelTrainFlags
  windows : Nat
  prompt : String
  generate : Nat
  temperature : Float
  topK : Nat
  seed : Nat
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

def parseTrainOptions (args : List String) : Except String (TrainOptions × List String) := do
  let (base, args) ← Common.parseModelTrainFlags exeName args defaultLogJson 200 0.002
  let (windows?, args) ← CLI.takeNatFlagOnce args "windows"
  let (prompt?, args) ← CLI.takeFlagValueOnce args "prompt"
  let (generate?, args) ← CLI.takeNatFlagOnce args "generate"
  let (temperature?, args) ← CLI.takeFloatFlagOnce args "temperature"
  let (topK?, args) ← CLI.takeNatFlagOnce args "top-k"
  let (seed?, args) ← CLI.takeNatFlagOnce args "sample-seed"
  let windows := windows?.getD 128
  if windows = 0 then
    throw s!"{exeName}: --windows must be > 0"
  let temperature := temperature?.getD 0.9
  if temperature <= 0.0 then
    throw s!"{exeName}: --temperature must be > 0"
  pure ({ base := base
          windows := windows
          prompt := prompt?.getD "First Citizen:"
          generate := generate?.getD 48
          temperature := temperature
          topK := topK?.getD 16
          seed := seed?.getD 0 }, args)

def castTensor {s : Shape} (t : Tensor Float s) : Tensor Float s :=
  t

def sampleFromTokenIds (ids : List Nat) : API.sample.Supervised Float σ τ :=
  let (xF, yF) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := cfg.vocab) ids
  API.sample.mk (castTensor xF) (castTensor yF)

def samplesFromCorpus (input prompt : String) (windows : Nat) :
    Array (API.sample.Supervised Float σ τ) :=
  let toks := tokenizer.encode input
  let promptToks := tokenizer.encode prompt
  let promptOffset? := text.Corpus.findWindow? toks.toArray promptToks.toArray
  let offsets :=
    match promptOffset? with
    | none => nn.models.mambaTrainingOffsets toks.length seqLen windows
    | some _ => text.Corpus.promptAwareOffsets toks.length seqLen windows promptOffset?
  offsets.toArray.map (fun off =>
    -- This Mamba example is single-sequence rather than batch-first, but it still uses the same text
    -- window contract: corpus tokens are sliced into `(seqLen + 1)` next-token windows, then turned
    -- into one-hot `(x, shifted y)` samples.  The public GPT helpers live in `NN.API.Text`; this
    -- unbatched model keeps the direct matrix sample until we add a batch-first Mamba command.
    let ids := text.tokenWindow tokenizer (seqLen + 1) input (offset := off) (padId := 32)
    sampleFromTokenIds ids)

def firstSample (samples : Array (API.sample.Supervised Float σ τ)) :
    API.sample.Supervised Float σ τ :=
  samples.getD 0 (sampleFromTokenIds (List.replicate (seqLen + 1) 32))

def printPredictionProbe (label prompt : String) (logits : Tensor Float τ) : IO Unit := do
  IO.println s!"  {label} pred={text.escapeForDisplay (text.decodeArgmaxLogits tokenizer logits)}"
  IO.println s!"  prompt={text.escapeForDisplay (text.decodeWindow tokenizer seqLen prompt (padId := 32))}"
  IO.println s!"  target={text.escapeForDisplay (text.decodeWindow tokenizer seqLen prompt (offset := 1) (padId := 32))}"

def inputTensorFromIds (ids : List Nat) : Tensor Float σ :=
  let (xF, _) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := cfg.vocab) ids
  xF

def logitsArrayAt (logits : Tensor Float τ) (pos : Nat) : Array Float :=
  let pos : Fin seqLen :=
    ⟨Nat.min pos (seqLen - 1),
      Nat.lt_of_le_of_lt (Nat.min_le_right pos (seqLen - 1)) (by decide)⟩
  match logits with
  | Tensor.dim rows =>
      match rows pos with
      | Tensor.dim cols =>
          Array.ofFn (fun j : Fin cfg.vocab =>
            match cols j with
            | Tensor.scalar x => x)

def greedyTokenAt (logits : Tensor Float τ) (pos : Nat) : Nat :=
  let scores := logitsArrayAt logits pos
  (text.topKIndices scores 1).head?.getD 32

def sampleFromLogitsAt (logits : Tensor Float τ) (pos : Nat)
    (temperature : Float) (topK seed counter : Nat) : Nat := Id.run do
  let scores := logitsArrayAt logits pos
  text.sampleTopKIndex scores temperature topK seed counter

partial def generateSampled
    (opts : Runtime.Autograd.Torch.Options) (model : nn.Sequential σ τ)
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (prompt : String) (steps : Nat) (temperature : Float) (topK seed : Nat) : IO String := do
  let rec loop (ids : List Nat) : Nat → IO (List Nat)
    | 0 => pure ids
    | n + 1 => do
        let generatedSoFar := steps - (n + 1)
        let start := if ids.length > seqLen then ids.length - seqLen else 0
        let window := (ids.drop start).take seqLen
        let predPos := if window.isEmpty then 0 else window.length - 1
        let padded := window ++ List.replicate (seqLen - window.length) 32
        let logits ← nn.eval1NoGrad (α := Float) opts model params (inputTensorFromIds padded)
        let nextTok :=
          if topK = 1 then
            greedyTokenAt logits predPos
          else
            sampleFromLogitsAt logits predPos temperature topK seed generatedSoFar
        loop (ids ++ [nextTok]) n
  let ids ← loop (tokenizer.encode prompt) steps
  pure (tokenizer.decode ids)

def meanLossOnSamples
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (nn.paramShapes model) [σ, τ])
    (samples : Array (API.sample.Supervised Float σ τ)) : IO Float := do
  -- Keep reporting cheap and memory-bounded. The trainer can cycle over hundreds of windows, but a
  -- full eager-CUDA scalar forward over every window is not needed just to print a representative
  -- curve point.
  let evalCount := Nat.min samples.size 32
  let mut total := 0.0
  for i in [0:evalCount] do
    let sample := samples.getD i (firstSample samples)
    let loss ← TorchLean.Module.forward (α := Float) m sample
    total := total + Tensor.toScalar loss
  pure (total / Float.ofNat (Nat.max 1 evalCount))

def trainOnText (opts : Runtime.Autograd.Torch.Options) (input : String) (train : TrainOptions) :
    IO (Float × Float) := do
  nn.withModel mkModel fun model => do
    let samples := samplesFromCorpus input train.prompt train.windows
    let probeSample := sampleFromTokenIds (text.tokenWindow tokenizer (seqLen + 1) train.prompt
      (padId := 32))
    let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts

    let logits0 ← nn.eval1NoGrad (α := Float) opts model m.trainer.params
      (NN.API.sample.x probeSample)
    printPredictionProbe "before" train.prompt logits0
    let L0 ← meanLossOnSamples model m samples

    let opt := TorchLean.Optim.adam (α := Float)
      (paramShapes := nn.paramShapes model)
      (lr := train.lr)
      (beta1 := 0.9)
      (beta2 := 0.999)
      (epsilon := 1e-8)
    let optH ← TorchLean.Optim.handle (α := Float) m opt
    if train.steps > 0 then
      for step in [0:train.steps] do
        let sample := samples.getD (step % Nat.max 1 samples.size) (firstSample samples)
        optH.step sample

    let L1 ← meanLossOnSamples model m samples
    let logits1 ← nn.eval1NoGrad (α := Float) opts model m.trainer.params
      (NN.API.sample.x probeSample)
    printPredictionProbe "after " train.prompt logits1
    let generated ← generateSampled opts model m.trainer.params train.prompt train.generate
      train.temperature train.topK train.seed
    IO.println s!"  generated={text.escapeForDisplay generated}"
    IO.println s!"  corpus_bytes={input.toByteArray.size} windows={samples.size}"
    IO.println s!"  steps={train.steps} lr={train.lr} loss0={L0} loss1={L1}"
    IO.println s!"  sampling=top_k({train.topK}), temperature={train.temperature}, seed={train.seed}"
    pure (L0, L1)

def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (path, rest) ← Common.orThrow exeName <| RealData.parseTextFlags rest
      let (train, rest) ← Common.orThrow exeName <| parseTrainOptions rest
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let input ← RealData.readTextCorpus exeName path
      let (L0, L1) ← trainOnText opts input train
      Common.writeBeforeAfterLossLogTo train.log "Mamba text training"
        train.steps L0 L1
        #[s!"data={path}", s!"device={if opts.useGpu then "cuda" else "cpu"}",
          s!"windows={train.windows}", s!"lr={train.lr}",
          s!"prompt={text.escapeForDisplay train.prompt}", s!"generate={train.generate}",
          s!"temperature={train.temperature}", s!"top_k={train.topK}", s!"sample_seed={train.seed}"]
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: Mamba text training (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Sequence.Mamba
