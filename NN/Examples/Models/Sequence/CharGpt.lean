/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Models.Gpt2
public import NN.Runtime.Autograd.TorchLean.NN
public import NN.API.Runtime

/-!
# Char-GPT (minGPT-style) Example

This example mirrors the classic "character-level GPT on a single text file" walkthrough popularized
by Andrej Karpathy's minGPT/nanoGPT teaching material:

- build an alphabet (`itos`) from the training text,
- build a `stoi` tokenizer from that alphabet,
- train a small causal Transformer to predict the next character,
- sample text continuations from a prompt.

It uses TorchLean's one-hot token interface (`batch × seqLen × vocab`) so the whole demo stays in
the same typed tensor world as the rest of the codebase.

Implementation note: training draws a fresh deterministic random window each step (minGPT-style
`get_batch`). The `--windows` flag is still accepted for compatibility, but it no longer controls
how many windows are precomputed.

```bash
lake build -R -K cuda=true torchlean:exe
lake exe torchlean chargpt --cuda --tiny-shakespeare --steps 500 \
  --prompt \"First Citizen:\" --generate 200
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.CharGpt

def exeName : String := "torchlean chargpt"

def tinyShakespearePath : System.FilePath :=
  "data/real/text/tiny_shakespeare.txt"

def missingTextHint : String :=
  "Download Tiny Shakespeare with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --tiny-shakespeare"

def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName tinyShakespearePath
    [("--tiny-shakespeare", tinyShakespearePath)] missingTextHint args

def buildAlphabet (s : String) : Array Char :=
  let chars : List Char := s.toList.eraseDups
  -- Deterministic order: sort by codepoint.
  let sorted := List.mergeSort chars (fun a b => decide (a.toNat ≤ b.toNat))
  sorted.toArray

structure TrainOptions where
  base : Common.ModelTrainFlags
  batch : Nat
  seqLen : Nat
  /--
  Accepted for CLI compatibility with fixed-window trainer entrypoints.

  CharGPT draws a fresh random window each step (minGPT-style), so training does not depend on
  precomputing a fixed `windows` array. We still accept `--windows` so scripts don't break.
  -/
  windows : Nat
  prompt : String
  generate : Nat
  temperature : Float
  topK : Nat
  repeatPenalty : Float
  repeatWindow : Nat
  seed : Nat
  asciiOnly : Bool
  loadParams? : Option System.FilePath
  saveParams? : Option System.FilePath
deriving Repr

def defaultLogJson : System.FilePath := "data/model_zoo/chargpt_trainlog.json"

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
  let defaultSteps : Nat := if opts.useGpu then 500 else 0
  let (base, args) ← Common.parseModelTrainFlags exeName args defaultLogJson defaultSteps 0.0005
    (allowZeroSteps := true)
  let (batch?, args) ← CLI.takeNatFlagOnce args "batch"
  let (seqLen?, args) ← CLI.takeNatFlagOnce args "seq-len"
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
        else asciiOnlyFlag
  let (loadParamsRaw?, args) ← CLI.takeFlagValueOnce args "load-params"
  let (saveParamsRaw?, args) ← CLI.takeFlagValueOnce args "save-params"
  let temperature := temperature?.getD 0.9
  if temperature <= 0.0 then
    throw s!"{exeName}: --temperature must be > 0"
  let repeatPenalty := repeatPenalty?.getD 1.15
  if repeatPenalty < 0.0 then
    throw s!"{exeName}: --repeat-penalty must be >= 0"
  let batch := batch?.getD 4
  if batch = 0 then
    throw s!"{exeName}: --batch must be > 0"
  let seqLen := seqLen?.getD 128
  if seqLen = 0 then
    throw s!"{exeName}: --seq-len must be > 0"
  let windows := windows?.getD 256
  if windows = 0 then
    throw s!"{exeName}: --windows must be > 0"
  match asciiOnlyRaw? with
  | none => pure ()
  | some v =>
      if v = "true" || v = "1" || v = "false" || v = "0" then pure ()
      else throw s!"{exeName}: --ascii-only expects true/false (or 1/0), got {v}"
  pure ({ base := base
          batch := batch
          seqLen := seqLen
          windows := windows
          prompt := prompt?.getD "First Citizen:"
          generate := generate?.getD 200
          temperature := temperature
          topK := topK?.getD 12
          repeatPenalty := repeatPenalty
          repeatWindow := repeatWindow?.getD 64
          seed := seed?.getD 7
          asciiOnly := asciiOnly
          loadParams? := loadParamsRaw?.map (fun p => (p : System.FilePath))
          saveParams? := saveParamsRaw?.map (fun p => (p : System.FilePath)) }, args)

def escapeCharIdsForDisplay (t : text.Tokenizer) (ids : List Nat) : String :=
  text.escapeForDisplay (t.decode ids)

def asciiAllowed (c : Char) : Bool :=
  c = '\n' || (32 ≤ c.toNat && c.toNat ≤ 126)

partial def generateSampledFromIds
    (batch seqLen vocab : Nat)
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]))
    (params : TorchLean.ParamList Float (nn.paramShapes model))
    (promptIds : List Nat)
    (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (allowId : Nat → Bool := fun _ => true)
    (padId : Nat := 0) : IO (List Nat) := do
  if hSeqLen : seqLen = 0 then
    -- The CLI rejects this case, but keeping the definition total makes the helper reusable.
    pure promptIds
  else if hBatch : batch = 0 then
    -- Likewise, `--batch 0` is rejected by the CLI parser.
    pure promptIds
  else
  let b0 : Fin batch := ⟨0, Nat.pos_of_ne_zero hBatch⟩
  let rec loop (ids : List Nat) : Nat → IO (List Nat)
    | 0 => pure ids
    | n + 1 => do
        let generatedSoFar := steps - (n + 1)
        let start := if ids.length > seqLen then ids.length - seqLen else 0
        let window := (ids.drop start).take seqLen
        let predPos := if window.isEmpty then 0 else window.length - 1
        let padded := window ++ List.replicate (seqLen - window.length) padId
        let x2D : Tensor Float (NN.Tensor.Shape.Mat seqLen vocab) :=
          Tensor.dim (fun t => text.oneHotTokenFloat vocab (padded.getD t.val padId))
        let x : Tensor Float (shape![batch, seqLen, vocab]) :=
          Tensor.dim (fun _ => x2D)
        let logits ← nn.eval1NoGrad (α := Float) opts model params x
        let recent :=
          if repeatWindow = 0 then
            []
          else
            ids.drop (ids.length - Nat.min ids.length repeatWindow)
        let scores : Array Float :=
          match logits with
          | Tensor.dim batches =>
              match batches b0 with
              | Tensor.dim rows =>
                  let pos : Nat := Nat.min predPos (seqLen - 1)
                  have hpos : pos < seqLen := by
                    have hle : pos ≤ (seqLen - 1) := Nat.min_le_right _ _
                    have hlt : (seqLen - 1) < seqLen :=
                      Nat.sub_lt (Nat.pos_of_ne_zero hSeqLen) (by decide)
                    exact Nat.lt_of_le_of_lt hle hlt
                  let posFin : Fin seqLen := ⟨pos, hpos⟩
                  match rows posFin with
                  | Tensor.dim cols =>
                      Array.ofFn (fun j : Fin vocab =>
                        match cols j with
                        | Tensor.scalar x => x)
                  | _ => Array.replicate vocab 0.0
              | _ => Array.replicate vocab 0.0
          | _ => Array.replicate vocab 0.0
        let scores := text.penalizeRepeats scores recent repeatPenalty
        let scores := text.restrictScores scores allowId
        let nextTok : Nat :=
          if topK = 1 then
            text.greedyIndex scores
          else
            text.sampleTopKIndex scores temperature topK seed generatedSoFar
        loop (ids ++ [nextTok]) n
  loop promptIds steps

def main (args : List String) : IO UInt32 := do
  if args.contains "--cuda" || CLI.hasFlagValue args "log" then
    TorchLean.Module.run exeName args
      (.float (fun opts rest => do
        let (corpus, rest) ← takeInputText rest
        let (train, rest) ← Common.orThrow exeName <| parseTrainOptions opts rest
        Common.orThrow exeName <| CLI.requireNoArgs rest
        let alphabet := buildAlphabet corpus
        let tok := text.Tokenizer.ofAlphabet alphabet (unkId := 0) (unkChar := '?')
        let vocab := tok.vocabSize
        let batch := train.batch
        let seqLen := train.seqLen

        -- Shapes depend on runtime flags and the derived alphabet size.
        let σ : Shape := shape![batch, seqLen, vocab]
        let τ : Shape := σ
        let cfg : nn.models.CausalOneHotConfig :=
          { batch := batch
            seqLen := seqLen
            vocab := vocab
            numHeads := 2
            headDim := 32
            ffnHidden := 256
            layers := 2 }
        let mkModel : nn.M (nn.Sequential σ τ) :=
          if hSeq : seqLen = 0 then
            -- Keep the definition total even though the CLI rejects `seqLen=0`.
            nn.linear vocab vocab (pfx := NN.Tensor.Shape.Mat batch seqLen)
          else
            have h_dModel : cfg.dModel ≠ 0 := by
              -- For this example, `dModel = numHeads * headDim = 2 * 32`.
              -- Unfolding `cfg` lets `simp` compute the constant.
              simp [cfg, nn.models.CausalOneHotConfig.dModel]
            nn.models.causalTransformerOneHot cfg (h_seqLen := hSeq) (h_dModel := h_dModel)

        let toksList := tok.encode corpus
        let toks := toksList.toArray
        let usableStarts : Nat := text.Corpus.usableTokenStarts toks.size seqLen

        let mkBatchSample (step : Nat) : API.sample.Supervised Float σ τ :=
          -- CharGPT follows the usual minGPT data rule: each optimizer step draws a fresh random
          -- batch of token windows from the corpus.  The helper in `NN.API.Text` makes this the
          -- same reusable contract as the other text models: token array + `(seed, step)` ->
          -- one `(batch, seqLen, vocab)` causal-LM sample.
          let idsAt := text.Corpus.randomBatchTokenWindows toks batch seqLen train.seed step (padId := 0)
          text.causalLmSampleOneHotBatchRows (α := Float) batch seqLen vocab idsAt (padId := 0)

        let firstSample : API.sample.Supervised Float σ τ :=
          mkBatchSample 0

        nn.withModel mkModel fun model => do
          let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
          let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
          match train.loadParams? with
          | none => pure ()
          | some path =>
              TorchLean.ParamIO.loadModuleParamsBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ])
                m path
          let loss0 ← TorchLean.Module.forward (α := Float) m firstSample
          let L0 := Tensor.toScalar loss0

          if train.steps != 0 then
            let opt := TorchLean.Optim.adam (α := Float)
              (paramShapes := nn.paramShapes model)
              (lr := train.lr)
              (beta1 := 0.9)
              (beta2 := 0.999)
              (epsilon := 1e-8)
            let optH ← TorchLean.Optim.handle (α := Float) m opt
            for step in [0:train.steps] do
              let sample := mkBatchSample step
              optH.step sample

          let loss1 ← TorchLean.Module.forward (α := Float) m firstSample
          let L1 := Tensor.toScalar loss1

          let promptIds := tok.encode train.prompt
          let allowId : Nat → Bool :=
            if train.asciiOnly then
              fun i => asciiAllowed (alphabet.getD i '?')
            else
              fun _ => true
          let outIds ←
            generateSampledFromIds batch seqLen vocab opts model m.trainer.params promptIds
              train.generate train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty
              (allowId := allowId) (padId := 0)
          let sampled := escapeCharIdsForDisplay tok outIds
          IO.println s!"  vocab={vocab} (unique chars)"
          IO.println s!"  sampled={sampled}"
          Common.writeBeforeAfterLossLogTo train.log "CharGPT (minGPT-style)" train.steps L0 L1
            #[s!"device={if opts.useGpu then "cuda" else "cpu"}",
              s!"vocab={vocab}",
              s!"usable_windows={usableStarts}",
              s!"prompt={text.escapeForDisplay train.prompt}",
              s!"generated={sampled}"]
          match train.saveParams? with
          | none => pure ()
          | some path =>
              TorchLean.ParamIO.saveModuleParamsBits (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ])
                m path
              IO.println s!"  wrote params: {path}"
      ))
      { banner? := some (fun opts =>
          s!"{exeName}: char-level GPT training (device={if opts.useGpu then "cuda" else "cpu"})")
        printOk := true }
  else
    TorchLean.Module.run exeName args
      (.float (fun _ _ => do
        throw <| IO.userError s!"{exeName}: use --cuda (CPU char-gpt is extremely slow in eager mode)"
      ))
      { printOk := true }

end NN.Examples.Models.Sequence.CharGpt
