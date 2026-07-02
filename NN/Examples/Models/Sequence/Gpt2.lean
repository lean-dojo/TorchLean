/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA text example:
  lake build -R -K cuda=true && lake exe torchlean gpt2 --cuda --steps 1 --windows 1 --generate 0
  lake exe -K cuda=true torchlean gpt2 --cuda --tiny-shakespeare --prompt "First Citizen:" --steps 1 \
    --windows 1 --generate 0 --temperature 0.85 --top-k 12 --sample-seed 7
  lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare --steps 1 --windows 1 \
    --save-params data/model_zoo/gpt2_shakespeare.params.json
  lake exe -K cuda=true torchlean gpt2_saved --cuda --fast-kernels --params data/model_zoo/gpt2_shakespeare.params.json \
    --prompt "First Citizen:" --generate 0

Dataset example:
  python3 scripts/datasets/download_example_data.py --tiny-shakespeare
  lake exe -K cuda=true torchlean gpt2 --cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
  lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare --steps 1 --windows 1 --generate 0

This is a GPT-2-style *causal* language-model command (byte-level tokens).

Performance note: use CUDA for this example. The pure Lean CPU path exists for debugging tiny model
states, but Transformer workloads are too slow there for a useful run. The default command uses one
training window so it finishes quickly; pass larger `--steps`, `--windows`, and `--generate` values
when you want a real text experiment.
The command exercises masked self-attention, LayerNorm, and feed-forward blocks through the public
`TorchLean.nn` model constructors and `TorchLean.text` token tools.

After a run that writes `--log <path>`, you can view the prompt and sampled continuation in the
infoview via:

`#gpt2_train_log_file_view "<path>"`
-/

module


public import NN
public import NN.Examples.Models.Common.RealData

/-!
GPT-2 style sequence model example.

The runnable causal language-model path includes training, generation, and infoview support. It uses
the same public TorchLean model API that the command-line example uses.
-/

/-!
# GPT-2-Style Causal Language Model Example

Runnable `torchlean gpt2` example. It builds a GPT-2-style causal transformer over
byte-level tokens, with optional real text input from tiny-shakespeare or `--data-file PATH`.

For the simplest "Karpathy-style single text file" path, use `torchlean chargpt`
(character-level tokenizer). This `gpt2` command is byte-level and shows the Transformer block
wiring and save/reload loop.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake build -R -K cuda=true && lake exe torchlean gpt2 --cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Gpt2

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "gpt2"

/-- Batch size for the byte-level causal Transformer. -/
def batch : Nat := 1

/-- Prompt/target window length for the runnable GPT example. -/
def seqLen : Nat := 1

/-- Byte vocabulary width used by the one-hot tokenizer. -/
def vocab : Nat := 8

/-- Number of attention heads in the miniature Transformer block. -/
def numHeads : Nat := 1

/-- Per-head embedding width. The model dimension is `numHeads * headDim`. -/
def headDim : Nat := 1

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Hidden width of the feed-forward sublayer. -/
def ffnHidden : Nat := 2

/-- Number of Transformer encoder blocks. -/
def layers : Nat := 1

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

/-- Input shape: batched byte-level one-hot token windows. -/
abbrev σ : Shape :=
  shape![batch, seqLen, vocab]

/-- Output shape: one vocabulary-logit row for every input token position. -/
abbrev τ : Shape :=
  σ

/-- Public GPT-style causal Transformer constructor specialized to the byte-level config. -/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.CausalTransformerOneHot
    { batch := batch
      seqLen := seqLen
      vocab := vocab
      numHeads := numHeads
      headDim := headDim
      ffnHidden := ffnHidden
      layers := layers }

/-- Build a batched causal-LM sample by repeating one token window across all rows. -/
def mkSampleFromTokenIds (toks : List Nat) : SupervisedSample Float σ τ :=
  Data.causalLmOneHotSample (α := Float) batch seqLen vocab (toks.map (· % vocab))
    (padId := 0)

/--
Build a batch sample from per-row token windows.

`idsByBatch[i]` is the `(seqLen + 1)`-token window for batch row `i`. If fewer than `batch` windows
are provided, the final window is repeated to fill the batch.
-/
def mkSampleBatchFromTokenIds (idsByBatch : Array (List Nat)) :
    SupervisedSample Float σ τ :=
  let fallback : List Nat := idsByBatch.getD 0 (List.replicate (seqLen + 1) 32)
  let idsByBatch := idsByBatch.map (fun ids => ids.map (· % vocab))
  let fallback := fallback.map (· % vocab)
  Data.causalLmOneHotSampleRowsFromArray
    (α := Float) batch seqLen vocab idsByBatch fallback (padId := 0)

/--
Parse GPT-2-specific data flags and return the training corpus plus remaining runtime flags.
-/
def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName RealData.tinyShakespearePath
    [("--tiny-shakespeare", RealData.tinyShakespearePath),
      ("--tinystories-valid", RealData.tinyStoriesValidPath)]
    RealData.missingTinyShakespeareOrTinyStoriesHint args

/-- Byte-token window used for reporting prompt/target text. -/
def tokenWindowIds (input : String) (offset : Nat) : List Nat :=
  text.tokenWindow text.Tokenizer.byte seqLen input (offset := offset) (padId := 32)

/-- Print a compact before/after language-model report for the first batch row. -/
def printPredictionReport (label : String) (input : String) (logits : Tensor.T Float σ) :
    IO Unit := do
  let predIds := text.argmaxTokenIdsFromBatchLogits (α := Float) logits ⟨0, by decide⟩
  IO.println s!"  {label} pred={text.escapeByteIdsForDisplay predIds}"
  IO.println s!"  prompt={text.escapeByteIdsForDisplay (tokenWindowIds input 0)}"
  IO.println s!"  target={text.escapeByteIdsForDisplay (tokenWindowIds input 1)}"

/-- Convert byte ids into the typed batched one-hot input tensor used for generation. -/
def inputTensorFromIds (ids : List Nat) : Tensor.T Float σ :=
  let (x2DF, _) := text.causalLmXYOneHotMatFloat
    (seqLen := seqLen) (vocab := vocab) (ids.map (· % vocab)) (padId := 0)
  let x : Tensor.T Float σ := Spec.Tensor.dim (fun _bi => x2DF)
  x

/--
Fitted byte-level GPT predictor.

Training, saved-checkpoint inference, and future compiled runners all provide this one closure.
Generation only needs a logit-producing function; it does not depend on where the logits came from.
-/
abbrev Predictor :=
  Tensor.T Float σ → IO (Tensor.T Float τ)

mutual

/-- Autoregressively extend byte token ids using a trained byte-level GPT model. -/
partial def generateSampledFromIds
    (predict : Predictor)
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
  let ids ←
    text.autoregressiveTokenIds seqLen 32 promptIds gen
      (fun padded predPos => do
        let logits ← predict (inputTensorFromIds padded)
        pure (text.batchLogitScoresAt logits ⟨0, by decide⟩ predPos))
      (allowId := allowId)
      -- Keep the same "space" fallback convention across byte-level text commands.
      (sanitize := fun tok => if tok < vocab then tok else 0)
  pure ids

/-- Encode a string prompt and autoregressively extend it. -/
partial def generateSampled
    (predict : Predictor)
    (prompt : String) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (List Nat) := do
  let init := text.Tokenizer.byte.encode prompt
  generateSampledFromIds predict init steps temperature topK seed repeatWindow repeatPenalty
    asciiOnly

end

/-- Build a finite cyclic training set from corpus text, biased toward the prompt when present. -/
def samplesFromCorpus (input _prompt : String) (windows : Nat) :
    Array (SupervisedSample Float σ τ) :=
  let toks := text.Tokenizer.byte.encode input
  let offs := (text.Corpus.promptAwareOffsets toks.length seqLen windows none).toArray
  offs.map (fun off =>
    let idsByBatch : Array (List Nat) :=
      Array.ofFn (fun i : Fin batch =>
        let off' := (off + i.val * (seqLen / 2 + 1)) % Nat.max 1 (toks.length - (seqLen + 1))
        text.tokenWindow text.Tokenizer.byte (seqLen + 1) input (offset := off') (padId := 32))
    mkSampleBatchFromTokenIds idsByBatch)

/--
Interactive prompt loop for the in-memory Float model.

Each line is appended to the current byte context, decoded through the trained local model, and then
kept as context for the next prompt unless the user clears it.
-/
partial def interactiveLoopFloat
    (predict : Predictor)
    (train : text.InteractiveCheckpointedWindowedTrainGenerationOptions) :
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
        generateSampledFromIds predict inputIds train.generate
          train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let genOnly := outIds.drop inputIds.length
      IO.println s!"  generated={text.escapeByteIdsForDisplay genOnly}"
      loop outIds
  loop []

/--
Float-specialized training path with decoded prediction reports.

The CUDA executable uses Lean `Float` tensors, so this branch can show actual prompt,
target, and predicted text before and after training. The polymorphic path above remains useful for
checking the same training loop over other scalar backends.
-/
def unitTrainStepsFloat (opts : Options) (input : String)
    (train : text.InteractiveCheckpointedWindowedTrainGenerationOptions) :
    IO (Float × Float × String) := do
  let samples := samplesFromCorpus input train.prompt train.windows
  let reportSample := Data.textCausalBatchSample (α := Float) batch seqLen vocab train.prompt
  let run := Trainer.runConfig opts { optimizer := optim.adam { lr := train.lr } }
  let trainer := Trainer.new model <|
    Trainer.Config.fromRunConfig run .crossEntropy
  trainer.printInfo

  /-
  The GPT-2 command trains on a bounded, prompt-aware window table.  That makes the training
  schedule explicit and reproducible, and it lets the public trainer own checkpointing and optimizer
  state.  The example stays focused on text windows, decoding, and generation instead of runtime
  module bookkeeping.
  -/
  let trained ← trainer.train
    (Data.floatSampleArray samples)
    { steps := train.steps
      log := .disabled
      loadParams? := train.loadParams?
      saveParams? := train.saveParams? }
  let (L0, L1) ←
    Trainer.TrainSummary.requireAndPrintFloatLosses exeName trained.report
      (steps? := some train.steps) (lr? := some train.lr)

  let logits0 ← trained.eval (Sample.x reportSample)
  printPredictionReport "after " train.prompt logits0
  let generatedIds ← generateSampled trained.eval train.prompt train.generate
    train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
  let generated := text.escapeByteIdsForDisplay generatedIds
  IO.println s!"  generated={generated}"
  IO.println s!"  corpus_bytes={input.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({train.topK}), temperature={train.temperature}, seed={train.seed}"
  IO.println s!"  repetition_penalty={train.repeatPenalty} repeat_window={train.repeatWindow}"
  if train.interactive then
    interactiveLoopFloat trained.eval train
  let cudaMemWatch := ModelZoo.effectiveCudaMemWatch opts train.steps train.cudaMemWatch
  text.writeGenerationTrainLog
    train.log "GPT-2 byte prompt training" train.steps L0 L1
    train.toGenerationOptions generated
    #[ModelZoo.deviceNote opts,
      s!"windows={train.windows}",
      s!"cuda_mem_watch={cudaMemWatch}"]
  pure (L0, L1, generated)

/-- CLI entrypoint for byte-level GPT training, sampling, logging, and checkpointing. -/
def main (args : List String) : IO UInt32 := do
  ModelZoo.runFloat exeName args
    (banner := ModelZoo.bannerWithDevice exeName "causal LM training")
    (k := fun opts rest => do
      let (input, rest) ← takeInputText rest
      let defaultSteps : Nat := if opts.useGpu then 1 else 0
      let (train, rest) ← ModelZoo.orThrow exeName <|
        text.InteractiveCheckpointedWindowedTrainGenerationOptions.parse
          exeName rest defaultLogJson defaultSteps 0.001 1
            { prompt := "First Citizen:"
              generate := 0
              temperature := 0.85
              topK := 12
              repeatPenalty := 1.25
              repeatWindow := 24
              seed := 0
              asciiOnly := false }
            (allowZeroSteps := true)
      CLI.requireNoArgs exeName rest
      let (_L0, _L1, _generated) ← unitTrainStepsFloat opts input train)

end NN.Examples.Models.Sequence.Gpt2
