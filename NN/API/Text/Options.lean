/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Tokenizer
public import NN.API.CLI.Parser
public import NN.API.Trainer.Reporting
public import NN.API.CLI.Training -- shake: keep
public import NN.Tensor.Conversion -- shake: keep

/-!
# Text Workflow Configuration

Display helpers, generation and corpus option records, training-log metadata, and CLI parsers for
text workflows.
-/

@[expose] public section

namespace TorchLean
namespace text

/-! ## Causal LM Display Helpers -/

/--
Return a fixed-length token window from a text string.

`offset = 0` is the model prompt window; `offset = 1` is the usual next-token target window for
causal language modeling. Missing tokens are padded with `paddingTokenId`, matching
`Data.CausalLM.oneHotSample`.

Example:
```lean
-- `offset = 0` is the prompt window and `offset = 1` is its next-token target. That pair is the
-- whole of causal language-model supervision.
def prompt : Tensor Nat [8] :=
  text.tokenWindow text.Tokenizer.byte 8 "hello world"

def target : Tensor Nat [8] :=
  text.tokenWindow text.Tokenizer.byte 8 "hello world" (offset := 1)
```
-/
def tokenWindow
    (tokenizer : Tokenizer)
    (length : Nat)
    (text : String)
    (offset : Nat := 0)
    (paddingTokenId : Nat := 0) :
    Tensor Nat [length] :=
  let tokens := tokenizer.encode text
  TorchLean.Tensor.ofFn fun position =>
    tokens.getD (offset + position.val) paddingTokenId

/-- Decode a fixed token window extracted by `tokenWindow`. -/
def decodeWindow
    (tokenizer : Tokenizer)
    (length : Nat)
    (text : String)
    (offset : Nat := 0)
    (paddingTokenId : Nat := 0) :
    String :=
  tokenizer.decode <| Tensor.to
    (tokenWindow
      tokenizer length text (offset := offset) (paddingTokenId := paddingTokenId))
    (Array Nat)

/--
Escape a short text fragment for one-line terminal output.

Display-only: this does not change tokenizer semantics. Quotes and backslashes use their usual
escapes, common whitespace controls use `\\n`, `\\r`, and `\\t`, and every other ASCII control
character is written as `\\xNN`. Thus byte-token predictions cannot turn a log into a binary file.

The argument is called `fragment` rather than `text` so it cannot shadow the `text` namespace inside
the body; a local named `text` makes every `text.foo` spelling in scope resolve to a field access
instead, which is a genuinely confusing error to read.
-/
def escape (fragment : String) : String :=
  let hexDigit := fun n =>
    (#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']).getD
      (n % 16) '0'
  let escapeChar := fun c =>
    match c with
    | '\\' => "\\\\"
    | '"' => "\\\""
    | '\n' => "\\n"
    | '\r' => "\\r"
    | '\t' => "\\t"
    | _ =>
        let n := c.toNat
        if n < 32 || n = 127 then
          "\\x" ++ String.singleton (hexDigit (n / 16)) ++ String.singleton (hexDigit n)
        else
          String.singleton c
  "\"" ++ String.join (fragment.toList.map escapeChar) ++ "\""

/-! ## Sampling Helpers (Top-k) -/

/--
Shared text-generation flags for GPT-style examples.

Example:
```lean
-- `topK := 1` is greedy decoding; anything larger samples, and `seed` is what makes that sampling
-- reproducible from one run to the next.
def greedy : text.GenerationOptions :=
  { prompt := "Once upon a time"
    newTokenCount := 64
    temperature := 1.0
    topK := 1
    repeatPenalty := 0.0
    repeatWindow := 0
    seed := 0
    asciiOnly := true }
```
-/
structure GenerationOptions where
  /-- Prompt used to seed autoregressive generation. -/
  prompt : String
  /-- Number of new tokens to append. -/
  newTokenCount : Nat
  /-- Softmax temperature. Must be finite and positive for sampling; ignored by greedy decoding. -/
  temperature : Float
  /-- Top-k cutoff. `0` samples the full vocabulary; `1` gives greedy decoding. -/
  topK : Nat
  /-- Finite nonnegative penalty subtracted for repeated recent tokens. `0` disables it. -/
  repeatPenalty : Float
  /-- Number of recent tokens considered by the repeat penalty. `0` disables the window. -/
  repeatWindow : Nat
  /-- Deterministic RNG seed for sampling. -/
  seed : Nat
  /-- Restrict generated ids to a model-specific ASCII allow-list. -/
  asciiOnly : Bool
deriving Repr

namespace Internal

/--
Parse `--ascii-only`, accepting either a bare flag or a `true`/`false` value.

Internal on purpose: `GenerationOptions.parse` is the entry point, and parsing this flag on its own
would let a command accept it without recording it in the training log.
-/
def parseAsciiOnlyFlag (exeName : String) (arguments : List String) (default : Bool) :
    Except String (Bool × List String) := do
  match TorchLean.CLI.takeSwitch arguments "ascii-only" (default := default) with
  | .ok result => pure result
  | .error e => throw s!"{exeName}: {e}"

end Internal

namespace GenerationOptions

/--
Parse the generation flags shared by GPT-style examples.

The model command supplies its concrete default prompt and sampling policy. This parser owns only
the stable generation flags and returns arguments belonging to the caller.
-/
def parse
    (exeName : String)
    (arguments : List String)
    (defaults : GenerationOptions) :
    Except String (GenerationOptions × List String) := do
  let (prompt, arguments) ←
    TorchLean.CLI.takeFlagValue arguments "prompt" (default := defaults.prompt)
  let (newTokenCount, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "generate" (default := defaults.newTokenCount)
  let (temperature, arguments) ←
    TorchLean.CLI.takePositiveFloatFlag
      arguments exeName "temperature" (default := defaults.temperature)
  let (topK, arguments) ← TorchLean.CLI.takeNatFlag arguments "top-k" (default := defaults.topK)
  let (repeatPenalty, arguments) ←
    TorchLean.CLI.takeNonnegativeFloatFlag
      arguments exeName "repeat-penalty" (default := defaults.repeatPenalty)
  let (repeatWindow, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "repeat-window" (default := defaults.repeatWindow)
  let (seed, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "sample-seed" (default := defaults.seed)
  let (asciiOnly, arguments) ← Internal.parseAsciiOnlyFlag exeName arguments defaults.asciiOnly
  pure
    ({ prompt
       newTokenCount
       temperature
       topK
       repeatPenalty
       repeatWindow
       seed
       asciiOnly },
     arguments)

end GenerationOptions

/-! ## Text Workflow Option Records -/

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
    TorchLean.CLI.requirePathFlag arguments "data-file" (exeName := exeName)
  let (allowSmallData, arguments) ← TorchLean.CLI.takeBoolFlag arguments "allow-small-data"
  pure ({ dataFile := dataFile, allowSmallData := allowSmallData }, arguments)

end CorpusFileOptions

/-- Optional text-corpus path selected by `--data-file`, with caller-supplied default. -/
structure CorpusPathOptions where
  /-- Local text corpus path. -/
  path : System.FilePath
deriving Repr

namespace CorpusPathOptions

/-- Parse an optional `--data-file` flag using the supplied default path. -/
def parse
    (arguments : List String)
    (defaultPath : System.FilePath) :
    Except String (CorpusPathOptions × List String) := do
  let (path, arguments) ← TorchLean.CLI.takePathFlag arguments "data-file" (default := defaultPath)
  pure ({ path := path }, arguments)

end CorpusPathOptions

/-- Optional second corpus pass after the main training run. -/
structure FinetuneOptions where
  /-- Optional corpus used for a second fine-tuning pass. -/
  finetuneFile? : Option System.FilePath
  /-- Number of optimizer steps used on that second corpus when present. -/
  finetuneSteps : Nat
deriving Repr

namespace FinetuneOptions

/--
Parse the optional `--finetune-file` / `--finetune-steps` pair.

The caller supplies the default step count so commands can reuse their main training-step default.
-/
def parse
    (arguments : List String)
    (defaultSteps : Nat) :
    Except String (FinetuneOptions × List String) := do
  let (finetuneFile?, arguments) ← TorchLean.CLI.takePathFlag? arguments "finetune-file"
  let (finetuneSteps, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "finetune-steps" (default := defaultSteps)
  pure ({ finetuneFile? := finetuneFile?
          finetuneSteps := finetuneSteps }, arguments)

end FinetuneOptions

/-- Optional GPT-2 BPE tokenizer bundle plus an optional bounded-text cap. -/
structure BpeCorpusOptions where
  /-- Optional GPT-2 `vocab.json` path. Must be paired with `mergesFile?`. -/
  vocabularyFile? : Option System.FilePath
  /-- Optional GPT-2 `merges.txt` path. Must be paired with `vocabularyFile?`. -/
  mergesFile? : Option System.FilePath
  /-- Optional text-character cap for bounded local BPE runs. -/
  maximumCharacters? : Option Nat
deriving Repr

namespace BpeCorpusOptions

/--
Parse the optional GPT-2 BPE tokenizer bundle.

`--bpe-vocab` and `--bpe-merges` must appear together; `--max-chars` is independent.
-/
def parse
    (arguments : List String) :
    Except String (BpeCorpusOptions × List String) := do
  let ((vocabularyFile?, mergesFile?), arguments) ←
    TorchLean.CLI.takePairedPathFlags arguments "bpe-vocab" "bpe-merges"
  let (maximumCharacters?, arguments) ←
    TorchLean.CLI.takeNatFlag? arguments "max-chars"
  pure ({ vocabularyFile?, mergesFile?, maximumCharacters? }, arguments)

end BpeCorpusOptions

/-- Shared terminal-REPL toggle used by interactive text examples. -/
structure InteractiveOptions where
  /-- Keep the trained model alive and read prompts from stdin. -/
  interactive : Bool
deriving Repr

namespace InteractiveOptions

/-- Parse the shared `--interactive` flag used by text examples with a terminal prompt loop. -/
def parse
    (arguments : List String) :
    Except String (InteractiveOptions × List String) := do
  let (interactive, arguments) ← TorchLean.CLI.takeBoolFlag arguments "interactive"
  pure ({ interactive := interactive }, arguments)

end InteractiveOptions

/-- Shared prompt plus continuation-length options for simple text-generation commands. -/
structure PromptGenerationOptions where
  /-- Prompt used for before/after reports and generation. -/
  prompt : String
  /-- Number of generated tokens or characters after training. -/
  newTokenCount : Nat
deriving Repr

namespace PromptGenerationOptions

/-- Parse the shared `--prompt` / `--generate` flags. -/
def parse
    (arguments : List String)
    (defaults : PromptGenerationOptions) :
    Except String (PromptGenerationOptions × List String) := do
  let (prompt, arguments) ←
    TorchLean.CLI.takeFlagValue arguments "prompt" (default := defaults.prompt)
  let (newTokenCount, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "generate" (default := defaults.newTokenCount)
  pure ({ prompt, newTokenCount }, arguments)

end PromptGenerationOptions

/-! ## Text TrainLog Notes -/

namespace Internal

/--
TrainLog note fields for generation-capable text commands.

The stable generation surface is prompt, continuation length, temperature/top-k, repetition
control, RNG seed, and ASCII-only filtering. Model commands can prepend dataset or architecture
notes through `extra`.

Internal on purpose: these note arrays only make sense inside the `Log.write*` wrappers below,
which pair them with the matching loss comparison.
-/
def generationNotes
    (options : GenerationOptions)
    (generated? : Option String := none)
    (extraNotes : Array String := #[]) : Array String :=
  extraNotes ++
    #[s!"prompt={escape options.prompt}",
      s!"generate={options.newTokenCount}",
      s!"temperature={options.temperature}",
      s!"top_k={options.topK}",
      s!"sample_seed={options.seed}",
      s!"repeat_penalty={options.repeatPenalty}",
      s!"repeat_window={options.repeatWindow}",
      s!"ascii_only={options.asciiOnly}"] ++
    match generated? with
    | some generated => #[s!"generated={generated}"]
    | none => #[]

/-- TrainLog note fields for prompt commands that do not expose the full sampling surface. -/
def promptGenerationNotes
    (options : PromptGenerationOptions)
    (generated? : Option String := none)
    (extraNotes : Array String := #[]) : Array String :=
  extraNotes ++
    #[s!"prompt={escape options.prompt}",
      s!"generate={options.newTokenCount}"] ++
    match generated? with
    | some generated => #[s!"generated={generated}"]
    | none => #[]

end Internal

/-!
### Training logs

`text.Log` is the whole logging surface for text commands. It lives in its own namespace so that
`text.` completion shows tokenizers, sampling, and option parsers rather than a pair of long
`write*TrainLog` names.
-/

namespace Log

/-- Write a before/after loss log for a generation-capable text training command. -/
def writeGeneration
    (destination : Runtime.Training.LogDestination)
    (title : String)
    (trainingSteps : Nat)
    (lossBefore lossAfter : Float)
    (options : GenerationOptions)
    (generated? : Option String := none)
    (extraNotes : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLossComparison
    destination title trainingSteps lossBefore lossAfter
    (Internal.generationNotes options generated? extraNotes)

/-- Write a before/after loss log for a prompt-based text training command. -/
def writePrompt
    (destination : Runtime.Training.LogDestination)
    (title : String)
    (trainingSteps : Nat)
    (lossBefore lossAfter : Float)
    (options : PromptGenerationOptions)
    (generated? : Option String := none)
    (extraNotes : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLossComparison
    destination title trainingSteps lossBefore lossAfter
    (Internal.promptGenerationNotes options generated? extraNotes)

end Log

/-! ## Text Training Option Combinators -/

/-- Number of corpus windows used by a finite or cyclic text-training command. -/
structure WindowOptions where
  /-- Number of windows available to the training sampler. -/
  windowCount : Nat
deriving Repr

namespace WindowOptions

/-- Parse a positive `--windows` value. -/
def parse
    (exeName : String)
    (arguments : List String)
    (defaultWindows : Nat) :
    Except String (WindowOptions × List String) := do
  let (windowCount, arguments) ←
    TorchLean.CLI.takePositiveNatFlag arguments exeName "windows" (default := defaultWindows)
  pure ({ windowCount }, arguments)

end WindowOptions

/-- Optional model-checkpoint paths for text training and generation. -/
structure CheckpointOptions where
  /-- Checkpoint loaded before training or generation. -/
  loadCheckpoint? : Option System.FilePath
  /-- Checkpoint written after training. -/
  saveCheckpoint? : Option System.FilePath
deriving Repr

namespace CheckpointOptions

/-- Parse `--load-checkpoint` and `--save-checkpoint`. -/
def parse (arguments : List String) : Except String (CheckpointOptions × List String) := do
  let (loadCheckpoint?, arguments) ← TorchLean.CLI.takePathFlag? arguments "load-checkpoint"
  let (saveCheckpoint?, arguments) ← TorchLean.CLI.takePathFlag? arguments "save-checkpoint"
  pure ({ loadCheckpoint?, saveCheckpoint? }, arguments)

end CheckpointOptions

end text
end TorchLean
