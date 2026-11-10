/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Text.Tokenizer
public import TorchLean.NN.CLI.Training -- shake: keep

/-!
# Text Workflow Configuration

Display helpers, generation options, the generation training log, and checkpoint flags for text
commands.
-/

@[expose] public section

namespace TorchLean
namespace Text

/-! ## Causal LM Display Helpers -/

/--
Encode `text` and return the `length` tokens that start at position `offset`, with
`paddingTokenId` past the end of the encoding.

`offset = 0` gives the model's prompt window and `offset = 1` its next-token target, the input and
target that `Data.CausalLM.oneHotSample` cuts from one window of `length + 1` tokens.

Example:
```lean
-- `target` is `prompt` shifted left by one token.
def prompt : Tensor Nat [8] :=
  Text.tokenWindow Text.Tokenizer.byte 8 "hello world"

def target : Tensor Nat [8] :=
  Text.tokenWindow Text.Tokenizer.byte 8 "hello world" (offset := 1)
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
Wrap a short text fragment in double quotes and escape it for one-line terminal output.

Quotes and backslashes get their usual escapes, newline, carriage return and tab print as `\n`,
`\r` and `\t`, and every other ASCII control character (DEL included) prints as `\xNN`, so
byte-token predictions cannot turn a log into a binary file.
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
Text-generation settings shared by the GPT-style examples, one field for each flag of `parse`.

Example:
```lean
-- `topK := 1` decodes greedily. Any other cutoff samples, and `seed` makes the samples
-- reproducible from one run to the next.
def greedy : Text.GenerationOptions :=
  { prompt := "Once upon a time"
    continuationLength := 64
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
  /-- Number of tokens to generate after the prompt. -/
  continuationLength : Nat
  /-- Softmax temperature: finite and positive for sampling, ignored by greedy decoding. -/
  temperature : Float
  /-- Top-k cutoff: `0` samples the full vocabulary and `1` decodes greedily. -/
  topK : Nat
  /-- Finite nonnegative penalty, subtracted from a token's logit once per occurrence in the
  recent window. `0` disables it. -/
  repeatPenalty : Float
  /-- Number of recent tokens considered by the repeat penalty. `0` disables the window. -/
  repeatWindow : Nat
  /-- Deterministic RNG seed for sampling. -/
  seed : Nat
  /-- Restrict generated ids to a model-specific ASCII allow-list. -/
  asciiOnly : Bool
deriving Repr

namespace GenerationOptions

/--
Require a finite nonnegative repetition penalty and, for sampling, a finite positive temperature.
Greedy decoding (`topK = 1`) ignores temperature.
-/
def validate (options : GenerationOptions) : Except String Unit := do
  unless options.repeatPenalty.isFinite && 0.0 <= options.repeatPenalty do
    throw "generation repeat penalty must be finite and nonnegative"
  unless options.topK = 1 || (options.temperature.isFinite && 0.0 < options.temperature) do
    throw "generation temperature must be finite and positive"

/--
Parse `--prompt`, `--generate`, `--temperature`, `--top-k`, `--repeat-penalty`, `--repeat-window`,
`--sample-seed` and `--ascii-only`, the generation flags shared by the GPT-style examples.

The model command supplies its default prompt and sampling policy in `defaults`, and every other
argument goes back to the caller. Validation runs on the parsed options, so with `--top-k 1` any
temperature that parses is accepted.
-/
def parse
    (exeName : String)
    (arguments : List String)
    (defaults : GenerationOptions) :
    Except String (GenerationOptions × List String) := do
  let (prompt, arguments) ←
    TorchLean.CLI.takeFlagValue arguments "prompt" (default := defaults.prompt)
  let (newTokenCount, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "generate" (default := defaults.continuationLength)
  let (temperature, arguments) ←
    TorchLean.CLI.takeFloatFlag arguments "temperature" (default := defaults.temperature)
  let (topK, arguments) ← TorchLean.CLI.takeNatFlag arguments "top-k" (default := defaults.topK)
  let (repeatPenalty, arguments) ←
    TorchLean.CLI.takeFloatFlag arguments "repeat-penalty" (default := defaults.repeatPenalty)
  let (repeatWindow, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "repeat-window" (default := defaults.repeatWindow)
  let (seed, arguments) ←
    TorchLean.CLI.takeNatFlag arguments "sample-seed" (default := defaults.seed)
  let (asciiOnly, arguments) ←
    (CLI.takeSwitch arguments "ascii-only" (default := defaults.asciiOnly)).mapError
      (fun message => s!"{exeName}: {message}")
  let options : GenerationOptions :=
    { prompt
      continuationLength := newTokenCount
      temperature
      topK
      repeatPenalty
      repeatWindow
      seed
      asciiOnly }
  options.validate.mapError (fun message => s!"{exeName}: {message}")
  pure (options, arguments)

end GenerationOptions

/-! ## Text TrainLog Notes -/

/--
TrainLog note fields for generation-capable text commands: the escaped prompt, continuation
length, temperature, top-k, sampling seed, repetition penalty and window, the ASCII-only flag and,
when given, the generated text.

Model commands prepend dataset or architecture notes through `extraNotes`. The notes only make
sense inside `Log.writeGeneration`, which pairs them with the matching loss comparison.
-/
def GenerationOptions.notes
    (options : GenerationOptions)
    (generated? : Option String := none)
    (extraNotes : Array String := #[]) : Array String :=
  extraNotes ++
    #[s!"prompt={escape options.prompt}",
      s!"generate={options.continuationLength}",
      s!"temperature={options.temperature}",
      s!"top_k={options.topK}",
      s!"sample_seed={options.seed}",
      s!"repeat_penalty={options.repeatPenalty}",
      s!"repeat_window={options.repeatWindow}",
      s!"ascii_only={options.asciiOnly}"] ++
    match generated? with
    | some generated => #[s!"generated={generated}"]
    | none => #[]

/-!
### Training logs

`Text.Log` holds the training-log writer for generation-capable text commands. A separate
namespace keeps the writer out of `text.` completion.
-/

namespace Log

/-- Write a before/after loss log for a generation-capable text training command. -/
def writeGeneration
    (destination : Training.LogDestination)
    (title : String)
    (trainingSteps : Nat)
    (lossBefore lossAfter : Float)
    (options : GenerationOptions)
    (generated? : Option String := none)
    (extraNotes : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLossComparison
    destination title trainingSteps lossBefore lossAfter
    (GenerationOptions.notes options generated? extraNotes)

end Log

/-! ## Text Training Option Combinators -/

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

end Text
end TorchLean
