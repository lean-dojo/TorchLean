/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA-only minGPT-style addition walkthrough:
  scripts/lake.sh -Kcuda=true exe torchlean gpt_adder --device cuda \
    --steps 1 --optim adam --lr 0.005 \
    --a 7 --b 8
  scripts/lake.sh -Kcuda=true exe torchlean gpt_adder --device cuda \
    --steps 1 --optim sgd --lr 0.05 \
    --a 7 --b 8

Interactive addition REPL:
  scripts/lake.sh -Kcuda=true exe torchlean gpt_adder --device cuda --steps 1 --interactive
-/

module

public import TorchLean.NN.Data.CausalLM
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Text.Generation
public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train
public import TorchLeanExamples.Models.Training
public import TorchLean.NN.Models.CausalTransformer

/-!
# minGPT-Style Addition Example

The example is a TorchLean-native take on Karpathy's `minGPT/projects/adder` experiment, which
trains a compact GPT to complete digit strings of the form

`digits(a) ++ digits(b) ++ reverseDigits(a+b)`.

For example, in the one-digit setting $8+7=15$ is represented as the digit sequence
`8 7 5 1`. At inference time the model sees `8 7` and greedily generates the two result digits
`5 1`, which reverse back to `15`.

The arithmetic task is a controlled test of the CUDA GPT training loop:

* synthetic data is generated in Lean,
* the model is a GPT-style causal Transformer built from TorchLean layers,
* training runs only on CUDA,
* `--optim` picks one of six optimizers (AdamW by default), with minGPT's `beta2 = 0.95` for Adam
  and AdamW and minGPT's weight decay `0.1` for AdamW, applied here to every trainable parameter,
* evaluation greedily completes every one-digit addition problem.

Training uses the eager CUDA runtime. Lean records a fresh autograd tape for each step and
composes attention from LibTorch matrix products and softmax, without fused attention or CUDA
graph replay.

The GPT-shaped architecture comes from the public TorchLean model constructor
`nn.CausalTransformer.oneHot`, so the example can focus on the mechanics of the adder task.

Reference: <https://github.com/karpathy/minGPT/tree/master/projects/adder>.
-/

@[expose] public section

open TorchLean.Backend.Device TorchLean.Trainer.Objective

namespace TorchLean.Examples.GPTAdder

/-- CLI subcommand label used by the shared model runner. -/
def exeName : String := "gpt_adder"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "gpt_adder"

/--
Number of input digits per operand.

The one-digit curriculum is small enough to train in the eager CUDA runtime and includes carry
examples such as $8+7=15$.
-/
def operandDigits : Nat := 1

/-- Digit-only vocabulary, matching minGPT's adder task (`0..9`). -/
def vocabularySize : Nat := 10

/--
Full one-digit table batch size.

The batch holds 100 rows because single-row workloads underuse the GPU. In all-pairs mode one
optimizer step sees every one-digit addition problem, and evaluation completes the whole table with
two batched greedy forward passes.
-/
def batchSize : Nat := 100

local instance : NeZero batchSize := ⟨by decide⟩

/--
Number of training problems with `--train-split`, the first 80 rows of the row-major table, so the
20 held-out problems are those with first operand 8 or 9. Karpathy's adder also holds out a test
split, which for one digit is 80 train and 20 test problems.
-/
def trainCount : Nat := 80

/-- Held-out one-digit examples when `--train-split` is enabled. -/
def testCount : Nat := 20

/--
GPT block size: `a`, `b`, and all but the final reversed result digit.

For `operandDigits = 1`, a full rendered example has length $1+1+2=4$ and a model input has length
`3`, as in minGPT's `get_block_size = 3 * operandDigits + 1 - 1`.
-/
def contextLength : Nat := 3 * operandDigits

/--
Number of attention heads.

Karpathy's minGPT default for the adder is `gpt-nano` (`3` heads, width `48`). TorchLean's eager
CUDA trainer is tape-based, so this example uses a smaller model (`2` heads, width `32`) that keeps
`torchlean gpt_adder` practical to run.
-/
def attentionHeads : Nat := 2

/-- Per-head width. -/
def attentionHeadWidth : Nat := 16

/-- Transformer embedding width. -/
def modelWidth : Nat := attentionHeads * attentionHeadWidth

/-- Feed-forward hidden width ($4$ times `modelWidth`, matching the common GPT MLP ratio). -/
def feedForwardWidth : Nat := 128

/-- Number of Transformer blocks. -/
def transformerLayers : Nat := 2

/-- Number of positions per row that contribute to the minGPT adder loss. -/
def activeTargetPositions : Nat :=
  contextLength - (2 * operandDigits - 1)

/--
Number of non-ignored next-token targets in the training batch.

Training targets set the ignored prefix positions to all-zero rows, and the adder loss divides the
summed one-hot cross-entropy by this count. The result is the mean over active next-token labels,
the normalization that minGPT gets from `ignore_index=-1`.
-/
def activeTargetCount : Nat :=
  batchSize * activeTargetPositions

local instance : NeZero contextLength := ⟨by decide⟩
local instance : NeZero modelWidth := ⟨by decide⟩

/-- GPT configuration shared by the typed shapes and model constructor. -/
abbrev modelConfig : nn.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := vocabularySize
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := transformerLayers
    }

/-- Shape of the batched one-hot digit sequences, and of the model's digit logits, one row per
position. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/-- Compact GPT-style causal Transformer for digit addition. -/
def model : nn.Builder (nn.Sequential input input) :=
  nn.CausalTransformer.oneHot modelConfig (batchShape := [batchSize])

/--
Adder-specific scalar loss: the one-hot cross-entropy summed over all positions and divided by
`activeTargetCount`.

Ignored prefix positions have all-zero one-hot target rows (`maskTargets`), so they add zero
to the one-hot cross-entropy sum. The sum is divided by the number of active target positions,
following minGPT's `ignore_index`-style normalization, so the ignored prefix rows do not count
toward the mean.
-/
def loss {α : Type} [Storage α] [Context α] :
    Program α [input, input] ([] : Shape) :=
  fun {m} _ _ logits targetOneHot =>
    show m (Ops.Ref (m := m) (α := α) ([] : Shape)) from do
      let summed ← Loss.oneHotCrossEntropy
        (m := m) (α := α) (s := input) 2 logits targetOneHot (reduction := Loss.Reduction.sum)
      Ops.apply (m := m) (α := α) (.scale (s := ([] : Shape)))
          (summed, .data
            (Ops.dataConst (m := m) (α := α)
              (Tensor.scalar ((1 : α) / (activeTargetCount : α)))), ())

/-- Render `n` as exactly `width` base-10 digits, most-significant first. -/
def fixedDigits (width n : Nat) : Tensor Nat [width] :=
  Tensor.ofFn fun index => (n / Nat.pow 10 (width - index.val - 1)) % 10

/--
minGPT adder rendering.

For `operandDigits = 1`, `a = 8`, `b = 7` becomes `[8, 7, 5, 1]`, i.e. the sum `15` is stored
reversed as `5, 1`. Reversing the output digits makes carry propagation local in left-to-right
generation.
-/
def renderExample (a b : Nat) : Tensor Nat [contextLength + 1] :=
  let sumDigits := fixedDigits (operandDigits + 1) (a + b)
  let reversed := Tensor.ofFn fun index : Fin (operandDigits + 1) => sumDigits[index.rev]
  let values := Tensor.concat
    (Tensor.concat (fixedDigits operandDigits a) (fixedDigits operandDigits b)) reversed
  Tensor.window values (contextLength + 1) 0 0

/--
Whether target position `t` lies past the masked operand prefix, that is,
`t ≥ 2 * operandDigits - 1`.

In `projects/adder/adder.py`, the target vector `y` is shifted by one token and then
`y[:operandDigits*2-1] = -1`, where `-1` is PyTorch's "ignore index" for cross entropy. TorchLean's
one-hot cross entropy has no ignore-index target, so we represent the same idea by
using an all-zero one-hot vector on ignored positions. Because the loss is
$-\sum y\log p$, these
positions contribute exactly zero gradient.
-/
def keepTargetPosition (t : Nat) : Bool :=
  t ≥ 2 * operandDigits - 1

/-- Apply the minGPT adder loss mask to a shifted one-hot target matrix. -/
def maskTargets {α : Type} [Storage α] [Zero α]
    (y : Tensor α [contextLength, vocabularySize]) :
    Tensor α [contextLength, vocabularySize] :=
  Tensor.stack 0 fun t =>
    if keepTargetPosition t.val then
      y[t]
    else
      Tensor.zeros [vocabularySize]

/--
Build one unbatched one-hot causal-LM sample for an addition row, then apply the minGPT-style
ignored-prefix mask to its target matrix.
-/
def rowSample (a b : Nat) :
    Except String
      (Sample.Supervised Float [contextLength, vocabularySize]
        [contextLength, vocabularySize]) := do
  let tokens ← Tensor.checkIndices vocabularySize (renderExample a b)
  pure <| Sample.mapTarget maskTargets <|
    Data.CausalLM.oneHotSample (α := Float) [] contextLength vocabularySize tokens

/-- Build one supervised next-digit sample from an addition problem. -/
def additionSample (a b : Nat) :
    Except String (Sample.Supervised Float input input) := do
  let row ← rowSample a b
  pure
    { input := Tensor.repeatAxis 0 batchSize row.input
      target := Tensor.repeatAxis 0 batchSize row.target }

/-- Problem `i` of the one-digit table in row-major order, `(j / 10, j % 10)` with `j = i % 100`. -/
def pairAt (i : Nat) : Nat × Nat :=
  let j := i % 100
  (j / 10, j % 10)

/-- Problem for training row `i`. In split mode the 100 rows cycle through the first 80 problems. -/
def trainPairAt (trainSplit : Bool) (i : Nat) : Nat × Nat :=
  if trainSplit then
    pairAt (i % trainCount)
  else
    pairAt i

/-- Parse `a+b` into a one-digit operand pair, or return `none` for a malformed prompt. -/
def parseProbe? (s : String) : Option (Nat × Nat) :=
  let parts := s.trimAscii.toString.splitOn "+"
  match parts with
  | aStr :: bStr :: rest =>
      if rest.isEmpty then
        match aStr.toNat?, bStr.toNat? with
        | some a, some b =>
            if a < 10 && b < 10 then some (a, b) else none
        | _, _ => none
      else
        none
  | _ => none

/--
Parse a comma-separated list of one-digit `a+b` checks, skipping empty entries and rejecting
malformed ones.
-/
def parseProbeArray (s : String) : Except String (Array (Nat × Nat)) := do
  let raw := s.splitOn "," |>.filter (fun p => p.trimAscii.toString != "")
  let mut out : Array (Nat × Nat) := #[]
  for p in raw do
    match parseProbe? p with
    | some pair => out := out.push pair
    | none =>
        throw (s!"bad --probes entry {p}; "
          ++ "expected comma-separated one-digit prompts like 0+0,7+8")
  pure out

/-- Build a batched supervised sample with one row per one-digit addition problem. -/
def tableSample (trainSplit : Bool) :
    Except String (Sample.Supervised Float input input) := do
  let windows : Tensor Nat [batchSize, contextLength + 1] := Tensor.stack 0 fun bi =>
    let (a, b) := trainPairAt trainSplit bi.val
    renderExample a b
  let tokens ← Tensor.checkIndices vocabularySize windows
  let sample := Data.CausalLM.oneHotSample
    (α := Float) [batchSize] contextLength vocabularySize tokens
  pure <| Sample.mapTarget (fun targets =>
    Tensor.stack 0 fun bi => maskTargets (targets.unstack bi)) sample

/-- Decode reversed generated result digits back into a natural number. -/
def decodeResult {count : Nat} (revDigits : Tensor Nat [count]) : Nat :=
  let digits := Tensor.ofFn fun index : Fin count => revDigits[index.rev]
  digits.foldl (fun acc digit => acc * 10 + digit) 0

/-- Argmax token id at a sequence position for a chosen batch row. -/
def argmax
    (logits : Tensor Float input)
    (bi : Fin batchSize)
    (pos : Fin contextLength) : Nat :=
  (Tensor.argmax ((logits.unstack bi).unstack pos) ⟨0, Nat.one_pos⟩ false (by decide)).item.val

/--
Build a model input tensor from the current generated digit prefix. The prefix is zero-padded or
cut to `contextLength` and repeated across the batch.
-/
def inputFromDigits {count : Nat} (digits : Tensor Nat [count]) :
    Except String (Tensor Float input) := do
  let window : Tensor Nat [contextLength] :=
    Tensor.window digits contextLength 0 0
  let tokens ← Tensor.checkIndices vocabularySize window
  pure <| Tensor.repeatAxis 0 batchSize <|
    Tensor.oneHotIndices (α := Float) vocabularySize tokens

/-- Build a batched model input from one digit prefix per row. -/
def inputFromRows {count : Nat} (rows : Tensor Nat [batchSize, count]) :
    Except String (Tensor Float input) := do
  let windows : Tensor Nat [batchSize, contextLength] :=
    Tensor.mapLeading [batchSize] (fun row => Tensor.window row contextLength 0 0) rows
  let tokens ← Tensor.checkIndices vocabularySize windows
  pure <| Tensor.oneHotIndices (α := Float) vocabularySize tokens

/-- Fitted adder predictor returned by the public trainer. -/
abbrev Predictor :=
  Tensor Float input → IO (Tensor Float input)

/--
Greedily complete `operandDigits + 1` result digits from the operand digits.

For a prefix of length `k`, the next-token prediction is read at position $k-1$ of the zero-padded
window, which is the final position only when the prefix fills the window.
-/
def generateResultDigits (predict : Predictor) (a b : Nat) :
    IO (Tensor Nat [operandDigits + 1]) := do
  let prompt := Tensor.concat (fixedDigits operandDigits a) (fixedDigits operandDigits b)
  let options : Text.GenerationOptions :=
    { prompt := "", continuationLength := operandDigits + 1, temperature := 1.0, topK := 1,
      repeatPenalty := 0.0, repeatWindow := 0, seed := 0, asciiOnly := false }
  let generated ← Text.generate contextLength 0 prompt options
    (fun digits position => do
      let input ← CLI.orThrow exeName <| inputFromDigits digits
      let logits ← predict input
      pure ((logits.unstack 0).unstack position))
  pure (Tensor.window generated (operandDigits + 1) (2 * operandDigits) 0)

/-- Exact-match counts for train/test/all one-digit addition rows. -/
structure Score where
  /-- Correct rows in the training split. -/
  train : Nat
  /-- Correct rows in the held-out split. -/
  test : Nat
  /-- Correct rows across all one-digit additions. -/
  total : Nat
deriving Repr

/--
Evaluate all 100 additions with batched greedy decoding.

For one-digit operands, generation needs two result digits. The first batched forward pass predicts
the ones digit from rows `[a,b]`, and the second predicts the tens (carry) digit from rows
`[a,b,pred₀]`. The train count covers the first `trainCount` rows of the table and the test count
the rest.
-/
def evaluate
    (predict : Predictor) :
    IO Score := do
  let operandRows : Tensor Nat [batchSize, 2 * operandDigits] := Tensor.dim fun bi =>
    let (a, b) := pairAt bi.val
    show Tensor Nat [2 * operandDigits] from by
      simpa only [Nat.two_mul] using
        Tensor.concat (fixedDigits operandDigits a) (fixedDigits operandDigits b)
  let input0 ← CLI.orThrow exeName <| inputFromRows operandRows
  let logits0 ← predict input0
  let firstDigit : Tensor Nat [batchSize] := Tensor.ofFn fun bi =>
    argmax logits0 bi ⟨2 * operandDigits - 1, by decide⟩
  let withFirst : Tensor Nat [batchSize, 2 * operandDigits + 1] := Tensor.dim fun bi =>
    Tensor.concat operandRows[bi] (Tensor.full [1] firstDigit[bi])
  let input1 ← CLI.orThrow exeName <| inputFromRows withFirst
  let logits1 ← predict input1
  let secondDigit : Tensor Nat [batchSize] := Tensor.ofFn fun bi =>
    argmax logits1 bi ⟨2 * operandDigits, by decide⟩
  let correct : Tensor Bool [batchSize] := Tensor.ofFn fun bi =>
    let (a, b) := pairAt bi.val
    firstDigit[bi] + 10 * secondDigit[bi] == a + b
  let trainMask : Tensor Nat [batchSize] := Tensor.ofFn fun bi =>
    if correct[bi] && bi.val < trainCount then 1 else 0
  let train := trainMask.sum
  let total := (correct.map (fun valid => if valid then 1 else 0) : Tensor Nat [batchSize]).sum
  pure { train, test := total - train, total }

/-- Print one addition check in the same digit convention used for training. -/
def printProbe (predict : Predictor) (a b : Nat) : IO Unit := do
  let revDigits ← generateResultDigits predict a b
  let pred := decodeResult revDigits
  IO.println s!"  check {a}+{b}: reversed-digits={revDigits}, pred={pred}, target={a + b}"

/-- Adder-specific CLI options. -/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Terminal prompt-loop policy. -/
  interactive : Bool
  /--
  Optimizer selected by `--optim`, `adamw` by default.

  `adamw` uses minGPT's adder settings `beta2 = 0.95` and weight decay `0.1`, but decays every
  trainable parameter, where minGPT decays only the weights of its linear layers. The other
  choices are useful for debugging and comparisons.
  -/
  optim : optim.Optimizer
  /-- Operand `a` used by the highlighted addition check. -/
  a : Nat
  /-- Operand `b` used by the highlighted addition check. -/
  b : Nat
  /-- Extra comma-separated addition checks, e.g. `0+0,4+5,9+9`. -/
  probes : Array (Nat × Nat)
  /-- Train on the first 80 of the 100 one-digit additions and hold out the other 20. -/
  trainSplit : Bool
  /--
  Train only the selected pair, useful for checking that the CUDA GPT can overfit one addition.
  -/
  overfitProbe : Bool
deriving Repr

namespace Options

/-- Help text for the one-digit addition curriculum and its training controls. -/
def usage : String :=
  String.intercalate "\n" [
    "Usage: scripts/lake.sh exe torchlean gpt_adder [options]",
    "",
    "Curriculum:",
    "  --optim adamw|adam|sgd|adagrad|rmsprop|adadelta",
    "  --a N --b N          highlighted one-digit addition",
    "  --probes EXPRS       comma-separated checks such as 0+0,4+5,9+9",
    "  --train-split       train on the fixed 80/20 split",
    "  --overfit-probe     train only the selected pair",
    "  --interactive       keep the trained model in a terminal loop",
    "",
    "Training:",
    "  --steps N --lr X --log PATH|false --cuda-mem-watch N",
    "",
    "Runtime:",
    "  --device cuda --seed N --show-backend"
  ]

/-- Default extra addition checks shown after training when `--probes` is omitted. -/
def defaultProbes : Array (Nat × Nat) :=
  #[(0, 0), (1, 2), (4, 5), (7, 8), (9, 9)]

/--
The optimizer named by `--optim` at learning rate `lr`. Adam and AdamW use minGPT's `beta2 = 0.95`
and AdamW its weight decay `0.1`, which `optim.adamW` applies to every trainable parameter where
minGPT decays only the weights of its linear layers. The other optimizers keep their defaults.
-/
def parseOptimizer (lr : Float) (value : String) : Except String optim.Optimizer :=
  match value with
  | "sgd" => pure (optim.sgd { lr })
  | "adagrad" => pure (optim.adaGrad { lr })
  | "rmsprop" => pure (optim.rmsProp { lr })
  | "adam" => pure (optim.adam { lr, beta2 := 0.95 })
  | "adamw" => pure (optim.adamW { lr, weightDecay := 0.1, beta2 := 0.95 })
  | "adadelta" => pure (optim.adaDelta { lr })
  | _ =>
      throw s!"bad --optim {value}; expected sgd, adagrad, rmsprop, adam, adamw, or adadelta"

/--
Parse adder-specific CLI options. `--batch-size` is rejected because each dataset item is already
a full model batch. The defaults are 1000 steps at learning rate `5e-4` with AdamW, and the
highlighted check is `7+8`.
-/
def parse (args : List String) : Except String (Options × List String) := do
  if CLI.hasFlagValue args "batch-size" then
    throw (s!"{exeName}: --batch-size is unavailable because each dataset item is already "
      ++ "the complete typed model batch")
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 1000) (defaultLr := 5e-4)
  let (interactive, args) ← CLI.takeBoolFlag args "interactive"
  let (optim, args) ← CLI.takeParsedFlag args "optim" (default := "adamw")
    (parseOptimizer training.lr)
  let (a, args) ← CLI.takeNatFlag args "a" (default := 7)
  let (b, args) ← CLI.takeNatFlag args "b" (default := 8)
  let (probes?, args) ← CLI.takeFlagValue? args "probes"
  let (trainSplit, args) ← CLI.takeBoolFlag args "train-split"
  let (overfitProbe, args) ← CLI.takeBoolFlag args "overfit-probe"
  if a ≥ 10 || b ≥ 10 then
    throw "--a and --b must be one-digit numbers in 0..9"
  let probes ←
    match probes? with
    | some s => parseProbeArray s
    | none => pure defaultProbes
  pure ({ training
          interactive
          optim := optim
          a := a
          b := b
          probes := probes
          trainSplit := trainSplit
          overfitProbe := overfitProbe }, args)

/-- Standard TrainLog notes for the adder training loop. -/
def logNotes (config : Options) (runtime : Session.Config) : Array String :=
  #[s!"optimizer={config.optim.describe}", Training.deviceNote runtime]

end Options

/-- Training/evaluation curriculum used by the adder runner. -/
inductive Curriculum where
  /-- Train on the single pair `a + b` of the options (`--overfit-probe`). -/
  | overfitPair
  /-- Train on the first 80 one-digit additions and hold out the other 20 (`--train-split`). -/
  | trainSplit
  /-- Train on all 100 one-digit additions. -/
  | fullTable
deriving DecidableEq, Repr

namespace Curriculum

/-- Pick the curriculum the adder options request. `--overfit-probe` wins over `--train-split`. -/
def select (config : Options) : Curriculum :=
  if config.overfitProbe then
    .overfitPair
  else if config.trainSplit then
    .trainSplit
  else
    .fullTable

/-- Startup note for the selected curriculum. -/
def intro (mode : Curriculum) (config : Options) : String :=
  match mode with
  | .overfitPair =>
      s!"  curriculum=overfit-pair pair={config.a}+{config.b}"
  | .trainSplit =>
      s!"  curriculum=train/test split ({trainCount} train / {testCount} test; "
        ++ s!"train rows repeat to fill batch={batchSize})"
  | .fullTable =>
      "  curriculum=all 100 one-digit addition pairs"

/-- Training sample corresponding to the selected curriculum. -/
def sample (mode : Curriculum) (config : Options) :
    Except String (Sample.Supervised Float input input) :=
  match mode with
  | .overfitPair => additionSample config.a config.b
  | .trainSplit => tableSample true
  | .fullTable => tableSample false

/-- Final evaluation line for the selected curriculum, if any. -/
def final?
    (mode : Curriculum)
    (predict : Predictor) :
    IO (Option String) := do
  match mode with
  | .overfitPair =>
      pure none
  | .trainSplit =>
      let result ← evaluate predict
      pure <| some
        (s!"  final train={result.train}/{trainCount} test={result.test}/{testCount} "
          ++ s!"all={result.total}/100")
  | .fullTable =>
      let correct := (← evaluate predict).total
      pure <| some s!"  final exact={correct}/100"

end Curriculum

/--
Terminal REPL for the trained CUDA model. It reads one-digit prompts such as `7+8` and stops on an
empty line, `:q` or `:quit`.
-/
partial def interactiveLoop (predict : Predictor) :
    IO Unit := do
  IO.println "  interactive: enter one-digit prompts like 7+8; empty line or :q exits"
  let stdin ← IO.getStdin
  let rec loop : IO Unit := do
    IO.print "  add> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else
      match parseProbe? prompt with
      | none =>
          IO.println "  expected one-digit prompt like 7+8"
          loop
      | some (a, b) =>
          printProbe predict a b
          loop
  loop

/-- Train the minGPT-style adder from scratch and report exact addition accuracy. -/
def train (runtime : Session.Config) (options : Options) :
    IO Unit := do
  let curriculum := Curriculum.select options
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := options.optim
          objective := custom loss }
  IO.println (s!"  mode=adder operandDigits={operandDigits} vocabularySize={vocabularySize} "
    ++ s!"contextLength={contextLength} steps={options.training.steps}")
  trainer.printSummary
  IO.println (s!"  attentionHeads={attentionHeads} attentionHeadWidth={attentionHeadWidth} "
    ++ s!"modelWidth={modelWidth} feedForwardWidth={feedForwardWidth} "
    ++ s!"activeTargets/step={activeTargetCount}")
  IO.println s!"  optimizer={options.optim.describe}"
  IO.println s!"  minGPT encoding example 8+7 -> {
    renderExample 8 7} (sum digits reversed)"
  IO.println <| Curriculum.intro curriculum options

  /-
  The one-digit adder has a finite training batch, so the public custom trainer sees a dataset with
  one supervised sample: the full table, the repeated train split, or the selected overfit pair. The
  custom `loss` keeps the minGPT-style ignore-prefix normalization, and the optimizer loop runs
  inside `trainer.train`.
  -/
  let trainSample ← CLI.orThrow exeName <| Curriculum.sample curriculum options
  let trained ← trainer.train
    (Data.fromSamples #[trainSample])
    { steps := options.training.steps
      logDestination := options.training.logDestination
      logEvery := Nat.max 1 (options.training.steps / 10)
      cudaMemorySampleEvery := options.training.cudaMemorySampleEvery
      logTitle := "GPT adder training"
      logNotes := options.logNotes runtime }
  trained.printSummary
  match (← Curriculum.final? curriculum trained.predict) with
  | some line => IO.println line
  | none => pure ()
  printProbe trained.predict options.a options.b
  if !options.probes.isEmpty then
    IO.println "  extra checks:"
    for (a, b) in options.probes do
      printProbe trained.predict a b
  if options.interactive then
    interactiveLoop trained.predict

/--
CLI entrypoint for the CUDA GPT adder command. The command requires `--device cuda` and supplies it
when no device is given.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner exeName "minGPT-style addition training"
      usage? := some Options.usage
      printSuccess := true
      runtime := { device? := some cuda } })
    exeName args
    (fun runtime rest => do
      let (options, rest) ← CLI.orThrow exeName <| Options.parse rest
      CLI.requireNoArgs exeName rest
      train runtime options)

end TorchLean.Examples.GPTAdder
