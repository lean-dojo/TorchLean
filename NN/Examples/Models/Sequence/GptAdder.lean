/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA-only minGPT-style addition walkthrough:
  lake -R -K cuda=true exe torchlean gpt_adder --device cuda --steps 1 --optim adam --lr 0.005 \
    --a 7 --b 8
  lake -R -K cuda=true exe torchlean gpt_adder --device cuda --steps 1 --optim sgd --lr 0.05 \
    --a 7 --b 8

Interactive addition REPL:
  lake -R -K cuda=true exe torchlean gpt_adder --device cuda --steps 1 --interactive
-/

module

public import NN.API
public import NN.Examples.Support

/-!
# minGPT-Style Addition Example

This is a TorchLean-native version of the spirit of Karpathy's `minGPT/projects/adder`
experiment. The original minGPT adder trains a compact GPT to complete digit strings of the form

`digits(a) ++ digits(b) ++ reverseDigits(a+b)`.

For example, in the one-digit setting $8+7=15$ is represented as the digit sequence
`8 7 5 1`.  At inference time the model sees `8 7` and greedily generates the two result digits
`5 1`, which we reverse back to `15`.

This controlled arithmetic sequence task exercises the CUDA GPT training loop:

* synthetic data is generated in Lean,
* the model is a GPT-style causal Transformer built from TorchLean layers,
* training is CUDA-only by default,
* optimizer choices follow the minGPT-style setup (`adamw`, `adam`, or `sgd`),
* evaluation greedily completes every one-digit addition problem.

Performance note: this uses the eager CUDA runtime, not a persistent CUDA graph.
The heavy tensor operations run on the GPU, including fused attention,
but each step still records a fresh autograd tape and synchronizes parameter refs through the
current scalar training bridge. This is the correctness-facing example; full PyTorch-style
throughput requires persistent device parameters plus future graph fusion and scheduling.

The GPT-shaped architecture is constructed through the public TorchLean model constructor
`nn.models.CausalTransformer.oneHot`, so the example can stay focused on the adder task mechanics.

Reference: <https://github.com/karpathy/minGPT/tree/master/projects/adder>.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.GptAdder

/-- CLI subcommand label used by the shared model runner. -/
def exeName : String := "gpt_adder"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "gpt_adder"

/--
Number of input digits per operand.

The one-digit curriculum trains directly in the eager CUDA runtime while still including carry
examples such as $8+7=15$.
-/
def operandDigits : Nat := 1

/-- Digit-only vocabulary, matching minGPT's adder task (`0..9`). -/
def vocabularySize : Nat := 10

/--
Full one-digit table batch size.

This is `100`, not `1`: scalar-sized GPU workloads underutilize the device. In all-pairs mode one
optimizer step sees every one-digit addition problem, and evaluation completes the whole table with
two batched greedy forward passes.
-/
def batchSize : Nat := 100

local instance : NeZero batchSize := ⟨by decide⟩

/-- Karpathy's adder uses a held-out split; for one digit this is 80 train / 20 test. -/
def trainCount : Nat := 80

/-- Held-out one-digit examples when `--train-split` is enabled. -/
def testCount : Nat := 20

/--
GPT block size: `a`, `b`, and all but the final reversed result digit.

For `operandDigits = 1`, full rendered examples have length $1+1+2=4$; model inputs have
length `3`, exactly as in minGPT's `get_block_size = 3 * operandDigits + 1 - 1`.
-/
def contextLength : Nat := 3 * operandDigits

/--
Number of attention heads.

Karpathy's minGPT default for the adder is `gpt-nano` (`3` heads, width `48`). TorchLean's eager
CUDA trainer is tape-based, so we use a middle-sized model that is substantially larger than the
original compact setup (1,050 params) while keeping `torchlean gpt_adder` practical to run.
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

The adder loss below masks ignored prefix positions to all-zero targets, then computes summed
one-hot cross-entropy divided by this count. That matches minGPT's `ignore_index=-1` normalization:
average over active next-token labels, not over every `(batch, position, vocab)` entry.
-/
def activeTargetCount : Nat :=
  batchSize * activeTargetPositions

local instance : NeZero contextLength := ⟨by decide⟩
local instance : NeZero modelWidth := ⟨by decide⟩

/-- GPT configuration shared by the typed shapes and model constructor. -/
abbrev modelConfig : nn.models.CausalTransformer.Config :=
  { sequenceLength := contextLength
    vocabularySize := vocabularySize
    headCount := attentionHeads
    headWidth := attentionHeadWidth
    feedForwardWidth := feedForwardWidth
    layerCount := transformerLayers
    }

/-- Input shape: batched one-hot digit sequences. -/
abbrev input : Shape :=
  [batchSize, contextLength, vocabularySize]

/-- Output shape: one digit-logit row per input position. -/
abbrev output : Shape :=
  input

/-- Compact GPT-style causal Transformer for digit addition. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.CausalTransformer.oneHot modelConfig (batchShape := [batchSize])

/-- Cross-entropy summed over non-ignored adder targets, normalized like minGPT `ignore_index`. -/
def adderLoss {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Runtime.Ops (m := m) (α := α)]
    (logits targetOneHot : Runtime.ValueRef (m := m) (α := α) output) :
    m (Runtime.ValueRef (m := m) (α := α) ([] : Shape)) := do
  let summed ← Loss.oneHotCrossEntropy
    (m := m) (α := α) (s := output) 2 logits targetOneHot (reduction := .sum)
  Runtime.scale (m := m) (α := α) (s := ([] : Shape))
    summed ((1 : α) / (activeTargetCount : α))

/--
Adder-specific scalar loss.

Ignored prefix positions are encoded by all-zero one-hot rows (`maskAdderTargets`), so they
contribute exactly zero to one-hot cross entropy.  We divide the summed loss by the number of
active target positions, matching minGPT's `ignore_index`-style normalization rather than averaging
over ignored prefix rows.
-/
def adderLossProgram {α : Type} [Storage α] [Context α] :
    Runtime.Program α [output, output] ([] : Shape) :=
  fun {m} _ _ =>
    fun logits targetOneHot =>
      adderLoss (m := m) (α := α) logits targetOneHot

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
Karpathy/minGPT masks the loss on the operand-prefix positions.

In `projects/adder/adder.py`, the target vector `y` is shifted by one token and then
`y[:operandDigits*2-1] = -1`, where `-1` is PyTorch's "ignore index" for cross entropy. TorchLean's
current one-hot cross entropy does not have an ignore-index target, so we represent the same idea by
using an all-zero one-hot vector on ignored positions. Because the loss is
$-\sum y\log p$, these
positions contribute exactly zero gradient.
-/
def keepTargetPosition (t : Nat) : Bool :=
  t ≥ 2 * operandDigits - 1

/-- Apply the minGPT adder loss mask to a shifted one-hot target matrix. -/
def maskAdderTargets {α : Type} [Storage α] [Zero α]
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
  pure <| Sample.mapTarget maskAdderTargets <|
    Data.CausalLM.oneHotSample (α := Float) [] contextLength vocabularySize tokens

/-- Build one supervised next-digit sample from an addition problem. -/
def additionSample (a b : Nat) :
    Except String (Sample.Supervised Float input output) := do
  let row ← rowSample a b
  pure
    { input := Tensor.repeatAxis 0 batchSize row.input
      target := Tensor.repeatAxis 0 batchSize row.target }

/-- Deterministic exhaustive one-digit dataset order. -/
def pairAt (i : Nat) : Nat × Nat :=
  let j := i % 100
  (j / 10, j % 10)

/-- Training row assignment. In split mode, rows repeat the first 80 train examples. -/
def trainPairAt (trainSplit : Bool) (i : Nat) : Nat × Nat :=
  if trainSplit then
    pairAt (i % trainCount)
  else
    pairAt i

/-- Parse `a+b` into a one-digit operand pair; returns `none` for malformed prompts. -/
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

/-- Comma-separated list of one-digit `a+b` checks. -/
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
    Except String (Sample.Supervised Float input output) := do
  let windows : Tensor Nat [batchSize, contextLength + 1] := Tensor.stack 0 fun bi =>
    let (a, b) := trainPairAt trainSplit bi.val
    renderExample a b
  let tokens ← Tensor.checkIndices vocabularySize windows
  let sample := Data.CausalLM.oneHotSample
    (α := Float) [batchSize] contextLength vocabularySize tokens
  pure <| Sample.mapTarget (fun targets =>
    Tensor.stack 0 fun bi => maskAdderTargets (targets.get bi)) sample

/-- Decode reversed generated result digits back into a natural number. -/
def decodeResult {count : Nat} (revDigits : Tensor Nat [count]) : Nat :=
  let digits := Tensor.ofFn fun index : Fin count => revDigits[index.rev]
  digits.foldl (fun acc digit => acc * 10 + digit) 0

/-- Argmax token id at a sequence position for a chosen batch row. -/
def argmaxAtBatch
    (logits : Tensor Float output)
    (bi : Fin batchSize)
    (pos : Fin contextLength) : Nat :=
  match Metrics.argmax? (text.batchLogitScoresAt logits bi pos) with
  | some token => token.val
  | none => 0

/-- Argmax token id at sequence position `pos` in the first batch row. -/
def argmaxAt (logits : Tensor Float output) (pos : Fin contextLength) : Nat :=
  argmaxAtBatch logits (Fin.ofNat batchSize 0) pos

/-- Build a model input tensor from the current generated digit prefix. -/
def inputFromDigits {count : Nat} (digits : Tensor Nat [count]) :
    Except String (Tensor Float input) := do
  let window : Tensor Nat [contextLength] :=
    Tensor.window digits contextLength 0 0
  let tokens ← Tensor.checkIndices vocabularySize window
  pure <| Tensor.repeatAxis 0 batchSize <|
    Data.CausalLM.oneHotInputs (α := Float) vocabularySize tokens

/-- Build a batched model input from one digit prefix per row. -/
def inputFromRows {count : Nat} (rows : Tensor Nat [batchSize, count]) :
    Except String (Tensor Float input) := do
  let windows : Tensor Nat [batchSize, contextLength] :=
    Tensor.mapLeading [batchSize] (fun row => Tensor.window row contextLength 0 0) rows
  let tokens ← Tensor.checkIndices vocabularySize windows
  pure <| Data.CausalLM.oneHotInputs (α := Float) vocabularySize tokens

/-- Fitted adder predictor returned by the public trainer. -/
abbrev Predictor :=
  Tensor Float input → IO (Tensor Float output)

/--
Greedily complete `operandDigits + 1` result digits from the operand digits.

The key detail is that when the current prefix has length `k`, the next-token prediction lives at
position $k-1$, not always at the final padded position.
-/
def generateResultDigits (predict : Predictor) (a b : Nat) :
    IO (Tensor Nat [operandDigits + 1]) := do
  let prompt := Tensor.concat (fixedDigits operandDigits a) (fixedDigits operandDigits b)
  let options : text.GenerationOptions :=
    { prompt := "", newTokenCount := operandDigits + 1, temperature := 1.0, topK := 1,
      repeatPenalty := 0.0, repeatWindow := 0, seed := 0, asciiOnly := false }
  let generated ← text.autoregressiveTokenIds contextLength 0 prompt options
    (fun digits position => do
      let input ← CLI.orThrow exeName <| inputFromDigits digits
      let logits ← predict input
      pure (text.batchLogitScoresAt logits 0 position))
  pure (Tensor.window generated (operandDigits + 1) (2 * operandDigits) 0)

/-- Predict $a+b$ by greedy decoding and reversing the minGPT result digits. -/
def predictSum (predict : Predictor) (a b : Nat) : IO Nat := do
  let revDigits ← generateResultDigits predict a b
  pure (decodeResult revDigits)

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

For one-digit operands, generation needs two result digits.  We first predict the ones digit from
rows `[a,b]`, append it, and then predict the carry/tens digit from rows `[a,b,pred₀]`.
-/
def score
    (predict : Predictor) :
    IO Score := do
  let operandRows : Tensor Nat [batchSize, 2 * operandDigits] := Tensor.stackLeading fun bi =>
    let (a, b) := pairAt bi.val
    show Tensor Nat [2 * operandDigits] from by
      simpa only [Nat.two_mul] using
        Tensor.concat (fixedDigits operandDigits a) (fixedDigits operandDigits b)
  let input0 ← CLI.orThrow exeName <| inputFromRows operandRows
  let logits0 ← predict input0
  let firstDigit : Tensor Nat [batchSize] := Tensor.ofFn fun bi =>
    argmaxAtBatch logits0 bi ⟨2 * operandDigits - 1, by decide⟩
  let withFirst : Tensor Nat [batchSize, 2 * operandDigits + 1] := Tensor.stackLeading fun bi =>
    Tensor.concat operandRows[bi] (Tensor.full [1] firstDigit[bi])
  let input1 ← CLI.orThrow exeName <| inputFromRows withFirst
  let logits1 ← predict input1
  let secondDigit : Tensor Nat [batchSize] := Tensor.ofFn fun bi =>
    argmaxAtBatch logits1 bi ⟨2 * operandDigits, by decide⟩
  let correct : Tensor Bool [batchSize] := Tensor.ofFn fun bi =>
    let (a, b) := pairAt bi.val
    firstDigit[bi] + 10 * secondDigit[bi] == a + b
  let trainMask : Tensor Nat [batchSize] := Tensor.ofFn fun bi =>
    if correct[bi] && bi.val < trainCount then 1 else 0
  let train := trainMask.sum
  let total := (correct.map (fun valid => if valid then 1 else 0) : Tensor Nat [batchSize]).sum
  pure { train, test := total - train, total }

/-- Batched exact-match score over all one-digit additions. -/
def totalCorrect (predict : Predictor) :
    IO Nat := do
  pure (← score predict).total

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
  interaction : text.InteractiveOptions
  /--
  Optimizer.

  `adamw` is closest to minGPT's adder recipe. `adam` and `sgd` are useful for debugging and
  comparisons.
  -/
  optim : optim.Algorithm
  /-- Operand `a` used by the highlighted addition check. -/
  a : Nat
  /-- Operand `b` used by the highlighted addition check. -/
  b : Nat
  /-- Extra comma-separated addition checks, e.g. `0+0,4+5,9+9`. -/
  probes : Array (Nat × Nat)
  /-- Train on an 80/20 train/test split instead of all 100 one-digit additions. -/
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
    "Usage: lake exe torchlean gpt_adder [options]",
    "",
    "Curriculum:",
    "  --optim adamw|adam|sgd",
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
    "  --device cuda --execution eager --seed N --show-backend"
  ]

/-- Default extra addition checks shown after training when `--probes` is omitted. -/
def defaultProbes : Array (Nat × Nat) :=
  #[(0, 0), (1, 2), (4, 5), (7, 8), (9, 9)]

/-- Parse adder-specific CLI options. -/
def parse (args : List String) : Except String (Options × List String) := do
  if CLI.hasFlagValue args "batch-size" then
    throw (s!"{exeName}: --batch-size is unavailable because each dataset item is already "
      ++ "the complete typed model batch")
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 1000) (defaultLearningRate := 5e-4)
  let (interactive, args) ← text.InteractiveOptions.parse args
  let (optim, args) ← CLI.takeParsedFlag args "optim" (default := "adamw") optim.Algorithm.parse
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
          interaction := interactive
          optim := optim
          a := a
          b := b
          probes := probes
          trainSplit := trainSplit
          overfitProbe := overfitProbe }, args)

/-- Standard TrainLog notes for the adder training loop. -/
def logNotes (config : Options) (runtime : Runtime.Config) : Array String :=
  #[s!"optimizer={optim.Algorithm.displayName config.optim}",
    s!"lr={config.training.learningRate}", Support.deviceNote runtime]

end Options

/-- Training/evaluation curriculum used by the adder runner. -/
inductive Curriculum where
  | overfitPair
  | trainSplit
  | fullTable
deriving DecidableEq, Repr

namespace Curriculum

/-- Decide which curriculum the current adder options request. -/
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
    Except String (Sample.Supervised Float input output) :=
  match mode with
  | .overfitPair => additionSample config.a config.b
  | .trainSplit => tableSample true
  | .fullTable => tableSample false

/-- Per-step progress line for the selected curriculum. -/
def progress
    (mode : Curriculum)
    (predict : Predictor)
    (config : Options)
    (done : Nat)
    (lossVal : Float) : IO String := do
  match mode with
  | .overfitPair =>
      let pred ← predictSum predict config.a config.b
      pure s!"  step={done} loss={lossVal} pairPred={pred} target={config.a + config.b}"
  | .trainSplit =>
      let result ← score predict
      pure (s!"  step={done} loss={lossVal} train={result.train}/{trainCount} "
        ++ s!"test={result.test}/{testCount} all={result.total}/100")
  | .fullTable =>
      let correct ← totalCorrect predict
      pure s!"  step={done} loss={lossVal} exact={correct}/100"

/-- Final evaluation line for the selected curriculum, if any. -/
def final?
    (mode : Curriculum)
    (predict : Predictor) :
    IO (Option String) := do
  match mode with
  | .overfitPair =>
      pure none
  | .trainSplit =>
      let result ← score predict
      pure <| some
        (s!"  final train={result.train}/{trainCount} test={result.test}/{testCount} "
          ++ s!"all={result.total}/100")
  | .fullTable =>
      let correct ← totalCorrect predict
      pure <| some s!"  final exact={correct}/100"

end Curriculum

/-- Simple terminal REPL for the trained CUDA model. -/
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
def train (runtime : Runtime.Config) (options : Options) :
    IO Unit := do
  let curriculum := Curriculum.select options
  let trainer :=
    Trainer.new model <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := options.optim.configure options.training.learningRate })
        (.custom adderLossProgram) (seed := runtime.seed)
  IO.println (s!"  mode=adder operandDigits={operandDigits} vocabularySize={vocabularySize} "
    ++ s!"contextLength={contextLength} steps={options.training.steps}")
  trainer.printSummary
  IO.println (s!"  attentionHeads={attentionHeads} attentionHeadWidth={attentionHeadWidth} "
    ++ s!"modelWidth={modelWidth} feedForwardWidth={feedForwardWidth} "
    ++ s!"activeTargets/step={activeTargetCount}")
  IO.println (s!"  optimizer={optim.Algorithm.displayName options.optim} "
    ++ s!"lr={options.training.learningRate}")
  IO.println s!"  minGPT encoding example 8+7 -> {
    renderExample 8 7} (sum digits reversed)"
  IO.println <| Curriculum.intro curriculum options

  /-
  The one-digit adder has a finite training batch, so the public custom trainer sees a dataset with
  exactly one supervised sample: that sample is either the full table, the repeated train split, or
  the selected overfit pair.  The custom loss `adderLossProgram` preserves the minGPT-style
  ignore-prefix normalization while still moving the optimizer loop behind `trainer.train`.
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
  if options.interaction.interactive then
    interactiveLoop trained.predict

/-- CLI entrypoint for the CUDA GPT adder command. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDevice exeName "minGPT-style addition training"
      usage? := some Options.usage
      printSuccess := true
      runtime := { device? := some .cuda, execution? := some .eager } })
    exeName args
    (.native fun runtime rest => do
      let (options, rest) ← CLI.orThrow exeName <| Options.parse rest
      CLI.requireNoArgs exeName rest
      train runtime options)

end NN.Examples.Models.Sequence.GptAdder
