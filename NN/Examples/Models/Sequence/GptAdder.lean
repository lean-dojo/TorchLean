/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA-only minGPT-style addition walkthrough:
  lake build -R -K cuda=true
  lake exe torchlean gpt_adder --steps 1 --optim adam --lr 0.005 --a 7 --b 8
  lake exe torchlean gpt_adder --steps 1 --optim sgd --lr 0.05 --a 7 --b 8

Interactive addition REPL:
  lake exe torchlean gpt_adder --steps 1 --interactive
-/

module

public import NN

/-!
# minGPT-Style Addition Example

This is a TorchLean-native version of the spirit of Karpathy's `minGPT/projects/adder`
experiment. The original minGPT adder trains a compact GPT to complete digit strings of the form

`digits(a) ++ digits(b) ++ reverseDigits(a+b)`.

For example, in the one-digit setting `8 + 7 = 15` is represented as the digit sequence
`8 7 5 1`.  At inference time the model sees `8 7` and greedily generates the two result digits
`5 1`, which we reverse back to `15`.

This is not a text chatbot. It is a controlled algorithmic sequence task for the CUDA GPT training
loop:

* synthetic data is generated in Lean,
* the model is a GPT-style causal Transformer built from TorchLean layers,
* training is CUDA-only by default,
* optimizer choices follow the minGPT-style setup (`adamw`, `adam`, or `sgd`),
* evaluation greedily completes every one-digit addition problem.

Performance note: this uses the eager CUDA runtime, not a persistent CUDA graph.
The heavy tensor operations run on the GPU, including fused attention when `--fast-kernels` is on,
but each step still records a fresh autograd tape and synchronizes parameter refs through the
current scalar training bridge. This is the correctness-facing example; full PyTorch-style
throughput requires persistent device parameters plus compiled/fused graph execution.

The GPT-shaped architecture is constructed through the public TorchLean model constructor
`nn.models.CausalTransformerOneHot`, so the example can stay focused on the adder task mechanics.

Reference: <https://github.com/karpathy/minGPT/tree/master/projects/adder>.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.GptAdder

/-- CLI subcommand label used by the shared model runner. -/
def exeName : String := "torchlean gpt_adder"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "gpt_adder"

/--
Number of input digits per operand.

The one-digit curriculum trains directly in the eager CUDA runtime while still including carry
examples such as `8 + 7 = 15`.
-/
def ndigit : Nat := 1

/-- Digit-only vocabulary, matching minGPT's adder task (`0..9`). -/
def vocab : Nat := 10

/--
Full one-digit table batch size.

This is `100`, not `1`: scalar-sized GPU workloads underutilize the device. In all-pairs mode one
optimizer step sees every one-digit addition problem, and evaluation completes the whole table with
two batched greedy forward passes.
-/
def batch : Nat := 100

/-- Karpathy's adder uses a held-out split; for one digit this is 80 train / 20 test. -/
def trainCount : Nat := 80

/-- Held-out one-digit examples when `--train-split` is enabled. -/
def testCount : Nat := 20

/--
GPT block size: `a`, `b`, and all but the final reversed result digit.

For `ndigit = 1`, full rendered examples have length `1 + 1 + 2 = 4`; model inputs have length
`3`, exactly as in minGPT's `get_block_size = 3 * ndigit + 1 - 1`.
-/
def seqLen : Nat := 3 * ndigit

/--
Number of attention heads.

Karpathy's minGPT default for the adder is `gpt-nano` (`3` heads, width `48`). TorchLean's eager
CUDA trainer is tape-based, so we use a middle-sized model that is substantially larger than the
original compact setup (1,050 params) while keeping `torchlean gpt_adder` practical to run.
-/
def numHeads : Nat := 2

/-- Per-head width. -/
def headDim : Nat := 16

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Feed-forward hidden width (`4 * dModel`, matching the common GPT MLP ratio). -/
def ffnHidden : Nat := 128

/-- Number of Transformer blocks. -/
def layers : Nat := 2

/-- Number of positions per row that contribute to the minGPT adder loss. -/
def activeTargetPositions : Nat :=
  seqLen - (2 * ndigit - 1)

/--
Number of non-ignored next-token targets in the training batch.

The adder loss below masks ignored prefix positions to all-zero targets, then computes summed
one-hot cross-entropy divided by this count. That matches minGPT's `ignore_index=-1` normalization:
average over active next-token labels, not over every `(batch, position, vocab)` entry.
-/
def activeTargetCount : Nat :=
  batch * activeTargetPositions

/-- Count scalar entries across a list of parameter shapes. -/
def paramCountShapes : List Shape → Nat
  | [] => 0
  | s :: ss => Shape.size s + paramCountShapes ss

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

/-- GPT configuration shared by the typed shapes and model constructor. -/
def cfg : nn.models.CausalOneHotConfig :=
  { batch := batch
    seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers
    seedStride := 100 }

/-- Input shape: batched one-hot digit sequences. -/
abbrev σ : Shape :=
  nn.models.causalOneHotShape cfg

/-- Output shape: one digit-logit row per input position. -/
abbrev τ : Shape :=
  σ

/-- Compact GPT-style causal Transformer for digit addition. -/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.CausalTransformerOneHot cfg

/-- Cross-entropy summed over non-ignored adder targets, normalized like minGPT `ignore_index`. -/
def adderLoss {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Runtime.Ops (m := m) (α := α)]
    (logits targetOneHot : Runtime.RefTy (m := m) (α := α) τ) :
    m (Runtime.RefTy (m := m) (α := α) Shape.scalar) := do
  let summed ← Loss.crossEntropyOneHot
    (m := m) (α := α) (s := τ) logits targetOneHot (reduction := .sum)
  Ops.scale (m := m) (α := α) (s := Shape.scalar)
    summed ((1 : α) / (activeTargetCount : α))

/--
Adder-specific scalar loss.

Ignored prefix positions are encoded by all-zero one-hot rows (`maskAdderTargets`), so they
contribute exactly zero to one-hot cross entropy.  We divide the summed loss by the number of
active target positions, matching minGPT's `ignore_index`-style normalization rather than averaging
over ignored prefix rows.
-/
def adderLossProgram {α : Type} [Runtime.TensorScalar α] [DecidableEq Shape] :
    Runtime.Program α [τ, τ] Shape.scalar :=
  fun {m} _ _ =>
    fun logits targetOneHot =>
      adderLoss (m := m) (α := α) logits targetOneHot

/-- Render `n` as exactly `width` base-10 digits, most-significant first. -/
def fixedDigits (width n : Nat) : List Nat :=
  (List.range width).map (fun i =>
    let pow := Nat.pow 10 (width - i - 1)
    (n / pow) % 10)

/--
minGPT adder rendering.

For `ndigit = 1`, `a = 8`, `b = 7` becomes `[8, 7, 5, 1]`, i.e. the sum `15` is stored reversed
as `5, 1`.  Reversing the output digits makes carry propagation local in left-to-right generation.
-/
def renderExample (a b : Nat) : List Nat :=
  fixedDigits ndigit a ++ fixedDigits ndigit b ++ (fixedDigits (ndigit + 1) (a + b)).reverse

/--
Karpathy/minGPT masks the loss on the operand-prefix positions.

In `projects/adder/adder.py`, the target vector `y` is shifted by one token and then
`y[:ndigit*2-1] = -1`, where `-1` is PyTorch's "ignore index" for cross entropy. TorchLean's
current one-hot cross entropy does not have an ignore-index target, so we represent the same idea by
using an all-zero one-hot vector on ignored positions. Because the loss is `-sum(y * log p)`, these
positions contribute exactly zero gradient.
-/
def keepTargetPosition (t : Nat) : Bool :=
  t ≥ 2 * ndigit - 1

/-- Apply the minGPT adder loss mask to a shifted one-hot target matrix. -/
def maskAdderTargets {α : Type} [Zero α]
    (y : Tensor.T α (Shape.mat seqLen vocab)) :
    Tensor.T α (Shape.mat seqLen vocab) :=
  Spec.Tensor.dim (fun t =>
    if keepTargetPosition t.val then
      match y with
      | Spec.Tensor.dim rows => rows t
    else
      Tensor.fill 0 (shape![vocab]))

/--
Build one unbatched one-hot causal-LM sample for an addition row, then apply the minGPT-style
ignored-prefix mask to its target matrix.
-/
def mkRowSample (a b : Nat) :
    SupervisedSample Float (Shape.mat seqLen vocab) (Shape.mat seqLen vocab) :=
  Sample.mapY maskAdderTargets <|
    Data.causalLmOneHotMatSample (α := Float) seqLen vocab (renderExample a b)

/-- Build one supervised next-digit sample from an addition problem. -/
def mkSample (a b : Nat) : SupervisedSample Float σ τ :=
  let row := mkRowSample a b
  let x2D := Sample.x row
  let y2D := Sample.y row
  Sample.mk (Spec.Tensor.dim (fun _ => x2D)) (Spec.Tensor.dim (fun _ => y2D))

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
  | [aStr, bStr] =>
      match aStr.toNat?, bStr.toNat? with
      | some a, some b =>
          if a < 10 && b < 10 then some (a, b) else none
      | _, _ => none
  | _ => none

/-- Comma-separated list of one-digit `a+b` checks. -/
def parseProbeList (s : String) : Except String (List (Nat × Nat)) := do
  let raw := s.splitOn "," |>.filter (fun p => p.trimAscii.toString != "")
  let mut out : List (Nat × Nat) := []
  for p in raw do
    match parseProbe? p with
    | some pair => out := out ++ [pair]
    | none => throw s!"bad --probes entry {p}; expected comma-separated one-digit prompts like 0+0,7+8"
  pure out

/-- Build a batched supervised sample with one row per one-digit addition problem. -/
def mkTrainSample (trainSplit : Bool) : SupervisedSample Float σ τ :=
  let row (bi : Fin batch) : Tensor.T Float (Shape.mat seqLen vocab) :=
    let (a, b) := trainPairAt trainSplit bi.val
    Sample.x <| mkRowSample a b
  let target (bi : Fin batch) : Tensor.T Float (Shape.mat seqLen vocab) :=
    let (a, b) := trainPairAt trainSplit bi.val
    Sample.y <| mkRowSample a b
  Sample.mk (Spec.Tensor.dim row) (Spec.Tensor.dim target)

/-- Decode reversed generated result digits back into a natural number. -/
def decodeResult (revDigits : List Nat) : Nat :=
  revDigits.reverse.foldl (fun acc d => acc * 10 + d) 0

/-- Argmax token id at sequence position `pos`. -/
def argmaxAt (logits : Tensor.T Float τ) (pos : Nat) : Nat :=
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float) (batchIdx := ⟨0, by decide⟩) logits
  ids.getD pos 0

/-- Argmax token id at a sequence position for a chosen batch row. -/
def argmaxAtBatch (logits : Tensor.T Float τ) (bi : Fin batch) (pos : Nat) : Nat :=
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float) (batchIdx := bi) logits
  ids.getD pos 0

/-- Build a model input tensor from the current generated digit prefix. -/
def inputFromDigits (digits : List Nat) : Tensor.T Float σ :=
  text.causalLmXOneHotBatch (α := Float) batch seqLen vocab digits

/-- Build a batched model input from one digit prefix per row. -/
def inputFromRows (rows : Fin batch → List Nat) : Tensor.T Float σ :=
  text.causalLmXOneHotBatchRows (α := Float) batch seqLen vocab rows

/-- Fitted adder predictor returned by the public trainer handle. -/
abbrev Predictor :=
  Tensor.T Float σ → IO (Tensor.T Float τ)

/--
Greedily complete `ndigit + 1` result digits from the operand digits.

The key detail is that when the current prefix has length `k`, the next-token prediction lives at
position `k - 1`, not always at the final padded position.
-/
def generateResultDigits
    (predict : Predictor)
    (a b : Nat) : IO (List Nat) := do
  let mut digits := fixedDigits ndigit a ++ fixedDigits ndigit b
  let mut out : List Nat := []
  for _ in [0:ndigit + 1] do
    let pos := if digits.length = 0 then 0 else Nat.min (digits.length - 1) (seqLen - 1)
    let logits ← predict (inputFromDigits digits)
    let next := argmaxAt logits pos
    digits := digits ++ [next]
    out := out ++ [next]
  pure out

/-- Predict `a + b` by greedy decoding and reversing the minGPT result digits. -/
def predictSum (predict : Predictor) (a b : Nat) : IO Nat := do
  let revDigits ← generateResultDigits predict a b
  pure (decodeResult revDigits)

/-- Evaluate all 100 one-digit additions. -/
def evalAllSlow (predict : Predictor) :
    IO Nat := do
  let mut correct := 0
  for i in [0:100] do
    let (a, b) := pairAt i
    let pred ← predictSum predict a b
    if pred = a + b then
      correct := correct + 1
  pure correct

/-- Exact-match counts for train/test/all one-digit addition rows. -/
structure EvalScore where
  /-- Correct rows in the training split. -/
  trainCorrect : Nat
  /-- Correct rows in the held-out split. -/
  testCorrect : Nat
  /-- Correct rows across all one-digit additions. -/
  allCorrect : Nat
deriving Repr

/--
Evaluate all 100 additions with batched greedy decoding.

For one-digit operands, generation needs two result digits.  We first predict the ones digit from
rows `[a,b]`, append it, and then predict the carry/tens digit from rows `[a,b,pred₀]`.
-/
def evalBatched
    (predict : Predictor) :
    IO EvalScore := do
  let operandRows : Fin batch → List Nat := fun bi =>
    let (a, b) := pairAt bi.val
    fixedDigits ndigit a ++ fixedDigits ndigit b
  let logits0 ← predict (inputFromRows operandRows)
  let firstDigit : Fin batch → Nat := fun bi => argmaxAtBatch logits0 bi (2 * ndigit - 1)
  let withFirst : Fin batch → List Nat := fun bi => operandRows bi ++ [firstDigit bi]
  let logits1 ← predict (inputFromRows withFirst)
  let secondDigit : Fin batch → Nat := fun bi => argmaxAtBatch logits1 bi (2 * ndigit)
  let mut trainCorrect := 0
  let mut testCorrect := 0
  let mut allCorrect := 0
  for i in [0:100] do
    if h : i < batch then
      let bi : Fin batch := ⟨i, h⟩
      let (a, b) := pairAt i
      let pred := decodeResult [firstDigit bi, secondDigit bi]
      if pred = a + b then
        allCorrect := allCorrect + 1
        if i < trainCount then
          trainCorrect := trainCorrect + 1
        else
          testCorrect := testCorrect + 1
  pure { trainCorrect := trainCorrect, testCorrect := testCorrect, allCorrect := allCorrect }

/-- Batched exact-match score over all one-digit additions. -/
def evalAllBatched (predict : Predictor) :
    IO Nat := do
  pure (← evalBatched predict).allCorrect

/-- Print one addition check in the same digit convention used for training. -/
def printProbe (predict : Predictor) (a b : Nat) : IO Unit := do
  let revDigits ← generateResultDigits predict a b
  let pred := decodeResult revDigits
  IO.println s!"  check {a}+{b}: reversed-digits={revDigits}, pred={pred}, target={a + b}"

/-- Adder-specific CLI options layered on top of the shared interactive text training flags. -/
structure AdderOptions extends text.InteractiveTrainOptions where
  /--
  Optimizer.

  `adamw` is closest to minGPT's adder recipe. `adam` and `sgd` are useful for debugging and
  comparisons.
  -/
  optim : optim.Kind
  /-- Operand `a` used by the highlighted addition check. -/
  a : Nat
  /-- Operand `b` used by the highlighted addition check. -/
  b : Nat
  /-- Extra comma-separated addition checks, e.g. `0+0,4+5,9+9`. -/
  probes : List (Nat × Nat)
  /-- Train on an 80/20 train/test split instead of all 100 one-digit additions. -/
  trainSplit : Bool
  /-- Train only the selected pair, useful for checking that the CUDA GPT can overfit one addition. -/
  overfitProbe : Bool
deriving Repr

namespace AdderOptions

/-- Default extra addition checks shown after training when `--probes` is omitted. -/
def defaultProbes : List (Nat × Nat) :=
  [(0, 0), (1, 2), (4, 5), (7, 8), (9, 9)]

/-- Parse adder-specific CLI options. -/
def parse (args : List String) : Except String (AdderOptions × List String) := do
  let (base, args) ←
    text.InteractiveTrainOptions.parse exeName args defaultLogJson 1000 5e-4
  let (optim, args) ← CLI.takeParsedFlagDefault args "optim" "adamw" optim.Kind.parse
  let (a, args) ← CLI.takeNatFlagDefault args "a" 7
  let (b, args) ← CLI.takeNatFlagDefault args "b" 8
  let (probes?, args) ← CLI.takeFlagValueOnce args "probes"
  let (trainSplit, args) ← CLI.takeBoolFlagOnce args "train-split"
  let (overfitProbe, args) ← CLI.takeBoolFlagOnce args "overfit-probe"
  if a ≥ 10 || b ≥ 10 then
    throw "--a and --b must be one-digit numbers in 0..9"
  let probes ←
    match probes? with
    | some s => parseProbeList s
    | none => pure defaultProbes
  pure ({ toInteractiveTrainOptions := base
          optim := optim
          a := a
          b := b
          probes := probes
          trainSplit := trainSplit
          overfitProbe := overfitProbe }, args)

/-- Standard TrainLog notes for the adder training loop. -/
def logNotes (cfg : AdderOptions) (opts : Options) : Array String :=
  #[s!"optimizer={cfg.optim.name}", s!"lr={cfg.lr}", ModelZoo.deviceNote opts]

end AdderOptions

/-- Training/evaluation curriculum used by the adder runner. -/
inductive CurriculumMode where
  | overfitPair
  | trainSplit
  | fullTable
deriving DecidableEq, Repr

namespace CurriculumMode

/-- Decide which curriculum the current adder options request. -/
def ofOptions (cfg : AdderOptions) : CurriculumMode :=
  if cfg.overfitProbe then
    .overfitPair
  else if cfg.trainSplit then
    .trainSplit
  else
    .fullTable

/-- Startup note for the selected curriculum. -/
def intro (mode : CurriculumMode) (cfg : AdderOptions) : String :=
  match mode with
  | .overfitPair =>
      s!"  curriculum=overfit-pair pair={cfg.a}+{cfg.b}"
  | .trainSplit =>
      s!"  curriculum=train/test split ({trainCount} train / {testCount} test; train rows repeat to fill batch={batch})"
  | .fullTable =>
      "  curriculum=all 100 one-digit addition pairs"

/-- Training sample corresponding to the selected curriculum. -/
def sample (mode : CurriculumMode) (cfg : AdderOptions) : SupervisedSample Float σ τ :=
  match mode with
  | .overfitPair => mkSample cfg.a cfg.b
  | .trainSplit => mkTrainSample true
  | .fullTable => mkTrainSample false

/-- Per-step progress line for the selected curriculum. -/
def progressLine
    (mode : CurriculumMode)
    (predict : Predictor)
    (cfg : AdderOptions)
    (done : Nat)
    (lossVal : Float) : IO String := do
  match mode with
  | .overfitPair =>
      let pred ← predictSum predict cfg.a cfg.b
      pure s!"  step={done} loss={lossVal} pairPred={pred} target={cfg.a + cfg.b}"
  | .trainSplit =>
      let score ← evalBatched predict
      pure s!"  step={done} loss={lossVal} train={score.trainCorrect}/{trainCount} test={score.testCorrect}/{testCount} all={score.allCorrect}/100"
  | .fullTable =>
      let score ← evalAllBatched predict
      pure s!"  step={done} loss={lossVal} exact={score}/100"

/-- Final evaluation line for the selected curriculum, if any. -/
def finalLine?
    (mode : CurriculumMode)
    (predict : Predictor) :
    IO (Option String) := do
  match mode with
  | .overfitPair =>
      pure none
  | .trainSplit =>
      let score ← evalBatched predict
      pure <| some
        s!"  final train={score.trainCorrect}/{trainCount} test={score.testCorrect}/{testCount} all={score.allCorrect}/100"
  | .fullTable =>
      let score ← evalAllBatched predict
      pure <| some s!"  final exact={score}/100"

end CurriculumMode

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
def trainAdderFloat (opts : Options) (trainOpts : AdderOptions) :
    IO Unit := do
  let mode := CurriculumMode.ofOptions trainOpts
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.runConfig opts { optimizer := trainOpts.optim.toOptimizer trainOpts.lr })
        (.custom adderLossProgram)
  IO.println s!"  mode=adder ndigit={ndigit} vocab={vocab} seqLen={seqLen} steps={trainOpts.steps}"
  trainer.printInfo
  IO.println s!"  heads={numHeads} headDim={headDim} dModel={dModel} ffnHidden={ffnHidden} activeTargets/step={activeTargetCount}"
  IO.println s!"  optimizer={trainOpts.optim.name} lr={trainOpts.lr}"
  IO.println s!"  minGPT encoding example 8+7 -> {renderExample 8 7} (sum digits reversed)"
  IO.println <| CurriculumMode.intro mode trainOpts

  /-
  The one-digit adder has a finite training batch, so the public custom trainer sees a dataset with
  exactly one supervised sample: that sample is either the full table, the repeated train split, or
  the selected overfit pair.  The custom loss `adderLossProgram` preserves the minGPT-style
  ignore-prefix normalization while still moving the optimizer loop behind `trainer.train`.
  -/
  let trainSample : SupervisedSample Float σ τ :=
    CurriculumMode.sample mode trainOpts
  let trained ← trainer.train
    (Data.floatSamples [trainSample])
    { steps := trainOpts.steps
      log := trainOpts.toInteractiveTrainOptions.toModelTrainFlags.log
      logEvery := Nat.max 1 (trainOpts.steps / 10)
      title := "GPT adder training"
      notes := trainOpts.logNotes opts }
  trained.printSummary
  match (← CurriculumMode.finalLine? mode trained.eval) with
  | some line => IO.println line
  | none => pure ()
  printProbe trained.eval trainOpts.a trainOpts.b
  if !trainOpts.probes.isEmpty then
    IO.println "  extra checks:"
    for (a, b) in trainOpts.probes do
      printProbe trained.eval a b
  if trainOpts.interactive then
    interactiveLoop trained.eval

/-- CLI entrypoint for the CUDA GPT adder command. -/
def main (args : List String) : IO UInt32 := do
  ModelZoo.runGpuEagerFloat exeName args
    (banner := ModelZoo.bannerWithDevice exeName "minGPT-style addition training")
    (k := fun opts rest => do
      if !opts.useGpu then
        throw <| IO.userError s!"{exeName}: CUDA runtime was not selected"
      let (trainOpts, rest) ← ModelZoo.orThrow exeName <| AdderOptions.parse rest
      CLI.requireNoArgs exeName rest
      trainAdderFloat opts trainOpts)

end NN.Examples.Models.Sequence.GptAdder
