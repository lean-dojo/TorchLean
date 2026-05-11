/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA-only minGPT-style addition walkthrough:
  lake build -R -K cuda=true
  lake exe torchlean gpt_adder --steps 500 --log-every 100 --optim adam --lr 0.005 --a 7 --b 8
  lake exe torchlean gpt_adder --steps 500 --log-every 100 --optim sgd --lr 0.05 --a 7 --b 8

Interactive addition REPL:
  lake exe torchlean gpt_adder --steps 1000 --interactive
-/

module

public import NN
public import NN.API.Models.Gpt2
public import NN.Runtime.Autograd.Torch.Core
public import NN.Runtime.Autograd.TorchLean.NN

/-!
# minGPT-Style Addition Demo

This file is a TorchLean-native version of the spirit of Karpathy's `minGPT/projects/adder`
experiment.  The original minGPT adder trains a compact GPT to complete digit strings of the form

`digits(a) ++ digits(b) ++ reverseDigits(a+b)`.

For example, in the one-digit setting `8 + 7 = 15` is represented as the digit sequence
`8 7 5 1`.  At inference time the model sees `8 7` and greedily generates the two result digits
`5 1`, which we reverse back to `15`.

This is intentionally not a text chatbot.  It is a controlled "does the CUDA GPT training loop
actually learn an algorithmic task?" walkthrough:

* synthetic data is generated in Lean,
* the model is a GPT-style causal Transformer built from TorchLean layers,
* training is CUDA-only by default,
* optimizer choices are deliberately close to minGPT (`adamw`, `adam`, or `sgd`),
* evaluation greedily completes every one-digit addition problem.

Performance note: this uses the eager CUDA runtime, not a persistent CUDA graph.
The heavy tensor operations run on the GPU, including fused attention when `--fast-kernels` is on,
but each step still records a fresh autograd tape and synchronizes parameter refs through the
scalar-trainer API. This is the correctness-facing example; full PyTorch-style throughput requires
persistent device parameters plus compiled/fused graph
execution.

The GPT-shaped architecture is constructed using the shared API helper in `NN.API.Models.Gpt2`
(`nn.models.causalTransformerOneHot`), so this file can stay focused on the adder task mechanics.

Reference: <https://github.com/karpathy/minGPT/tree/master/projects/adder>.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.GptAdder

/-- Runner subcommand name. -/
def exeName : String := "torchlean gpt_adder"
def defaultLogJson : System.FilePath := "data/model_zoo/gpt_adder_trainlog.json"

/--
Number of input digits per operand.

We start with the one-digit curriculum because it is small enough to train quickly in the
eager CUDA runtime while still including carry examples such as `8 + 7 = 15`.
-/
def ndigit : Nat := 1

/-- Digit-only vocabulary, matching minGPT's adder task (`0..9`). -/
def vocab : Nat := 10

/--
Full one-digit table batch size.

This is intentionally `100`, not `1`: very small scalar workloads underutilize GPUs. In all-pairs mode one
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

def cfg : nn.models.CausalOneHotConfig :=
  { batch := batch
    seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers
    seedStride := 100 }

abbrev σ : Shape :=
  nn.models.causalOneHotShape cfg

abbrev τ : Shape :=
  σ

/-- Compact GPT-style causal Transformer for digit addition. -/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot cfg

/-- Number of trainable scalar parameters in the current compile-time adder model. -/
def modelParamCount : Nat :=
  paramCountShapes (nn.paramShapes (nn.build 0 mkModel))

/-- Number of parameter tensors in the current compile-time adder model. -/
def modelParamTensorCount : Nat :=
  (nn.paramShapes (nn.build 0 mkModel)).length

/-- Cross-entropy summed over non-ignored adder targets, normalized like minGPT `ignore_index`. -/
def adderLoss {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (logits targetOneHot :
      _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) Shape.scalar) := do
  let summed ← _root_.Runtime.Autograd.TorchLean.Loss.crossEntropyOneHot
    (m := m) (α := α) (s := τ) logits targetOneHot (reduction := .sum)
  _root_.Runtime.Autograd.Torch.Ops.scale (m := m) (α := α) (s := Shape.scalar)
    summed ((1 : α) / (activeTargetCount : α))

/--
Adder-specific scalar loss.

Ignored prefix positions are encoded by all-zero one-hot rows (`maskAdderTargets`), so they
contribute exactly zero to one-hot cross entropy.  We divide the summed loss by the number of
active target positions, matching minGPT's `ignore_index`-style normalization rather than averaging
over ignored prefix rows.
-/
def adderScalarModuleDef (model : nn.Sequential σ τ) :
    TorchLean.Module.ScalarModuleDef (nn.paramShapes model) [σ, τ] :=
  nn.scalarModuleDef model (loss := fun {α} _ _ =>
    fun {m} _ _ =>
      fun logits targetOneHot =>
        adderLoss (m := m) (α := α) logits targetOneHot)

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
    (y : Tensor α (NN.Tensor.Shape.Mat seqLen vocab)) :
    Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
  Tensor.dim (fun t =>
    if keepTargetPosition t.val then
      match y with
      | Tensor.dim rows => rows t
    else
      Tensor.fill 0 (shape![vocab]))

/-- Build one supervised next-digit sample from an addition problem. -/
def mkSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (a b : Nat) : sample.Supervised α σ τ :=
  let (x2DF, y2DF) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab)
    (renderExample a b)
  let x2D : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat x2DF
  let y2D : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    maskAdderTargets (Common.castTensor Runtime.ofFloat y2DF)
  sample.mk (Tensor.dim (fun _ => x2D)) (Tensor.dim (fun _ => y2D))

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

/-- Parse `a+b` into a probe pair; returns `none` for non-one-digit prompts. -/
def parseProbe? (s : String) : Option (Nat × Nat) :=
  let parts := s.trimAscii.toString.splitOn "+"
  match parts with
  | [aStr, bStr] =>
      match aStr.toNat?, bStr.toNat? with
      | some a, some b =>
          if a < 10 && b < 10 then some (a, b) else none
      | _, _ => none
  | _ => none

/-- Comma-separated list of one-digit `a+b` probes. -/
def parseProbeList (s : String) : Except String (List (Nat × Nat)) := do
  let raw := s.splitOn "," |>.filter (fun p => p.trimAscii.toString != "")
  let mut out : List (Nat × Nat) := []
  for p in raw do
    match parseProbe? p with
    | some pair => out := out ++ [pair]
    | none => throw s!"bad --probes entry {p}; expected comma-separated one-digit prompts like 0+0,7+8"
  pure out

/-- Build a batched supervised sample with one row per one-digit addition problem. -/
def mkTrainSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] (trainSplit : Bool) :
    sample.Supervised α σ τ :=
  let row (bi : Fin batch) : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    let (a, b) := trainPairAt trainSplit bi.val
    let (x2DF, _) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab)
      (renderExample a b)
    Common.castTensor Runtime.ofFloat x2DF
  let target (bi : Fin batch) : Tensor α (NN.Tensor.Shape.Mat seqLen vocab) :=
    let (a, b) := trainPairAt trainSplit bi.val
    let (_, y2DF) := text.causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab)
      (renderExample a b)
    maskAdderTargets (Common.castTensor Runtime.ofFloat y2DF)
  sample.mk (Tensor.dim row) (Tensor.dim target)

/-- Decode reversed generated result digits back into a natural number. -/
def decodeResult (revDigits : List Nat) : Nat :=
  revDigits.reverse.foldl (fun acc d => acc * 10 + d) 0

/-- Argmax token id at sequence position `pos`. -/
def argmaxAt (logits : Tensor Float τ) (pos : Nat) : Nat :=
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float) (batchIdx := ⟨0, by decide⟩) logits
  ids.getD pos 0

/-- Argmax token id at a sequence position for a chosen batch row. -/
def argmaxAtBatch (logits : Tensor Float τ) (bi : Fin batch) (pos : Nat) : Nat :=
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float) (batchIdx := bi) logits
  ids.getD pos 0

/-- Build a model input tensor from the current generated digit prefix. -/
def inputFromDigits (digits : List Nat) : Tensor Float σ :=
  text.causalLmXOneHotBatch (α := Float) batch seqLen vocab digits

/-- Build a batched model input from one digit prefix per row. -/
def inputFromRows (rows : Fin batch → List Nat) : Tensor Float σ :=
  let row (bi : Fin batch) : Tensor Float (NN.Tensor.Shape.Mat seqLen vocab) :=
    let x2DF : Tensor Float (NN.Tensor.Shape.Mat seqLen vocab) :=
      Tensor.dim (fun t => text.oneHotTokenFloat vocab ((rows bi).getD t.val 0))
    x2DF
  Tensor.dim row

/--
Run a model forward through the eager runtime and return logits.

We keep this helper local instead of importing the larger GPT-2 example module. That keeps the
adder executable focused on the task-specific data and evaluation path.
-/
def runtimePredictFloat {σ τ : Shape}
    (opts : Runtime.Autograd.Torch.Options) (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ])
    (x : Tensor Float σ) : IO (Tensor Float τ) := do
  nn.eval1
    (α := Float) opts model m.trainer.params x

/--
Greedily complete `ndigit + 1` result digits from the operand digits.

The key detail is that when the current prefix has length `k`, the next-token prediction lives at
position `k - 1`, not always at the final padded position.
-/
def generateResultDigits
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ])
    (a b : Nat) : IO (List Nat) := do
  let mut digits := fixedDigits ndigit a ++ fixedDigits ndigit b
  let mut out : List Nat := []
  for _ in [0:ndigit + 1] do
    let pos := if digits.length = 0 then 0 else Nat.min (digits.length - 1) (seqLen - 1)
    let logits ← runtimePredictFloat opts model m (inputFromDigits digits)
    let next := argmaxAt logits pos
    digits := digits ++ [next]
    out := out ++ [next]
  pure out

/-- Predict `a + b` by greedy decoding and reversing the minGPT result digits. -/
def predictSum
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ])
    (a b : Nat) : IO Nat := do
  let revDigits ← generateResultDigits opts model m a b
  pure (decodeResult revDigits)

/-- Evaluate all 100 one-digit additions. -/
def evalAllSlow
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ]) :
    IO Nat := do
  let mut correct := 0
  for i in [0:100] do
    let (a, b) := pairAt i
    let pred ← predictSum opts model m a b
    if pred = a + b then
      correct := correct + 1
  pure correct

structure EvalScore where
  trainCorrect : Nat
  testCorrect : Nat
  allCorrect : Nat
deriving Repr

/--
Evaluate all 100 additions with batched greedy decoding.

For one-digit operands, generation needs two result digits.  We first predict the ones digit from
rows `[a,b]`, append it, and then predict the carry/tens digit from rows `[a,b,pred₀]`.
-/
def evalBatched
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ]) :
    IO EvalScore := do
  let operandRows : Fin batch → List Nat := fun bi =>
    let (a, b) := pairAt bi.val
    fixedDigits ndigit a ++ fixedDigits ndigit b
  let logits0 ← runtimePredictFloat opts model m (inputFromRows operandRows)
  let firstDigit : Fin batch → Nat := fun bi => argmaxAtBatch logits0 bi (2 * ndigit - 1)
  let withFirst : Fin batch → List Nat := fun bi => operandRows bi ++ [firstDigit bi]
  let logits1 ← runtimePredictFloat opts model m (inputFromRows withFirst)
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

def evalAllBatched
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ]) :
    IO Nat := do
  pure (← evalBatched opts model m).allCorrect

/-- Print one addition probe in the same digit convention used for training. -/
def printProbe
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ])
    (a b : Nat) : IO Unit := do
  let revDigits ← generateResultDigits opts model m a b
  let pred := decodeResult revDigits
  IO.println s!"  probe {a}+{b}: reversed-digits={revDigits}, pred={pred}, target={a + b}"

/-- Optimizer choice for this addition run. -/
inductive OptimKind where
  | sgd
  | adam
  | adamw
deriving DecidableEq, Repr

def OptimKind.parse (s : String) : Except String OptimKind :=
  if s == "sgd" then
    pure .sgd
  else if s == "adam" then
    pure .adam
  else if s == "adamw" then
    pure .adamw
  else
    throw s!"bad --optim {s}; expected sgd, adam, or adamw"

def OptimKind.name : OptimKind → String
  | .sgd => "SGD"
  | .adam => "Adam"
  | .adamw => "AdamW"

/-- Local options for the adder runner. -/
structure TrainOptions where
  /-- Number of optimizer steps. -/
  steps : Nat
  /-- Print loss every `logEvery` steps. -/
  logEvery : Nat
  /-- JSON training log artifact path. -/
  logPath : System.FilePath
  /--
  Optimizer.

  `adamw` is closest to minGPT's adder recipe.  `adam` and `sgd` are kept for debugging and
  comparisons.
  -/
  optim : OptimKind
  /-- Learning rate. -/
  lr : Float
  /-- Probe operand `a`. -/
  a : Nat
  /-- Probe operand `b`. -/
  b : Nat
  /-- Extra comma-separated prompt probes, e.g. `0+0,4+5,9+9`. -/
  probes : List (Nat × Nat)
  /-- Train on an 80/20 train/test split instead of all 100 one-digit additions. -/
  trainSplit : Bool
  /-- Train only the probe pair, useful for verifying the CUDA GPT can overfit one addition. -/
  overfitProbe : Bool
  /-- Keep the trained CUDA model alive and read `a+b` prompts from stdin. -/
  interactive : Bool
deriving Repr

/-- Parse adder-specific CLI options. -/
def parseTrainOptions (args : List String) : Except String (TrainOptions × List String) := do
  let (steps, args) ← CLI.takeStepsOrEpochs args 1000
  let (logEvery?, args) ← CLI.takeNatFlagOnce args "log-every"
  let (logPath?, args) ← CLI.takePathFlagOnce args "log"
  let (optim?, args) ← CLI.takeFlagValueOnce args "optim"
  let (lr?, args) ← CLI.takeFloatFlagOnce args "lr"
  let (a?, args) ← CLI.takeNatFlagOnce args "a"
  let (b?, args) ← CLI.takeNatFlagOnce args "b"
  let (probes?, args) ← CLI.takeFlagValueOnce args "probes"
  let (trainSplit, args) ← CLI.takeBoolFlagOnce args "train-split"
  let (overfitProbe, args) ← CLI.takeBoolFlagOnce args "overfit-probe"
  let (interactive, args) ← CLI.takeBoolFlagOnce args "interactive"
  let lr ←
    match lr? with
    | some v => pure v
    | none => pure 5e-4
  let optim ←
    match optim? with
    | some s => OptimKind.parse s
    | none => pure .adamw
  let a := a?.getD 7
  let b := b?.getD 8
  if a ≥ 10 || b ≥ 10 then
    throw "--a and --b must be one-digit numbers in 0..9"
  let probes ←
    match probes? with
    | some s => parseProbeList s
    | none => pure [(0, 0), (1, 2), (4, 5), (7, 8), (9, 9)]
  pure ({ steps := steps
          logEvery := logEvery?.getD 100
          logPath := logPath?.getD defaultLogJson
          optim := optim
          lr := lr
          a := a
          b := b
          probes := probes
          trainSplit := trainSplit
          overfitProbe := overfitProbe
          interactive := interactive }, args)

/-- Force CUDA and fused kernels, because this example is meant to exercise the GPU path. -/
def forceCudaArgs (args : List String) : Except String (List String) := do
  if args.contains "--cpu" then
    throw "gpt_adder is CUDA-only; remove --cpu"
  if args.contains "--backend=compiled" then
    throw "gpt_adder requires --backend eager; compiled is proof-compiled host execution, not CUDA graph execution"
  let rec hasCompiledBackend : List String → Bool
    | "--backend" :: "compiled" :: _ => true
    | _ :: rest => hasCompiledBackend rest
    | [] => false
  if hasCompiledBackend args then
    throw "gpt_adder requires --backend eager; compiled is proof-compiled host execution, not CUDA graph execution"
  let args := if args.contains "--cuda" then args else "--cuda" :: args
  let args := if args.contains "--fast-kernels" then args else "--fast-kernels" :: args
  pure args

/-- Simple terminal REPL for the trained CUDA model. -/
partial def interactiveLoop
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential σ τ)
    (m : TorchLean.Module.ScalarModule Float (TorchLean.NN.Seq.paramShapes model) [σ, τ]) :
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
          printProbe opts model m a b
          loop
  loop

/-- Train the minGPT-style adder from scratch and report exact addition accuracy. -/
def trainAdderFloat (opts : Runtime.Autograd.Torch.Options) (trainOpts : TrainOptions) :
    IO Unit := do
  nn.withModel mkModel fun model => do
    let modDef := adderScalarModuleDef model
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let sample0 := mkSample (α := Float) 0 0
    let loss0 ← TorchLean.Module.forward (α := Float) m sample0
    IO.println s!"  mode=adder ndigit={ndigit} vocab={vocab} seqLen={seqLen} steps={trainOpts.steps}"
    IO.println s!"  model layers={layers} heads={numHeads} headDim={headDim} dModel={dModel} ffnHidden={ffnHidden}"
    IO.println s!"  parameters={modelParamCount} tensors={modelParamTensorCount} activeTargets/step={activeTargetCount}"
    IO.println s!"  optimizer={trainOpts.optim.name} lr={trainOpts.lr}"
    IO.println s!"  initial loss={Tensor.toScalar loss0}"
    IO.println s!"  minGPT encoding example 8+7 -> {renderExample 8 7} (sum digits reversed)"
    if trainOpts.overfitProbe then
      IO.println s!"  curriculum=overfit-probe pair={trainOpts.a}+{trainOpts.b}"
    else if trainOpts.trainSplit then
      IO.println s!"  curriculum=train/test split ({trainCount} train / {testCount} test; train rows repeat to fill batch={batch})"
    else
      IO.println "  curriculum=all 100 one-digit addition pairs"

    /-
    The one-digit adder dataset is static, so build the supervised batch once.

    This matters for runtime measurements: constructing a `Tensor.dim` tree in Lean every optimizer
    step can dominate very small GPU experiments and obscure the cost of the CUDA kernels. Real data
    loaders should still stream/minibatch, but this finite full-table task intentionally trains
    against the same batch each step.
    -/
    let trainSample : sample.Supervised Float σ τ :=
      if trainOpts.overfitProbe then
        mkSample (α := Float) trainOpts.a trainOpts.b
      else
        mkTrainSample (α := Float) trainOpts.trainSplit
    let trainLoss0 ← TorchLean.Module.forward (α := Float) m trainSample
    let trainLoss0Val := Tensor.toScalar trainLoss0

    let stepSample : sample.Supervised Float σ τ → IO Unit ←
      match trainOpts.optim with
      | .sgd =>
          let opt := TorchLean.Optim.sgd (α := Float)
            (paramShapes := nn.paramShapes model) trainOpts.lr
          let optH ← TorchLean.Optim.handle (α := Float) m opt
          pure optH.step
      | .adam =>
          let opt := TorchLean.Optim.adam (α := Float)
            (paramShapes := nn.paramShapes model)
            (lr := trainOpts.lr)
            (beta1 := 0.9)
            (beta2 := 0.95)
            (epsilon := 1e-8)
          let optH ← TorchLean.Optim.handle (α := Float) m opt
          pure optH.step
      | .adamw =>
          let opt := TorchLean.Optim.adamw (α := Float)
            (paramShapes := nn.paramShapes model)
            (lr := trainOpts.lr)
            (weightDecay := 0.1)
            (beta1 := 0.9)
            (beta2 := 0.95)
            (epsilon := 1e-8)
          let optH ← TorchLean.Optim.handle (α := Float) m opt
          pure optH.step

    for step in [0:trainOpts.steps] do
      stepSample trainSample
      let done := step + 1
      if trainOpts.logEvery != 0 && done % trainOpts.logEvery == 0 then
        let loss ← TorchLean.Module.forward (α := Float) m trainSample
        let lossVal := Tensor.toScalar loss
        Common.check exeName s!"non-finite training loss at step {done}" (lossVal == lossVal)
        if trainOpts.overfitProbe then
          let pred ← predictSum opts model m trainOpts.a trainOpts.b
          IO.println s!"  step={done} loss={lossVal} probePred={pred} target={trainOpts.a + trainOpts.b}"
        else if trainOpts.trainSplit then
          let score ← evalBatched opts model m
          IO.println s!"  step={done} loss={lossVal} train={score.trainCorrect}/{trainCount} test={score.testCorrect}/{testCount} all={score.allCorrect}/100"
        else
          let score ← evalAllBatched opts model m
          IO.println s!"  step={done} loss={lossVal} exact={score}/100"

    if trainOpts.overfitProbe then
      pure ()
    else
      if trainOpts.trainSplit then
        let score ← evalBatched opts model m
        IO.println s!"  final train={score.trainCorrect}/{trainCount} test={score.testCorrect}/{testCount} all={score.allCorrect}/100"
      else
        let score ← evalAllBatched opts model m
        IO.println s!"  final exact={score}/100"
    let trainLoss1 ← TorchLean.Module.forward (α := Float) m trainSample
    let trainLoss1Val := Tensor.toScalar trainLoss1
    Common.writeBeforeAfterLossLog trainOpts.logPath "GPT adder training" trainOpts.steps
      trainLoss0Val trainLoss1Val
      #[s!"optimizer={trainOpts.optim.name}", s!"lr={trainOpts.lr}",
        s!"device={if opts.useGpu then "cuda" else "cpu"}"]
    printProbe opts model m trainOpts.a trainOpts.b
    if !trainOpts.probes.isEmpty then
      IO.println "  extra probes:"
      for (a, b) in trainOpts.probes do
        printProbe opts model m a b
    if trainOpts.interactive then
      interactiveLoop opts model m

def main (args : List String) : IO UInt32 := do
  match forceCudaArgs args with
  | .error e =>
      IO.eprintln s!"{exeName}: {e}"
      pure 1
  | .ok args =>
      TorchLean.Module.run exeName args
        (.float (fun opts rest => do
          if !opts.useGpu then
            throw <| IO.userError s!"{exeName}: CUDA runtime was not selected"
          let (trainOpts, rest) ← Common.orThrow exeName <| parseTrainOptions rest
          Common.orThrow exeName <| CLI.requireNoArgs rest
          trainAdderFloat opts trainOpts))
        { banner? := some (fun opts =>
            s!"{exeName}: minGPT-style addition training (device={if opts.useGpu then "cuda" else "cpu"})")
          printOk := true }

end NN.Examples.Models.Sequence.GptAdder
