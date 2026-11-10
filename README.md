<h1 align="center">
  <img src="home_page/assets/media/brand/torchlean-logo.png" alt="TorchLean logo" width="88" align="center">
  Formalizing Neural Networks in Lean
</h1>

TorchLean brings general tensor computations, machine learning, and mathematical proofs together
in Lean. We can use its shape-checked tensors for linear algebra and numerical algorithms, write
our own tensor functions, or build and train neural networks, from small examples to larger
GPU-backed training runs. Lean is both a functional programming language and a theorem prover,
so we can run a calculation and reason about it in the same language.

The same model definition serves training, graph transformations, certificate checks and proofs.
The verification tools answer questions such as whether a classifier keeps its prediction across
an input region. Through FloatLib, we can choose the numerical precision and prove bounds on
rounding and numerical error under stated assumptions. The CPU tape, the GPU tape and the
verifier's graph lowering all record operators from one table, `TorchLean.Op`. Each operator has
one rule for its value and gradient. The CPU tape runs that rule, and the GPU tape runs the
operator's LibTorch implementation. Automatic differentiation is TorchLean's own: the tape
and the backward pass are Lean code, and on the GPU, LibTorch's ATen operations compute the tensor
values and local gradients. That external GPU code is a trust boundary, outside the Lean proofs.

## Installation

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
scripts/lake.sh exe cache get
scripts/lake.sh build
```

`scripts/lake.sh build` builds the four libraries `TorchLean.Tensor`, `TorchLean.CUDA`,
`TorchLean.NN`, and `TorchLean.Verify`. To build one layer, name it:
`scripts/lake.sh build TorchLean.Tensor`.

The [installation guide](https://lean-dojo.github.io/TorchLean/installation/) covers Linux, macOS,
Windows/WSL and CUDA with LibTorch, and explains TorchLean's backend architecture.

## Quickstart

```bash
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 10 --arithmetic ieee
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 10

# Optional GPU run with a CUDA-enabled LibTorch SDK, matching toolkit, and NVIDIA GPU:
export TORCHLEAN_LIBTORCH_HOME=/absolute/path/to/libtorch
scripts/lake.sh -Kcuda=true build
scripts/lake.sh -Kcuda=true exe torchlean quickstart_mlp --device cuda --steps 10
```

The first quickstart command uses [FloatLib](https://github.com/lean-dojo/FloatLib)'s binary32
arithmetic and the second uses Lean's native `Float32`. Typed tensors and models accept other
FloatLib binary formats too (see [Precision](#precision)). The CUDA command runs on LibTorch's GPU
runtime while TorchLean keeps its own differentiation tape, and it reports an error when CUDA is
unavailable.

Application code writes concrete tensor types as `Tensor α [dims...]`, with the element type first.
For example, `Tensor Float [4, 2]` is a four-by-two tensor of `Float` values:

```lean
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
open TorchLean
open Trainer.Objective (mse)

/-- A two-layer regression model. The dimensions are checked when the layers are composed. -/
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
    nn.linear 8 1
  ]

-- Four input rows, each containing two features.
def xs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

-- One regression target for each input row.
def ys : Tensor Float [4, 1] :=
  [[0.2], [1.0], [1.0], [1.8]]

-- The leading `4` counts samples; each sample has shapes `[2]` and `[1]`.
def data : Trainer.Dataset [2] [1] := Data.fromTensors xs ys

def trainOnce : IO Unit := do
  -- Select the loss and train on the CPU with FloatLib binary32 arithmetic.
  let trainer :=
    Trainer.create model
      { objective := mse
        optimizer := optim.sgd { lr := 0.05 }
        device := .cpu
        arithmetic := .ieee }
  -- Inspect the initialized model before any parameter updates.
  let initialPrediction ← trainer.predict ([0.5, -0.25] : Tensor Float [2])
  IO.println s!"initial={reprStr initialPrediction}"
  -- Each step averages 16 sample gradients at one parameter point, then updates once.
  -- Training returns a result that retains the updated parameters and run report.
  let trained ← trainer.train data { steps := 200, samplesPerStep := 16, logEvery := 25 }
  trained.printSummary
```

### Compile a Python model

From the repository root, install the Python package with `python -m pip install -e python`,
build the checker with `scripts/lake.sh build verify`, and put `@torchlean.compile` on a model
class:

```python
import torch
import torchlean

@torchlean.compile
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
        )

    def forward(self, inputs):
        return self.layers(inputs)

model = MLP()
batch = torch.ones(32, 2)
prediction = model(batch)
prediction.sum().backward()
print(model.torchlean.report["ok"])
```

TorchLean detects PyTorch or JAX automatically. The first call captures the computation for the
whole batch and checks its IR structure and types. Later compatible calls reuse that graph. There is
no extra loop over samples. PyTorch or JAX runs the captured computation itself, on the same device
and with its own automatic differentiation. Use the same decorator on an ordinary PyTorch or JAX
function, or use `model = torchlean.compile(model)` for a constructed PyTorch module, whose
parameters and training methods remain available. For a class that overrides `__call__`, put the
decorator on the class. Unsupported captures and failed checks raise an error.

These checks are about the imported graph. They don't prove that it matches the Python program, or
that it is correct on every input. The
[captured-computation guide](https://lean-dojo.github.io/TorchLean/examples/verify/) covers graph
inspection, sample comparisons and Lean proof exports.

An [optional Python binding](bindings/python/README.md) runs the same checker inside the Python
process. `torchlean.compile` and `torchlean.verify` have the same interface with or without it.

## Commands

```bash
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe verify --help
scripts/lake.sh exe verify -- torchlean-ibp
```

For the maintained examples:

```bash
scripts/lake.sh build TorchLeanExamples
```

## Use TorchLean From Another Lean Project

TorchLean is a normal Lake package. You can depend on the Git repository directly:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then run from the downstream project's root, using its own Lake configuration:

```bash
lake update
lake exe cache get
lake build
```

Model, data and training code imports the modules of the application API that define what it uses,
as the quickstart above does. [`TorchLean/NN`](TorchLean/NN/README.md) holds `TorchLean.nn`,
`TorchLean.Data`, `TorchLean.Trainer` and `TorchLean.optim`. Import `TorchLean.Verify.Trainer`
to call `trained.verify center (radius := r)` after ordinary training, and
`TorchLean.Verify.Lowering` to lower a model to a verifier graph yourself.

### Choosing layers

A project that requires TorchLean and imports only `TorchLean.Tensor.*` modules compiles against
FloatLib and Mathlib and links only the CPU tensor core, `csrc/cpu/torchlean_tensor.c`, with the
pinned allocator on Linux. Once its imports reach `TorchLean.CUDA`, as the quickstart's imports
do, an executable also links the stub compiled from `csrc/libtorch/unavailable.c`, which reports
the GPU backend as not linked. Setting the Lake option `cuda` to `true` links the LibTorch backend
in its place: `-Kcuda=true` when building TorchLean itself, or the `with` clause of `require` in
a project that depends on it.

### Custom tensor computations

We can write a scalar calculation as an ordinary Lean function, apply it to a tensor, and choose
the execution device at the call site:

```lean
import TorchLean.NN.Kernel.Function
open TorchLean

def square := fun (x : Float32) => x * x

def input : Tensor Float32 [3] := Tensor.ofFn fun i => Float32.ofNat (i.val + 1)
def onCpu : IO (Tensor Float32 [3]) := square.run input (device := .cpu)
def onGpu : IO (Tensor Float32 [3]) := square.run input (device := .cuda)

#eval square.run input (device := .cpu)
-- [1.000000, 4.000000, 9.000000]
```

CPU is the default and runs the Lean function itself. On GPU, supported calculations compile
through NVRTC, with native FP32/FP64 or configured binary arithmetic. A scalar type or format that
only the CPU supports runs on CPU at its own precision, with a diagnostic. Unsupported code for a
GPU-capable type, or an unavailable GPU, returns an error. Functions can take runtime scalar
parameters, and `f.zip left right` combines two tensors of the same shape. Indexed programs
support bounded loops and fixed-parameter numerical recurrences, with a Lean-checked
correspondence to the recursive equations. Whole-tensor functions use the same call,
`f.run input`. Add `(grad := true)` to record supported operations on the autograd tape. Such a
result has `.value` and `.backward`. Shapes and storage are inferred from the input.
The tape does not record arbitrary indexed bodies, and `Program.gradient` compiles their
gradients into kernels, proved over the reals. The
[custom-operation guide](TorchLean/NN/Kernel/README.md) and the
[CUDA compiler guide](TorchLean/CUDA/Kernel/README.md) cover the supported operations, indexed
programs, lowering proofs and the native execution boundary.

### Proved gradients

Over the reals, the stored gradients of the operator table are proved to be adjoints of Fréchet
derivatives. The operators with a fixed number of inputs (all but `cat` and `custom`) have 166 value
inputs, and the gradient of each has a `derivative` certificate in all the value inputs together.
The certificate holds at the points its hypotheses name, such as nonzero divisors for division.
`scripts/checks/Library.lean` fails when one of these gradients has no such certificate. `cat` has
`cat_backward_eq_adjoint`. At kinks, such as `relu` at zero, `convention` certificates state the
stored backward. `#op_certs Op.relu` lists the certificates of `relu`.

`op_deriv` proves the same statement for a backward pass written by hand, here for softplus
applied entrywise:

```lean
import TorchLean.NN.Tactic.OpDeriv
open TorchLean

example {s : Shape} (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => Real.log (1 + Real.exp a)) y)
        x ∧
      δ * Tensor.map (fun a => Real.exp a / (1 + Real.exp a)) x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.map (fun a => Real.log (1 + Real.exp a)) y)
          x).adjoint δ := by
  op_deriv
```

A wrong backward pass fails with the equations the tactic could not close. The certificates and
the tactic are about real arithmetic and bound no rounding error. The
[tactic guide](TorchLean/NN/Tactic/README.md#backward-passes) lists the rules and side conditions.

### Recursion over trees

`TorchLean.CUDA.Tree` runs recursive functions over inductive types on the GPU. A type that derives
`Sig.Inductive` is stored by levels, the children of level `r` on level `r + 1`. `Fold.of` turns a
structurally recursive function into a fold, one kernel launch per level, with a proof that the fold
computes the function:

```lean
import TorchLean.CUDA.Tree.Elab
import TorchLean.CUDA.Tree.Run
open TorchLean.Kernel

inductive BTree where
  | leaf
  | node (l : BTree) (x : Float) (r : BTree)
  deriving Sig.Inductive

def BTree.sum : BTree → Float
  | .leaf => 0
  | .node l x r => l.sum + x + r.sum

def sumFold := Fold.of BTree.sum

def small : BTree := .node (.node .leaf 1 .leaf) 2 (.node .leaf 3 .leaf)

#eval sumFold.eval (Reader.ofArrays #[])
  (Arena.encode (σ := BTree.sig) (T := fun _ => BTree) (s := 0) small)
-- Except.ok 6.000000
```

`Fold.run` runs the fold over an arena that `Arena.upload` puts in device memory. Hylomorphisms
cover recursion whose calls do not follow a value, and unfolds build trees on the GPU. The
[tree guide](TorchLean/CUDA/Tree/README.md) states what is proved and what is trusted.

### Precision

[FloatLib](https://github.com/lean-dojo/FloatLib) supplies configurable arithmetic and numerical
proofs. TorchLean's CPU tensors and typed models can use binary32, binary128 or a custom binary
format without converting through native floats. Custom GPU computations accept configured binary
types too. They pick compatible hardware operations automatically and use integer-limb software
arithmetic for custom formats and for precisions beyond native floats. The LibTorch tape keeps
`Float32` in binary32 and `Float` in binary64, gradients and optimizer state included. Configured
binary arithmetic also records on the GPU tape, which saves values and gradients as complete words.
The arithmetic operations in the
[custom-operation guide](TorchLean/NN/Kernel/README.md#recording-a-calculation-for-differentiation)
record this way, while native model operators and optimizer checkpoints require native dtypes.
Decimal and posit formats run on CPU.

FloatLib's `roundoff` tactic reads the Lean code of an operator and proves a bound on its rounding
error, absolute or mixed relative and absolute, for every IEEE format. TorchLean registers its
tensors with the tactic (`TorchLean.Verify.RuntimeApprox.Roundoff`), so it also reads tensor
entries, folds and the general operators, at any axes and shape. The
[numerical runtime example](https://lean-dojo.github.io/TorchLean/examples/numerical-runtime/)
explains its theorems. `#roundoff_coverage` runs it on every registered operator and tape
constructor, and `scripts/lake.sh lint` fails when a public one isn't registered. The guide's
[operator chapter](https://lean-dojo.github.io/TorchLean/blueprint/Semantics-and-Graphs/The-Operator-Table/)
lists the operators of the table by family, with their CUDA entries, certificates and bound
propagations.

`TorchLean.Verify.RuntimeApprox.OnlineSoftmax` bounds the rounding error of FlashAttention's online
softmax in every IEEE format, with the scores and values as inputs.
`TorchLean.Verify.RuntimeApprox.TransformerBlock` bounds the rounding error of a pre-normalization
transformer block in any arithmetic that meets its error contract, given bounds on the exact stages
and the weights.

The [typed-training example](TorchLeanExamples/Quickstart/TypedTraining.lean) trains in binary128,
and the [tensor guide](https://lean-dojo.github.io/TorchLean/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/)
covers precision, shapes and derivatives. Import `TorchLean.NN.Sequential.Training` for typed graphs
and SGD on typed model state, `TorchLean.NN.Trainer.Session` for typed trainer sessions, or
`FloatLib` itself for standalone numerical work.

## Repository Map

- [`TorchLean/Tensor`](TorchLean/Tensor/README.md): shape-indexed tensors and their operations,
  packed CPU storage, scalar formats, mathematical tensor and layer definitions with the operator
  table, proof automation, and tensor proofs.
- [`TorchLean/CUDA`](TorchLean/CUDA/README.md): the LibTorch boundary, the CUDA compiler for
  custom computations, and folds over trees on the GPU.
- [`TorchLean/NN`](TorchLean/NN/README.md): the application API, model and dynamical-system
  specifications, executable autograd, optimizers, training loops, PyTorch import/export, Jaxpr
  graph import, RL runtime support, backend plans, the graph IR, learning theory, and proofs.
- [`TorchLean/Verify`](TorchLean/Verify/README.md): CROWN and IBP, certificate checkers and CLI
  workflows, roundoff proofs, and the floating-point bridges and contracts.
- [`TorchLeanExamples`](TorchLeanExamples/README.md): quickstarts, runnable model examples,
  widgets, bundled verification assets, and interoperability workflows.
- [`TorchLeanTest`](TorchLeanTest/README.md): the tests behind `scripts/lake.sh test`, which check
  TorchLean against PyTorch, LibTorch and CUDA, ONNX, native floating point and outside file
  formats.
- `csrc`: native sources, with the CPU tensor core in `csrc/cpu`, the LibTorch backend and its
  stub in [`csrc/libtorch`](csrc/libtorch/README.md), and the pinned Linux allocator in
  `csrc/runtime`.
- `python/torchlean`: the Python package that compiles PyTorch and JAX graphs to TorchLean IR and
  checks them (`@torchlean.compile`), installed with `python -m pip install -e python`.
- FloatLib dependency: configurable executable formats, reference semantics, rounding proofs,
  and scalar intervals.
- `home_page/blueprint/TorchLeanBlueprint/Guide`: source for the guide.
- `home_page`: project website sources.

## Proofs And Runtime Boundaries

TorchLean proves properties of definitions written in Lean. It also checks certificates produced
by external tools, including bound-propagation and scientific-computing workflows. An executable
certificate check reports that the checker accepted the certificate. A semantic guarantee also
needs the checker's soundness theorem and its hypotheses, and a Lean proof needs kernel-checked
evidence of acceptance. None of these checks certifies the program that produced the certificate.

CPU instructions, CUDA kernels, cuBLAS, LibTorch, PyTorch, Julia and other external systems are
runtime providers. [`docs/TRUST_BOUNDARIES.md`](docs/TRUST_BOUNDARIES.md) lists their interfaces,
their assumptions and the checks available for each.
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) lists third-party sources and
licenses, and [the AI usage disclosure](docs/CONTRIBUTING.md#ai-usage-disclosure) describes how we
used coding assistants.

The papers, books and standards that the specifications follow are listed in
[docs/REFERENCES.md](docs/REFERENCES.md).

Contribution guidelines are in [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Citation

If TorchLean is useful in your work, please cite
[*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631):

```bibtex
@misc{george2026torchlean,
  title         = {TorchLean: Formalizing Neural Networks in Lean},
  author        = {George, Robert Joseph and Cruden, Jennifer and Adkisson, Will and
                   Zhong, Xiangru and Zhang, Huan and Anandkumar, Anima},
  year          = {2026},
  eprint        = {2602.22631},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MS},
  url           = {https://arxiv.org/abs/2602.22631}
}
```

## License

TorchLean is released under the MIT License. See `LICENSE`.
