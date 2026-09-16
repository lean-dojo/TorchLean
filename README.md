<h1 align="center">
  <img src="home_page/assets/media/brand/torchlean-logo.png" alt="TorchLean logo" width="88" align="center">
  Formalizing Neural Networks in Lean
</h1>

TorchLean brings neural-network programming and formal reasoning into one Lean project. Tensor
shapes are part of the types, models are executable Lean programs, and the same definitions can be
used by training code, graph transformations, certificate checkers, and proofs. CPU and CUDA
backends handle numerical work; the Lean library records the mathematical meaning and assumptions
attached to each path.

## Installation

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
```

For Linux, macOS, Windows/WSL, CUDA, optional LibTorch support, and an explanation of
TorchLean's backend architecture, see the [Installation guide](https://lean-dojo.github.io/TorchLean/installation/).

## Quickstart

```bash
lake exe torchlean quickstart_mlp --device cpu --steps 10 --arithmetic ieee --execution eager
lake exe torchlean quickstart_mlp --device cpu --steps 10 --execution eager

# Optional CUDA run, if the CUDA toolkit and an NVIDIA GPU are available:
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 1000
```

The first quickstart uses [FloatLib](https://github.com/lean-dojo/FloatLib)'s binary32 arithmetic.
The second uses Lean's native `Float32`.
For more precision, choose a FloatLib binary format directly in typed tensors and models, as shown
below.
The CUDA command selects the native GPU runtime and reports an error when CUDA is unavailable.

Application code writes concrete tensor types as `Tensor α [dims...]`, with the element type first.
For example, `Tensor Float [4, 2]` is a four-by-two tensor of `Float` values:

```lean
import NN.API
open TorchLean

/-- A two-layer regression model. The dimensions are checked when the layers are composed. -/
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
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
  -- Select the loss and train through a typed graph with FloatLib binary32 arithmetic.
  let trainer :=
    Trainer.new model
      { objective := .meanSquaredError
        optimizer := optim.sgd { learningRate := 0.05 }
        execution := .typedGraph
        device := .cpu
        arithmetic := .ieee }
  -- Inspect the initialized model before any parameter updates.
  let initialPrediction ← trainer.predict ([0.5, -0.25])
  IO.println s!"initial={reprStr initialPrediction}"
  -- Each step averages 16 sample gradients at one parameter point, then updates once.
  -- Training returns a result that retains the updated parameters and run report.
  let trained ← trainer.train data { steps := 200, samplesPerStep := 16, logEvery := 25 }
  trained.printSummary
```

## Commands

```bash
lake exe torchlean --help
lake exe verify --help
lake exe verify -- torchlean-ibp
```

For the maintained examples:

```bash
lake build NNExamples
```

## Use TorchLean From Another Lean Project

TorchLean is a normal Lake package. You can depend on the Git repository directly:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then run:

```bash
lake update
lake exe cache get
lake build
```

Use `import NN.API` for model, data, and training code. It provides `TorchLean.nn`,
`TorchLean.Data`, `TorchLean.Trainer`, and `TorchLean.optim` without exposing the full proof and
backend trees. Use `NN.API.Verification` to call
`trained.verify center (radius := r) (norm := .inf)` after ordinary training. Explicit verifier
graph lowering has the focused import `NN.API.Verification.Lowering`. Use `import NN` when the
same file also needs proofs or backend infrastructure; focused imports such as `NN.GraphSpec`,
`NN.Runtime`, or `NN.Proofs` are available for subsystem work.

Downstream model and training files should start from:

```lean
import NN.API
open TorchLean
```

For scalar arithmetic and numerical proofs, import FloatLib together with `NN.API.Precision`.
The precision API supplies TorchLean's `Context` instances and rational casts. Use FloatLib's
scalar types directly: choose binary32, binary128, or your own exponent and fraction widths:

```lean
import FloatLib
import NN.API.Precision
open FloatLib.Floats

-- Binary32 stores 23 fraction bits: 24 significant bits for a normal value.
abbrev Binary32 :=
  ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)

-- Binary128 has 113 significant bits, including the implicit leading bit.
abbrev Binary128 :=
  ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)

-- A custom format with binary64's exponent range and 81 significant bits.
abbrev Custom :=
  ExecFloat.Binary (exponentBits := 11) (fractionBits := 80)

def small : Binary32 := 1.5
def wide : Binary128 := Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
def custom : Custom := Rat.cast (1 + 1 / (2 ^ 70 : Nat) : Rat)

-- Ordinary addition uses the implementation covered by this theorem.
example (a b : Binary128) :
    a + b = ExecFloat.Spec.add a b :=
  ExecFloat.Proof.add_eq_spec a b
```

The type selects the format; there is no fixed upper precision chosen by TorchLean. FloatLib
requires at least two exponent bits, a positive fraction width, and a valid exponent bias.
Storage and execution cost grow with the selected widths. Numerical literals and rational casts
round directly into the destination format, without a native `Float` intermediate.
`ExecFloat.Binary.toRat?` recovers the exact rational value of a finite result; it returns `none`
for NaN or infinity. The theorem above relates executable addition to its reference semantics,
including exceptional values. Real error bounds add the relevant finiteness and range hypotheses.

FloatLib also supplies other binary widths, decimal formats, posits, fixed-point arithmetic, and
intervals. Its scalar import does not depend on TorchLean's tensors, models, or CUDA runtime.
Configured binary formats support CPU tensor and typed-model execution at the selected precision.
The native CUDA providers support binary32 and binary64; selecting another FloatLib binary format
does not create a GPU provider for it. TorchLean's tensor quantization and runtime-approximation
connections remain separate from the scalar library.

For tensors and models at a chosen precision, `NN.API` includes `NN.API.Precision`. A
`Tensor Binary128 shape` holds configured scalars, and a typed graph with
`nn.State Binary128 (nn.stateShapes model)` uses that scalar for parameters, activations, and
derivatives. Initialize the state directly from exact literals or rationals when those digits
matter. `nn.sgdStep model learningRate state gradient` applies an SGD update with the learning
rate and gradients in the same type. It preserves frozen state and rejects models with buffer-update
hooks, including BatchNorm. The [typed training example](NN/Examples/Quickstart/TypedTraining.lean)
combines a typed VJP with this update to fit an affine model in binary128.

Take $a = 1 + 2^{-100}$ and start with the affine model $x \mapsto ax + a$.
For input $2$ and target $0$, the gradients of half squared error are $6a$ for the weight
and $3a$ for the bias. One SGD step with learning rate $1/8$ gives weight $a/4$ and bias $5a/8$:

```lean
import NN.API
open TorchLean
open FloatLib.Floats

abbrev Scalar := ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)

def affine : nn.Sequential [1] [1] := nn.build 0 (nn.linear 1 1)

def trainOnce : IO (Option Rat) := do
  let a : Scalar := Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
  let state : nn.State Scalar (nn.stateShapes affine) := nn.State.full a
  let input : Tensor Scalar [1] := Tensor.full [1] 2
  let graph ← nn.lowerToTypedGraph affine (α := Scalar) (mode := .train)
  let prediction := nn.TypedGraphModel.forward graph state input
  -- For target zero, the prediction itself is the loss derivative.
  let (gradient, _) := nn.TypedGraphModel.vjp graph state input prediction
  let next ← match nn.sgdStep affine (Rat.cast (1 / 8 : Rat)) state gradient with
    | .ok next => pure next
    | .error message => throw <| IO.userError message
  let output := nn.TypedGraphModel.forward graph next input
  return ExecFloat.Binary.toRat? (output.getScalar ⟨0, by decide⟩)

#eval trainOnce
```

The final prediction is $9a/8$. Returning an exact rational observation preserves the digits
that would disappear in a conversion to binary64. The same typed interfaces accept other valid
configured binary formats; their rounding can change the arithmetic result.

The supervised trainer's dataset, reporting, and checkpoint boundaries use `Float`; changing its
arithmetic option does not turn that interface into a general-precision data path. The
[tensor guide](https://lean-dojo.github.io/TorchLean/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/)
works through a typed binary128 model and checks its output and derivatives against exact rationals.

For local development against a checkout, use a path dependency instead:

```lean
require TorchLean from "../TorchLean"
```

### Finding The Numerical API

The scalar implementation and its generic proofs live in FloatLib. Choose the import for the
numerical work:

| Numerical work | Import |
| --- | --- |
| Standalone scalar arithmetic | `FloatLib` |
| Formats, rounding, and real error bounds | `FloatLib.Floats.Formats.Flocq` |
| Mantissa/exponent calculations | `FloatLib.Floats.Formats.Flocq.Calculation.Round` or `FloatLib.Floats.Formats.Flocq.Calculation.Operations` |
| Configurable binary arithmetic | `FloatLib.Floats.Formats.BinaryInterchange.Configured` |
| Native logical-model proofs | `FloatLib.Floats.Formats.IEEE754` |
| Generic intervals | `FloatLib.Floats.Interval` |

Generic rounded-real definitions live in `FloatLib.Floats.Formats.Flocq`; executable values use
`FloatLib.Floats.ExecFloat`. The binary elementary functions need the explicit import
`FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals`.

Native conversions now use `ExecFloat.Binary.ofFloat32` and `toFloat32` directly. FloatLib proves
their exact native round trip, addition/subtraction agreement for finite operands, and square-root
agreement through native export for every configured input. The finite-input addition theorem
allows overflow in the result. These are logical-model theorems; they do not certify compiled
CPU or CUDA instructions.

TorchLean retains `NN.Floats.FP32` and its tensor connections. Rounded binary32 reduction-tree
theorems are imported through
`NN.Proofs.RuntimeApprox.Reductions.IEEE32`. FloatLib's
`FloatLib.Floats.Formats.BinaryInterchange.Configured.Reduction` instead describes exact
accumulation followed by one rounding; it does not replace theorems about rounding at every node.

For quantization, `FloatLib.Numerics.Quantization.Affine` supplies executable rational
nearest-even quantization. The real-scale quantizer remains in `NN.Floats.Quantization`, with
tensor lifts in `NN.Spec.Quantization` and `NN.Spec.Quantization.Rational`.

## Repository Map

- `NN.lean`: complete import for model, tensor, data, training, verification, and proof workflows.
- `NN/API`: the application API exported by `import NN.API` and included by `import NN`.
- `NN/Tensor`: the shared shape-indexed tensor type, packed CPU storage, conversions, and operations.
- `NN/Spec`: mathematical tensor, layer, model, and dynamical-system definitions.
- `NN/Runtime`: executable autograd, optimizers, training loops, CUDA boundary,
  PyTorch import/export, and RL runtime support.
- `NN/Backend`: contract-carrying kernel capsules, the planner, backend profiles, execution
  audits, and the contract check that accepts or rejects a kernel plan.
- `NN/IR` and `NN/GraphSpec`: graph IR, graph semantics, and typed architecture
  descriptions.
- `NN/Proofs`: tensor algebra, selected autograd correctness theorems, analytic derivatives,
  runtime approximation, and bridge proofs.
- `NN/Floats`: TorchLean's scalar integration and numerical proof adapters.
- FloatLib dependency: configurable executable formats, reference semantics, rounding proofs,
  and scalar intervals.
- `NN/MLTheory`: learning theory, robustness, CROWN/LiRPA, generative objectives,
  optimization theory, and related proof layers.
- `NN/Verification`: certificate checkers and CLI workflows.
- `NN/Examples`: quickstarts, runnable model examples, widgets, bundled verification assets,
  and interoperability workflows.
- `home_page/blueprint/TorchLeanBlueprint/Guide`: source for the guide.
- `home_page`: project website sources.

## Proofs And Runtime Boundaries

TorchLean proves properties of explicit Lean definitions. It also checks certificates produced by
external tools, including bound-propagation and scientific-computing workflows. An executable
certificate check reports acceptance by that checker. A semantic guarantee additionally requires
the checker's soundness theorem and its hypotheses; a Lean proof needs kernel-checked evidence of
acceptance. None of these checks certifies the program that produced the certificate.

CPU instructions, CUDA kernels, cuBLAS, LibTorch, PyTorch, Julia, and other external systems are
runtime providers. Their interfaces, assumptions, and available checks are listed in
[`docs/TRUST_BOUNDARIES.md`](docs/TRUST_BOUNDARIES.md). Third-party sources and licenses are listed in
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md), and
[`docs/AI_USAGE.md`](docs/AI_USAGE.md) describes the
project's use of coding assistants.

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
