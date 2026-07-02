<h1 align="center">
  <img src="home_page/assets/media/brand/torchlean-logo.png" alt="TorchLean logo" width="88" align="center">
  Formalizing Neural Networks in Lean
</h1>

TorchLean is a Lean 4 framework for writing, running, inspecting, and verifying
neural-network programs. It provides typed tensors and model APIs, a shared graph
IR, runtime/autograd support, finite-precision semantics, certificate checkers,
CUDA/runtime boundaries, and examples across modern ML and scientific ML.

## Quickstart

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake build
python3 scripts/datasets/download_example_data.py --auto-mpg
lake exe torchlean mlp --cpu --steps 10
lake exe torchlean mlp --steps 10 --dtype float --backend eager

# Optional CUDA run, if the CUDA toolkit and an NVIDIA GPU are available:
lake build -K cuda=true
lake exe -K cuda=true torchlean mlp --cuda --steps 1000
```

The first MLP command uses the executable IEEE-style Float32 path. The second
uses Lean's builtin `Float` runtime path. The CUDA command uses the native GPU
runtime path and checks that the CUDA backend is available; it is not a trusted
proof boundary. Handwritten fused CUDA kernels are opt-in performance paths via
`--fast-kernels`; keep them off unless you are explicitly testing that backend.

TorchLean is pinned by `lean-toolchain` and currently builds with
`leanprover/lean4:v4.31.0`.

The public code shape is:

```lean
import NN
open TorchLean

def model :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def xs : Tensor.T Float (shape![4, 2]) :=
  tensorND! [4, 2] [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensorND! [4, 1] [0.2, 1.0, 1.0, 1.8]

def data : Trainer.Dataset (Shape.vec 2) (Shape.vec 1) :=
  Data.tensorDataset xs ys

def trainOnce : IO Unit := do
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.sgd { lr := 0.05 }
        backend := .compiled
        dtype := .float32 }
  let y0 ← trainer.eval (tensorND! [2] [0.5, -0.25])
  IO.println s!"initial={Tensor.pretty y0}"
  let trained ← trainer.train data { steps := 200, batchSize := 16, logEvery := 25 }
  trained.printSummary
```

## First Things To Try

```bash
lake exe torchlean --help
lake exe verify --help
lake exe verify -- torchlean-ibp
```

For the maintained example surface:

```bash
lake build NN.Examples.Zoo
```

## Documentation

- Project site: <https://lean-dojo.github.io/TorchLean/>
- Guide: <https://lean-dojo.github.io/TorchLean/blueprint/>
- API reference: <https://lean-dojo.github.io/TorchLean/docs/>
- Updates and recent validation notes: <https://lean-dojo.github.io/TorchLean/updates/>
- Paper: [*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631)
  (arXiv:2602.22631)

Detailed tutorials, verification chapters, model walkthroughs, CUDA notes, and
API-level reference material belong in the guide and generated docs rather than
in this README.

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

Use `import NN` for model, data, training, verification, and proof workflows. Focused imports such
as `NN.API.*`, `NN.GraphSpec.*`, `NN.Runtime.*`, or `NN.Proofs.*` are for files that deliberately
work inside one subsystem.

Downstream model and training files should start from:

```lean
import NN
open TorchLean
```

For local development against a checkout, use a path dependency instead:

```lean
require TorchLean from "../TorchLean"
```

## Repository Map

- `NN.lean`: canonical umbrella import for model, tensor, data, training, verification,
  and proof workflows.
- `NN/API`: subsystem APIs behind `import NN`.
- `NN/Spec`: mathematical tensor, layer, model, and dynamical-system definitions.
- `NN/Runtime`: executable autograd, optimizers, training loops, CUDA boundary,
  PyTorch import/export, and RL runtime support.
- `NN/IR` and `NN/GraphSpec`: graph IR, graph semantics, and typed architecture
  descriptions.
- `NN/Proofs`: tensor algebra, autograd correctness, analytic derivatives,
  runtime approximation, and bridge proofs.
- `NN/Floats`: finite-precision models, IEEE-style executable semantics,
  NeuralFloat formats, and error-bound infrastructure.
- `NN/MLTheory`: learning theory, robustness, CROWN/LiRPA, generative objectives,
  optimization theory, and related proof layers.
- `NN/Verification`: certificate checkers and CLI workflows.
- `NN/Examples`: quickstarts, model zoo commands, widgets, bundled verification assets,
  and interoperability workflows.
- `blueprint/TorchLeanBlueprint/Guide`: source for the public guide.
- `home_page`: project website sources.

## Correctness and Boundaries

For correctness claims, trust assumptions, and third-party tooling:

- `TRUST_BOUNDARIES.md`
- `AI_USAGE.md`
- `THIRD_PARTY_NOTICES.md`
- `CONTRIBUTING.md`

Lean proofs, executable checkers, Lake builds, tests, and explicit
trust-boundary documentation are the source of truth for what is proved,
checked, or assumed.

## Citation

For the current paper BibTeX, use the citation metadata on arXiv:

```bibtex
@misc{george2026torchleanformalizingneuralnetworks,
      title={TorchLean: Formalizing Neural Networks in Lean},
      author={Robert Joseph George and Jennifer Cruden and Will Adkisson and Xiangru Zhong and Huan Zhang and Anima Anandkumar},
      year={2026},
      eprint={2602.22631},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2602.22631},
}
```

## License

TorchLean is released under the MIT License. See `LICENSE`.
