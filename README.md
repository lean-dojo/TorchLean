<h1 align="center">
  <img src="home_page/assets/media/brand/torchlean-logo.png" alt="TorchLean logo" width="88" align="center">
  Formalizing Neural Networks in Lean
</h1>

TorchLean is a Lean 4 framework for writing, running, inspecting, and verifying
neural-network programs. It provides typed tensors and model APIs, a shared graph
IR, runtime/autograd support, finite-precision semantics, certificate checkers,
CUDA/runtime boundaries, and examples across modern ML and scientific ML.

The detailed story lives in the project site, guide, API docs, and paper. This
README is the quick entry point: build the repo, run the first examples, find the
docs, and cite the project.

## Quickstart

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake build
lake exe torchlean mlp --cpu --steps 10
```

TorchLean is pinned by `lean-toolchain` and currently builds with
`leanprover/lean4:v4.29.0`.

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

Most downstream model and training files should start from the public facade:

```lean
import NN.API.Public
```

Use the broader umbrella when you want the maintained specification, IR, proof,
verification, examples, and widget surface:

```lean
import NN.Library
```

For local development against a checkout, use a path dependency instead:

```lean
require TorchLean from "../TorchLean"
```

Reservoir is not required for Git or path dependencies. It is only for package
discovery and versioned `require` lines once the repository is indexed.

## Repository Map

- `NN/API`: public facade for model, tensor, data, and training workflows.
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
- `NN/Examples`: quickstarts, model zoo commands, widgets, verification fixtures,
  and interoperability demos.
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
      author={Robert Joseph George and Jennifer Cruden and Xiangru Zhong and Huan Zhang and Anima Anandkumar},
      year={2026},
      eprint={2602.22631},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2602.22631},
}
```

## License

TorchLean is released under the MIT License. See `LICENSE`.
