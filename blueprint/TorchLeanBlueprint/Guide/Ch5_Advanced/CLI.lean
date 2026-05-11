import VersoManual

open Verso.Genre Manual

#doc (Manual) "Command-Line Tools" =>
%%%
tag := "cli"
%%%

The public command line has three jobs: run model demos, run individual Lean example files, and run
verification tools. Everything else should be treated as internal unless a page points to it
explicitly.

TorchLean keeps that public surface intentionally small. The idea is not to expose every internal
script as a public command. The idea is to make the first few commands easy to remember, easy to
teach, and easy to cite in the book:

- `lake exe torchlean <demo> [args...]` for the main model zoo,
- `lake env lean --run NN/Examples/.../Foo.lean -- [args...]` for files in the
  [examples tree](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/),
- `lake exe verify -- ...` for verification workflows and checkers.

Three command families cover most everyday use: run a model zoo demo, run an advanced example
file, and run a verifier tool.

| Command | Purpose |
|---|---|
| `lake build` | build the project |
| `lake build NN.Examples.Zoo` | build curated examples |
| `lake exe torchlean --help` | list model demos |
| `lake exe torchlean <demo>` | run a model-zoo demo |
| `lake env lean --run NN/Examples/.../Foo.lean` | run a direct example file |
| `lake exe verify -- list` | list verifier tools |
| `lake exe verify -- <tool>` | run a verifier/checker |
| `scripts/docs/build_site.sh` | build the website/docs |

# Building with CUDA (optional)

GPU-backed demos require a CUDA-enabled build of the Lean project so the native archives in the
[CUDA source tree](https://github.com/lean-dojo/TorchLean/tree/main/csrc/cuda/) link against the toolkit:

```
lake build -R -K cuda=true
lake exe -K cuda=true torchlean gpt2 --cuda --steps 1
```

If `cuda=true` is not set, the same symbols resolve to CPU stubs and `--cuda` may error or fall back
depending on the demo. See *GPU and CUDA* for the build/runtime split and each example's module
header for model specific flags. Verification CLI tools (`lake exe verify`) do not need CUDA.

# The Two Commands To Remember

Run one model zoo demo:

```
lake exe torchlean <demo> [args...]
```

Run one advanced example file:

```
lake env lean --run NN/Examples/.../Foo.lean -- [args...]
```

List verification workflows:

```
lake exe verify -- list
```

Run one verification workflow:

```
lake exe verify -- <tool> [args...]
```

That is most of the public CLI. The usual loop is list, pick a name, run it, then dig into source
only when output or errors warrant it.

Common failure modes are usually simple: CUDA examples need `-K cuda=true`; real-data demos need
the dataset files under `data/real`; some verifier tools need an external artifact; and Python
producer workflows need their Python dependencies installed before Lean can check the exported
artifact.

# A Good First Smoke Test

After cloning, a fast smoke test is:

```
lake build
lake build NN.Examples.Zoo
lake env lean --run NN/Examples/Quickstart/TensorBasics.lean
lake exe torchlean mlp --cpu --steps 10
lake exe verify -- torchlean-ibp
```

Those commands answer four different questions:

- does the project build?
- does the typed-tensor layer work?
- does the public model runner and training API feel reasonable?
- does the verification pipeline complete?

This sequence is a confidence check rather than full coverage: within a minute or two it shows
whether the project builds, whether a small training run works, and whether the verifier surface is
present.

# Examples In Practice

Model examples use the `torchlean` runner:

```
lake exe torchlean --help
lake exe torchlean cnn --cpu --n-total 20 --steps 1
```

Some tutorial and advanced examples are ordinary Lean `--run` programs under `NN/Examples/*`. Pick
one and run it directly:

```
lake env lean --run NN/Examples/Quickstart/TensorBasics.lean
```

For data-backed model runs, prepare the public example datasets first:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --tinystories-valid --cifar10
```

# Verification In Practice

The verification side has the same small feel: list the registered tools, then run one directly.

- Show registered verification tools:

```
lake exe verify -- list
```

- Run the smallest IR-to-bounds workflow:

```
lake exe verify -- torchlean-ibp
```

- Run the CROWN operator demo:

```
lake exe verify -- torchlean-crown-ops
```

These commands are intentionally plain, so the verification workflow is easy to invoke without
knowing the internal directory structure of the repository.

# Website Build

The website combines the API docs, this book, and the homepage. The local build command is kept in
the repository:

```
scripts/docs/build_site.sh
```

That script builds the API docs with equation rendering disabled, rebuilds the Verso guide (from
the `blueprint/` package), and installs the homepage bundle.

For more runnable demos, see *Example Walkthroughs*. For what `verify` subcommands do internally, see
*Verification*.
