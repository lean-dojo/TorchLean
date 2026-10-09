# Scripts

Support tools for building TorchLean, preparing data, producing certificates, and publishing the
website. Run commands from the repository root; use `--help` on the command-line tools below.

| Path | Purpose |
| --- | --- |
| `lake.sh` | Run Lake with separate `cpu` and `cuda-libtorch` build caches. |
| `libtorch_build.py` | Build and cache the complete LibTorch C++ backend through SDK CMake. |
| `checks/repo_lint.py` | Check source hygiene, public interfaces, and documentation references. |
| `checks/dependency_audit.py` | Inspect imports and check library dependency boundaries. |
| `checks/cuda.sh` | GPU checks: `parity` compares binary32 with ATen/C ABI; `sanitize` runs Compute Sanitizer. |
| `lean_allocator.py`, `checks/allocator.py` | Build the private Linux allocator and check its memory-release behavior. |
| `docs/build_site.sh` | Build the API reference, Lean guide, import graph, and Jekyll website. |
| `datasets/` | Download example data and convert tensor files. |
| `rl/gymnasium_server.py` | Serve Gymnasium environments or export a seeded rollout with `--out`. |
| `verification/` | Produce artifacts consumed by Lean checkers. |
| `generate_unicode_table.py` | Regenerate the lexer table with pinned CPython/Unicode versions. |

```bash
python3 scripts/datasets/download_example_data.py --help
python3 scripts/datasets/download_example_data.py --wikitext --config wikitext-2-raw-v1
python3 scripts/datasets/torchlean_data_convert.py --help
python3 scripts/rl/gymnasium_server.py --help
scripts/docs/build_site.sh
```

For library build targets and contribution checks, see
[Contributing](../docs/CONTRIBUTING.md#build-and-check).

## CPU checks

The default build provides the `pureLean` and `portableCPU` runtimes. It needs no LibTorch SDK or
CUDA toolkit, and the GPU buffer symbols fail with a message to rebuild.
Hosted CI selects `cuda=false` explicitly;
its results establish CPU behavior and do not establish GPU correctness.

Install the pinned Python dependencies for CPU interoperability checks:

```bash
python3 -m pip install -r scripts/checks/requirements-interop.txt
python3 -c 'import torch, numpy, onnx; print(torch.__version__, numpy.__version__, onnx.__version__)'
scripts/lake.sh -Kcuda=false build NN.CI.All
TORCHLEAN_REQUIRE_INTEROP=1 scripts/lake.sh -Kcuda=false test
scripts/lake.sh -Kcuda=false lint
```

That requirements file pins CPU PyTorch 2.8.0 for hosted interoperability checks. It does not
select the LibTorch backend SDK. Use a separate Python environment for these dependencies;
installing the CPU requirements into a GPU SDK environment can replace its CUDA PyTorch package.

On Linux, the build links a private allocator object rather than modifying the installed Lean
runtime. If you change its patch or update Lean, check the built object with:

```bash
torchlean_build_dir="$(scripts/lake.sh --torchlean-build-dir)"
scripts/lake.sh build torchlean_allocator
python3 scripts/checks/allocator.py --lean-prefix "$(lean --print-prefix)" \
  --object "$torchlean_build_dir/torchlean_allocator.o" --output _out/allocator
```

The harness compares the installed and private allocators, including delayed arena purging and
concurrent allocations. See [Trust boundaries](../docs/TRUST_BOUNDARIES.md#native-host-allocation)
for the scope of these checks.

## LibTorch CUDA build

The CUDA build needs a Linux host with a CUDA-capable GPU and a full LibTorch SDK. See the
[backend's SDK notes](../csrc/libtorch/README.md#tested-sdk-versions) for tested configurations.
Attention and its local VJP are composed in Lean from LibTorch numerical primitives; there is no
fused attention selector. Rerun the GPU checks when changing SDKs.

`cuda=true` selects the complete LibTorch backend. The SDK root must contain `include/`,
`lib/`, and `share/cmake/Torch/TorchConfig.cmake`. A CUDA-enabled Python PyTorch installation can
provide this root. Partial header snapshots alone cannot build the backend.

SDK selection uses `-Klibtorch_home=PATH`, then `TORCHLEAN_LIBTORCH_HOME`, then `libtorch/` under
the package root. Relative configured paths are package-relative. The build currently requires
Linux, Python 3, CMake 3.22 or newer, Make, the pinned Lean headers, and a C++20 compiler compatible
with the SDK. SDK CMake supplies ABI flags, stricter C++ standard requirements, libraries, and
runtime library paths. The build includes a compiler/ABI/link check; numerical compatibility still
requires the GPU regressions.

```bash
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
scripts/lake.sh -Kcuda=true build NN NNCI NNExamples NNTests \
  torchlean verify nn_tests_suite
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
scripts/lake.sh -Kcuda=true build NN.CI.All
scripts/lake.sh -Kcuda=true lint
```

`TORCHLEAN_REQUIRE_CUDA=1` rejects builds without LibTorch and missing CUDA devices.
The sanitizer wrapper enables this mode automatically. GPU execution needs a supported visible
device; SDK configuration may also probe
visible devices. TorchLean builds one shared C++ adapter and has no independent
nvcc prerequisite. The SDK's own CMake package may require a matching CUDA development toolkit
and invoke its compiler during discovery.

Build controls:

| Setting | Meaning |
| --- | --- |
| `TORCHLEAN_CXX`, otherwise `CXX` | C++ compiler executable; defaults to `c++`. |
| `TORCHLEAN_CMAKE` | CMake executable; defaults to `cmake`. |
| `TORCHLEAN_LIBTORCH_JOBS` | Positive build parallelism; defaults to `2`. |
| `TORCHLEAN_LIBTORCH_CMAKE_ARGS` | JSON array of individual `-Dkey=value` options for SDK discovery. SDK/compiler/output/rpath settings owned by the helper cannot be overridden here. |
| `CXXFLAGS`, `LDFLAGS` | Compiler/linker flags read by CMake; ABI or arithmetic changes require renewed validation. |
| `-Kcuda_home=PATH` | Optional development toolkit root for SDK discovery. |
| `TORCH_CUDA_ARCH_LIST` | SDK architecture discovery control, for example `8.0` for its configure probes. |

TorchLean sets no architecture override. SDK architecture controls affect discovery probes;
they do not change the architectures or arithmetic of the SDK's packaged GPU kernels.
Repeat SDK/toolkit configuration on later build, `exe`, and `env` invocations.

The wrapper uses separate `cpu` and `cuda-libtorch` cache directories, selected through
`.lake/build`, and holds a checkout lock. `TORCHLEAN_BUILD_ROOT` changes their location;
`TORCHLEAN_BUILD_PROFILE` supplies a custom profile name. Use distinct custom names for
different backends. `--torchlean-build-dir` prints the selected path without building.
An existing real `.lake/build` directory must be moved aside by its owner before using the
wrapper; the wrapper will not delete it.

For an early C++ build before the full Lean build, run the helper directly:

```bash
export TORCHLEAN_LEAN_PREFIX="$(lean --print-prefix)"
torchlean_build_dir="$(scripts/lake.sh -Kcuda=true --torchlean-build-dir)"
python3 scripts/libtorch_build.py --package-dir "$PWD" \
  --build-dir "$torchlean_build_dir" \
  --lean-include "$TORCHLEAN_LEAN_PREFIX/include"
export TORCHLEAN_BACKEND_LIBRARY="$torchlean_build_dir/libtorch/libtorchlean_libtorch.so"
```

The helper builds `torchlean.cpp` with `TORCHLEAN_LIBTORCH` defined.
Its stdout is a cache fingerprint; CMake output
goes to stderr. SDK/version/ABI headers, compiler identity, flags, sources, and discovered build
dependencies are tracked in `libtorch/build.json`. SDK library replacements are tracked by file
metadata; the backend output is hashed. Changed inputs trigger a clean private CMake build.
`libtorch/cmake/sdk.txt` records discovered SDK/compiler settings. The link-check executable is
built in the same CMake project as the backend, so SDK alias targets remain available; it is never
executed.

Lake links one shared backend with SDK-derived transitive libraries and rpath. Keep the backend
at its resolved absolute build path and retain the selected SDK when running linked executables.
The package's existing CPU dynamic library for native `#eval` calls remains the only automatically
loaded evaluation library; CUDA validation uses compiled executables.

## GPU regression tools

After building the production backend, `cuda.sh parity` builds an ordinary C++ regression
executable against that library, the same SDK, and the Lean runtime:

```bash
scripts/checks/cuda.sh parity \
  --libtorch-home "$TORCHLEAN_LIBTORCH_HOME" \
  --backend-library "$TORCHLEAN_BACKEND_LIBRARY" \
  --lean-prefix "$TORCHLEAN_LEAN_PREFIX" --sweep 200000 --keep
scripts/checks/cuda.sh sanitize --libtorch-home "$TORCHLEAN_LIBTORCH_HOME"
scripts/checks/cuda.sh sanitize --libtorch-home "$TORCHLEAN_LIBTORCH_HOME" --all-tools
```

The C++ regression runs separately from `lake test` and hosted CPU CI. The wrapper uses a temporary
build directory so generating the CPU Lean reference can switch profiles. Keep manual CMake test
builds outside the `.lake/build` symlink and pass an absolute backend path.
See [the harness README](../csrc/libtorch/tests/elementwise/README.md) for coverage and direct CMake
commands, or use `cuda.sh parity --help` and `cuda.sh sanitize --help` for options.
These checks cover executed paths, not every input or native memory operation.

## Documentation and data

Build the full site from the repository root with `scripts/docs/build_site.sh`. It builds the
library, API reference, guide, dependency graph and Jekyll pages, then checks local links.
The API post-processor removes local dependency pages and redirects their links upstream.
Its `--importgraph PAGE` option also handles the import viewer. The guide post-processor checks
the KaTeX runtime, section anchors and page layout after polishing the generated HTML.

The performance page reads `home_page/assets/ci-timings.json`, refreshed by the site build.
For a Markdown-only preview, refresh it with `python3 scripts/docs/ci_timings.py` before Jekyll.
The updater uses `GITHUB_TOKEN` or `GH_TOKEN` when available and keeps the saved snapshot if
GitHub cannot be reached. Tokens stay in the build environment; visitors make no GitHub API calls.

| Page | Source to edit |
| --- | --- |
| Home, installation and getting started | `home_page/index.md`, `home_page/installation/index.md`, `home_page/start/index.md`, shared CSS/JS/assets |
| Examples | `home_page/examples/**/index.md` and matching `NN/Examples/**` sources |
| Guide and formalization map | `home_page/blueprint/TorchLeanBlueprint/{Guide,FormalizationMap}/` |
| API reference | `NN/**/*.lean` module and declaration docstrings |
| Graphs and import explorer | `home_page/graphs/index.md`, `checks/dependency_audit.py` |
| Companion projects | `home_page/tools/index.md`; link to each project's own documentation |
| Performance and updates | `home_page/performance/index.md`, `home_page/updates/index.md` |
| CUDA and trust boundaries | `home_page/cuda/index.md`, `docs/TRUST_BOUNDARIES.md` |

Edit sources rather than generated HTML under `home_page/docs`, `home_page/importgraph` or
`home_page/_site`. The guide package is excluded from Jekyll; the full builder installs its
generated pages at `home_page/_site/blueprint` after Jekyll finishes.

For local preview, use Ruby 3.2 or newer (below 4.0) and Bundler 2.3.14
(`gem install bundler:2.3.14` if needed). Native gem extensions need Ruby headers and build tools,
typically `ruby-dev` and `build-essential` on Ubuntu.

```bash
cd home_page
bundle config set path vendor/bundle
bundle _2.3.14_ install
bundle _2.3.14_ exec jekyll serve --config _config.yml,_config_dev.yml --port 4000
```

Open `http://127.0.0.1:4000/`; change `--port` if it is occupied. A Markdown-only build uses
`bundle _2.3.14_ exec jekyll build --config _config.yml,_config_dev.yml`. This does not regenerate
the Lean guide or API reference. For a guide-only edit, build and check the blueprint package,
then copy its assets and run the guide post-processor as shown in `docs/build_site.sh`.

The full builder sets `DISABLE_EQUATIONS=1` for DocGen and clears its cached database and doc data:
the cache does not track that setting. This avoids rendering every imported equation lemma while
retaining declaration types, docstrings and source links. Native source notes come from
`NN.Runtime.Autograd.Engine.LibTorch.Trusted`.

DocGen and Jekyll use `$...$` for inline math and `$$...$$` for display math through MathJax.
Verso uses its bundled KaTeX: prefix a code literal with `$` or `$$`; ordinary backticks stay code.
Before publishing, rebuild the affected output and check example commands, artifact paths and
producer/native trust boundaries. The formalization map shows selected declarations and proofs;
the import explorer shows source dependencies, and the runtime IR represents model computations.

Verification producers are grouped by artifact: `lirpa`, `abcrown`, `geometry3d`, `pinn`,
`robustness`, `splines`, and `two_stage`. Their example documentation gives the matching producer
and `verify` commands.

Example dataset downloads require Python 3.12+ and NumPy. WikiText also needs PyArrow. This is
separate from the basic Bash/Python 3 build wrapper; CPU installation does not download datasets.

LiRPA fixtures use one producer: `verification/lirpa/export_cert.py MODEL`, where `MODEL` is
`mlp`, `cnn`, `attention`, `gru`, `transformer`, or `all`. Pass `--out-dir PATH` to avoid
replacing the bundled certificates.

PINN training uses `verification/pinn/train_pinn.py evolution` for space-time problems or
`stationary` for two spatial coordinates. The choices share network and derivative setup,
but keep their distinct initial-condition and boundary sampling schedules.
Use `train_pinn.py export --in-dim 1 --out PATH` to export a fresh fixed-width tanh model,
or add `--ckpt PATH` to convert an existing checkpoint without training.

Geometry3D uses one renderer for both image overlays and interval diagnostics. Pass
`--view intervals --cert PATH --out PATH` to compare the exported bounding boxes without
running the Lean checker; the default overlay still runs it unless `--no-verify` is given.

Keep downloaded data and generated artifacts under `data/`, `_out/`, or a temporary directory.
