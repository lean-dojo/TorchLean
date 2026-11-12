# Scripts

Tools for building TorchLean, preparing data, producing certificates and publishing the website.
Run commands from the repository root, and pass `--help` to any command-line tool below.

| Path | Purpose |
| --- | --- |
| `lake.sh` | Run Lake with separate build caches for each toolchain and CPU/LibTorch profile. |
| `libtorch_build.py` | Build and cache the complete LibTorch C++ backend through SDK CMake. |
| `checks/repo_lint.py` | Check source hygiene, public interfaces, and documentation references. |
| `checks/dependency_audit.py` | Inspect imports and check library dependency boundaries. |
| `checks/libtorch_abi.py` | Check the ATen operator schemas and torch C functions the LibTorch bridge calls against `docs/TRUST_BOUNDARIES.md`, and regenerate `csrc/libtorch/schemas.h`. `lake.sh lint` runs the source checks, and `libtorch_build.py` checks the LibTorch SDK and the built library. |
| `checks/Library.lean` | Check that every theorem of the libraries depends on at most `propext`, `Classical.choice` and `Quot.sound`, that the operators have their `#roundoff_coverage` registrations, equation lemmas and routes, and that the outcomes of their analyses match `checks/roundoff_baseline.jsonl`. `lake.sh lint` runs it. A worse outcome fails, and so does a new or better one until `TORCHLEAN_ROUNDOFF_BASELINE=update scripts/lake.sh env lean scripts/checks/Library.lean` rewrites the file; that refuses while there are regressions, which `TORCHLEAN_ROUNDOFF_BASELINE=accept` writes too. |
| `checks/cuda.sh` | Run GPU `parity` (binary32 vs ATen/C ABI) or `sanitize` (Compute Sanitizer). |
| `lean_allocator.py`, `checks/allocator.py` | Build the private Linux allocator and check its memory-release behavior. |
| `docs/build_site.sh` | Build the API reference, Lean guide, import graph, and Jekyll website. |
| `bench/` | Measure runtime and elaboration times, or compare them between two checkouts. |
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

The default build provides the CPU runtime and needs no LibTorch SDK or CUDA toolkit. Its GPU
buffer symbols fail with a message to rebuild with `-K cuda=true`. The commands below check CPU
behavior only.

Install the pinned Python dependencies for CPU interoperability checks:

```bash
python3 -m pip install -r scripts/checks/requirements-interop.txt
python3 -c 'import torch, numpy, onnx; print(torch.__version__, numpy.__version__, onnx.__version__)'
scripts/lake.sh -Kcuda=false build TorchLean.Tensor TorchLean.CUDA TorchLean.NN TorchLean.Verify \
  TorchLeanExamples TorchLeanTest verify
TORCHLEAN_REQUIRE_INTEROP=1 scripts/lake.sh -Kcuda=false test
scripts/lake.sh -Kcuda=false lint
```

That requirements file pins CPU PyTorch 2.8.0 for the interoperability checks. The live operator
captures in `scripts/checks/test_ir_capture.py` and the differential fuzzer
`scripts/checks/test_frontend_fuzz.py` (seeded random programs through torch.export, FX, Dynamo and
eager JAX, every value compared with the framework's; `TORCHLEAN_FUZZ_SEED`, `TORCHLEAN_FUZZ_COUNT`)
also need the `verify` executable built above; `lake test` runs both.
Their JAX cases run only where JAX is installed, and `TORCHLEAN_REQUIRE_JAX=1` turns a missing JAX
into an error. The requirements file does not select the LibTorch backend SDK. Install the
requirements in a separate Python environment: in a GPU SDK environment they can replace its
CUDA PyTorch package.

On Linux, the build links a private allocator object and leaves the installed Lean runtime
unmodified. It downloads the pinned mimalloc source archive once and checks its SHA-256 digest.
For offline builds, set `TORCHLEAN_MIMALLOC_ARCHIVE` to a local copy of that archive. If you
change the allocator patch or update Lean, check the built object with:

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

`-Kcuda=true` builds the complete LibTorch backend and needs a Linux host with a CUDA-capable GPU
and a full LibTorch SDK of torch 2.13 or later, whose root contains `include/`, `lib/` and
`share/cmake/Torch/TorchConfig.cmake`. A CUDA-enabled Python PyTorch installation can provide this
root, but partial header snapshots cannot. The backend calls torch through the LibTorch Stable
ABI, so the built library also runs with the runtime of a later release, and the [backend's notes
on torch versions](../csrc/libtorch/README.md#torch-versions) list the tested releases. Attention
and its local VJP are composed in Lean from LibTorch numerical primitives. There is no fused
attention selector.

The SDK root is `-Klibtorch_home=PATH` if given, then `TORCHLEAN_LIBTORCH_HOME`, then `libtorch/`
under the package root, and relative paths resolve against the package. The build requires Linux,
Python 3, CMake 3.22 or newer, Make, the pinned Lean headers and a C++20 compiler compatible with
the SDK, whose CMake package supplies the ABI flags, any stricter C++ standard requirement, the
libraries and the runtime library paths. The build includes a compiler/ABI/link check and checks
the operators and torch functions the built library uses against `docs/TRUST_BOUNDARIES.md`.
Numerical compatibility still requires the GPU regressions, so rerun them after a change of the
SDK or of the torch runtime.

```bash
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
scripts/lake.sh -Kcuda=true build TorchLean.Tensor TorchLean.CUDA TorchLean.NN TorchLean.Verify \
  TorchLeanExamples TorchLeanTest torchlean verify nn_tests_suite
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
scripts/lake.sh -Kcuda=true lint
```

`TORCHLEAN_REQUIRE_CUDA=1` makes the test suite fail on a build without LibTorch or a host with no
CUDA device, and the sanitizer wrapper sets it. GPU execution needs a supported visible device, and
SDK configuration may probe the visible devices too. TorchLean builds one shared C++ adapter and
needs no `nvcc` itself, though the SDK's CMake package may require a matching CUDA development
toolkit and run its compiler during discovery.

Build controls:

| Setting | Meaning |
| --- | --- |
| `TORCHLEAN_CXX`, otherwise `CXX` | C++ compiler executable (default `c++`). |
| `TORCHLEAN_CMAKE` | CMake executable (default `cmake`). |
| `TORCHLEAN_LIBTORCH_JOBS` | Parallel build jobs, a positive integer (default `2`). |
| `TORCHLEAN_LIBTORCH_CMAKE_ARGS` | JSON array of individual `-Dkey=value` SDK discovery options. |
| `CXXFLAGS`, `LDFLAGS` | Compiler and linker flags read by CMake. |
| `-Kcuda_home=PATH` | Optional development toolkit root for SDK discovery. |
| `TORCH_CUDA_ARCH_LIST` | Architecture list for the SDK's configure probes, for example `8.0`. |

`TORCHLEAN_LIBTORCH_CMAKE_ARGS` cannot override the helper's SDK, compiler, output or rpath
settings. Rerun the GPU checks after a change to `CXXFLAGS` or `LDFLAGS` that affects the ABI or
arithmetic.

TorchLean sets no architecture override, and the SDK's architecture controls leave the
architectures and arithmetic of its packaged GPU kernels unchanged. Repeat the SDK and toolkit
settings on every later `build`, `exe` and `env` invocation.

The wrapper keeps separate `cpu` and `cuda-libtorch` build directories for each checkout and
toolchain under `${XDG_CACHE_HOME:-$HOME/.cache}/torchlean/build`, selects one through the
`.lake/build` symlink, and holds a checkout lock. `TORCHLEAN_BUILD_ROOT` sets another location,
and `TORCHLEAN_BUILD_PROFILE` supplies a custom profile name. Use distinct custom names for
different backends. `--torchlean-build-dir` prints the selected path without building.
The wrapper does not delete a real `.lake/build` directory: move it aside once before using the
wrapper.

For an early C++ build before the full Lean build, run the helper directly:

```bash
export TORCHLEAN_LEAN_PREFIX="$(lean --print-prefix)"
torchlean_build_dir="$(scripts/lake.sh -Kcuda=true --torchlean-build-dir)"
python3 scripts/libtorch_build.py --package-dir "$PWD" \
  --build-dir "$torchlean_build_dir" \
  --lean-include "$TORCHLEAN_LEAN_PREFIX/include"
export TORCHLEAN_BACKEND_LIBRARY="$torchlean_build_dir/libtorch/libtorchlean_libtorch.so"
```

The helper builds the backend's sources in `csrc/libtorch` (`torchlean.cpp`, `conv_pool.cpp`,
`tree.cpp`, `kernels.cpp` and `unstable.cpp`) with `TORCHLEAN_LIBTORCH` defined, runs
`scripts/checks/libtorch_abi.py` on the library, prints a cache fingerprint on stdout and sends
CMake's output to stderr. `libtorch/build.json` tracks the SDK's headers,
version and ABI, the compiler identity, flags, sources and discovered build dependencies, with
SDK libraries tracked by file metadata and the backend output by hash. Any changed input
triggers a clean private CMake build, and `libtorch/cmake/sdk.txt` records the discovered SDK
and compiler settings. The link-check executable is built in the same CMake project as the
backend, so the SDK's alias targets remain available. It is never executed.

Lake links the executables with one shared backend, which finds the SDK's libraries through the
runpath the SDK supplies. A runpath serves only the backend's own dependencies, so each SDK library
must find its own, `libc10` among them, through its own search path, as pip torch's libraries do
with `$ORIGIN`, or through `LD_LIBRARY_PATH` or `ldconfig`. Keep the backend at its resolved
absolute build path and the selected SDK in place while running linked executables.
`LD_LIBRARY_PATH` naming another release's `torch/lib` and NVIDIA libraries runs them with that
release's runtime. Run the executables directly then: the build environment includes
`LD_LIBRARY_PATH`, so a Lake command with another value rebuilds the backend. The package's CPU
dynamic library for native `#eval` calls is the only automatically loaded evaluation library.
CUDA validation uses compiled executables.

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

The C++ regression runs separately from `lake test`, in a temporary build directory so that
generating the CPU Lean reference can switch profiles. Keep manual CMake test builds outside the
`.lake/build` symlink and pass an absolute backend path.
See [the harness README](../csrc/libtorch/tests/elementwise/README.md) for coverage and direct CMake
commands, or use `cuda.sh parity --help` and `cuda.sh sanitize --help` for options.
These checks cover the paths a run executes. They do not cover every input or every native memory
operation.

## Documentation and data

Build the full site with `scripts/docs/build_site.sh`. It builds the library, API reference, guide,
dependency graph and Jekyll pages, then checks local links. Named steps run parts of it: `lean`,
`api`, `guide`, `graphs`, `bundle`, `timings`, `website` and `links`, for example
`scripts/docs/build_site.sh guide website`. With `JEKYLL_ENV=production` it builds and checks the
site for the published base URL `/TorchLean` instead of a local preview. The API post-processor
removes local dependency pages and points their links upstream, and a second run with
`--importgraph PAGE` styles the import viewer. The guide post-processor polishes the generated
HTML and checks the KaTeX runtime, section anchors and page layout.

Public-site CI runs these steps in production mode, except `lean` (the API step builds the
modules it documents) and `bundle` (`ruby/setup-ruby` installs the bundle). It caches the
generated API reference, guide and import graphs, keyed on their Lean, dependency, guide-asset and
build-tool inputs. On an exact cache hit, a Markdown or CSS change rebuilds only Jekyll and runs
the link checks before deployment. A changed input or an unavailable cache runs the full
documentation build, and generated pages are reused only under their own input key. The first
successful build seeds the cache.

The performance page reads `home_page/assets/ci-timings.json`, which the `timings` step
refreshes. It shows the API documentation, guide, website and link-check times of the CI job
`Build project`, found by step name, so the CI step names must match the names that
`home_page/performance/index.md` reads. For a Markdown-only preview, refresh it with
`python3 scripts/docs/ci_timings.py` before running Jekyll. The updater uses `GITHUB_TOKEN` or
`GH_TOKEN` when set and keeps the saved snapshot if GitHub is unreachable. Tokens stay in the
build environment, and visitors make no GitHub API calls.

The runtime section of the same page shows `home_page/assets/runtime-bench.json`, the results
of one `scripts/bench/bench.py` run. `python3 scripts/docs/runtime_bench.py --from SUMMARY.json`
saves a run's results there, and the `timings` step renders the section from the saved file.

| Page | Source to edit |
| --- | --- |
| Home, installation and getting started | `home_page/index.md`, `home_page/installation/index.md`, `home_page/start/index.md`, shared CSS/JS/assets |
| Examples | `home_page/examples/**/index.md` and matching `TorchLeanExamples/**` sources |
| Guide and formalization map | `home_page/blueprint/TorchLeanBlueprint/{Guide,FormalizationMap}/` |
| API reference | `TorchLean/**/*.lean` and `TorchLeanExamples/**/*.lean` module and declaration docstrings |
| Graphs and import explorer | `home_page/graphs/index.md`, `checks/dependency_audit.py` |
| Companion projects | `home_page/tools/index.md` (link to each project's own documentation) |
| Performance and updates | `home_page/performance/index.md`, `home_page/updates/index.md` |
| CUDA and trust boundaries | `home_page/cuda/index.md`, `docs/TRUST_BOUNDARIES.md` |

Edit the sources. The HTML under `home_page/docs`, `home_page/importgraph` and `home_page/_site` is
generated. Jekyll excludes the guide package, and the full builder copies the guide's
generated pages to `home_page/_site/blueprint` after Jekyll finishes.

For local preview, use Ruby 3.2 or newer (below 4.0) and Bundler 2.3.14
(`gem install bundler:2.3.14` if needed). Native gem extensions need Ruby headers and build tools,
typically `ruby-dev` and `build-essential` on Ubuntu.

```bash
cd home_page
bundle config set path vendor/bundle
bundle _2.3.14_ install
bundle _2.3.14_ exec jekyll serve --config _config.yml,_config_dev.yml --port 4001
```

Open `http://127.0.0.1:4001/`, the address `home_page/_config_dev.yml` sets. A Markdown-only build
uses `bundle _2.3.14_ exec jekyll build --config _config.yml,_config_dev.yml`, which does not
regenerate the Lean guide or the API reference. For a guide-only edit, run
`scripts/docs/build_site.sh guide website`.

The full builder runs DocGen with `DISABLE_EQUATIONS=1`, which stops it rendering every imported
equation lemma but keeps declaration types, docstrings and source links. It clears DocGen's cached
database and doc data first, because the cache does not track that setting. Native source notes
come from `TorchLean.CUDA.LibTorch.Trusted`.

DocGen and Jekyll use `$...$` for inline math and `$$...$$` for display math through MathJax.
Verso uses its bundled KaTeX: write math as a code literal with a `$` or `$$` prefix, and ordinary
backticks stay code. Before publishing, rebuild the affected output and check the example commands,
artifact paths and producer and native trust boundaries. The formalization map shows selected
declarations and proofs, the import explorer shows source dependencies, and the runtime IR
represents model computations.

`verification/abcrown/export_leaf_artifact.py` converts a raw leaf dump from an external
alpha-beta-CROWN-style verifier into the JSON that `verify -- abcrown-leaf` checks.
`verification/interval.py` holds the outward-rounded binary64 interval helpers that the PINN and
IBP certificate producers import. The other producers live beside their examples under
`TorchLeanExamples/Verification/`.

Example dataset downloads need Python 3.12 or newer and NumPy, and WikiText also needs PyArrow.
The basic build wrapper needs only Bash and Python 3, and a CPU installation downloads no datasets.

Keep downloaded data and generated artifacts under `data/`, `_out/`, or a temporary directory.
