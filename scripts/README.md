# Scripts

Support tools for building TorchLean, preparing data, producing certificates, and publishing the
website. Run commands from the repository root. Each Python command provides `--help`.

| Path | Purpose |
| --- | --- |
| `lake.sh` | Run Lake with separate `cpu` and `cuda-libtorch` build caches. |
| `libtorch_build.py` | Build and cache the complete LibTorch C++ backend through SDK CMake. |
| `checks/check.sh` | Run the local build, test, and lint gate. |
| `checks/repo_lint.py` | Check source hygiene, public interfaces, and documentation references. |
| `checks/dependency_audit.py` | Inspect imports and check library dependency boundaries. |
| `checks/check_interop_dependencies.py` | Check the active Python's PyTorch, NumPy, and ONNX imports. |
| `checks/cuda_sanitize_tests.sh` | Run the LibTorch CUDA suite under Compute Sanitizer. |
| `checks/cuda_float32_parity.sh` | Compare Lean binary32 cases with ATen and the production C ABI. |
| `docs/build_site.sh` | Build the API reference, Lean guide, import graph, and Jekyll website. |
| `datasets/` | Download example data, convert tensor files, and plot training logs. |
| `rl/gymnasium_server.py` | Serve Gymnasium environments or export a seeded rollout with `--out`. |
| `verification/` | Produce artifacts consumed by Lean checkers. |
| `generate_unicode_table.py` | Regenerate the lexer table with pinned CPython/Unicode versions. |

```bash
scripts/lake.sh build
scripts/checks/check.sh
python3 scripts/datasets/download_example_data.py --help
python3 scripts/datasets/torchlean_data_convert.py --help
python3 scripts/rl/gymnasium_server.py --help
scripts/docs/build_site.sh
```

## CPU checks

The default build provides the `pureLean` and `portableCPU` runtimes. It needs no LibTorch SDK or
CUDA toolkit, and the GPU buffer symbols fail with a message to rebuild.
Hosted CI selects `cuda=false` explicitly;
its results establish CPU behavior and do not establish GPU correctness.

Install the pinned Python dependencies for CPU interoperability checks:

```bash
python3 -m pip install -r scripts/checks/requirements-interop.txt
python3 scripts/checks/check_interop_dependencies.py
TORCHLEAN_REQUIRE_INTEROP=1 scripts/checks/check.sh --ci-all
```

That requirements file pins CPU PyTorch 2.8.0 for hosted interoperability checks. It does not
select the LibTorch backend SDK. Use a separate Python environment for these dependencies;
installing the CPU requirements into a GPU SDK environment can replace its CUDA PyTorch package.

## LibTorch CUDA build

The CUDA build needs a Linux host with a CUDA-capable GPU and a full LibTorch SDK. This tree was
tested locally against pip torch 2.13.0+cu130 with CUDA 13.0 on A100, and previously against a
PyTorch 2.12 nightly. The backend uses a few internal ATen entry points (the fused attention
selector and its forward and backward kernels), so other SDK versions may fail to compile or need
the GPU regressions rerun.

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
  torchlean verify nn_tests_suite libtorch_sdpa_test
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true exe libtorch_sdpa_test
scripts/checks/check.sh --libtorch-home "$TORCHLEAN_LIBTORCH_HOME" --ci-all
```

`TORCHLEAN_REQUIRE_CUDA=1` rejects builds without LibTorch and missing CUDA devices.
`check.sh --cuda`, its SDK/toolkit options, and the sanitizer wrapper enable this mode
automatically. GPU execution needs a supported visible device; SDK configuration may also probe
visible devices. TorchLean builds six ordinary C++ translation units and has no independent
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
`check.sh` and the sanitizer wrapper expose `--libtorch-home` and `--cuda-home`; repeat the same
configuration on later build, `exe`, and `env` invocations.

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

The helper builds `runtime.cpp`, `elementwise.cpp`, `kernels.cpp`, `conv_pool.cpp`, `attention.cpp`,
and `blas.cpp` with `TORCHLEAN_LIBTORCH` defined. Its stdout is a cache fingerprint; CMake output
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

After building the production backend, the Float32 wrapper builds an ordinary C++ regression
executable against that library, the same SDK, and the Lean runtime:

```bash
scripts/checks/cuda_float32_parity.sh \
  --libtorch-home "$TORCHLEAN_LIBTORCH_HOME" \
  --backend-library "$TORCHLEAN_BACKEND_LIBRARY" \
  --lean-prefix "$TORCHLEAN_LEAN_PREFIX" --sweep 200000 --keep
scripts/checks/cuda_sanitize_tests.sh --libtorch-home "$TORCHLEAN_LIBTORCH_HOME"
scripts/checks/cuda_sanitize_tests.sh --libtorch-home "$TORCHLEAN_LIBTORCH_HOME" --all-tools
```

The C++ regression is a standalone CMake project, separate from `lake test` and hosted CPU CI.
The wrapper uses a temporary build directory so generating the CPU Lean reference can safely
switch profiles. Keep any manual CMake test build directory outside the `.lake/build` symlink,
and pass an absolute production backend path.

The Float32 harness consumes Lean's add/mul/div/sqrt/fma reference stream, checks ATen and
production C ABI behavior, and includes staged Adam and no-autograd regressions. Finite values
and signed zeros require exact agreement; NaN encoding differences are reported separately.
See [the harness README](../csrc/libtorch/tests/elementwise/README.md) for coverage and direct
CMake commands. It tests the selected prebuilt SDK. A finite sweep is validation, not a universal
proof.

The sanitizer wrapper forwards all SDK/toolkit options to both build and execution, defaults to
`memcheck`, and makes findings fail with exit code 99. `--all-tools` adds `racecheck`, `initcheck`,
and `synccheck`. The default `--target-processes application-only` allows the suite's intentional
self-reexec probe; use `all` only when child-process instrumentation is intended. With
`--skip-build`, select the same profile and SDK as the existing executable. These runs cover
executed paths in the native boundary and SDK; they do not establish general memory safety.

## Documentation and data

The site builder runs the documentation post-processors and link checker. The guide source lives
in `home_page/blueprint/`; generated pages go to `home_page/_site/blueprint/`.

The fragment-alias regression uses temporary HTML fixtures and the site link checker:

```bash
python3 -B -m unittest discover -s scripts/docs -p 'test_*.py'
```

Verification producers are grouped by artifact: `lirpa`, `abcrown`, `geometry3d`, `pinn`,
`robustness`, `splines`, and `two_stage`. Their example documentation gives the matching producer
and `verify` commands. `normalization_contract_probe.py` measures PyTorch normalization residuals
for the BugZoo examples.

Keep downloaded data and generated artifacts under `data/`, `_out/`, or a temporary directory.
Use temporary checks for routine refactors and remove them after validation. Retain small tests
for numerical and native-code behavior outside Lean's proofs.
