# Scripts

Support tools for building TorchLean, preparing data, producing certificates, and publishing the
website. Run commands from the repository root. Each Python command provides `--help`.

| Path | Purpose |
| --- | --- |
| `lake.sh` | Run Lake with separate CPU, CUDA, and LibTorch build caches. |
| `checks/check.sh` | Run the local build, test, and lint gate. |
| `checks/repo_lint.py` | Check source hygiene, public interfaces, and documentation references. |
| `checks/dependency_audit.py` | Inspect imports and check library dependency boundaries. |
| `checks/cuda_sanitize_tests.sh` | Run native CUDA checks under Compute Sanitizer. |
| `checks/cuda_float32_parity.sh` | Compare executable binary32 semantics with native CPU/CUDA bits. |
| `docs/build_site.sh` | Build the API reference, Lean guide, import graph, and Jekyll website. |
| `datasets/` | Download example data, convert tensor files, and plot training logs. |
| `rl/gymnasium_server.py` | Serve Gymnasium environments or export a seeded rollout with `--out`. |
| `verification/` | Produce artifacts consumed by Lean checkers. |
| `generate_unicode_table.py` | Regenerate the lexer table with pinned CPython/Unicode versions. |

```bash
scripts/lake.sh build
scripts/checks/check.sh
scripts/checks/cuda_float32_parity.sh
python3 scripts/datasets/download_example_data.py --help
python3 scripts/datasets/torchlean_data_convert.py --help
python3 scripts/rl/gymnasium_server.py --help
scripts/docs/build_site.sh
```

CUDA builds default to `cuda_arch=all-major`. For a known deployment target, pass its architecture
explicitly; these commands select `sm_80` for an A100:

```bash
scripts/lake.sh -R -K cuda=true -K cuda_arch=sm_80 build nn_tests_suite
scripts/checks/check.sh --cuda-arch sm_80
scripts/checks/cuda_sanitize_tests.sh --cuda-arch sm_80 --all-tools
scripts/checks/cuda_float32_parity.sh --cuda-arch sm_80
```

`check.sh --cuda-arch` enables CUDA. The sanitizer wrapper passes the option to both the build and
the Lake environment used to run the executable. The Float32 parity wrapper compiles its CUDA
comparison directly with `nvcc`; its existing `--arch` spelling remains an alias. All three accept
`--cuda-home` to select the toolkit. Keep the same architecture and toolkit on later invocations,
including sanitizer runs with `--skip-build`. `native` is rejected because it depends on the
builder's visible GPUs; an explicit target also works when the build machine has no GPU.

The site builder runs the documentation post-processors and link checker. The guide source lives
in `home_page/blueprint/`; generated pages go to `home_page/_site/blueprint/`.

Verification producers are grouped by artifact: `lirpa`, `abcrown`, `geometry3d`, `pinn`,
`robustness`, `splines`, and `two_stage`. Their example documentation gives the matching producer
and `verify` commands. `normalization_contract_probe.py` measures PyTorch normalization residuals
for the BugZoo examples.

Keep downloaded data and generated artifacts under `data/`, `_out/`, or a temporary directory.
Use temporary checks for routine refactors and remove them after validation. Retain small tests
for numerical and native-code behavior outside Lean's proofs.
