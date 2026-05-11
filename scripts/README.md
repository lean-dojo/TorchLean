# TorchLean Scripts

This directory contains support commands for local checks, repository hygiene audits,
site generation, dataset preparation, artifact producers, and optional sandboxed Lean checking.

Generated files such as `__pycache__/`, downloaded datasets, built documentation, and local output
artifacts should not live here. Keep those in ignored output directories such as `data/`, `_out/`,
or `home_page/_site/`.

## Release Surface

For a release, keep this directory small, but do not delete scripts only because they are Python.
Everything is organized by purpose:

- `checks/`: local CI, repository lint, dependency audit, CUDA sanitizer/profiling helpers, and
  Lake's Lean lint driver.
- `docs/`: public site, DocGen, and Verso post-processing.
- `datasets/`: dataset download, conversion, and training-log plotting helpers.
- `verification/`: certificate producers and artifact-regeneration workflows.
- `rl/`: optional reinforcement-learning bridge demos.
- `sandbox/`: comparator/untrusted-Lean helper tooling.

### Required Release Support

These are part of the release/check/build path and should stay:

- `checks/check.sh`
- `checks/cuda_sanitize_tests.sh`
- `checks/cuda_profile_tests.sh`
- `checks/check_case_collisions.py`
- `checks/repo_lint.py`
- `checks/TorchLeanLint.lean`
- `checks/dependency_audit.py`
- `docs/build_site.sh`
- `docs/polish_docgen.py`
- `docs/polish_verso_guide.py`
- `sandbox/run_comparator.py`
- `comparator/nn_ci_all.json`

`docs/polish_docgen.py` and `docs/polish_verso_guide.py` are not verification logic. They are the
final website post-processors called by `docs/build_site.sh`: DocGen and Verso generate correct
HTML, then these scripts add the TorchLean landing page, navigation polish, responsive figures, copy
buttons, asset copying, and public-site styling. Delete them only together with a matching
`docs/build_site.sh` and accept rough generated docs.

### Required reproducibility helpers

These are referenced by examples, docs, or artifact-regeneration workflows:

- `datasets/download_example_data.py`
- `datasets/torchlean_data_convert.py`
- `datasets/plot_trainlog.py`
- `verification/regenerate_assets.py`
- `verification/robustness/*`
- `verification/lirpa/*`
- `verification/pinn/*`
- `verification/splines/*`
- `verification/two_stage/*`
- `verification/geometry3d/*`

Use the subfolder paths directly, for example
`python3 scripts/datasets/download_example_data.py --cifar10`.

### Optional examples and research demos

These are not release blockers, but they are documented examples rather than junk:

- `datasets/download_wikitext.py`
- `rl/gymnasium_server.py`
- `rl/export_gymnasium_rollout.py`
- `rl/train_ppo_cartpole_sb3.py`

### Delete on sight

These should never be checked in:

- `__pycache__/`
- `*.pyc`
- local checkpoints, downloaded data, generated plots, generated JSON output
- ad-hoc notebooks or temporary one-off scripts unless they are promoted into one of the buckets
  above

## Plot and Asset Policy

Most plots are generated output and should stay ignored. The release source should track only small,
intentional documentation assets that are referenced by Markdown or Verso pages.

Keep tracked:

- `home_page/assets/media/**` when referenced by homepage Markdown.
- `blueprint/TorchLeanBlueprint/Guide/Assets/**` when referenced by the Verso guide.

Do not track:

- `_external/**`: local model outputs, Geometry3D certificates, overlays, and diagnostic PNGs.
- `_out/**`: generated Verso/blueprint build output.
- `home_page/docs/**`: generated DocGen API HTML.
- `home_page/_site/**`: generated Jekyll output.
- `data/**`: downloaded datasets, training logs, audit plots, and model outputs.
- `Two-Stage_Neural_Controller_Training/**`: local research/training output.

## Checks

- `checks/check.sh`: local verification gate for `lake build`, `lake test`, `lake lint`,
  and optional CUDA / `NN.CI.All` checks.
- `checks/cuda_sanitize_tests.sh`: CUDA sanitizer runner for the CUDA runtime test suite.
- `checks/cuda_profile_tests.sh`: optional Nsight Systems / Nsight Compute wrapper for CUDA
  performance reports.
- `checks/check_case_collisions.py`: CI guard for case-insensitive filesystem name
  collisions.
- `checks/repo_lint.py`: repository lint used by `lake lint`.
- `checks/dependency_audit.py`: lightweight module/import graph audit inspired by
  Li et al., "The Network Structure of Mathlib" (arXiv:2604.24797). It reports
  broad imports, layer-boundary smells, fan-in/fan-out hubs, and Markdown/JSON
  summaries for repository hygiene.
- `checks/TorchLeanLint.lean`: Lean-side lint entry point used by Lake.

Useful commands:

```bash
scripts/checks/check.sh --ci-all
scripts/checks/cuda_profile_tests.sh --both
python3 scripts/checks/repo_lint.py --fail-on-warn
python3 scripts/checks/dependency_audit.py --markdown /tmp/torchlean_dependency_audit.md --fail-on-error
python3 scripts/checks/check_case_collisions.py
```

## Data

- `datasets/download_example_data.py`: downloads the small public datasets used by the
  runnable examples, including CIFAR-10 shards, UCI household-power forecasting
  windows, and tiny text corpora.
- `datasets/download_wikitext.py`: optional WikiText preparation helper for text-model
  experiments. Requires `pyarrow`.
- `datasets/torchlean_data_convert.py`: converts `.npy`, `.npz`, `.mat`, `.pt/.pth`, CSV,
  and image-folder datasets into TorchLean's `.npy` tensor format. Optional formats require the
  corresponding Python package (`scipy`, `torch`, or `pillow`).
  For ImageNet-style diffusion demos, use:

  ```bash
  python3 scripts/datasets/torchlean_data_convert.py image-folder \
    --input /path/to/imagenet/train \
    --x-output data/real/imagenet64/imagenet64_train_X.npy \
    --y-output data/real/imagenet64/imagenet64_train_y.npy \
    --height 64 --width 64 --labels-from-dirs --limit 2000
  ```

## Documentation

- `docs/build_site.sh`: rebuilds the generated API docs, the Verso guide (from the
  `blueprint/` package), dependency graph JSON, and homepage bundle.

`docs/build_site.sh` is the full site build: it rebuilds Lean modules, DocGen, the Verso guide,
the dependency graph JSON, and the Jekyll site. For a quick graph-only refresh, run
`checks/dependency_audit.py` directly.

## Sandboxed Lean Checking

- `sandbox/run_comparator.py`: optional wrapper for `leanprover/comparator`.
- `comparator/nn_ci_all.json`: Comparator config for the `NN.CI.ComparatorAll`
  marker theorem.
