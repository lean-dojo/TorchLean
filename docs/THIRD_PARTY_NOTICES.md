# Third-Party Notices

TorchLean is MIT-licensed; see [LICENSE](../LICENSE). This file records external datasets, tokenizer assets,
bundled generated artifacts, optional third-party code, and the main external projects TorchLean
builds on. Paths and commands below are relative to the repository root.

The repository builds from source without downloading real datasets. When an example needs public
data, the helper scripts below place it under the ignored `data/` directory.

## Lean Ecosystem and Build Dependencies

TorchLean source files are authored for this repository. Many of those files import Mathlib, Std,
Batteries, or other Lean packages. Those imports are ordinary library use, analogous to relying on
NumPy or SciPy in a Python scientific-computing project.

The main Lean dependencies are pinned in `lake-manifest.json` and `home_page/blueprint/lake-manifest.json`.
After `lake update`, their license files are available under `.lake/packages/` and
`home_page/blueprint/.lake/packages/`.

| Component | Role in TorchLean | Notice |
| --- | --- | --- |
| Lean 4, Lake, and Std | Compiler, package manager, and standard library, selected by `lean-toolchain`. | Source project: `leanprover/lean4`. |
| mimalloc 3.4.4 | Private allocator object for TorchLean's Linux native executables and shared libraries, with a narrow global purge-wakeup repair. The installed Lean compiler allocator is unchanged. | Source project: `microsoft/mimalloc`, MIT licensed. `scripts/lean_allocator.py` pins and checks the source archive and SDK header; the archive includes `LICENSE`. |
| Mathlib | Formal mathematics used by specifications, proofs, probability, real analysis, tensors, and optimization files. | Source project: `leanprover-community/mathlib4`, pinned to the Lean 4.34 line here. Mathlib is Apache-2.0 licensed. |
| FloatLib | Configurable executable scalars, reference semantics, generic rounding theory, and arithmetic refinement proofs. | Source project: `lean-dojo/FloatLib`, tracking `main` and locked to `c9a051f08f948ce473036b815b0669fafb61f601` in `lake-manifest.json`. MIT licensed; its license and third-party notices are distributed with the dependency. |
| Batteries, Aesop, Qq, ProofWidgets, LeanSearchClient, importGraph, plausible, Cli, leansqlite, MD4Lean, BibtexQuery, UnicodeBasic | Lean ecosystem packages pulled directly or transitively for proofs, tactics, widgets, documentation, dependency analysis, and command-line tooling. | Their license files are fetched with the Lake package cache. |
| doc-gen4 | Generates the API documentation built with `lake build NN:docs`. | Source project: `leanprover/doc-gen4`, Apache-2.0 licensed. |
| Verso, VersoBlueprint, SubVerso, Illuminate | Build the TorchLean guide/blueprint documentation. | Upstream Lean documentation tooling. |
| Lean4Checker | Optional kernel-level checking infrastructure used by verification workflows. | Upstream Lean project pinned in the Lake manifests. |
| CUDA toolkit, cuBLAS, cuFFT | Optional external NVIDIA libraries used when building with `lake build -K cuda=true`. | Users provide their own CUDA installation. |
| PyTorch, ATen, libtorch, ONNX tooling | Optional runtime/import/export providers for checkpoints, graph capture, IR exchange, and fast forward kernels. | PyTorch/ATen/libtorch and ONNX remain external systems; TorchLean checks imported JSON/IR artifacts and documents runtime trust boundaries separately. |
| Jekyll and Ruby gems | Website build tooling for `home_page/`. | Used to build the public site. |

## Local Data Policy

The root `data/` directory is a local workspace for downloaded datasets, Hugging Face caches,
tokenizer files, training logs, plots, and other generated artifacts. `lake build` and CI do not
download real datasets. These commands populate the common example cache:

```bash
python3 -m pip install numpy scipy pyarrow

# CIFAR-10, Tiny Shakespeare, and TinyStories validation text.
python3 scripts/datasets/download_example_data.py --all

# WikiText-2 raw train split.
python3 scripts/datasets/download_wikitext.py \
  --config wikitext-2-raw-v1 \
  --split train \
  --output data/real/text/wikitext2_train.txt

# 1D Burgers FNO operator-learning arrays.
python3 NN/Examples/Data/prepare_fno1d_burgers.py \
  --download \
  --out-dir data/real/fno

# GPT-2 tokenizer files for BPE text examples.
mkdir -p data/real/gpt2
curl -L https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json \
  -o data/real/gpt2/vocab.json
curl -L https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt \
  -o data/real/gpt2/merges.txt
```

By default, `scripts/datasets/download_example_data.py --cifar10` exports a small
interactive CIFAR-10 subset: 200 train images and 100 test images. To export the
full 50,000/10,000 split, use:

```bash
python3 scripts/datasets/download_example_data.py \
  --cifar10 \
  --cifar10-limit-train -1 \
  --cifar10-limit-test -1
```

## Downloaded Dataset and Asset Sources

These are the public data sources used by the runnable examples when a user chooses to populate
`data/`.

| Asset | Source | Local path | Preparation command | License / provenance note |
| --- | --- | --- | --- | --- |
| CIFAR-10 | `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` | `data/real/cifar10/*.npy`, archive cache under `data/real/raw/` | `python3 scripts/datasets/download_example_data.py --cifar10` | CIFAR-10 is the Krizhevsky/Nair/Hinton image dataset. The downloader checks MD5 `c58f30108f718f92721af3b95e74349a`. |
| Tiny Shakespeare | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` | `data/real/text/tiny_shakespeare.txt` | `python3 scripts/datasets/download_example_data.py --tiny-shakespeare` | Public text corpus popularized by the `char-rnn` examples. |
| TinyStories validation split | `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt` | `data/real/text/tinystories_valid.txt` | `python3 scripts/datasets/download_example_data.py --tinystories-valid` | TinyStories is the Eldan/Li synthetic stories corpus on Hugging Face. |
| WikiText | Hugging Face dataset `Salesforce/wikitext` via the Dataset Viewer API | `data/real/text/wikitext2_train.txt`, optional cache under `data/real/hf_cache/wikitext/` | `python3 scripts/datasets/download_wikitext.py --config wikitext-2-raw-v1 --split train --output data/real/text/wikitext2_train.txt` | The script records the upstream license note: WikiText is CC BY-SA 3.0 / GFDL; see the Hugging Face dataset card for details. |
| GPT-2 tokenizer files | Hugging Face repository `openai-community/gpt2` | `data/real/gpt2/vocab.json`, `data/real/gpt2/merges.txt` | Use the `curl` commands in the Local Data Policy section. | Used by `NN.API.Text.Bpe` and `torchlean text_gpt2` for standard GPT-2 byte-pair encoding. |
| 1D Burgers FNO dataset | `https://huggingface.co/datasets/kks32/sciml-dataset/resolve/main/fno/burgers_data_R10.mat` | `data/real/fno/burgers_*.npy`, `data/real/fno/burgers_meta.json` | `python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --out-dir data/real/fno` | Public SciML/FNO Burgers operator-learning data. The script converts the `.mat` file to native NPY arrays used by `torchlean fno1d_burgers`. |
| Viscous Burgers PINN reference data | `https://github.com/AdrianDario10/Burgers_Equation1D` (`burgers_shock.mat`) | User-selected path, commonly `/tmp/burgers_dataset.json` after conversion | `git clone --depth 1 https://github.com/AdrianDario10/Burgers_Equation1D.git /tmp/Burgers_Equation1D` then `python3 scripts/verification/pinn/import_burgers_shock_mat.py --mat /tmp/Burgers_Equation1D/burgers_shock.mat --out /tmp/burgers_dataset.json` | Classic Raissi et al. viscous Burgers PINN reference dataset as distributed by the external repository. |

For larger language-model experiments, `scripts/datasets/download_wikitext.py` can also
export bounded WikiText-103 text:

```bash
python3 scripts/datasets/download_wikitext.py \
  --config wikitext-103-raw-v1 \
  --split train \
  --max-bytes 120000000 \
  --output data/real/text/wikitext103_train_120mb.txt
```

## Bundled Example Artifacts

The repository includes generated artifacts so examples and tests can run without network access:

| Artifact | Path | Origin / notice |
| --- | --- | --- |
| Small regression CSV/NPY | `NN/Examples/Data/small_regression.csv`, `NN/Examples/Data/small_regression_X.npy`, `NN/Examples/Data/small_regression_y.npy` | Synthetic TorchLean example data generated by `NN/Examples/Data/generate_small_data.py`, covered by the repository MIT license. |
| Small CIFAR-10-like NPY | `NN/Examples/Data/small_cifar10like_X.npy`, `NN/Examples/Data/small_cifar10like_y.npy` | Synthetic TorchLean image-shaped data generated by `NN/Examples/Data/generate_small_data.py`, covered by the repository MIT license. |
| Robustness digits JSON artifacts | `NN/Examples/Verification/Robustness/digits_test.json`, `NN/Examples/Verification/Robustness/digits_linear_weights.json`, `NN/Examples/Verification/Robustness/digits_linear_margin_cert.json` | Derived artifacts produced from scikit-learn's built-in `load_digits` dataset and local training/certification scripts. scikit-learn is BSD-3-Clause licensed. |
| LiRPA/CROWN fixture JSON | `NN/Examples/Verification/LiRPA/*.json` | Small deterministic artifacts produced by scripts under `scripts/verification/lirpa/`; Lean checkers consume the represented bounds. |
| Alpha-beta-CROWN-style leaf artifacts | `NN/Examples/Verification/AbCrown/*.json` | Compact raw/converted terminal-leaf artifacts for the TorchLean structural checker. The external verifier/search remains a producer boundary. |
| ODE/PINN/spline certificate fixtures | `NN/Examples/Verification/{ODE,PINN,Splines}/` | Small curated artifacts produced by local scripts or hand-written fixtures so verification commands run offline. Larger trained checkpoints stay outside git. |

Bundled JSON fixtures should stay small enough to review. If an artifact depends on a large model,
dataset, solver run, or external training process, keep the large output under `data/`, `_external/`,
`/tmp`, or another ignored directory and document the regeneration command instead.

## Generated Local Outputs

The following files are generated by examples and are not third-party datasets:

| Output kind | Typical path | Notes |
| --- | --- | --- |
| RL widget logs and policies | `data/rl/*.json` | Produced by `torchlean ppo_gridworld`, `torchlean ppo_cartpole`, and related widget commands. |
| FNO prediction CSV/PNG files | `data/real/fno*/predictions*.csv`, `data/real/fno*/predictions*.png` | Produced by `torchlean fno1d_burgers` and `NN/Examples/Data/plot_fno1d_burgers.py`. |
| Data conversion manifests | beside converted `.npy` files under `data/real/**` | Produced by `scripts/datasets/torchlean_data_convert.py --manifest`; record source path/key/shape metadata for later inspection. |
| Verification producer output | `_external/**`, `/tmp/**`, or user-selected output paths | Produced by scripts under `scripts/verification/**`; Lean checkers should name the artifact schema they consume. |
| Hugging Face or raw archive caches | `data/real/hf_cache/`, `data/real/raw/` | Local caches created by dataset-preparation scripts. |

## Optional Third-Party Code

- `Two-Stage_Neural_Controller_Training/`
  - This path is gitignored in TorchLean; it is a local checkout for the two-stage controller
    experiments rather than part of the library tree.
  - If you check out the external project locally, see
    `Two-Stage_Neural_Controller_Training/LICENSE`.
  - That external checkout may itself vendor the `alphabetaCROWN` code and data; see:
    - `Two-Stage_Neural_Controller_Training/src/alphabetaCROWN/LICENSE`
    - `Two-Stage_Neural_Controller_Training/src/alphabetaCROWN/README.md`

Questions or corrections about provenance are welcome as issues or pull requests.
