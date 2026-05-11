# Third-Party Notices

TorchLean is MIT-licensed; see `LICENSE`. This file records external datasets, tokenizer assets,
generated fixtures, optional third-party code, and release hygiene notes.

The short version: the repository builds without downloading real datasets. Large or external data
lives under the ignored `data/` directory and should not be included in source releases unless a
separate data release audits the licenses and citations.

## Local Data Policy

The root `data/` directory is intentionally ignored by git. It is a local
workspace for downloaded public datasets, Hugging Face caches, tokenizer files,
training logs, plots, and other generated artifacts. `lake build` and CI do not
download real datasets.

> The commands below recreate local caches for examples. They are not part of the default build and
> should not be committed.

To recreate the commonly used real-data cache from a clean checkout:

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

These assets are not redistributed by TorchLean unless explicitly noted. Users
download them into `data/` when they want to run examples on real data.

| Asset | Source | Local path | Preparation command | License / provenance note |
| --- | --- | --- | --- | --- |
| CIFAR-10 | `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` | `data/real/cifar10/*.npy`, archive cache under `data/real/raw/` | `python3 scripts/datasets/download_example_data.py --cifar10` | CIFAR-10 is the Krizhevsky/Nair/Hinton image dataset. The downloader checks MD5 `c58f30108f718f92721af3b95e74349a`. Users should follow the terms published with the Toronto CIFAR-10 distribution. |
| Tiny Shakespeare | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` | `data/real/text/tiny_shakespeare.txt` | `python3 scripts/datasets/download_example_data.py --tiny-shakespeare` | Public text corpus popularized by the `char-rnn` examples. Keep attribution to the upstream source when redistributing derived artifacts. |
| TinyStories validation split | `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt` | `data/real/text/tinystories_valid.txt` | `python3 scripts/datasets/download_example_data.py --tinystories-valid` | TinyStories is the Eldan/Li synthetic stories corpus on Hugging Face. Check the dataset card for current license and citation requirements before redistributing. |
| WikiText | Hugging Face dataset `Salesforce/wikitext` via the Dataset Viewer API | `data/real/text/wikitext2_train.txt`, optional cache under `data/real/hf_cache/wikitext/` | `python3 scripts/datasets/download_wikitext.py --config wikitext-2-raw-v1 --split train --output data/real/text/wikitext2_train.txt` | The script records the upstream license note: WikiText is CC BY-SA 3.0 / GFDL; see the Hugging Face dataset card for details. |
| GPT-2 tokenizer files | Hugging Face repository `openai-community/gpt2` | `data/real/gpt2/vocab.json`, `data/real/gpt2/merges.txt` | Use the `curl` commands in the Local Data Policy section. | Used by `NN.API.Text.Bpe` and `torchlean text_gpt2` for standard GPT-2 byte-pair encoding. Check the Hugging Face model card for current terms. |
| 1D Burgers FNO dataset | `https://huggingface.co/datasets/kks32/sciml-dataset/resolve/main/fno/burgers_data_R10.mat` | `data/real/fno/burgers_*.npy`, `data/real/fno/burgers_meta.json` | `python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --out-dir data/real/fno` | Public SciML/FNO Burgers operator-learning data. The script converts the `.mat` file to native NPY arrays used by `torchlean fno1d_burgers`; check the dataset card before redistributing. |
| Viscous Burgers PINN reference data | `https://github.com/AdrianDario10/Burgers_Equation1D` (`burgers_shock.mat`) | User-selected path, commonly `/tmp/burgers_dataset.json` after conversion | `git clone --depth 1 https://github.com/AdrianDario10/Burgers_Equation1D.git /tmp/Burgers_Equation1D` then `python3 scripts/verification/pinn/import_burgers_shock_mat.py --mat /tmp/Burgers_Equation1D/burgers_shock.mat --out /tmp/burgers_dataset.json` | Classic Raissi et al. viscous Burgers PINN reference dataset as distributed by the external repository. Treat it as opt-in external data and preserve the upstream repository's license/citation information. |

For larger language-model experiments, `scripts/datasets/download_wikitext.py` can also
export bounded WikiText-103 text:

```bash
python3 scripts/datasets/download_wikitext.py \
  --config wikitext-103-raw-v1 \
  --split train \
  --max-bytes 120000000 \
  --output data/real/text/wikitext103_train_120mb.txt
```

## Bundled Demo Fixtures

The repository includes small generated fixtures so tutorials and tests can run
without network access:

| Fixture | Path | Origin / notice |
| --- | --- | --- |
| Toy regression CSV/NPY | `NN/Examples/Data/toy_regression.csv`, `NN/Examples/Data/toy_regression_X.npy`, `NN/Examples/Data/toy_regression_y.npy` | Synthetic TorchLean tutorial data, covered by the repository MIT license. |
| Toy CIFAR-10-like NPY | `NN/Examples/Data/toy_cifar10like_X.npy`, `NN/Examples/Data/toy_cifar10like_y.npy` | Synthetic TorchLean fixture generated by `NN/Examples/Data/gen_toy_cifar10like.py`, covered by the repository MIT license. |
| Robustness digits JSON fixtures | `NN/Examples/Verification/Robustness/digits_test.json`, `NN/Examples/Verification/Robustness/digits_linear_weights.json`, `NN/Examples/Verification/Robustness/digits_linear_margin_cert.json` | Derived demo artifacts produced from scikit-learn's built-in `load_digits` dataset and local training/certification scripts. scikit-learn is BSD-3-Clause licensed; keep scikit-learn and original digits-dataset attribution when reusing these fixtures outside TorchLean. |

## Generated Local Outputs

The following files are generated by examples and are not third-party datasets:

| Output kind | Typical path | Notes |
| --- | --- | --- |
| RL widget logs and policies | `data/rl/*.json` | Produced by `torchlean ppo_gridworld`, `torchlean ppo_cartpole`, and related widget demos. |
| FNO prediction CSV/PNG files | `data/real/fno*/predictions*.csv`, `data/real/fno*/predictions*.png` | Produced by `torchlean fno1d_burgers` and `NN/Examples/Data/plot_fno1d_burgers.py`. |
| Hugging Face or raw archive caches | `data/real/hf_cache/`, `data/real/raw/` | Local caches created by dataset-preparation scripts. Do not commit them. |

## Optional Third-Party Code

- `Two-Stage_Neural_Controller_Training/`
  - This path is gitignored in TorchLean and is not part of the public library
    release unless a separate release bundle explicitly includes it.
  - If you check out the upstream project locally, see
    `Two-Stage_Neural_Controller_Training/LICENSE`.
  - This subtree also vendors the `alphabetaCROWN` code and data; see:
    - `Two-Stage_Neural_Controller_Training/src/alphabetaCROWN/LICENSE`
    - `Two-Stage_Neural_Controller_Training/src/alphabetaCROWN/README.md`

## Release Checklist

Before publishing a release or dataset bundle:

1. Do not include the root `data/` directory unless the release is
   explicitly a data release and every included asset has been audited.
2. Preserve upstream license files and citation metadata for any redistributed
   external asset.
3. Prefer shipping scripts and checksums over downloaded archives.
4. Keep generated TorchLean logs, plots, and caches out of source releases.

If you believe a third-party component is missing attribution or licensing
information, please open an issue or pull request.
