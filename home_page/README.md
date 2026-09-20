# TorchLean Website (Jekyll)

This folder contains the TorchLean landing page and the small amount of glue that assembles the
public site:

- API reference under `/docs/` (built by DocGen4)
- Verso Blueprint guide under `/blueprint/`, including the curated declaration and proof map
- Curated runnable examples under `/examples/` (a maintained Jekyll page)
- Dependency and import graph pages under `/graphs/` and `/importgraph/`
- Companion projects and their documentation under `/tools/`
- CI timing history under `/performance/`
- Status/update notes under `/updates/`

Each generated page has an owning source. Edit that source, then rebuild the generated output.

The Tools directory is maintained in `home_page/tools/index.md`. Its links should point to the
documentation owned by each companion repository rather than copying those manuals into this site.

The rest of the source map is:

| Public page | Source to edit |
| --- | --- |
| Landing page | `home_page/index.md`, site CSS/JS/assets |
| Getting started | `home_page/start/index.md` |
| Examples pages | `home_page/examples/**/index.md` plus matching `NN/Examples/**` README/source |
| Guide | `home_page/blueprint/TorchLeanBlueprint/Guide/**/*.lean` |
| Formalization map | `home_page/blueprint/TorchLeanBlueprint/FormalizationMap/*.lean` |
| API reference | `NN/**/*.lean` module docstrings and declaration docstrings |
| Graph pages | `home_page/graphs/index.md`, `scripts/checks/dependency_audit.py`, generated graph JSON |
| Performance page | `home_page/performance/index.md`, site CSS |
| CUDA page | `home_page/cuda/index.md` plus the guide's CUDA/trust-boundary chapters |
| Trust/provenance claims | `docs/TRUST_BOUNDARIES.md`, `docs/THIRD_PARTY_NOTICES.md`, relevant checker README |

Avoid editing generated HTML by hand. Regenerate `home_page/docs`, `home_page/_site/blueprint`,
`home_page/importgraph`, and `home_page/_site` from their sources.

## Local preview

Use Ruby 3.2 or newer (below 4.0), as required by `Gemfile`, and Bundler 2.3.14:

```bash
cd home_page
bundle config set path vendor/bundle
bundle _2.3.14_ install
bundle _2.3.14_ exec jekyll serve --config _config.yml,_config_dev.yml
```

Then open `http://127.0.0.1:4000/`.

If port `4000` is already in use, run:

```bash
bundle _2.3.14_ exec jekyll serve --config _config.yml,_config_dev.yml --port 4001
```

## Building Generated Assets

The public website expects a few generated directories under `home_page/`:

- `home_page/docs/` (DocGen4 HTML API reference)
- `home_page/_site/blueprint/` (Verso guide HTML, copied after Jekyll builds)
- the generated `dependency-audit.json` data used by the interactive graph explorer

CI populates these via `.github/workflows/blueprint.yml`. Run the commands below from the
repository root.

### API Reference (DocGen4)

```bash
rm -rf .lake/build/doc .lake/build/doc-data .lake/build/api-docs.db
DISABLE_EQUATIONS=1 scripts/lake.sh build TorchLeanDocs:docs
rm -rf home_page/docs
cp -r .lake/build/doc home_page/docs
find home_page/docs -name "*.trace" -delete
find home_page/docs -name "*.hash" -delete
python3 scripts/docs/polish_docgen.py --docs home_page/docs
```

Native CUDA/C source notes are documented by the Lean module
`NN.Runtime.Autograd.Engine.Cuda.Trusted`, so they are generated as part of `/docs/`.

`scripts/docs/polish_docgen.py` keeps the generated docs focused on TorchLean's `NN` modules. It
removes local copies of Lean, Std, Mathlib, and other dependency pages, then rewrites dependency
links to the upstream generated documentation so declaration links do not become local 404s.

`DISABLE_EQUATIONS=1` is intentional for public site builds. It keeps DocGen from trying to render
equation lemmas for every imported Lean and Mathlib definition, which otherwise produces many
non-fatal timeout warnings. Clear `.lake/build/doc-data` first when switching to this mode because
Lake can otherwise replay old DocGen artifacts.

Lean documentation comments use `$...$` for inline math and `$$...$$` for display math; DocGen
loads MathJax on every declaration page. Jekyll pages load MathJax from the shared layout. Verso
pages use their bundled KaTeX runtime: prefix a Verso code literal with `$` for inline math or `$$`
for display math. Ordinary backticks are for code and stay monospace.

### Verso Guide (Blueprint Package)

The Lean package lives in `home_page/blueprint/`, which Jekyll excludes. Run the following
after building Jekyll; the final copy installs the guide at `/blueprint/`.

```bash
TORCHLEAN_PACKAGE_ROOT="$PWD/home_page/blueprint" \
  scripts/lake.sh exe vbp build --output ../../_out/blueprint
TORCHLEAN_PACKAGE_ROOT="$PWD/home_page/blueprint" \
  scripts/lake.sh exe vbp check --site ../../_out/blueprint
if [ -d home_page/blueprint/TorchLeanBlueprint/Guide/Assets ]; then
  mkdir -p _out/blueprint/html-multi/Guide/Assets
  cp -r home_page/blueprint/TorchLeanBlueprint/Guide/Assets/* _out/blueprint/html-multi/Guide/Assets/
fi
python3 scripts/docs/polish_verso_guide.py --guide _out/blueprint/html-multi
rm -rf home_page/_site/blueprint
mkdir -p home_page/_site/blueprint
cp -r _out/blueprint/html-multi/. home_page/_site/blueprint/
```

The guide’s Formalization Map follows selected definitions and theorems across subsystems. The
`/graphs/` page answers a different question: it audits source-module imports. The runtime graph IR
is the third graph in the repository, representing a neural-network computation rather than Lean
source or proof dependencies.

### Dependency graph explorer

The `/graphs/` page reads a JSON snapshot generated from the current Lean imports.

```bash
python3 scripts/checks/dependency_audit.py \
  --json home_page/graphs/dependency-audit.json \
  --fail-on-error
```

### One command rebuild

From the repo root:

```bash
scripts/docs/build_site.sh
```

That script rebuilds Lean modules, DocGen, the Verso guide, dependency graph artifacts, the import
graph viewer, installs the Jekyll bundle, and writes `home_page/_site`.

For a lighter pass after editing only Jekyll Markdown:

```bash
cd home_page
bundle _2.3.14_ exec jekyll build --config _config.yml,_config_dev.yml
```

For a lighter pass after editing only the Verso guide, run from the repository root:

```bash
TORCHLEAN_PACKAGE_ROOT="$PWD/home_page/blueprint" \
  scripts/lake.sh exe vbp build --output ../../_out/blueprint
TORCHLEAN_PACKAGE_ROOT="$PWD/home_page/blueprint" \
  scripts/lake.sh exe vbp check --site ../../_out/blueprint
python3 scripts/docs/polish_verso_guide.py --guide _out/blueprint/html-multi
rm -rf home_page/_site/blueprint
mkdir -p home_page/_site/blueprint
cp -r _out/blueprint/html-multi/. home_page/_site/blueprint/
```

## Site Review Checklist

Before publishing a docs-heavy change, check:

- the page explains what object is being discussed: model, graph, artifact, checker, or theorem;
- runtime examples name their data path, command, and output artifacts;
- verification pages name the predicate Lean recomputes and the producer boundary that remains;
- CUDA/LibTorch/PyTorch text does not imply external backward or native kernels are proved unless a
  theorem says so;
- generated pages were rebuilt from source rather than edited directly;
- `bundle _2.3.14_ exec jekyll build --config _config.yml,_config_dev.yml` succeeds.

## Troubleshooting (Common)

- `bundle` not found: install Bundler for your Ruby, or use `ruby/setup-ruby` if you are in CI.
- Bundler version warning: install the lockfile version with `gem install bundler:2.3.14`, then run
  `bundle _2.3.14_ install`.
- Native extension build failures (e.g. `commonmarker`): install Ruby headers / build tools
  (on Ubuntu this is typically `ruby-dev` + `build-essential`).
