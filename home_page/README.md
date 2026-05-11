# TorchLean Website (Jekyll)

This folder contains the TorchLean landing page and the small amount of glue that assembles the
public site:

- API docs under `/docs/` (built by DocGen4)
- Verso guide (built from the `blueprint/` Lake package) under `/blueprint/`
- Curated runnable demos under `/examples/` (a maintained Jekyll page)

## Local preview

If you have Ruby + Bundler available:

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

- `home_page/docs/` (DocGen4 HTML API docs)
- `home_page/blueprint/` (Verso guide HTML)
- `home_page/graphs/dependency-audit.json` (module/import graph audit for the
  interactive dependency explorer)

CI populates these via `.github/workflows/blueprint.yml`. To reproduce that locally:

### API docs (DocGen4)

```bash
cd ..
rm -rf .lake/build/doc .lake/build/doc-data .lake/build/api-docs.db
DISABLE_EQUATIONS=1 lake build NN:docs
rm -rf home_page/docs
cp -r .lake/build/doc home_page/docs
find home_page/docs -name "*.trace" -delete
find home_page/docs -name "*.hash" -delete
```

Native CUDA/C source notes are documented by the Lean module
`NN.Runtime.Autograd.Engine.Cuda.NativeSources`, so they are generated as part of `/docs/`.

`DISABLE_EQUATIONS=1` is intentional for public site builds. It keeps DocGen from trying to render
equation lemmas for every imported Lean and Mathlib definition, which otherwise produces many
non-fatal timeout warnings. Clear `.lake/build/doc-data` first when switching to this mode because
Lake can otherwise replay stale DocGen artifacts.

### Verso Guide (Blueprint Package)

```bash
cd ../blueprint
lake exe blueprint-gen --output ../_out/blueprint
cd ..
rm -rf home_page/blueprint
mkdir -p home_page/blueprint
cp -r _out/blueprint/html-multi/* home_page/blueprint/
```

### Dependency graph explorer

The `/graphs/` page uses a compact JSON audit generated from Lean imports. It is inspired by
Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797), but keeps
TorchLean's public site lightweight by starting at module/import granularity.

```bash
cd ..
python3 scripts/checks/dependency_audit.py \
  --json home_page/graphs/dependency-audit.json \
  --markdown _out/torchlean_dependency_audit.md \
  --fail-on-error
```

### One command rebuild

From the repo root:

```bash
scripts/docs/build_site.sh
```

## Troubleshooting (Common)

- `bundle` not found: install Bundler for your Ruby, or use `ruby/setup-ruby` if you are in CI.
- Bundler version warning: install the lockfile version with `gem install bundler:2.3.14`, then run
  `bundle _2.3.14_ install`.
- Native extension build failures (e.g. `commonmarker`): install Ruby headers / build tools
  (on Ubuntu this is typically `ruby-dev` + `build-essential`).
