#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "==> Building Lean modules"
lake build

echo "==> Building DocGen API docs"
# DocGen can try to render equations for every imported definition, including Lean and Mathlib
# internals. That is noisy and not useful for the public site, so site builds disable equation
# rendering while keeping declaration types, docstrings, source links, search, and module pages.
#
# Lake does not include DISABLE_EQUATIONS in the docInfo trace, so remove the cached DocGen DB/data
# before rebuilding. Otherwise Lake may replay stale noisy docInfo artifacts.
rm -rf .lake/build/doc .lake/build/doc-data .lake/build/api-docs.db
DISABLE_EQUATIONS=1 lake build NN:docs

echo "==> Copying DocGen output"
# The public site serves DocGen from `home_page/docs`. Strip trace/hash files so
# the checked-in preview tree contains browser assets rather than Lake internals.
rm -rf home_page/docs
cp -r .lake/build/doc home_page/docs
find home_page/docs -name "*.trace" -delete
find home_page/docs -name "*.hash" -delete
python3 scripts/docs/polish_docgen.py --docs home_page/docs

rm -rf home_page/manual

echo "==> Building Verso Guide (Blueprint Package)"
(cd blueprint && lake exe blueprint-gen --output ../_out/blueprint)
# Verso does not automatically copy arbitrary guide assets in every local build
# mode, so mirror the guide asset directory before polishing the generated HTML.
if [ -d blueprint/TorchLeanBlueprint/Guide/Assets ]; then
  mkdir -p _out/blueprint/html-multi/Guide/Assets
  cp -r blueprint/TorchLeanBlueprint/Guide/Assets/* _out/blueprint/html-multi/Guide/Assets/
fi
python3 scripts/docs/polish_verso_guide.py --guide _out/blueprint/html-multi
rm -rf home_page/blueprint
mkdir -p home_page/blueprint
cp -r _out/blueprint/html-multi/* home_page/blueprint/

echo "==> Building dependency graph audit"
# The homepage graph page consumes this JSON directly; the Markdown file is a
# readable companion artifact for debugging site builds.
python3 scripts/checks/dependency_audit.py \
  --root "$ROOT" \
  --json home_page/graphs/dependency-audit.json \
  --markdown home_page/graphs/dependency-audit.md \
  --fail-on-error

echo "==> Building interactive import graph HTML"
# The import graph is generated from Lean imports after the library build, so it
# reflects the same module graph users get from the current checkout.
mkdir -p home_page/importgraph
lake exe graph --to NN.Library home_page/importgraph/index.html
python3 - <<'PY'
from pathlib import Path

page = Path("home_page/importgraph/index.html")
html = page.read_text(encoding="utf-8")
html = html.replace('  <link rel="stylesheet" href="style.css" />\n', "")
html = html.replace(
    'var docs_url = params.get("docs_url") || "https://leanprover-community.github.io/mathlib4_docs/";',
    'var docs_url = params.get("docs_url") || new URL("../docs/", window.location.href).href;',
)
page.write_text(html, encoding="utf-8")
PY

echo "==> Installing Jekyll bundle"
# Prefer the lockfile Bundler version when installed, but keep local previewing
# usable on machines that only have a newer default Bundler.
if bundle _2.3.14_ --version >/dev/null 2>&1; then
  BUNDLE_CMD=(bundle _2.3.14_)
else
  echo "note: Bundler 2.3.14 is not installed; using the default Bundler." >&2
  echo "      Install it with: gem install bundler:2.3.14" >&2
  BUNDLE_CMD=(bundle)
fi
(cd home_page && "${BUNDLE_CMD[@]}" config set path vendor/bundle && "${BUNDLE_CMD[@]}" install)

echo "==> Building Jekyll site"
(cd home_page && "${BUNDLE_CMD[@]}" exec jekyll build --config _config.yml,_config_dev.yml)

cat <<'EOF'

Site assets and Jekyll output are rebuilt.

Preview with:
  cd home_page
  bundle _2.3.14_ exec jekyll serve --config _config.yml,_config_dev.yml

If Bundler warns about its version, install the lockfile version once:
  gem install bundler:2.3.14
  bundle _2.3.14_ config set path vendor/bundle
  bundle _2.3.14_ install
EOF
