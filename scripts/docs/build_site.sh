#!/usr/bin/env bash
# Build the Lean library, generated documentation, guide, and website.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$ROOT/scripts/lake.sh}"
cd "$ROOT"

echo "==> Building Lean modules"
"$LAKE" build

echo "==> Building DocGen API reference"
# DocGen can try to render equations for every imported definition, including Lean and Mathlib
# internals. That is noisy and not useful for the public site, so site builds disable equation
# rendering while keeping declaration types, docstrings, source links, search, and module pages.
#
# Lake does not include DISABLE_EQUATIONS in the docInfo trace, so remove the cached DocGen DB/data
# before rebuilding. Otherwise Lake may replay old noisy docInfo artifacts.
if [ "${SKIP_DOCGEN:-0}" = "1" ]; then
  if [ ! -d .lake/build/doc ]; then
    echo "error: SKIP_DOCGEN=1 requires an existing .lake/build/doc directory" >&2
    exit 1
  fi
  echo "    Reusing .lake/build/doc"
else
  rm -rf .lake/build/doc .lake/build/doc-data .lake/build/api-docs.db
  DISABLE_EQUATIONS=1 "$LAKE" build TorchLeanDocs:docs
fi

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
rm -rf _out/blueprint
TORCHLEAN_PACKAGE_ROOT="$ROOT/home_page/blueprint" \
  "$LAKE" exe vbp build --output ../../_out/blueprint
TORCHLEAN_PACKAGE_ROOT="$ROOT/home_page/blueprint" \
  "$LAKE" exe vbp check --site ../../_out/blueprint
test -s _out/blueprint/html-multi/-verso-data/blueprint-manifest.json
test -s _out/blueprint/html-multi/-verso-data/blueprint-html-cache.json
# Verso does not automatically copy arbitrary guide assets in every local build
# mode, so mirror the guide asset directory before polishing the generated HTML.
if [ -d home_page/blueprint/TorchLeanBlueprint/Guide/Assets ]; then
  mkdir -p _out/blueprint/html-multi/Guide/Assets
  cp -r home_page/blueprint/TorchLeanBlueprint/Guide/Assets/* _out/blueprint/html-multi/Guide/Assets/
fi
python3 scripts/docs/polish_verso_guide.py --guide _out/blueprint/html-multi
python3 scripts/docs/check_verso_layout.py --guide _out/blueprint/html-multi

echo "==> Building dependency graph audit"
# The Graphs page reads this JSON to populate the import explorer.
python3 scripts/checks/dependency_audit.py \
  --root "$ROOT" \
  --json home_page/graphs/dependency-audit.json \
  --fail-on-error

echo "==> Building interactive import graph HTML"
# The import graph is generated from Lean imports after the library build, so it
# reflects the same module graph users get from the current checkout.
mkdir -p home_page/importgraph
"$LAKE" exe graph --to NN home_page/importgraph/index.html
python3 scripts/docs/postprocess_importgraph.py home_page/importgraph/index.html

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
(cd home_page && rm -rf _site && "${BUNDLE_CMD[@]}" exec jekyll build --config _config.yml,_config_dev.yml)
mkdir -p home_page/_site/blueprint
cp -r _out/blueprint/html-multi/. home_page/_site/blueprint/
jekyll_content_pages=0
while IFS= read -r -d '' page; do
  if grep -Fq "<main" "$page"; then
    jekyll_content_pages=$((jekyll_content_pages + 1))
    grep -Fq "mathjax@3/es5/tex-chtml.js" "$page" || {
      echo "error: MathJax is not loaded by $page" >&2
      exit 1
    }
    grep -Fq "packages: {'[-]': ['noundefined']}" "$page" || {
      echo "error: MathJax does not report unsupported TeX commands in $page" >&2
      exit 1
    }
  fi
done < <(
  find home_page/_site -type f -name "*.html" \
    ! -path "home_page/_site/blueprint/*" \
    ! -path "home_page/_site/docs/*" \
    ! -path "home_page/_site/importgraph/*" \
    -print0
)
if [ "$jekyll_content_pages" -eq 0 ]; then
  echo "error: no Jekyll content pages were generated" >&2
  exit 1
fi
python3 - <<'PY'
import re
from pathlib import Path

site = Path("home_page/_site")
excluded = ("blueprint/", "docs/", "importgraph/")
ignored_html = re.compile(
    r"<(script|style|pre|code)\b[^>]*>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
inline_math = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", flags=re.DOTALL)

for page in site.rglob("*.html"):
    relative = page.relative_to(site).as_posix()
    if relative.startswith(excluded):
        continue
    html = ignored_html.sub("", page.read_text(encoding="utf-8"))
    for formula in inline_math.finditer(html):
        if "<" in formula.group(1):
            raise SystemExit(
                f"error: Jekyll inserted HTML inside inline math in {page}; "
                "escape Markdown-sensitive characters in the source formula"
            )
PY

echo "==> Checking generated links and anchors"
python3 scripts/docs/check_site_links.py home_page/_site \
  --site-url https://lean-dojo.github.io/TorchLean

cat <<'EOF'

Site assets and Jekyll output are rebuilt.

Preview with:
  cd home_page
  bundle _2.3.14_ exec jekyll serve --config _config.yml,_config_dev.yml --port 4001

If Bundler warns about its version, install the lockfile version once:
  gem install bundler:2.3.14
  bundle _2.3.14_ config set path vendor/bundle
  bundle _2.3.14_ install
EOF
