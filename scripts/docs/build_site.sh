#!/usr/bin/env bash
# Build the Lean library, generated documentation, guide, and website.
#
# Usage: scripts/docs/build_site.sh [STEP...]
# The steps run in the order given, and without arguments all of them run in this order:
#   lean     build the four Lean libraries
#   api      render the DocGen API reference into home_page/docs
#   guide    build and check the Verso guide in _out/blueprint/html-multi
#   graphs   write the dependency audit JSON and the import graph page
#   bundle   install the Jekyll bundle
#   timings  refresh the CI timing history and render the runtime section of the performance page
#   website  build the Jekyll site in home_page/_site and check its pages
#   links    check the links and anchors of home_page/_site
# With SKIP_DOCGEN=1, `api` copies the DocGen output of the build directory without rendering it.
# JEKYLL_ENV=production builds and checks the site for its published base URL /TorchLean (the CI
# build). Any other value builds a local preview with home_page/_config_dev.yml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$ROOT/scripts/lake.sh}"
cd "$ROOT"

all_steps=(lean api guide graphs bundle timings website links)
if [ "$#" -eq 0 ]; then
  steps=("${all_steps[@]}")
else
  steps=("$@")
fi
for step in "${steps[@]}"; do
  case " ${all_steps[*]} " in
    *" $step "*) ;;
    *)
      echo "error: unknown step '$step' (steps: ${all_steps[*]})" >&2
      exit 2
      ;;
  esac
done

if [ "${JEKYLL_ENV:-}" = "production" ]; then
  jekyll_config="_config.yml"
  baseurl="/TorchLean"
else
  jekyll_config="_config.yml,_config_dev.yml"
  baseurl=""
fi

step_lean() {
  echo "==> Building Lean modules"
  "$LAKE" build
}

step_api() {
  # Keep generated-document paths tied to this build profile even if another invocation switches
  # the checkout's `.lake/build` symlink between CPU and GPU checks.
  local build_dir
  build_dir="$("$LAKE" --torchlean-build-dir)"

  echo "==> Building DocGen API reference"
  # DocGen can render equations for every imported definition, Lean and Mathlib internals included.
  # The public site has no use for them, so site builds turn equation rendering off and keep
  # declaration types, docstrings, source links, search and module pages.
  #
  # Lake does not include DISABLE_EQUATIONS in the docInfo trace, so remove the cached DocGen DB
  # and data before rebuilding. Otherwise Lake may replay docInfo artifacts rendered with equations.
  if [ "${SKIP_DOCGEN:-0}" = "1" ]; then
    if [ ! -d "$build_dir/doc" ]; then
      echo "error: SKIP_DOCGEN=1 needs DocGen output in $build_dir/doc" >&2
      exit 1
    fi
    echo "    Reusing $build_dir/doc"
  else
    rm -rf "$build_dir/doc" "$build_dir/doc-data" "$build_dir/api-docs.db"
    # The TorchLean libraries have no root modules, so their `:docs` facets render nothing. Build
    # the docInfo of every module of the four libraries, of the examples and of FloatLib, then
    # render every module in the database.
    DISABLE_EQUATIONS=1 "$LAKE" build TorchLean.Tensor:docInfo TorchLean.CUDA:docInfo \
      TorchLean.NN:docInfo TorchLean.Verify:docInfo TorchLeanExamples:docInfo +FloatLib:docInfo
    DISABLE_EQUATIONS=1 "$LAKE" exe doc-gen4 fromDb \
      --build "$build_dir" --manifest "$build_dir/doc-manifest.json" "$build_dir/api-docs.db"
  fi

  echo "==> Copying DocGen output"
  # The public site serves DocGen from `home_page/docs`. Strip trace/hash files right after the
  # copy, so the polisher and the generated preview tree see browser assets and no Lake internals.
  rm -rf home_page/docs
  cp -r "$build_dir/doc" home_page/docs
  find home_page/docs -name "*.trace" -delete
  find home_page/docs -name "*.hash" -delete
  python3 scripts/docs/polish_docgen.py --docs home_page/docs
  du -sh home_page/docs
}

step_guide() {
  echo "==> Building Verso Guide (Blueprint Package)"
  rm -rf _out/blueprint
  TORCHLEAN_PACKAGE_ROOT="$ROOT/home_page/blueprint" \
    "$LAKE" exe vbp build --output ../../_out/blueprint
  TORCHLEAN_PACKAGE_ROOT="$ROOT/home_page/blueprint" \
    "$LAKE" exe vbp check --site ../../_out/blueprint
  test -s _out/blueprint/html-multi/-verso-data/blueprint-manifest.json
  test -s _out/blueprint/html-multi/-verso-data/blueprint-html-cache.json
  # Verso does not copy arbitrary guide assets in every local build mode, so mirror the guide
  # asset directory before polishing the generated HTML.
  if [ -d home_page/blueprint/TorchLeanBlueprint/Guide/Assets ]; then
    mkdir -p _out/blueprint/html-multi/Guide/Assets
    cp -r home_page/blueprint/TorchLeanBlueprint/Guide/Assets/* \
      _out/blueprint/html-multi/Guide/Assets/
  fi
  python3 scripts/docs/polish_verso_guide.py --guide _out/blueprint/html-multi
}

step_graphs() {
  echo "==> Building dependency graph audit"
  # The Graphs page reads this JSON to populate the import explorer.
  python3 scripts/checks/dependency_audit.py \
    --root "$ROOT" \
    --json home_page/graphs/dependency-audit.json \
    --fail-on-error

  echo "==> Building interactive import graph HTML"
  # The import graph is read from the built Lean modules, so it shows the module graph of this
  # checkout.
  mkdir -p home_page/importgraph
  # A library sink is a library module that no library module imports. Every library module is
  # in the import closure of a sink, so the graph of the sinks shows all of Tensor, CUDA, NN and
  # Verify.
  local sinks
  sinks="$(python3 -B scripts/checks/dependency_audit.py --sinks)"
  "$LAKE" exe graph --to "${sinks//$'\n'/,}" home_page/importgraph/index.html
  python3 scripts/docs/polish_docgen.py --importgraph home_page/importgraph/index.html
}

BUNDLE_CMD=()
select_bundler() {
  # Prefer the lockfile Bundler version when installed, but keep local previewing usable on
  # machines that only have a newer default Bundler.
  if [ "${#BUNDLE_CMD[@]}" -gt 0 ]; then
    return
  fi
  if bundle _2.3.14_ --version >/dev/null 2>&1; then
    BUNDLE_CMD=(bundle _2.3.14_)
  else
    echo "note: Bundler 2.3.14 is not installed, so the default Bundler runs." >&2
    echo "      Install it with: gem install bundler:2.3.14" >&2
    BUNDLE_CMD=(bundle)
  fi
}

step_bundle() {
  echo "==> Installing Jekyll bundle"
  select_bundler
  (cd home_page && "${BUNDLE_CMD[@]}" config set path vendor/bundle && "${BUNDLE_CMD[@]}" install)
}

step_timings() {
  echo "==> Refreshing build timing history"
  python3 scripts/docs/ci_timings.py
  python3 scripts/docs/runtime_bench.py
}

step_website() {
  echo "==> Building Jekyll site"
  # The API reference, guide and graphs come from their steps or, in CI, from the cache.
  local generated
  for generated in home_page/docs/index.html \
    _out/blueprint/html-multi/-verso-data/blueprint-manifest.json \
    _out/blueprint/html-multi/-verso-data/blueprint-html-cache.json \
    home_page/importgraph/index.html \
    home_page/graphs/dependency-audit.json; do
    if [ ! -s "$generated" ]; then
      echo "error: generated input $generated is missing or empty" >&2
      exit 1
    fi
  done
  select_bundler
  (cd home_page && rm -rf _site && "${BUNDLE_CMD[@]}" exec jekyll build --config "$jekyll_config")
  mkdir -p home_page/_site/blueprint
  cp -r _out/blueprint/html-multi/. home_page/_site/blueprint/
  local jekyll_content_pages=0
  local page
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
}

step_links() {
  echo "==> Checking generated links and anchors"
  python3 scripts/docs/check_site_links.py home_page/_site \
    --baseurl "$baseurl" --site-url https://lean-dojo.github.io/TorchLean
}

for step in "${steps[@]}"; do
  "step_$step"
done

if [ -z "$baseurl" ] && [[ " ${steps[*]} " == *" website "* ]]; then
  cat <<'HINT'

Site assets and Jekyll output are rebuilt.

Preview with:
  cd home_page
  bundle _2.3.14_ exec jekyll serve --config _config.yml,_config_dev.yml --port 4001

If Bundler warns about its version, install the lockfile version once:
  gem install bundler:2.3.14
  bundle _2.3.14_ config set path vendor/bundle
  bundle _2.3.14_ install
HINT
fi
