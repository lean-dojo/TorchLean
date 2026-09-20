#!/usr/bin/env bash
# Run Lake with isolated local build profiles outside the source checkout.
set -euo pipefail

default_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${TORCHLEAN_PACKAGE_ROOT:-$default_repo_root}" && pwd)"
# Resolve elan shims against this package's pinned toolchain, even when invoked elsewhere.
cd "$repo_root"
lake_bin="${TORCHLEAN_LAKE:-lake}"
lean_bin="${TORCHLEAN_LEAN:-lean}"

cuda=false
libtorch=false
cuda_home="/usr/local/cuda"
print_build_dir=false
declare -a forwarded=()
declare -a arguments=("$@")

set_config_flag() {
  local config="$1"
  local key="${config%%=*}"
  local value=""
  if [[ "$config" == *=* ]]; then
    value="${config#*=}"
  fi
  case "$key" in
    cuda)
      case "$value" in
        true|1) cuda=true ;;
        *) cuda=false ;;
      esac
      ;;
    cuda_home)
      # Match Lake's whitespace normalization, including an explicitly empty override.
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      cuda_home="${value:-/usr/local/cuda}"
      ;;
    libtorch)
      case "$value" in
        true|1) libtorch=true ;;
        *) libtorch=false ;;
      esac
      ;;
  esac
}

i=0
while [[ $i -lt ${#arguments[@]} ]]; do
  argument="${arguments[$i]}"
  case "$argument" in
    --torchlean-build-dir)
      print_build_dir=true
      ;;
    --)
      forwarded+=("$argument")
      i=$((i + 1))
      while [[ $i -lt ${#arguments[@]} ]]; do
        forwarded+=("${arguments[$i]}")
        i=$((i + 1))
      done
      break
      ;;
    -K)
      forwarded+=("$argument")
      i=$((i + 1))
      if [[ $i -ge ${#arguments[@]} ]]; then
        echo "error: -K requires a key=value argument" >&2
        exit 2
      fi
      config="${arguments[$i]}"
      forwarded+=("$config")
      set_config_flag "$config"
      ;;
    -K*)
      forwarded+=("$argument")
      config="${argument#-K}"
      config="${config#=}"
      set_config_flag "$config"
      ;;
    *)
      forwarded+=("$argument")
      ;;
  esac
  i=$((i + 1))
done

if [[ -n "${TORCHLEAN_BUILD_PROFILE:-}" ]]; then
  profile="$TORCHLEAN_BUILD_PROFILE"
elif [[ "$cuda" == true && "$libtorch" == true ]]; then
  profile="cuda-libtorch"
elif [[ "$cuda" == true ]]; then
  profile="cuda"
elif [[ "$libtorch" == true ]]; then
  profile="cpu-libtorch"
else
  profile="cpu"
fi

if [[ -n "${TORCHLEAN_BUILD_ROOT:-}" ]]; then
  build_root="$TORCHLEAN_BUILD_ROOT"
elif [[ -d /mnt/build && -w /mnt/build ]]; then
  build_root="/mnt/build/TorchLean"
else
  build_root="${XDG_CACHE_HOME:-$HOME/.cache}/torchlean/build"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: scripts/lake.sh requires python3 for build paths and locking" >&2
  exit 127
fi

build_dir_for() {
  # Resolve missing paths without GNU realpath, including on macOS.
  python3 - "$build_root" "$1" "$2" <<'PY'
import hashlib
import os
import sys

build_root, package_root, profile = sys.argv[1:]
workspace_hash = hashlib.sha256(os.fsencode(package_root)).hexdigest()[:12]
workspace_id = os.path.basename(package_root) + "-" + workspace_hash
print(os.path.realpath(os.path.join(build_root, workspace_id, profile)))
PY
}

build_dir="$(build_dir_for "$repo_root" "$profile")"

if [[ "$print_build_dir" == true ]]; then
  printf '%s\n' "$build_dir"
  exit 0
fi

if [[ ! -f "$repo_root/lakefile.lean" && ! -f "$repo_root/lakefile.toml" ]]; then
  echo "error: Lake package file not found under: $repo_root" >&2
  exit 2
fi
if ! command -v "$lake_bin" >/dev/null 2>&1; then
  echo "error: Lake executable not found: $lake_bin" >&2
  exit 127
fi
if [[ ${#forwarded[@]} -eq 1 && "${forwarded[0]}" == "--version" ]]; then
  exec "$lake_bin" --version
fi
if ! command -v "$lean_bin" >/dev/null 2>&1; then
  echo "error: Lean executable not found: $lean_bin" >&2
  exit 127
fi
if [[ "$cuda" == true ]]; then
  if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
    echo "error: CUDA compiler not found: $cuda_home/bin/nvcc" >&2
    echo "hint: pass the toolkit root with -K cuda_home=/path/to/cuda" >&2
    exit 127
  fi
  export PATH="$cuda_home/bin:$PATH"
  export LD_LIBRARY_PATH="$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$repo_root/.lake" "$default_repo_root/.lake"

# A blueprint build also writes its parent path dependency's artifacts. All packages invoked
# through this wrapper share one checkout lock, held for Lake's lifetime.
exec 9>"$default_repo_root/.lake/profile.lock"
python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX)'

use_build_dir() {
  local package_root="$1"
  local target="$2"
  local build_link="$package_root/.lake/build"
  if [[ -e "$build_link" && ! -L "$build_link" ]]; then
    cat >&2 <<EOF
error: $build_link is a real directory

scripts/lake.sh will not delete build artifacts. Move or remove that directory once, then rerun;
the wrapper will replace it with a symlink into:
  $target
EOF
    exit 1
  fi

  mkdir -p "$target"
  if [[ -L "$build_link" ]]; then
    local current_target
    current_target="$(readlink "$build_link")"
    if [[ "$current_target" != "$target" ]]; then
      rm "$build_link"
      ln -s "$target" "$build_link"
    fi
  else
    ln -s "$target" "$build_link"
  fi
}

use_build_dir "$repo_root" "$build_dir"
if [[ "$repo_root" == "$default_repo_root/home_page/blueprint" ]]; then
  # The blueprint's path dependency explicitly requests cuda=false. Its default buildDir uses
  # this symlink, which may still point at CUDA artifacts from the last root-package invocation.
  use_build_dir "$default_repo_root" "$(build_dir_for "$default_repo_root" cpu)"
fi

lean_prefix="$("$lean_bin" --print-prefix)"
lean_system_lib="$lean_prefix/lib"
lean_library_lib="$lean_prefix/lib/lean"
export TORCHLEAN_BUILD_DIR="$build_dir"
export LD_LIBRARY_PATH="$lean_system_lib:$lean_library_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "$repo_root" == "$default_repo_root" ]]; then
  exec "$lake_bin" -R "-KtorchleanBuildDir=$build_dir" "${forwarded[@]}"
fi
exec "$lake_bin" -R "${forwarded[@]}"
