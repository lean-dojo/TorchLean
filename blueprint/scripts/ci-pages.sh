#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

lake update
lake exe blueprint-gen --output ../_out/blueprint
if [ -d TorchLeanBlueprint/Guide/Assets ]; then
  mkdir -p ../_out/blueprint/html-multi/Guide/Assets
  cp -r TorchLeanBlueprint/Guide/Assets/* ../_out/blueprint/html-multi/Guide/Assets/
fi
