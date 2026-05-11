# TorchLean Verso Book

This directory contains the Lean-native Verso book for TorchLean. The rendered `/blueprint/` page is
the public guide used by the website.

The public site serves the rendered output under `/blueprint/`; the source of truth is Lean/Verso:

- `TorchLeanBlueprint/Guide.lean` assembles the public book.
- `TorchLeanBlueprint/Guide/**/*.lean` contains the narrative guide chapters.
- `TorchLeanBlueprintMain.lean` is the HTML generator executable.

The book is organized as a guide rather than a raw module dump. The generated API docs and import
graph provide declaration-level lookup; this guide explains the main concepts, workflows, examples,
and verification boundaries.

Local build:

```bash
cd blueprint
lake update
lake exe blueprint-gen --output ../_out/blueprint
mkdir -p ../_out/blueprint/html-multi/Guide/Assets
cp -r TorchLeanBlueprint/Guide/Assets/* ../_out/blueprint/html-multi/Guide/Assets/
cd ..
python3 scripts/docs/polish_verso_guide.py --guide _out/blueprint/html-multi
```

The generated site lives in `../_out/blueprint/html-multi/`. The repository site build copies that
directory to `home_page/blueprint/`.
