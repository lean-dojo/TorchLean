# Contributing to TorchLean

Use the pinned Lean 4.34.0 toolchain and mathlib dependency, and keep changes focused. FloatLib tracks
`main` in `lakefile.lean`; `lake-manifest.json` locks the revision used by a checkout. Update it with
`scripts/lake.sh update floatlib`, rebuild and test, and commit the updated manifest with any fixes.
Paths below are relative to the repository root.

The guide has its own manifest. After updating FloatLib, refresh its inherited dependencies with
`TORCHLEAN_PACKAGE_ROOT="$PWD/home_page/blueprint" scripts/lake.sh update TorchLean` and check the
guide build against the same revision.

## Build and Check

```bash
scripts/lake.sh build
scripts/lake.sh test
scripts/lake.sh lint
```

The wrapper requires Bash and Python 3. It keeps CPU, CUDA, and optional LibTorch builds in
separate local cache directories and holds a checkout lock while Lake runs. Use it consistently
when switching profiles; direct `lake` commands do not acquire that lock. Blueprint builds select
the parent library's CPU cache even after a CUDA build.

Set `TORCHLEAN_BUILD_ROOT` to choose the cache location, or use `--torchlean-build-dir` to print
the selected path. `TORCHLEAN_BUILD_PROFILE` overrides the profile name; use distinct names for
different backend configurations. If `.lake/build` is an existing directory, move it aside once
before using the wrapper. The wrapper preserves it and reports the destination it needs.

`import NN.API` provides the application API; `import NN` includes specifications, runtime,
verification, and proofs. Neither imports executable examples or tests.

| Target | Contents |
| --- | --- |
| `NN` | Library specifications, runtimes, APIs, proofs, and checkers |
| `NNExamples` | Runnable examples and model commands |
| `NNTests` | Test modules and compiled documentation snippets |
| `NNCI` | Maintained modules outside the downstream umbrella |
| `NNSlowProofs` | Proof-heavy IR semantic equivalence |
| `TorchLeanDocs:docs` | Generated API documentation |

Use Lean's types and proofs for mathematical behavior. Run development checks, tactic experiments,
and broad input sweeps in temporary scratch files outside the repository, then remove them after
validation. Keep a permanent test only for a specific numerical, parser, or CUDA/FFI gap not covered
by the proofs or existing tests. Prefer extending an existing focused check over adding another
test module. Useful teaching examples belong in `NN/Examples`, not duplicated in a test catalogue.

## Library and Examples

Reusable validation, execution, data processing, and training belong in the library. Examples
construct inputs and models, call library operations, and explain results. Local formatting is
fine; duplicating a runtime or implementing a model-specific replacement for a general operation
is not. Reusable modules must not import examples, tests, CI aggregates, or executable wrappers.
Keep global `main` declarations in executable modules.

Run examples and artifact checkers through their shared commands:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe verify -- list
```

Keep fixtures small. Document external data requirements and record provenance in
[third-party notices](THIRD_PARTY_NOTICES.md).

## Module Boundaries

Import the smallest module that provides what you use. The Torch runtime separates its
`Functional.Ops` interface, `Session` internals, and `Trainer.Types` contract from backend
construction. The parent modules re-export these parts for callers that need the whole API.
Keep `Core.Types` free of typed-graph and operation imports. Eager operation files import the
CPU and CUDA primitives they dispatch to; graph modules import their own recorder and proofs.

Use `public import` for dependencies in the public interface and ordinary `import` for
implementation details. Keep runtime bodies hidden with `public section`. Add `@[expose]` where
downstream elaboration or proofs need to reduce a definition, as with the curried argument types
and operation instances. A runtime factory does not need its implementation exposed.
Some polymorphic factories still need public backend imports for Lean's compiler specialization;
check compiled consumers before making those imports private.

After changing imports, build the affected modules and ask Shake what they actually use:

```bash
scripts/lake.sh build NN.Runtime.Autograd.Torch.Core.Trainer
scripts/lake.sh shake --explain --keep-public NN.Runtime.Autograd.Torch.Core.Trainer
```

Review suggestions before applying them. A facade's public re-exports are part of its API, even
when the facade has no declarations of its own. Rebuild consumers after narrowing imports; they
may have relied on a dependency arriving indirectly.

## Adding an Operator

1. Define its mathematical meaning under `NN/Spec/Core` or `NN/Spec/Layers`.
2. Add forward/VJP contracts under `NN/Spec/Autograd` if differentiation is supported.
3. For graph primitives, update `NN.IR.OpKind`, denotation, and shape inference/checking.
4. Add runtime support and any required verification transfers.
5. Check the relevant declarations and execution behavior. State unsupported paths explicitly.

A runtime-only or verifier-only operation must identify that boundary in its documentation.
Theorems, conditional contracts, checker acceptance, and native execution carry different
assumptions; see [trust boundaries](TRUST_BOUNDARIES.md). AI assistance is disclosed in
[AI usage](AI_USAGE.md).

## Names and Proof Style

- Use UpperCamelCase for types and namespaces, lowerCamelCase for functions, and snake_case for
  theorems. Use `theorem` for theorem declarations.
- Prefer `open TorchLean` and the short `Tensor`/`Storage` names within a module. Qualify names
  where needed to resolve ambiguity. Do not create local aliases for an already-open name.
- Keep one canonical implementation. Public facades may re-export it; do not preserve unused
  compatibility synonyms or duplicate Option/Except versions of the same operation.
- Avoid repeating a namespace in its declaration names. Name options records `Options`.
  Use `batch` as the prefix for a batched counterpart.
- Lowercase application namespaces such as `nn`, `optim`, and `text` follow the public API.
  Definition-specific auxiliary namespaces use their definition's spelling; other helpers belong
  under `Internal`. Keep top-level API entrypoints as focused import modules.
- Use `Internal.create` for sealed-structure construction. Exposed definitions need accessible
  helpers; use private helpers only when their callers can also hide their bodies.
- Keep imports narrow, lines at most 100 characters, and comments about mathematical intent or
  non-obvious invariants. Do not add unresolved `sorry`, `native_decide`, custom axioms, or
  proof-level heartbeat overrides. The custom-axiom allowlist is empty.
- Preserve theorem statements. Split expensive proofs into reusable lemmas rather than weakening
  hypotheses or raising resource limits.

Run `python3 scripts/checks/repo_lint.py --fail-on-warn` for source and API checks.

Reusable proof automation lives under `NN/Tactic`; see the [tactic guide](../NN/Tactic/README.md).
Extend `autograd` with a proved rule when adding a differentiable operation, rather than adding
another expression representation or a model-specific proof solver.
Tensor expression elaborators stay under `NN/Tensor/Internal/Elab`, model-building macros under
`NN/API/Macros`, and widget commands under `NN/Widgets`. Those construct programs or displays;
they are not proof tactics. Keep single-file proof helpers local rather than exporting them just
to place every macro in one directory.

## Documentation

Every Lean module needs a module docstring; public declarations need useful documentation or
`@[inherit_doc ...]`. Describe the operation, shape conventions, failure conditions, and relevant
trust assumptions. Update nearby user documentation when behavior changes; avoid repeating a
complete API catalogue in several files.

Guide source lives in `home_page/blueprint/`. Build the website with:

```bash
scripts/docs/build_site.sh
```

See [website development](../home_page/README.md) for preview commands. Edit source files,
not generated pages under `home_page/docs` or `home_page/_site`.

### Docstring Examples

API docstrings with an `Example:` block use compiler-checked snippets under
`NN/Tests/API/DocExamples/`. The linter synchronizes their text with the docstrings.

Adding one is three steps:

1. Write the snippet in the mirror module for its area (`Neural.lean`, `Trainer.lean`, `Data.lean`,
   `Tensor.lean`, `Text.lean`), inside a namespace of its own, preceded by a marker naming the
   declaration it documents:

   ```lean
   -- doc-example: NN/API/Seeded.lean :: def dropout
   namespace Dropout

   -- Active in `.train` mode and the identity in `.eval` mode, which the trainer selects for you.
   def model : nn.Builder (nn.Sequential [64] [64]) :=
     nn.dropout 0.1

   end Dropout
   ```

   The part after `::` is a prefix of the declaration line, and it has to name exactly one
   declaration in that file; the linter says so when it does not.

2. Run `python3 scripts/checks/repo_lint.py --sync-doc-examples`. That copies the namespace body
   into the docstring verbatim as a fenced `lean` block under an `Example:` heading, expanding a
   one-line docstring into the multi-line form if it has to.

3. Build. `lake build NNTests` compiles the snippet, and `lake lint` compares the docstring against
   it, so the two cannot drift apart afterwards.

Snippets use `--` line comments, never nested docstrings: a `-/` inside the snippet would close the
docstring it is being pasted into. Nothing in the mirror modules runs, and that is deliberate.
Elaboration is the property worth checking, and a snippet that has to run needs fixtures, a device,
and a seed, which is what the example programs under `NN/Examples/` are for.

The build sets `warningAsError`, so any warning fails `lake build`. Check a single file the same
way the build will:

```bash
lake env lean -DwarningAsError=true NN/Path/To/File.lean
```

## Before Review

- Relevant build, proof, numerical, and lint checks pass.
- Temporary refactor tests and exploratory code are removed.
- Public documentation matches the final implementation and states remaining assumptions.
- Dataset and third-party provenance is retained.
