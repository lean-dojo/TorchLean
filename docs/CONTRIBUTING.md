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

The wrapper requires Bash and Python 3. It keeps `cpu` and `cuda-libtorch` builds in
separate local cache directories and holds a checkout lock while Lake runs. Use it consistently
when switching profiles; direct `lake` commands do not acquire that lock. Blueprint builds select
the parent library's CPU cache even after a CUDA build.

For GPU checks, select a CUDA-enabled LibTorch SDK and require a visible device:

```bash
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
scripts/lake.sh -Kcuda=true build nn_tests_suite
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
```

The GPU implementation lives in `csrc/libtorch`. TorchLean owns the tape and calls ATen for forward operations and their VJPs. See
[native build instructions](../scripts/README.md#libtorch-cuda-build) for SDK selection and
compiler requirements. CPU checks do not exercise the GPU backend.

Custom scalar computations use the checked `NN.Kernel` frontend and NVRTC rather than ATen's
operator catalogue. Its [guide](../NN/Kernel/README.md) describes source correspondence, supported
precision, and the separate native execution boundary.

`import NN.API` provides the application API; `import NN` includes specifications, runtime,
verification, and proofs. Neither imports executable examples or tests.

| Target | Contents |
| --- | --- |
| `NN` | Library specifications, runtimes, APIs, proofs, and checkers |
| `NNExamples` | Runnable examples and model commands |
| `NNTests` | Test modules |
| `NNCI` | Maintained modules outside the downstream umbrella |
| `NNSlowProofs` | Proof-heavy IR semantic equivalence |
| `TorchLeanDocs:docs` | Generated API documentation |

For public API or proof changes, also run `scripts/lake.sh build NNCI NNSlowProofs`.
These import targets check maintained library modules; examples and tests have their own targets.
Large proofs may elaborate without intermediate progress output.

Use Lean's types and proofs for mathematical behavior. Run development checks, tactic experiments,
and broad input sweeps in temporary scratch files outside the repository, then remove them after
validation. Keep a permanent test only for a specific numerical, parser, or CUDA/FFI gap not covered
by the proofs or existing tests. Prefer extending an existing focused check over adding another
test module. Useful teaching examples belong in `NN/Examples`, not duplicated in a test catalogue.

## Library and Examples

Application examples use `open TorchLean` and named device, execution and arithmetic choices:
`device := gpu`, `execution := eager`, and `arithmetic := ieee`. These names re-export the existing
constructors; they do not define different settings. For names shared by several APIs, open the
specific namespace locally, such as `open Trainer.Objective (mse)`, or qualify the value.
Do not remove dots from field access, method calls or constructor patterns mechanically.

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
[AI usage](#ai-usage-disclosure).

## Names and Proof Style

- Use UpperCamelCase for types and namespaces, lowerCamelCase for functions, and snake_case for
  theorems. Use `theorem` for theorem declarations.
- Prefer `open TorchLean` and the short `Tensor`/`Storage` names within a module. Qualify names
  where needed to resolve ambiguity. Do not create local aliases for an already-open name.
- Keep one canonical implementation. Public facades may re-export it; do not preserve unused
  compatibility synonyms or duplicate Option/Except versions of the same operation.
- Avoid repeating a namespace in its declaration names. Name options records `Options`.
  Prefer a named `batch` option when the input and result shapes remain clear. Keep meaningful
  spatial dimensions in names, such as `conv1d`. Scalar type names such as `Float32` name the
  representation; arithmetic functions should take or infer the format/type instead of exposing
  separate `32`/`64` suffixed APIs. Avoid arbitrary version suffixes.
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

See [website development](../scripts/README.md#documentation-and-data) for preview commands. Edit source files,
not generated pages under `home_page/docs` or `home_page/_site`.

### Docstring Examples

Keep short examples directly in API docstrings. Check changed snippets in temporary Lean files
outside the repository, then remove the scratch files. Runnable tutorials belong in `NN/Examples`;
do not maintain a second copy under the test tree.

## Before Review

The build treats warnings as errors. Check an individual module with
`scripts/lake.sh env lean -DwarningAsError=true NN/Path/To/File.lean`.

- Relevant build, proof, numerical, and lint checks pass.
- Temporary refactor tests and exploratory code are removed.
- Public documentation matches the final implementation and states remaining assumptions.
- Dataset and third-party provenance is retained.

## AI Usage Disclosure

TorchLean has been developed and formalized primarily by hand over an extended
period of work. The core definitions, architecture, theorem statements, proof
decisions, runtime boundaries, examples, and release choices were made and
reviewed by the maintainers. We used AI assistance only as a limited support
tool for some of the harder proof engineering and debugging work, not as an
oracle and not as a replacement for manual formalization.

This included autograd proofs, runtime approximation, CROWN verification explanations,
and IR lowering correctness, as well as native-runtime debugging and documentation.

### Tools We Used

- OpenAI GPT-5.2 Pro was used selectively as an interactive assistant for a few
  difficult proof and engineering tasks. In particular, it helped with proof
  planning, Lean search, refactoring long proof scripts, debugging stubborn Lean
  goals, and explaining possible ways to organize large correctness arguments.
- GPT-5.6 Pro was also useful while working around CUDA and native runtime
  boundaries: reading error logs, thinking through FFI and memory ownership
  issues, checking documentation language, and helping us separate what Lean
  proves from what the CUDA/cuBLAS runtime must be trusted to do.
- More recently, OpenAI Codex has helped with repository-wide cleanup, API and
  module organization, upgrades to newer Lean releases, and the build, test,
  and documentation checks that accompany those changes. The maintainers
  reviewed the resulting code and decided which changes belonged in TorchLean.
- Harmonic was used in a narrower exploratory role for a small amount of
  definition design and mathematical organization, especially before committing
  some concepts to Lean.
