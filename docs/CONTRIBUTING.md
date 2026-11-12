# Contributing to TorchLean

Use the pinned Lean 4.35.0 toolchain and Mathlib, and keep each change focused. Paths are relative
to the repository root. FloatLib tracks `main` in `lakefile.lean`, and `lake-manifest.json` locks
the revision. To update it, run `scripts/lake.sh update floatlib`, rebuild and test, then commit
the manifest with any fixes.

The guide has its own manifest. After updating FloatLib, refresh its inherited dependencies with
`TORCHLEAN_PACKAGE_ROOT="$PWD/home_page/blueprint" scripts/lake.sh update TorchLean` and check the
guide build against the same revision.

## Build and Check

```bash
scripts/lake.sh build
scripts/lake.sh test
scripts/lake.sh lint
```

The only GitHub Actions workflow publishes the documentation and doesn't run the runtime tests, so
run the CPU and CUDA checks yourself before merging a runtime change.

`scripts/lake.sh` needs Bash and Python 3. It keeps separate `cpu` and `cuda-libtorch` caches for
each pinned toolchain and holds a checkout lock that plain `lake` doesn't take, so use the wrapper
for every build when you switch profiles. Blueprint builds use the parent library's CPU cache, even
after a CUDA build.

For GPU checks, select a CUDA-enabled LibTorch SDK of torch 2.13 or later and require a visible
device:

```bash
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
scripts/lake.sh -Kcuda=true build nn_tests_suite
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
```

The GPU implementation lives in `csrc/libtorch`. TorchLean owns the tape and calls ATen for each
forward operation and its VJP. The
[native build instructions](../scripts/README.md#libtorch-cuda-build) cover SDK selection and
compiler requirements. The CPU checks don't exercise the GPU backend.

Custom scalar computations go through the checked `Kernel` frontend and NVRTC instead of ATen's
operator catalogue. Its [guide](../TorchLean/CUDA/Kernel/README.md) covers source correspondence,
supported precision and the separate native execution boundary.

Each library holds the modules of one directory:

| Library | Contents |
| --- | --- |
| `TorchLean.Tensor` | Tensors, shapes, scalar formats, operator specifications, and proofs |
| `TorchLean.CUDA` | LibTorch buffers and kernels, and the CUDA kernel frontend |
| `TorchLean.NN` | Autograd, runtimes, the application API, the operator IR, interop, and proofs |
| `TorchLean.Verify` | CROWN and IBP, certificates, roundoff proofs, and FP32 contracts |
| `TorchLeanExamples` | Runnable examples, model commands, and worked proofs |
| `TorchLeanTest` | Test modules |

`scripts/lake.sh build` builds the four libraries. After a change to the public API or to proofs,
also run `scripts/lake.sh build TorchLeanExamples TorchLeanTest`. Large proofs may elaborate for a
while without printing progress.

Express mathematical behavior with Lean's types and proofs. Run development checks, tactic
experiments and broad input sweeps in scratch files outside the repository, and delete them
afterwards. Keep a permanent test only for a specific numerical, parser, or CUDA/FFI gap not covered
by the proofs or the other tests. Prefer extending a focused check over adding another test module.
Teaching examples belong in `TorchLeanExamples` and aren't copied into the test tree.

## Library and Examples

Application examples use `open TorchLean` and name their device and arithmetic with the
constructors of `Backend.Device` and `Arithmetic`, as in `device := .cuda` and
`arithmetic := .ieee`, never by an alias. For a name shared by several APIs, open its namespace
locally, as in `open Trainer.Objective (mse)`, or qualify it. Don't strip the dots from field
access, method calls or constructor patterns mechanically.

Reusable validation, execution, data processing and training belong in the library. An example
builds its inputs and model, calls library operations and explains the results. It may format its
output but must not duplicate a runtime or swap a general operation for a model-specific one.
Reusable modules must not import examples, tests, CI aggregates or executable wrappers, and global
`main` declarations stay in executable modules.

Run examples and artifact checkers through their shared commands:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe verify -- list
```

Keep fixtures small. Document external data requirements and record provenance in
[third-party notices](THIRD_PARTY_NOTICES.md).

## Module Boundaries

Import the module that defines a declaration you use, not a parent module that imports it. The
Torch runtime keeps its `Ops` interface (`TorchLean.NN.Ops`), `Session` internals and trainer
contract (`TorchLean.NN.ScalarTrainer`) apart from backend construction
(`TorchLean.NN.ScalarTrainer.Construction`). Keep `TorchLean.NN.Session.Config` free of
typed-graph and operation imports. Eager operation files import the CPU and CUDA primitives they
dispatch to, and graph modules import their own recorder and proofs.

Use `public import` for dependencies of the public interface and plain `import` for implementation
details. Keep runtime bodies hidden with `public section`, and add `@[expose]` where downstream
elaboration or proofs need to reduce a definition, as for the curried argument types and operation
instances. Runtime factories don't need exposed implementations, but some polymorphic ones need
public backend imports for Lean's compiler specialization, so check the compiled consumers before
making those imports private.

After changing imports, build the affected modules and ask Shake what they actually use:

```bash
scripts/lake.sh build TorchLean.NN.ScalarTrainer.Construction
scripts/lake.sh shake --explain --keep-public TorchLean.NN.ScalarTrainer.Construction
```

Review Shake's suggestions before applying them. After narrowing imports, rebuild the consumers,
which may have relied on a dependency arriving indirectly. Module structure follows four rules:

- No umbrella or import-only modules: a file imports the modules that define what it uses.
- No alias files, modules that only `export` names from others.
- One definition per concept, without compatibility synonyms or duplicate `Option`/`Except`
  versions of an operation.
- Layer order `TorchLean.Tensor` < `TorchLean.CUDA` < `TorchLean.NN` < `TorchLean.Verify`: a
  module imports only modules of its own layer or of lower ones. `TorchLeanExamples` ranks above
  the four layers, and `TorchLeanTest` may import any library.

The layering gate in `scripts/checks/dependency_audit.py` checks the first, second, and fourth
rules. It also checks Lake's glob ownership, where `@[extern]` declarations live and their
agreement with the C sources, and the imports of the library check `scripts/checks/Library.lean`,
which must reach every module of the libraries but the root modules of the executables.
`scripts/lake.sh lint` builds the libraries and then runs the `torchlean_lint` executable, which
runs `scripts/checks/repo_lint.py`, the gate, the LibTorch ABI check
`scripts/checks/libtorch_abi.py` and the library check. The ABI check fails unless the LibTorch
bridge's sources call exactly the ATen operators of
[the trust boundaries](TRUST_BOUNDARIES.md#libtorch-stable-abi) and only C shims listed there. The
library check fails unless every theorem of the libraries depends on at most `propext`,
`Classical.choice` and `Quot.sound`.
Its coverage check fails unless every public operator specification and tape constructor under
`TorchLean/` is registered with `#roundoff_coverage`, every operator of the table with a rule has
its equation lemma and, when one of its ports holds the scalar, a route analyzed for the configured
scalar, and every `op_cert` certificate speaks about a route of its operator. It also fails when
the outcomes of the analyses differ from `scripts/checks/roundoff_baseline.jsonl`. Run the gate
alone with:

```bash
python3 scripts/checks/dependency_audit.py --fail-on-error
```

## Adding an Operator

Every frontend runs operators through the table `TorchLean.Op`
(`TorchLean/Tensor/Op/Table.lean`, its rules in `TorchLean/Tensor/Op/Table/Rule.lean`), so an
operator is added there once:

1. Define its mathematical meaning and pullback as tensor operations under
   `TorchLean/Tensor` or `TorchLean/Tensor/Layers`.
2. Add an `Op` constructor with its attributes, indexed by its input and output ports
   (`TorchLean.Port`: value, frozen, detached, index, Boolean or integer), and a docstring stating
   what it computes and its pullback.
3. Give it an arm of `Op.rule?` with the forward map and the dense pullback, plus a prepared
   pullback, contribution order, alias rule or value-domain check where these differ from the
   defaults. Restate the arm as the equation lemma `Op.rule?_<op>` (for the two forms of `cast`,
   `Op.rule?_cast_bool` and `Op.rule?_cast_index`) in the module of its operator family under
   `TorchLean/Tensor/Op/Table/`. Attributes that the type does not check go in `Op.check`.
4. In `TorchLean/Tensor/Op/Need.lean`, give the operator an arm of `Op.need` if its rule reads an
   optional capability (`Rounding`, `Erf`, `InverseTrig`, `Cbrt`, `NextAfter` or
   `Bits`). `Op.needsCapability` is computed from `Op.need`. In `TorchLean/Tensor/Op/Invoke.lean`,
   list the operator in `Op.checksValues` if it checks its input values.
   `Op.rule?_capabilities` and `Op.validate_eq_ok` there, and `Node.eval_ports` in
   `TorchLean/NN/IR/Eval.lean`, go by cases on every constructor, and an arm that splits on an
   attribute or reads a capability needs its own case.
5. Add its `Op.encode` and `Op.decode` arms in `TorchLean/Tensor/Op/Codec.lean`, with a
   `decode_encode_<op>` lemma and its case in `Op.decode_encode`. A change that makes an existing
   record name another operator or other attributes, or fail to decode, must raise
   `Op.Wire.formatVersion`.
6. If PyTorch has a function for it, add its row to the PyTorch exporter
   (`TorchLean/NN/PyTorch/Export/IR.lean`), which refuses an operator without one.
7. If LibTorch has a kernel for it, add its CUDA entry to `LibTorch.Kernel.ofOp`
   (`TorchLean/NN/LibTorch/Kernel.lean`) with a contract in
   `TorchLean/NN/Backend/LibTorch.lean` and a test against the CPU tape (see the [native
   guide](../csrc/libtorch/README.md)). Add its reference contract to
   `Backend.Reference.contract?` so CPU planning offers a capsule.
8. Register its routes with `#roundoff_coverage` in the matching module of
   `TorchLean/Verify/Coverage/`.
9. Model code reaches the operator through a method of the `Ops` interface
   (`TorchLean/NN/Ops.lean`). Its eager instance records it
   with `Session.apply`, and its recorder instance with `IR.record`.
10. Check the arm's values (`TorchLeanTest/OpTable.lean` holds hand-computed checks) and any
    certificate registered with `@[op_cert]`. Document each path that doesn't support it.

The CPU tape needs nothing further: `Tape.apply` records every operator through
its arm.

A runtime-only or verifier-only operation must say so in its documentation. Theorems, conditional
contracts, checker acceptance and native execution carry different assumptions, set out in
[trust boundaries](TRUST_BOUNDARIES.md). AI assistance is disclosed in
[AI usage](#ai-usage-disclosure).

## Names and Proof Style

- Use UpperCamelCase for types and namespaces, lowerCamelCase for functions, and snake_case for
  theorems, declared with `theorem`.
- Within a module, prefer `open TorchLean` and the short `Tensor`/`Storage` names, and qualify a
  name where it would be ambiguous. Don't create local aliases for an already-open name.
- Don't repeat a namespace in its declaration names. Name options records `Options`. Prefer a
  named `batch` option when the input and result shapes stay clear, and keep spatial distinctions
  that change an operation's meaning. A scalar type name such as `Float32` names a representation,
  so arithmetic functions take or infer the format or type instead of offering `32`- and
  `64`-suffixed variants. Avoid arbitrary version suffixes.
- `TorchLean.nn`, `TorchLean.optim` and `TorchLean.autograd` stay lowercase, as the public API
  spells them. Every other namespace names a concept in UpperCamelCase, never a folder, file, role
  or stage such as `Internal`, `Core`, `Spec` or `Proofs`. A definition's helpers live in its own
  namespace.
- Construct sealed structures with `create`. Exposed definitions need accessible helpers, so use
  private helpers only when their callers can hide their bodies too.
- Keep imports narrow, lines at most 100 characters, and comments about mathematical intent or
  non-obvious invariants. Don't add `sorry`, `native_decide`, custom axioms or proof-level
  heartbeat overrides. The custom-axiom allowlist (`ALLOWED_AXIOMS` in
  `scripts/checks/repo_lint.py`) is empty.
- Preserve theorem statements. Split expensive proofs into reusable lemmas instead of weakening
  hypotheses or raising resource limits.

The [tensor tactic guide](../TorchLean/Tensor/Tactic/README.md) describes the reusable proof
automation under `TorchLean/Tensor/Tactic`, and the [derivative tactic
guide](../TorchLean/NN/Tactic/README.md) covers `autograd` and `converges` under
`TorchLean/NN/Tactic`. When you add a differentiable operation, extend `autograd` with a proved rule
instead of adding another expression representation or a model-specific proof solver. Tensor
expression elaborators stay under `TorchLean/Tensor/Einops/Elab` and in
`TorchLean/Tensor/Literal.lean`, model-building macros in `TorchLean/NN/Sequential/Literal.lean`
and widget commands under `TorchLean/NN/Widget`: they build programs or displays, not proof
tactics. Keep single-file proof helpers local instead of exporting them just to put every macro in
one directory.

## Documentation

Every Lean module needs a module docstring, and every public declaration needs documentation or
`@[inherit_doc ...]` covering the operation, shape conventions, failure conditions and relevant
trust assumptions. When behavior changes, update the user documentation next to it, and keep any
complete API catalogue in one file.

Guide source lives in `home_page/blueprint/`. Build the website with:

```bash
scripts/docs/build_site.sh
```

[Website development](../scripts/README.md#documentation-and-data) has the preview commands. Edit
the source files and leave the generated pages under `home_page/docs` and `home_page/_site` alone.

### Docstring Examples

Keep short examples in the API docstrings themselves, and check a changed snippet in a scratch
Lean file outside the repository.

## Before Review

The build treats warnings as errors. Check an individual module with
`scripts/lake.sh env lean -DwarningAsError=true TorchLean/NN/Path/To/File.lean`.

- Relevant build, proof, numerical, and lint checks pass.
- Temporary refactor tests and exploratory code are removed.
- Public documentation matches the final implementation and states remaining assumptions.
- Dataset and third-party provenance is recorded in `docs/THIRD_PARTY_NOTICES.md`.

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
- OpenAI Codex has helped with repository-wide cleanup, API and module
  organization, Lean toolchain upgrades, and the build, test, and
  documentation checks that accompany those changes. The maintainers
  reviewed the resulting code and decided which changes belonged in TorchLean.
- Harmonic was used in a narrower exploratory role for a small amount of
  definition design and mathematical organization, especially before committing
  some concepts to Lean.
