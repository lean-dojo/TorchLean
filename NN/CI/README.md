# NN/CI

This directory holds import checks for maintained modules outside the downstream `NN` umbrella.

`NN.CI.All` covers ordinary library modules. `NN.CI.SlowProofs` isolates the end-to-end IR semantic
equivalence proof, which the documentation workflow typechecks before DocGen. Examples and tests
use `NNExamples` and `NNTests`; they do not pass through a second CI import tree.

See:

* `NN/Runtime/Autograd/IRExec/Correctness.lean` (runtime lowering correctness,
  including the semantic equivalence theorem)
* `NN/CI/All.lean` (ordinary modules omitted from the downstream umbrella)
* `NN/CI/SlowProofs.lean` (the explicit proof-heavy target)

If you run one of these locally and it appears to pause, Lean is often elaborating one large module
without intermediate progress output.

## CI Targets

CI targets should answer questions that are broader than a local executable regression check:

- does the curated public API still build after a refactor?
- do slow proof modules still elaborate on the pinned Lean toolchain?
- do generated or bundled verification artifacts still parse under the current checker code?
- do CUDA and non-CUDA builds still expose the same Lean names where the public API expects them?
- do import umbrellas still expose every intended public feature?

This directory is not a second documentation tree and does not wrap examples or tests. It contains
only import targets whose job is to exercise maintained library modules.

## Suggested Local Checks

For ordinary code changes that stay inside Lean definitions, examples, or docs:

```bash
lake build
lake build NNCI NNExamples NNTests
lake test
```

For proof-heavy or public API changes:

```bash
lake build NNCI NNSlowProofs
```

For CUDA changes:

```bash
lake -R -K cuda=true build NN NNCI NNExamples NNTests
lake -R -K cuda=true exe nn_tests_suite
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

The CUDA sanitizer run is expensive, but it is the right evidence for memory and synchronization
hazards at the native boundary. A Lean proof about the spec does not replace that native check.

For public command changes, run the affected command with a small input after compiling
the examples. For website changes, rebuild the site:

```bash
lake build NNExamples
cd home_page
bundle _2.3.14_ exec jekyll build --config _config.yml,_config_dev.yml
```

For docs that mention verification tools, compare the command names with:

```bash
lake exe verify --help
lake exe torchlean --help
```

CI should keep these evidence types distinct. A theorem target proves a mathematical statement. A
verifier command checks a concrete artifact against its schema and predicate. A runtime regression
exercises the code path users run. A sanitizer run checks native CUDA behavior around the Lean FFI
boundary. A site build proves the public pages can be regenerated from source.
