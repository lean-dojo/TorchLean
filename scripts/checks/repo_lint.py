#!/usr/bin/env python3
"""
TorchLean repo lints (project-specific).

This linter stays dependency-free so it can run in CI and locally.

Checks are split into:
  - errors: must be fixed (fail CI)
  - warnings: reported for visibility (do not fail by default)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LINT_SCOPE_SENTINEL = REPO_ROOT / "NN/MLTheory/CROWN/Lyapunov/Oracle.lean"

# External trees that may exist in a developer checkout but are not part of TorchLean's core sources
# and must not affect repo policy/CI. These are user-cloned repos outside TorchLean's source tree.
#
# Important: `Path.rglob` walks the filesystem, not git-tracked files, so this linter explicitly skips
# these directories to ensure `lake lint` does not start depending on optional checkouts.
VENDORED_DIR_NAMES = {
    "Two-Stage_Neural_Controller_Training",  # optional external checkout (α,β-CROWN workflows)
    "PINN_verification",  # user-cloned external repo (gitignored)
}

# Keep the trusted boundary explicit: axioms must be quarantined, named, and documented.
ALLOWED_AXIOMS = {
    "NN/MLTheory/CROWN/Lyapunov/Oracle.lean": {"crown_oracle"},
    "NN/Runtime/Autograd/Engine/Cuda/Trusted.lean": {"instNonemptyBuffer"},
}

# Narrow allowlist for linter suppressions that are noisy in facade files but do
# not weaken proofs. Keep this list small and review each addition.
ALLOWED_LINTER_SUPPRESSION_FILES = {
    "NN/Tensor/API.lean",
}

PUBLIC_EXAMPLE_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Models/",
    "NN/Examples/Data/",
)

PUBLIC_TUTORIAL_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Data/",
)

PUBLIC_GUIDE_PREFIXES = (
    "README.md",
    "NN/Examples/README.md",
    "NN/Examples/Quickstart/README.md",
    "NN/Examples/Models/README.md",
    "blueprint/TorchLeanBlueprint/Guide/Ch1_Introduction/",
    "blueprint/TorchLeanBlueprint/Guide/Ch2_Frontend/",
    "blueprint/TorchLeanBlueprint/Guide/Ch5_Advanced/",
)

PUBLIC_GUIDE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bTrainer\.NewConfig\b|\bNewConfig\b"),
        "public guides should use `Trainer.Config`; `Trainer.NewConfig` was removed during the unified Trainer cleanup.",
    ),
    (
        re.compile(r"\bTrainer\.(regression|classifier|crossEntropy|custom)\b"),
        "public guides should use `Trainer.new ... { task := ... }`; specialized `Trainer.*` constructors are removed.",
    ),
    (
        re.compile(r"\btrainer\.fit\b"),
        "public guides should call `trainer.train`; do not reintroduce the old `fit` public API.",
    ),
    (
        re.compile(r"\bIO\.println\s+(report|fit)\.summary\b"),
        "public guides should use `trained.printSummary` / `report.printSummary` instead of printing `.summary` directly.",
    ),
    (
        re.compile(r"\blet\s+report\s+←\s+trainer\.train\b"),
        "public guides should call the trained handle `trained`; `trainer.train` returns a reusable trained object, not just a report.",
    ),
    (
        re.compile(r"\bRuntimeFit\b|\bparseRuntimeFit\b|\bparsed\.fit\b"),
        "quickstart docs should use `RuntimeTrain`, `parseRuntimeTrain`, and `parsed.trainOptions`.",
    ),
    (
        re.compile(r"\bfitOptionsWhenLogRequested\b|\bfitOptions\b"),
        "public guides should use `trainOptions` terminology, not old `fitOptions` spellings.",
    ),
    (
        re.compile(r"\bTrainer\.FitOptions\b"),
        "`Trainer.FitOptions` was removed; public guides should name `Trainer.TrainOptions`.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\b"),
        "`Trainer.FitSummary` was removed; public guides should use `Trainer.TrainSummary`.",
    ),
    (
        re.compile(
            r"\bfitCsvRegression\b|\brunCsvRegressionTrain\b|\bfitNpyRegression\b|"
            r"\brunCifar(Classifier|Regression|Curve)Train\b|"
            r"\brun(RegressionCsv|ClassificationNpy|RegressionNpy|ForecastWindow)\b"
        ),
        "public guides should show `Trainer.new` / `trainer.train` or `Trainer.Command` runners, not removed command-wrapper names.",
    ),
    (
        re.compile(r"\bModelZoo\.Command\b|\bTrainCommand\.run\b"),
        "public guides should use `Trainer.Command` for runnable model-zoo command glue.",
    ),
    (
        re.compile(r"\btrain\.\*"),
        "public guides should teach the `Trainer` facade, not `train.*`.",
    ),
    (
        re.compile(r"\btrain\.stepEpochLR\b"),
        "public guides should say `Trainer.stepEpochLR`, not `train.stepEpochLR`.",
    ),
    (
        re.compile(r"\btrain\.Advanced\b"),
        "public guides should say `Trainer.Advanced` for escape-hatch code, not `train.Advanced`.",
    ),
    (
        re.compile(r"\bNN\.API\.train\.Advanced\b"),
        "public guides should not teach the internal `NN.API.train.Advanced` namespace; explain it as an advanced runtime escape hatch.",
    ),
]

PUBLIC_EXAMPLE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bsample\.Supervised\b"),
        "public examples should use `SupervisedSample`, not the internal `sample.Supervised` spelling.",
    ),
    (
        re.compile(r"_root_\.NN\.API\.sample\.Supervised\b"),
        "public examples should use `SupervisedSample`, not the fully-qualified internal sample type.",
    ),
    (
        re.compile(r"\bSample\.Supervised\b"),
        "public examples should use `SupervisedSample` as the sample type; reserve `Sample.*` for constructors/accessors.",
    ),
    (
        re.compile(r"\bsample\.mk\b"),
        "public examples should use `Sample.mk`, not the internal `sample.mk` spelling.",
    ),
    (
        re.compile(r"\bnn\.sequential!\b"),
        "public examples should use `nn.Sequential!`, not the lowercase macro spelling.",
    ),
    (
        re.compile(r"\bnn\.summary\b"),
        "public examples should print `model.info`; do not introduce a second model-summary spelling.",
    ),
    (
        re.compile(r"\bShape\.(Vec|Mat|Image|Images|NCHW)\b"),
        "public examples should use the canonical lowercase shape helpers: `Shape.vec`, `Shape.mat`, `Shape.image`, `Shape.images`, `Shape.nchw`.",
    ),
    (
        re.compile(r"\bSemantics\.Scalar\b"),
        "public examples should use `Runtime.SemanticScalar`, not the lower internal `Semantics.Scalar` spelling.",
    ),
    (
        re.compile(r"\bTaskRunner\b"),
        "public examples should not expose `TaskRunner`; use `Trainer`/`Module` helpers instead.",
    ),
    (
        re.compile(r"\bTrainer\.Advanced\.trainLoaderWith\b"),
        "public examples should use `Trainer.RunConfig` + `Trainer.TrainOptions` with `trainer.train`, not `Trainer.Advanced.trainLoaderWith`.",
    ),
    (
        re.compile(r"\bTrainer\.Advanced\.logLossEvery\b"),
        "public examples should keep logging inline or use `Trainer.TrainSummary`; do not expose `Trainer.Advanced.logLossEvery`.",
    ),
    (
        re.compile(r"\bfitWithParams\b"),
        "public examples should prefer `fitWithCompiledForward1` or other public trainer/verifier bridges instead of reopening raw post-training parameter callbacks.",
    ),
    (
        re.compile(r"\bTList\b"),
        "public examples should not expose `TList`; use model/trainer/checkpoint helpers instead.",
    ),
    (
        re.compile(r"\btlist\.TList\b|\btlist!\b"),
        "public examples should not expose raw `tlist` packs; use public tensor/model helpers instead.",
    ),
    (
        re.compile(r"\bModule\.instantiateWithOptions\b"),
        "public examples should use `Module.instantiate`, `Module.instantiateMse`, or `Module.instantiateCrossEntropyOneHot`, not `Module.instantiateWithOptions` directly.",
    ),
    (
        re.compile(r"\bTorchLean\.Module\.run\b"),
        "public examples should use `Runtime.runFloat`, `Runtime.withOptions`, or `ModelZoo.runFloat`, not the raw `TorchLean.Module.run` dispatcher.",
    ),
    (
        re.compile(r"\bRuntime\.withSelected(NoCast)?\b"),
        "public examples should use `Runtime.withOptions` / `Runtime.withOptionsNoCast`, which pass parsed runtime options through the facade.",
    ),
    (
        re.compile(r"\bModule\.(withMseModel|withCrossEntropyOneHotModel|withScalarLossModel)\b"),
        "public model/example training should use `Trainer.*` handles, not raw `Module.with*Model` setup.",
    ),
    (
        re.compile(r"\bModule\.(lossScalar|optimizerStep|predict1|predict1NoGrad)\b"),
        "public model/example training should use trained handles (`trained.predict`, callbacks, or `verify`), not raw module stepping/prediction.",
    ),
    (
        re.compile(
            r"\bRealData\.fit(CifarClassifierModel|CifarRegressionModel|CsvRegressionModel|HouseholdPowerRegressionModel)\b"
        ),
        "public model-zoo examples should use the current `Trainer.Command` runners, not the old `*Model` wrappers.",
    ),
    (
        re.compile(r"\bTrainer\.NewConfig\b|\bNewConfig\b"),
        "public examples should use `Trainer.Config`; `Trainer.NewConfig` was removed during the unified Trainer cleanup.",
    ),
    (
        re.compile(
            r"\bfitCsvRegression\b|\brunCsvRegressionTrain\b|\bfitNpyRegression\b|"
            r"\brunCifar(Classifier|Regression|Curve)Train\b|"
            r"\brun(RegressionCsv|ClassificationNpy|RegressionNpy|ForecastWindow)\b"
        ),
        "public examples should use `Trainer.new` / `trainer.train` or `Trainer.Command` runners, not removed command-wrapper names.",
    ),
    (
        re.compile(r"\bModelZoo\.Command\b|\bTrainCommand\.run\b"),
        "public examples should use `Trainer.Command` for runnable model-zoo command glue.",
    ),
    (
        re.compile(r"\bSimpleText\.main\b"),
        "shared sequence-model code should expose one executable entrypoint, not nested `*.main` actions.",
    ),
    (
        re.compile(r"\.verify\s*\(\s*Trainer\.Verify\.lInfIBP\b"),
        "public examples should prefer `trained.verifyRobustLInf x eps` over manually building a `Trainer.Verify.lInfIBP` request.",
    ),
    (
        re.compile(r"\bTrainer\.(FitOptions|TrainOptions)\.forSteps\b"),
        "public examples should prefer record literals such as `{ steps := n }`, which match the trainer.train API shown in quickstarts.",
    ),
    (
        re.compile(r"\bTrainer\.FitOptions\b"),
        "`Trainer.FitOptions` was removed; public examples should name `Trainer.TrainOptions`.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\b"),
        "`Trainer.FitSummary` was removed; public examples should use `Trainer.TrainSummary`.",
    ),
    (
        re.compile(r"\bTrainer\.(regression|classifier|crossEntropy|custom)\b"),
        "public examples should use `Trainer.new ... { task := ... }`; specialized `Trainer.*` constructors are removed.",
    ),
    (
        re.compile(r"\bTrainer\.(Regression|Classifier|CrossEntropy|Custom)(\.|\b)"),
        "public examples should stay on the unified `Trainer` facade, not specialized trainer implementation handles.",
    ),
    (
        re.compile(r"\bstructure\s+RunConfig\b"),
        "public examples should not define their own `RunConfig`; reserve that name for `Trainer.RunConfig` and use domain-specific option names.",
    ),
    (
        re.compile(r"\btrainer\.fit\b"),
        "public examples should call `trainer.train`; `trained` is the conventional local name for the trained result.",
    ),
    (
        re.compile(r"\bRuntimeFit\b|\bparseRuntimeFit\b|\bparsed\.fit\b"),
        "quickstart examples should use `RuntimeTrain`, `parseRuntimeTrain`, and `parsed.trainOptions`.",
    ),
    (
        re.compile(r"\bfitOptionsWhenLogRequested\b|\bfitOptions\b"),
        "public examples should use `trainOptions` terminology, not old `fitOptions` spellings.",
    ),
    (
        re.compile(r"\.fit(StreamFloat|PairStreamFloat|SelectedCrossEntropy)\b|\bfit(StreamFloat|PairStreamFloat|SelectedCrossEntropy)\b"),
        "public examples should use the `train*` trainer methods, not old stream/selected-training helpers.",
    ),
    (
        re.compile(r"\(\s*\{[^}]*optimizer\s*:=.*\}\s*:\s*Trainer\.RunConfig\s*\)\.withOptions\s+opts"),
        "public examples should use `Trainer.runConfig opts { optimizer := ... }`, not a type-ascribed RunConfig followed by `.withOptions opts`.",
    ),
    (
        re.compile(r"\b(backend := opts\.backend|device := if opts\.useGpu|fastKernels := opts\.fastKernels|fastGpuMatmulPrecision := opts\.fastGpuMatmulPrecision)\b"),
        "public examples should use `Trainer.runConfig opts { ... }` or `Trainer.runtimeSettings opts { ... }` instead of manually copying runtime fields from `opts`.",
    ),
    (
        re.compile(r"\bnn\.(mseScalarModuleDef|crossEntropyOneHotScalarModuleDef)\b"),
        "public examples should use public `Module.instantiate*` helpers instead of spelling raw loss-module defs.",
    ),
    (
        re.compile(r"\bfun\s+\{α\}"),
        "public examples should avoid raw polymorphic runtime callbacks in user-facing code.",
    ),
    (
        re.compile(r"^\s*open\s+NN\.API\b", flags=re.MULTILINE),
        "public examples should `open TorchLean`, not `open NN.API`.",
    ),
    (
        re.compile(r"^\s*(public\s+)?import\s+NN\.API\b", flags=re.MULTILINE),
        "public examples should import `TorchLean`, not `NN.API.*` directly.",
    ),
    (
        re.compile(r"\bNN\.API\."),
        "public examples should go through the `TorchLean` facade, not fully-qualified `NN.API.*` implementation paths.",
    ),
    (
        re.compile(r"IO\.println\s+\"model\s*="),
        "public examples should print `model.info`, not a hardcoded model banner.",
    ),
    (
        re.compile(r"\bIO\.println\s+trainer\.info\b"),
        "public examples should use `trainer.printInfo` so model-summary formatting stays consistent.",
    ),
    (
        re.compile(r"\bIO\.println\s+\w*Trainer\.info\b"),
        "public examples should use `trainer.printInfo` / `trainer.printInfoAs`, not direct trainer-info printing.",
    ),
    (
        re.compile(r"\bIO\.println\s+(report|fit)\.summary\b"),
        "public examples should use `report.printSummary` / `trained.printSummary` for trained results.",
    ),
    (
        re.compile(r"\blet\s+report\s+←\s+(trainer\.train|train\s+opts\s+flags)\b"),
        "public examples should call trained results `trained`, not `report`; `trainer.train` returns a trained handle, not just a summary.",
    ),
    (
        re.compile(r"\bfit\.fit\.predict(Batch)?\b"),
        "public stream examples should use `trained.predict` / `trained.predictBatch`; do not expose the nested trained-handle field.",
    ),
    (
        re.compile(r"\bfit\.curve\.values\b"),
        "public paired-stream examples should use `trained.printCurveSummary` for before/after curve summaries.",
    ),
    (
        re.compile(r"\bIO\.println\s+cert\.summary\b"),
        "public examples should use `cert.printSummary` for verification reports.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\.parseFloat\?\b"),
        "public examples should use `Trainer.TrainSummary.requireAndPrintFloatLosses` when numeric losses are required, not hand-rolled optional parsing.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\.requireFloatLosses\b"),
        "public examples should use `Trainer.TrainSummary.requireAndPrintFloatLosses` when they need numeric losses for logs.",
    ),
]

PUBLIC_TUTORIAL_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\btrainer\.trainClassifier\b"),
        "public examples should batch classifier datasets with `Data.batchDataset` and call ordinary `trainer.train`; `trainer.trainClassifier` was removed.",
    ),
    (
        re.compile(r"\btrainClassifierWithFlags\b"),
        "public examples should batch classifier datasets with `Data.batchDataset` and call ordinary `trainer.train`; classifier-specific trainer loops were removed.",
    ),
]

TOP_LEVEL_API_DECL_RE = re.compile(
    r"^\s*(def|structure|inductive|class|abbrev|instance|theorem|lemma)\s+",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class Finding:
    """One repository-lint warning or error."""

    level: str  # "ERROR" | "WARN"
    path: pathlib.Path
    line: int | None
    col: int | None
    message: str

    def render(self) -> str:
        """Format the finding for terminal and CI output."""
        rel = self.path.relative_to(REPO_ROOT)
        if self.line is None:
            return f"{self.level}: {rel}: {self.message}"
        if self.col is None:
            return f"{self.level}: {rel}:{self.line}: {self.message}"
        return f"{self.level}: {rel}:{self.line}:{self.col}: {self.message}"


def _iter_lean_files() -> Iterable[pathlib.Path]:
    """Yield project Lean files while skipping vendored and generated trees."""
    for p in REPO_ROOT.rglob("*.lean"):
        # Vendored dependencies are checked by their own projects.
        if ".lake" in p.parts:
            continue
        if any(d in p.parts for d in VENDORED_DIR_NAMES):
            continue
        # `_out` can contain generated artifacts; keep policy focused on sources.
        if "_out" in p.parts:
            continue
        yield p


def _iter_generated_script_artifacts() -> Iterable[pathlib.Path]:
    """Generated files that stay outside the checked-in `scripts/` tree."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"} or p.name == ".DS_Store":
            yield p


def _iter_script_files() -> Iterable[pathlib.Path]:
    """Yield checked-in support scripts and helper files under `scripts/`."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if p.is_file():
            yield p


def _is_executable(path: pathlib.Path) -> bool:
    """Return whether any executable bit is set for `path`."""

    return bool(path.stat().st_mode & 0o111)


def _has_shebang(text: str) -> bool:
    """Return whether `text` starts with a Unix shebang line."""

    return text.startswith("#!")


def _has_python_module_docstring(text: str) -> bool:
    """Return whether a Python script starts with a module docstring after an optional shebang."""

    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    body = "\n".join(lines).lstrip()
    return body.startswith(('"""', "'''"))


def _script_needs_readme_entry(path: pathlib.Path) -> bool:
    """Return whether `path` should be mentioned explicitly in `scripts/README.md`."""

    if path.name == "README.md":
        return False
    if path.suffix in {".json", ".txt"}:
        return False
    return path.is_file()


def _line_col(text: str, idx: int) -> tuple[int, int]:
    """Translate a string offset into 1-based line and column coordinates."""
    # 1-based (Lean-style).
    line = text.count("\n", 0, idx) + 1
    last_nl = text.rfind("\n", 0, idx)
    col = idx - last_nl
    return line, col


def _has_nn_header(path: pathlib.Path, text: str) -> bool:
    """Check whether an `NN/` source file carries the standard TorchLean header."""
    # TorchLean policy: NN sources carry a consistent header at the top of the file.
    if not path.is_relative_to(REPO_ROOT / "NN"):
        return True
    head = "\n".join(text.splitlines()[:10])
    return "Copyright (c) 2026 TorchLean" in head


def _has_lean_module_docstring(text: str) -> bool:
    """Return whether a Lean source contains a module docstring (`/-! ... -/`)."""
    return "/-!" in text


def _mask_lean_comments_and_strings(text: str) -> str:
    """
    Return a same-length string where Lean comments/docstrings and string literals are replaced
    with spaces (newlines preserved).

    This prevents repo-lint regexes like `\\bsorry\\b` from triggering on policy mentions in
    docstrings/comments (and avoids false positives in string literals).

    Notes:
      - Lean block comments `/- ... -/` nest; the scanner tracks nesting depth.
      - The scanner is lexical (no full parser), but it avoids treating comment
        markers inside strings as comments.
    """

    out = list(text)
    n = len(text)
    i = 0

    in_line_comment = False
    block_depth = 0
    in_string = False

    while i < n:
        ch = text[i]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                i += 1
            else:
                out[i] = " "
                i += 1
            continue

        if block_depth > 0:
            if text.startswith("/-", i):
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                block_depth += 1
                i += 2
                continue
            if text.startswith("-/", i):
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                block_depth -= 1
                i += 2
                continue
            if ch == "\n":
                i += 1
            else:
                out[i] = " "
                i += 1
            continue

        if in_string:
            # Mask string contents while preserving newlines. Lean strings should
            # not contain raw newlines, but the scanner stays defensive so a
            # missing quote does not mask the rest of the file.
            if ch == "\n":
                in_string = False
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                # Escape sequence: mask both chars.
                out[i] = " "
                if text[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            out[i] = " "
            if ch == '"':
                in_string = False
            i += 1
            continue

        # Outside comments/strings: detect comment/string starts.
        if text.startswith("--", i):
            out[i] = " "
            if i + 1 < n:
                out[i + 1] = " "
            in_line_comment = True
            i += 2
            continue

        if text.startswith("/-", i):
            out[i] = " "
            if i + 1 < n:
                out[i + 1] = " "
            block_depth = 1
            i += 2
            continue

        if ch == '"':
            out[i] = " "
            in_string = True
            i += 1
            continue

        i += 1

    return "".join(out)


def lint_repo(*, fail_on_warn: bool) -> list[Finding]:
    """Run TorchLean's repository hygiene checks and return all findings."""
    findings: list[Finding] = []

    if not LINT_SCOPE_SENTINEL.exists():
        findings.append(
            Finding(
                "ERROR",
                LINT_SCOPE_SENTINEL,
                None,
                None,
                "repo linter is not rooted at TorchLean; expected to see NN/MLTheory/CROWN/Lyapunov/Oracle.lean.",
            )
        )

    for path in _iter_generated_script_artifacts():
        findings.append(
            Finding(
                "ERROR",
                path,
                None,
                None,
                "generated Python/cache artifact under `scripts/`; remove it from the source tree.",
            )
        )

    scripts_readme = REPO_ROOT / "scripts/README.md"
    try:
        scripts_readme_text = scripts_readme.read_text(encoding="utf-8")
    except OSError as e:
        scripts_readme_text = ""
        findings.append(Finding("ERROR", scripts_readme, None, None, f"failed to read file: {e}"))

    for path in _iter_script_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as e:
            findings.append(Finding("ERROR", path, None, None, f"failed to read file: {e}"))
            continue

        if path.suffix in {".py", ".sh"}:
            has_shebang = _has_shebang(text)
            is_executable = _is_executable(path)
            if has_shebang and not is_executable:
                findings.append(
                    Finding("ERROR", path, 1, 1, "script has a shebang but is not executable.")
                )
            if is_executable and not has_shebang:
                findings.append(
                    Finding("ERROR", path, 1, 1, "executable script should start with a shebang.")
                )

        if path.suffix == ".py" and not _has_python_module_docstring(text):
            findings.append(
                Finding("ERROR", path, 1, 1, "Python scripts/helpers should start with a module docstring.")
            )

        if _script_needs_readme_entry(path):
            script_rel = rel.removeprefix("scripts/")
            if script_rel not in scripts_readme_text:
                findings.append(
                    Finding("ERROR", path, None, None, "`scripts/README.md` should explain this script.")
                )

    banned_regexes: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bnative_decide\b"), "`native_decide` is banned in TorchLean."),
        (re.compile(r"\bsorry\b"), "`sorry` is banned in TorchLean sources."),
        (re.compile(r"\badmit\b"), "`admit` is banned in TorchLean sources."),
        (
            re.compile(r"\b(FitConfig|LoaderFitConfig|FitReport)\b"),
            "old lower training names are removed; use `TrainConfig`, `LoaderTrainConfig`, and `TrainReport`.",
        ),
        (
            re.compile(r"\beffectiveFitBatchSize\b"),
            "old lower training helper names are removed; use `effectiveTrainBatchSize`.",
        ),
        (
            re.compile(r"\b(FitResult|StreamFitResult|PairStreamFitResult)\b"),
            "old trainer result names are removed; use `TrainResult`, `StreamTrainResult`, and `PairStreamTrainResult`.",
        ),
        (
            re.compile(r"\bverifyLInfIBP\b|\b(Trainer\.)?Verify\.robustLInf\b"),
            "duplicate verification helper names are removed; use `verifyRobustLInf` on trained results or `Trainer.Verify.lInfIBP` for requests.",
        ),
        (re.compile(r"\bby\s+omega\b"), "`omega` is banned in TorchLean; prefer `linarith`/`nlinarith`/`grind` or small arithmetic lemmas."),
        (re.compile(r"^\s*omega\b", flags=re.MULTILINE), "`omega` is banned in TorchLean; prefer `linarith`/`nlinarith`/`grind` or small arithmetic lemmas."),
        (re.compile(r"\bsimp\s*\[\s*\*(\s*[,\]])"), "`simp [*]` is banned; prefer `simp [h₁, h₂]` or `simp (config := ...)` with explicit hypotheses."),
        (
            re.compile(r"^\s*public\s+import\s+Mathlib\.Tactic\b", flags=re.MULTILINE),
            "Do not `public import Mathlib.Tactic.*`; import the specific tactic modules you use (non-public).",
        ),
        (
            re.compile(r"^\s*import\s+Mathlib\.Tactic(?!\.)\b", flags=re.MULTILINE),
            "Do not `import Mathlib.Tactic` (umbrella import). Import the specific `Mathlib.Tactic.*` modules you use.",
        ),
        (re.compile(r"@\[\s*de" r"precated\b"), "`@[de" "precated]` is banned in TorchLean sources."),
    ]

    axiom_re = re.compile(r"^\s*axiom\s+([A-Za-z0-9_'.]+)\b", flags=re.MULTILINE)

    for path in _iter_lean_files():
        try:
            raw = path.read_bytes()
        except OSError as e:
            findings.append(Finding("ERROR", path, None, None, f"failed to read file: {e}"))
            continue

        # Enforce LF-only; CRLF and stray CR cause confusing diffs and occasional parser weirdness.
        if b"\r" in raw:
            findings.append(Finding("ERROR", path, None, None, "contains CR (`\\r`) characters (use LF)."))
            continue

        text = raw.decode("utf-8", errors="replace")
        masked = _mask_lean_comments_and_strings(text)

        if not _has_nn_header(path, text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing TorchLean header in the first ~10 lines (expected `Copyright (c) 2026 TorchLean`).",
                )
            )

        if path.is_relative_to(REPO_ROOT / "NN") and not _has_lean_module_docstring(text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing Lean module docstring (`/-! ... -/`); add purpose, main declarations, and import guidance.",
                )
            )

        # Line-level whitespace hygiene: Lean rejects tabs, and trailing whitespace is noisy in reviews.
        for i, line in enumerate(text.splitlines(), start=1):
            if "\t" in line:
                col = line.find("\t") + 1
                findings.append(Finding("ERROR", path, i, col, "tab character found (use spaces)."))
            if line.endswith(" "):
                findings.append(Finding("ERROR", path, i, len(line), "trailing whitespace."))

        for rx, msg in banned_regexes:
            for m in rx.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))

        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith("NN/API/Public/Facade/Trainer/Train/") and rel.endswith(".lean"):
            if "(opts : Options)" in masked and "(opts : TrainOptions" in masked:
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        None,
                        None,
                        "trainer train implementation should distinguish runtime `Options` from `TrainOptions` (use names like `runtimeOpts` and `trainOpts`).",
                    )
                )

        if (
            rel.startswith("NN/API/")
            and rel.count("/") == 2
            and rel.endswith(".lean")
        ):
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "top-level `NN/API/*.lean` files must be import entrypoints only; move implementation declarations into a matching subfolder.",
                    )
                )

        if (
            rel.startswith("NN/API/Public/")
            and rel.count("/") == 3
            and rel.endswith(".lean")
        ):
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "direct `NN/API/Public/*.lean` files must be import entrypoints only; move implementation declarations into a matching subfolder.",
                    )
                )

        if (
            rel.startswith("NN/API/Public/Facade/")
            and rel.count("/") == 4
            and rel.endswith(".lean")
        ):
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "direct `NN/API/Public/Facade/*.lean` files must be import entrypoints only; move implementation declarations into a matching subfolder.",
                    )
                )

        if rel == "NN/API/Public/Facade/Base/Core.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "`NN.API.Public.Facade.Base.Core` must stay an import-only aggregator; put base facade implementation in `NN.API.Public.Facade.Base.*` modules.",
                    )
                )

        if rel == "NN/API/Public/Facade/Runtime/Core.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "`NN.API.Public.Facade.Runtime.Core` must stay an import-only aggregator; put runtime facade implementation in `NN.API.Public.Facade.Runtime.*` modules.",
                    )
                )

        if rel == "NN/API/Public/Facade/NN/Core.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "`NN.API.Public.Facade.NN.Core` must stay an import-only aggregator; put NN facade implementation in `NN.API.Public.Facade.NN.*` modules.",
                    )
                )

        if rel == "NN/API/Public/Facade/Data/Core.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "`NN.API.Public.Facade.Data.Core` must stay an import-only aggregator; put data facade implementation in `NN.API.Public.Facade.Data.*` modules.",
                    )
                )

        if rel == "NN/API/Public/Facade/Trainer/Train.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "`NN.API.Public.Facade.Trainer.Train` must stay an import-only aggregator; put training implementation in `NN.API.Public.Facade.Trainer.Train.*` modules.",
                    )
                )

        if any(rel == prefix or rel.startswith(prefix) for prefix in PUBLIC_GUIDE_PREFIXES):
            for rx, msg in PUBLIC_GUIDE_BANNED_PATTERNS:
                for m in rx.finditer(text):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        if rel == "NN/API/Public.lean" and re.search(
            r"^\s*public\s+import\s+NN\.API\.Public\.Training\b", masked, flags=re.MULTILINE
        ):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Public` must not re-export `NN.API.Public.Training`; use `TorchLean.Trainer` for ordinary code and import the advanced training module explicitly when needed.",
                )
            )
        is_trainer_facade = (
            rel == "NN/API/Public/Facade/Trainer.lean"
            or rel.startswith("NN/API/Public/Facade/Trainer/")
        )
        is_training_entrypoint = rel == "NN/API/Public/Training.lean"
        if re.search(
            r"^\s*public\s+import\s+NN\.API\.Public\.Training\b", masked, flags=re.MULTILINE
        ) and not (is_trainer_facade or is_training_entrypoint):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Public.Training` should only be imported by the Trainer facade; keep the callback-heavy training layer out of broad public facade imports.",
                )
            )

        if rel.endswith(".lean") and any(rel.startswith(prefix) for prefix in PUBLIC_EXAMPLE_PREFIXES):
            for rx, msg in PUBLIC_EXAMPLE_BANNED_PATTERNS:
                for m in rx.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        if rel.endswith(".lean") and any(rel.startswith(prefix) for prefix in PUBLIC_TUTORIAL_PREFIXES):
            for rx, msg in PUBLIC_TUTORIAL_BANNED_PATTERNS:
                for m in rx.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        # Axioms must be quarantined and named explicitly.
        allowed_axiom_names = ALLOWED_AXIOMS.get(rel, set())
        for m in axiom_re.finditer(masked):
            axiom_name = m.group(1)
            if axiom_name not in allowed_axiom_names:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"axiom `{axiom_name}` is not allowlisted; quarantine and document trusted axioms.",
                    )
                )

        # Warn by default; callers can promote these warnings with `--fail-on-warn`.
        if "set_option linter." in masked and " false" in masked:
            suppressions = list(
                re.finditer(r"set_option\s+linter\.([A-Za-z0-9_]+)\s+false(?:\s+in)?", masked)
            )
            disallowed_suppressions = [
                m for m in suppressions
                if not (m.group(1) == "auxLemma" and m.group(0).rstrip().endswith(" in"))
            ]
            # Only a coarse signal; report once per file.
            if disallowed_suppressions:
                rel_posix = path.relative_to(REPO_ROOT).as_posix()
                # Some executable examples, tests, and maintenance scripts scope linter options
                # locally. Keep this warning focused on library-facing code where suppressions are
                # part of the public proof surface.
                if (
                    rel_posix in ALLOWED_LINTER_SUPPRESSION_FILES
                    or
                    rel_posix.startswith("NN/Examples/")
                    or rel_posix.startswith("NN/Tests/")
                    or rel_posix.startswith("scripts/")
                    # The compiled correctness proofs scope linter options locally to
                    # keep proof scripts readable; warning here is usually not actionable.
                    or rel_posix.startswith("NN/Runtime/Autograd/Compiled/IRExec/Correctness/")
                ):
                    pass
                else:
                    findings.append(
                        Finding(
                            "WARN",
                            path,
                            None,
                            None,
                            "suppresses a linter (`set_option linter.* false`). Prefer fixing the warning or scoping the option tightly.",
                        )
                    )

    if not fail_on_warn:
        return findings
    # Promote warnings to errors.
    return [
        Finding("ERROR" if f.level == "WARN" else f.level, f.path, f.line, f.col, f.message)
        for f in findings
    ]


def main() -> int:
    """CLI entry point used by local checks and CI."""
    ap = argparse.ArgumentParser(description="TorchLean repo lints (project policies).")
    ap.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Treat warnings as errors (useful for tightening policies over time).",
    )
    args = ap.parse_args()

    findings = lint_repo(fail_on_warn=args.fail_on_warn)
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]

    for f in findings:
        print(f.render())

    if errors:
        print(f"\nFAILED: {len(errors)} error(s), {len(warns)} warning(s).")
        return 1

    if warns:
        print(f"\nOK (with warnings): {len(warns)} warning(s).")
        return 0

    print("OK: no issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
