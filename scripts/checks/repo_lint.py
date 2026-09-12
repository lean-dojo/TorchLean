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
import os
import pathlib
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LINT_SCOPE_SENTINEL = REPO_ROOT / "NN/MLTheory/CROWN/Lyapunov/Certificate.lean"

# External trees that may exist in a developer checkout but are not part of TorchLean's core sources
# and must not affect repo policy/CI. These are user-cloned repos outside TorchLean's source tree.
#
# Source discovery skips these directories so optional checkouts do not affect `lake lint`.
VENDORED_DIR_NAMES = {
    "Two-Stage_Neural_Controller_Training",  # optional external checkout (α,β-CROWN workflows)
    "PINN_verification",  # user-cloned external repo (gitignored)
}

# Prune these before descent: filtering yielded paths still crawls their contents on EFS.
SOURCE_EXCLUDED_DIR_NAMES = VENDORED_DIR_NAMES | {
    ".git",
    ".lake",
    ".cache",
    ".venv",
    ".bundle",
    ".jekyll-cache",
    ".sass-cache",
    ".verso",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "_out",
    "_site",
    "vendor",
}

# These output paths have ordinary names that can also occur in authored source directories.
GENERATED_DOC_DIRS = {
    "docs/manual",
    "home_page/docs",
    "home_page/manual",
    "home_page/blueprint/print",
    "home_page/blueprint/web",
}

# Keep the trusted boundary explicit: axioms must be quarantined, named, and documented.
# TorchLean currently has no custom axioms.
ALLOWED_AXIOMS: dict[str, set[str]] = {}

# These modules were pure compatibility routes or duplicate import surfaces. New code must use the
# canonical subsystem umbrellas and namespaces instead of recreating them.
REMOVED_COMPATIBILITY_PATHS = {
    "NN/API/TorchLean/Optimizers.lean",
    "NN/API/TensorPack.lean",
    "NN/Spec/Core/Tensor/API.lean",
    "NN/Examples/Verification/LiRPA.lean",
    "NN/GraphSpec/Models/TorchLean/Fno1d.lean",
    "NN/GraphSpec/Models/TorchLean.lean",
    "NN/GraphSpec/Models/TorchLean/Autoencoder.lean",
    "NN/Library.lean",
    "NN/API/TorchLean/Trainer/Verify.lean",
    "NN/API/TorchLean/Data/DotInfo.lean",
    "NN/API/Neural/FunctionalBatch.lean",
    "NN/API/Models/Gpt2.lean",
    "NN/API/Models/Mlp.lean",
    "NN/Runtime/Autograd/Engine/Core/Core.lean",
    "NN/Runtime/Autograd/IRExec/Helpers.lean",
    "NN/Runtime/Autograd/Torch/Utils.lean",
    "NN/Runtime/Autograd/Utils.lean",
    "NN/Proofs/RuntimeApprox/NF/Utils.lean",
    "NN/Spec/Core/Utils.lean",
    "NN/Spec/Models/CommonHelpers.lean",
    "NN/Spec/Layers/Utils.lean",
    "NN/Verification/Builtin/Proved/Correctness/Eval/Coverage.lean",
    "NN/MLTheory/CROWN/Lyapunov/Oracle.lean",
    "NN/MLTheory/CROWN/Tactics/CrownOracle.lean",
    "NN/Spec/Layers/Pooling/Aliases.lean",
    "NN/Spec/Layers/Pooling/PaddedTwoD.lean",
    "NN/Spec/Layers/Pooling/TwoD.lean",
    "NN/Spec/Layers/Conv/TwoD/Padding.lean",
    "NN/Spec/Core/Tensor/Vec.lean",
    "NN/Spec/Core/TensorArray.lean",
    "NN/Spec/Core/TensorBridge.lean",
    "NN/GraphSpec/Models/TorchLean/Cnn.lean",
    "NN/GraphSpec/Models/TorchLean/Mlp.lean",
    "NN/GraphSpec/Models/TorchLean/TransformerBlock.lean",
    "NN/Proofs/Autograd/Tape/Ops/Norm/BatchNormChannelFirst.lean",
    "NN/Proofs/RuntimeApprox/NF/Conv.lean",
    "NN/Proofs/RuntimeApprox/NF/ConvBackward.lean",
    "NN/Proofs/RuntimeApprox/NF/ConvForward.lean",
    "NN/Spec/Models/Unet.lean",
    "NN/Spec/Models/Vit.lean",
    "NN/Runtime/Autograd/Train/TensorLoader.lean",
    "NN/Runtime/Autograd/Train/Dataset.lean",
    "NN/Runtime/Autograd/Train/IoLoader.lean",
    "NN/Runtime/Autograd/Train/IoLoader/Parsing.lean",
    "NN/Runtime/Autograd/Train/IoLoader/Csv.lean",
    "NN/Runtime/Autograd/Train/IoLoader/Npy.lean",
    "NN/Verification/Builtin/Verified.lean",
    "NN/Verification/Builtin/Proved/Correctness/Eval/Pooling.lean",
    "NN/Runtime/Context.lean",
    "NN/Widgets/Runtime/Context.lean",
    "NN/Runtime/Optim/GradientUtils.lean",
    "NN/Spec/Core/Tensor/Packed.lean",
    "NN/Proofs/Autograd/Runtime/PackedTensor.lean",
    "NN/API/Data/PackedDataset.lean",
    "NN/API/Data/TensorDataset.lean",
    "NN/API/Models/Transformer.lean",
    "NN/API/Neural/Heads.lean",
    "NN/API/Neural/Layers/Normalization.lean",
    "NN/API/Scalar.lean",
    "NN/API/Trainer/Manual.lean",
    "NN/API/Trainer/Manual/Control.lean",
    "NN/API/Trainer/Manual/Core.lean",
    "NN/API/Trainer/Manual/Evaluation.lean",
    "NN/API/Trainer/Manual/Execution.lean",
    "NN/API/Trainer/Manual/Loaders.lean",
    "NN/API/Trainer/Manual/Loops.lean",
    "NN/API/Trainer/Manual/Optimizer.lean",
    "NN/API/Trainer/Manual/Stepper.lean",
    "NN/API/Trainer/Manual/Streams.lean",
    "NN/API/Trainer/Predict.lean",
    "NN/API/Trainer/Train/Custom.lean",
    "NN/API/Trainer/Train/OneHotCrossEntropy.lean",
    "NN/API/Trainer/Train/Regression.lean",
    "NN/API/Trainer/Train/Streams.lean",
    "NN/API/Verification/Trainer.lean",
    "NN/Tensor/Printing.lean",
    "NN/Tensor/Syntax.lean",
    "NN/Runtime/Autograd/Model/Random.lean",
}

REMOVED_COMPATIBILITY_PREFIXES = (
    "NN/Entrypoint/",
    "NN/API/Public/",
)

# Documentation may mention producer-side environment variables only when the implementation hook
# exists in source. This prevents guide text from advertising phantom integration flags.
DOCUMENTED_ENV_VAR_IMPLEMENTATIONS = {
    "ABCROWN_ARTIFACT_OUT": "scripts/verification/abcrown/export_leaf_artifact.py",
}

TRUST_BOUNDARY_DECL_REFS = {
    "NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownTransferSound": (
        "NN/MLTheory/CROWN/Proofs/GraphCrownCertSoundness.lean",
        re.compile(r"\bdef\s+CrownTransferSound\b"),
    ),
    "NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.OpsExact.Sound": (
        "NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean",
        re.compile(r"\bclass\s+Sound\s*:\s*Prop\b"),
    ),
}

DOC_FACT_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"leaf (?:artifact )?JSON format exported from (?:α,β-CROWN|alpha-beta-CROWN)",
            flags=re.IGNORECASE,
        ),
        "TorchLean alpha-beta-CROWN leaf JSON is produced by TorchLean's converter from raw terminal-domain data; do not imply the external verifier natively exports the TorchLean schema.",
    ),
    (
        re.compile(
            r"Set\s+`?ABCROWN_ARTIFACT_OUT`?.{0,80}(?:before running|when running)\s+(?:α,β-CROWN|alpha-beta-CROWN)",
            flags=re.IGNORECASE,
        ),
        "`ABCROWN_ARTIFACT_OUT` belongs to TorchLean's exporter/helper boundary around alpha-beta-CROWN.",
    ),
    (
        re.compile(
            r"(?:external tool|alpha-beta-CROWN)\s+writes\s+the\s+JSON",
            flags=re.IGNORECASE,
        ),
        "TorchLean's alpha-beta-CROWN JSON schema is written by the TorchLean exporter/helper, not vanilla alpha-beta-CROWN.",
    ),
    (
        re.compile(
            r"Two-Stage tooling can emit a small JSON \*leaf\s+certificate\*",
            flags=re.IGNORECASE,
        ),
        "Say that an instrumented external verifier exposes terminal domains and TorchLean's helper converts them; do not imply the external Two-Stage tooling natively emits TorchLean's schema.",
    ),
    (
        re.compile(
            r"External JSON artifacts are treated as untrusted\.\s+Checkers parse them, validate shapes, and compare\s+them against Lean recomputation",
            flags=re.IGNORECASE,
        ),
        "Not every JSON artifact is recomputed in Lean; distinguish structural checks, recomputation checks, and theorem-backed checks.",
    ),
    (
        re.compile(
            r"A CUDA run proves that the CUDA path executed",
            flags=re.IGNORECASE,
        ),
        "Reserve `prove` for Lean/checker claims; a CUDA run is runtime evidence, not a proof.",
    ),
    (
        re.compile(
            r"verified reverse mode autograd",
            flags=re.IGNORECASE,
        ),
        "Do not imply all reverse-mode autograd is verified; say selected reverse-mode/autograd proofs.",
    ),
    (
        re.compile(
            r"bundled (?:α,β-CROWN|alpha-beta-CROWN) leaf certificate",
            flags=re.IGNORECASE,
        ),
        "The bundled alpha-beta-CROWN file is a structural leaf artifact, not a proof-backed certificate.",
    ),
    (
        re.compile(
            r"External (?:α,β-CROWN|alpha-beta-CROWN) artifact.*JSON leaf certificate",
            flags=re.IGNORECASE,
        ),
        "Call the alpha-beta-CROWN import a JSON leaf artifact unless the computation is replayed or proof-backed.",
    ),
    (
        re.compile(
            r"\bleaf_cert\.json\b|<cert\.json>|Output certificate path",
            flags=re.IGNORECASE,
        ),
        "Alpha-beta-CROWN-facing docs and CLI help should use `leaf_artifact.json` / artifact wording; the checker is structural unless a separate proof/replay path is named.",
    ),
    (
        re.compile(
            r"Autograd correctness.*backprop computes the adjoint derivative",
            flags=re.IGNORECASE,
        ),
        "Autograd correctness claims must name the supported tape node or graph fragment.",
    ),
    (
        re.compile(
            r"JSON \*leaf\s+certificate\*",
            flags=re.IGNORECASE,
        ),
        "Use `leaf artifact` for alpha-beta-CROWN structural JSON unless the computation is replayed or proof-backed.",
    ),
    (
        re.compile(
            r"abcrown-leaf` to check a JSON certif(?:icate)? against TorchLean's semantics",
            flags=re.IGNORECASE,
        ),
        "Use artifact wording: abcrown-leaf checks a converted structural leaf artifact plus a local witness predicate at the TorchLean boundary.",
    ),
]

PUBLIC_DOC_API_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bData\.(?:TensorSource|SupervisedEpochs)\b|\bTensorSource\.load\b"
        ),
        "single tensor files load through `Tensor.load`; dataset source records are reserved "
        "for paired supervised or labeled data, and typed manual loops use `Data.Loader`.",
    ),
    (
        re.compile(r"\b(?:inputShape|outputShape|targetShape)\b(?!\?)"),
        "public tensor signatures use `input`, `output`, or `target`; use `σ`/`τ` for "
        "invisible generic shape indices and reserve a `Shape` suffix for names that "
        "distinguish multiple shapes.",
    ),
    (
        re.compile(r"\bdata/model_zoo/"),
        "example artifacts live under `data/examples`; the removed ModelZoo name must not "
        "reappear in public documentation.",
    ),
    (
        re.compile(r"\bRuntime\.(?:mm|bmm)\b"),
        "public docs should use generic `Runtime.matmul`; the rank-specific `mm` and `bmm` APIs "
        "were removed.",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_])\.dim\b"),
        "public docs should write concrete tensor shapes with dimension-list syntax.",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9_])\.scalar\b|\b(?:Shape|Tensor)\.scalar\b"
        ),
        "public docs should write rank-zero tensor shapes as `[]` and rank-zero values as "
        "ordinary typed literals, not recursive representation constructors.",
    ),
    (
        re.compile(r"(?<![0-9])\.[12]\b(?!\.)"),
        "public docs should destructure multi-value results instead of using anonymous `.1` and "
        "`.2` projections.",
    ),
    (
        re.compile(r"\b(?:stateGradient|stateCotangent|inputCotangent)\b"),
        "public docs should name destructured VJP values `gradient` and `inputGradient`.",
    ),
    (
        re.compile(
            r"\b(?:ValueAndGradient|valueAndGradient|LossAndGradients|lossAndGradients|"
            r"VjpResult|LossAndGradient|lossAndGradient)\b"
        ),
        "public docs should use `grad ... (value := true)` and destructure multi-value results.",
    ),
    (
        re.compile(r"\b(?:gradAndValue|stepWithLoss|stepWithGradients)\b"),
        "public docs should use one base operation with named options: "
        "`grad ... (value := true)`, `step ... (loss := true)`, or `update`.",
    ),
    (
        re.compile(
            r"\b(?:TrainReport|LossEndpoints|initialValue|finalValue|initialLoss|finalLoss)\b"
        ),
        "public docs should describe training loss through `Training.LossProgress` and its "
        "`.before` / `.after` fields.",
    ),
    (
        re.compile(r"\bloss[01]="),
        "public docs should render training progress as `loss=before -> after`.",
    ),
    (
        re.compile(
            r"\b(?:takeFlagValueOnce|takeFlagValueDefault|takeRequiredFlagValue|"
            r"takeParsedFlagDefault|takeBoolFlagOnce|takePositionalDefault|"
            r"takeNatFlagOnce|takeNatFlagDefault|takeFloatFlagDefault|"
            r"takeRequiredFloatFlag|takeBoolValueFlagDefault|takeSwitchDefault|"
            r"takePathFlagOnce|takePathFlagDefault|takeRequiredPathFlag|"
            r"takeStepsFlagDefault)\b"
        ),
        "public docs should use `takeX?` for optional values, `takeX (default := ...)` for "
        "defaults, and `requireX` for required values.",
    ),
    (
        re.compile(
            r"\b(?:writeLogTo|writeLossComparisonTo)\b|"
            r"\bLogDestination\.(?:parseValue\b|parse\?(?![A-Za-z0-9_])|pathD\b)"
        ),
        "public docs should use the destination-based `writeLog` / `writeLossComparison` "
        "operations and `LogDestination.parse`, `resolve`, or `path?`.",
    ),
    (
        re.compile(
            r"\b(?:resolvedLogPath|nextEpochWith|collectRolloutSessionWith|"
            r"collectRolloutCheckedSessionWith|collectRolloutWith|collectRolloutNativeWith)\b|"
            r"\bLogDestination\.path(?!\?)\b"
        ),
        "public docs should keep one log destination, use `mapNextEpoch`, and name rollout "
        "sources explicitly with `collectRolloutFrom...`.",
    ),
    (
        re.compile(
            r"\b(?:ProgramWithNatInputs|natInputShapes|validateNatInputs|"
            r"NatVecRef|inputNatVec|getNatVec|setNatVec)\b"
        ),
        "public docs should describe discrete data through typed data references, not "
        "implementation-specific input packs.",
    ),
    (
        re.compile(r"\bNN\.API\.[a-z][A-Za-z0-9_]*\b"),
        "public docs should use the exported `TorchLean.*` namespace, not a lower-case "
        "`NN.API.*` implementation namespace.",
    ),
    (
        re.compile(r"\bTensor\b[^\n]*(?:leading\s*\+\+|spatial\.toList)"),
        "public docs should describe tensor dimensions with list-shaped types or named shape "
        "accessors, not expanded internal shape expressions.",
    ),
]

DOC_PROSE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:In )?[Tt]his (?:chapter|section|page|guide) "
            r"(?:covers|discusses|examines|explores|provides|will cover|will discuss|"
            r"will examine|will explore)\b"
        ),
        "start with the subject instead of meta prose about what the document will cover.",
    ),
    (
        re.compile(r"\b(?:It is|It's) (?:important|worth) to note\b", flags=re.IGNORECASE),
        "state the relevant fact directly instead of announcing that it is important.",
    ),
]

PUBLIC_WEBSITE_API_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\{[A-Za-z][A-Za-z0-9_']*\s*:\s*Spec\.Shape\}"),
        "website examples should use inferred or list-shaped tensor dimensions, not an explicit "
        "internal `Spec.Shape` binder.",
    ),
]

TORCHLEAN_SOURCE_LINK_RE = re.compile(
    r"https://github\.com/lean-dojo/TorchLean/blob/main/([^\s\)\]`]+)"
)

DOCGEN_API_LINK_RE = re.compile(
    r"""(?:['"(])(/docs/[A-Za-z0-9_./-]+\.html(?:#[A-Za-z0-9_'.:-]+)?)"""
)

LOCAL_SOURCE_REF_RE = re.compile(
    r"`((?:NN|blueprint|home_page|scripts|csrc)/[^`\s]+"
    r"\.(?:lean|md|py|json|sh|cu|c|h|yml|yaml))`"
)

LEAN_DOC_COMMENT_RE = re.compile(r"/-(?:!|-).*?-/", flags=re.DOTALL)
# A single backslash starts the delimiters that MD4Lean does not recognize.
# The negative lookbehind leaves TeX line breaks such as `\\[1ex]` alone.
DOCGEN_UNSUPPORTED_MATH_DELIMITER_RE = re.compile(r"(?<!\\)\\[\(\[]")
DOCGEN_DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
DOCGEN_MARKDOWN_LIST_IN_DISPLAY_MATH_RE = re.compile(
    r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+",
    flags=re.MULTILINE,
)
VERSO_TEX_IN_ORDINARY_CODE_RE = re.compile(
    r"(?<!\$)`[^`\n]*(?:\\[A-Za-z]+|_\{[^}`]+\})[^`\n]*`"
)
FORMALIZATION_MATH_IN_ORDINARY_CODE_RE = re.compile(
    r"(?<!\$)`[^`\n]*(?:\s[\^+*/<>]=?\s|≤|≥|±)[^`\n]*`"
)

PUBLIC_EXAMPLE_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Models/",
    "NN/Examples/Data/",
    "NN/Examples/Factorization/",
    "NN/Examples/Functional/",
    "NN/Examples/Interop/",
    "NN/Examples/Support",
)

PUBLIC_TUTORIAL_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Data/",
)

PUBLIC_NUMERICAL_EXAMPLE_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Models/",
    "NN/Examples/Data/",
    "NN/Examples/Functional/",
    "NN/Examples/Factorization/",
    "NN/Examples/Interop/",
)

PUBLIC_NUMERICAL_SPEC_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bSpec\.(?:fill|mseSpec|getSpec|toScalarSpec|qrQSpec|qrRSpec|qrSpec|"
            r"choleskySpec|linearSpec|matVecMulSpec|matMulSpec|"
            r"convOutSpatial|poolOutSpatialPad)\b|"
            r"\bActivation\.(?:reluSpec|sigmoidSpec|tanhSpec)\b|"
            r"\b(?:Spec\.)?Tensor\.(?:addSpec|subSpec|mulSpec|divSpec|scaleSpec)\b"
        ),
        "runnable numerical examples should call the public `Tensor.*` or `nn.*` operation; "
        "reserve direct computational `Spec.*` calls for proof and specification examples.",
    ),
    (
        re.compile(r"^\s*def\s+[A-Z]\s*(?::|:=)", flags=re.MULTILINE),
        "runnable numerical examples should use descriptive lowerCamelCase names for tensor values; "
        "single-letter uppercase names are reserved for mathematical prose and type-level names.",
    ),
    (
        re.compile(r"\[[^\]\n]+\]!"),
        "runnable numerical examples should use bounded tensor indices or checked container lookup, "
        "not forced indexing.",
    ),
    (
        re.compile(r"\bunreachable!"),
        "runnable numerical examples should report invalid runtime input explicitly, not use "
        "`unreachable!`.",
    ),
]

# Root-driven Lake targets that collectively typecheck maintained modules outside the dedicated
# example and test libraries. Keep this list aligned with `lakefile.lean`.
LEAN_TYPECHECK_ROOTS = {
    "NN",
    "NN.CI.All",
    "NN.CI.SlowProofs",
    "NN.Docs",
    "NN.Verification.Main",
}

LEAN_TYPECHECK_GLOB_PREFIXES = (
    "NN/Examples/",
    "NN/Tests/",
)

# The tensor compiler's internal language uses Lean vectors for compiler indices,
# proof-recursive lists for syntax.
# Public numerical-container policies apply at `NN.Tensor`, not inside this
# implementation namespace.
TENSOR_INTERNAL_PREFIX = "NN/Tensor/Internal/"
TENSOR_VECTOR_BOUNDARY_FILES = {
    "NN/Tensor/Conversion.lean",
    "NN/Tests/Tensor/Storage.lean",
}

PUBLIC_GUIDE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bScalarShape\b|shape!|\bTensor\.T\b"),
        "public guides should write `Tensor α [dims]` and list-shaped model/dataset types.",
    ),
    (
        re.compile(r"\bSpec\.Tensor\.(?:dim|scalar|vecGet|vector|matrix)\b"),
        "public guides should use `Tensor` constructors and general indexing, not the recursive spec representation.",
    ),
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
        re.compile(
            r"\bRuntime(?:Fit|Train)\b|\bparseRuntime(?:Fit|Train)\b|"
            r"\bparsed\.(?:fit|trainOptions)\b"
        ),
        "quickstart docs should use internal `TrainingArgs`, `parseTrainingArgs`, and "
        "`parsed.options`.",
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
        "public guides should show `Trainer.new` / `trainer.train`, not removed command-wrapper names.",
    ),
    (
        re.compile(r"\bSupport\.Command\b|\bTrainer\.Command\b|\bTrainCommand\.run\b"),
        "public guides should teach `Trainer.new` / `trainer.train`; repository command glue belongs in examples.",
    ),
    (
        re.compile(r"\btrain\.\*"),
        "public guides should teach the `Trainer` API, not `train.*`.",
    ),
    (
        re.compile(r"\btrain\.stepEpochLR\b"),
        "public guides should say `Trainer.stepEpochLR`, not `train.stepEpochLR`.",
    ),
    (
        re.compile(r"\bNN\.API\.nn\b"),
        "public guides should use the canonical `TorchLean.nn` namespace.",
    ),
    (
        re.compile(r"\bNN\.API\.Models\.TrainFixed\b"),
        "fixed-sample training now lives at `TorchLean.Trainer.FixedSample`.",
    ),
]

PUBLIC_EXAMPLE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bdata/model_zoo/"),
        "runnable example artifacts live under `data/examples`; the removed ModelZoo name must "
        "not reappear.",
    ),
    (
        re.compile(r"^\s*def\s+(?:cfg|modelCfg|cfgFor)\b", flags=re.MULTILINE),
        "public example configuration values should use a descriptive name such as "
        "`modelConfig` or contextual `config`, not an abbreviated top-level name.",
    ),
    (
        re.compile(
            r"\b(?:observationShape|rolloutLeadingShape|rolloutStateShape|"
            r"rolloutLogitsShape|rolloutValueShape|actionLogitsShape|valueShape)\b"
        ),
        "public examples should use contextual tensor names such as `observation`, "
        "`rolloutStates`, `rolloutLogits`, and `value` instead of repeating `Shape`.",
    ),
    (
        re.compile(
            r"^\s*(?:def|abbrev)\s+(?:batchShape|latentShape|dataShape|scoreShape|obsShape)\b",
            flags=re.MULTILINE,
        ),
        "public example shape aliases should use the contextual tensor role directly, such as "
        "`batch`, `latent`, `data`, `score`, or `observation`.",
    ),
    (
        re.compile(r"\bScalarShape\b|shape!|\bTensor\.T\b"),
        "public examples should write `Tensor α [dims]` and list-shaped model/dataset types.",
    ),
    (
        re.compile(r"\bSpec\.Tensor\.(?:dim|scalar|vecGet|vector|matrix)\b"),
        "public examples should use `Tensor` constructors and general indexing, not the recursive spec representation.",
    ),
    (
        re.compile(r"\b(?:TorchLean\.)?Tensor\.Internal\b"),
        "public examples should use the ordinary `Tensor` API, not its physical representation.",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_])\.dim\b"),
        "public fixed-shape examples should write `Tensor α [dims]`; `.dim` is reserved "
        "for recursive shape implementations.",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9_])\.scalar\b|\b(?:Shape|Tensor)\.scalar\b"
        ),
        "public examples should write rank-zero tensors with shape `[]` and ordinary typed "
        "literals; recursive scalar constructors belong to implementation code.",
    ),
    (
        re.compile(r"(?<![0-9])\.[12]\b(?!\.)"),
        "public examples should destructure multi-value results instead of using anonymous `.1` "
        "and `.2` projections.",
    ),
    (
        re.compile(r"\b(?:stateGradient|stateCotangent|inputCotangent)\b"),
        "public examples should name destructured VJP values `gradient` and `inputGradient`.",
    ),
    (
        re.compile(
            r"\b(?:ValueAndGradient|valueAndGradient|LossAndGradients|lossAndGradients|"
            r"VjpResult|LossAndGradient|lossAndGradient)\b"
        ),
        "public examples should use `grad ... (value := true)` and destructure multi-value results.",
    ),
    (
        re.compile(r"\b(?:gradAndValue|stepWithLoss|stepWithGradients)\b"),
        "public examples should use one base operation with named options: "
        "`grad ... (value := true)`, `step ... (loss := true)`, or `update`.",
    ),
    (
        re.compile(
            r"\b(?:TrainReport|LossEndpoints|initialValue|finalValue|initialLoss|finalLoss)\b"
        ),
        "public examples should use `Training.LossProgress` and its `.before` / `.after` "
        "fields for training loss.",
    ),
    (
        re.compile(r"\bNN\.API\.nn\b"),
        "public examples should use the canonical `TorchLean.nn` namespace.",
    ),
    (
        re.compile(r"\bNN\.API\.Models\.TrainFixed\b"),
        "public examples should use `TorchLean.Trainer.FixedSample`.",
    ),
    (
        re.compile(r"\bsample\.Supervised\b"),
        "public examples should use `Sample.Supervised`, not the internal `sample.Supervised` spelling.",
    ),
    (
        re.compile(r"_root_\.NN\.API\.sample\.Supervised\b"),
        "public examples should use `Sample.Supervised`, not the fully-qualified internal sample type.",
    ),
    (
        re.compile(r"\bSupervisedSample\b"),
        "the `SupervisedSample` alias was removed; use the canonical `Sample.Supervised` type.",
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
        re.compile(r"\bShape\.(?:Vec|Mat|Image|Images|NCHW|vec|mat|image|images|nchw)\b"),
        "public examples should express fixed dimensions as lists or use `Shape.ofList` for computed dimensions, not domain- or layout-specific shape aliases.",
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
        re.compile(r"\bTrainer\.Manual\.trainLoaderWith\b"),
        "public examples should use `Trainer.RunConfig` + `Trainer.TrainOptions` with `trainer.train`, not `Trainer.Manual.trainLoaderWith`.",
    ),
    (
        re.compile(r"\bTrainer\.Manual\.logLossEvery\b"),
        "public examples should keep logging inline or use `Trainer.Report`; do not expose `Trainer.Manual.logLossEvery`.",
    ),
    (
        re.compile(r"\bfitWithParams\b"),
        "public examples should prefer the public trainer/verifier bridges instead of reopening raw post-training parameter callbacks.",
    ),
    (
        re.compile(r"\bModule\.instantiateConfigured\b"),
        "public examples should use `Module.instantiate` or the `Trainer` API, not `Module.instantiateConfigured` directly.",
    ),
    (
        re.compile(r"\bTorchLean\.Module\.run\b"),
        "public examples should use the `Trainer` API or `Module.Command.run`, not a removed raw `TorchLean.Module.run` dispatcher.",
    ),
    (
        re.compile(
            r"\b(?:TorchLean\.)?Module\.(?:loss|lossValue|gradState|lossAndGradState|"
            r"initOptimizer|optimizerStep(?:WithLoss)?|state)\b"
        ),
        "public model/example training should use `Trainer`; advanced manual code should keep "
        "objective operations under `Module.Objective`.",
    ),
    (
        re.compile(r"\.loadState\b"),
        "in-memory state replacement is `setState`; reserve `load` for checkpoints and external data.",
    ),
    (
        re.compile(r"\bTensor\.(?:pretty|print)\b"),
        "tensors already use Lean's standard `Repr`; use `#eval tensor` or "
        "`IO.println (reprStr tensor)` instead of a second display API.",
    ),
    (
        re.compile(
            r"\bRealData\.fit(CifarClassifierModel|CifarRegressionModel|CsvRegressionModel|HouseholdPowerRegressionModel)\b"
        ),
        "public model examples should use the shared example `TrainCommand` runners, not the old `*Model` wrappers.",
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
        "public examples should use `Trainer.new` / `trainer.train` or the shared example `TrainCommand` runners, not removed command-wrapper names.",
    ),
    (
        re.compile(r"\bTrainer\.Command\b"),
        "repository command glue belongs under `NN.Examples.Support`, outside the public Trainer namespace.",
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
        "`Trainer.FitSummary` was removed; public examples should use `Trainer.Report`.",
    ),
    (
        re.compile(r"\bTrainer\.(regression|classifier|crossEntropy|custom)\b"),
        "public examples should use `Trainer.new ... { task := ... }`; specialized `Trainer.*` constructors are removed.",
    ),
    (
        re.compile(r"\bTrainer\.(Regression|Classifier|OneHotCrossEntropy|Custom)(\.|\b)"),
        "public examples should stay on the unified `Trainer` API, not specialized trainer implementation handles.",
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
        re.compile(
            r"\bRuntime(?:Fit|Train)\b|\bparseRuntime(?:Fit|Train)\b|"
            r"\bparsed\.(?:fit|trainOptions)\b"
        ),
        "quickstart examples should use internal `TrainingArgs`, `parseTrainingArgs`, and "
        "`parsed.options`.",
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
        "public examples should use `Trainer.RunConfig.ofRuntimeOptions opts { optimizer := ... }`, not a type-ascribed RunConfig followed by `.withRuntimeOptions opts`.",
    ),
    (
        re.compile(
            r"\b(execution := opts\.execution|device := opts\.device|"
            r"backendProfile\? := opts\.backendProfile\?|device := if opts\.usesCuda|"
            r"fastKernels := opts\.fastKernels|"
            r"fastGpuMatmulPrecision := opts\.fastGpuMatmulPrecision)\b"
        ),
        "public examples should use `Trainer.RunConfig.ofRuntimeOptions opts { ... }` instead of manually copying runtime fields from `opts`.",
    ),
    (
        re.compile(r"\bnn\.(mseScalarModuleDef|crossEntropyOneHotScalarModuleDef)\b"),
        "public examples should use public `Module.instantiate*` helpers instead of spelling raw objective definitions.",
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
        re.compile(r"^\s*(public\s+)?import\s+NN\s*$", flags=re.MULTILINE),
        "public examples should import the focused `NN.API`, not the complete `NN` umbrella.",
    ),
    (
        re.compile(
            r"^(?!\s*(?:public\s+)?import\s+NN\.API\.).*\bNN\.API\.",
            flags=re.MULTILINE,
        ),
        "public examples should go through the `TorchLean` API, not fully-qualified `NN.API.*` implementation paths.",
    ),
    (
        re.compile(r"IO\.println\s+\"model\s*="),
        "public examples should use the structured model summary instead of a hardcoded model banner.",
    ),
    (
        re.compile(r"\bIO\.println\s+trainer\.summary\b"),
        "public examples should use `trainer.printSummary` so model-summary formatting stays consistent.",
    ),
    (
        re.compile(r"\bIO\.println\s+\w*Trainer\.summary\b"),
        "public examples should use `trainer.printSummary`, not direct model-summary printing.",
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
        "public stream examples should use `trained.predict` / `trained.predictMany`; do not expose internal training state.",
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
        "public examples should use `Trainer.Report.numeric?` or `requireNumeric` for numeric metrics.",
    ),
    (
        re.compile(r"\bTrainer\.FitSummary\.requireFloatLosses\b"),
        "public examples should use `Trainer.Report.requireNumeric` or `printNumeric` for numeric metrics.",
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

PUBLIC_DECL_RE = re.compile(
    r"^\s*(?:public\s+)?(?:def|opaque|structure|inductive|class|abbrev|theorem|lemma)\s+"
    r"(?P<name>[A-Za-z0-9_'.]+)\b",
    flags=re.MULTILINE,
)

# Public tensor and model APIs describe axes through shapes and rank-one tensors. Layout spellings and fixed
# spatial ranks belong in low-level kernels or domain examples, not in user-facing declaration names.
PUBLIC_LAYOUT_NAME_RE = re.compile(
    r"(?:[123][dD](?=[A-Z_]|$)|(?:One|Two|Three)D(?=[A-Z_]|$)|"
    r"(?:^|_)(?:chw|nchw|nhwc|hwc)(?:_|$))"
)

PUBLIC_IMPORT_RE = re.compile(
    r"^\s*public\s+import\s+(?P<module>[A-Za-z0-9_.]+)\s*$",
    flags=re.MULTILINE,
)

# Import-only API umbrellas should compose focused public modules. Re-exporting these implementation
# roots makes runtime internals part of the user API by accident.
BROAD_LOW_LEVEL_IMPORTS = {
    "NN",
    "NN.Proofs",
    "NN.Runtime",
    "NN.Spec",
    "NN.Verification",
    "NN.Runtime.Autograd.Model",
}

BROAD_LOW_LEVEL_IMPORT_PREFIXES = (
    "NN.Runtime.Autograd.Engine.",
    "NN.Runtime.Autograd.Torch.Core.",
    "NN.Spec.Core.Tensor.Internal.",
)

CONTRACT_SOURCE_FILE_RE = re.compile(
    r"\.sourceFile\s*\{(?P<body>[^{}]*)\}",
    flags=re.DOTALL,
)
CONTRACT_NATIVE_SYMBOL_RE = re.compile(
    r"\.nativeSymbol\s*\{(?P<body>[^{}]*)\}",
    flags=re.DOTALL,
)
CONTRACT_GUARD_SOURCE_PATH_RE = re.compile(
    r"\.runtimeGuard\s+\"[^\"]*\.(?:c|cc|cpp|cu|cuh|h|hpp)\"",
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
    """Yield tracked and non-ignored project Lean sources without crawling build trees."""
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "*.lean",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("repo lint requires Git to enumerate project sources") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"failed to enumerate project sources with Git: {message}") from error

    for relative_bytes in result.stdout.split(b"\0"):
        if not relative_bytes:
            continue
        relative = pathlib.Path(relative_bytes.decode("utf-8", errors="surrogateescape"))
        if any(directory in relative.parts for directory in VENDORED_DIR_NAMES):
            continue
        if "_out" in relative.parts:
            continue
        path = REPO_ROOT / relative
        if path.is_file():
            yield path

# `NN/Floats` is scheduled for extraction into its own package and is frozen until then; the
# style rules below therefore apply to everything else.
FROZEN_STYLE_PREFIXES = ("NN/Floats/",)

MAX_LINE_LENGTH = 100


def _check_line_style(path: pathlib.Path, rel: str, text: str, findings: list[Finding]) -> None:
    """Mathlib-style line rules: at most 100 columns and no em-dashes in prose."""
    if rel.startswith(FROZEN_STYLE_PREFIXES):
        return
    for lineno, line in enumerate(text.split("\n"), start=1):
        # Verso block directives (`:::theorem "label" (lean := "...")`) must stay on one line, so
        # the column limit does not apply to them.
        if line.lstrip().startswith(":::"):
            continue
        if len(line) > MAX_LINE_LENGTH:
            findings.append(
                Finding("ERROR", path, lineno, MAX_LINE_LENGTH + 1,
                        f"line exceeds {MAX_LINE_LENGTH} characters; wrap it.")
            )
        if "\u2014" in line:
            findings.append(
                Finding("ERROR", path, lineno, line.index("\u2014") + 1,
                        "em-dash in source; use a comma, colon, parentheses, or a new sentence.")
            )


def _check_lean_target_coverage(findings: list[Finding]) -> None:
    """Require every maintained `NN` module to belong to a typecheck target."""

    nn_files = {
        path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", "."): path
        for path in _iter_lean_files()
        if path == REPO_ROOT / "NN.lean" or path.is_relative_to(REPO_ROOT / "NN")
    }
    import_re = re.compile(
        r"^\s*(?:public\s+)?(?:meta\s+)?import\s+([A-Za-z0-9_.]+)",
        flags=re.MULTILINE,
    )
    imports: dict[str, list[str]] = {}
    for module, path in nn_files.items():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        masked = _mask_lean_comments_and_strings(text)
        imports[module] = [name for name in import_re.findall(masked) if name in nn_files]

    covered: set[str] = set()
    pending = list(LEAN_TYPECHECK_ROOTS)
    while pending:
        module = pending.pop()
        if module in covered or module not in nn_files:
            continue
        covered.add(module)
        pending.extend(imports.get(module, []))

    for module, path in sorted(nn_files.items()):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if module in covered or rel.startswith(LEAN_TYPECHECK_GLOB_PREFIXES):
            continue
        findings.append(
            Finding(
                "ERROR",
                path,
                None,
                None,
                "maintained Lean module is not reachable from `NN`, `NNCI`, `NNSlowProofs`, "
                "`TorchLeanDocs`, or an executable root, and is not covered by the `NNExamples` "
                "or `NNTests` globs.",
            )
        )


def _iter_source_files(
    root: pathlib.Path, suffix: str, *, exclude_dirs: Iterable[str] = ()
) -> Iterable[pathlib.Path]:
    """Yield authored files, including untracked sources, without entering generated trees."""

    excluded_names = SOURCE_EXCLUDED_DIR_NAMES | set(exclude_dirs)
    excluded_paths = {REPO_ROOT / relative for relative in GENERATED_DOC_DIRS}
    if root.name in excluded_names or root in excluded_paths:
        return
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        parent = pathlib.Path(directory)
        directories[:] = sorted(
            name for name in directories
            if name not in excluded_names and parent / name not in excluded_paths
        )
        for name in sorted(files):
            if name.endswith(suffix):
                yield parent / name


def _iter_authored_public_docs() -> Iterable[pathlib.Path]:
    """Yield maintained guide and website sources, excluding generated and vendored trees."""

    yield REPO_ROOT / "README.md"
    yield from (REPO_ROOT / "docs").glob("*.md")
    yield from _iter_source_files(REPO_ROOT / "home_page/blueprint/TorchLeanBlueprint", ".lean")
    yield from _iter_source_files(
        REPO_ROOT / "home_page", ".md", exclude_dirs={"blueprint", "docs"}
    )


def _iter_doc_fact_paths() -> Iterable[pathlib.Path]:
    """Yield guide, website, blueprint, and source-local documentation for factual checks."""

    yield from REPO_ROOT.glob("README.md")
    yield from (REPO_ROOT / "docs").glob("*.md")
    yield from _iter_source_files(REPO_ROOT / "home_page/blueprint", ".lean")
    for relative in ("home_page", "NN", "scripts"):
        yield from _iter_source_files(REPO_ROOT / relative, ".md")


def _normalized_prose_paragraphs(text: str) -> Iterable[tuple[str, int]]:
    """Yield long prose paragraphs suitable for exact-duplication checks."""

    offset = 0
    for paragraph in re.split(r"\n\s*\n", text):
        start = text.find(paragraph, offset)
        offset = start + len(paragraph)
        normalized = " ".join(line.strip() for line in paragraph.splitlines())
        if len(normalized) < 240:
            continue
        if any(marker in paragraph for marker in ("```", ":::", "https://", ":=")):
            continue
        if normalized.startswith(("import ", "public import ", "#", "<")):
            continue
        yield normalized, start


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


def _mask_verso_prose(text: str) -> str:
    """Preserve Lean examples in a `#doc` body without treating its prose as Lean code."""

    doc = re.search(r"^#doc\b[^\n]*=>[ \t]*\n", text, flags=re.MULTILINE)
    if doc is None:
        return text
    out = [text[:doc.end()]]
    fence: str | None = None
    lean_block = False
    for line in text[doc.end():].splitlines(keepends=True):
        marker = re.match(r"^[ \t]*(`{3,}|~{3,})(.*?)[\r\n]*$", line)
        if marker is not None and fence is None:
            fence = marker.group(1)
            language = marker.group(2).strip().split()
            lean_block = not language or language[0] in {"lean", "leanTerm", "leanInit"}
        elif (
            marker is not None
            and fence is not None
            and marker.group(1)[0] == fence[0]
            and len(marker.group(1)) >= len(fence)
            and not marker.group(2).strip()
        ):
            fence = None
            lean_block = False
        elif lean_block:
            out.append(line)
            continue
        # Inline Lean roles also elaborate terms; retain them for the banned-construct checks.
        masked = list(re.sub(r"[^\r\n]", " ", line))
        if fence is None:
            for role in re.finditer(r"\{lean(?:\s[^}\n]*)?\}(`+)(.*?)\1", line):
                start, end = role.span(2)
                masked[start:end] = line[start:end]
        out.append("".join(masked))
    return "".join(out)


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


def _internal_namespace_lines(masked: str) -> set[int]:
    """Return lines nested under a namespace with an `Internal` component.

    This lightweight namespace scan is deliberately narrower than a Lean parser. It only supports
    the ordinary one-line `namespace`, `section`, and `end` forms used by TorchLean source files.
    """

    frames: list[tuple[str, tuple[str, ...]]] = []
    internal_lines: set[int] = set()
    namespace_re = re.compile(r"^\s*namespace\s+([A-Za-z0-9_.]+)\s*$")
    section_re = re.compile(r"^\s*(?:@\[[^]]+\]\s*)?(?:public\s+)?section(?:\s+([A-Za-z0-9_]+))?\s*$")
    end_re = re.compile(r"^\s*end(?:\s+([A-Za-z0-9_.]+))?\s*$")

    for line_number, line in enumerate(masked.splitlines(), start=1):
        if any(kind == "namespace" and "Internal" in components
               for kind, components in frames):
            internal_lines.add(line_number)

        if match := namespace_re.match(line):
            frames.append(("namespace", tuple(match.group(1).split("."))))
            continue
        if match := section_re.match(line):
            name = (match.group(1),) if match.group(1) else ()
            frames.append(("section", name))
            continue
        if match := end_re.match(line):
            name = match.group(1)
            if name is None:
                if frames:
                    frames.pop()
                continue
            final_component = name.rsplit(".", 1)[-1]
            for index in range(len(frames) - 1, -1, -1):
                if frames[index][1] and frames[index][1][-1] == final_component:
                    del frames[index:]
                    break

    return internal_lines


def _check_local_source_refs(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check backtick-quoted local source paths in authored docs/comments."""

    for m in LOCAL_SOURCE_REF_RE.finditer(text):
        raw_target = m.group(1).split("#", 1)[0]
        if any(marker in raw_target for marker in ("*", "<", ">", "...")):
            continue
        target = pathlib.Path(urllib.parse.unquote(raw_target))
        if target.is_absolute():
            continue
        if not (REPO_ROOT / target).exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"dead local source reference: `{raw_target}` does not exist in this checkout.",
                )
            )


def _check_docgen_api_links(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check website links to generated API pages against the corresponding Lean source."""

    for m in DOCGEN_API_LINK_RE.finditer(text):
        url = urllib.parse.unquote(m.group(1))
        module_path = url.removeprefix("/docs/").split("#", 1)[0].removesuffix(".html")
        target = REPO_ROOT / f"{module_path}.lean"
        if not target.exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"dead generated API link: `/docs/{module_path}.html` has no `{module_path}.lean` source.",
                )
            )


def _check_lean_doc_math(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Reject documentation math that DocGen cannot pass intact to MathJax."""

    for comment in LEAN_DOC_COMMENT_RE.finditer(text):
        for match in DOCGEN_UNSUPPORTED_MATH_DELIMITER_RE.finditer(comment.group()):
            offset = comment.start() + match.start()
            line, col = _line_col(text, offset)
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "DocGen does not preserve this math delimiter; use `$...$` or `$$...$$`.",
                )
            )
        for display in DOCGEN_DISPLAY_MATH_RE.finditer(comment.group()):
            for match in DOCGEN_MARKDOWN_LIST_IN_DISPLAY_MATH_RE.finditer(display.group(1)):
                offset = comment.start() + display.start(1) + match.start()
                line, col = _line_col(text, offset)
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "a display-math line starts like a Markdown list item; move the operator "
                        "to the preceding TeX line so DocGen keeps the equation together.",
                    )
                )


def _check_verso_math_roles(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Catch mathematical TeX that would remain an ordinary monospace code span."""

    patterns = [VERSO_TEX_IN_ORDINARY_CODE_RE]
    if path.is_relative_to(REPO_ROOT / "home_page/blueprint/TorchLeanBlueprint/FormalizationMap"):
        patterns.append(FORMALIZATION_MATH_IN_ORDINARY_CODE_RE)
    for pattern in patterns:
        for match in pattern.finditer(text):
            line, col = _line_col(text, match.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "mathematical prose is in an ordinary code span; use a Verso `$` math role.",
                )
            )


def _lean_string_field(body: str, field: str) -> str | None:
    m = re.search(rf"\b{re.escape(field)}\s*:=\s*\"([^\"]+)\"", body)
    return m.group(1) if m else None


def _lean_optional_string_field(body: str, field: str) -> str | None:
    m = re.search(rf"\b{re.escape(field)}\s*:=\s*some\s+\"([^\"]+)\"", body)
    return m.group(1) if m else None


def _lake_declares_target(lake_text: str, target: str) -> bool:
    return re.search(
        rf"^\s*(?:target|lean_exe|lean_lib)\s+{re.escape(target)}\b",
        lake_text,
        flags=re.MULTILINE,
    ) is not None


def _check_backend_contract_refs(
    path: pathlib.Path,
    text: str,
    lake_text: str,
    findings: list[Finding],
) -> None:
    """Check structured backend contract references to local sources and native symbols."""

    for m in CONTRACT_GUARD_SOURCE_PATH_RE.finditer(text):
        line, col = _line_col(text, m.start())
        findings.append(
            Finding(
                "ERROR",
                path,
                line,
                col,
                "native source paths belong in structured `.sourceFile` or `.nativeSymbol` provenance, not a runtime-guard label.",
            )
        )

    for m in CONTRACT_SOURCE_FILE_RE.finditer(text):
        body = m.group("body")
        raw_path = _lean_string_field(body, "path")
        if raw_path is None:
            line, col = _line_col(text, m.start())
            findings.append(Finding("ERROR", path, line, col, "`.sourceFile` provenance is missing `path := ...`."))
            continue
        source = REPO_ROOT / raw_path
        if not source.exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.sourceFile` provenance points to missing source `{raw_path}`.",
                )
            )

    for m in CONTRACT_NATIVE_SYMBOL_RE.finditer(text):
        body = m.group("body")
        raw_path = _lean_string_field(body, "path")
        symbol = _lean_string_field(body, "symbol")
        build_target = _lean_optional_string_field(body, "buildTarget?")
        line, col = _line_col(text, m.start())
        if raw_path is None:
            findings.append(Finding("ERROR", path, line, col, "`.nativeSymbol` provenance is missing `path := ...`."))
            continue
        if symbol is None:
            findings.append(Finding("ERROR", path, line, col, "`.nativeSymbol` provenance is missing `symbol := ...`."))
            continue
        source = REPO_ROOT / raw_path
        if not source.exists():
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance points to missing source `{raw_path}`.",
                )
            )
            continue
        try:
            source_text = source.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            findings.append(Finding("ERROR", source, None, None, f"failed to read file: {e}"))
            continue
        if symbol not in source_text:
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance names `{symbol}`, but it does not occur in `{raw_path}`.",
                )
            )
        if build_target is not None and not _lake_declares_target(lake_text, build_target):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance names Lake target `{build_target}`, but `lakefile.lean` does not declare it.",
                )
            )


# --- Compiled docstring examples ----------------------------------------------------------------
#
# Every `Example:` block in an API docstring is generated from a snippet that lives in a mirror
# module under `NN/Tests/API/DocExamples/`, where the compiler checks it on every build. Copying an
# example out of a docstring is only useful if the example still elaborates, and the way to keep
# that true is to make the docstring a copy of something that gets compiled rather than prose that
# nobody rechecks. This check compares the two and refuses to let them drift; the writing direction
# is `--sync-doc-examples`, mirror module first, docstring second.

DOC_EXAMPLE_MIRROR_DIR = "NN/Tests/API/DocExamples"
DOC_EXAMPLE_MARKER = re.compile(r"^--\s*doc-example:\s*(\S+)\s*::\s*(.+?)\s*$")
DOC_EXAMPLE_SYNC_HINT = "run `python3 scripts/checks/repo_lint.py --sync-doc-examples`"
# An anchor may only stop at a boundary, so `def train` never matches `def trainStream`.
DOC_EXAMPLE_NAME_CHARS = set("!'?_")


@dataclass
class DocExample:
    """One compiled snippet and the declaration whose docstring should carry it."""

    mirror: pathlib.Path
    mirror_line: int
    target: pathlib.Path
    target_rel: str
    anchor: str
    snippet: list[str]


def _strip_blank_edges(lines: list[str]) -> list[str]:
    """Drop leading and trailing blank lines, keeping the interior spacing intact."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _collect_doc_examples(findings: list[Finding]) -> list[DocExample]:
    """Read every `doc-example:` marker and the namespace body that follows it."""
    mirror_root = REPO_ROOT / DOC_EXAMPLE_MIRROR_DIR
    if not mirror_root.is_dir():
        return []
    examples: list[DocExample] = []
    for path in sorted(mirror_root.rglob("*.lean")):
        lines = path.read_text(encoding="utf-8").split("\n")
        index = 0
        while index < len(lines):
            match = DOC_EXAMPLE_MARKER.match(lines[index].strip())
            if match is None:
                index += 1
                continue
            marker_line = index + 1
            target_rel, anchor = match.group(1), match.group(2)
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            opener = lines[cursor] if cursor < len(lines) else ""
            if not opener.startswith("namespace ") or len(opener.split()) != 2:
                findings.append(
                    Finding("ERROR", path, marker_line, None,
                            "a doc-example marker must be followed by `namespace <Name>` holding "
                            "the snippet.")
                )
                index = cursor + 1
                continue
            closer = f"end {opener.split()[1]}"
            body_start = cursor + 1
            body_end = body_start
            while body_end < len(lines) and lines[body_end].rstrip() != closer:
                body_end += 1
            if body_end >= len(lines):
                findings.append(
                    Finding("ERROR", path, marker_line, None,
                            f"doc-example namespace is never closed with `{closer}`.")
                )
                break
            target = REPO_ROOT / target_rel
            snippet = _strip_blank_edges(lines[body_start:body_end])
            if not target.is_file():
                findings.append(
                    Finding("ERROR", path, marker_line, None,
                            f"doc-example target does not exist: `{target_rel}`.")
                )
            elif not snippet:
                findings.append(
                    Finding("ERROR", path, marker_line, None,
                            "doc-example namespace is empty; write the snippet inside it.")
                )
            else:
                examples.append(
                    DocExample(path, marker_line, target, target_rel, anchor, snippet)
                )
            index = body_end + 1
    return examples


def _anchor_matches(line: str, anchor: str) -> bool:
    """Match an anchor as a whole prefix: the character after it must end an identifier."""
    if not line.startswith(anchor):
        return False
    if len(line) == len(anchor):
        return True
    following = line[len(anchor)]
    return not (following.isalnum() or following in DOC_EXAMPLE_NAME_CHARS)


def _docstring_span(lines: list[str], declaration: int) -> tuple[int, int] | None:
    """Locate the docstring directly above a declaration, skipping attribute lines."""
    cursor = declaration - 1
    while cursor >= 0 and lines[cursor].lstrip().startswith("@["):
        cursor -= 1
    if cursor < 0 or not lines[cursor].rstrip().endswith("-/"):
        return None
    end = cursor
    if lines[end].lstrip().startswith("/--"):
        return (end, end)
    start = end - 1
    while start >= 0 and not lines[start].lstrip().startswith("/--"):
        if lines[start].rstrip().endswith("-/"):
            return None
        start -= 1
    return None if start < 0 else (start, end)


def _fenced_example(body: list[str]) -> tuple[int, int] | None:
    """Return the line range of an existing `Example:` block inside a docstring body."""
    for index, line in enumerate(body):
        if line.strip() != "Example:":
            continue
        if index + 1 >= len(body) or body[index + 1].strip() != "```lean":
            return None
        closing = index + 2
        while closing < len(body) and body[closing].strip() != "```":
            closing += 1
        return None if closing >= len(body) else (index, closing)
    return None


def _render_docstring(block: list[str], snippet: list[str]) -> list[str]:
    """Rewrite a docstring so its `Example:` block is exactly `snippet`."""
    indent = block[0][: len(block[0]) - len(block[0].lstrip())]
    example = ["Example:", "```lean", *snippet, "```"]
    if len(block) == 1:
        # A one-line docstring grows into the multi-line form to make room for the example.
        summary = block[0].strip()
        body = [summary[len("/--"):-len("-/")].strip()]
    else:
        body = [line[len(indent):] if line.startswith(indent) else line.lstrip()
                for line in block[1:-1]]
        found = _fenced_example(body)
        if found is not None:
            body = body[: found[0]] + example + body[found[1] + 1:]
            example = []
    if example:
        body = _strip_blank_edges(body) + [""] + example
    rendered = ["/--", *body, "-/"]
    return [indent + line if line else line for line in rendered]


def _sync_doc_examples(*, write: bool) -> tuple[list[Finding], list[str]]:
    """Compare (or rewrite) every docstring `Example:` block against its compiled mirror."""
    findings: list[Finding] = []
    updates: list[str] = []
    by_target: dict[pathlib.Path, list[DocExample]] = {}
    for example in _collect_doc_examples(findings):
        by_target.setdefault(example.target, []).append(example)

    for target, group in sorted(by_target.items()):
        lines = target.read_text(encoding="utf-8").split("\n")
        located: list[tuple[int, tuple[int, int], DocExample]] = []
        for example in group:
            hits = [index for index, line in enumerate(lines)
                    if _anchor_matches(line, example.anchor)]
            if not hits:
                findings.append(
                    Finding("ERROR", example.mirror, example.mirror_line, None,
                            f"doc-example anchor `{example.anchor}` is not a declaration in "
                            f"`{example.target_rel}`.")
                )
                continue
            if len(hits) > 1:
                findings.append(
                    Finding("ERROR", example.mirror, example.mirror_line, None,
                            f"doc-example anchor `{example.anchor}` matches "
                            f"{len(hits)} declarations in `{example.target_rel}`; extend it until "
                            "it names exactly one.")
                )
                continue
            span = _docstring_span(lines, hits[0])
            if span is None:
                findings.append(
                    Finding("ERROR", target, hits[0] + 1, None,
                            "a doc-example target needs its own docstring to carry the "
                            "`Example:` block.")
                )
                continue
            located.append((hits[0], span, example))

        changed = False
        # Rewrite from the bottom of the file upward so earlier spans keep their line numbers.
        for _, (start, end), example in sorted(located, key=lambda item: -item[0]):
            block = lines[start:end + 1]
            rendered = _render_docstring(block, example.snippet)
            if rendered == block:
                continue
            if write:
                lines[start:end + 1] = rendered
                updates.append(f"{example.target_rel}: {example.anchor}")
                changed = True
            else:
                mirror_rel = example.mirror.relative_to(REPO_ROOT).as_posix()
                findings.append(
                    Finding("ERROR", target, start + 1, None,
                            f"the `Example:` block for `{example.anchor}` does not match its "
                            f"compiled snippet in `{mirror_rel}`; {DOC_EXAMPLE_SYNC_HINT}.")
                )
        if changed:
            target.write_text("\n".join(lines), encoding="utf-8")

    return findings, updates


def _check_doc_examples(findings: list[Finding]) -> None:
    """Fail when a docstring example no longer matches the snippet the compiler checks."""
    drift, _ = _sync_doc_examples(write=False)
    findings.extend(drift)



def lint_repo(*, fail_on_warn: bool) -> list[Finding]:
    """Run TorchLean's repository hygiene checks and return all findings."""
    findings: list[Finding] = []
    try:
        lake_text = (REPO_ROOT / "lakefile.lean").read_text(encoding="utf-8")
    except OSError as e:
        lake_text = ""
        findings.append(Finding("ERROR", REPO_ROOT / "lakefile.lean", None, None, f"failed to read file: {e}"))

    if not LINT_SCOPE_SENTINEL.exists():
        findings.append(
            Finding(
                "ERROR",
                LINT_SCOPE_SENTINEL,
                None,
                None,
                "repo linter is not rooted at TorchLean; expected to see "
                "NN/MLTheory/CROWN/Lyapunov/Certificate.lean.",
            )
        )

    _check_lean_target_coverage(findings)
    _check_doc_examples(findings)

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



    for env_var, rel_impl in DOCUMENTED_ENV_VAR_IMPLEMENTATIONS.items():
        docs_mention = False
        for path in _iter_authored_public_docs():
            try:
                if env_var in path.read_text(encoding="utf-8"):
                    docs_mention = True
                    break
            except OSError:
                continue
        if docs_mention:
            impl = REPO_ROOT / rel_impl
            try:
                impl_text = impl.read_text(encoding="utf-8")
            except OSError as e:
                findings.append(Finding("ERROR", impl, None, None, f"documented env var `{env_var}` has no readable implementation: {e}"))
                continue
            if env_var not in impl_text:
                findings.append(
                    Finding(
                        "ERROR",
                        impl,
                        None,
                        None,
                        f"documented env var `{env_var}` is not implemented in its declared producer helper.",
                    )
                )

    trust_file = REPO_ROOT / "docs/TRUST_BOUNDARIES.md"
    try:
        trust_text = trust_file.read_text(encoding="utf-8")
    except OSError as e:
        trust_text = ""
        findings.append(Finding("ERROR", trust_file, None, None, f"failed to read file: {e}"))

    for fq_name, (rel_source, decl_re) in TRUST_BOUNDARY_DECL_REFS.items():
        if fq_name not in trust_text:
            findings.append(
                Finding(
                    "ERROR",
                    trust_file,
                    None,
                    None,
                    f"trust-boundary declaration `{fq_name}` is missing from docs/TRUST_BOUNDARIES.md.",
                )
            )
        source = REPO_ROOT / rel_source
        try:
            source_text = source.read_text(encoding="utf-8")
        except OSError as e:
            findings.append(Finding("ERROR", source, None, None, f"failed to read file: {e}"))
            continue
        if not decl_re.search(source_text):
            findings.append(
                Finding(
                    "ERROR",
                    source,
                    None,
                    None,
                    f"docs/TRUST_BOUNDARIES.md cites `{fq_name}`, but the expected declaration was not found.",
                )
            )

    authored_public_docs = set(_iter_authored_public_docs())
    seen_prose: dict[str, tuple[pathlib.Path, int]] = {}
    for path in _iter_doc_fact_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if path.suffix != ".lean":
            _check_local_source_refs(path, text, findings)
            _check_docgen_api_links(path, text, findings)
        elif path.is_relative_to(REPO_ROOT / "home_page/blueprint/TorchLeanBlueprint"):
            _check_verso_math_roles(path, text, findings)
        for rx, msg in DOC_FACT_BANNED_PATTERNS:
            for m in rx.finditer(text):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))
        for rx, msg in PUBLIC_DOC_API_BANNED_PATTERNS:
            for m in rx.finditer(text):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))
        if path in authored_public_docs:
            for rx, msg in PUBLIC_GUIDE_BANNED_PATTERNS + DOC_PROSE_BANNED_PATTERNS:
                for m in rx.finditer(text):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))
            for paragraph, start in _normalized_prose_paragraphs(text):
                previous = seen_prose.get(paragraph)
                if previous is None:
                    seen_prose[paragraph] = (path, start)
                    continue
                previous_path, previous_start = previous
                line, col = _line_col(text, start)
                previous_line, _ = _line_col(
                    previous_path.read_text(encoding="utf-8"), previous_start
                )
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "repeats a long prose paragraph from "
                        f"`{previous_path.relative_to(REPO_ROOT)}:{previous_line}`; keep one "
                        "account and link or summarize it here.",
                    )
                )
            if path.suffix == ".md" and (
                path == REPO_ROOT / "README.md"
                or path.is_relative_to(REPO_ROOT / "home_page")
            ):
                for rx, msg in PUBLIC_WEBSITE_API_BANNED_PATTERNS:
                    for m in rx.finditer(text):
                        line, col = _line_col(text, m.start())
                        findings.append(Finding("ERROR", path, line, col, msg))
        for m in TORCHLEAN_SOURCE_LINK_RE.finditer(text):
            raw_target = m.group(1).split("#", 1)[0]
            target = urllib.parse.unquote(raw_target)
            if not (REPO_ROOT / target).exists():
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"dead TorchLean source link: `{raw_target}` does not exist in this checkout.",
                    )
                )

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

    for rel in sorted(REMOVED_COMPATIBILITY_PATHS):
        path = REPO_ROOT / rel
        if path.exists():
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "removed compatibility module has been restored; use the canonical API or subsystem import.",
                )
            )
    for prefix in REMOVED_COMPATIBILITY_PREFIXES:
        directory = REPO_ROOT / prefix
        if directory.exists() and any(directory.rglob("*.lean")):
            findings.append(
                Finding(
                    "ERROR",
                    directory,
                    None,
                    None,
                    "removed compatibility import tree has been restored; use the canonical subsystem umbrellas.",
                )
            )

    banned_regexes: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"\b(?:takeFlagValueOnce|takeFlagValueDefault|takeRequiredFlagValue|"
                r"takeParsedFlagDefault|takeBoolFlagOnce|takePositionalDefault|"
                r"takeNatFlagOnce|takeNatFlagDefault|takeFloatFlagDefault|"
                r"takeRequiredFloatFlag|takeBoolValueFlagDefault|takeSwitchDefault|"
                r"takePathFlagOnce|takePathFlagDefault|takeRequiredPathFlag|"
                r"takeStepsFlagDefault)\b"
            ),
            "removed CLI parser variant found; use `takeX?` for optional values, `takeX "
            "(default := ...)` for defaults, and `requireX` for required values.",
        ),
        (
            re.compile(
                r"\b(?:writeLogTo|writeLossComparisonTo)\b|"
                r"\bLogDestination\.(?:parseValue\b|parse\?(?![A-Za-z0-9_])|pathD\b)"
            ),
            "removed duplicate logging operation found; use the destination-based `writeLog` / "
            "`writeLossComparison` operations and `LogDestination.parse`, `resolve`, or `path?`.",
        ),
        (
            re.compile(
                r"\b(?:resolvedLogPath|nextEpochWith|collectRolloutSessionWith|"
                r"collectRolloutCheckedSessionWith|collectRolloutWith|collectRolloutNativeWith)\b|"
                r"\bLogDestination\.path(?!\?)\b"
            ),
            "removed duplicate or implementation-shaped API found; keep one log destination, use "
            "`mapNextEpoch`, and name rollout sources explicitly with `collectRolloutFrom...`.",
        ),
        (
            re.compile(r"\b(?:rowSoftmaxFwd|rowLogSoftmaxFwd)\b"),
            "obsolete value-only CUDA softmax adapters were removed; retain the workspace from "
            "`rowSoftmaxForward` or `rowLogSoftmaxForward` for explicit buffer ownership.",
        ),
        (
            re.compile(
                r"\b(?:packedResultOrPanic|LinearAlgebraImpl|ShapeChangeImpl|ReductionImpl|"
                r"appendTimeChannelImpl|bmmLikeSpec|"
                r"OperationImpl\.(?:linearLast|stack)|linearLast)\b"
            ),
            "removed implementation-shaped tensor or lowering helper found; keep recursive tensor "
            "workers private and make lowered closures return `Except` for checked results.",
        ),
        (
            re.compile(r"\b[A-Za-z0-9_]*TList[A-Za-z0-9_]*\b|\btlist!\b"),
            "the old tensor-list API was removed; use the canonical `TorchLean.TensorPack` API.",
        ),
        (re.compile(r"\bnative_decide\b"), "`native_decide` is banned in TorchLean."),
        (re.compile(r"\bsorry\b"), "`sorry` is banned in TorchLean sources."),
        (re.compile(r"\badmit\b"), "`admit` is banned in TorchLean sources."),
        (
            re.compile(r"\bnamespace Private\b|\bSpec\.Private\b"),
            "use a subsystem-scoped `Internal` namespace instead of a shared `Private` namespace.",
        ),
        (
            re.compile(r"\b(FitConfig|LoaderFitConfig|FitReport)\b"),
            "old lower training names are removed; use `TrainConfig`, `LoaderTrainConfig`, "
            "and `Training.LossProgress`.",
        ),
        (
            re.compile(r"\b(?:TrainReport|LossEndpoints)\b"),
            "duplicate training endpoint records are removed; use `Training.LossProgress`.",
        ),
        (
            re.compile(
                r"\b(?:stateGradient|stateCotangent|inputCotangent|initialValue|finalValue|"
                r"initialLoss|finalLoss)\b"
            ),
            "removed public result vocabulary found; use clear local gradient names or "
            "`loss.before` / `loss.after`.",
        ),
        (
            re.compile(r"\bpositiveDimensions\b"),
            "removed shape-helper vocabulary found; use `natValues` for a tensor's natural-number "
            "contents.",
        ),
        (
            re.compile(
                r"\b(?:ValueAndGradient|valueAndGradient|LossAndGradients|lossAndGradients|"
                r"VjpResult|LossAndGradient|lossAndGradient)\b"
            ),
            "removed autograd result vocabulary found; use `grad ... (value := true)` and "
            "destructure its result.",
        ),
        (
            re.compile(
                r"\bgradAndValue\b|"
                r"\bTorchLean\.Module\.Objective\.(?:diff|stepWithLoss|stepWithGradients)\b"
            ),
            "removed compound public operation found; use `grad` or `step` with named options, "
            "or `update` for an already-computed gradient.",
        ),
        (
            re.compile(r"\bRuntime\.Autograd\.Train\.(?:Dataset|DataLoader)\b"),
            "generic data containers live in `TorchLean.Data`; do not place them under the autograd runtime.",
        ),
        (
            re.compile(
                r"\b(?:ProgramWithNatInputs|natInputShapes|validateNatInputs|"
                r"NatVecRef|inputNatVec|getNatVec|setNatVec)\b"
            ),
            "Nat-specific program input plumbing was removed; use generic typed data inputs.",
        ),
        (
            re.compile(r"\b(?:castRankOneDim|relaxRankOne|relaxRankOneLower)\b"),
            "removed rank-specific helper name found; use `Tensor.castShape`, `relaxVector`, or "
            "`relaxVectorLower`.",
        ),
        (
            re.compile(r"\b(?:PackedDataset|PackedTensorErrorLe|RealPackedTensorEnclosed)\b"),
            "old packed-tensor names were removed; use `TensorDataset` and the `SomeTensor` predicates.",
        ),
        (
            re.compile(r"\beffectiveFitBatchSize\b"),
            "old lower training helper names are removed; use `effectiveTrainBatchSize`.",
        ),
        (
            re.compile(
                r"\b(?:FitResult|StreamFitResult|PairStreamFitResult|"
                r"TrainResult|StreamTrainResult|AlternatingTrainResult|"
                r"TrainSummary|NumericSummary|LoaderTrainConfig|LoaderTrainingResult)\b"
            ),
            "old trainer API names are removed; use `Trainer.Result`, `Trainer.Report`, "
            "`Trainer.Manual.LoaderConfig`, and `Trainer.Manual.LoaderResult`.",
        ),
        (
            re.compile(
                r"\bTrainer\.Metrics\b|"
                r"\bReport\.(?:numeric\?|requireNumeric|printNumeric|toTrainLog\?)\b"
            ),
            "trainer reports carry host `Float` losses directly; read `report.loss` or use "
            "`Report.toTrainLog`.",
        ),
        (
            re.compile(
                r"\b(?:BurgersOptions|LoadedData|EvalData|DiffusionOptions|AdderOptions|"
            r"SavedOptions|CorpusOptions|ExperimentConfig)\b"
        ),
            "leaf examples use namespace-local `Options`, `Preset`, `Splits`, and `Evaluation` "
            "instead of repeating the command name.",
        ),
        (
            re.compile(
                r"\b(?:TensorSource|matrixFromArray|matrixStorage|"
                r"trainCorpusFloat|trainBpeCorpusFloat|SupervisedEpochs|epochLoader|"
                r"shuffleEachEpoch|shuffleSeed|dropIncompleteBatch)\b"
            ),
            "removed representation-shaped API or example name found; use `Tensor.load` and "
            "short contextual names such as `Data.Loader`, `shuffle`, `seed`, and `dropLast`.",
        ),
    (
        re.compile(
            r"\b(?:cleanImageShape|noisyInputShape|modelConfigFor|modelFor|"
            r"cifarCleanImageBatch|imageNet64CleanImageBatch|"
            r"loadCifarCleanImageBatches|loadImageNet64CleanImageBatches|"
            r"trainCurveFloat|writeTrainingLog|runTypedDataset|"
            r"EvalScore|evalBatched|evalAllBatched|CurriculumMode|trainAdderFloat|"
            r"cudaMemoryCadence|writePredictionProbe|metricHistory|writeMetricLog|"
            r"evaluationData|pushLossPoint|evalLosses|recordEval|runPortableDense|"
            r"logRunHeader|runOptionsUsage)\b"
        ),
        "removed verbose example identifier found; use the short contextual name used by the "
        "model command.",
    ),
        (
            re.compile(r"\b(?:modelSummary|printModelSummary)\b"),
            "old trainer summary names are removed; use `trainer.summary` and `trainer.printSummary`.",
        ),
        (
            re.compile(r"\bverifyLInfIBP\b|\b(Trainer\.)?Verify\.robustLInf\b"),
            "duplicate verification helper names are removed; use `verifyRobustLInf` on trained results or `Trainer.Verify.lInfIBP` for requests.",
        ),
        (
            re.compile(
                r"\b(?:RuntimeSettings|argmaxRankOne|rankOneTensorToArray|rankOneTensorToPy|"
                r"matrixTensorToPy|rankFourTensorToPy)\b|"
                r"\b(?:argmaxRankOne|correctOneHotRankOne)\?"
            ),
            "removed compatibility name found; use `Trainer.RunConfig`, the axis-general metric "
            "operations, `Tensor.to tensor (Array α)`, or arbitrary-rank `tensorToPyString`.",
        ),
        (
            re.compile(
                r"\b(?:Conv2dLayer|Core\.oneHotAction|scalarValue|CROWNNodeCertificate|"
                r"ToTorchLean\.Sequential|mseSpecBasic|mseDerivSpecBasic|"
                r"layerNorm2dParams|appendCore|weakenContextCore|"
                r"parseValueGraphUnchecked|parseGraphUnchecked|"
                r"getRaw|singleRaw|getCLMRaw|stepRaw|RefT|"
                r"GpuMatmulPrecision|matmulForward|matmulForwardcuBLAS(?:32|64|With)|"
                r"embeddingRowsNat|embeddingBatchSeqNat|mlpGo|encoderStackGo|defaultTensor|"
                r"ResnetConfig|Cnn2|Unet2|mlpRelu|nn\.models\.MlpConfig|"
                r"oneHotTokenOrZero|parseExecutionMode|requestsCuda|AnyBatchLoader|loaderAny|"
                r"asTextTokenizer|loadRolloutCast|ioSingletonFloat|oneHotFloat|castFloat|"
                r"oneHotSequenceOrZero|oneHotBatchOrZero|Synthetic\.oneHot)\b"
            ),
            "removed duplicate name found; use the canonical declaration directly.",
        ),
        (
            re.compile(
                r"\b(?:ExecConfig|runFloat32|runCudaFloat32|runCudaEagerFloat32)\b|"
                r"\b(?:TorchLean\.)?Module\.(?:withRuntime|withModule)\b"
            ),
            "removed executable-command wrapper found; use `Module.RuntimeSelection`, "
            "`Module.withSelectedRuntime`, or the single "
            "`Module.Command.run` entrypoint with explicit runtime requirements.",
        ),
        (
            re.compile(
                r"\b(?:SgdConfig|MomentumSgdConfig|AdagradConfig|RmspropConfig|"
                r"AdadeltaConfig|momentumSgd)\b|"
                r"\boptim\.Kind\b|\bKind\.(?:name|toOptimizer)\b"
            ),
            "removed optimizer API name found; use the acronym-correct public config types, "
            "`optim.sgd { momentum := ... }`, or `optim.Algorithm.displayName/build`.",
        ),
        (
            re.compile(r"\b(?:tensorFloat|xavierW|kaimingW)\b"),
            "removed initializer API name found; use scalar-polymorphic `Init.tensor`, "
            "`Init.xavierUniform`, or `Init.kaimingUniform`.",
        ),
        (
            re.compile(
                r"\b(?:Data\.(?:floatSamples|singletonFloatIO)|"
                r"Trainer\.Probe\.ofFloatTensor|"
                r"TensorDataset\.(?:ofSupervisedFloatPairs|ofLabeledFloatPairs|ofBatchedFloat)|"
                r"Sample\.mapXY)\b"
            ),
            "removed Float-specific data API found; use `Data.samples`, `Data.singletonIO`, "
            "`Trainer.Probe.tensor`, generic `TensorDataset` constructors, or `Sample.map`.",
        ),
        (
            re.compile(
                r"\bRunConfig\.(?:ofRuntimeOptions|parseRuntimeArgs|parseRuntimeArgsOrThrow|"
                r"parse|parseCommandLine|cliArguments|"
                r"withScalar|withExecution|eager|typedGraph|cpu|cuda)\b|"
                r"\bTrainer\.TrainOptions\.(?:forSteps|withLogEvery|withCudaMemWatch|"
                r"withBatchSize|withScheduler|withoutScheduler|withLog|disableLog|withTitle|"
                r"withNotes|withLoadCheckpoint|withSaveCheckpoint)\b"
            ),
            "removed trainer convenience API found; use `RunConfig.fromRuntime`, direct "
            "`TrainOptions` record syntax, or the example-owned `TrainerFlags` parser.",
        ),
        (
            re.compile(r"\bTensor\.QR\b"),
            "removed tensor linear-algebra type found; QR decomposition results use "
            "`Tensor.QRFactors`.",
        ),
        (
            re.compile(
                r"\b(?:ParamsT|netT|updateAtT|energyT|seqStatesT|"
                r"VitPatchOutH|VitPatchOutW|VitPatchCount|"
                r"approxT|approxTTol|scaleT|toVecT|ofVecT|getYorT|"
                r"actual_index_lt|getMinorSpec|leadingEigenpairPowerIterationApproxSpec|"
                r"normalizeProbsSpec|getValueAtPosition|extractWindow|padMultiChannel|"
                r"extractMultiWindow|padChannelsZero|setValueAtPosition|addValueAtPosition|"
                r"get_at_or_zero_pad_multi_channel(?:_shift)?|matrixFromRowsPadTo|"
                r"fullLike|zerosLike|onesLike|useFin|expandLastDim|tlistBang)\b|"
                r"\babbrev\s+StateT\b|tlist!"
            ),
            "removed opaque model name found; use the descriptive tensor or ViT declaration name.",
        ),
        (
            re.compile(
                r"\b(?:DVal|FlatDVal|flatDValShape|flatDValTensor|permuteDVal|"
                r"mseLossDVal|getDVal|toDVal|dValOfAny|dValsOfCtx)\b"
            ),
            "removed shape-tagged-value alias found; use `Spec.SomeTensor` and its canonical operations.",
        ),
        (
            re.compile(
                r"\bautograd\.func\b|"
                r"\bautograd\.model\.(?:initState(?:With)?|OutputLoss|"
                r"valueAndGrad(?:Tensor)?|valueAndAllGrads|gradState|gradInputs|"
                r"vjpState|vjpInput|jacrevState|jvpState|hvpState)\b|"
                r"\babbrev\s+Fn\b"
            ),
            "removed autograd API name found; use `autograd.Function`, top-level tensor "
            "transforms, `autograd.model.State.init`, `autograd.model.Loss`, and the "
            "short model transforms.",
        ),
        (
            re.compile(
                r"\b(?:SequentialModel|ModelBuilder|modelParamShapes|LossReduction)\b|"
                r"\bTorchLean\.ParamTensors\b"
            ),
            "removed API alias found; use the canonical `nn`, `Module`, `Loss`, or `TensorPack` name.",
        ),
        (
            re.compile(
                r"\b(?:oneHotNat|oneHotToken|oneHotSequence|matrixPadTo|vectorFromArray|"
                r"vectorFromArrayD|"
                r"flattenKeep0|instantiateHostFloat|curveHostFloat|trainFixedCurveHostFloat|"
                r"RawDataLoader|EmbeddingOptions|oneHotAccuracyBatched|oneHotAccuracyLoader|"
                r"oneHotMetricsBatched|classProbesBatched|ScalarModuleDef|ScalarModule|"
                r"ppoActorCriticScalarModuleDef|ScalarEvaluator|lossScalar|"
                r"LossModuleDef|LossModule|LossEvaluator|lossModuleWithMode|lossModule)\b"
            ),
            "removed misleading API name found; use the canonical name that states its fallback or scalar semantics.",
        ),
        (
            re.compile(
                r"\b(?:Runtime\.Autograd\.(?:Torch|TorchLean)|TorchLean\.Runtime)\."
                r"(?:mm|bmm)\b"
            ),
            "rank-specific public matrix products were removed; use generic `matmul`.",
        ),
        (
            re.compile(
                r"\b(?:flattenBatch|flattenBatchPrefix|flattenLeading|zipWithLeading|"
                r"classifierBatch|regressorBatch|"
                r"uniformND|maskND|randND|loadCsvTensorND)\b"
            ),
            "removed dimension-specific API name found; use `flattenAfter`, a head with an explicit leading shape, or shape-indexed `rand.uniform`/`rand.mask`.",
        ),
        (
            re.compile(
                r"\b(?:linear2d|catAxisDyn|catAxis2Dyn|float32Vector|float32Matrix|"
                r"ieee32ExecVector|ieee32ExecMatrix|singletonVectorFloat|pointVectorFloat|"
                r"concatVectors|mapBatch0|boolMask01|multiheadAttentionWith|"
                r"multiheadAttention|cycleDataset|cycleDatasetOrError|firstArrayOrError|"
                r"shapeOfDims|ofDims|numelDims|tensorpack|exportGeneralModel|"
                r"mapSequenceSpec|zipWithSequenceSpec|reduceSumSequenceSpec|reverseSequenceSpec|"
                r"regressionGrid|regressionTargetsFloat|affinePlane|"
                r"classifyVector|classVectorProbes|classVectorProbesBatch|"
                r"datasetOfListVectors|readCsvDatasetPairs|readCsvVectorDataset|"
                r"readNpyVector|readNpyMatrix|"
                r"mlpInShape|mlpOutShape|cnnInShape|cnnOutShape|"
                r"resnetInShape|resnetHiddenShape|resnetOutShape|"
                r"epsConvNetInShape|epsConvNetOutShape|"
                r"kanInShape|kanOutShape|"
                r"vectorGenerativeConfig|vectorDataShape|vectorLatentShape|vectorVaeOutShape|"
                r"recurrentInShape|recurrentOutShape|"
                r"fnoInShape|fnoOutShape|"
                r"PPOActorCriticConfig|ppoActorInShape|ppoActorOutShape|ppoCriticOutShape|"
                r"ppoActor|ppoCritic|"
                r"transformerEncoderShape|vitInShape|vitConvOutShape|vitTokensShape|vitOutShape|"
                r"vitMaeInShape|vitMaeOutShape|"
                r"scalarTensor|vectorTensor|matrixTensor|nDArrayTensor|vectorN|matrixMN|"
                r"ofArray1D|ofArray2d|ofArrayDim|CausalTransformerOneHot|"
                r"toScalar|ofScalar|ofVecFn|ofMatFn|shapeToList|listToShape|"
                r"eval1NoGrad|eval1CompiledNoGrad|predict1|forwardCompiled|compileOut|"
                r"curveFloat64|trainFixedCurveFloat64|Probe\.point|ClassProbe|"
                r"uniformDims|maskDims|randDims|repeatBatch|"
                r"VectorGenerativeConfig|vectorAutoencoder|vectorVae|vectorVqVae|"
                r"vectorGanGenerator|vectorGanDiscriminator|vectorMaskedAutoencoder|"
                r"loadCifarVectorBatch|cifarVectorDataset|MLPConfig|CNNConfig|FNOConfig|"
                r"KANConfig|ViTConfig|ViTMAEConfig|round₃₂|ulp₃₂|eps₃₂|"
                r"round₃₂_eq_round32|ulp₃₂_eq_ulp32|eps₃₂_eq_eps32|"
                r"adaptFlatBatch|applyBatch|KANEdgeFamily|KANPiecewiseLinear|"
                r"RMSpropConfig|PPOFlags|BPECorpusOptions|optimizerLR|stepLR|"
                r"bitsToα|[A-Za-z0-9_]*StateWithLR|VectorMAE|MaskedPrediction|vectorOfArrayD|"
                r"expandVecToBatchSpec|batchToEndSpec|channelFirstToLastSpec|"
                r"collectAtIndexSpec|linearBatchedSpec|sliceVectorSpec|"
                r"sumLeading|reverseLeading|autoencoderBatchedForwardSpec|"
                r"decisionTreeForwardSpecN|decisionTreeBatchedForwardSpecN|"
                r"gradientBoostedTreesBatchedForwardSpec|"
                r"shapeDims|permuteDyn|reduceSumDimsDyn|reduceMeanDimsDyn|sliceRangeAxisDyn|"
                r"softmaxDyn|logSoftmaxDyn|unsqueezeDyn|squeezeDyn|concatAxisDyn|stackAxisDyn|"
                r"splitAxisDyn|chunkAxisDyn|einsumDyn|"
                r"cnnSpec|cnnWithReluSpec|cnnForward|CNN2Config|CNN2Spec|CNN2Grads|"
                r"Cnn2Config|Cnn2Spec|Cnn2Grads|UNet2Config|UNet2Spec|UNet2Grads|"
                r"UNetDownH|UNetDownW|UNetUpH|UNetUpW|Models\.CNN|"
                r"simpleRnnModelSpec|rnnClassifierModelSpec|multilayerRnnSpec|"
                r"rnnLanguageModelSpec|SimpleRNNModel|MultiLayerRNNModel|RNNClassifier|"
                r"RNNGenerator|BiRNNModel|simpleRnnForward|simpleRnnSequenceForward|"
                r"rnnClassifierForward|rnnGeneratorForward|birnnForward|"
                r"multilayerRnnForwardSingle|simpleRnnBackward|sequenceClassificationLoss|"
                r"simpleRNNToModuleSpec|rnnClassifierToModuleSpec|"
                r"simpleGruModelSpec|gruClassifierModelSpec|multilayerGruSpec|"
                r"gruLanguageModelSpec|SimpleGRUModel|MultiLayerGRUModel|GRUClassifier|"
                r"GRUGenerator|BiGRUModel|GRULanguageModel|GRUEncoderDecoder|"
                r"AttentionGRUModel|ResidualGRUModel|simpleGruForward|"
                r"simpleGruSequenceForward|gruClassifierForward|gruGeneratorForward|"
                r"bigruForward|multilayerGruForward|gruLmForward|gruEncoderDecoderForward|"
                r"simpleGruBackward|residualGruForward|simpleGRUToModuleSpec|"
                r"gruClassifierToModuleSpec|biGRUToModuleSpec|gruGeneratorToModuleSpec|"
                r"LSTMGrads|SimpleLSTMModelGrads|simpleLstmModelSpec|"
                r"lstmClassifierModelSpec|multilayerLstmSpec|lstmLanguageModelSpec|"
                r"SimpleLSTMModel|MultiLayerLSTMModel|LSTMClassifier|LSTMGenerator|"
                r"BiLSTMModel|LSTMLanguageModel|AttentionLSTMModel|simpleLstmForward|"
                r"simpleLstmSequenceForward|simpleLstmBackward|simpleLstmMseLoss|"
                r"simpleLstmMseGrad|lstmClassifierForward|lstmClassifierBackward|"
                r"lstmGeneratorForward|bilstmForward|multilayerLstmForward|lstmLmForward|"
                r"simpleLSTMToModuleSpec|lstmClassifierToModuleSpec|biLSTMToModuleSpec|"
                r"ModSpec|NNModuleSpec|SpecChain|ExportFunctions|export_func|SpecModule|"
                r"Proofs\.RuntimeApprox\.TList|"
                r"[A-Za-z0-9_]*ModuleSpec|composeRight|extractLayerInfo|"
                r"exportSpecChain|exportMLPFromSpecChain|toModuleSpec|"
                r"flatIndexAux|flatIndexAux_lt|andThenAux|eval_andThenAux|"
                r"TypedGraphWithAux|lowerToTypedGraphWithAux|"
                r"indexAux|hiddenList|applyAux|inSortedRangesAux|"
                r"flattenFloatAux|flattenBoolMaskAux|unflattenFloatAux|"
                r"unflattenBoolMaskAux|uniformAux|maskAux|normalAux|diagMaskSpecAux|argsAux|"
                r"layernormPure|layernormPure_eq_spec|"
                r"linearInterpolationRaw|cosineAnnealRaw|"
                r"swapAtDepthHelper|swapAtDepthHelper_zero|swapAtDepthSpec|"
                r"choleskyColsImpl|cholSolveImpl|solveRidgeImpl|"
                r"unflattenFloatUnsafe|unflattenBoolMaskUnsafe|"
                r"alphaBarsLinear|"
                r"pretokenizeAux|parseJsonStringAux|toSeqAux|buildNodesAux|verifySegmentAux|"
                r"detInitParamsAux|runListAux|runListAux_nil|runListAux_outputs_length|"
                r"selectiveMamba_runListAux_append_outputs_prefix|neuralTruncateAux|"
                r"neuralTruncateAux_remainder_bounds|neuralTruncateAux_mantissa_decomposition|"
                r"neuralTruncateAux_brackets|"
                r"DynTensor|dynamicOfList|reduceDimsDynCore|generatePyTorchHelperModules|"
                r"countsFindD|dimFindD|vectorOfArrayWithDefault|oneHotBatchFromRows|"
                r"byteAtD|charAtD|SessionImpl|encodeVec|encodeBatchVec|"
                r"getDimSize|TorchLean\.Loss\.dimSize|Shape\.(?:dimSize|innerDimSize|isMatrix|isVector)|"
                r"gatherScalarNat|gatherVecNat|gatherRowsNat|"
                r"gatherScalarRef|gatherRowRef|gatherVecRef|gatherRowsRef|"
                r"nllNat|crossEntropyNat|"
                r"rowTargetFlatIndices|"
                r"text\.causalMask|"
                r"Data\.(?:fromList|toList|size|isEmpty))\b"
            ),
            "removed specialized or inconsistently named helper found; use the canonical shape-general API.",
        ),
        (
            re.compile(r"\bnn\.(?:manualSeed|runGlobal|freshSeed|freshSeeds)\b"),
            "global seed state belongs to `rand`; use `rand.manualSeed`, `rand.runGlobal`, or `rand.nextSeedGlobal`.",
        ),
        (
            re.compile(
                r"\bReport\.(?:probes|meanLossLoader|oneHotMetrics|oneHotMetricsLoader)\b|"
                r"\bReport\.Objective\.meanLoss\b"
            ),
            "removed reporting wrapper found; call the canonical dataset or loader metric directly.",
        ),
        (
            re.compile(
                r"\b(?:NativeOptimizerCheckpoint|CudaAdamSchema|saveNativeOptimizerState|"
                r"loadNativeOptimizerState|projectedSGDUpdate_identity_eq_sgd|"
                r"update_identity_param_eq_momentumSGD)\b"
            ),
            "removed checkpoint or definitional-theorem name found; use the shared optimizer-state checkpoint API.",
        ),
        (
            re.compile(
                r"\b(?:conv2d|convTranspose2d|maxPool2d|maxPool2dPad|smoothMaxPool2d|"
                r"smoothMaxPool2dPad|avgPool2d|avgPool2dPad|batchNormChannelFirst|"
                r"batchNorm2d|batchNorm2dNchw|batchNorm2dChwEval|instanceNorm2dNchw|"
                r"groupNorm2dNchw|transpose2d|transpose3dFirstToLast|"
                r"transpose3dLastToFirst|transpose3dLastTwo|nchwToNhwc|nhwcToNchw|"
                r"padChannelsFirst2d|Conv2dSpec|ConvTranspose2dSpec|MaxPool2dSpec|"
                r"AvgPool2dSpec|AdaptiveAvgPool2dSpec|AdaptiveMaxPool2dSpec)\b"
            ),
            "fixed-rank spatial API name found; use the rank-polymorphic convolution, pooling, normalization, permutation, or adaptive-pooling API.",
        ),
        (re.compile(r"\bsimp\s*\[\s*\*(\s*[,\]])"), "`simp [*]` is banned; prefer `simp [h₁, h₂]` or `simp (config := ...)` with explicit hypotheses."),
        (
            re.compile(r"\bset_option\s+maxHeartbeats\b"),
            "proof-level `maxHeartbeats` overrides are not allowed; split the declaration or isolate expensive normalization behind reusable lemmas.",
        ),
        (
            re.compile(r"^\s*public\s+import\s+Mathlib\.Tactic\b", flags=re.MULTILINE),
            "Do not `public import Mathlib.Tactic.*`; import the specific tactic modules you use (non-public).",
        ),
        (
            re.compile(r"^\s*import\s+Mathlib\.Tactic(?!\.)\b", flags=re.MULTILINE),
            "Do not `import Mathlib.Tactic` (umbrella import). Import the specific `Mathlib.Tactic.*` modules you use.",
        ),
        (re.compile(r"@\[\s*de" r"precated\b"), "`@[de" "precated]` is banned in TorchLean sources."),
        (
            re.compile(
                r"\b(?:compatibility (?:alias|shim|wrapper|layer|re-export)|"
                r"legacy (?:alias|name|spelling)|historical (?:alias|name|spelling)|"
                r"deprecated alias|old import path|migration shim|kept for compatibility)\b",
                flags=re.IGNORECASE,
            ),
            "compatibility aliases and shims are not allowed; migrate callers to the canonical API and delete the old route.",
        ),
        (
            re.compile(
                r"\b(?:Spec\.(?:fill|zeros|ones)|Tensor\.fill|broadcastLike|broadcastFill|"
                r"Runtime\.Autograd\.Model\.Random|TorchLean\.Einops)\b"
            ),
            "removed tensor, syntax-scope, or RNG compatibility name found; use `Tensor.full`, "
            "`Tensor.zeros`, `Tensor.ones`, `replicate`, `TorchLean.Tensor`, or `Spec.Random`.",
        ),
        (
            re.compile(r"\bmapEach\b"),
            "`mapEach` is ambiguous across tensor, module, builder, and runtime layers; use "
            "`Tensor.mapLeading`, `Module.liftLeading`, or the layer-appropriate `mapLeading`.",
        ),
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
        masked = _mask_lean_comments_and_strings(_mask_verso_prose(text))
        rel = path.relative_to(REPO_ROOT).as_posix()
        _check_line_style(path, rel, text, findings)
        internal_namespace_lines = _internal_namespace_lines(masked)
        _check_local_source_refs(path, text, findings)
        _check_lean_doc_math(path, text, findings)
        _check_backend_contract_refs(path, text, lake_text, findings)

        if rel.startswith(("NN/API/", "NN/Examples/")):
            for match in re.finditer(r"_root_\.", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public API and example code must resolve canonical namespaces without "
                        "`_root_.`; fix the namespace or import boundary instead.",
                    )
                )

        if rel.startswith("NN/Examples/Quickstart/"):
            quickstart_internals = [
                (re.compile(r"\bTensorPack\b"), "heterogeneous runtime packs"),
                (re.compile(r"\bRuntime\.Autograd\b"), "runtime autograd internals"),
                (re.compile(r"\bNN\.IR\b"), "raw compiler IR"),
                (re.compile(r"\bTensor\.ofFn\b"), "proof-level tensor construction"),
                (re.compile(r"\bList\.finRange\b"), "bounded-index proof plumbing"),
            ]
            for pattern, description in quickstart_internals:
                for match in pattern.finditer(masked):
                    line, col = _line_col(text, match.start())
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"quickstarts must use reader-facing APIs, not {description}; "
                            "move the internal demonstration to `DeepDives` or add a public wrapper.",
                        )
                    )

        runtime_example_prefixes = (
            "NN/Examples/Quickstart/",
            "NN/Examples/Data/",
            "NN/Examples/Factorization/",
            "NN/Examples/Models/Supervised/",
            "NN/Examples/Models/Vision/",
            "NN/Examples/Models/Sequence/",
            "NN/Examples/Models/Generative/",
            "NN/Examples/Models/Operators/",
        )
        if rel.startswith(runtime_example_prefixes):
            for match in re.finditer(r"\bSpec\.", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "runtime-facing examples must call the public executable API, not `Spec`; "
                        "move proof-only material to a proof example or add a public operation.",
                    )
                )

        if rel.startswith("NN/Examples/"):
            for match in re.finditer(r"\b(?:nn\.)?State\.Internal\b", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "examples must use the public opaque `nn.State` and public execution or "
                        "verification APIs, not unpack its internal tensor representation.",
                    )
                )

        if rel.startswith("NN/Examples/Models/"):
            prefixed_name_re = re.compile(
                r'\bdef\s+exeName\s*:\s*String\s*:=\s*"torchlean(?:\s|")'
            )
            for match in prefixed_name_re.finditer(text):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "model examples store only their CLI subcommand in `exeName`; help text "
                        "adds `lake exe torchlean` at the rendering boundary.",
                    )
                )

        if rel.startswith(
            (
                "NN/API/",
                "NN/Examples/Models/",
                "home_page/blueprint/TorchLeanBlueprint/Guide/",
            )
        ):
            redundant_shape_name_re = re.compile(
                r"\b(?:inputShape|outputShape|targetShape)\b(?!\?)"
            )
            for match in redundant_shape_name_re.finditer(masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public tensor signatures use `input`, `output`, or `target`; use `σ`/`τ` "
                        "for invisible generic shape indices and reserve a `Shape` suffix for names "
                        "that distinguish multiple shapes.",
                    )
                )

        if rel.startswith("NN/Tensor/Internal/"):
            removed_tensor_checker_names_re = re.compile(
                r"\b(?:"
                r"NormalizedTransform\.inputShape|"
                r"TransformPlan\.(?:outputShape|inferredInputShape|inferredOutputShape)|"
                r"CheckedPack\.outputShape|"
                r"CheckedEinsum\.outputShape|"
                r"einsumOutputShape"
                r")\b"
            )
            for match in removed_tensor_checker_names_re.finditer(masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "tensor checker records already identify shape-valued fields by type; "
                        "use `input`, `output`, `inferredInput`, or `inferredOutput`.",
                    )
                )

        if rel.startswith("NN/Examples/Models/") or rel == "NN/API/RL/Cli.lean":
            for match in re.finditer(r"\.drop\s+10\b", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "do not recover a CLI subcommand by stripping a fixed prefix; store the "
                        "subcommand directly.",
                    )
                )

        removed_model_api_patterns = (
            (r"\bMamba\.textLM\b", "`Mamba.languageModel`"),
            (r"\bDenseGenerative\b", "`Generative`"),
            (r"\bGenerative\.ganGenerator\b", "`Generative.generator`"),
            (r"\bGenerative\.ganDiscriminator\b", "`Generative.discriminator`"),
            (r"\bConvolutionActivation\.Config\b", "`ConvBlock.Config`"),
            (r"\bConvolutionActivationPooling\.Config\b", "`ConvPoolBlock.Config`"),
            (r"\bconvAct\b", "`convBlock`"),
            (r"\bconvActPool\b", "`convPoolBlock`"),
        )
        for pattern, replacement in removed_model_api_patterns:
            for match in re.finditer(pattern, masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"removed model API name found; use {replacement}.",
                    )
                )

        if rel == "NN/API/Models/Transformer.lean":
            for match in re.finditer(r"\babbrev\s+shape\b", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "the Transformer config shape is specifically a token shape; use "
                        "`tokenShape`.",
                    )
                )

        if rel.startswith("NN/API/") and rel != "NN/API/Neural/Builders.lean":
            for match in re.finditer(
                r"\bRuntime\.Autograd\.TorchLean\.NN\.Seq\.id\b", masked
            ):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public API implementations should construct identity branches through "
                        "`nn.Sequential.identity`.",
                    )
                )

        import_directives: dict[str, int] = {}
        import_re = re.compile(
            r"^\s*((?:(?:public|private)\s+)?(?:meta\s+)?import\s+([A-Za-z0-9_.]+))\s*$",
            flags=re.MULTILINE,
        )
        # Lean module imports form one contiguous block at the start of a file. Restrict the
        # check to that block so `import ...` lines in Verso code examples are not mistaken for
        # dependencies of the documentation module itself.
        import_header_lines: list[str] = []
        saw_import = False
        for header_line in masked.splitlines(keepends=True):
            if import_re.fullmatch(header_line.rstrip("\r\n")):
                saw_import = True
                import_header_lines.append(header_line)
            elif not saw_import or not header_line.strip():
                import_header_lines.append(header_line)
            else:
                break
        import_header = "".join(import_header_lines)
        for match in import_re.finditer(import_header):
            directive = " ".join(match.group(1).split())
            module_name = match.group(2)
            line, col = _line_col(text, match.start())
            if rel.startswith("NN/CI/") and directive.startswith("public import"):
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "CI import targets compile dependencies but must not re-export them; "
                        "use a private `import`.",
                    )
                )
            previous_line = import_directives.get(directive)
            if previous_line is None:
                import_directives[directive] = line
            else:
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"duplicate import `{module_name}`; it was already imported on line "
                        f"{previous_line}.",
                    )
                )

        if rel.startswith("NN/API") and TOP_LEVEL_API_DECL_RE.search(masked) is None:
            for match in PUBLIC_IMPORT_RE.finditer(import_header):
                module_name = match.group("module")
                if (
                    module_name in BROAD_LOW_LEVEL_IMPORTS
                    or module_name.startswith(BROAD_LOW_LEVEL_IMPORT_PREFIXES)
                ):
                    line, col = _line_col(text, match.start("module"))
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"import-only API umbrella re-exports low-level module `{module_name}`; "
                            "export a focused API module instead.",
                        )
                    )

        ownership_sensitive_cuda_modules = {
            "NN/Runtime/Autograd/Engine/Cuda/Buffer.lean",
            "NN/Runtime/Autograd/Engine/Cuda/Kernels.lean",
            "NN/Runtime/Autograd/Engine/Cuda/ConvPool.lean",
        }
        if rel in ownership_sensitive_cuda_modules:
            unsafe_extern = re.compile(r"@\[(?![^\]]*\bnever_extract\b)[^\]]*\bextern\b[^\]]*\]")
            for m in unsafe_extern.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "CUDA buffer externs must use `never_extract`; these calls allocate, "
                        "observe, or mutate native resources and may not be commoned or deleted.",
                    )
                )

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

        if rel.startswith("NN/"):
            fixed_vector_re = re.compile(r"\b(?:List\.)?Vector\b|#v\[")
            if (
                not rel.startswith(TENSOR_INTERNAL_PREFIX)
                and rel not in TENSOR_VECTOR_BOUNDARY_FILES
            ):
                for m in fixed_vector_re.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            "fixed-shape numerical data must use `TorchLean.Tensor`; use `Array` for "
                            "dynamic homogeneous storage.",
                        )
                    )

            removed_numeric_container_re = re.compile(
                r"\b(?:Tensor\.ofList|NN\.Tensor\.ofList|someTensorOfArray|"
                r"ofArrayDynamic|"
                r"someTensorOfList|fromFloatList|"
                r"Dataset\.ofList|Dataset\.toList|cycleList(?:OrError)?|floatSampleArray|"
                r"vectorTensorTo(?:List|Py)|tensorOfFlatListExact)\b"
            )
            for m in removed_numeric_container_re.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "removed numerical-container API: use a shaped `Tensor` or an `Array` "
                        "boundary instead.",
                    )
                )

            removed_public_ops_re = re.compile(r"(?<!\.)\bTorchLean\.Ops\b")
            for m in removed_public_ops_re.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "removed public runtime namespace: use `TorchLean.Runtime` operations.",
                    )
                )

            removed_public_ref_re = re.compile(r"\b(?:TorchLean\.)?Runtime\.RefTy\b")
            for m in removed_public_ref_re.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "removed public runtime handle: use `TorchLean.Runtime.ValueRef`.",
                    )
                )

            dynamic_numeric_list_re = re.compile(
                r"\bList\s+(?:Float|Rat|Int|Bool|UInt8|UInt16|UInt32|UInt64)\b|"
                r"\bList\s*\(\s*(?:Probe|Sample\.Supervised|LinParams)\b|"
                r"\bList\s*\(\s*FlatAffine\b|"
                r"\bList\s+PinnLayer\b|"
                r"\bIO\.Ref\s*\(\s*List\s+Nat\s*\)|"
                r"\bList\s+NN\.Backend\.(?:AcceptedKernel|Provider)\b|"
                r"\bList\s*\(\s*NN\.Backend\.KernelHandler\b|"
                r"\bhiddenDims\s*:\s*List\s+Nat\b"
            )
            if not rel.startswith(TENSOR_INTERNAL_PREFIX):
                for m in dynamic_numeric_list_re.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            "dynamic homogeneous numerical collections must use `Array`; reserve "
                            "`List` for type-level or proof-recursive structure.",
                        )
                    )

        for rx, msg in banned_regexes:
            if rel.startswith(TENSOR_INTERNAL_PREFIX) and msg.startswith(
                "removed specialized or inconsistently named helper"
            ):
                continue
            for m in rx.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))

        rel = path.relative_to(REPO_ROOT).as_posix()

        # Keep the numerical library reusable without importing tensors, models, runtimes, or
        # verification. TorchLean-specific adapters must point into `NN.Floats`, never the reverse.
        if rel == "NN/Floats.lean" or rel.startswith("NN/Floats/"):
            for m in re.finditer(
                r"^\s*(?:public\s+)?import\s+(NN\.[A-Za-z0-9_.]+)\s*$",
                masked,
                flags=re.MULTILINE,
            ):
                imported = m.group(1)
                if not (imported == "NN.Floats" or imported.startswith("NN.Floats.")
                        or imported == "NN.Core" or imported.startswith("NN.Core.")):
                    line, col = _line_col(text, m.start(1))
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"floating-point core imports `{imported}`; move this integration to the spec, proof, runtime, or verification layer.",
                        )
                    )

        is_shape_generic_public_api = (
            rel == "NN/API/Tensor.lean"
            or rel.startswith("NN/API/Models/")
            or rel.startswith("NN/API/Neural/")
        )
        if is_shape_generic_public_api:
            for m in re.finditer(r"\bPNat\b", masked):
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public tensor and model APIs use ordinary `Nat` dimensions and validate "
                        "positivity at construction time; do not expose proof-carrying `PNat` "
                        "configuration fields.",
                    )
                )
            for m in re.finditer(r"\bList\s+Nat\b", masked):
                line_start = masked.rfind("\n", 0, m.start()) + 1
                line_end = masked.find("\n", m.end())
                if line_end < 0:
                    line_end = len(masked)
                source_line = masked[line_start:line_end]
                if "hiddenWidths" in source_line:
                    continue
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public tensor and model geometry must use `Spec.Shape` for static "
                        "shape indices or `Tensor Nat [d]` for computed geometry, not `List Nat`; "
                        "ordinary lists are reserved for explicitly named recursive architecture "
                        "plans such as `hiddenWidths`.",
                    )
                )
            for declaration in PUBLIC_DECL_RE.finditer(masked):
                name = declaration.group("name")
                line, col = _line_col(text, declaration.start("name"))
                if line not in internal_namespace_lines and PUBLIC_LAYOUT_NAME_RE.search(name):
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"public declaration `{name}` encodes a fixed rank or memory layout; "
                            "express axes through `Spec.Shape`, `TorchLean.Tensor Nat [d]`, or a domain-specific "
                            "example outside the public tensor/model API.",
                        )
                    )

        if rel.startswith("NN/API/Trainer/Train/") and rel.endswith(".lean"):
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

        if rel == "NN/API/Trainer/Train.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                            "`NN.API.Trainer.Train` must stay an import-only aggregator; put training implementation in `NN.API.Trainer.Train.*` modules.",
                    )
                )

        if rel == "NN/API/Neural.lean" and re.search(
            r"^\s*public\s+import\s+NN\.API\.Trainer\s*$", masked, flags=re.MULTILINE
        ):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Neural` must not re-export `NN.API.Trainer`; use `TorchLean.Trainer` for ordinary code and import the advanced training module explicitly when needed.",
                )
            )
        is_trainer_api = rel == "NN/API/Trainer.lean" or rel.startswith("NN/API/Trainer/")
        is_training_entrypoint = rel in {"NN/API.lean", "NN/API/Data/Training.lean"}
        if re.search(
            r"^\s*public\s+import\s+NN\.API\.Trainer\s*$", masked, flags=re.MULTILINE
        ) and not (is_trainer_api or is_training_entrypoint):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Trainer` should only be imported by the Trainer API; keep the callback-heavy training layer out of broad application API imports.",
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

        if rel.endswith(".lean") and any(
            rel.startswith(prefix) for prefix in PUBLIC_NUMERICAL_EXAMPLE_PREFIXES
        ):
            for rx, msg in PUBLIC_NUMERICAL_SPEC_BANNED_PATTERNS:
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

        # Warning and visibility checks are part of the build contract. Do not hide them locally:
        # fix the declaration or proof that emits the diagnostic.
        suppressed_linter_re = re.compile(
            r"set_option\s+linter\.([A-Za-z0-9_]+)\s+false(?:\s+in)?"
        )
        for m in suppressed_linter_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"suppresses Lean linter `{m.group(1)}`; fix the diagnostic instead.",
                )
            )

        nolint_re = re.compile(
            r"(?:@\[\s*|attribute\s+\[\s*)nolint\b"
        )
        for m in nolint_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "suppresses a Lean linter with `nolint`; fix the diagnostic instead.",
                )
            )

        private_compat_re = re.compile(r"\bbackward\.privateInPublic(?:\.warn)?\b")
        for m in private_compat_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "overrides Lean's strict private-in-public boundary; remove the override.",
                )
            )

        hidden_warning_re = re.compile(
            r"(?:set_option\s+warningAsError\s+false|"
            r"⟨\s*`warningAsError\s*,\s*false\s*⟩)"
        )
        for m in hidden_warning_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "disables warnings-as-errors; keep compiler warnings visible and fix them.",
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
    ap.add_argument(
        "--sync-doc-examples",
        action="store_true",
        help="Copy every compiled snippet from NN/Tests/API/DocExamples into its docstring.",
    )
    args = ap.parse_args()

    if args.sync_doc_examples:
        findings, updates = _sync_doc_examples(write=True)
        for update in updates:
            print(f"synced: {update}")
        for f in findings:
            print(f.render())
        if any(f.level == "ERROR" for f in findings):
            print(f"\nFAILED: could not sync every doc example.")
            return 1
        print(f"OK: {len(updates)} docstring example(s) updated.")
        return 0

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
