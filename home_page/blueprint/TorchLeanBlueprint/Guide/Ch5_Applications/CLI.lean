import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Command-Line Reference" =>
%%%
tag := "cli"
%%%

TorchLean has two command dispatchers. `torchlean` runs models, demonstrations, and data checks,
while `verify` runs verification workflows and artifact checkers:

```
# Replace the placeholders with a registered command and its
# own flags.
scripts/lake.sh exe torchlean <example> [flags...]
scripts/lake.sh exe verify -- <tool> [args...]
```

Both commands can print numerical results. A training command's success means it completed its
runtime checks, while a verification command's success depends on that tool's declared checks. For
example, validating an artifact's schema is different from establishing its claimed bounds.

Both dispatch tables are ordinary Lean definitions:

- {src "TorchLeanExamples/Runner.lean"}[`TorchLeanExamples/Runner.lean`] routes the `torchlean`
  subcommands.
- {src "TorchLeanExamples/Verification/CLI.lean"}[`TorchLeanExamples/Verification/CLI.lean`] routes
  the `verify` tools.

You reach these workflows through the commands below, but the help from your executable remains
authoritative for command names and flags. Terminal transcripts below are recorded examples:
numerical output, banners, and backend reports can change with the build.

# Discovering Commands

You get grouped help and a machine-readable list from the dispatcher:

```terminal
# Compare grouped help with the flat list used by scripts.
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe torchlean --list
```

`--help` explains the command groups and gives a starting path. `--list` prints exactly one
registered command name per line, which is what you want in a script.
Running `scripts/lake.sh exe torchlean`
with no arguments prints the same help and exits with status `1`, so a bare invocation in a
Makefile fails rather than silently doing nothing.

Use the generated help for the command groups and descriptions, which come from the runner's
registry and not from a second catalogue kept on this page.

`--list` prints the registered command names with no grouping and no descriptions:

```
quickstart_tensors
quickstart_autograd
quickstart_mlp
mlp
...
torch_ir_pytorch
```

An unknown name is refused before anything runs, and the help follows so the correct spelling is on
screen:

```terminal
# An unknown name exercises dispatch before any model is
# constructed.
scripts/lake.sh exe torchlean nosuchthing
```

```
Unknown example: nosuchthing

TorchLean runnable examples
...
```

That invocation exits with status `1`.

# Command Implementation

You can see how the shared parser connects to a trainer in the MLP quickstart, whose `main` first
handles help and arguments, then constructs the training configuration:

```
-- Dispatch selects the registered command; that command
-- parses its remaining arguments.
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args
  let (steps, args) ← CLI.orThrow exeName <|
    CLI.takePositiveNatFlag args exeName "steps" 200
  let runtime ← Trainer.RunConfig.parseCommandLine exeName args

  let trainer := Trainer.create model
    { runtime with
        objective := mse
        optimizer := optim.adam { lr := 0.03 }
        seed := seed }
  ...
```

Before constructing the trainer, the quickstart drops a leading `--`, answers `--help` if present,
and parses the remaining arguments (the full file is
{src "TorchLeanExamples/Quickstart/MLP.lean"}[`MLP.lean`]). Each binding of
`args` holds only the arguments the previous parser left, and after `--seed` and `--steps` are
consumed, `Trainer.RunConfig.parseCommandLine` handles runtime flags and checks for anything left
over. In {srcDir "TorchLean/NN/CLI"}[`TorchLean/NN/CLI/`], the shared functions expose this
remainder in their return types:

```lean (name := cliTypes)
-- The result carries both a typed optional value and the
-- arguments still to parse.
#check @CLI.takeFlagValue?
#check @CLI.takeNatFlag?
#check @CLI.takePathFlag?
#check @CLI.takeBoolValueFlag?
#check @CLI.checkNoArgs
```

```leanOutput cliTypes (whitespace := lax)
CLI.takeFlagValue? : List String → String →
  Except String (Option String × List String)
```

```leanOutput cliTypes (whitespace := lax)
CLI.takeNatFlag? : List String → String →
  Except String (Option ℕ × List String)
```

```leanOutput cliTypes (whitespace := lax)
CLI.takePathFlag? : List String → String →
  Except String (Option System.FilePath × List String)
```

```leanOutput cliTypes (whitespace := lax)
CLI.takeBoolValueFlag? : List String → String →
  Except String (Option Bool × List String)
```

```leanOutput cliTypes (whitespace := lax)
CLI.checkNoArgs : List String → Except String Unit
```

All four `take...` functions return the parsed optional value together with the remaining
arguments, and each can also return an error message. `checkNoArgs` returns only `Unit` on success,
and succeeds only for an empty list. Calling it on the final remainder rejects flags that no parser
consumed.

If you know Python's `argparse`, `parse_args` also rejects unknown arguments by default, while
`parse_known_args` returns them to its caller. Here we can see that choice in the call to
`checkNoArgs`.

In these signatures, `Except` and `Option` answer separate questions. `Except.error` means that the
supplied command line can't be used. Inside `Except.ok`, `none` means that this particular flag was
absent, while `some value` means it was present and parsed. That distinction lets a command supply
a default without silently replacing a misspelled number. Just as important is the second
component: it records the arguments left for another parser to account for. A successful individual
parser doesn't establish that the whole command line is valid.

# Parser Results

You can evaluate the parsers on their own. This call consumes `--steps 20` and leaves the unrelated
path flag in the remainder:

```lean (name := cliTake)
-- The steps parser removes its flag while preserving the
-- unrelated path flag.
#eval CLI.takeNatFlag?
  ["--steps", "20", "--x", "a.npy"] "steps"
```

```leanOutput cliTake (whitespace := lax)
Except.ok (some 20, ["--x", "a.npy"])
```

That call took the `20` out and handed `--x a.npy` back untouched. A flag written as `--key=value`
is accepted as the same thing:

```lean (name := cliEq)
-- The equals form produces the same typed value as the
-- separated form.
#eval CLI.takeNatFlag? ["--steps=20"] "steps"
```

```leanOutput cliEq (whitespace := lax)
Except.ok (some 20, [])
```

A repeated flag produces an error:

```lean (name := cliDup)
-- Two occurrences are rejected even when both values are
-- individually valid.
#eval CLI.takeNatFlag?
  ["--steps", "1", "--steps", "2"] "steps"
```

```leanOutput cliDup (whitespace := lax)
Except.error "--steps: duplicate flag"
```

A shell script that appends options from several variables can supply conflicting values for the
same flag. Rejecting duplicates requires the caller to resolve that conflict before running.

An absent flag is not an error, because callers supply defaults:

```lean (name := cliAbsent)
-- Absence leaves the other arguments intact so the caller
-- can choose a default.
#eval CLI.takeNatFlag? ["--x", "a.npy"] "steps"
```

```leanOutput cliAbsent (whitespace := lax)
Except.ok (none, ["--x", "a.npy"])
```

A present flag with an unusable value is:

```lean (name := cliBad)
-- A present but malformed value is an error, not a request
-- for the default.
#eval CLI.takeNatFlag? ["--steps", "two"] "steps"
```

```leanOutput cliBad (whitespace := lax)
Except.error "--steps: expected a natural number, got `two`"
```

Arguments left unconsumed are reported by the final check:

```lean (name := cliRest)
-- The final parser rejects a flag that no earlier parser
-- recognized.
#eval CLI.checkNoArgs ["--rollout", "8"]
```

```leanOutput cliRest (whitespace := lax)
Except.error "unexpected arguments: [--rollout, 8]"
```

```lean (name := cliRestOk)
-- An empty remainder confirms that every supplied argument
-- was consumed.
#eval CLI.checkNoArgs []
```

```leanOutput cliRestOk (whitespace := lax)
Except.ok ()
```

Paths and Booleans go through typed variants, so a path becomes a `System.FilePath` at the boundary
rather than being carried around as a string:

```lean (name := cliPath)
-- Parsing constructs a path value. Opening the file
-- happens later.
#eval CLI.takePathFlag?
  ["--log", "/tmp/trainlog.json"] "log"
```

```leanOutput cliPath (whitespace := lax)
Except.ok (some (FilePath.mk "/tmp/trainlog.json"), [])
```

```lean (name := cliBool)
-- This flag consumes an explicit Boolean value rather than
-- toggling on presence.
#eval CLI.takeBoolValueFlag?
  ["--shuffle", "false"] "shuffle"
```

```leanOutput cliBool (whitespace := lax)
Except.ok (some false, [])
```

Both remaining helpers deal with the shape of the command line, not its contents. A leading
separator is dropped for the benefit of wrappers that insist on one:

```lean (name := cliDash)
-- Remove the leading separator before interpreting command
-- arguments.
#eval CLI.dropDashDash
  ["--", "quickstart_mlp", "--steps", "20"]
```

```leanOutput cliDash (whitespace := lax)
["quickstart_mlp", "--steps", "20"]
```

```lean (name := cliHelp)
-- Help detection recognizes the request without running the
-- selected model.
#eval CLI.hasHelp ["quickstart_mlp", "--help"]
```

```leanOutput cliHelp (whitespace := lax)
true
```

`hasHelp` is checked before the other parsers, so help remains available even if another argument
on the command line is invalid.

Both the path and the Boolean outputs above describe parsing, not the effects of running the
command. `FilePath.mk` doesn't check whether a file exists, whether its contents have the required
shape, or whether its directory is writable (those checks belong to the reader or writer that uses
the path). Similarly, successfully parsing `false` says which Boolean was requested, but the
command must still use that value when it constructs its configuration. So when a flag seems to
have no effect, follow the parsed value into the configuration.

# Command Grammar

A command line normally has this form:

```
# Runtime and command-specific flags share the argument list
# after the command.
scripts/lake.sh exe torchlean <subcommand> [runtime flags] [command flags]
```

For example:

```terminal
# Select CPU execution while setting the training length and
# initialization seed.
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu --steps 20 --seed 2026
```

Runtime flags may also precede the subcommand:

```terminal
# The dispatcher also accepts the runtime selection before
# the command name.
scripts/lake.sh exe torchlean --device cpu quickstart_mlp \
  --steps 20 --seed 2026
```

The runner collects prefix flags, finds the command, and forwards those flags together with the
arguments after it, so the selected command parses both. I put the command first in these examples
so we can read what will run before its options. A leading separator is accepted for wrappers that
require one:

```terminal
# The separator is removed before the command parses its
# remaining flags.
scripts/lake.sh exe torchlean -- quickstart_mlp --device cpu --steps 20
```

# Strict Parsing

GridWorld's command has a fixed source-level horizon and doesn't accept `--rollout`:

```terminal
# GridWorld does not accept this spelling of the rollout
# option.
scripts/lake.sh exe torchlean ppo_gridworld \
  --device cpu --updates 1 --rollout 8
```

```terminal +output
[TorchLean] device: cpu
ppo_gridworld: PPO on Lean-native GridWorld (4x4, horizon=64) (device=cpu)
  env: pure Lean dynamics + boundary contract check + formal MDP validity proof available
error: ppo_gridworld: unexpected arguments: [--rollout, 8]
```

That run exits with status `1` before training or writing artifacts. Its runtime selection and
description appear first because the command prints them before the final `checkNoArgs`. In the
runner, `TorchLean.CLI.exitOnError` catches the error and adds the `error:` prefix. Scripts should
test the exit status instead of inferring success from a banner or requiring an error message to
start with the command name.

Strict parsing also catches flags borrowed from a neighbouring command. Because the FNO application
takes `--x` and `--y` for its training arrays, the plausible-looking `--train-x` fails:

```terminal
# A dataset option from another command is rejected by the
# Burgers example.
scripts/lake.sh exe torchlean fno1d_burgers --train-x a.npy
```

```terminal +output
fno1d_burgers: native FNO1D Burgers (device=cpu)
error: fno1d_burgers: unexpected arguments: [--train-x, a.npy]
```

Independently written parsers compose because of the leftover-argument check. A training parser can
consume `--steps`, then hand the remainder to the runtime parser without knowing every device
option itself. At the end, `checkNoArgs` closes that process: an unrecognized flag can't quietly
survive the chain. That's why the errors above show the original unused arguments. When you add a
new option, consuming it and placing its value in the eventual configuration both matter.

# Runtime Flags

Here's what the common runtime parser recognizes:

:::table +header
*
  * Flag
  * Current meaning
*
  * `--device auto|cpu|gpu|cuda`
  * requested execution device
*
  * `--arithmetic native|ieee|complex`
  * arithmetic runtime, where the command supports it
*
  * `--seed N`
  * explicit random seed
*
  * `--show-backend`
  * print selected backend capsules
:::

Per-command help narrows this shared list. For instance, the convolutional classifier advertises
only native arithmetic:

```terminal
# Read the CNN-specific defaults before preparing image
# batches.
scripts/lake.sh exe torchlean cnn --help
```

```
Usage: scripts/lake.sh exe torchlean cnn [options]

Data:
  --x PATH           feature/image NPY file
  --y PATH           class-label NPY file
  --n-total N        rows to load

Training:
  --steps N          optimizer updates
  --batch-size N     dataset items accumulated per update (default: 1)
  --lr X             learning rate
  --log PATH|false   write a TrainLog JSON, or disable logging
  --cuda-mem-watch N sample CUDA allocator state every N updates

Runtime:
  --device auto|cpu|gpu|cuda
  --arithmetic native
  --seed N           runtime seed (default: 0)
  --show-backend     print backend capsules as they are selected
```

Read the `Runtime:` block as this command's contract, not as a copy of the global list.

## Devices

The parser accepts `cpu`, `cuda`, `gpu` (another name for `cuda`) and `auto`, which chooses the
portable CPU runtime. Any other name is rejected with the list of known devices:

```terminal
# An unknown device name fails during parsing rather than
# backend execution.
scripts/lake.sh exe torchlean quickstart_mlp --device rocm --steps 1
```

```terminal +output
error: quickstart_mlp: unknown device rocm (known devices:
cpu|gpu|cuda)
```

The request exits with status `1` and doesn't fall back to CPU.

A CPU-only build rejects a CUDA request at the runtime boundary. Build CUDA support before
requesting that device, and use `--show-backend` to inspect the selected implementation and its
evidence. A device name alone is not a proof that a particular kernel ran.

# CPU And CUDA Builds

CPU execution uses the ordinary build:

```terminal
# Build the CPU configuration before asking the executable
# to use CPU storage.
scripts/lake.sh build
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 20
```

CUDA execution requires a CUDA-enabled LibTorch SDK of torch 2.13 or later and the Lake option:

```terminal
# Set this to the installed SDK before linking the CUDA configuration.
export TORCHLEAN_LIBTORCH_HOME=/path/to/libtorch
scripts/lake.sh -R -K cuda=true build
scripts/lake.sh -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke
```

Since the wrapper isolates CPU and CUDA build profiles, pass `-Kcuda=true` on each CUDA invocation.
`-R` forces Lake to reconfigure the package. Without `cuda=true`, the CUDA symbols fail with a
message to rebuild. Your SDK, CUDA toolkit, and compiler must be compatible
({ref "gpu-and-cuda"}[GPU And CUDA] describes the native build).

Add `--show-backend` to inspect what was selected:

```terminal
# Inspect the contracts selected by this runtime profile.
scripts/lake.sh exe torchlean quickstart_mlp --steps 1 --show-backend
```

For each operation the report names its provider, trust level, VJP source, and reduction order,
then four evidence lines. Wrapped to fit this page, the matrix-multiply entry reads:

```
  matmul: reference.matmul provider=reference trust=checked
      vjp=torchlean-tape reduction=fixed-left
    shape: shape safety for matmul; guarded at runtime by
      portable runtime shape checks
    layout: canonical-tensor layout compatibility for matmul;
      guarded at runtime by typed tensor layout
    value: matmul forward refines its TorchLean semantics;
      uses canonical definition TorchLean.Op.rule? directly
    vjp: matmul torchlean-tape VJP refines its TorchLean
      semantics; uses canonical definition TorchLean.Op.rule? directly
```

For that one-step run the report covers seven operations: `reshape`, `permute`, `matmul`,
`broadcast`, `add`, `relu`, and `mse_loss`. `reduction=fixed-left` records a specific summation
order, which matters when reproducing floating-point results {Informal.citep goldberg1991}[].
`trust=checked` points to runtime guards and tests, while the two value/VJP lines say that the
reference implementation uses the canonical definition directly. Those lines don't identify a
kernel-checked refinement proof.

Capsule lines may interleave with the application's output, so redirect to a file and search for
`provider=` to find the report. {ref "backend-selection"}[Backend Selection] explains what the
fields mean and where they come from.

The device identifies where operations run, while the capsule identifies the implementation
selected for each operation on that device.

A build flag and a runtime flag act at different stages. `-Kcuda=true` changes the build
configuration and linked backend support, while `--device cuda` asks an already built executable to
select that backend. Recording only the latter leaves out whether the executable could satisfy the
request. Capsule reporting describes selection before execution, so a report can precede a
missing-implementation error. A successful native call requires the linked implementation and a
usable device as well as an admissible capsule.

# Arithmetic

Most model commands use native `Float32`, but the arithmetic-polymorphic quickstarts and numerical
workflows also accept `--arithmetic ieee`, which selects FloatLib's executable binary32:

```terminal
# Keep the model configuration fixed while changing the
# arithmetic implementation.
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 1 --seed 2026
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 1 --seed 2026 --arithmetic ieee
```

Native arithmetic executes hardware floating-point operations, while `--arithmetic ieee` selects
`ExecFloat.Binary 8 23`, with binary32 rounding defined in FloatLib. Compare the reported losses
and held-out prediction. Running both can expose a disagreement between the executable reference
and the runtime, but matching decimal output alone doesn't compare every bit or cover other inputs.
You'll find the proof statements and their assumptions in {ref "fp32-soundness"}[Float32
Soundness], and {ref "floats"}[Floating-Point Semantics] describes the scalar types
{Informal.citep flocq2011}[].

A command that cannot implement the requested semantics says so instead of approximating:

```terminal
# Complex arithmetic is not an available training mode for
# this command.
scripts/lake.sh exe torchlean quickstart_mlp --steps 1 --arithmetic complex
```

```terminal +output
error: quickstart_mlp: TorchLean.Trainer: supervised training
supports native or IEEE arithmetic; complex arithmetic requires an explicit
complex-valued training API
```

The shared parser recognizes names across commands, but each selected command still checks whether
it supports the requested arithmetic and device.

For complex training, the explicit-state example has its own command:

```terminal
scripts/lake.sh exe torchlean complex_regression \
  --steps 50 --save-checkpoint /tmp/complex-regression.state
```

That command minimizes a real squared-residual loss over complex binary32 weights and bias, then
prints a held-out complex prediction. `--save-checkpoint` saves both complex components of the
final state. Its coordinate-forward gradient is intended for small models, not as a native complex
GPU training benchmark.

Device and arithmetic describe two choices: where tensor operations run and how scalar arithmetic
is implemented. It's useful to record them separately even when only some combinations are
implemented. The native and IEEE `quickstart_mlp` commands make a deliberately small comparison
with the same training arguments. Reading a numerical difference as an arithmetic effect requires
keeping the initialization, data, optimizer settings, and number of updates fixed as well. A
completed command establishes that its selected path ran, and the printed loss describes that run's
objective.

# The Interactive Device Chooser

Use `--choose` when running a command by hand and you do not want to remember the device flag:

```terminal
# Choose the device interactively for the selected
# quickstart command.
scripts/lake.sh exe torchlean --choose quickstart_mlp --steps 1
```

```
TorchLean runtime chooser
Runtime device:
  1) CPU    portable default
  2) CUDA   GPU runtime, requires `scripts/lake.sh -K cuda=true exe ...`
Select device [1]:
```

Pressing Enter selects CPU and the run proceeds. We keep the chooser opt-in so shell scripts,
tests, and continuous integration never block waiting for input.

The command registry records which examples accept device selection, and the chooser prompts only
for those. Fixed-runtime commands run directly, while help, unknown commands, and incomplete
invocations are handled without prompting. For example:

```terminal
# A command with a fixed runtime can bypass the device
# chooser.
scripts/lake.sh exe torchlean --choose quickstart_tensors
```

That command runs the tensor quickstart without a prompt. A `--device` given on the command line is
preserved and suppresses the prompt, but the selected command still validates that value.

# Command-Specific Flags

Ask the subcommand:

```terminal
# Inspect the text model dimensions, data path, and
# evaluation options together.
scripts/lake.sh exe torchlean chargpt --help
```

```
torchlean chargpt: character-level GPT training

Usage:
  scripts/lake.sh -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare \
    --preset PRESET [flags]

Presets:
  smoke       two-update end-to-end CUDA check
  karpathy    full Tiny Shakespeare lecture experiment

Architecture:
  --width N       embedding width
  --heads N       attention heads; must divide width
  --layers N      Transformer blocks
  --dropout P     dropout probability in [0, 1)
  --batch-size N  training windows per update
  --seq-len N     context length
  --steps N       optimizer updates
  --lr FLOAT      AdamW learning rate
  --eval-every N  validation cadence; 0 disables intermediate evaluation
  --eval-iters N  batches averaged at each validation point
  --load-checkpoint P restore model state before training
  --save-checkpoint P save final model state

Notes:
  - Training and validation use disjoint 90/10 corpus splits.
  - CPU and CUDA runs both use native Float32 arithmetic.
```

Presets, architecture and training options, and checkpoint paths appear in that help, along with
constraints such as the requirement that `--heads` divide `--width`. Batch size is set with
`--batch-size`, and `--batch` is rejected.

Command-specific help isn't equally detailed for every application. If a generic runtime page is
printed, read the module documentation or the parser next to that application (the source links in
the preceding chapters point at the maintained definitions).

Dimensions and training controls share a group in the text-model help because they constrain each
other. Context length determines how many token positions participate in an example, and batch size
determines how many such examples are processed together. An evaluation batch count controls how
many losses are averaged, so changing it changes the measurement even when the model parameters are
fixed. When reproducing a reported result, keep these values alongside the seed and step count. A
command name alone doesn't identify the experiment.

# Preparing Data

Synthetic examples need no dataset download. Real-data examples have separate preparation steps
(check the command, because some text workflows can download their corpus). One script prepares
several public datasets:

```terminal
# Prepare the input datasets before running the examples
# that consume them.
python3 scripts/datasets/download_example_data.py \
  --auto-mpg --tiny-shakespeare --tinystories-valid --cifar10
```

That script writes under `data/real/`, and `--cifar10-limit-train -1` opts into all 50000 training
images instead of the small default subset. Forecasting and the Burgers operator use dedicated
preparation:

```terminal
# Convert the source data into the formats expected by the
# sequence and operator examples.
python3 scripts/datasets/download_example_data.py \
  --household-power --household-power-windows 512

python3 TorchLeanExamples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

Convert a local labeled image folder with:

```terminal
# Match the converted image dimensions and labels to the
# image training command.
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/images \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 \
  --labels-from-dirs --limit 2000
```

These Python tools download and preprocess data, including image decoding and array conversion.
Then the Lean loaders check file existence, array metadata, dimensions, finiteness where required,
and conversion to typed tensors. Those checks don't establish that the downloaded labels or the
preprocessing script are semantically correct. {ref "datasets-loaders"}[Datasets And Loaders]
describes the loader contracts.

Data preparation determines the interface that the trainer will see. Resizing an image changes its
spatial dimensions, choosing a label column changes the target, and converting a sequence fixes the
ordering in which observations are presented. These choices should agree with the model
configuration before training begins. A limit on input rows is also different from a minibatch
size: one limits the available dataset, while the other controls how examples are grouped for an
update. Conversion commands and training commands expose different parts of that contract.

# Artifact Paths

Pass explicit paths when an output will be inspected later:

```terminal
# Save the autoencoder training log at an explicit path.
scripts/lake.sh exe torchlean autoencoder --device cpu \
  --n-total 1 --steps 1 \
  --log /tmp/autoencoder-trainlog.json
```

```terminal
# Keep the policy checkpoint, rollout record, and training
# log as separate artifacts.
scripts/lake.sh exe torchlean ppo_gridworld --device cpu \
  --updates 1 --eval-every 1 \
  --log /tmp/ppo-trainlog.json \
  --policy /tmp/ppo-policy.json \
  --path /tmp/ppo-path.json
```

```terminal
# Write the sampled image to the requested PPM file.
scripts/lake.sh -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 8 --steps 20 --T 20 \
  --sample-ppm /tmp/diffusion-sample.ppm
```

`--log PATH|false` appears in most training commands, and passing `false` turns logging off rather
than writing to a file named `false`. That path is part of the experiment: it tells a widget,
plotting script, or checker which exact object it is reading. {ref "widgets"}[Interactive Widgets]
consumes these files.

An artifact path gives a run a destination, but it doesn't by itself describe the artifact's
provenance. Keep the command, configuration, and input identity with a checkpoint or log if it will
be compared with another run later. Explicit paths are particularly useful when invoking the same
example repeatedly: a default filename otherwise refers to whichever run wrote it most recently.
PPO's three outputs serve different purposes, so preserving a log doesn't replace preserving the
policy parameters or the recorded rollout.

# Verification Commands

List the registry, then choose one workflow:

```terminal
# Discover the checker names, then run the in-memory IBP
# workflow explicitly.
scripts/lake.sh exe verify -- list
scripts/lake.sh exe verify -- torchlean-ibp
```

Each row below is one `Tool` record in
{src "TorchLeanExamples/Verification/CLI.lean"}[`TorchLeanExamples/Verification/CLI.lean`], and the
last column is its `includeInAll` field:

:::table +header
*
  * Tool
  * Reads
  * In `all`
*
  * `ibp-cert`
  * IBP certificate for a 3 → 4 → 2 ReLU MLP
  * yes
*
  * `ir`
  * typed graph structure, dtype checks, and optional binary sample comparison
  * no
*
  * `camera-box3d-cert`
  * 3D camera-box projection certificate
  * yes
*
  * `pinn-cert`
  * PINN certificate recomputation check
  * yes
*
  * `spline-cert`
  * piecewise-polynomial certificate
  * yes
*
  * `pinn-cli`
  * interactive PINN residual bounding
  * no
*
  * `abcrown-leaf`
  * α,β-CROWN leaf artifact consistency
  * yes
*
  * `crown-query`
  * exact rational CROWN output query
  * no
*
  * `torchlean-ibp`
  * TorchLean to IR to IBP, in memory
  * no
*
  * `torchlean-transformer-ibp`
  * the same for attention and encoder blocks
  * no
*
  * `torchlean-crown-ops`
  * IBP and CROWN for softmax and loss operations
  * no
*
  * `torchlean-mlp-workflow`
  * train, then check robustness
  * no
*
  * `pinn-dataset-check`
  * pointwise interval containment for a dataset
  * no
*
  * `ode`
  * ODE enclosure with network sub- and super-solutions
  * no
*
  * `vnncomp-mnistfc`
  * VNN-COMP-style MNIST-FC suite
  * no
*
  * `twostage-torchlean-cegis-van`
  * all-in-Lean two-stage Lyapunov refinement
  * no
:::

`ibp-cert`, `torchlean-ibp` and `torchlean-transformer-ibp` use interval bound propagation
{Informal.citep gowal2018}[], and the CROWN tools use the linear relaxations of the LiRPA framework
{Informal.citep autolirpa2020}[]. The other names refer to α,β-CROWN
{Informal.citep betacrown2021}[] for `abcrown-leaf`, physics-informed residuals
{Informal.citep pinn2019}[] for the PINN tools, and neural Lyapunov functions
{Informal.citep neurallyapunov2019}[] for the two-stage workflow.

Both in-memory computations and readers of external artifacts appear in the table. An in-memory
workflow such as `torchlean-ibp` builds a model, lowers it to IR, and runs a bound algorithm inside
the current process:

```terminal
# This workflow builds and lowers its model inside the
# current process.
scripts/lake.sh exe verify -- torchlean-ibp
```

```terminal +output
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] arithmetic: native binary32
lowered IR nodes: 18
output box lo: [1.903999]
output box hi: [2.256001]
```

That workflow computes the output box from its built-in model. An artifact checker such as
`pinn-cert` or `abcrown-leaf` parses a file produced elsewhere and applies its declared checks.
Declared checks differ between checkers: the α,β-CROWN leaf reader checks that the leaves cover the
root and clear their thresholds without recomputing the bounds, while a semantic certificate
checker must also establish the claimed bounds. That distinction matters for proof-carrying code
{Informal.citep necula1997}[]: the producer may be untrusted as long as the consumer can check the
evidence. A checker accepts only its declared schema and semantic fragment, and it doesn't
retroactively verify the process that produced the artifact.

## `all` Coverage

```terminal
# Run the registry entries whose includeInAll field is
# enabled.
scripts/lake.sh exe verify -- all
```

runs the bundled checks marked `yes` above, one after another:

```terminal +output
== ibp-cert ==
IBP certificate accepted: serialized bounds contain Lean's host-Float recomputation.
...
== abcrown-leaf ==
[artifact] Checked 1 leaves: ok=1, bad=0
[artifact] consistent: the leaves cover the root and every leaf clears its threshold.
[artifact] The lower bounds are the producer's claims; TorchLean did not recompute them.
```

Skipped tools are marked `no` above: interactive, externally dependent, long-running, or awaiting a
caller-supplied artifact. That list includes every `torchlean-*` in-memory workflow and `ir`, so a
green `all` is not evidence that model lowering and bound propagation still work. Run those by
name.

In the leaf reader's output, `ok=1` records acceptance under its consistency checks. Interpret each
message against the tool's contract before treating it as evidence of a semantic bound.

`verify all` runs each included tool with an empty argument list. Extra arguments after `all`
aren't forwarded as shared checker options, so to choose an artifact path or another tool-specific
argument, invoke that tool on its own. This behavior differs from the strict per-command parsers
illustrated earlier, and it follows from the `all` branch of the verification dispatcher rather
than from `checkNoArgs`.

# Validation Sequence

After a fresh clone or a substantial local change, run these commands in order:

```terminal
# Check elaboration, small executions, training, and
# verification in sequence.
scripts/lake.sh build
scripts/lake.sh build TorchLeanExamples
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 20 --seed 2026
scripts/lake.sh exe verify -- torchlean-ibp
```

Those commands check different things:

:::table +header
*
  * Command
  * Checks
*
  * `scripts/lake.sh build`
  * whether the project elaborates and links
*
  * `scripts/lake.sh build TorchLeanExamples`
  * whether the curated examples elaborate
*
  * tensor/autograd quickstarts
  * whether small executable values and derivatives behave as expected
*
  * MLP training
  * whether a complete optimizer path runs
*
  * TorchLean IBP
  * whether model lowering and bound propagation complete
:::

For CUDA changes, follow with CUDA-specific builds and runtime checks. For external tools, prepare
their dependencies separately. Their dependencies and acceptance conditions differ from the CPU
checks above.

# Building The Website

You build the guide, API reference, and application pages with:

```terminal
# Regenerate the blueprint before serving the website that
# includes it.
scripts/docs/build_site.sh
```

For a local Jekyll preview after the generated site exists:

```terminal
# Serve the generated site from the website directory.
cd home_page
bundle _2.3.14_ exec jekyll serve \
  --config _config.yml,_config_dev.yml \
  --host 127.0.0.1 --port 4001
```

Then open:

```
http://127.0.0.1:4001/
http://127.0.0.1:4001/blueprint/
http://127.0.0.1:4001/examples/
http://127.0.0.1:4001/docs/
```

If the source changed but the browser didn't, rebuild the generated site before restarting Jekyll.
Building the blueprint checks the document's executable pieces and regenerates the guide, while
Jekyll serves the resulting files. Restarting the server against an older generated tree will still
show the older examples. After rebuilding, read the edited prose beside its output blocks in the
served page: elaboration checks the executable pieces, while this reading checks whether the
explanation still makes sense with them.

# Command Exercises

1. Run `scripts/lake.sh exe torchlean --list | wc -l` and account for the commands across the
   groups shown by `--help`.
2. Pick any training command and pass a flag from a different one. Read the message and identify
   which parser rejected it.
3. Run one command twice with `--seed 1` and `--seed 2`, then twice with the same seed, and confirm
   which pairs agree exactly.
4. Run `quickstart_mlp --steps 1` with `--arithmetic native` and `--arithmetic ieee` and diff the
   two outputs. Then try `--steps 200` and compare the losses and predictions again.
5. Redirect `--show-backend` output to a file and count the distinct `provider=` values.
6. Run `scripts/lake.sh exe verify -- all`, then run each `torchlean-*` tool by hand, and note how
   much of the registry a green `all` left untouched.
7. Write a `--log` under a new directory and inspect the created parent directories. Then try a
   path whose parent is a regular file and inspect the reported write failure.

# Command Results

A successful process, a backend selection report, and an accepted certificate answer different
questions. When recording a result, keep the command, selected implementation, and the tool's
actual acceptance condition together. {ref "verification"}[Verification Overview] describes the
claims that the verification commands check.
