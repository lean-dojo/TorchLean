import VersoManual
import NN.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Command-Line Reference" =>
%%%
tag := "cli"
%%%

TorchLean has two command dispatchers. `torchlean` runs models, demonstrations, and data checks;
`verify` runs verification workflows and artifact checkers:

```
# Replace the placeholders with a registered command and its
# own flags.
lake exe torchlean <example> [flags...]
lake exe verify -- <tool> [args...]
```

Both commands can print numerical results. A training command's success means it completed its
runtime checks; a verification command's success depends on that tool's declared checks. For
example, validating an artifact's schema is different from establishing its claimed bounds.

The dispatch tables are ordinary Lean definitions:

- {src "NN/Examples/Runner.lean"}[`NN/Examples/Runner.lean`] routes the `torchlean` subcommands;
- {src "NN/Verification/CLI.lean"}[`NN/Verification/CLI.lean`] routes the `verify` tools.

The transcripts illustrate these workflows. Help from your executable remains authoritative for
command names and flags; numerical output and backend reports can change with the build.

# Discovering Commands

The dispatcher provides grouped help and a machine-readable list:

```terminal
# Compare grouped help with the flat list used by scripts.
lake exe torchlean --help
lake exe torchlean --list
```

`--help` explains the command groups and gives a starting path. `--list` prints exactly one
registered command name per line, which is what you want in a script. Running `lake exe torchlean`
with no arguments prints the same help and exits with status `1`, so a bare invocation in a
Makefile fails rather than silently doing nothing.

The full help is the inventory of the runnable surface:

```
TorchLean runnable examples

Usage:
  lake exe torchlean <example> [flags...]
  lake exe torchlean --choose <example> [flags...]
  lake exe torchlean <example> --help
  lake exe torchlean --list

Start here:
  lake exe torchlean quickstart_tensors
  lake exe torchlean quickstart_autograd
  lake exe torchlean quickstart_mlp --steps 20

Quickstarts:
  quickstart_tensors - construct, transform, and print typed tensors
  quickstart_autograd - differentiate tensor functions and model losses
  quickstart_mlp - train and evaluate a small regression MLP

Supervised and vision:
  mlp - train an MLP on supervised CSV data
  kan - train a Kolmogorov-Arnold network on CSV data
  lstm_regression - forecast a scalar sequence with an LSTM
  cnn - train a convolutional image classifier
  resnet - train a residual image classifier
  vit - train a compact vision transformer

Sequence:
  rnn - train a vanilla recurrent text model
  lstm - train an LSTM text model
  transformer - train a small transformer text model
  chargpt - train an indexed character-level GPT
  gpt2 - train and sample a byte-level GPT
  gpt2_saved - load a saved GPT checkpoint and generate text
  text_gpt2 - train byte or BPE GPT models on a text corpus
  gpt_adder - train a GPT on digit addition
  mamba - train a compact Mamba-style sequence model

Generative and operator learning:
  autoencoder - train an image autoencoder
  mae - train a masked image autoencoder
  diffusion - train and sample an image diffusion model
  fno1d_burgers - learn the one-dimensional Burgers operator

Reinforcement learning:
  ppo_cartpole - train and evaluate PPO on CartPole
  ppo_gridworld - train PPO against a checked GridWorld contract
  ppo_pong_ram - train PPO on Pong RAM observations
  dqn_replay - inspect checked DQN replay-buffer behavior

PyTorch interop:
  pytorch_roundtrip - round-trip model weights through PyTorch

Data:
  data_csv - load and validate supervised CSV tensors
  data_npy - load and validate NumPy tensor files
  data_cifar10 - load prepared CIFAR-10 image tensors

Numerical checks:
  factorizations - check Cholesky and QR reconstruction properties
  transcendentals - check transcendental autograd rules

Deep dives:
  autograd_transforms - run Jacobian, Hessian, JVP, VJP, and detach examples
  floats_arb_ieee_compare - compare arbitrary precision and IEEE execution
  float32_semantics - inspect executable binary32 semantics
  numerical_certificate - check a graph-level numerical certificate
  graphspec - build and inspect a typed graph specification
  ir_axis_ops - lower axis operations into executable graph IR
  one_semantic_universe - relate executable and proof-oriented scalar semantics
  torch_ir_pytorch - export TorchLean IR for PyTorch execution

Runtime flags:
  --choose                         ask for runtime choices before running
  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external
  --arithmetic native|ieee|complex
      arithmetic availability depends on the example; check its --help
  --execution eager|typed-graph
  --seed N
  --show-backend

Verification commands live under `lake exe verify -- list`.
Use `lake exe torchlean <example> --help` for command-specific flags.
```

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
lake exe torchlean nosuchthing
```

```
Unknown example: nosuchthing

TorchLean runnable examples
...
```

The exit status is `1`.

# Command Implementation

The MLP quickstart shows how the shared parser connects to a trainer. Its `main` first handles help
and arguments, then constructs the training configuration:

```
-- Dispatch selects the registered command; that command
-- parses its remaining arguments.
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let flags ← parseFlags exeName args (defaultSteps := 200)

  let trainer := Trainer.new model
    { flags.runtime with
        objective := .meanSquaredError
        optimizer := optim.adam { learningRate := 0.03 }
        seed := flags.seed }
  ...
```

The command drops a leading `--`, answers `--help` if present, and parses the remaining arguments
before constructing the trainer. The full file is
{src "NN/Examples/Quickstart/SimpleMlpTrain.lean"}[`SimpleMlpTrain.lean`]. Its `parseFlags`
helper in
{src "NN/Examples/Quickstart/Common.lean"}[`Common.lean`] passes each parser the previous remainder:

```
-- Each parser consumes its own flag and passes the
-- remainder to the next parser.
def parseFlags (exeName : String) (args : List String) (defaultSteps : Nat) : IO Flags := do
  let (seed, args) ← CLI.seed exeName args
  let (steps, args) ← CLI.positiveNatFlag exeName args "steps" defaultSteps
  let runtime ← CLI.Trainer.parseCommandLine exeName args
  pure { seed, steps, runtime }
```

Each binding of `args` contains only the arguments left by the previous parser. After consuming
`--seed` and `--steps`, `CLI.Trainer.parseCommandLine` handles runtime flags and checks for anything
left over. The shared functions in {src "NN/API/CLI.lean"}[`NN/API/CLI.lean`] expose this remainder
in their return types:

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

The four `take...` functions return the parsed optional value together with the remaining
arguments. Each can also return an error message. `checkNoArgs` returns only `Unit` on success,
and succeeds only for an empty list. Calling it on the final remainder rejects flags that no
parser consumed.

The final remainder check matters in any parser. Python's `argparse.parse_args` also rejects unknown
arguments by default; `parse_known_args` instead returns them to its caller. Here, `checkNoArgs`
makes that final validation explicit.

The `Except` and `Option` in these signatures answer separate questions. `Except.error` means
that the supplied command line cannot be used. Inside `Except.ok`, `none` means that this
particular flag was absent, while `some value` means it was present and parsed. This distinction
lets a command supply a default without silently replacing a misspelled number. The second
component is just as important: it records the arguments that another parser still has to
account for. A successful individual parser does not establish that the whole command line is
valid.

# Parser Results

The parsers can be evaluated directly. This call consumes `--steps 20` and leaves the unrelated
path flag in the remainder:

```lean (name := cliTake)
-- The steps parser removes its flag while preserving the
-- unrelated seed flag.
#eval CLI.takeNatFlag?
  ["--steps", "20", "--x", "a.npy"] "steps"
```

```leanOutput cliTake (whitespace := lax)
Except.ok (some 20, ["--x", "a.npy"])
```

The `20` came out, and `--x a.npy` was handed back untouched. The `--key=value` spelling is accepted
as the same thing:

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

The final check reports arguments left unconsumed:

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
-- Parsing constructs a path value; opening the file happens
-- later.
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

The two remaining helpers handle the shape of the command line rather than its contents. A leading
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

The path and Boolean outputs above describe parsing, not the effects of running the command.
`FilePath.mk` does not check whether a file exists, whether its contents have the required shape,
or whether its directory is writable. Those checks belong to the reader or writer that uses the
path. Similarly, successfully parsing `false` says which Boolean was requested; the command must
still use that value when it constructs its configuration. Following the parsed value into the
configuration is the useful next step when a flag appears to have no effect.

# Command Grammar

The normal form is:

```
# Runtime and command-specific flags share the argument list
# after the command.
lake exe torchlean <subcommand> [runtime flags] [command flags]
```

For example:

```terminal
# Select CPU execution while setting the training length and
# initialization seed.
lake exe torchlean quickstart_mlp \
  --device cpu --steps 20 --seed 2026
```

Runtime flags may also precede the subcommand:

```terminal
# The dispatcher also accepts the runtime selection before
# the command name.
lake exe torchlean --device cpu quickstart_mlp \
  --steps 20 --seed 2026
```

Both forms reach the same parser, because the runtime flags are removed from the list wherever they
appear. Documentation uses the first because the application name reads better before its options.
A leading separator is accepted for wrappers that require one:

```terminal
# The separator is removed before the command parses its
# remaining flags.
lake exe torchlean -- quickstart_mlp --device cpu --steps 20
```

# Strict Parsing

The GridWorld command has a fixed source-level horizon and does not accept `--rollout`:

```terminal
# GridWorld does not accept this spelling of the rollout
# option.
lake exe torchlean ppo_gridworld \
  --device cpu --updates 1 --rollout 8
```

```terminal +output
[TorchLean] execution: eager
[TorchLean] device: cpu
ppo_gridworld: PPO on Lean-native GridWorld (4x4, horizon=64) (device=cpu)
  env: pure Lean dynamics + boundary contract check + formal MDP validity proof available
error: ppo_gridworld: unexpected arguments: [--rollout, 8]
```

The command exits with status `1` before training or writing artifacts. Its runtime selection and
description appear first because it prints them before the final `checkNoArgs`. The runner's
`TorchLean.CLI.exitOnError` catches the error and adds the `error:` prefix. Scripts should test
the exit status rather than infer success from a banner or require an error message to start with
the command name.

The same rule catches flags borrowed from a neighbouring command. The FNO application takes `--x`
and `--y` for its training arrays, so the plausible-looking `--train-x` fails:

```terminal
# A dataset option from another command is rejected by the
# Burgers example.
lake exe torchlean fno1d_burgers --train-x a.npy
```

```terminal +output
fno1d_burgers: native FNO1D Burgers (device=cpu)
error: fno1d_burgers: unexpected arguments: [--train-x, a.npy]
```

The leftover-argument check makes independently written parsers compose. A training parser can
consume `--steps`, then hand the remainder to the runtime parser without knowing every device
option itself. The final `checkNoArgs` closes that process: an unrecognized flag cannot quietly
survive the chain. This is why the errors above show the original unused arguments. When adding
a new option, both consuming it and placing its value in the eventual configuration matter.

# Runtime Flags

The common runtime parser recognizes:

:::table +header
*
  * Flag
  * Current meaning
*
  * `--device auto|cpu|cuda|...`
  * requested execution device
*
  * `--arithmetic native|ieee|complex`
  * arithmetic runtime, where the command supports it
*
  * `--execution eager|typed-graph`
  * eager autograd or shape-indexed typed graph host path where supported
*
  * `--seed N`
  * explicit random seed
*
  * `--show-backend`
  * print selected backend capsules
:::

Per-command help narrows this shared list. The convolutional classifier, for instance, advertises
only native arithmetic:

```terminal
# Read the CNN-specific defaults before preparing image
# batches.
lake exe torchlean cnn --help
```

```
Usage: lake exe torchlean cnn [options]

Data:
  --x PATH           feature/image NPY file
  --y PATH           class-label NPY file
  --n-total N        rows to load

Training:
  --steps N          optimizer updates
  --batch-size N     dataset items accumulated per update
  --lr X             learning rate
  --log PATH|false   write a TrainLog JSON, or disable logging
  --cuda-mem-watch N sample CUDA allocator state every N updates

Runtime:
  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external
  --execution eager|typed-graph
  --arithmetic native
  --seed N --show-backend
```

Read the `Runtime:` block as this command's contract, not as a copy of the global list.

## Devices

The parser accepts `rocm`, `metal`, `wasm`, `tpu`, `trainium`, `custom`, and `external`. They are
planning targets in the backend registry, not completed runtimes, and asking for one says so:

```terminal
# A known device name can still lack an implementation for
# this command.
lake exe torchlean quickstart_mlp --device rocm --steps 1
```

```terminal +output
error: quickstart_mlp: device `rocm` has no maintained
runtime profile; use a programmatic backend profile
```

A name outside the registry fails differently, and lists the registry:

```terminal
# An unknown device name fails during parsing rather than
# backend execution.
lake exe torchlean quickstart_mlp --device banana --steps 1
```

```terminal +output
error: quickstart_mlp: unknown device banana (known targets:
cpu | cuda | rocm | metal | wasm | tpu | trainium | custom | external)
```

Both requests exit with status `1`; neither silently selects CPU. An unsupported device and an
unrecognized device name are separate errors.

A CPU-only build rejects a CUDA request at the runtime boundary. Build CUDA support before
requesting that device, and use `--show-backend` to inspect the selected implementation and its
evidence. A device name alone is not a proof that a particular kernel ran.

# CPU And CUDA Builds

CPU execution uses the ordinary build:

```terminal
# Build the CPU configuration before asking the executable
# to use CPU storage.
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 20
```

Real CUDA execution must be compiled with the Lake option:

```terminal
# Link the CUDA configuration before selecting CUDA at
# runtime.
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke
```

Keep `-R` when changing a Lake configuration so affected native archives are rebuilt. Without
`cuda=true`, TorchLean links portable CUDA stubs.

Add `--show-backend` to inspect what was selected:

```terminal
# Inspect the capabilities compiled into this executable.
lake exe torchlean quickstart_mlp --steps 1 --show-backend
```

The report names, for each operation, its provider, trust level, VJP source, and reduction order,
then four evidence lines. Wrapped to fit this page, the matrix-multiply entry reads:

```
  matmul: reference.matmul provider=reference trust=checked
      vjp=torchlean-tape reduction=fixed-left
    shape: shape safety for matmul; guarded at runtime by
      portable runtime shape checks
    layout: canonical-tensor layout compatibility for matmul;
      guarded at runtime by typed tensor layout
    value: matmul forward refines its TorchLean semantics;
      covered by test suite NN.Tests.Runtime.Floats.Suite
    vjp: matmul torchlean-tape VJP refines its TorchLean
      semantics; covered by test suite NN.Tests.Runtime.Floats.Suite
```

For that one-step run the report covers seven operations: `reshape`, `permute`, `matmul`,
`broadcast`, `add`, `relu`, and `mse_loss`. `reduction=fixed-left` records a specific summation
order, which matters when reproducing floating-point results ({Informal.citep goldberg1991}[]).
The `trust=checked` entry points to runtime guards and tests; the two value/VJP lines name the test
suite that supports those claims. They do not identify a kernel-checked refinement proof.

Capsule lines may interleave with the application's output. Redirect to a file and search for
`provider=` to find the report. {ref "backend-selection"}[Backend Selection] explains
what the fields mean and where they come from.

Device selection answers "where did this run?"; capsule reporting answers the more precise question
"which implementation was selected for each operation?"

A build flag and a runtime flag act at different stages. `-Kcuda=true` changes the build
configuration and linked backend support. `--device cuda` asks an already built executable to
select that backend. Recording only the latter leaves out whether the executable could satisfy
the request. The backend capsule helps resolve this ambiguity by reporting compiled capabilities;
the command's own runtime messages then show the path selected for that particular run.

# Arithmetic

Most model commands use native `Float32`. The arithmetic-polymorphic quickstarts and numerical
workflows also accept `--arithmetic ieee`, which swaps the scalar type for TorchLean's executable
software binary32:

```terminal
# Keep the model configuration fixed while changing the
# arithmetic implementation.
lake exe torchlean quickstart_mlp --steps 1
lake exe torchlean quickstart_mlp --steps 1 --arithmetic ieee
```

```
steps=1 arithmetic=native scalar=Float32 loss=1.159370 -> 1.100337
steps=1 arithmetic=ieee scalar=IEEE32Exec loss=1.159370 -> 1.100337
```

The two implementations agree to six printed decimals on this update. Native arithmetic executes
hardware floating-point operations; `IEEE32Exec` computes binary32 operations with rounding defined
in Lean. Running both can expose a disagreement between the executable reference and the runtime,
although this transcript neither compares every bit nor covers other inputs. The proof statements
and their assumptions are in {ref "fp32-soundness"}[Float32 Soundness];
{ref "floats"}[Floating-Point Semantics] describes the scalar types ({Informal.citep flocq2011}[]).

A command that cannot implement the requested semantics says so instead of approximating:

```terminal
# Complex arithmetic is not an available training mode for
# this command.
lake exe torchlean quickstart_mlp --steps 1 --arithmetic complex
```

```terminal +output
error: quickstart_mlp: TorchLean.Trainer: supervised training
supports native or IEEE arithmetic; complex arithmetic requires an explicit
complex-valued training API
```

The shared parser recognizes names across commands. Each selected command still checks whether it
supports the requested arithmetic, device, and execution mode.

Device, arithmetic, and execution mode describe three choices: where tensor operations run,
how scalar arithmetic is implemented, and how the program is evaluated. They are useful to
record separately even when only some combinations are implemented. The two loss lines above
come from a deliberately small comparison with the same training arguments. Reading a numerical
difference as an arithmetic effect requires keeping the initialization, data, optimizer settings,
and number of updates fixed as well. A completed command establishes that its selected path ran;
the printed loss describes that run's objective.

# Execution Modes

`--execution typed-graph` selects the shape-indexed graph host path for commands that implement it.
For the MLP quickstart the two paths agree:

```terminal
# Keep the training arguments fixed while comparing the two
# execution modes.
lake exe torchlean quickstart_mlp --steps 1 --execution eager
lake exe torchlean quickstart_mlp --steps 1 --execution typed-graph
```

Both print `loss=1.159370 -> 1.100337` and the same held-out prediction. This is one numerical
comparison of the eager and typed-graph paths. The flag selects a shape-indexed host execution
path; it does not request CUDA graph capture, compiler optimization, or a derivative proof.
Some specialized CUDA applications require eager execution.
{ref "execution-modes"}[Execution Modes] compares the paths.

# The Interactive Device Chooser

Use `--choose` when running a command by hand and you do not want to remember the device flag:

```terminal
# Enter the interactive dispatcher without selecting a
# command in advance.
lake exe torchlean --choose quickstart_mlp --steps 1
```

```
TorchLean runtime chooser
Runtime device:
  1) CPU    portable default
  2) CUDA   GPU runtime, requires `lake -R -K cuda=true exe ...`
Select device [1]:
```

Pressing Enter selects CPU and the run proceeds. The chooser is opt-in so shell scripts, tests, and
continuous integration never block waiting for input, and it offers only the two implemented
runtimes rather than the whole device registry.

The command registry records which examples accept device selection. The chooser prompts only for
those examples; fixed-runtime commands run directly, and help, unknown commands, and incomplete
invocations are handled without prompting. For example:

```terminal
# A command with a fixed runtime can bypass the device
# chooser.
lake exe torchlean --choose quickstart_tensors
```

This runs the tensor quickstart directly. An explicit `--device` is preserved and suppresses the
prompt; its value is still validated by the selected command.

# Command-Specific Flags

Ask the subcommand:

```terminal
# Inspect the text model dimensions, data path, and
# evaluation options together.
lake exe torchlean chargpt --help
```

```
torchlean chargpt: character-level GPT training

Usage:
  lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare \
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

The help lists presets, architecture and training options, and checkpoint paths. It also gives
constraints such as the requirement that `--heads` divide `--width`. The batch flag is
`--batch-size`; `--batch` is rejected.

Not every application has equally detailed command-specific help yet. If a generic runtime page is
printed, read the module documentation or the parser next to that application; the source links in
the preceding chapters point at the maintained definitions.

The text-model help groups dimensions with training controls because they constrain each other.
Context length determines how many token positions participate in an example; batch size determines
how many such examples are processed together. An evaluation batch count controls how many losses
are averaged, so changing it changes the measurement even when the model parameters are fixed.
When reproducing a reported result, keep these values alongside the seed and step count. A command
name alone does not identify the experiment.

# Preparing Data

Synthetic examples need no dataset download. Real-data examples have separate preparation steps;
check the command because some text workflows can download their corpus. The dataset script prepares
several public datasets:

```terminal
# Prepare the input datasets before running the examples
# that consume them.
python3 scripts/datasets/download_example_data.py \
  --auto-mpg --tiny-shakespeare --tinystories-valid --cifar10
```

It writes under `data/real/`, and `--cifar10-limit-train -1` opts into all 50000 training images
instead of the small default subset. Forecasting and the Burgers operator use dedicated preparation:

```terminal
# Convert the source data into the formats expected by the
# sequence and operator examples.
python3 scripts/datasets/download_example_data.py \
  --household-power --household-power-windows 512

python3 NN/Examples/Data/prepare_fno1d_burgers.py \
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
The Lean loaders then check file existence, array metadata, dimensions, finiteness where required,
and conversion to typed tensors. These checks do not establish that the downloaded labels or the
preprocessing script are semantically correct.
{ref "datasets-loaders"}[Datasets And Loaders] describes the loader contracts.

Data preparation determines the interface that the trainer will see. Resizing an image changes
its spatial dimensions, choosing a label column changes the target, and converting a sequence
fixes the ordering in which observations are presented. These choices should agree with the
model configuration before training begins. A limit on input rows is also different from a
minibatch size: one limits the available dataset, while the other controls how examples are
grouped for an update. The conversion and training commands expose different parts of that
contract.

# Artifact Paths

Pass explicit paths when an output will be inspected later:

```terminal
# Save the autoencoder training log at an explicit path.
lake exe torchlean autoencoder --device cpu \
  --n-total 1 --steps 1 \
  --log /tmp/autoencoder-trainlog.json
```

```terminal
# Keep the policy checkpoint, rollout record, and training
# log as separate artifacts.
lake exe torchlean ppo_gridworld --device cpu \
  --updates 1 --eval-every 1 \
  --log /tmp/ppo-trainlog.json \
  --policy /tmp/ppo-policy.json \
  --path /tmp/ppo-path.json
```

```terminal
# Write the sampled image to the requested PPM file.
lake -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 8 --steps 20 --T 20 \
  --sample-ppm /tmp/diffusion-sample.ppm
```

`--log PATH|false` appears in most training commands, and passing `false` turns logging off rather
than writing to a file named `false`. The path is part of the experiment: it tells a widget,
plotting script, or checker which exact object it is reading. {ref "widgets"}[Interactive Widgets]
consumes these files.

An artifact path gives a run a destination, but it does not by itself describe the artifact's
provenance. Keep the command, configuration, and input identity with a checkpoint or log if it
will be compared with another run later. Explicit paths are particularly useful when invoking the
same example repeatedly: a default filename otherwise refers to whichever run wrote it most
recently. The three PPO outputs serve different purposes, so preserving a log does not replace
preserving the policy parameters or the recorded rollout.

# Verification Commands

List the registry, then choose one workflow:

```terminal
# Discover the checker names, then run the in-memory IBP
# workflow explicitly.
lake exe verify -- list
lake exe verify -- torchlean-ibp
```

The registry currently holds 23 tools. Each row below is one `Tool` record in
{src "NN/Verification/CLI.lean"}[`NN/Verification/CLI.lean`], and the last column is its
`includeInAll` field:

:::table +header
*
  * Tool
  * Reads
  * In `all`
*
  * `lirpa-mlp`
  * IBP certificate for a feed-forward MLP
  * yes
*
  * `lirpa-cnn`
  * IBP certificate for a convolution and head
  * yes
*
  * `lirpa-attention`
  * IBP certificate for an attention softmax block
  * yes
*
  * `lirpa-gru`
  * IBP certificate for a GRU gate
  * yes
*
  * `lirpa-encoder`
  * IBP certificate for a transformer encoder block
  * yes
*
  * `camera-box3d-cert`
  * 3D camera-box projection certificate
  * yes
*
  * `pinn-cert`
  * PINN residual-bound certificate replay
  * yes
*
  * `spline-cert`
  * piecewise-polynomial certificate
  * yes
*
  * `abcrown-leaf`
  * α,β-CROWN leaf artifact structure
  * yes
*
  * `margin-report`
  * consistency of an exported logit-bound report
  * yes
*
  * `pinn-cli`
  * interactive PINN residual bounding
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
  * `digits`
  * certified accuracy over sklearn digits
  * no
*
  * `digits-train-certify`
  * train digits, lower to IR, then report
  * no
*
  * `vnncomp-mnistfc`
  * VNN-COMP-style MNIST-FC suite
  * no
*
  * `twostage-pythononly-certgen`
  * Lyapunov certificate from an external script
  * no
*
  * `twostage-hybrid-van-stage2`
  * PyTorch initialization refined in Lean
  * no
*
  * `twostage-torchlean-cegis-van`
  * all-in-Lean two-stage Lyapunov refinement
  * no
:::

The families behind those names are documented where they belong: interval bound propagation
({Informal.citep gowal2018}[]) and the LiRPA framework ({Informal.citep autolirpa2020}[]) for the
`lirpa-*` checkers, α,β-CROWN ({Informal.citep betacrown2021}[]) for `abcrown-leaf`,
physics-informed residuals ({Informal.citep pinn2019}[]) for the PINN tools, and neural Lyapunov
functions ({Informal.citep neurallyapunov2019}[]) for the two-stage workflows.

The table includes both in-memory computations and readers of external artifacts. An in-memory
workflow such as `torchlean-ibp` builds a model, lowers it to IR, and runs a bound algorithm inside
the current process:

```terminal
# This workflow builds and lowers its model inside the
# current process.
lake exe verify -- torchlean-ibp
```

```terminal +output
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] arithmetic: native binary32
lowered IR nodes: 18
output box lo: [1.904000]
output box hi: [2.256001]
```

Nothing was read from disk, and the printed box is a result computed here. An artifact checker such
as `pinn-cert` or `abcrown-leaf` instead parses a file produced elsewhere and applies its declared
checks. These differ: the α,β-CROWN leaf reader validates artifact structure, while a semantic
certificate checker must also establish the claimed bounds. This distinction matters for
proof-carrying code
({Informal.citep necula1997}[]): the producer may be untrusted as long as the consumer can check the
evidence. A checker accepts only its declared schema and semantic fragment, and it does not
retroactively verify the process that produced the artifact.

## `all` Coverage

```terminal
# Run the registry entries whose includeInAll field is
# enabled.
lake exe verify -- all
```

runs 10 of the 23 tools, one after another:

```terminal +output
== lirpa-mlp ==
IBP certificate verified: serialized bounds enclose Lean recomputation.

== lirpa-cnn ==
IBP certificate verified: serialized bounds enclose Lean recomputation.
...
== abcrown-leaf ==
[artifact] Checked 1 leaves: ok=1, bad=0

== margin-report ==
[margin report] examples=360
[margin report] nominal_ok=349 (requires 'pred' in examples)
[margin report] positive_margin=318
```

The 13 tools it skips are the ones marked `no` above: interactive, externally dependent, or
long-running. That list includes every `torchlean-*` in-memory workflow, so a green `all` is not
evidence that model lowering and bound propagation still work. Run those explicitly.

`margin-report` reports a positive margin for 318 of 360 examples, and `all` still exits `0`.
The tool checks report consistency; it does not require every example to have a positive margin.
Likewise, the leaf reader's `ok=1` records acceptance under its structural checks. Interpret each
message against the tool's contract before treating it as evidence of a semantic bound.

There is one dispatch detail worth making explicit here: `verify all` runs each included tool
with an empty argument list. Extra arguments after `all` are not forwarded as shared checker
options. To choose an artifact path or another tool-specific argument, invoke that tool directly.
This behavior differs from the strict per-command parsers illustrated earlier, and it follows
from the `all` branch of the verification dispatcher rather than from `checkNoArgs`.

# Validation Sequence

After a fresh clone or a substantial local change, run these commands in order:

```terminal
# Check elaboration, small executions, training, and
# verification in sequence.
lake build
lake build NNExamples
lake exe torchlean quickstart_tensors
lake exe torchlean quickstart_autograd
lake exe torchlean quickstart_mlp --device cpu --steps 20 --seed 2026
lake exe torchlean numerical_certificate
lake exe verify -- torchlean-ibp
```

They answer different questions:

:::table +header
*
  * Command
  * Question
*
  * `lake build`
  * does the project elaborate and link?
*
  * `lake build NNExamples`
  * do the curated examples elaborate?
*
  * tensor/autograd quickstarts
  * do small executable values and derivatives behave as expected?
*
  * MLP training
  * does a complete optimizer path run?
*
  * numerical certificate
  * are positive *and* negative certificate cases handled?
*
  * TorchLean IBP
  * does model lowering and bound propagation complete?
:::

The numerical-certificate command includes a negative control alongside successful certificate
and replay cases:

```terminal
# Exercise both accepted certificates and the deliberately
# tampered negative case.
lake exe torchlean numerical_certificate
```

```terminal +output
TorchLean numerical runtime certificate
  ok  base certificate
  ok  base IEEE replay
  ok  tampered range rejected
  ok  two-layer MLP certificate
  ok  two-layer MLP IEEE replay
All numerical certificate checks passed.
```

The negative control changes an addition range and confirms rejection. The remaining rows check
successful generation and replay of the small graph and the MLP. For the broader native numerical
checks, use the curated suite; this example keeps the artifact workflow visible.

For CUDA changes, follow with CUDA-specific builds and runtime checks. For external tools, prepare
their dependencies separately. Their dependencies and acceptance conditions differ from the CPU
checks above.

# Building The Website

The guide, API reference, and application pages are built by:

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

If the source changed but the browser did not, rebuild the generated site before restarting Jekyll.
A running web server cannot regenerate Lean documentation on its own.

The blueprint build and the local web server perform separate jobs. The build checks and
regenerates the guide content; the server makes the resulting website available to a browser.
Starting a server against an older generated tree does not update the Lean examples in that tree.
When reviewing an edited chapter, rebuild first, then inspect both the prose and its neighboring
output blocks in the served page. A successful build checks the document's executable pieces;
reading the page checks whether their explanations remain intelligible in context.

# Command Exercises

1. Run `lake exe torchlean --list | wc -l` and compare the count with the ten groups in `--help`.
2. Pick any training command and pass a flag from a different one. Read the message and identify
   which parser rejected it.
3. Run one command twice with `--seed 1` and `--seed 2`, then twice with the same seed, and confirm
   which pairs agree exactly.
4. Run `quickstart_mlp --steps 1` with `--arithmetic native` and `--arithmetic ieee` and diff the
   two transcripts. Then try `--steps 200` and see whether the agreement survives.
5. Redirect `--show-backend` output to a file and count the distinct `provider=` values.
6. Run `lake exe verify -- all`, then run each `torchlean-*` tool by hand, and note how much of the
   registry a green `all` left untouched.
7. Write a `--log` under a new directory and inspect the created parent directories. Then try a
   path whose parent is a regular file and inspect the reported write failure.

# Command Results

A successful process, a backend selection report, and an accepted certificate answer different
questions. When recording a result, keep the command, selected implementation, and the tool's
actual acceptance condition together. {ref "verification"}[Verification Overview] describes the
claims that the verification commands check.