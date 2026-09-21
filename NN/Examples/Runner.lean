/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models
public import NN.Examples.DeepDives.AutogradTransforms
public import NN.Examples.DeepDives.Floats.ArbIEEEExecCompare
public import NN.Examples.DeepDives.Floats.Float32Semantics
public import NN.Examples.DeepDives.Floats.GraphNumericalCertificate
public import NN.Examples.DeepDives.GraphSpec.Tutorial
public import NN.Examples.DeepDives.IRAxisOps
public import NN.Examples.DeepDives.OneSemanticUniverse
public import NN.Examples.DeepDives.TorchIRPyTorch
public import NN.Examples.Quickstart.AutogradBasics
public import NN.Examples.Quickstart.SimpleMlpTrain
public import NN.Examples.Quickstart.TensorBasics
public import NN.Examples.Data.Loaders.Csv
public import NN.Examples.Data.Loaders.Npy
public import NN.Examples.Data.Loaders.Cifar10Images
public import NN.Examples.Factorization.Check
public import NN.Examples.Functional.Transcendentals
public import NN.Examples.Interop.PyTorch.Roundtrip

/-!
# TorchLean Example Runner

Command registry and dispatcher for the `torchlean` example runner. Each example keeps its own
namespace; this module selects one from a subcommand such as `mlp` or `gpt2`.
`NN.Examples.RunnerMain` supplies the global executable entrypoint.
-/

@[expose] public section

open System
open TorchLean

namespace NN.Examples.Runner

/-- One runnable example, as the dispatcher sees it. -/
structure Command where
  /-- Subcommand typed by the user, for example `quickstart_mlp`. -/
  name : String
  /-- One-line summary shown in the usage text; `--list` prints only command names. -/
  description : String
  /-- The example's entry point; the returned code becomes the process exit status. -/
  run : List String → IO UInt32
  /-- Whether the example accepts `--device`, allowing the chooser to supply a missing value. -/
  supportsDevice : Bool := false

/-- A titled section of the usage listing, so related examples appear together. -/
structure CommandGroup where
  /-- Section heading, for example "Quickstarts". -/
  title : String
  /-- Examples in this section, in the order they should be listed. -/
  commands : List Command

/--
Wrap an example whose `main` returns `Unit`: finishing without an exception counts as success.
-/
def unitCommand (name description : String) (run : List String → IO Unit)
    (supportsDevice : Bool := false) : Command :=
  { name, description, supportsDevice
    run := fun args => do
      run args
      pure 0 }

/--
Wrap an example that reports its own exit status, typically because it can fail a numeric check.
-/
def statusCommand (name description : String) (run : List String → IO UInt32)
    (supportsDevice : Bool := false) : Command :=
  { name, description, run, supportsDevice }

/-- The full example catalogue.

This list is the single place a new example has to be registered; `usage`, `--list` and the
dispatcher are all derived from it, so they cannot drift apart. -/
def commandGroups : List CommandGroup :=
  [
    { title := "Quickstarts"
      commands :=
        [ unitCommand "quickstart_tensors" "construct, transform, and print typed tensors"
            NN.Examples.Quickstart.TensorBasics.main
        , unitCommand "quickstart_autograd" "differentiate tensor functions and model losses"
            NN.Examples.Quickstart.AutogradBasics.main
        , unitCommand "quickstart_mlp" "train and evaluate a small regression MLP"
            NN.Examples.Quickstart.SimpleMlpTrain.main (supportsDevice := true)
        ] }
  , { title := "Supervised and vision"
      commands :=
        [ statusCommand "mlp" "train an MLP on supervised CSV data"
            NN.Examples.Models.Supervised.Mlp.main (supportsDevice := true)
        , statusCommand "kan" "train a Kolmogorov-Arnold network on CSV data"
            NN.Examples.Models.Supervised.Kan.main (supportsDevice := true)
        , statusCommand "lstm_regression" "forecast a scalar sequence with an LSTM"
            NN.Examples.Models.Supervised.LstmRegression.main (supportsDevice := true)
        , statusCommand "cnn" "train a convolutional image classifier"
            NN.Examples.Models.Vision.Cnn.main (supportsDevice := true)
        , statusCommand "resnet" "train a residual image classifier"
            NN.Examples.Models.Vision.ResNet.main (supportsDevice := true)
        , statusCommand "vit" "train a compact vision transformer"
            NN.Examples.Models.Vision.Vit.main (supportsDevice := true)
        ] }
  , { title := "Sequence"
      commands :=
        [ statusCommand "rnn" "train a vanilla recurrent text model"
            NN.Examples.Models.Sequence.Rnn.main (supportsDevice := true)
        , statusCommand "lstm" "train an LSTM text model"
            NN.Examples.Models.Sequence.Lstm.main (supportsDevice := true)
        , statusCommand "transformer" "train a small transformer text model"
            NN.Examples.Models.Sequence.Transformer.main (supportsDevice := true)
        , statusCommand "chargpt" "train an indexed character-level GPT"
            NN.Examples.Models.Sequence.CharGpt.main (supportsDevice := true)
        , statusCommand "gpt2" "train and sample a byte-level GPT"
            NN.Examples.Models.Sequence.Gpt2.main (supportsDevice := true)
        , statusCommand "gpt2_saved" "load a saved GPT checkpoint and generate text"
            NN.Examples.Models.Sequence.Gpt2Saved.main (supportsDevice := true)
        , statusCommand "text_gpt2" "train byte or BPE GPT models on a text corpus"
            NN.Examples.Models.Sequence.TextGpt2.main (supportsDevice := true)
        , statusCommand "gpt_adder" "train a GPT on digit addition"
            NN.Examples.Models.Sequence.GptAdder.main (supportsDevice := true)
        , statusCommand "mamba" "train a compact Mamba-style sequence model"
            NN.Examples.Models.Sequence.Mamba.main (supportsDevice := true)
        ] }
  , { title := "Generative and operator learning"
      commands :=
        [ statusCommand "autoencoder" "reconstruct a compact vector of CIFAR values"
            NN.Examples.Models.Generative.Autoencoder.main (supportsDevice := true)
        , statusCommand "mae" "train a masked image autoencoder"
            NN.Examples.Models.Generative.Mae.main (supportsDevice := true)
        , statusCommand "diffusion" "train and sample an image diffusion model"
            NN.Examples.Models.Generative.Diffusion.main (supportsDevice := true)
        , statusCommand "fno1d_burgers" "learn the one-dimensional Burgers operator"
            NN.Examples.Models.Operators.Fno1dBurgers.main (supportsDevice := true)
        , unitCommand "pinn" "train a neural field from an equation and boundary conditions"
            NN.Examples.Models.Operators.Pinn.main
        , unitCommand "complex_regression" "fit complex parameters using a real loss"
            NN.Examples.Models.Operators.ComplexRegression.main
        ] }
  , { title := "Reinforcement learning"
      commands :=
        [ statusCommand "ppo_cartpole" "train and evaluate PPO on CartPole"
            NN.Examples.Models.RL.PPOCartPole.main (supportsDevice := true)
        , statusCommand "ppo_gridworld" "train PPO against a checked GridWorld contract"
            NN.Examples.Models.RL.PPOGridWorld.main (supportsDevice := true)
        , statusCommand "ppo_pong_ram" "train PPO on Pong RAM observations"
            NN.Examples.Models.RL.PPOPongRam.main (supportsDevice := true)
        , statusCommand "dqn_replay" "inspect checked DQN replay-buffer behavior"
            NN.Examples.Models.RL.DqnReplay.main
        ] }
  , { title := "PyTorch interop"
      commands :=
        [ unitCommand "pytorch_roundtrip" "export PyTorch source or import reference JSON weights"
            NN.Examples.Interop.PyTorch.Roundtrip.main
        ] }
  , { title := "Data"
      commands :=
        [ unitCommand "data_csv" "load and validate supervised CSV tensors"
            NN.Examples.Data.Loaders.Csv.main (supportsDevice := true)
        , unitCommand "data_npy" "load and validate NumPy tensor files"
            NN.Examples.Data.Loaders.Npy.main (supportsDevice := true)
        , unitCommand "data_cifar10" "load prepared CIFAR-10 image tensors"
            NN.Examples.Data.Loaders.Cifar10Images.main (supportsDevice := true)
        ] }
  , { title := "Numerical checks"
      commands :=
        [ unitCommand "factorizations" "check Cholesky and QR reconstruction properties"
            NN.Examples.Factorization.main
        , unitCommand "transcendentals" "check transcendental autograd rules"
            NN.Examples.Functional.Transcendentals.main
        ] }
  , { title := "Deep dives"
      commands :=
        [ unitCommand "autograd_transforms" "run Jacobian, Hessian, JVP, VJP, and detach examples"
            NN.Examples.DeepDives.AutogradTransforms.main
        , statusCommand "floats_arb_ieee_compare" "compare arbitrary precision and IEEE execution"
            NN.Examples.DeepDives.Floats.ArbIEEEExecCompare.main
        , unitCommand "float32_semantics" "inspect executable binary32 semantics"
            NN.Examples.DeepDives.Floats.Float32Semantics.main
        , statusCommand "numerical_certificate" "check a graph-level numerical certificate"
            NN.Examples.DeepDives.Floats.GraphNumericalCertificate.main
        , unitCommand "graphspec" "build and inspect a typed graph specification"
            NN.Examples.DeepDives.GraphSpec.Tutorial.main (supportsDevice := true)
        , unitCommand "ir_axis_ops" "lower axis operations into executable graph IR"
            NN.Examples.DeepDives.IRAxisOps.main (supportsDevice := true)
        , unitCommand "one_semantic_universe"
            "relate executable and proof-oriented scalar semantics"
            NN.Examples.DeepDives.OneSemanticUniverse.main
        , unitCommand "torch_ir_pytorch" "export TorchLean IR for PyTorch execution"
            NN.Examples.DeepDives.TorchIRPyTorch.main
        ] }
  ]

/-- Usage lines for one group: a blank separator, the heading, then one line per example. -/
def commandGroupUsage (group : CommandGroup) : List String :=
  [ "", group.title ++ ":" ] ++
    group.commands.map fun command => s!"  {command.name} - {command.description}"

/-- Every registered subcommand name, used by `--list` and by the shell completion helper. -/
def commandNames : List String :=
  commandGroups.flatMap fun group => group.commands.map (·.name)

/-- The help text, assembled from `commandGroups` so it always matches what can actually be run. -/
def usage : String :=
  String.intercalate "\n"
    ([ "TorchLean runnable examples"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean <example> [flags...]"
    , "  scripts/lake.sh exe torchlean --choose <example> [flags...]"
    , "  scripts/lake.sh exe torchlean <example> --help"
    , "  scripts/lake.sh exe torchlean --list"
    , ""
    , "Start here:"
    , "  scripts/lake.sh exe torchlean quickstart_tensors"
    , "  scripts/lake.sh exe torchlean quickstart_autograd"
    , "  scripts/lake.sh exe torchlean quickstart_mlp --steps 20"
    ] ++ commandGroups.flatMap commandGroupUsage ++
    [ ""
    , "Runtime flags:"
    , "  --choose                         ask for a device when the example supports --device"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --arithmetic native|ieee|complex"
    , "      arithmetic availability depends on the example; check its --help"
    , "  --execution eager|typed-graph"
    , "  --seed N"
    , "  --show-backend"
    , ""
    , "Verification commands live under `scripts/lake.sh exe verify -- list`."
    , "Use `scripts/lake.sh exe torchlean <example> --help` for command-specific flags."
    ])

/-- Runtime flags that consume the following command-line token. -/
def prefixFlagTakesValue (a : String) : Bool :=
  a == "--device" ||
  a == "--arithmetic" ||
  a == "--execution" ||
  a == "--seed"

/--
Split CLI arguments into `(prefixFlags, command, commandArgs)`.

We allow "global" runtime flags before the command name so users can write either:
- `torchlean mlp --device cpu`, or
- `torchlean --device cpu mlp`.
-/
def splitCommandArgs? (args : List String) : Option (List String × String × List String) :=
  let rec go (prefixRev : List String) : List String → Option (List String × String × List String)
    | .nil => none
    | a :: b :: rest =>
        if prefixFlagTakesValue a then
          go (b :: a :: prefixRev) rest
        else if a.startsWith "-" then
          go (a :: prefixRev) (b :: rest)
        else
          some (prefixRev.reverse, a, b :: rest)
    | a :: rest =>
        if a.startsWith "-" then
          go (a :: prefixRev) rest
        else
          some (prefixRev.reverse, a, rest)
  go [] args

/-- Detect whether the command line already selects a runtime device. -/
def hasDeviceFlag : List String → Bool
  | .nil => false
  | "--device" :: _ => true
  | a :: rest =>
      a.startsWith "--device=" || hasDeviceFlag rest

/-- Read one line, treating EOF as the default answer. -/
def readPromptLine : IO String := do
  try
    let line ← (← IO.getStdin).getLine
    pure line.trimAscii.toString
  catch _ =>
    pure ""

/-- Ask for the device in the interactive runtime chooser. -/
partial def askDevice : IO (List String) := do
  IO.println "Runtime device:"
  IO.println "  1) CPU    portable default"
  IO.println "  2) CUDA   GPU runtime, requires `scripts/lake.sh -K cuda=true exe ...`"
  IO.print "Select device [1]: "
  (← IO.getStdout).flush
  match (← readPromptLine).toLower with
  | "" | "1" | "cpu" => pure ["--device", "cpu"]
  | "2" | "cuda" => pure ["--device", "cuda"]
  | _ =>
      IO.println "Please choose 1/cpu or 2/cuda."
      askDevice

/-- Strip top-level runner flags that are not meant for individual examples. -/
def stripFlag (flag : String) : List String → List String
  | .nil => []
  | a :: rest =>
      if a == flag then stripFlag flag rest else a :: stripFlag flag rest

/-- Look up a subcommand inside a single group. -/
def findCommandIn (name : String) : List Command → Option Command
  | .nil => none
  | command :: rest =>
      if command.name == name then some command else findCommandIn name rest

/-- Look up a subcommand across every group, taking the first match. -/
def findCommand (name : String) : List CommandGroup → Option Command
  | .nil => none
  | group :: rest =>
      match findCommandIn name group.commands with
      | some command => some command
      | none => findCommand name rest

/-- Whether a known, device-selectable command still needs an interactive device choice.

Explicit device flags, including a missing value, are left to the command's normal parser.
-/
def needsDeviceChoice (args : List String) : Bool :=
  if args.contains "--help" || args.contains "-h" || hasDeviceFlag args then
    false
  else
    match splitCommandArgs? args with
    | none => false
    | some (_, name, _) =>
        match findCommand name commandGroups with
        | none => false
        | some command => command.supportsDevice

/-- Add a chooser response only where the selected command accepts a missing device flag. -/
def applyDeviceChoice (args deviceArgs : List String) : List String :=
  if needsDeviceChoice args then deviceArgs ++ args else args

/-- Fill in a missing device choice for a registered command that supports `--device`.

Fixed-runtime commands, help, absent commands, and unknown names never prompt.
-/
def chooseRuntimeArgs (args : List String) : IO (List String) := do
  if !needsDeviceChoice args then return args
  IO.println "TorchLean runtime chooser"
  pure (applyDeviceChoice args (← askDevice))

/-- Run one example by name, printing the usage text and failing if the name is unknown. -/
def runCmd (cmd : String) (args : List String) : IO UInt32 := do
  match findCommand cmd commandGroups with
  | some command => command.run args
  | none =>
      IO.eprintln s!"Unknown example: {cmd}"
      IO.eprintln ""
      IO.eprintln usage
      pure 1

/-- Entry point of the `torchlean` executable.

Handles the runner's own flags (`--help`, `--list`, `--choose`) and otherwise forwards the remaining
arguments untouched, so an example's own flag parsing never sees runner flags. -/
def main (args : List String) : IO UInt32 := do
  let args := CLI.dropDashDash args
  let choose := args.contains "--choose" && !(args.contains "--help") && !(args.contains "-h")
  let args := NN.Examples.Runner.stripFlag "--choose" args
  match args with
  | .nil =>
      IO.eprintln NN.Examples.Runner.usage
      pure 1
  | "--help" :: _ | "-h" :: _ =>
      IO.println NN.Examples.Runner.usage
      pure 0
  | "--list" :: _ =>
      commandNames.forM IO.println
      pure 0
  | _ =>
      let args ←
        if choose then
          NN.Examples.Runner.chooseRuntimeArgs args
        else
          pure args
      match NN.Examples.Runner.splitCommandArgs? args with
      | none =>
          IO.eprintln NN.Examples.Runner.usage
          pure 1
      | some (pref, cmd, commandArgs) =>
          if pref.contains "--help" || pref.contains "-h" then
            IO.println NN.Examples.Runner.usage
            pure 0
          else
            runCmd cmd (pref ++ commandArgs)

end NN.Examples.Runner
