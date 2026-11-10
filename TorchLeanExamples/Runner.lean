/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.Supervised.MLP
public import TorchLeanExamples.Models.Supervised.KAN
public import TorchLeanExamples.Models.Supervised.LSTMRegression
public import TorchLeanExamples.Models.Vision.CNN
public import TorchLeanExamples.Models.Vision.ResNet
public import TorchLeanExamples.Models.Vision.ViT
public import TorchLeanExamples.Models.Sequence.Recurrent
public import TorchLeanExamples.Models.Sequence.Transformer
public import TorchLeanExamples.Models.Sequence.GPT2Saved
public import TorchLeanExamples.Models.Sequence.TextGPT2
public import TorchLeanExamples.Models.Sequence.Mamba
public import TorchLeanExamples.Models.Sequence.GPTAdder
public import TorchLeanExamples.Models.Sequence.CharGPT
public import TorchLeanExamples.Models.Generative.Autoencoder
public import TorchLeanExamples.Models.Generative.MAE
public import TorchLeanExamples.Models.Generative.Diffusion
public import TorchLeanExamples.Models.Operators.ComplexRegression
public import TorchLeanExamples.Models.Operators.FNOBurgers
public import TorchLeanExamples.Models.Operators.PINNPoisson
public import TorchLeanExamples.Models.RL.PPOCartPole
public import TorchLeanExamples.Models.RL.PPOGridWorld
public import TorchLeanExamples.Models.RL.DQNReplay
public import TorchLeanExamples.DeepDives.AutogradTransforms
public import TorchLeanExamples.DeepDives.Floats.Float32Semantics
public import TorchLeanExamples.DeepDives.GraphSemantics
public import TorchLeanExamples.DeepDives.IRAxisOps
public import TorchLeanExamples.DeepDives.Sequential
public import TorchLeanExamples.DeepDives.TorchIRPyTorch
public import TorchLeanExamples.Tree.Bench
public import TorchLeanExamples.Quickstart.Autograd
public import TorchLeanExamples.Quickstart.MLP
public import TorchLeanExamples.Quickstart.Tensors
public import TorchLeanExamples.Data.Loaders.CSV
public import TorchLeanExamples.Data.Loaders.NPY
public import TorchLeanExamples.Data.Loaders.CifarImages
public import TorchLeanExamples.Interop.PyTorch.Roundtrip
public import TorchLeanExamples.Interop.JAX

/-!
# TorchLean Example Runner

Command registry and dispatcher for the `torchlean` executable. Each example keeps its own
namespace, and the runner picks one by its subcommand name, such as `mlp` or `gpt2`. Run
`scripts/lake.sh exe torchlean --list` for every name. `TorchLeanExamples.RunnerMain` supplies the
global executable entry point.
-/

@[expose] public section

open System

namespace TorchLean.Examples.Runner

/-- One runnable example, as the dispatcher sees it. -/
structure Command where
  /-- Subcommand typed by the user, for example `quickstart_mlp`. -/
  name : String
  /-- One-line summary shown next to the name in the usage text (`--list` prints names only). -/
  description : String
  /-- The example's entry point, whose return value becomes the process exit status. -/
  run : List String → IO UInt32
  /-- Whether the interactive chooser asks for a device: the example accepts `--device` and runs on
  more than one. CUDA-only examples take `--device cuda` themselves. -/
  offersDeviceChoice : Bool := false

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
    (offersDeviceChoice : Bool := false) : Command :=
  { name, description, offersDeviceChoice
    run := fun args => do
      run args
      pure 0 }

/--
Wrap an example that reports its own exit status, typically because it can fail a numeric check.
-/
def statusCommand (name description : String) (run : List String → IO UInt32)
    (offersDeviceChoice : Bool := false) : Command :=
  { name, description, run, offersDeviceChoice }

/-- The full example catalogue, in listing order.

An example is registered here and only here: `usage`, `--list` and the dispatcher all read this
list, so they cannot drift apart. -/
def commandGroups : List CommandGroup :=
  [
    { title := "Quickstarts"
      commands :=
        [ unitCommand "quickstart_tensors" "construct, transform, and print typed tensors"
            Examples.Quickstart.Tensors.main
        , unitCommand "quickstart_autograd" "differentiate tensor functions and model losses"
            Examples.Quickstart.Autograd.main
        , unitCommand "quickstart_mlp" "train and evaluate a small regression MLP"
            Examples.Quickstart.MLP.main (offersDeviceChoice := true)
        ] }
  , { title := "Supervised and vision"
      commands :=
        [ statusCommand "mlp" "train an MLP on supervised CSV data"
            Examples.MLP.main (offersDeviceChoice := true)
        , statusCommand "kan" "train a Kolmogorov-Arnold network on CSV data"
            Examples.KAN.main (offersDeviceChoice := true)
        , statusCommand "lstm_regression" "forecast a scalar sequence with an LSTM"
            Examples.LSTMRegression.main (offersDeviceChoice := true)
        , statusCommand "cnn" "train a convolutional image classifier"
            Examples.CNN.main (offersDeviceChoice := true)
        , statusCommand "resnet" "train a residual image classifier"
            Examples.ResNet.main (offersDeviceChoice := true)
        , statusCommand "vit" "train a compact vision transformer"
            Examples.ViT.main (offersDeviceChoice := true)
        ] }
  , { title := "Sequence"
      commands :=
        [ statusCommand "rnn" "train a vanilla recurrent text model"
            (Examples.Recurrent.main .rnn) (offersDeviceChoice := true)
        , statusCommand "lstm" "train an LSTM text model"
            (Examples.Recurrent.main .lstm) (offersDeviceChoice := true)
        , statusCommand "transformer" "train a small transformer text model"
            Examples.Transformer.main (offersDeviceChoice := true)
        , statusCommand "chargpt" "train an indexed character-level GPT"
            Examples.CharGPT.main (offersDeviceChoice := true)
        , statusCommand "gpt2" "train and sample a byte-level GPT"
            Examples.GPT2.main (offersDeviceChoice := true)
        , statusCommand "gpt2_saved" "load a saved GPT checkpoint and generate text"
            Examples.GPT2Saved.main (offersDeviceChoice := true)
        , statusCommand "text_gpt2" "train byte or BPE GPT models on a text corpus"
            Examples.TextGPT2.main
        , statusCommand "gpt_adder" "train a GPT on digit addition"
            Examples.GPTAdder.main
        , statusCommand "mamba" "train a compact Mamba-style sequence model"
            Examples.Mamba.main (offersDeviceChoice := true)
        ] }
  , { title := "Generative and operator learning"
      commands :=
        [ statusCommand "autoencoder" "reconstruct a compact vector of CIFAR values"
            Examples.Autoencoder.main (offersDeviceChoice := true)
        , statusCommand "mae" "train a masked image autoencoder"
            Examples.MAE.main (offersDeviceChoice := true)
        , statusCommand "diffusion" "train and sample an image diffusion model"
            Examples.Diffusion.main (offersDeviceChoice := true)
        , statusCommand "fno1d_burgers" "learn the one-dimensional Burgers operator"
            Examples.FNOBurgers.main (offersDeviceChoice := true)
        , unitCommand "pinn" "train a neural field from an equation and boundary conditions"
            Examples.PINNPoisson.main
        , unitCommand "complex_regression" "fit complex parameters using a real loss"
            Examples.ComplexRegression.main
        ] }
  , { title := "Reinforcement learning"
      commands :=
        [ statusCommand "ppo_cartpole" "train and evaluate PPO on CartPole"
            Examples.PPOCartPole.main (offersDeviceChoice := true)
        , statusCommand "ppo_gridworld" "train PPO against a checked GridWorld contract"
            Examples.PPOGridWorld.main (offersDeviceChoice := true)
        , statusCommand "dqn_replay" "print replay-buffer sizes and DQN minibatch losses"
            Examples.DQNReplay.main
        ] }
  , { title := "External graphs and weights"
      commands :=
        [ unitCommand "pytorch_roundtrip" "run Lean models on checked-in PyTorch state dicts"
            Examples.PyTorchRoundtrip.main
        , unitCommand "jaxpr" "import captured JAX graphs and compare their Lean outputs"
            Examples.Jaxpr.run
        ] }
  , { title := "Data"
      commands :=
        [ unitCommand "data_csv" "load and validate supervised CSV tensors"
            Examples.CSV.main (offersDeviceChoice := true)
        , unitCommand "data_npy" "load and validate NumPy tensor files"
            Examples.NPY.main (offersDeviceChoice := true)
        , unitCommand "data_cifar10" "load prepared CIFAR-10 image tensors"
            Examples.CifarImages.main (offersDeviceChoice := true)
        ] }
  , { title := "Deep dives"
      commands :=
        [ unitCommand "autograd_transforms" "run Jacobian, Hessian, JVP, HVP and detach examples"
            Examples.AutogradTransforms.main
        , unitCommand "float32_semantics" "inspect executable binary32 semantics"
            Examples.Float32Semantics.main
        , unitCommand "sequential" "evaluate a sequential model, then train it"
            Examples.Sequential.main (offersDeviceChoice := true)
        , unitCommand "ir_axis_ops" "evaluate axis operations as graphs of registered operators"
            Examples.IRAxisOps.main
        , unitCommand "graph_semantics"
            "evaluate a graph and propagate its interval bounds"
            Examples.GraphSemantics.main
        , unitCommand "torch_ir_pytorch" "export TorchLean IR for PyTorch execution"
            Examples.TorchIRPyTorch.main
        , statusCommand "tree_bench" "time tree arenas on the host and the device"
            Examples.Tree.Bench.main (offersDeviceChoice := true)
        ] }
  ]

/-- Usage lines for one group: a blank separator, the heading, then one line per example. -/
def commandGroupUsage (group : CommandGroup) : List String :=
  [ "", group.title ++ ":" ] ++
    group.commands.map fun command => s!"  {command.name} - {command.description}"

/-- Every registered subcommand name, used by `--list`. -/
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
    , s!"  --device auto|{Backend.Device.help}"
    , "  --arithmetic native|ieee|complex"
    , "      arithmetic availability depends on the example; check its --help"
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
  a == "--seed"

/--
Split CLI arguments into `(prefixFlags, command, commandArgs)`, or `none` if no command name
appears.

Runtime flags may come before the command name, so `torchlean mlp --device cpu` and
`torchlean --device cpu mlp` both work.
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

/-- Whether the arguments already select a device, as `--device` or `--device=...`. -/
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

/-- Prompt for CPU or CUDA and return the matching `--device` arguments, asking again after any
other answer. An empty answer picks CPU. -/
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

/-- Look up a subcommand across every group, taking the first match. -/
def findCommand (name : String) : List CommandGroup → Option Command
  | .nil => none
  | group :: rest =>
      match group.commands.find? (·.name == name) with
      | some command => some command
      | none => findCommand name rest

/-- Whether `--choose` should prompt for a device: the command is registered with
`offersDeviceChoice`, and the arguments hold no help flag and no device flag.

A `--device` with a missing value counts as a device flag, so the command's own parser reports it.
-/
def needsDeviceChoice (args : List String) : Bool :=
  if CLI.hasHelp args || hasDeviceFlag args then
    false
  else
    match splitCommandArgs? args with
    | none => false
    | some (_, name, _) =>
        match findCommand name commandGroups with
        | none => false
        | some command => command.offersDeviceChoice

/-- Prepend the chooser's device arguments when `needsDeviceChoice args` holds. -/
def applyDeviceChoice (args deviceArgs : List String) : List String :=
  if needsDeviceChoice args then deviceArgs ++ args else args

/-- Fill in a missing device choice for a registered command that supports `--device`.

Help requests, fixed-runtime commands, a missing command name and unknown names never prompt.
-/
def chooseDevice (args : List String) : IO (List String) := do
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

The runner answers `--help` before the command name and a leading `--list` itself, and it drops
`--choose` wherever it appears. All other arguments go to the example: runtime flags written
before the command name come first, then the example's own arguments. -/
def main (args : List String) : IO UInt32 := do
  let args := CLI.dropDashDash args
  let choose := args.contains "--choose" && !CLI.hasHelp args
  let args := args.filter (· != "--choose")
  match args with
  | .nil =>
      IO.eprintln Examples.Runner.usage
      pure 1
  | "--help" :: _ | "-h" :: _ =>
      IO.println Examples.Runner.usage
      pure 0
  | "--list" :: _ =>
      commandNames.forM IO.println
      pure 0
  | _ =>
      let args ←
        if choose then
          Examples.Runner.chooseDevice args
        else
          pure args
      match Examples.Runner.splitCommandArgs? args with
      | none =>
          IO.eprintln Examples.Runner.usage
          pure 1
      | some (pref, cmd, commandArgs) =>
          if CLI.hasHelp pref then
            IO.println Examples.Runner.usage
            pure 0
          else
            runCmd cmd (pref ++ commandArgs)

end TorchLean.Examples.Runner
