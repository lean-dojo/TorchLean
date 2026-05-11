/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models
public import NN.Examples.Advanced
public import NN.Examples.Quickstart
public import NN.Examples.Data.Loaders.Csv
public import NN.Examples.Data.Loaders.Npy
public import NN.Examples.Data.Loaders.Cifar10Images
public import NN.Examples.Interop.PyTorch.Roundtrip
public import NN.Examples.Interop.PyTorch.TorchExportSmoke

/-!
# TorchLean Example Runner

This module is the executable root for the `torchlean` example runner.

We keep each example's implementation namespaced (so `NN.Examples.Zoo` can import them all),
and select which example to run based on a subcommand argument (e.g. `mlp`, `gpt2`).
-/

@[expose] public section

open System

namespace NN.Examples.Models.Runner

def usage : String :=
  String.intercalate "\n"
    [ "TorchLean runnable examples"
    , ""
    , "Usage:"
    , "  lake exe torchlean --help"
    , "  lake exe torchlean <example> [flags...]"
    , ""
    , "Examples:"
    , "  quickstart_tensors | quickstart_autograd | quickstart_mlp"
    , "  mlp | cnn | diffusion | fno1d_burgers"
    , "  autoencoder | mae | vae | vqvae | gan"
    , "  rnn | lstm | lstm_regression | transformer | vit | resnet | gpt2 | text_gpt2"
    , "  gpt2_saved"
    , "  chargpt"
    , "  gpt_adder"
    , "  mamba"
    , "  ppo_cartpole | ppo_gridworld | ppo_pong_ram | dqn_replay"
    , "  pytorch_roundtrip | pytorch_export_smoke"
    , "  data_csv | data_npy | data_cifar10"
    , "  floats_arb_ieee_compare | float32_modes | graphspec"
    , "  ir_axis_ops | one_semantic_universe | torch_ir_pytorch"
    , ""
    , "Notes:"
    , "  - You can put runtime flags (e.g. `--cpu`, `--cuda`, `--dtype`, `--backend`) either before or"
    , "    after the example name; the runner forwards them to the example entrypoint."
    , "  - If you are used to `lean --run`, a leading `--` is accepted and ignored:"
    , "      lake exe torchlean -- mlp --cpu"
    , ""
    , "Examples:"
    , "  python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10"
    , "  lake exe torchlean mlp --cpu --steps 10"
    , "  lake exe torchlean cnn --cuda --steps 10"
    , "  lake exe torchlean quickstart_tensors"
    , "  lake exe torchlean quickstart_autograd --dtype float --backend eager"
    , "  lake exe torchlean quickstart_mlp --steps 20 --dtype float --backend eager"
    , "  lake build -R -K cuda=true && lake exe torchlean gpt2 --cuda --steps 1"
    , "  lake exe torchlean gpt2 --flash-attention-only"
    , "  lake exe torchlean fno1d_burgers --cuda --fast-kernels --steps 50 --plot-csv data/real/fno/predictions.csv"
    , "  lake exe torchlean gpt2 --cuda --tiny-shakespeare --steps 100"
    , "  python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512"
    , "  lake exe torchlean lstm_regression --cuda --steps 200 --windows 96"
    , "  lake exe torchlean text_gpt2 --data-file data/real/text/tinystories_valid.txt --steps 100"
    , "  lake exe torchlean mamba --cuda --tiny-shakespeare --steps 25"
    , "  lake exe torchlean mae --cuda --steps 25 --log data/model_zoo/mae_trainlog.json"
    , "  lake exe torchlean vae --cuda --steps 25 --log data/model_zoo/vae_trainlog.json"
    , "  lake exe torchlean gan --cuda --steps 25 --log data/model_zoo/gan_trainlog.json"
    , "  python3 scripts/datasets/torchlean_data_convert.py image-folder --input /path/to/imagenet/train --x-output data/real/imagenet64/imagenet64_train_X.npy --y-output data/real/imagenet64/imagenet64_train_y.npy --height 64 --width 64 --labels-from-dirs --limit 2000"
    , "  lake exe torchlean diffusion --cuda --fast-kernels --dataset imagenet64 --n-total 800 --steps 200 --hidden-c 8 --T 100 --beta-end 0.12 --sample-ppm data/model_zoo/imagenet64_sample.ppm"
    , "  lake exe torchlean pytorch_roundtrip --model mlp --action import"
    , "  lake exe torchlean data_csv --epochs 1 --batch 5 --dtype float --backend eager"
    , "  lake exe torchlean data_npy --epochs 1 --batch 5 --dtype float --backend eager"
    , "  lake exe torchlean data_cifar10 --quick --epochs 1 --batch 4 --train-size 8 --n-total 20"
    , "  lake exe torchlean pytorch_export_smoke"
    , "  lake exe torchlean float32_modes"
    , "  lake exe torchlean graphspec --backend eager"
    , "  lake exe torchlean ir_axis_ops --dtype float --backend eager"
    , "  lake exe torchlean one_semantic_universe --samples 50"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py"
    , "  lake exe torchlean gpt_adder --steps 1000 --a 7 --b 8"
    ]

/--
Split CLI arguments into `(prefixFlags, command, commandArgs)`.

We allow "global" runtime flags before the command name so users can write either:
- `torchlean mlp --cpu`, or
- `torchlean --cpu mlp`.
-/
def splitCommandArgs? (args : List String) : Option (List String × String × List String) :=
  let rec go (prefixRev : List String) : List String → Option (List String × String × List String)
    | [] => none
    | a :: rest =>
        if a.startsWith "-" then
          go (a :: prefixRev) rest
        else
          some (prefixRev.reverse, a, rest)
  go [] args

def runCmd (cmd : String) (args : List String) : IO UInt32 := do
  match cmd with
  | "quickstart_tensors" =>
      NN.Examples.Quickstart.TensorBasics.main args
      pure 0
  | "quickstart_autograd" =>
      NN.Examples.Quickstart.AutogradBasics.main args
      pure 0
  | "quickstart_mlp" =>
      NN.Examples.Quickstart.SimpleMLPTrain.main args
      pure 0
  | "mlp" => NN.Examples.Models.Supervised.Mlp.main args
  | "cnn" => NN.Examples.Models.Vision.Cnn.main args
  | "diffusion" => NN.Examples.Models.Generative.Diffusion.main args
  | "fno1d_burgers" => NN.Examples.Models.Operators.Fno1dBurgers.main args
  | "autoencoder" => NN.Examples.Models.Generative.Autoencoder.main args
  | "mae" => NN.Examples.Models.Generative.Mae.main args
  | "vae" => NN.Examples.Models.Generative.Vae.main args
  | "vqvae" => NN.Examples.Models.Generative.VqVae.main args
  | "gan" => NN.Examples.Models.Generative.Gan.main args
  | "rnn" => NN.Examples.Models.Sequence.Rnn.main args
  | "lstm" => NN.Examples.Models.Sequence.Lstm.main args
  | "lstm_regression" => NN.Examples.Models.Supervised.LstmRegression.main args
  | "transformer" => NN.Examples.Models.Sequence.Transformer.main args
  | "vit" => NN.Examples.Models.Vision.Vit.main args
  | "resnet" => NN.Examples.Models.Vision.Resnet.main args
  | "gpt2" => NN.Examples.Models.Sequence.Gpt2.main args
  | "gpt2_saved" => NN.Examples.Models.Sequence.Gpt2Saved.main args
  | "text_gpt2" => NN.Examples.Models.Sequence.TextGpt2.main args
  | "chargpt" => NN.Examples.Models.Sequence.CharGpt.main args
  | "gpt_adder" => NN.Examples.Models.Sequence.GptAdder.main args
  | "mamba" => NN.Examples.Models.Sequence.Mamba.main args
  | "ppo_cartpole" => NN.Examples.Models.RL.PPOCartPole.main args
  | "ppo_gridworld" => NN.Examples.Models.RL.PPOGridWorld.main args
  | "ppo_pong_ram" => NN.Examples.Models.RL.PPOPongRam.main args
  | "dqn_replay" => NN.Examples.Models.RL.DQNReplay.main args
  | "data_csv" =>
      NN.Examples.Data.Loaders.Csv.main args
      pure 0
  | "data_npy" =>
      NN.Examples.Data.Loaders.Npy.main args
      pure 0
  | "data_cifar10" =>
      NN.Examples.Data.Loaders.Cifar10Images.main args
      pure 0
  | "pytorch_roundtrip" =>
      NN.Examples.Interop.PyTorch.Roundtrip.main args
      pure 0
  | "pytorch_export_smoke" => NN.Examples.Interop.PyTorch.TorchExportSmoke.main args
  | "floats_arb_ieee_compare" => NN.Examples.Advanced.Floats.ArbIEEEExecCompare.main args
  | "float32_modes" =>
      NN.Examples.Advanced.Floats.Float32Modes.main args
      pure 0
  | "graphspec" =>
      NN.Examples.Advanced.GraphSpec.Tutorial.main args
      pure 0
  | "ir_axis_ops" =>
      NN.Examples.Advanced.IRAxisOps.main args
      pure 0
  | "one_semantic_universe" =>
      NN.Examples.Advanced.OneSemanticUniverse.main args
      pure 0
  | "torch_ir_pytorch" =>
      NN.Examples.Advanced.TorchIRPyTorch.main args
      pure 0
  | _ =>
      IO.eprintln s!"Unknown example: {cmd}"
      IO.eprintln ""
      IO.eprintln usage
      pure 1

end NN.Examples.Models.Runner

def main (args : List String) : IO UInt32 := do
  let args := NN.API.CLI.dropDashDash args
  match args with
  | [] =>
      IO.eprintln NN.Examples.Models.Runner.usage
      pure 1
  | "--help" :: _ | "-h" :: _ =>
      IO.println NN.Examples.Models.Runner.usage
      pure 0
  | _ =>
      match NN.Examples.Models.Runner.splitCommandArgs? args with
      | none =>
          IO.eprintln NN.Examples.Models.Runner.usage
          pure 1
      | some (pref, cmd, commandArgs) =>
          NN.Examples.Models.Runner.runCmd cmd (pref ++ commandArgs)
