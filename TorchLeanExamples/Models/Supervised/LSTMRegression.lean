/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Models.Recurrent
public import TorchLeanExamples.Models.Training

/-!
# LSTM Seasonal Regression / Forecasting

The `lstm_regression` command trains an LSTM on a real-valued forecasting task and takes the same
CPU and CUDA runtime flags as the other model commands.

The default data path uses the UCI Individual Household Electric Power Consumption dataset:
minute-level power readings from one household over almost four years. The preparation script
averages the `Global_active_power` readings over each hour, normalizes them to $[0,1]$ and cuts the
hourly series into one-step forecasting windows:

`past 24 hours -> next 24 shifted-by-one-hour targets`

Prepare the real data once:

```bash
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
```

Recommended runs:
- `--steps 1` checks that the runtime, the data loader and the CUDA path agree on shapes.
- `--steps 200 --windows 96` is a short training run with forecast reports before and after.
- A different `--report-offset` reports the forecast for another part of the power curve.
- Lower `--lr` if the reported forecast error increases.

```bash
scripts/lake.sh -Kcuda=true exe torchlean lstm_regression --device cuda --steps 1 --windows 1
scripts/lake.sh -Kcuda=true exe torchlean lstm_regression --device cuda --steps 200 --windows 96
```

Dataset citation: Hebrail and Berard, "Individual Household Electric Power Consumption", UCI Machine
Learning Repository, DOI `10.24432/C58K54`, CC BY 4.0.
-/

@[expose] public section

open TorchLean.Data TorchLean.Trainer.Objective

namespace TorchLean.Examples.LSTMRegression

/-- Runner subcommand: `scripts/lake.sh exe torchlean lstm_regression ...`. -/
def exeName : String := "lstm_regression"

/--
Default JSON path for the before/after loss.

Pass `--log PATH` to write somewhere else, or `--log disabled` when you only want terminal output.
-/
def defaultLogPath : System.FilePath := Training.artifactPath "lstm_regression"

/-- Prepared household-power windows contain one day of hourly samples. -/
def sequenceLength : Nat := 24

/--
One feature per timestep, the normalized hourly power. Prepared data with more features per
timestep needs a matching value here, because `availableWindows` checks the `X.npy` shape.
-/
def featureCount : Nat := 1

/-- Width of the LSTM hidden state. -/
def hiddenWidth : Nat := 8

/-- One scalar power-consumption prediction at each timestep. -/
def outputWidth : Nat := 1

/--
Recurrent-model configuration: 24 timesteps, one input feature, one LSTM layer of width 8 and one
output feature.

The model, the input shape and the output shape all read this value. At each time step the output
feature predicts the power reading one hour later.
-/
abbrev modelConfig : nn.Recurrent.Config :=
  { sequenceLength := sequenceLength
    inputWidth := featureCount
    hiddenWidths := [hiddenWidth]
    outputWidth := outputWidth }

/-- Input shape `[24, 1]`: one scalar observation at each timestep. -/
abbrev input : Shape :=
  modelConfig.inputShape

/-- Target and prediction shape `[24, 1]`: the next hour's value at each timestep. -/
abbrev output : Shape :=
  modelConfig.outputShape

/--
The forecaster.

`nn.Recurrent.lstm modelConfig` expands to:

`nn.lstm sequenceLength featureCount hiddenWidth`
followed by a time-distributed `nn.linear hiddenWidth outputWidth`.

Each timestep emits a scalar forecast from its hidden state, and the loss covers all 24 forecasts.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.Recurrent.lstm modelConfig

/-- Data source tags for terminal logs and JSON metadata. -/
def dataTags (xPath yPath : System.FilePath) : Array String :=
  #["data=uci-household-power", s!"x={xPath}", s!"y={yPath}"]

/-- Check that `X.npy` has shape `(N, 24, 1)` and return the window count `N`. -/
def availableWindows (xPath : System.FilePath) :
    IO Nat :=
  Data.NPY.availableRows xPath [sequenceLength, featureCount]
    s!"X.npy shape (N,{sequenceLength},{featureCount})"

/--
Load the first `windows` prepared windows as `Float` samples. The same array is the training data
and the source of the report probe.
-/
def loadReportSamples (xPath yPath : System.FilePath) (windows : Nat) :
    IO (Array (Sample.Supervised Float input output)) := do
  let source := Data.SupervisedSource.fromFiles xPath yPath windows
    [sequenceLength, featureCount] [sequenceLength, outputWidth]
  let samples ← source.load (α := Float)
  pure <| by
    simpa [input, output, modelConfig, source,
      Data.SupervisedSource.fromFiles] using samples.toArray

/--
Render the first eight target values of one forecast window, labeled `t+1` to `t+8`.
-/
def targetSummary (sample : Sample.Supervised Float input output) : String :=
  String.intercalate ", " <|
    (List.finRange (Nat.min sequenceLength 8)).map (fun i =>
      let row : Fin sequenceLength :=
        ⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_left sequenceLength 8)⟩
      s!"t+{i.val + 1}={sample.target (row, ⟨0, by decide⟩, PUnit.unit)}")

/--
Trainer probe named `forecast` for one window. Its input text names the window index, and its
expected text lists the first eight targets.
-/
def probe
    (sample : Sample.Supervised Float input output)
    (index : Nat) :
    Trainer.Probe input :=
  Trainer.Probe.tensor
    "forecast"
    sample.input
    (inputText := s!"report_index={index}")
    (expected := some (targetSummary sample))

/--
Train with Adam on the mean squared error over the first `--windows` windows of the `--x` and `--y`
files, and fail when the input file holds fewer. The forecast for window `--report-offset` modulo
the window count is reported before and after training.
-/
def train (runtime : Session.Config)
    (flags : Training.Options Forecast.Options) :
    IO (Trainer.Result input output) := do
  Data.requirePairedFiles exeName
    "household-power inputs" flags.data.xPath
    "household-power targets" flags.data.yPath
    RealData.missingHouseholdPowerHint
  let available ← availableWindows flags.data.xPath
  if flags.data.windows > available then
    throw <| IO.userError
      (s!"{exeName}: requested --windows {flags.data.windows}, "
        ++ s!"but {flags.data.xPath} only contains {available} windows")
  let samples ←
    loadReportSamples flags.data.xPath flags.data.yPath flags.data.windows
  let probeIndex := flags.data.reportOffset % Nat.max 1 samples.size
  let probeSample ←
    match samples[probeIndex]? with
    | some sample => pure sample
    | none => throw <| IO.userError s!"{exeName}: no training windows loaded"
  let reportProbe := probe probeSample probeIndex
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := optim.adam { lr := flags.training.lr }
          objective := mse }
  trainer.train
    (Data.fromSamples samples)
    (flags.training.trainOptions
      (logTitle := "LSTM seasonal regression")
      (logNotes := Forecast.Options.logNotes flags.data ++
        #[Training.cudaMemoryNote runtime flags.training.steps
            flags.training.cudaMemorySampleEvery,
          "task=next-step household power forecasting"] ++
        dataTags flags.data.xPath flags.data.yPath))
    #[reportProbe]

/--
CLI entry point for CPU or CUDA training. The defaults are 100 steps at learning rate `0.01`, 512
windows and report offset 96.
-/
def main (args : List String) : IO UInt32 :=
  Forecast.train exeName args
    (fun rest =>
      Training.Options.parse exeName rest defaultLogPath 100 0.01
        (parseData := fun args => Forecast.Options.parse exeName args 512 96))
    (Training.banner exeName "LSTM time-series regression")
    train (fun result => result.printSummary)

end TorchLean.Examples.LSTMRegression
