/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# LSTM Seasonal Regression / Forecasting

This is the runnable supervised sequence example: an LSTM trains on a real-valued forecasting task
and works with the same CPU/CUDA runtime flags as the other model commands.

The default data path uses the UCI Individual Household Electric Power Consumption dataset:
minute-level power readings from one household over almost four years. The preparation script turns
that into hourly one-step forecasting windows:

`past 24 hours -> next 24 shifted-by-one-hour targets`

Prepare the real data once:

```bash
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
```

Recommended runs:
- use `--steps 1` to check that the runtime, data loader, and CUDA path agree on shapes;
- use `--steps 200 --windows 96` for a short training run with before/after forecast reports;
- change `--report-offset` to evaluate a different part of the power curve;
- lower `--lr` if the reported forecast error increases.

```bash
scripts/lake.sh -Kcuda=true exe torchlean lstm_regression --device cuda --steps 1 --windows 1
scripts/lake.sh -Kcuda=true exe torchlean lstm_regression --device cuda --steps 200 --windows 96
```

Dataset citation: Hebrail and Berard, "Individual Household Electric Power Consumption", UCI Machine
Learning Repository, DOI `10.24432/C58K54`, CC BY 4.0.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Supervised.LstmRegression

/-- Runner subcommand: `scripts/lake.sh exe torchlean lstm_regression ...`. -/
def exeName : String := "lstm_regression"

/--
Default JSON path for the before/after loss.

Pass `--log PATH` to write somewhere else, or `--log disabled` when you only want terminal output.
-/
def defaultLogPath : System.FilePath := Support.trainLogPath "lstm_regression"

/-- Prepared household-power windows contain one day of hourly samples. -/
def sequenceLength : Nat := 24

/-- One scalar feature. Increase this when the prepared sequence data gains more features. -/
def featureCount : Nat := 1

/-- Hidden width for the recurrent state used by this runnable example. -/
def hiddenWidth : Nat := 8

/-- One scalar power-consumption prediction at each timestep. -/
def outputWidth : Nat := 1

/--
Shared recurrent-model configuration.

The model constructor, input shape, and output shape all read from this value. The single output
feature is the next power-consumption prediction at each time step.
-/
abbrev modelConfig : nn.models.Recurrent.Config :=
  { sequenceLength := sequenceLength
    inputWidth := featureCount
    hiddenWidths := [hiddenWidth]
    outputWidth := outputWidth }

/-- Input shape: one scalar observation at each timestep. -/
abbrev input : Shape :=
  modelConfig.inputShape

/-- Target/prediction shape: one next-step scalar at each timestep. -/
abbrev output : Shape :=
  modelConfig.outputShape

/--
The actual forecaster.

`nn.models.lstm modelConfig` expands to:

`nn.lstm sequenceLength featureCount hiddenWidth`
followed by a time-distributed `nn.linear hiddenWidth outputWidth`.

So every timestep emits a scalar forecast. We are not using only the final hidden state here; the
loss checks the whole output sequence.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.lstm modelConfig

/-- Data source tags for terminal logs and JSON metadata. -/
def dataTags (xPath yPath : System.FilePath) : Array String :=
  #["data=uci-household-power", s!"x={xPath}", s!"y={yPath}"]

/-- Validate the prepared input file and return its available window count. -/
def availableWindows (xPath : System.FilePath) :
    IO Nat :=
  Data.availableNpyRows xPath [sequenceLength, featureCount]
    s!"X.npy shape (N,{sequenceLength},{featureCount})"

/-- Load the Float version once for reporting probes and short training. -/
def loadReportSamples (xPath yPath : System.FilePath) (windows : Nat) :
    IO (Array (Sample.Supervised Float input output)) := do
  let source := Data.SupervisedSource.fromFiles xPath yPath windows
    [sequenceLength, featureCount] [sequenceLength, outputWidth]
  let samples ← source.load (α := Float)
  pure <| by
    simpa [input, output, modelConfig, source,
      Data.SupervisedSource.fromFiles] using samples.toArray

/--
Render the first few target values for one forecast window.
-/
def targetSummary (sample : Sample.Supervised Float input output) : String :=
  String.intercalate ", " <|
    (List.finRange (Nat.min sequenceLength 8)).map (fun i =>
      let row : Fin sequenceLength :=
        ⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_left sequenceLength 8)⟩
      s!"t+{i.val + 1}={sample.target (row, ⟨0, by decide⟩, PUnit.unit)}")

/-- Public trainer probe for a deterministic forecast window. -/
def probe
    (sample : Sample.Supervised Float input output)
    (index : Nat) :
    Trainer.Probe input :=
  Trainer.Probe.tensor
    "forecast"
    sample.input
    (inputText := s!"report_index={index}")
    (expected := some (targetSummary sample))

/-- Train on the prepared windows and report predictions for one selected window. -/
def train (runtime : Runtime.Config)
    (flags : Support.Training.Options Support.Forecast.Options) :
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
    Trainer.new model <|
      { runtime with
          optimizer := optim.adam { learningRate := flags.training.learningRate }
          objective := mse }
  trainer.train
    (Data.fromSamples samples)
    (flags.training.trainOptions
      (logTitle := "LSTM seasonal regression")
      (logNotes := Support.Forecast.Options.logNotes flags.data ++
        #[s!"cuda_mem_watch={flags.training.cudaMemorySampleEvery}",
          "task=next-step household power forecasting"] ++
        dataTags flags.data.xPath flags.data.yPath))
    #[reportProbe]

/-- Executable entrypoint for CPU/CUDA Float training. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.forecast exeName args
    (fun rest =>
      Support.Training.Options.parse exeName rest defaultLogPath 100 0.01
        (parseData := fun args => RealData.Forecast.Options.parse exeName args 512 96))
    (Support.banner exeName "LSTM time-series regression")
    train (fun result => result.printSummary)

end NN.Examples.Models.Supervised.LstmRegression
