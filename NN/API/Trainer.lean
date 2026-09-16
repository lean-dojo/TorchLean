/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core
public import NN.API.Trainer.Dataset
public import NN.API.Trainer.Constructor
public import NN.API.Trainer.Results
public import NN.API.Trainer.Run
public import NN.API.Trainer.Scheduler
public import NN.API.Trainer.Session
public import NN.API.Trainer.Train
public import NN.API.Trainer.Reporting

/-!
# Training

The main training interface:

```lean
let trainer := Trainer.new model
  { objective := .meanSquaredError
    optimizer := optim.adam { learningRate := 0.03 } }
let y0 ← trainer.predict x
let trained ← trainer.train data { steps := 200, samplesPerStep := 16, logEvery := 25 }
trained.printSummary
trained.save "model.state"
```

The same interface supports regression, classification, custom losses, finite datasets, and
streaming batches. Public signatures use `Tensor Float`; the run executes in the binary32 scalar
selected by `arithmetic` (`Float32` or `ExecFloat.Binary 8 23`) and the report names it.

Programs that own the optimizer loop open the same trainer as a session:

```lean
let session ← trainer.open
for step in [0:steps] do
  let loss ← session.step (sampleAt step)
let trained ← session.finish { before, after }
```

`trainer.train` is exactly this pattern with dataset cycling, logging, and checkpoints added.
Benchmark and diagnostic code that repeatedly trains one sample should import
`NN.API.Trainer.FixedSample` explicitly.
-/
