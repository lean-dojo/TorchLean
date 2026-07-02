# NN/API

These modules back the public TorchLean API. Model and training code usually imports:

```lean
import NN
open TorchLean
```

Use the `NN.API.*` imports when you are extending TorchLean itself or working inside a subsystem:

* `NN.API.Public` backs the `TorchLean.nn`, `TorchLean.optim`, `TorchLean.Trainer`,
  `TorchLean.Data`, `TorchLean.Loss`, and `TorchLean.Metrics` namespaces.
* `NN.API.Runtime` exposes the executable runtime surface for code that works directly with tensor
  operations, module execution, autograd, supervised training, and session-level tools.
* Focused files under `NN/API` give subsystem code a smaller import target than the full `NN`
  umbrella.

The design rule is simple: examples and downstream projects should import `NN`, then use the
`TorchLean` namespace. Subsystem code can import the focused `NN.API.*` module it actually needs.

## Folder Rule

Top-level `NN/API/*.lean` files are import entrypoints. Definitions belong in matching subfolders:

* `NN.API.Core` imports `NN.API.Core.Basic`.
* `NN.API.Data` imports `NN.API.Data.Core`; transforms live in `NN.API.Data.Transforms`.
* `NN.API.Text` imports `NN.API.Text.Core`; BPE and Unicode tables live in `NN.API.Text.*`.
* `NN.API.Common` imports `NN.API.Common.Core`.
* `NN.API.CLI` imports `NN.API.CLI.Core`.
* `NN.API.Init`, `NN.API.Json`, `NN.API.Macros`, `NN.API.Rand`, and `NN.API.Samples` import their
  matching `*.Core` modules.
* `NN.API.Public.Facade.Base`, `NN.API.Public.Facade.NN`,
  `NN.API.Public.Facade.Runtime`, `NN.API.Public.Facade.Data`, and
  `NN.API.Public.Facade.ModelZoo` are import-only entrypoints for their matching `*.Core`
  definition files.
* `NN.API.Public.Facade.Base.Core` is itself an import-only aggregator. The root public API
  definitions are split into `Base.Root`, `Base.Verification`, `Base.CLI`, `Base.ModelZoo`,
  `Base.Runtime`, and `Base.Tensor`.
* `NN.API.Public.Facade.Runtime.Core` is itself an import-only aggregator. The runtime API
  definitions are split into `Runtime.Autograd`, `Runtime.TensorPack`,
  `Runtime.ObjectiveAdapters`, `Runtime.RL`, `Runtime.Module`, `Runtime.Supervised`,
  `Runtime.Optim`, `Runtime.LossMetrics`, `Runtime.Text`, and `Runtime.Adapters`.
* `NN.API.Public.Facade.NN.Core` is itself an import-only aggregator. The `TorchLean.nn` API is
  split into `NN.Basic`, `NN.Summary`, `NN.Params`, `NN.Runtime`, `NN.Layers`, and `NN.Models`.
* `NN.API.Public.Facade.Data.Core` is itself an import-only aggregator. The public data API is
  split into `Data.Sample`, `Data.Datasets`, `Data.Text`, `Data.Builtin`, `Data.Checkpoint`, and
  `Data.DotInfo`.
* `NN.API.Public.Facade.Trainer` imports focused trainer modules for summaries, core types,
  construction, trained-result handles, runtime options, verification APIs, training
  definitions, and trained-result methods.
* `NN.API.Public.Facade.Trainer.Train` is itself an import-only aggregator. Training
  definitions are split into `Train.Regression`, `Train.CrossEntropy`, `Train.Custom`, and
  `Train.Streams`.
* `NN.API.Public.Facade.ModelZoo.Core` holds the reusable flags, logging, paths, and banners used by
  repository examples; `Trainer.Command` in the example tree owns runnable command orchestration.
* `NN.API.Public.Autograd`, `NN.API.Public.Seeded`, and `NN.API.Public.TensorPack` are import-only
  entrypoints for their matching `*.Core` definition files.
* `NN.API.Public.Training` imports `NN.API.Public.Training.Core`; this is the advanced
  callback/runner layer, not the path ordinary examples should teach first.

Entrypoint files preserve short import names. Implementation files live in the matching
subdirectories.

The public surface exposes scalar evidence, shape-indexed parameter packs, graph semantics, and
checker boundaries when those objects are part of the example or proof. The import story is:

* `import NN` for model, data, training, verification, and proof workflows.
* Focused `NN.API.*`, `NN.Runtime.*`, `NN.Spec.*`, and `NN.Proofs.*` imports for subsystem files.

## Related References

* Lean language reference, for the module/import and namespace mechanisms that make this API
  structure possible: <https://lean-lang.org/doc/reference/latest/>
* PyTorch documentation, for the familiar model/optimizer/dataloader vocabulary that the public
  training API follows where it helps readability:
  <https://pytorch.org/docs/stable/index.html>
