# NN/API

This directory collects the public API modules for TorchLean. The goal is to give users a small set
of imports with names that stay stable even when the runtime or spec internals move around.

Recommended imports:

* `import NN` for most users.
* `import NN.API.Public` for the PyTorch style facade (`NN.API.nn`, `NN.API.optim`, `NN.API.train`, ...).
* `import NN.API.Runtime` when you need the lower level runtime surface (`NN.API.TorchLean.*`).

Design goal:

Keep the public surface stable and discoverable while letting internal runtime modules evolve without
asking users to chase imports across `NN/Runtime/*`.
