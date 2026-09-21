/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Backend
public import NN.Floats
public import NN.GraphSpec
public import NN.IR
public import NN.MLTheory
public import NN.Proofs
public import NN.Runtime
public import NN.Spec
public import NN.Tactic
public import NN.Tensor
public import NN.Verification
public import NN.Widgets

/-!
# NN

Complete TorchLean umbrella import. Use it when one file spans application code and the proof,
verification, graph, floating-point, widget, or backend subsystems:

```lean
import NN
open TorchLean
```

Application files that only need models, data, and training can use `import NN.API`. The
user-facing names live under the `TorchLean` namespace. Direct subsystem imports such as
`NN.Spec`, `NN.Runtime`, `NN.Proofs`, and `NN.Verification` are available when a file does not need
the full library.
-/

@[expose] public section
