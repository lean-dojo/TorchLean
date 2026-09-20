/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all (These imports define the CI build coverage.)

import NN.Runtime
import NN.Runtime.Autograd.IRExec.Correctness
import NN.Runtime.Training.Log

/-!
# Additional Runtime Modules

The runtime umbrella leaves correctness developments and focused logging support opt-in. This target
checks their ordinary modules without pulling the end-to-end semantic-equivalence proof into `NN`.
-/

@[expose] public section
