/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all (These imports define the CI build coverage.)

import FloatLib
import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
import NN.Proofs.RuntimeApprox.FP32

/-!
# Additional Floating-Point Modules

This target checks the shared FloatLib interface, its optional binary elementary functions, and
TorchLean's runtime approximation proofs.
-/

@[expose] public section
