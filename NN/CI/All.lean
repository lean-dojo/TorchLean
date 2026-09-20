/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all (These imports define the CI build coverage.)

import NN.CI.Floats
import NN.CI.Foundation
import NN.CI.Runtime
import NN.CI.Theory
import NN.CI.Verification

/-!
# Complete CI Import Surface

This CI-only umbrella imports maintained library modules that are intentionally absent from the
downstream `NN` umbrella. Examples, tests, and the end-to-end IR proof have their own Lake targets.

```bash
lake build NN.CI.All
```
-/
