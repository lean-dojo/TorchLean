/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Geometry3D.Box3D
public import NN.Verification.Geometry3D.CLI

/-!
# Geometry3D Verification

Reusable tensor-native checkers for 3D vision artifacts.

Real-model demo producers live under `scripts/verification/geometry3d` and write untrusted JSON
artifacts under `_external/geometry3d`. This namespace contains the Lean checker, theorem
statements, and a small CLI wrapper around the checker.
-/

@[expose] public section
