/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb.Oracle

/-!
# `NN.Floats.Arb`

Umbrella import for the Arb (ball arithmetic) oracle integration.

This is an **external** backend (python-flint / Arb/FLINT) called via `IO.Process`.
It is useful for:

- rigorous enclosures of real-valued expressions (ball arithmetic),
- tight bounds for monotone nonlinearities,
- cross-checking TorchLean’s numeric models (`ExecFloat.Binary 8 23`, `FP32`/`NF`).

Trust boundary: the Python/Arb toolchain is an oracle. This module parses and packages its output;
it does not by itself prove the Arb enclosure certificate inside Lean.
-/

@[expose] public section
