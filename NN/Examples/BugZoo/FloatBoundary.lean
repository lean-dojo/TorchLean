/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeInitFloat32

/-!
# BugZoo: floating-point trust boundaries

Floating point is not a cosmetic implementation detail. Robustness and equivalence proofs over real
numbers can become unsound when the deployed network runs with finite precision, fused operations,
different reduction order, denorm/flush behavior, or backend-specific kernels.

The key warning paper is:

- Jia and Rinard, “Exploiting Verified Neural Networks via Floating Point Numerical Error”,
  IEEE S&P Workshops 2020.
  https://doi.org/10.1109/SPW50608.2020.00058

TorchLean's response is not to pretend Lean's runtime `Float32` is transparent. It is opaque to the
Lean kernel. So we expose the trust boundary as a typeclass: if the runtime `Float32` primitive
matches the bit-level `IEEE32Exec` operation, we may rewrite runtime arithmetic into the executable
IEEE-754 model and then use the internal floating-point theorems.

This file is deliberately small because the important part is the boundary itself. The theorem below
says: once the runtime assumption is supplied, ordinary runtime addition is reduced to the explicit
bit-level `IEEE32Exec.add` operation.
-/

@[expose] public section

namespace NN.Examples.BugZoo.FloatBoundary

open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.Float32Bridge

/--
Runtime `Float32` addition rewrites to the explicit bit-level IEEE executor only under the named
runtime-conformance assumption.

That is the honest TorchLean claim: floating-point deployment semantics are not silently smuggled
into proofs; they are either modeled by `IEEE32Exec` or isolated as a trust obligation.
-/
theorem runtimeFloat32_add_rewrites_to_ieee32
    [RuntimeFloat32MatchesIEEE32Exec] (a b : F32) :
    toIEEE32Exec (a + b) =
      IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b) :=
  RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_add a b

/-- The same boundary is available for division, where invalid-domain bugs often surface first. -/
theorem runtimeFloat32_div_rewrites_to_ieee32
    [RuntimeFloat32MatchesIEEE32Exec] (a b : F32) :
    toIEEE32Exec (a / b) =
      IEEE32Exec.div (toIEEE32Exec a) (toIEEE32Exec b) :=
  RuntimeFloat32MatchesIEEE32Exec.toIEEE32Exec_div a b

end NN.Examples.BugZoo.FloatBoundary

