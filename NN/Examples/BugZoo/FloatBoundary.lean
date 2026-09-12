/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Floats.IEEEExec.Bridge.LeanFloat32

/-!
# BugZoo: floating-point trust boundaries

Floating point is not a cosmetic implementation detail. Robustness and equivalence proofs over real
numbers can become unsound when the deployed network runs with finite precision, fused operations,
different reduction order, denorm/flush behavior, or backend-specific kernels.

The key warning paper is:

- Jia and Rinard, “Exploiting Verified Neural Networks via Floating Point Numerical Error”,
  IEEE S&P Workshops 2020.
  https://doi.org/10.1109/SPW50608.2020.00058

Core `Float32` arithmetic has a logical definition through `Float32.Model`. TorchLean proves that
model agrees with its independently implemented `IEEE32Exec` arithmetic for the modeled core
binary32 operations.

Compiled CPU instructions and CUDA kernels still sit beyond that logical equality and require their
own backend-conformance evidence.
-/

@[expose] public section

namespace NN.Examples.BugZoo.FloatBoundary

open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.Float32Bridge

-- Classification agreement is unconditional.
example (a : Float32) :
    Float32.isFinite a = IEEE32Exec.isFinite (toIEEE32Exec a) :=
  float32_isFinite_eq_ieee32 a

-- IEEE comparison, including unordered NaN cases and equality of signed zeros, agrees as well.
example (a b : Float32) :
    Float32.lt a b =
      (IEEE32Exec.compare (toIEEE32Exec a) (toIEEE32Exec b) == some .lt) :=
  float32_lt_eq_ieee32 a b

-- Addition agreement is proved for every canonical binary32 value.
example (a b : Float32) :
    toIEEE32Exec (a + b) =
      canonicalize (IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)) :=
  toIEEE32Exec_add a b

-- Division agreement includes finite values, exceptional values, and NaN canonicalization.
example (a b : Float32) :
    toIEEE32Exec (a / b) =
      canonicalize (IEEE32Exec.div (toIEEE32Exec a) (toIEEE32Exec b)) :=
  toIEEE32Exec_div a b

-- Square root agreement also covers negative inputs, signed zeros, infinities, and NaNs.
example (a : Float32) :
    toIEEE32Exec (Float32.sqrt a) =
      canonicalize (IEEE32Exec.sqrt (toIEEE32Exec a)) :=
  toIEEE32Exec_sqrt a

end NN.Examples.BugZoo.FloatBoundary
