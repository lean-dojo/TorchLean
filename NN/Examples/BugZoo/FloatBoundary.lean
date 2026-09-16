/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all (Anonymous examples are checked but not retained as declarations.)

import FloatLib.Floats.Formats.IEEE754.Native.AddSub
import FloatLib.Floats.Formats.IEEE754.Native.Sqrt
import FloatLib.Floats.Formats.BinaryInterchange.Configured.Rounding.Proof

/-!
# BugZoo: floating-point trust boundaries

Floating point is not a cosmetic implementation detail. Robustness and equivalence proofs over real
numbers can become unsound when the deployed network runs with finite precision, fused operations,
different reduction order, denorm/flush behavior, or backend-specific kernels.

The key warning paper is:

- Jia and Rinard, “Exploiting Verified Neural Networks via Floating Point Numerical Error”,
  IEEE S&P Workshops 2020.
  https://doi.org/10.1109/SPW50608.2020.00058

Core `Float32` arithmetic has a logical definition through `Float32.Model`. FloatLib connects that
model to its configured software arithmetic. Its add/sub bridge requires finite operands;
its square-root bridge covers every input after NaN canonicalization. Configured division has
a total software-model refinement, which does not assert native Float32 division conformance.

Compiled CPU instructions and CUDA kernels still sit beyond that logical equality and require their
own backend-conformance evidence.
-/

@[expose] public section

namespace NN.Examples.BugZoo.FloatBoundary

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange
open ExecFloat.Binary

-- Importing and exporting a logical native value recovers it exactly.
example (a : Float32) :
    toFloat32 (ofFloat32 a) = a :=
  toFloat32_ofFloat32 a

-- Encoded NaN payloads are canonicalized by the logical native boundary.
example (x : Model FloatFormat.binary32) :
    Model.canonicalizeModel (Model.canonicalizeModel x) = Model.canonicalizeModel x :=
  Model.canonicalizeModel_idempotent x

-- Finite operands include signed zeros; the result may overflow.
example (a b : Float32) (ha : a.isFinite = true) (hb : b.isFinite = true) :
    ofFloat32 (a + b) = ofFloat32 a + ofFloat32 b :=
  ofFloat32_add_of_isFinite a b ha hb

example (a b : Float32) (ha : a.isFinite = true) (hb : b.isFinite = true) :
    ofFloat32 (a - b) = ofFloat32 a - ofFloat32 b :=
  ofFloat32_sub_of_isFinite a b ha hb

-- Configured division refines its model for every encoded pair and the selected rounding mode.
example (x y : ExecFloat.Binary 8 23) :
    toModel (ExecFloat.Binary.div x y .nearestEven) =
      Model.divWithRounding .nearestEven (toModel x) (toModel y) :=
  ExecFloat.Binary.toModel_div x y .nearestEven

-- Square root agreement also covers negative inputs, signed zeros, infinities, and NaNs.
example (a : Float32) :
    toModel (ofFloat32 a.sqrt) =
      Model.canonicalizeModel (Model.sqrt (toModel (ofFloat32 a))) :=
  toModel_ofFloat32_sqrt a

end NN.Examples.BugZoo.FloatBoundary
