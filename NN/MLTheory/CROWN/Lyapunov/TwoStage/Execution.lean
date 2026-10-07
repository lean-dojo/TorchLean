/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core

/-!
# Executable TwoStage Support

Executable support shared by pipelines (ii) and (iii).

These pipelines execute “inside Lean” using `α = IEEE32Exec` (an executable model of IEEE-754
float32). To keep the workflows reproducible, we provide:
- a deterministic sampler `UInt64 → (x ∈ [-rad, rad]^2)` that does not use `Float` anywhere, and
- a simple clamp routine for keeping PGD samples inside the training box.

This executable support layer lets the TwoStage pipelines run end-to-end under the verifier's
scalar semantics. The abstract CROWN theory does not depend on it.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.Execution

open NN.MLTheory.CROWN.Lyapunov.TwoStage.Core

local notation "Scalar" => (ExecFloat.Binary 8 23)

/-- Default learning rate used by the TwoStage workflows (`0.05`). -/
def defaultLr : Scalar := 1 / 20

/-- Default PGD step size used by the TwoStage workflows (`0.05`). -/
def defaultPgdStepSize : Scalar := 1 / 20

/-- Default sampling/clamp radius used by the TwoStage workflows (`2.0`). -/
def defaultRad : Scalar := 2

/-- Default check-box half-width used by the TwoStage workflows (`0.1`). -/
def defaultEpsCheck : Scalar := 1 / 10

/-- Clamp a scalar to `[lo, hi]`. -/
def clamp (lo hi x : Scalar) : Scalar :=
  if x < lo then lo else if x > hi then hi else x

/-- Clamp the two-dimensional state vector to `[lo, hi]^2`. -/
def clampStateTensor (lo hi : Scalar) (x : Tensor Scalar Core.xShape) : Tensor Scalar Core.xShape :=
  Tensor.map (clamp lo hi) x

/-!
Deterministic sampler: `UInt64` LCG → `α` in `[-rad, rad]`.

We use the top 24 bits of the LCG state as an integer in `[0, 2^24)`, then scale to `[0,1)`,
then to `[-rad, rad]`.
-/

/-- One step of Knuth's 64-bit linear congruential generator (MMIX constants).

We carry our own generator so a sampling run is reproducible from its seed alone, independent of
any platform RNG. -/
def lcgStep (s : UInt64) : UInt64 :=
  6364136223846793005 * s + 1442695040888963407

/-- Advance the LCG state and extract a 24-bit integer `u ∈ [0, 2^24)`. -/
def lcgU24 (s : UInt64) : UInt64 × Nat :=
  let s' := lcgStep s
  let u : UInt64 := (s' >>> 40) &&& 0xFFFFFF
  (s', u.toNat)

/-- Convert a 24-bit integer `u ∈ [0, 2^24)` to a scalar in `[0,1)`. -/
def unitIntervalSample (u : Nat) : Scalar :=
  (u : Scalar) / ((0x1000000 : Nat) : Scalar) -- divide by 2^24

/-- Build a state vector for the two-dimensional Lyapunov example. -/
def stateTensor (x1 x2 : Scalar) : Tensor Scalar Core.xShape :=
  Tensor.dim (n := Core.xDim) (fun i =>
    Tensor.scalar <|
      match i.val with
      | 0 => x1
      | _ => x2)

/-- Sample a point in `[-rad, rad]^2` from the LCG's 24-bit grid, using a deterministic seed. -/
def sampleStateTensor (seed : UInt64) (rad : Scalar) : UInt64 × Tensor Scalar Core.xShape :=
  let (s1, u1) := lcgU24 seed
  let (s2, u2) := lcgU24 s1
  let firstUniform : Scalar := unitIntervalSample u1
  let secondUniform : Scalar := unitIntervalSample u2
  let two : Scalar := 2
  let one : Scalar := 1
  let x1 := (two * firstUniform - one) * rad
  let x2 := (two * secondUniform - one) * rad
  (s2, stateTensor x1 x2)

end NN.MLTheory.CROWN.Lyapunov.TwoStage.Execution
