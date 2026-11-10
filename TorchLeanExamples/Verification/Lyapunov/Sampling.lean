/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange
public import TorchLeanExamples.Verification.Lyapunov

/-!
# Executable TwoStage Support

Executable support for the all-in-Lean pipeline (`AllInLean.lean`).

The pipeline executes “inside Lean” using `α = ExecFloat.Binary 8 23` (FloatLib's executable
IEEE-754 binary32). To keep the workflow reproducible, we provide:
- a deterministic sampler `UInt64 → (x ∈ [-rad, rad]^2)` that does not use `Float` anywhere, and
- a simple clamp routine for keeping PGD samples inside the training box.

This support layer lets the pipeline run under the verifier's scalar semantics once the initial
parameters, built in `Float`, are cast to binary32. The abstract CROWN theory does not depend on it.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open TorchLean.Tensor

namespace TorchLean.Examples.Lyapunov

local notation "Scalar" => FloatLib.Floats.Binary32

/-- Default learning rate used by the TwoStage workflow (`0.05`). -/
def defaultLr : Scalar := 1 / 20

/-- Default PGD step size used by the TwoStage workflow (`0.05`). -/
def defaultPgdStepSize : Scalar := 1 / 20

/-- Default sampling/clamp radius used by the TwoStage workflow (`2.0`). -/
def defaultRad : Scalar := 2

/-- Default check-box half-width used by the TwoStage workflow (`0.1`). -/
def defaultEpsCheck : Scalar := 1 / 10

/-- Clamp a scalar to `[lo, hi]`. -/
def clamp (lo hi x : Scalar) : Scalar :=
  if x < lo then lo else if x > hi then hi else x

/-- Clamp the two-dimensional state vector to `[lo, hi]^2`. -/
def clampStateTensor (lo hi : Scalar) (x : Tensor Scalar xShape) : Tensor Scalar xShape :=
  Tensor.map (clamp lo hi) x

/-!
Deterministic sampler: two keyed integers in `[0, 2^24)` from `Random.sampleNat`, scaled to
`[0, 1)` and then to `[-rad, rad]`. A run is reproducible from its seed alone.
-/

/-- Convert a 24-bit integer `u ∈ [0, 2^24)` to a scalar in `[0,1)`. -/
def unitIntervalSample (u : Nat) : Scalar :=
  (u : Scalar) / ((0x1000000 : Nat) : Scalar) -- divide by 2^24

/-- Build a state vector for the two-dimensional Lyapunov example. -/
def stateTensor (x1 x2 : Scalar) : Tensor Scalar xShape :=
  Tensor.dim (n := xDim) (fun i =>
    Tensor.scalar <|
      match i.val with
      | 0 => x1
      | _ => x2)

/-- Sample a point in `[-rad, rad]^2` on the 24-bit grid from the key `seed`, returning the next
key `Random.splitmix64 seed`. -/
def sampleStateTensor (seed : UInt64) (rad : Scalar) : UInt64 × Tensor Scalar xShape :=
  let u1 := Random.sampleNat seed 0 0x1000000
  let u2 := Random.sampleNat seed 1 0x1000000
  let firstUniform : Scalar := unitIntervalSample u1
  let secondUniform : Scalar := unitIntervalSample u2
  let two : Scalar := 2
  let one : Scalar := 1
  let x1 := (two * firstUniform - one) * rad
  let x2 := (two * secondUniform - one) * rad
  (Random.splitmix64 seed, stateTensor x1 x2)

end TorchLean.Examples.Lyapunov
