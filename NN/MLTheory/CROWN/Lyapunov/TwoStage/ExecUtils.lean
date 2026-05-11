/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.Spec
public import NN.Floats.IEEEExec
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core

/-!
# Executable TwoStage Utilities

Executable utilities shared by pipelines (ii) and (iii).

These pipelines execute “inside Lean” using `α = IEEE32Exec` (an executable model of IEEE-754
float32). To keep the workflows reproducible, we provide:
- a deterministic sampler `UInt64 → (x ∈ [-rad, rad]^2)` that does not use `Float` anywhere, and
- a simple clamp routine for keeping PGD samples inside the training box.

This file is not part of the abstract CROWN theory. It is the executable support layer that lets
the TwoStage pipelines run end-to-end under the same scalar semantics used by the verifier.
-/

@[expose] public section


open Spec
open Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils

open TorchLean.Floats.IEEE754
open NN.MLTheory.CROWN.Lyapunov.TwoStage.Core

/-- Executable float32 semantics used by the TwoStage workflows. -/
abbrev α : Type := IEEE32Exec

/-- Coerce a natural number into `IEEE32Exec`. -/
def nat (k : Nat) : α := ((k : Nat) : α)

/-- Default learning rate used by the TwoStage workflows (`0.05`). -/
def defaultLr : α := nat 1 / nat 20

/-- Default PGD step size used by the TwoStage workflows (`0.05`). -/
def defaultPgdStepSize : α := nat 1 / nat 20

/-- Default sampling/clamp radius used by the TwoStage workflows (`2.0`). -/
def defaultRad : α := nat 2

/-- Default check-box half-width used by the TwoStage workflows (`0.1`). -/
def defaultEpsCheck : α := nat 1 / nat 10

/-- Clamp a scalar to `[lo, hi]`. -/
def clamp (lo hi x : α) : α :=
  if x < lo then lo else if x > hi then hi else x

/-- Clamp a 2D state vector to `[lo, hi]^2`. -/
def clampVec2 (lo hi : α) (x : Tensor α Core.xShape) : Tensor α Core.xShape :=
  Tensor.dim (fun i =>
    let xi := Tensor.vecGet x i
    Tensor.scalar (clamp lo hi xi))

/-!
Deterministic sampler: `UInt64` LCG → `α` in `[-rad, rad]`.

We use the top 24 bits of the LCG state as a uniform integer in `[0, 2^24)`, then scale to `[0,1)`,
then to `[-rad, rad]`.
-/

def lcgStep (s : UInt64) : UInt64 :=
  6364136223846793005 * s + 1442695040888963407

/-- Advance the LCG state and extract a 24-bit uniform integer `u ∈ [0, 2^24)`. -/
def lcgU24 (s : UInt64) : UInt64 × Nat :=
  let s' := lcgStep s
  let u : UInt64 := (s' >>> 40) &&& 0xFFFFFF
  (s', u.toNat)

/-- Convert a 24-bit integer `u ∈ [0, 2^24)` to a scalar in `[0,1)`. -/
def u01 (u : Nat) : α :=
  (u : α) / ((0x1000000 : Nat) : α) -- divide by 2^24

/-- Build a length-2 state vector. -/
def vec2 (x1 x2 : α) : Tensor α Core.xShape :=
  Tensor.dim (n := Core.xDim) (s := .scalar) (fun i =>
    Tensor.scalar <|
      match i.val with
      | 0 => x1
      | _ => x2)

/-- Sample a point uniformly from `[-rad, rad]^2` using a deterministic PRNG seed. -/
def sampleVec2 (seed : UInt64) (rad : α) : UInt64 × Tensor α Core.xShape :=
  let (s1, u1) := lcgU24 seed
  let (s2, u2) := lcgU24 s1
  let r1 : α := u01 u1
  let r2 : α := u01 u2
  let two : α := nat 2
  let one : α := nat 1
  let x1 := (two * r1 - one) * rad
  let x2 := (two * r2 - one) * rad
  (s2, vec2 x1 x2)

end NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils
