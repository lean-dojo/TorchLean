/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor
public import NN.Entrypoint.Tensor
public import Std

/-!
# Floats Utils

Shared helpers for the Float runtime smoke tests.

These helpers are intentionally small: they keep the curated test files focused on the actual
regressions instead of re-declaring the same tensor accessors and approximate equality checks.
-/

@[expose] public section


open Spec
open Tensor

namespace Tests
namespace Floats
namespace Utils

/-- Approximate equality for smoke tests over `Float`. -/
def assertApprox (msg : String) (x y : Float) (tol : Float := 1e-5) : IO Unit := do
  if Float.abs (x - y) > tol then
    throw <| IO.userError s!"{msg}: got {x}, expected {y} (tol={tol})"

/-- Reject `NaN` and infinities in a smoke test. -/
def assertFinite (msg : String) (x : Float) : IO Unit := do
  if x.isNaN || x.isInf then
    throw <| IO.userError s!"{msg}: expected finite, got {x}"

/-- Check that a value lies in `[0, 1]` up to a small tolerance. -/
def assertIn01 (msg : String) (x : Float) : IO Unit := do
  if x < -1e-6 || x > 1.0 + 1e-6 then
    throw <| IO.userError s!"{msg}: expected in [0,1], got {x}"

/-- Read the scalar payload from a scalar tensor. -/
def scalarVal (t : Tensor Float Shape.scalar) : Float :=
  match t with
  | Tensor.scalar v => v

/-- Read one coordinate from a vector tensor. -/
def vecVal {n : Nat} (t : Tensor Float (.dim n .scalar)) (i : Fin n) : Float :=
  match t with
  | Tensor.dim f =>
      match f i with
      | Tensor.scalar v => v

/-- Read one coordinate from a matrix tensor. -/
def matVal {rows cols : Nat} (t : Tensor Float (.dim rows (.dim cols .scalar)))
    (i : Fin rows) (j : Fin cols) : Float :=
  match t with
  | Tensor.dim f =>
      match f i with
      | Tensor.dim g =>
          match g j with
          | Tensor.scalar v => v

/-- Read one coordinate from a `C × H × W` tensor. -/
def chwVal {c h w : Nat} (t : Tensor Float (.dim c (.dim h (.dim w .scalar))))
    (ci : Fin c) (hi : Fin h) (wi : Fin w) : Float :=
  match t with
  | Tensor.dim f =>
      match f ci with
      | Tensor.dim fh =>
          match fh hi with
          | Tensor.dim fw =>
              match fw wi with
              | Tensor.scalar v => v

end Utils
end Floats
end Tests
