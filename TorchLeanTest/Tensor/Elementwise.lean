/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Tactic.Init

/-!
# Elementwise operator checks

Hand-computed values of the elementwise operators of `TorchLean.Tensor.Elementwise` on
host binary64 tensors, at the cases where a policy matters: ties of `round`, signed zeros, NaN
operands of `sign`, `maximum` and `fmax`, the signs of `fmod` and `remainder`, and both branches of
`lerp`. The expected values are IEEE 754-2019's, Python's `%` on CPU for `remainder`
(`pythonRemainders`), and `jax.lax.sign`'s for `sign`. PyTorch 2.13 returns them too, except
`torch.sign` at a NaN and at `-0` and the sign of a zero `torch.remainder`. `run` executes the
compiled checks.
-/

@[expose] public section

namespace TorchLean.Test.Elementwise

open TorchLean.Tensor

/-- The entries of a vector, as an array. -/
def entries {n : Nat} (x : Tensor Float [n]) : Array Float := x.to (Array Float)

/-- Exact equality of arrays that may hold NaNs, compared by their bit patterns. -/
def sameBits (x y : Array Float) : Bool := x.map Float.toBits == y.map Float.toBits

/-- A NaN. -/
def nan : Float := 0.0 / 0.0

/-- Positive infinity. -/
def inf : Float := 1.0 / 0.0

/-- Ties of `round` from `-2.5` to `2.5`, also the inputs of `floor`, `ceil`, `trunc` and `frac`. -/
def halves : Tensor Float [6] := [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

/-- A NaN, `-0`, `3` and `-∞`, the inputs of `sign`, `isNaN` and `isFinite`. -/
def specials : Tensor Float [4] := [nan, -0.0, 3.0, -inf]

/-- Dividends of `fmod` and `remainder`, with a negative zero. -/
def dividends : Tensor Float [4] := [5.0, -5.0, 5.0, -0.0]

/-- Divisors of `fmod` and `remainder`, one of them negative. -/
def divisors : Tensor Float [4] := [3.0, 3.0, -3.0, 3.0]

/-- Left operands of `maximum` and `fmax`: a NaN, `1` and `-0`. -/
def left : Tensor Float [3] := [nan, 1.0, -0.0]

/-- Right operands of `maximum` and `fmax`: `1`, a NaN and `+0`. -/
def right : Tensor Float [3] := [1.0, nan, 0.0]

/-- Magnitudes for `copysign`. -/
def magnitudes : Tensor Float [3] := [3.0, 2.0, 0.0]

/-- Sign sources for `copysign`, with a negative zero. -/
def signSources : Tensor Float [3] := [-1.0, 1.0, -0.0]

/-- Dividends, divisors and Python's `a % b` for them, computed by CPython 3.12 on x86-64. At the
zero divisor, where Python raises, the entry is NumPy's NaN. -/
def pythonRemainders : List (Float × Float × Float) :=
  [(5.0, 3.0, 2.0), (-5.0, 3.0, 1.0), (5.0, -3.0, -1.0), (-5.0, -3.0, -2.0), (-0.0, 3.0, 0.0),
    (0.0, -3.0, -0.0), (-2.0, 1.0, 0.0), (2.0, -1.0, -0.0), (7.5, -0.25, -0.0), (-7.5, 0.25, 0.0),
    (-1.0e-30, 1.0, 1.0), (1.0e-30, -1.0, -1.0), (1.0, inf, 1.0), (-1.0, inf, inf),
    (1.0, -inf, -inf), (-0.0, inf, 0.0), (0.0, -inf, -0.0), (inf, 3.0, nan), (nan, 1.0, nan),
    (1.0, nan, nan), (1.0, 0.0, nan)]

/-- Fail with `label` unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"elementwise check failed: {label}"

/-- Run every check. -/
def run : IO Unit := do
  expect "round ties to even" <|
    sameBits (entries (Tensor.round halves)) #[-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]
  expect "floor" <|
    sameBits (entries (Tensor.floor halves)) #[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
  expect "ceil" <|
    sameBits (entries (Tensor.ceil halves)) #[-2.0, -1.0, -0.0, 1.0, 2.0, 3.0]
  expect "trunc" <|
    sameBits (entries (Tensor.trunc halves)) #[-2.0, -1.0, -0.0, 0.0, 1.0, 2.0]
  expect "frac" <|
    sameBits (entries (Tensor.frac halves)) #[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
  expect "sign at NaN and zeros" <|
    sameBits (entries (Tensor.sign specials)) #[nan, -0.0, 1.0, -1.0]
  expect "isNaN" <|
    entries (Tensor.map (fun b => if b then 1.0 else 0.0) (Tensor.isNaN specials)) ==
    #[1.0, 0.0, 0.0, 0.0]
  expect "isFinite" <|
    entries (Tensor.map (fun b => if b then 1.0 else 0.0) (Tensor.isFinite specials)) ==
    #[0.0, 1.0, 1.0, 0.0]
  expect "fmod sign of dividend" <|
    sameBits (entries (Tensor.fmod dividends divisors)) #[2.0, -2.0, 2.0, -0.0]
  expect "remainder sign of divisor" <|
    sameBits (entries (Tensor.remainder dividends divisors)) #[2.0, 1.0, -1.0, 0.0]
  for (a, b, r) in pythonRemainders do
    let value := Context.remainder a b
    expect s!"remainder {a} % {b} is Python's {r}" <|
      if r.isNaN then value.isNaN else value.toBits == r.toBits
  expect "maximum propagates NaN" <|
    (entries (Tensor.maximum left right)).map Float.isNaN == #[true, true, false]
  expect "maximum and minimum on signed zeros, the order of IEEE 754-2019 9.6" <|
    sameBits (entries (Tensor.maximum ([-0.0, 0.0] : Tensor Float [2]) [0.0, -0.0]) ++
      entries (Tensor.minimum ([-0.0, 0.0] : Tensor Float [2]) [0.0, -0.0])) #[0.0, 0.0, -0.0, -0.0]
  expect "clamp of -0 at a +0 lower bound" <|
    sameBits (entries (Tensor.clamp ([-0.0] : Tensor Float [1]) (Tensor.full [1] 0.0)
      (Tensor.full [1] 1.0))) #[0.0]
  expect "fmax ignores NaN" <|
    sameBits (entries (Tensor.fmax left right)) #[1.0, 1.0, 0.0]
  expect "fmax and fmin on signed zeros, IEEE 754-2019 maximumNumber and minimumNumber" <|
    sameBits (entries (Tensor.fmax ([-0.0, 0.0] : Tensor Float [2]) [0.0, -0.0]) ++
      entries (Tensor.fmin ([-0.0, 0.0] : Tensor Float [2]) [0.0, -0.0])) #[0.0, 0.0, -0.0, -0.0]
  expect "fmax and fmin of two NaNs" <|
    (Context.fmax nan nan).isNaN && (Context.fmin nan nan).isNaN
  expect "copysign sign bits" <|
    sameBits (entries (Tensor.copysign magnitudes signSources)) #[-3.0, 2.0, -0.0]
  expect "clamp lo > hi" <|
    sameBits (entries (Tensor.clamp ([0.5, 5.0, -5.0] : Tensor Float [3])
    (Tensor.full [3] 2.0) (Tensor.full [3] 1.0))) #[1.0, 1.0, 1.0]
  expect "select" <|
    entries (Tensor.select ([true, false] : Tensor Bool [2]) ([1.0, 2.0] : Tensor Float [2])
    ([3.0, 4.0] : Tensor Float [2])) == #[1.0, 4.0]
  expect "lerp both branches" <|
    entries (Tensor.lerp ([1.0e16, 1.0] : Tensor Float [2]) ([1.0, 1.0e16] : Tensor Float [2])
    ([0.75, -0.25] : Tensor Float [2])) == #[2500000000000001.0, -2499999999999999.0]
  expect "nanToNum" <|
    entries (Tensor.nanToNum ([nan, inf, -inf, 1.0] : Tensor Float [4]) 0.0 7.0 (-7.0)) ==
    #[0.0, 7.0, -7.0, 1.0]
  expect "addcmul" <|
    entries (Tensor.addcmul (Tensor.full [1] 1.0) (Tensor.full [1] 3.0)
    (Tensor.full [1] 0.5) 2.0) == #[4.0]
  expect "exp2" <|
    entries (Tensor.exp2 (Tensor.full [1] 10.0)) == #[1024.0]
  IO.println "  elementwise operators: passed"

end TorchLean.Test.Elementwise
