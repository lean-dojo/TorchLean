/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Fft
public import NN.Runtime.Autograd.TorchLean.Fft
public import Mathlib.Analysis.RCLike.Sqrt

/-!
# Runtime FFT transport lemmas (`NN.Runtime.*.Fft` → mathlib `ℂ`)

`NN.Runtime.Autograd.TorchLean.Fft` defines FFT/IFFT matrices using “twiddle factors” written as
`cos θ ± i sin θ` so the definitions work for TorchLean’s runtime complex scalar
`TorchLean.Complex β`.

For the **exact** DFT inversion proof we instead worked in mathlib’s `ℂ` using primitive roots of
unity (`ωₙ = exp(-2πi/n)`), since that is where the classical algebraic facts live.

This file bridges the two views:

- on `ℂ`, TorchLean’s `twiddle` factor is exactly `ωₙ^(j*k)`,
- on `ℂ`, TorchLean’s `twiddleInv` factor is exactly `ζₙ^(j*k)`,
- therefore the runtime DFT/IDFT *matrices* (viewed entrywise) coincide with the matrices in
  `NN.Proofs.Analysis.Fft`.

We keep the `Context ℂ` instance **local** to this module. Its ordering is arbitrary (real-part
order) and is irrelevant to FFT; it exists only to instantiate the scalar-polymorphic runtime
definitions at `α := ℂ`.

Why this is not merged into `Fft.lean`: `Fft.lean` is pure Fourier algebra over mathlib matrices.
This file imports the TorchLean runtime FFT definitions and therefore sits at the boundary between
runtime code and the exact theorem. The separation keeps the core inversion theorem lightweight and
lets downstream proofs import only the pure DFT facts when they do not need runtime transport.
-/

@[expose] public section

noncomputable section

namespace Proofs
namespace FftBridge

open scoped BigOperators

open Spec

-- ---------------------------------------------------------------------------
-- Local `Context ℂ` instance (only used to instantiate the runtime FFT definitions).
-- ---------------------------------------------------------------------------

section ComplexContext

open Complex

local instance : Coe Nat ℂ where
  coe n := (Nat.cast n : ℂ)

@[simp] private lemma coeNat_eq_natCast (n : Nat) : (Coe.coe n : ℂ) = (n : ℂ) := rfl

local instance : LT ℂ := ⟨fun x y => x.re < y.re⟩
local instance : LE ℂ := ⟨fun x y => x.re ≤ y.re⟩

local instance : Max ℂ where
  max x y := if x.re > y.re then x else y

local instance : Min ℂ where
  min x y := if x.re > y.re then y else x

local instance : BEq ℂ where
  beq x y := by
    classical
    exact decide (x = y)

/- The runtime FFT definitions are scalar-polymorphic over `Context α`.

Only `pi`, `cos`, `sin`, and `sqrt(-1)` are relevant for the FFT twiddle factors. The remaining
`MathFunctions` fields use local fallback definitions so we can instantiate the runtime definitions
at `α := ℂ` without making this file a global complex-analysis backend.
-/
noncomputable local instance : MathFunctions ℂ where
  exp := Complex.exp
  tanh := fun _ => 0
  cosh := fun _ => 0
  sqrt := fun z => sqrt z
  abs := fun z => z
  log := fun _ => 0
  pi := (Real.pi : ℂ)
  cos := Complex.cos
  sin := Complex.sin
  sinh := fun _ => 0

noncomputable local instance : Numbers ℂ where
  neg_point_five := (-1 : ℂ) / 2
  neg_one := (-1 : ℂ)
  pointone := (1 : ℂ) / 10
  pointfive := (1 : ℂ) / 2
  one := (1 : ℂ)
  zero := (0 : ℂ)
  two := (2 : ℂ)
  three := (3 : ℂ)
  four := (4 : ℂ)
  five := (5 : ℂ)
  ten := (10 : ℂ)
  log10 := 0
  log10000 := 0
  epsilon := (1 : ℂ) / 1000000
  neg_thousand := (-1000 : ℂ)

/- Local-only `Context ℂ`: the ordering is not mathematically meaningful for complex numbers.
It is present solely because the generic TorchLean runtime context class includes order-dependent
operations used by other tensor code. The FFT bridge below never relies on that order.
-/
local instance : Context ℂ := {
  decidable_gt := fun x y => inferInstanceAs (Decidable (x > y))
}

-- ---------------------------------------------------------------------------
-- Twiddle factors on `ℂ`.
-- ---------------------------------------------------------------------------

open Runtime.Autograd.TorchLean.NN

/--
TorchLean's runtime FFT uses `sqrt(-1)` as its scalar-polymorphic imaginary unit.

When we instantiate the runtime definitions at mathlib `ℂ`, that value is the usual
`Complex.I`.
-/
private lemma FFT1D_I_eq :
    Runtime.Autograd.TorchLean.NN.FFT1D.I (α := ℂ) = (Complex.I : ℂ) := by
  -- `FFT1D.I` is defined as `sqrt(-1)`.
  simpa [Runtime.Autograd.TorchLean.NN.FFT1D.I, Numbers.neg_one] using (Complex.sqrt_neg_one)

/--
Runtime forward twiddle factors match the negative-frequency root powers from the exact DFT
development.

TorchLean runtime definition:

`cos θ - I * sin θ`

Exact DFT definition:

`ωₙ^(j*k) = exp(-2π i j k / n)`.

The proof is just Euler's formula plus scalar normalization of the exponent.
-/
theorem twiddle_eq_omega_pow (n j k : Nat) (hn : n ≠ 0) :
    Runtime.Autograd.TorchLean.NN.FFT1D.twiddle (α := ℂ) n j k =
      Proofs.Fft.ω n ^ (j * k) := by
  have hn0 : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  set θ : ℂ := (Numbers.two : ℂ) * MathFunctions.pi * (j : ℂ) * (k : ℂ) / (n : ℂ)

  have hEuler :
      Complex.exp (-(θ * Complex.I)) =
        MathFunctions.cos θ - Complex.I * MathFunctions.sin θ := by
    calc
      Complex.exp (-(θ * Complex.I)) = Complex.exp ((-θ) * Complex.I) := by
        simp [neg_mul]
      _ = Complex.cos (-θ) + Complex.sin (-θ) * Complex.I := by
            simpa using (Complex.exp_mul_I (-θ))
      _ = Complex.cos θ - Complex.sin θ * Complex.I := by
            simp [Complex.cos_neg, Complex.sin_neg, sub_eq_add_neg]
      _ = Complex.cos θ - Complex.I * Complex.sin θ := by
            simp [mul_comm]
      _ = MathFunctions.cos θ - Complex.I * MathFunctions.sin θ := by
            rfl

  have htw :
      Runtime.Autograd.TorchLean.NN.FFT1D.twiddle (α := ℂ) n j k =
        Complex.exp (-(θ * Complex.I)) := by
    -- Unfold `twiddle` and rewrite `I` as `Complex.I`.
    simp [Runtime.Autograd.TorchLean.NN.FFT1D.twiddle, θ, FFT1D_I_eq, hEuler]

  have hω : Proofs.Fft.ω n = Complex.exp (-(2 * Real.pi * Complex.I / (n : ℂ))) := by
    simp [Proofs.Fft.ω, Proofs.Fft.ζ, Complex.exp_neg]

  have hωpow :
      Proofs.Fft.ω n ^ (j * k) =
        Complex.exp ((j * k : ℕ) * (-(2 * Real.pi * Complex.I / (n : ℂ)))) := by
    simpa [hω] using
      (Complex.exp_nat_mul (-(2 * Real.pi * Complex.I / (n : ℂ))) (j * k)).symm

  -- Match exponents: `θ = 2π * j*k / n` and multiplication is commutative.
  have hθ : θ = (2 * Real.pi * (j : ℂ) * (k : ℂ)) / (n : ℂ) := by
    simp [θ, Numbers.two, MathFunctions.pi, mul_left_comm, mul_comm]

  have hexp :
      -θ * Complex.I =
        (j * k : ℕ) * (-(2 * Real.pi * Complex.I / (n : ℂ))) := by
    have hjk : ((j * k : ℕ) : ℂ) = (j : ℂ) * (k : ℂ) := by
      simp [Nat.cast_mul]
    calc
      -θ * Complex.I
          = -((2 * Real.pi * (j : ℂ) * (k : ℂ)) / (n : ℂ)) * Complex.I := by
              simp [hθ]
      _ = ((j * k : ℕ) : ℂ) * (-(2 * Real.pi * Complex.I / (n : ℂ))) := by
              simp [hjk, div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm]
      _ = (j * k : ℕ) * (-(2 * Real.pi * Complex.I / (n : ℂ))) := by rfl

  calc
    Runtime.Autograd.TorchLean.NN.FFT1D.twiddle (α := ℂ) n j k
        = Complex.exp (-(θ * Complex.I)) := htw
    _ = Complex.exp (-θ * Complex.I) := by
          simp [neg_mul]
    _ = Complex.exp ((j * k : ℕ) * (-(2 * Real.pi * Complex.I / (n : ℂ)))) := by
          simp [hexp]
    _ = Proofs.Fft.ω n ^ (j * k) := by
          simp [hωpow]

/--
Runtime inverse twiddle factors match the positive-frequency root powers from the exact IDFT
development.

This is the inverse-direction analogue of `twiddle_eq_omega_pow`: the runtime
`cos θ + I * sin θ` term is `ζₙ^(j*k)`.
-/
theorem twiddleInv_eq_zeta_pow (n j k : Nat) (hn : n ≠ 0) :
    Runtime.Autograd.TorchLean.NN.FFT1D.twiddleInv (α := ℂ) n j k =
      Proofs.Fft.ζ n ^ (j * k) := by
  have hn0 : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  set θ : ℂ := (Numbers.two : ℂ) * MathFunctions.pi * (j : ℂ) * (k : ℂ) / (n : ℂ)

  have hEuler :
      Complex.exp (θ * Complex.I) =
        MathFunctions.cos θ + Complex.I * MathFunctions.sin θ := by
    calc
      Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.sin θ * Complex.I := by
        simpa using (Complex.exp_mul_I θ)
      _ = Complex.cos θ + Complex.I * Complex.sin θ := by
        simp [mul_comm]
      _ = MathFunctions.cos θ + Complex.I * MathFunctions.sin θ := by
        rfl

  have htw :
      Runtime.Autograd.TorchLean.NN.FFT1D.twiddleInv (α := ℂ) n j k =
        Complex.exp (θ * Complex.I) := by
    simp [Runtime.Autograd.TorchLean.NN.FFT1D.twiddleInv, θ, FFT1D_I_eq, hEuler]

  have hζ : Proofs.Fft.ζ n = Complex.exp (2 * Real.pi * Complex.I / (n : ℂ)) := by
    simp [Proofs.Fft.ζ]

  have hζpow :
      Proofs.Fft.ζ n ^ (j * k) =
        Complex.exp ((j * k : ℕ) * (2 * Real.pi * Complex.I / (n : ℂ))) := by
    simpa [hζ] using (Complex.exp_nat_mul (2 * Real.pi * Complex.I / (n : ℂ)) (j * k)).symm

  have hθ : θ = (2 * Real.pi * (j : ℂ) * (k : ℂ)) / (n : ℂ) := by
    simp [θ, Numbers.two, MathFunctions.pi, mul_left_comm, mul_comm]

  have hexp :
      θ * Complex.I =
        (j * k : ℕ) * (2 * Real.pi * Complex.I / (n : ℂ)) := by
    have hjk : ((j * k : ℕ) : ℂ) = (j : ℂ) * (k : ℂ) := by
      simp [Nat.cast_mul]
    calc
      θ * Complex.I
          = ((2 * Real.pi * (j : ℂ) * (k : ℂ)) / (n : ℂ)) * Complex.I := by
              simp [hθ]
      _ = ((j * k : ℕ) : ℂ) * (2 * Real.pi * Complex.I / (n : ℂ)) := by
              simp [hjk, div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm]
      _ = (j * k : ℕ) * (2 * Real.pi * Complex.I / (n : ℂ)) := by rfl

  calc
    Runtime.Autograd.TorchLean.NN.FFT1D.twiddleInv (α := ℂ) n j k
        = Complex.exp (θ * Complex.I) := htw
    _ = Complex.exp ((j * k : ℕ) * (2 * Real.pi * Complex.I / (n : ℂ))) := by
          simp [hexp]
    _ = Proofs.Fft.ζ n ^ (j * k) := by
          simp [hζpow]

-- ---------------------------------------------------------------------------
-- Matrix entries: runtime tensors coincide with the mathlib matrices from `NN.Proofs.Analysis.Fft`.
-- ---------------------------------------------------------------------------

/--
Entrywise bridge from the runtime tensor DFT matrix to the exact mathlib DFT matrix.

This is the first point where we leave pure root-of-unity algebra and connect to TorchLean's
shape-indexed tensor representation.
-/
theorem dftMatrix_entry_eq (n : Nat) (hn : n ≠ 0) (k j : Fin n) :
    Spec.get2 (Runtime.Autograd.TorchLean.NN.FFT1D.dftMatrix (α := ℂ) n) k j =
      Proofs.Fft.dftMatrix n k j := by
  -- `get2` reduces the tensor constructor and exposes `twiddle`.
  simp [Spec.get2, Spec.get, Spec.getAtSpec,
    Runtime.Autograd.TorchLean.NN.FFT1D.dftMatrix, Proofs.Fft.dftMatrix,
    twiddle_eq_omega_pow (n := n) (j := j.val) (k := k.val) hn]

/--
Entrywise bridge from the runtime tensor IDFT matrix to the exact mathlib inverse DFT matrix.

Together with `dftMatrix_entry_eq`, this is the transport layer needed to reuse the pure DFT
inversion theorem for runtime FFT matrix definitions.
-/
theorem idftMatrix_entry_eq (n : Nat) (hn : n ≠ 0) (j k : Fin n) :
    Spec.get2 (Runtime.Autograd.TorchLean.NN.FFT1D.idftMatrix (α := ℂ) n) j k =
      Proofs.Fft.idftMatrix n j k := by
  -- `get2` reduces the tensor constructor and exposes `twiddleInv`.
  simp [Spec.get2, Spec.get, Spec.getAtSpec,
    Runtime.Autograd.TorchLean.NN.FFT1D.idftMatrix, Proofs.Fft.idftMatrix,
    twiddleInv_eq_zeta_pow (n := n) (j := j.val) (k := k.val) hn]

end ComplexContext

end FftBridge
end Proofs
