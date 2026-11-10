/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Scalar

/-!
# NF Sum Reduction Bounds

Forward-error bounds for rounded sum reductions. The accumulator pairs the runtime value with its
error budget, and each addition grows that budget by the incoming element's error and one rounding
term. The bounds follow the left-to-right traversal of `Tensor.sum`, starting at rounded zero. A
different reduction tree needs its own bound, as does a native kernel whose accumulation order is
not known to match.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- One step of a rounded sum that carries its absolute-error budget. The budget grows by the
incoming element's error `epsElem` and half an ULP of the exact sum of accumulator and element. -/
def NF.sumStep (epsElem : ℝ) : (R × ℝ) → R → (R × ℝ)
  | (accR, epsAcc), xR =>
      let epsAcc' : ℝ :=
        epsAcc + epsElem +
          ulp β fexp
              (NF.toReal accR +
                NF.toReal xR) / 2
      (accR + xR, epsAcc')

/-- Fold `sumStep` over a tensor, carrying the rounded sum and its error budget together. -/
def NF.sumFoldState {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) : (R × ℝ) :=
  Tensor.foldl (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st tR

/-- Absolute-error budget for `Tensor.sum`, given an elementwise bound `epsElem`. It starts at
`ulp 0 / 2` and grows by `epsElem` plus one rounding term per addition. The starting term is
conservative, since an empty tensor sums to exactly zero. -/
def NF.sumBound {s : Shape} (epsElem : ℝ) (tR : Tensor R s) : ℝ :=
  (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem
    ((0 : R), ulp β fexp 0 / 2) tR).2

omit [ValidRndToNearest rnd] in
/-- The first component of `sumFoldState` is the plain rounded fold `Tensor.foldl (· + ·)`: the
bookkeeping changes neither the values nor the order of the additions. -/
private theorem NF.sumFoldState_fst_eq {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) :
    (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st tR).1 =
      Tensor.foldl (· + ·) st.1 tR := by
  simpa only [sumFoldState, Tensor.Rep.map_id] using
    Tensor.Rep.map_foldl tR
      (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st
      id Prod.fst (· + ·) (fun _ _ => rfl)

/-- The scalar addition bound is a fold invariant, for arbitrary tensor shapes. -/
private theorem NF.approx_sumFoldState {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {accS : ℝ} {st : R × ℝ} {epsElem : ℝ},
      abs (NF.toReal st.1 - accS) ≤ st.2 →
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR epsElem →
        abs
            (NF.toReal
                (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).1 -
              Tensor.foldl (· + ·) accS xS) ≤
          (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).2 := by
  intro xS xR accS st epsElem hAcc hx
  exact Tensor.Rep.foldl_rel xR xS
    (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) (· + ·)
    (fun state reference => |NF.toReal state.1 - reference| ≤ state.2) st accS hAcc
    (fun index state reference hstate => by
      simpa only [sumStep] using
        approx_add (β := β) (fexp := fexp) (rnd := rnd)
          (x := reference) (y := xS index) (xR := state.1) (yR := xR index)
          hstate (ApproxTensor.abs_sub_le hx index))

/-- If the inputs are elementwise within `eps`, the rounded and real sums differ by at most
`sumBound eps xR`. -/
theorem ApproxTensor.sum {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.scalar (Tensor.sum (α := ℝ) (s := s) xS))
          (Tensor.scalar (Tensor.sum (α := R) (s := s) xR))
          (NF.sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR) := by
  intro xS xR eps hx
  -- Start from accumulator 0 with a conservative rounding bound.
  let initEps : ℝ := ulp β fexp 0 / 2
  have hAcc : abs (NF.toReal (0 : R) - (0 : ℝ)) ≤ initEps := by
    have hnonneg : 0 ≤ initEps := div_nonneg (ulp.nonneg β fexp 0) (by norm_num)
    simpa [initEps] using hnonneg
  have h :=
    NF.approx_sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := xS) (xR := xR) (accS := (0 : ℝ)) (st := ((0 : R), initEps)) (epsElem := eps) hAcc hx
  have hfst :
      (NF.sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps ((0 : R), initEps) xR).1 =
        Tensor.sum (α := R) (s := s) xR := by
    simpa [Tensor.sum_eq_foldl] using
      (NF.sumFoldState_fst_eq (β := β) (fexp := fexp) (rnd := rnd) (s := s) (epsElem := eps)
        (st := ((0 : R), initEps)) (tR := xR))
  refine (approxTensor_scalar_iff (α := R) (toSpec := NF.toReal)
      (x := Tensor.sum (α := ℝ) (s := s) xS) (xR := Tensor.sum (α := R) (s := s) xR)
      (eps := NF.sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)).2 ?_
  have h' :
      abs
          (NF.toReal (Tensor.sum (α := R) (s := s) xR) -
            Tensor.sum (α := ℝ) (s := s) xS) ≤
        (NF.sumFoldState (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) eps ((0 : R), initEps) xR).2 := by
    simpa [hfst, Tensor.sum_eq_foldl] using h
  simpa [NF.sumBound, NF.sumFoldState, initEps] using h'

end

end TorchLean
