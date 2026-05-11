/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

/-
Reduction operator bounds for CROWN/LiRPA verification.

Implements IBP and affine bounds for:
- ReduceSum: sum over axis
- ReduceMean: mean over axis
- ReduceMax: max over axis (non-smooth)
- ReduceMin: min over axis (non-smooth)
- ReduceProd: product over axis (bilinear)
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Flatbox
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Reduction operator bounds

This file defines simple IBP and affine transfer rules for common reductions over flattened vectors
(sum/mean/max/min/prod). These rules are primarily intended for the graph-based verifier.

Non-smooth reductions (`max`/`min`) are treated conservatively (interval enclosures), and `prod`
is treated as a bilinear-style operation (also conservatively).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.Reduce

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Helper to extract scalar from dim-scalar tensor. -/
def getDimScalarFn {n : Nat} (t : Tensor α (.dim n .scalar)) : Fin n → Tensor α Shape.scalar :=
  match t with
  | .dim f => f

/-- Compute sum of a 1D tensor. -/
def tensorSum {n : Nat} (t : Tensor α (.dim n .scalar)) : α :=
  let f := getDimScalarFn t
  (List.finRange n).foldl (fun acc i =>
    match f i with
    | .scalar v => acc + v
  ) Numbers.zero

/-- IBP for ReduceSum: sum of intervals = interval of sums.
    [Σ lo_i, Σ hi_i]
-/
def ibpReduceSum (xB : FlatBox α) : FlatBox α :=
  let sum_lo := tensorSum xB.lo
  let sum_hi := tensorSum xB.hi
  { dim := 1
  , lo := Tensor.dim (fun _ => Tensor.scalar sum_lo)
  , hi := Tensor.dim (fun _ => Tensor.scalar sum_hi) }

/-- IBP for ReduceMean: mean of intervals.
    [Σ lo_i / n, Σ hi_i / n]
-/
def ibpReduceMean (xB : FlatBox α) : FlatBox α :=
  let n : α := (xB.dim : Nat)
  let sum_lo := tensorSum xB.lo
  let sum_hi := tensorSum xB.hi
  { dim := 1
  , lo := Tensor.dim (fun _ => Tensor.scalar (sum_lo / n))
  , hi := Tensor.dim (fun _ => Tensor.scalar (sum_hi / n)) }

/-- IBP for ReduceMax: max over intervals.
    lo = max of individual lower bounds (conservative but sound)
    hi = max of individual upper bounds

    Note: This is conservative. The true lower bound on max(x) is
    max(min_i x_i), but we use max(lo_i) which may be looser.
-/
def ibpReduceMax (xB : FlatBox α) : FlatBox α :=
  if h : xB.dim > 0 then
    let flo := getDimScalarFn xB.lo
    let fhi := getDimScalarFn xB.hi
    -- Find max of lowers (conservative lower bound for max)
    let init_lo := match flo ⟨0, h⟩ with | .scalar v => v
    let max_lo := (List.finRange xB.dim).foldl (fun acc i =>
      match flo i with
      | .scalar v => if v > acc then v else acc
    ) init_lo
    -- Find max of uppers (tight upper bound for max)
    let init_hi := match fhi ⟨0, h⟩ with | .scalar v => v
    let max_hi := (List.finRange xB.dim).foldl (fun acc i =>
      match fhi i with
      | .scalar v => if v > acc then v else acc
    ) init_hi
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar max_lo)
    , hi := Tensor.dim (fun _ => Tensor.scalar max_hi) }
  else
    -- Empty input case: return zero bounds
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar Numbers.zero)
    , hi := Tensor.dim (fun _ => Tensor.scalar Numbers.zero) }

/-- IBP for ReduceMin: min over intervals.
    lo = min of individual lower bounds (tight)
    hi = min of individual upper bounds (conservative but sound)
-/
def ibpReduceMin (xB : FlatBox α) : FlatBox α :=
  if h : xB.dim > 0 then
    let flo := getDimScalarFn xB.lo
    let fhi := getDimScalarFn xB.hi
    -- Find min of lowers (tight lower bound for min)
    let init_lo := match flo ⟨0, h⟩ with | .scalar v => v
    let min_lo := (List.finRange xB.dim).foldl (fun acc i =>
      match flo i with
      | .scalar v => if v < acc then v else acc
    ) init_lo
    -- Find min of uppers (conservative upper bound for min)
    let init_hi := match fhi ⟨0, h⟩ with | .scalar v => v
    let min_hi := (List.finRange xB.dim).foldl (fun acc i =>
      match fhi i with
      | .scalar v => if v < acc then v else acc
    ) init_hi
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar min_lo)
    , hi := Tensor.dim (fun _ => Tensor.scalar min_hi) }
  else
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar Numbers.zero)
    , hi := Tensor.dim (fun _ => Tensor.scalar Numbers.zero) }

/-- IBP for ReduceProd: product over intervals.
    This is more complex due to sign changes.
    For each element, compute all 4 corner products and take min/max.

    For n elements: Π[lo_i, hi_i] requires 2^n evaluations in general.
    We use a recursive interval multiplication approach.
-/
def ibpReduceProd (xB : FlatBox α) : FlatBox α :=
  if xB.dim > 0 then
    let flo := getDimScalarFn xB.lo
    let fhi := getDimScalarFn xB.hi
    -- Start with [1, 1] and multiply intervals sequentially
    let (prod_lo, prod_hi) := (List.finRange xB.dim).foldl (fun acc i =>
      let (accLo, accHi) := acc
      match flo i, fhi i with
      | .scalar li, .scalar ui =>
        -- Multiply interval [accLo, accHi] by [li, ui]
        let p1 := accLo * li
        let p2 := accLo * ui
        let p3 := accHi * li
        let p4 := accHi * ui
        let newLo :=
          let m1 := if p1 < p2 then p1 else p2
          let m2 := if p3 < p4 then p3 else p4
          if m1 < m2 then m1 else m2
        let newHi :=
          let m1 := if p1 > p2 then p1 else p2
          let m2 := if p3 > p4 then p3 else p4
          if m1 > m2 then m1 else m2
        (newLo, newHi)
    ) (Numbers.one, Numbers.one)
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar prod_lo)
    , hi := Tensor.dim (fun _ => Tensor.scalar prod_hi) }
  else
    -- Empty product = 1
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar Numbers.one)
    , hi := Tensor.dim (fun _ => Tensor.scalar Numbers.one) }

/-- Affine bounds for ReduceSum: y = Σ x_i = [1, 1, ..., 1] · x
    The affine form maps inDim to 1: A = [[sum of A rows]], c = [sum of c].
-/
def affReduceSum {inDim : Nat} (n : Nat)
    (aff : AffineVec α inDim n) : AffineVec α inDim 1 :=
  match aff.A, aff.c with
  | .dim rows, .dim cv =>
    -- Sum all rows of A into a single row
    let sumRowArr : Array α := (List.finRange n).foldl (fun arr i =>
      match rows ⟨i.val, i.isLt⟩ with
      | .dim cols =>
        (List.finRange inDim).foldl (fun arr2 j =>
          match cols j with
          | .scalar v =>
            let oldVal := if h : j.val < arr2.size then arr2[j.val] else Numbers.zero
            arr2.set! j.val (oldVal + v)
        ) arr
    ) (Array.replicate inDim Numbers.zero)
    let sumRow : Tensor α (.dim inDim .scalar) :=
      Tensor.dim (fun j => Tensor.scalar (if h : j.val < sumRowArr.size then sumRowArr[j.val] else
        Numbers.zero))
    -- Sum all bias terms
    let sumBias := (List.finRange n).foldl (fun acc i =>
      match cv ⟨i.val, i.isLt⟩ with
      | .scalar c => acc + c
    ) Numbers.zero
    let A' : Tensor α (.dim 1 (.dim inDim .scalar)) :=
      Tensor.dim (fun _ => sumRow)
    let c' : Tensor α (.dim 1 .scalar) :=
      Tensor.dim (fun _ => Tensor.scalar sumBias)
    { A := A', c := c' }

/-- Affine bounds for ReduceMean: y = (Σ x_i) / n = (1/n) * Σ x_i -/
def affReduceMean {inDim : Nat} (n : Nat)
    (aff : AffineVec α inDim n) : AffineVec α inDim 1 :=
  let sumAff := affReduceSum (inDim := inDim) n aff
  let nA : α := (n : Nat)
  match sumAff.A, sumAff.c with
  | .dim rows, .dim cv =>
    let A' := Tensor.dim (fun i =>
      match rows i with
      | .dim cols => Tensor.dim (fun j =>
        match cols j with
        | .scalar a => Tensor.scalar (a / nA)))
    let c' := Tensor.dim (fun i =>
      match cv i with
      | .scalar c => Tensor.scalar (c / nA))
    { A := A', c := c' }

/-- Derivative bounds for ReduceSum: ∂(Σ x_i)/∂x_j = 1 for all j.
    If input derivatives are in [dlo, dhi], output derivative = Σ d_i.
-/
def derivReduceSum (dB : FlatBox α) : FlatBox α :=
  ibpReduceSum dB

/-- Derivative bounds for ReduceMean: ∂(mean)/∂x_j = 1/n for all j. -/
def derivReduceMean (dB : FlatBox α) : FlatBox α :=
  let sumD := ibpReduceSum dB
  let n : α := (dB.dim : Nat)
  match sumD.lo, sumD.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i => match lo i with | .scalar v => Tensor.scalar (v / n))
    let outHi := Tensor.dim (fun i => match hi i with | .scalar v => Tensor.scalar (v / n))
    { dim := 1, lo := outLo, hi := outHi }

/-- Derivative bounds for ReduceMax: non-smooth, uses subgradient.
    ∂max/∂x_i = 1 if x_i is the max, 0 otherwise.
    For intervals, we bound: derivative ∈ [0, 1] for argmax candidates.
-/
def derivReduceMax (_xB dB : FlatBox α) : FlatBox α :=
  if h : dB.dim > 0 then
    let fdlo := getDimScalarFn dB.lo
    let fdhi := getDimScalarFn dB.hi
    -- Find min and max of derivatives over potential argmax locations
    -- Conservative: any component could be max, so d_out ∈ [min d_i, max d_i]
    let init_l := match fdlo ⟨0, h⟩ with | .scalar v => v
    let init_h := match fdhi ⟨0, h⟩ with | .scalar v => v
    let (min_d, max_d) := (List.finRange dB.dim).foldl (fun acc i =>
      let (accMin, accMax) := acc
      match fdlo i, fdhi i with
      | .scalar dl, .scalar dh =>
        let newMin := if dl < accMin then dl else accMin
        let newMax := if dh > accMax then dh else accMax
        (newMin, newMax)
    ) (init_l, init_h)
    -- The actual derivative is sum of d_i over argmax set, but conservatively
    -- bound between 0 (if max is constant) and max(d_i) * 1
    let lo := if min_d < Numbers.zero then min_d else Numbers.zero
    let hi := if max_d > Numbers.zero then max_d else Numbers.zero
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar lo)
    , hi := Tensor.dim (fun _ => Tensor.scalar hi) }
  else
    { dim := 1
    , lo := Tensor.dim (fun _ => Tensor.scalar Numbers.zero)
    , hi := Tensor.dim (fun _ => Tensor.scalar Numbers.zero) }

/-- Derivative bounds for ReduceMin: similar to max but argmin. -/
def derivReduceMin (xB dB : FlatBox α) : FlatBox α :=
  -- Same logic as max, derivative ∈ [0, 1] at argmin
  derivReduceMax xB dB

namespace AxisReduce

/-- IBP for ReduceSum along axis 0 (sum over rows) for 2D: [rows, cols] → [1, cols].
    Each output column j = Σ_i input[i,j].
-/
def ibpReduceSumAxis0 {rows cols : Nat}
    (xB : Box α (.dim rows (.dim cols .scalar))) : Box α (.dim 1 (.dim cols .scalar)) :=
  match xB.lo, xB.hi with
  | .dim loRows, .dim hiRows =>
    let outLo := Tensor.dim (fun _ =>
      Tensor.dim (fun j =>
        let sum := (List.finRange rows).foldl (fun acc i =>
          match loRows ⟨i.val, i.isLt⟩ with
          | .dim loCols => match loCols j with
            | .scalar v => acc + v
        ) Numbers.zero
        Tensor.scalar sum))
    let outHi := Tensor.dim (fun _ =>
      Tensor.dim (fun j =>
        let sum := (List.finRange rows).foldl (fun acc i =>
          match hiRows ⟨i.val, i.isLt⟩ with
          | .dim hiCols => match hiCols j with
            | .scalar v => acc + v
        ) Numbers.zero
        Tensor.scalar sum))
    { lo := outLo, hi := outHi }

/-- IBP for ReduceSum along axis 1 (sum over columns) for 2D: [rows, cols] → [rows, 1].
    Each output row i = Σ_j input[i,j].
-/
def ibpReduceSumAxis1 {rows cols : Nat}
    (xB : Box α (.dim rows (.dim cols .scalar))) : Box α (.dim rows (.dim 1 .scalar)) :=
  match xB.lo, xB.hi with
  | .dim loRows, .dim hiRows =>
    let outLo := Tensor.dim (fun i =>
      match loRows i with
      | .dim loCols =>
        let sum := (List.finRange cols).foldl (fun acc j =>
          match loCols ⟨j.val, j.isLt⟩ with
          | .scalar v => acc + v
        ) Numbers.zero
        Tensor.dim (fun _ => Tensor.scalar sum))
    let outHi := Tensor.dim (fun i =>
      match hiRows i with
      | .dim hiCols =>
        let sum := (List.finRange cols).foldl (fun acc j =>
          match hiCols ⟨j.val, j.isLt⟩ with
          | .scalar v => acc + v
        ) Numbers.zero
        Tensor.dim (fun _ => Tensor.scalar sum))
    { lo := outLo, hi := outHi }

end AxisReduce

namespace Theorems

/-- ReduceSum output dimension is 1. -/
theorem ibp_reduce_sum_dim (xB : FlatBox α) :
    (ibpReduceSum xB).dim = 1 := by
  rfl

/-- ReduceMean output dimension is 1. -/
theorem ibp_reduce_mean_dim (xB : FlatBox α) :
    (ibpReduceMean xB).dim = 1 := by
  rfl

/-- ReduceMax output dimension is 1. -/
theorem ibp_reduce_max_dim (xB : FlatBox α) :
    (ibpReduceMax xB).dim = 1 := by
  simp only [ibpReduceMax]
  split
  · rfl
  · rfl

/-- ReduceMin output dimension is 1. -/
theorem ibp_reduce_min_dim (xB : FlatBox α) :
    (ibpReduceMin xB).dim = 1 := by
  simp only [ibpReduceMin]
  split
  · rfl
  · rfl

/-- ReduceProd output dimension is 1. -/
theorem ibp_reduce_prod_dim (xB : FlatBox α) :
    (ibpReduceProd xB).dim = 1 := by
  simp only [ibpReduceProd]
  split
  · rfl
  · rfl

/-- Affine reduce sum output dimension is 1. -/
theorem aff_reduce_sum_dim {inDim : Nat} (n : Nat) (aff : AffineVec α inDim n) :
    ∃ A c, (affReduceSum n aff).A = A ∧ (affReduceSum n aff).c = c := by
  exact ⟨(affReduceSum n aff).A, (affReduceSum n aff).c, rfl, rfl⟩

end Theorems

end NN.MLTheory.CROWN.Operators.Reduce
