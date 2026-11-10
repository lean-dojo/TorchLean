/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Affine

/-!
# Directed backward coefficients

The state of a backward CROWN sweep at a rounded scalar is an interval coefficient per node and a
two-sided constant, both accumulated with outward rounding (`Directed.State`). The file gives the
coefficient rules of a linear map and a concatenation, the discharge of a coefficient against an
interval box, and the selection of the final affine forms in the input (`Directed.inputAffines`).
Nothing here depends on a graph representation.

The sweep follows the backward mode of auto_LiRPA (Sec. 3.1, Algorithm 2), whose soundness is
their Theorem 1. Their coefficient matrices are exact. Ours are intervals, because each product
and sum rounds outward, and the step that turns one into a slope and a constant
(`Directed.coeffAffine`) is our own, derived by hand. For an input in `[l, u]` with `l < 0 < u`
and a coefficient in `[a₋, a₊]`, the lower form keeps `a₋` and adds `(a₊ - a₋)·l`, the upper keeps
`a₊` and adds `(a₋ - a₊)·l`: at `[-1, 1]` and `[1, 2]` that gives `x - 1` and `2·x + 1`.
`Directed.coeffAffine_encloses` proves both at a lawful scalar. The other rules have
`Directed.dotBox_encloses`, `Directed.linear_encloses`, `Directed.consumeBox_encloses` and
`Directed.inputAffines_encloses`.

References:
- auto_LiRPA: Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond", NeurIPS 2020, arXiv:2002.12920, Sec. 3.1 (Algorithm 2 and Theorem 1).
-/

public section

namespace TorchLean.CROWN

open Tensor

variable {α : Type} [Storage α] [Context α]
variable [BoundOps α]

/--
Which end of the objective enclosure a backward sweep is computing.

Every relaxation choice below branches on this, because a lower bound wants the pessimistic
endpoint of each local relaxation and an upper bound the optimistic one.
-/
inductive Graph.BackwardDir where
  /-- Compute a lower bound on the objective. -/
  | lower
  /-- Compute an upper bound on the objective. -/
  | upper

open Graph

/--
Discharge an active objective against a node's IBP box.

Coordinatewise, the endpoint of the box that extremizes `aY i * y i` is selected from the sign of
the coefficient, and the products are accumulated with directed rounding. The result is therefore
sound even though the objective is not pushed any further back.
-/
@[expose] def Directed.consumeBox
    (dir : BackwardDir) (aY : FlatTensor α) (B : FlatBox α) :
    Option α :=
  if h : aY.n = B.dim then
    let aYv : Tensor α [B.dim] :=
      castDimScalar (α := α) (n := aY.n) (n' := B.dim) h aY.v
    let product (i : Fin B.dim) : α :=
      let ay := aYv.getScalar i
      let l := B.lo.getScalar i
      let u := B.hi.getScalar i
      let y :=
        if decide (ay > 0) then
          match dir with
          | .upper => u
          | .lower => l
        else
          match dir with
          | .upper => l
          | .lower => u
      match dir with
      | .lower => BoundOps.mulDown ay y
      | .upper => BoundOps.mulUp ay y
    some <| Fin.foldl B.dim (fun acc i =>
      (match dir with
      | .lower => BoundOps.addDown
      | .upper => BoundOps.addUp) acc (product i)) 0
  else
    none


/-- Regard a scalar objective enclosure as an affine form with zero input coefficients. -/
@[expose] def Directed.constantAffine
    (inputDim : Nat) (c : α) : Linear α inputDim 1 :=
  { weights := Tensor.full (α := α) (.dim 1 (.dim inputDim .scalar)) 0
    bias := Tensor.dim (fun _ => Tensor.scalar c) }

/-- Bound an objective directly from the output box, without affine reassociation. -/
@[expose] def Directed.outputBoxBound
    (dir : BackwardDir) (ibp : Array (Option (FlatBox α)))
    (outputId inputDim : Nat) (obj : FlatTensor α) : Option (Linear α inputDim 1) := do
  let some outputBox := ibp[outputId]?
    | none
  let outputBox ← outputBox
  let bound ← consumeBox (α := α) dir obj outputBox
  pure <| constantAffine (α := α) inputDim bound

/-!
## Directed affine propagation

An interval coefficient `[a₋, a₊]` encloses the mathematically exact backward coefficient, which
the verifier can only compute with outward rounding. Linear and structural nodes propagate these
interval coefficients. At a node without a directed affine rule, the active objective is
discharged against that node's directed IBP box.

The final conversion chooses one endpoint of each coefficient interval according to the sign of
the corresponding input interval. When an input interval crosses zero, a directed constant
correction accounts for the coefficient endpoint that is not selected.
-/

/--
Accumulator for the directed sweep: an interval coefficient per node and a two-sided constant.
-/
structure Directed.State (α : Type) [Storage α] [Context α] where
  /-- Interval coefficients per node, with `none` at nodes the objective has not reached yet. -/
  coeffs : Array (Option (FlatBox α))
  /-- Lower end of the accumulated constant term. -/
  cstLo : α
  /-- Upper end of the accumulated constant term. -/
  cstHi : α
  /-- Set once an active objective could not be propagated safely. -/
  failed : Bool := false

/-- Mark the directed sweep as failed. -/
@[expose] def Directed.State.fail
    (st : State α) : State α :=
  { st with failed := true }

/-- Accumulate an interval coefficient into node `pid`, adding outwards on both ends. A node out of
range, or a dimension mismatch with the coefficient already stored there, marks the sweep as
failed. -/
@[expose] def Directed.addCoeff
    (st : State α) (pid : Nat) (v : FlatBox α) :
    State α :=
  if pid < st.coeffs.size then
    match st.coeffs[pid]! with
    | none => { st with coeffs := st.coeffs.set! pid (some v) }
    | some w =>
        if h : w.dim = v.dim then
          let vlo : Tensor α [w.dim] :=
            castDimScalar (α := α) (n := v.dim) (n' := w.dim) h.symm v.lo
          let vhi : Tensor α [w.dim] :=
            castDimScalar (α := α) (n := v.dim) (n' := w.dim) h.symm v.hi
          let sum : FlatBox α :=
            { dim := w.dim
              lo := Tensor.zipWith BoundOps.addDown w.lo vlo
              hi := Tensor.zipWith BoundOps.addUp w.hi vhi }
          { st with coeffs := st.coeffs.set! pid (some sum) }
        else
          st.fail
  else
    st.fail

/-- Outward-rounded interval inner product, or `none` on a length mismatch. -/
@[expose] def Directed.dotBox (a b : FlatBox α) : Option (α × α) :=
  if h : a.dim = b.dim then
    let bLo : Tensor α [a.dim] :=
      castDimScalar (α := α) (n := b.dim) (n' := a.dim) h.symm b.lo
    let bHi : Tensor α [a.dim] :=
      castDimScalar (α := α) (n := b.dim) (n' := a.dim) h.symm b.hi
    let term (i : Fin a.dim) :=
      intervalMul (α := α)
        (getAtOrZero a.lo [i.val]) (getAtOrZero a.hi [i.val])
        (getAtOrZero bLo [i.val]) (getAtOrZero bHi [i.val])
    some <| Fin.foldl a.dim (fun acc i =>
      let p := term i
      (BoundOps.addDown acc.1 p.1, BoundOps.addUp acc.2 p.2)) (0, 0)
  else
    none

/-- Accumulate a two-sided constant contribution, rounding each end outwards. -/
@[expose] def Directed.addConstant
    (st : State α) (cLo cHi : α) : State α :=
  { st with
    cstLo := BoundOps.addDown st.cstLo cLo
    cstHi := BoundOps.addUp st.cstHi cHi }

/-- Discharge an active interval objective against a node's directed IBP box. -/
@[expose] def Directed.consume
    (st : State α) (aY By : FlatBox α) : State α :=
  match dotBox (α := α) aY By with
  | some (lo, hi) => addConstant (α := α) st lo hi
  | none => st.fail

/--
Push an interval objective back through `y = W x + b`.

The coefficient for input `j` is the interval sum of the products `aY i * W i j`, and the bias
contributes a two-sided constant.
-/
@[expose] def Directed.linear {m n : Nat}
    (aY : FlatBox α) (W : Tensor α [m, n])
    (b : Tensor α [m]) : Option (FlatBox α × (α × α)) :=
  if h : aY.dim = m then
    let aLo : Tensor α [m] :=
      castDimScalar (α := α) (n := aY.dim) (n' := m) h aY.lo
    let aHi : Tensor α [m] :=
      castDimScalar (α := α) (n := aY.dim) (n' := m) h aY.hi
    let coeffAt (j : Fin n) : α × α :=
      (List.finRange m).foldl (fun acc i =>
        let w := getAtOrZero W [i.val, j.val]
        let p := intervalMul (α := α)
          (getAtOrZero aLo [i.val]) (getAtOrZero aHi [i.val]) w w
        (BoundOps.addDown acc.1 p.1, BoundOps.addUp acc.2 p.2))
        (0, 0)
    let aX : FlatBox α :=
      { dim := n
        lo := Tensor.dim (fun j => Tensor.scalar (coeffAt j).1)
        hi := Tensor.dim (fun j => Tensor.scalar (coeffAt j).2) }
    let bBox : FlatBox α := { dim := m, lo := b, hi := b }
    dotBox (α := α) { dim := m, lo := aLo, hi := aHi } bBox |>.map fun c => (aX, c)
  else
    none

/-- Split interval coefficients into parent occurrences using the concat coordinate map. -/
@[expose] def Directed.splitCoeff (layout : ConcatLayout) (aY : FlatBox α) :
    Option (Array (FlatBox α)) :=
  if h : aY.dim = layout.outputShape.size then
    let lo := castDimScalar h aY.lo
    let hi := castDimScalar h aY.hi
    some <| (Array.finRange layout.lengths.length).map fun parent =>
      { dim := (layout.parentShape parent).size
        lo := layout.split lo parent
        hi := layout.split hi parent }
  else
    none


/-- Select affine endpoints for one interval coefficient, including the crossing-zero correction. -/
@[expose] def Directed.coeffAffine (l u al au : α) : (α × α) × (α × α) :=
  if decide (¬ l < 0) then
    ((al, 0), (au, 0))
  else if decide (¬ 0 < u) then
    ((au, 0), (al, 0))
  else
    let lowerCorrection :=
      BoundOps.mulDown (BoundOps.subUp au al) l
    let upperCorrection :=
      BoundOps.mulUp (BoundOps.subDown al au) l
    ((al, lowerCorrection), (au, upperCorrection))

/--
Turn the interval coefficients that reached the input into a lower and an upper affine form.

Each coefficient endpoint is selected from the sign of the corresponding input interval. When that
interval straddles zero, no single endpoint is valid throughout, and the one that was not selected
becomes a directed constant correction.
-/
@[expose] def Directed.inputAffines
    (inputDim : Nat) (xB aIn : FlatBox α) (cLo cHi : α) :
    Option (Linear α inputDim 1 × Linear α inputDim 1) :=
  if hx : xB.dim = inputDim then
    if ha : aIn.dim = inputDim then
      let xLo : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := xB.dim) (n' := inputDim) hx xB.lo
      let xHi : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := xB.dim) (n' := inputDim) hx xB.hi
      let aLo : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := aIn.dim) (n' := inputDim) ha aIn.lo
      let aHi : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := aIn.dim) (n' := inputDim) ha aIn.hi
      let selected (i : Fin inputDim) : (α × α) × (α × α) :=
        let l := getAtOrZero xLo [i.val]
        let u := getAtOrZero xHi [i.val]
        let al := getAtOrZero aLo [i.val]
        let au := getAtOrZero aHi [i.val]
        coeffAffine l u al au
      let lowerRow : Tensor α [inputDim] :=
        Tensor.dim (fun i => Tensor.scalar (selected i).1.1)
      let upperRow : Tensor α [inputDim] :=
        Tensor.dim (fun i => Tensor.scalar (selected i).2.1)
      let lowerCorrection := (List.finRange inputDim).foldl
        (fun acc i => BoundOps.addDown acc (selected i).1.2) 0
      let upperCorrection := (List.finRange inputDim).foldl
        (fun acc i => BoundOps.addUp acc (selected i).2.2) 0
      let lowerConstant := BoundOps.addDown cLo lowerCorrection
      let upperConstant := BoundOps.addUp cHi upperCorrection
      let lower : Linear α inputDim 1 :=
        { weights := Tensor.dim (fun _ => lowerRow)
          bias := Tensor.dim (fun _ => Tensor.scalar lowerConstant) }
      let upper : Linear α inputDim 1 :=
        { weights := Tensor.dim (fun _ => upperRow)
          bias := Tensor.dim (fun _ => Tensor.scalar upperConstant) }
      some (lower, upper)
    else
      none
  else
    none

end TorchLean.CROWN
