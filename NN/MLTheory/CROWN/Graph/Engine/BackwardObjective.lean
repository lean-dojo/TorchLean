/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Activations

/-!
# Objective-Dependent Backward CROWN

Forward CROWN gives nodewise bounds. This module handles the complementary use case: start from a
linear objective on an output node and propagate that objective backward through the graph, choosing
local relaxations from the sign of the downstream coefficients.

For exact scalar backends, the coefficient transformations are ordinary algebraic identities. For
rounded scalar backends, TorchLean carries intervals for the coefficients and evaluates every
coefficient product and sum outwards through `BoundOps`. The `DirectedBackward` proof modules
establish enclosure for the reverse sweep, its public fallback, and final interval evaluation,
under lawful directed arithmetic, real stored-parameter graph equations, and enclosing IBP boxes.
Relating a separate host runtime's evaluation order to this real graph requires an additional
runtime-approximation theorem.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]

open BoundOps
open NN.MLTheory.CROWN.Graph.Internal

/-!
The backward pass covers the same verifier dialect as `runCROWN` where objective-dependent
relaxations are available. Unsupported nodes consume already-computed IBP boxes conservatively.
-/

/--
Which end of the objective enclosure a backward sweep is computing.

Every relaxation choice below branches on this, because a lower bound wants the pessimistic
endpoint of each local relaxation and an upper bound the optimistic one.
-/
inductive Internal.BackwardDir where
  /-- Compute a lower bound on the objective. -/
  | lower
  /-- Compute an upper bound on the objective. -/
  | upper

/--
Accumulator for one backward sweep: the objective coefficients still owed to each node, together
with the constant that has already been split off.
-/
private structure BackwardState (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Objective coefficients per node; `none` for nodes the objective has not reached yet. -/
  coeffs : Array (Option (FlatTensor α))
  /-- Constant accumulated from biases and from objectives discharged against IBP boxes. -/
  cst    : α
  /-- Set once an active objective could not be propagated safely, which voids the sweep. -/
  failed : Bool := false

/-- Mark the sweep as failed; the entry points turn a failed state into `none`. -/
private def BackwardState.fail (st : BackwardState α) : BackwardState α :=
  { st with failed := true }

/-- Add two flat coefficient vectors, or fail when their lengths disagree. -/
private def flatTensorAdd (a b : FlatTensor α) : Option (FlatTensor α) :=
  if h : a.n = b.n then
    let bv : Tensor α [a.n] :=
      castDimScalar (α := α) (n := b.n) (n' := a.n) h.symm b.v
    some { n := a.n, v := Tensor.addSpec a.v bv }
  else
    none

/-- Scale every entry of a flat coefficient vector. -/
private def flatTensorScale (k : α) (v : FlatTensor α) : FlatTensor α :=
  { n := v.n, v := Tensor.scaleSpec v.v k }

/-- Accumulate a coefficient contribution into node `pid`, failing on a length mismatch. -/
private def addCoeff (st : BackwardState α) (pid : Nat) (v : FlatTensor α) : BackwardState α :=
  match st.coeffs[pid]! with
  | none => { st with coeffs := st.coeffs.set! pid (some v) }
  | some w =>
    match flatTensorAdd (α:=α) w v with
    | some s => { st with coeffs := st.coeffs.set! pid (some s) }
    | none   => st.fail

/-- Inner product of two flat vectors of equal length. -/
private def dotFlat {n : Nat} (a b : Tensor α [n]) : α :=
  TorchLean.Tensor.sumSpec (Tensor.mulSpec a b)

/--
Discharge an active objective against a node's IBP box.

Coordinatewise, the endpoint of the box that extremizes `aY i * y i` is selected from the sign of
the coefficient, and the products are accumulated with directed rounding. The result is therefore
sound even though the objective is not pushed any further back.
-/
@[expose] def Internal.consumeObjectiveFromBox
    (dir : BackwardDir) (aY : FlatTensor α) (B : FlatBox α) :
    Option α :=
  if h : aY.n = B.dim then
    let aYv : Tensor α [B.dim] :=
      castDimScalar (α := α) (n := aY.n) (n' := B.dim) h aY.v
    let fa := getDimScalarFn (α := α) aYv
    let flo := getDimScalarFn (α := α) B.lo
    let fhi := getDimScalarFn (α := α) B.hi
    let products : Array α :=
      (Array.finRange B.dim).map fun i =>
        let ay := (fa i).item
        let l := (flo i).item
        let u := (fhi i).item
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
    some <| products.foldl
      (match dir with
      | .lower => BoundOps.addDown
      | .upper => BoundOps.addUp)
      0
  else
    none

/-- Add a constant contribution while preserving the direction of the requested enclosure. -/
private def addConstant (dir : BackwardDir) (st : BackwardState α) (c : α) : BackwardState α :=
  { st with cst :=
      match dir with
      | .lower => BoundOps.addDown st.cst c
      | .upper => BoundOps.addUp st.cst c }

/-- Regard a scalar objective enclosure as an affine form with zero input coefficients. -/
@[expose] def Internal.constantObjectiveAffine
    (inputDim : Nat) (c : α) : AffineVec α inputDim 1 :=
  { A := Tensor.full (α := α) (.dim 1 (.dim inputDim .scalar)) 0
    c := Tensor.dim (fun _ => Tensor.scalar c) }

/-- Bound an objective directly from the output box, without affine reassociation. -/
@[expose] def Internal.objectiveFromOutputBox
    (dir : BackwardDir) (ibp : Array (Option (FlatBox α)))
    (outputId inputDim : Nat) (obj : FlatTensor α) : Option (AffineVec α inputDim 1) := do
  let some outputBox := ibp[outputId]?
    | none
  let outputBox ← outputBox
  let bound ← consumeObjectiveFromBox (α := α) dir obj outputBox
  pure <| constantObjectiveAffine (α := α) inputDim bound

/-!
## Directed affine propagation

An interval coefficient `[a₋, a₊]` records all values that a mathematically exact backward
coefficient may take after the verifier has evaluated its arithmetic with outward rounding. Linear
and structural nodes preserve these coefficients. At a node without a directed affine rule, the
active objective is discharged against that node's directed IBP box.

The final conversion chooses one endpoint of each coefficient interval according to the sign of
the corresponding input interval. When an input interval crosses zero, a directed constant
correction accounts for the coefficient endpoint that was not selected.
-/

/--
Accumulator for the directed sweep: an interval coefficient per node and a two-sided constant.
-/
structure Internal.DirectedBackwardState (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Interval coefficients per node; `none` for nodes the objective has not reached yet. -/
  coeffs : Array (Option (FlatBox α))
  /-- Lower end of the accumulated constant term. -/
  cstLo : α
  /-- Upper end of the accumulated constant term. -/
  cstHi : α
  /-- Set once an active objective could not be propagated safely. -/
  failed : Bool := false

/-- Mark the directed sweep as failed. -/
@[expose] def Internal.DirectedBackwardState.fail
    (st : DirectedBackwardState α) : DirectedBackwardState α :=
  { st with failed := true }

/-- View an exact coefficient vector as a degenerate interval coefficient. -/
@[expose] def Internal.pointCoeffBox (v : FlatTensor α) : FlatBox α :=
  { dim := v.n, lo := v.v, hi := v.v }

/-- Outward-rounded interval product: all four endpoint products, then the min and the max. -/
@[expose] def Internal.directedIntervalMul (aLo aHi bLo bHi : α) : α × α :=
  let p1Lo := BoundOps.mulDown aLo bLo
  let p2Lo := BoundOps.mulDown aLo bHi
  let p3Lo := BoundOps.mulDown aHi bLo
  let p4Lo := BoundOps.mulDown aHi bHi
  let p1Hi := BoundOps.mulUp aLo bLo
  let p2Hi := BoundOps.mulUp aLo bHi
  let p3Hi := BoundOps.mulUp aHi bLo
  let p4Hi := BoundOps.mulUp aHi bHi
  (min2 (min2 p1Lo p2Lo) (min2 p3Lo p4Lo),
    max2 (max2 p1Hi p2Hi) (max2 p3Hi p4Hi))

/-- Accumulate an interval coefficient into node `pid`, adding outwards on both ends. -/
@[expose] def Internal.addDirectedCoeff
    (st : DirectedBackwardState α) (pid : Nat) (v : FlatBox α) :
    DirectedBackwardState α :=
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
            lo := Tensor.map2Spec BoundOps.addDown w.lo vlo
            hi := Tensor.map2Spec BoundOps.addUp w.hi vhi }
        { st with coeffs := st.coeffs.set! pid (some sum) }
      else
        st.fail

/-- Negate an interval coefficient, which swaps its two endpoints. -/
@[expose] def Internal.negateDirectedCoeff (v : FlatBox α) : FlatBox α :=
  { dim := v.dim
    lo := Tensor.mapSpec (fun x => BoundOps.subDown 0 x) v.hi
    hi := Tensor.mapSpec (fun x => BoundOps.subUp 0 x) v.lo }

/-- Outward-rounded interval inner product, or `none` on a length mismatch. -/
@[expose] def Internal.directedDotBox (a b : FlatBox α) : Option (α × α) :=
  if h : a.dim = b.dim then
    let bLo : Tensor α [a.dim] :=
      castDimScalar (α := α) (n := b.dim) (n' := a.dim) h.symm b.lo
    let bHi : Tensor α [a.dim] :=
      castDimScalar (α := α) (n := b.dim) (n' := a.dim) h.symm b.hi
    let terms := (List.finRange a.dim).map fun i =>
      directedIntervalMul (α := α)
        (getAtOrZero a.lo [i.val]) (getAtOrZero a.hi [i.val])
        (getAtOrZero bLo [i.val]) (getAtOrZero bHi [i.val])
    let lo := terms.foldl (fun acc p => BoundOps.addDown acc p.1) 0
    let hi := terms.foldl (fun acc p => BoundOps.addUp acc p.2) 0
    some (lo, hi)
  else
    none

/-- Accumulate a two-sided constant contribution, rounding each end outwards. -/
@[expose] def Internal.addDirectedConstant
    (st : DirectedBackwardState α) (cLo cHi : α) : DirectedBackwardState α :=
  { st with
    cstLo := BoundOps.addDown st.cstLo cLo
    cstHi := BoundOps.addUp st.cstHi cHi }

/-- Discharge an active interval objective against a node's directed IBP box. -/
@[expose] def Internal.consumeDirectedObjective
    (st : DirectedBackwardState α) (aY By : FlatBox α) : DirectedBackwardState α :=
  match directedDotBox (α := α) aY By with
  | some (lo, hi) => addDirectedConstant (α := α) st lo hi
  | none => st.fail

/--
Push an interval objective back through `y = W x + b`.

The coefficient for input `j` is the interval sum of the products `aY i * W i j`, and the bias
contributes a two-sided constant.
-/
@[expose] def Internal.directedBackwardLinear {m n : Nat}
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
        let p := directedIntervalMul (α := α)
          (getAtOrZero aLo [i.val]) (getAtOrZero aHi [i.val]) w w
        (BoundOps.addDown acc.1 p.1, BoundOps.addUp acc.2 p.2))
        (0, 0)
    let aX : FlatBox α :=
      { dim := n
        lo := Tensor.dim (fun j => Tensor.scalar (coeffAt j).1)
        hi := Tensor.dim (fun j => Tensor.scalar (coeffAt j).2) }
    let bBox : FlatBox α := { dim := m, lo := b, hi := b }
    directedDotBox (α := α) { dim := m, lo := aLo, hi := aHi } bBox |>.map fun c => (aX, c)
  else
    none

/-- Split interval coefficients into parent occurrences using the concat coordinate map. -/
@[expose] def Internal.splitDirectedCoeff (layout : ConcatLayout) (aY : FlatBox α) :
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

/-- Read the diagonal of a square matrix, which is all a diagonal relaxation actually stores. -/
private def diagOfMat {n : Nat} (A : Tensor α [n, n]) : Tensor α [n] :=
  Tensor.ofFn fun i => Spec.get2 A i i

/--
Apply a diagonal relaxation `y = s ⊙ x + b` to a scalar objective.

Coordinatewise, the slope and the intercept are picked from the sign of the incoming coefficient,
so the resulting affine form is the tighter of the two candidate planes in the requested direction.
-/
private def backwardApplyDiag {n : Nat}
  (dir : BackwardDir)
  (aY : Tensor α [n])
  (sLo bLo sHi bHi : Tensor α [n]) :
  (Tensor α [n] × α) :=
  let fa := getDimScalarFn (α := α) aY
  let fsLo := getDimScalarFn (α := α) sLo
  let fbLo := getDimScalarFn (α := α) bLo
  let fsHi := getDimScalarFn (α := α) sHi
  let fbHi := getDimScalarFn (α := α) bHi
  let sChosen : Tensor α [n] :=
    Tensor.dim (fun i =>
      let ay := (fa i).item
      let slo := (fsLo i).item
      let shi := (fsHi i).item
      let s :=
        if decide (ay > 0) then
          match dir with
          | .upper => shi
          | .lower => slo
        else
          match dir with
          | .upper => slo
          | .lower => shi
      Tensor.scalar s)
  let bChosen : Tensor α [n] :=
    Tensor.dim (fun i =>
      let ay := (fa i).item
      let blo := (fbLo i).item
      let bhi := (fbHi i).item
      let b :=
        if decide (ay > 0) then
          match dir with
          | .upper => bhi
          | .lower => blo
        else
          match dir with
          | .upper => blo
          | .lower => bhi
      Tensor.scalar b)
  let aX := Tensor.mulSpec aY sChosen
  let cst := dotFlat (α:=α) aY bChosen
  (aX, cst)

/--
Backward step for a unary op whose relaxation is diagonal, which covers relu, exp, log, sigmoid,
tanh, softmax, and layernorm. The stored affine bounds are cast to the pre-activation width, their
diagonals extracted, and `backwardApplyDiag` picks the plane.
-/
private def backwardUnaryDiag
  (dir : BackwardDir) (preB : FlatBox α) (localB : FlatAffineBounds α)
  (aY : FlatTensor α) : Option (FlatTensor α × α) := by
  if h : aY.n = preB.dim then
    let n := preB.dim
    if hIn : localB.inDim = n then
      if hOut : localB.outDim = n then
        let aYv : Tensor α [n] :=
          castDimScalar (α := α) (n := aY.n) (n' := n) h aY.v
        let loAffN : AffineVec α n n :=
          castAffineIn (α:=α) (n:=localB.inDim) (n':=n) (m:=n) hIn
            (castAffineOut (α:=α) (n:=localB.inDim) (m:=localB.outDim) (m':=n) hOut localB.loAff)
        let hiAffN : AffineVec α n n :=
          castAffineIn (α:=α) (n:=localB.inDim) (n':=n) (m:=n) hIn
            (castAffineOut (α:=α) (n:=localB.inDim) (m:=localB.outDim) (m':=n) hOut localB.hiAff)
        let sLo := diagOfMat (α:=α) (n:=n) loAffN.A
        let bLo := castDimScalar (α:=α) (n:=localB.outDim) (n':=n) hOut localB.loAff.c
        let sHi := diagOfMat (α:=α) (n:=n) hiAffN.A
        let bHi := castDimScalar (α:=α) (n:=localB.outDim) (n':=n) hOut localB.hiAff.c
        let (aX, cst) := backwardApplyDiag (α:=α) (n:=n) dir aYv sLo bLo sHi bHi
        exact some ({ n := n, v := aX }, cst)
      else
        exact none
    else
      exact none
  else
    exact none

/-- Row vector times matrix, the shape a linear node needs on the way back. -/
private def matLeftMul {m n : Nat}
  (aY : Tensor α [m]) (W : Tensor α [m, n]) :
  Tensor α [n] :=
  Tensor.ofFn fun j =>
    (List.finRange m).foldl
      (fun acc i => acc + Tensor.getScalar aY i * Spec.get2 W i j)
      0

/-- Push an exact objective back through `y = W x + b`. -/
private def backwardLinear {m n : Nat}
  (aY : FlatTensor α) (W : Tensor α [m, n]) (b : Tensor α [m]) :
  Option (FlatTensor α × α) :=
  if h : aY.n = m then
    let aYv : Tensor α [m] :=
      castDimScalar (α := α) (n := aY.n) (n' := m) h aY.v
    let aX := matLeftMul (α:=α) (m:=m) (n:=n) aYv W
    let cst := dotFlat (α:=α) aYv b
    some ({ n := n, v := aX }, cst)
  else
    none

/-- The right operand of a subtraction receives the negated objective. -/
private def backwardSubRight (aY : FlatTensor α) : FlatTensor α :=
  flatTensorScale (α:=α) (k := (-1)) aY

/-- Split an objective into parent occurrences using the concat coordinate map. -/
private def backwardConcatSplit (layout : ConcatLayout) (aY : FlatTensor α) :
    Option (Array (FlatTensor α)) :=
  if h : aY.n = layout.outputShape.size then
    let value := castDimScalar h aY.v
    some <| (Array.finRange layout.lengths.length).map fun parent =>
      { n := (layout.parentShape parent).size, v := layout.split value parent }
  else
    none

/-- Check parent boxes as well as graph geometry before routing a concat objective. -/
@[expose] def Internal.concatBackwardLayout? (nodes : Array Node) (ibp : Array (Option (FlatBox α)))
    (node : Node) (axis : Nat) : Option ConcatLayout := do
  let layout ← concatNodeLayout? nodes node axis
  unless node.parents.size == layout.lengths.length do failure
  let parents : Fin layout.lengths.length → FlatBox α ←
    Tensor.Internal.sequenceFinM fun parent => do
      let id ← node.parents[parent.val]?
      (ibp[id]?).join
  unless (Array.finRange layout.lengths.length).all
      (fun parent => (parents parent).dim == (layout.parentShape parent).size) do failure
  pure layout

/-- Reindex a flat vector by a permutation of its coordinates. -/
@[expose] def Internal.backwardPermuteVec {n : Nat} (perm : Fin n → Fin n) (v : Tensor α [n]) :
  Tensor α [n] :=
  Tensor.ofFn fun i => Tensor.getScalar v (perm i)

/-- Pull a flat objective back through an arbitrary valid axis permutation. -/
private def backwardAxisPermutation? (outputShape : Shape) (forwardPerm : Array Nat)
    (aY : FlatTensor α) : Option (FlatTensor α) := do
  if aY.n = 0 then
    pure aY
  else
    let inverse ← (OpContracts.inversePerm forwardPerm).toOption
    let flatPerm ← flatAxisPermutation? outputShape inverse aY.n
    pure { n := aY.n, v := backwardPermuteVec (α := α) flatPerm aY.v }

/-- Pull interval coefficients back through a permutation without scalar arithmetic. -/
@[expose] def Internal.directedAxisPermutation? (outputShape : Shape) (forwardPerm : Array Nat)
    (aY : FlatBox α) : Option (FlatBox α) := do
  if aY.dim = 0 then
    pure aY
  else
    let inverse ← (OpContracts.inversePerm forwardPerm).toOption
    let flatPerm ← flatAxisPermutation? outputShape inverse aY.dim
    pure
      { dim := aY.dim
        lo := backwardPermuteVec flatPerm aY.lo
        hi := backwardPermuteVec flatPerm aY.hi }

/--
Push an objective back through a matrix product whose operands both vary.

That is not a linear node, so there is no single coefficient matrix to transpose: each coefficient
is bounded from the operand boxes, which is why `Bx` and `By` are required here.
-/
private def backwardMatmul
  (dir : BackwardDir)
  (aZ : FlatTensor α) (Bx By : FlatBox α)
  (sA sB : Shape) :
  Option ((FlatTensor α) × (FlatTensor α) × α) :=
  match (OpContracts.matmulDims sA sB).toOption with
  | none => none
  | some dims =>
    let dimA := sA.size
    let dimB := sB.size
    let outDim := dims.outShape.size
    if aZ.n = outDim ∧ Bx.dim = dimA ∧ By.dim = dimB then
      let (aArr, bArr, cst) : Array α × Array α × α := Id.run do
        let mut aArr : Array α := Array.replicate dimA 0
        let mut bArr : Array α := Array.replicate dimB 0
        let mut cst : α := 0
        for outIdx in List.range outDim do
          let az : α := getAtOrZero aZ.v [outIdx]
          for kk in List.range dims.inner do
            let aIdx := dims.leftIndex outIdx kk
            let bIdx := dims.rightIndex outIdx kk
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)

            -- Upper plane selection.
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let axU := if u1 < u2 then ly else uy
            let ayU := if u1 < u2 then ux else lx
            let bU := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))

            -- Lower plane selection.
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let axL := if l1 > l2 then ly else uy
            let ayL := if l1 > l2 then lx else ux
            let bL := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))

            let useUpper : Bool :=
              if decide (az > 0) then
                match dir with
                | .upper => true
                | .lower => false
              else
                match dir with
                | .upper => false
                | .lower => true

            let ax := if useUpper then axU else axL
            let ay := if useUpper then ayU else ayL
            let bb := if useUpper then bU else bL

            aArr := aArr.set! aIdx (aArr[aIdx]! + az * ax)
            bArr := bArr.set! bIdx (bArr[bIdx]! + az * ay)
            cst := cst + az * bb
        return (aArr, bArr, cst)

      let aT : Tensor α [dimA] :=
        Tensor.dim (fun i => Tensor.scalar (aArr[i.val]!))
      let bT : Tensor α [dimB] :=
        Tensor.dim (fun i => Tensor.scalar (bArr[i.val]!))
      some ({ n := dimA, v := aT }, { n := dimB, v := bT }, cst)
    else
      none

/--
Push an objective back through an elementwise product whose operands both vary.

Each element uses one McCormick plane, chosen at the interval midpoint, and the slack that the
choice gives up is absorbed into the constant term.
-/
private def backwardMulElem
  (dir : BackwardDir)
  (aZ : FlatTensor α) (Bx By : FlatBox α) :
  Option ((FlatTensor α) × (FlatTensor α) × α) :=
  if h : aZ.n = Bx.dim ∧ Bx.dim = By.dim then
    let n := Bx.dim
    let hZ : aZ.n = n := h.1
    let aZv : Tensor α [n] :=
      castDimScalar (α := α) (n := aZ.n) (n' := n) hZ aZ.v
    let xLo := getDimScalarFn (α := α) Bx.lo
    let xHi := getDimScalarFn (α := α) Bx.hi
    let yLo := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.lo)
    let yHi := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.hi)
    let aF := getDimScalarFn (α := α) aZv
    -- Choose one McCormick plane per element using the interval midpoint.
    let axU : Tensor α [n] :=
      Tensor.dim (fun i =>
        let lx := (xLo i).item
        let ux := (xHi i).item
        let ly := (yLo i).item
        let uy := (yHi i).item
        let mx := (lx + ux) * (1 / 2)
        let my := (ly + uy) * (1 / 2)
        let u1 := ux * my + ly * mx - ux * ly
        let u2 := lx * my + uy * mx - lx * uy
        let ax := if u1 < u2 then ly else uy
        Tensor.scalar ax)
    let ayU : Tensor α [n] :=
      Tensor.dim (fun i =>
        let lx := (xLo i).item
        let ux := (xHi i).item
        let ly := (yLo i).item
        let uy := (yHi i).item
        let mx := (lx + ux) * (1 / 2)
        let my := (ly + uy) * (1 / 2)
        let u1 := ux * my + ly * mx - ux * ly
        let u2 := lx * my + uy * mx - lx * uy
        let ay := if u1 < u2 then ux else lx
        Tensor.scalar ay)
    let bU : Tensor α [n] :=
      Tensor.dim (fun i =>
        let lx := (xLo i).item
        let ux := (xHi i).item
        let ly := (yLo i).item
        let uy := (yHi i).item
        let mx := (lx + ux) * (1 / 2)
        let my := (ly + uy) * (1 / 2)
        let u1 := ux * my + ly * mx - ux * ly
        let u2 := lx * my + uy * mx - lx * uy
        let b := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
        Tensor.scalar b)
    let axL : Tensor α [n] :=
      Tensor.dim (fun i =>
        let lx := (xLo i).item
        let ux := (xHi i).item
        let ly := (yLo i).item
        let uy := (yHi i).item
        let mx := (lx + ux) * (1 / 2)
        let my := (ly + uy) * (1 / 2)
        let l1 := lx * my + ly * mx - lx * ly
        let l2 := ux * my + uy * mx - ux * uy
        let ax := if l1 > l2 then ly else uy
        Tensor.scalar ax)
    let ayL : Tensor α [n] :=
      Tensor.dim (fun i =>
        let lx := (xLo i).item
        let ux := (xHi i).item
        let ly := (yLo i).item
        let uy := (yHi i).item
        let mx := (lx + ux) * (1 / 2)
        let my := (ly + uy) * (1 / 2)
        let l1 := lx * my + ly * mx - lx * ly
        let l2 := ux * my + uy * mx - ux * uy
        let ay := if l1 > l2 then lx else ux
        Tensor.scalar ay)
    let bL : Tensor α [n] :=
      Tensor.dim (fun i =>
        let lx := (xLo i).item
        let ux := (xHi i).item
        let ly := (yLo i).item
        let uy := (yHi i).item
        let mx := (lx + ux) * (1 / 2)
        let my := (ly + uy) * (1 / 2)
        let l1 := lx * my + ly * mx - lx * ly
        let l2 := ux * my + uy * mx - ux * uy
        let b := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))
        Tensor.scalar b)
    let axUFn := getDimScalarFn (α := α) axU
    let ayUFn := getDimScalarFn (α := α) ayU
    let bUFn := getDimScalarFn (α := α) bU
    let axLFn := getDimScalarFn (α := α) axL
    let ayLFn := getDimScalarFn (α := α) ayL
    let bLFn := getDimScalarFn (α := α) bL
    let aX : Tensor α [n] :=
      Tensor.dim (fun i =>
        let az := (aF i).item
        let axu := (axUFn i).item
        let axl := (axLFn i).item
        let ax :=
          if decide (az > 0) then
            match dir with
            | .upper => axu
            | .lower => axl
          else
            match dir with
            | .upper => axl
            | .lower => axu
        Tensor.scalar (az * ax))
    let aY : Tensor α [n] :=
      Tensor.dim (fun i =>
        let az := (aF i).item
        let ayu := (ayUFn i).item
        let ayl := (ayLFn i).item
        let ay :=
          if decide (az > 0) then
            match dir with
            | .upper => ayu
            | .lower => ayl
          else
            match dir with
            | .upper => ayl
            | .lower => ayu
        Tensor.scalar (az * ay))
    let biasProd : Tensor α [n] :=
      Tensor.dim (fun i =>
        let az := (aF i).item
        let bu := (bUFn i).item
        let bl := (bLFn i).item
        let b :=
          if decide (az > 0) then
            match dir with
            | .upper => bu
            | .lower => bl
          else
            match dir with
            | .upper => bl
            | .lower => bu
        Tensor.scalar (az * b))
    let cst := TorchLean.Tensor.sumSpec biasProd
    some ({ n := n, v := aX }, { n := n, v := aY }, cst)
  else
    none

/--
One backward step at node `id`: consume the objective owed to it and pass contributions to its
parents. A node with no backward rule discharges the objective against its IBP box instead.
-/
private def backwardNode (dir : BackwardDir)
  (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
  (ctx : AffineCtx) (st : BackwardState α) (id : Nat) : BackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
    let node := nodes[id]!
    match node.kind with
    | .input =>
      if node.id = ctx.inputId then
        st
      else
        match ibp[id]! with
        | some Bx =>
          match consumeObjectiveFromBox (α := α) (dir := dir) aY Bx with
          | some cadd => addConstant ( α := α) dir st cadd
          | none => st.fail
        | none => st.fail
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        if h : aY.n = v.n then
          let aYv : Tensor α [v.n] :=
            castDimScalar (α := α) (n := aY.n) (n' := v.n) h aY.v
          let add := dotFlat (α:=α) aYv v.v
          addConstant (α := α) dir st add
        else st.fail
      | none => st.fail
    | .detach =>
      match node.parents with
      | #[p1] => addCoeff (α := α) st p1 aY
      | _ => st.fail
    | .add =>
      match node.parents with
      | #[p1, p2] =>
        let st1 := addCoeff (α:=α) st p1 aY
        addCoeff (α:=α) st1 p2 aY
      | _ => st.fail
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        let st1 := addCoeff (α:=α) st p1 aY
        addCoeff (α:=α) st1 p2 (backwardSubRight (α:=α) aY)
      | _ => st.fail
    | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .sin | .cos | .maxElem |
      .minElem | .hardMaskedSoftmax _
    | .maxPool .. | .avgPool ..
    | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => addConstant (α := α) dir st cadd
        | none => st.fail
      | none => st.fail
    | .batchNormEval channelAxis _ =>
      match node.parents with
      | #[p1] =>
        match ps.batchNormEval[id]? with
        | some config =>
          match batchNormEvalLinear? (α := α) nodes[p1]!.outShape channelAxis config with
          | some p =>
            match backwardLinear (α := α) (m := p.m) (n := p.n) aY p.w p.b with
            | some (aX, cadd) =>
              let st' := addCoeff (α := α) st p1 aX
              addConstant (α := α) dir st' cadd
            | none => st.fail
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | .linear =>
      match node.parents with
      | #[p1] =>
        match ps.linearWB[id]? with
        | some p =>
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w p.b with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            addConstant (α := α) dir st' cadd
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | .matmul =>
      match node.parents with
      | #[p1, p2] =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMatmul (α:=α) (dir:=dir) aY Bx By (sA := nodes[p1]!.outShape) (sB :=
            nodes[p2]!.outShape) with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            addConstant (α := α) dir st2 cadd
          | none => st.fail
        | _, _ => st.fail
      | #[p1] =>
        match ps.matmulW[id]? with
        | some p =>
          let zb := Tensor.full (α := α) (.dim p.m .scalar) 0
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w zb with
          | some (aX, _cadd) =>
            addCoeff (α:=α) st p1 aX
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | .conv configuration =>
      match node.parents with
      | #[p1] =>
        match ps.convCfg[id]?, nodes[p1]? with
        | some config, some parent =>
          match planConvTransfer? configuration config parent.outShape node.outShape with
          | some leading =>
            let convAff := affOfConv (α := α) config leading
            match backwardLinear (α := α) aY convAff.A convAff.c with
            | some (aX, cadd) =>
              let st' := addCoeff (α := α) st p1 aX
              addConstant (α := α) dir st' cadd
            | none => st.fail
          | none => st.fail
        | _, _ => st.fail
      | _ => st.fail
    | .layernorm _ =>
      if !crownNodeSemanticsSupported (α := α) nodes ps id then
        st.fail
      else
        match ibp[id]! with
        | some By =>
          match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
          | some cadd => addConstant (α := α) dir st cadd
          | none => st.fail
        | none => st.fail
    | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softplus | .safeLog | .softmax _ =>
      -- The value pass has already applied the scalar backend's directed nonlinear capabilities.
      -- The default backward pass consumes that box rather than rebuilding a relaxation with
      -- exact-real algebra. The alpha-specific entry point below retains its explicit ReLU rule.
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => addConstant (α := α) dir st cadd
        | none => st.fail
      | none => st.fail
    | .mulElem =>
      match node.parents with
      | #[p1, p2] =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMulElem (α:=α) (dir:=dir) aY Bx By with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            addConstant (α := α) dir st2 cadd
          | none => st.fail
        | _, _ => st.fail
      | _ => st.fail
    | .sum =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]! with
        | some Bx =>
          if aY.n = 1 then
            let a0 : α := getAtOrZero aY.v [0]
            let out : FlatTensor α :=
              { n := Bx.dim, v := Tensor.full (α := α) (.dim Bx.dim .scalar) a0 }
            addCoeff (α:=α) st p1 out
          else st.fail
        | none => st.fail
      | _ => st.fail
    | .reshape _ _ | .flatten _ =>
      match node.parents with
      | #[p1] => addCoeff (α:=α) st p1 aY
      | _ => st.fail
    | .concat axis =>
      let contributions := do
        let layout ← concatBackwardLayout? (α := α) nodes ibp node axis
        let coefficients ← backwardConcatSplit (α := α) layout aY
        pure (node.parents.zip coefficients)
      match contributions with
      | some parents =>
        parents.foldl (fun state (parent, coefficient) =>
          addCoeff (α := α) state parent coefficient) st
      | none => st.fail
    | .transpose axis₁ axis₂ =>
      match node.parents with
      | #[p1] =>
        match OpContracts.transposePerm nodes[p1]!.outShape.rank axis₁ axis₂ with
        | .ok perm =>
          match backwardAxisPermutation? (α := α) node.outShape perm aY with
          | some aX => addCoeff (α := α) st p1 aX
          | none => st.fail
        | .error _ => st.fail
      | _ => st.fail
    | .permute perm =>
      match node.parents with
      | #[p1] =>
        match backwardAxisPermutation? (α := α) node.outShape perm aY with
        | some aX => addCoeff (α := α) st p1 aX
        | none => st.fail
      | _ => st.fail
    | .mseLoss =>
      -- The directed IBP pass already encloses the rounded subtraction, square, and mean. Reusing
      -- that enclosure avoids introducing an unqualified finite-precision quadratic relaxation.
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => addConstant (α := α) dir st cadd
        | none => st.fail
      | none => st.fail

/--
`backwardNode` with per-neuron relu slopes supplied from outside, which is the freedom
alpha-CROWN optimizes over. Every other node kind behaves exactly as it does in `backwardNode`.
-/
private def backwardNodeWithReluAlpha (dir : BackwardDir)
  (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
  (ctx : AffineCtx) (reluAlpha : Array (Option (FlatTensor α)))
  (st : BackwardState α) (id : Nat) : BackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
    let node := nodes[id]!
    match node.kind with
    | .relu =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]! with
        | some preB =>
          let n := preB.dim
          let idB := boundsIdentity (α:=α) n
          let localB? : Option (FlatAffineBounds α) :=
            match reluAlpha[id]? with
            | some (some a) =>
              if h : a.n = n then
                let aT : Tensor α [n] :=
                  castDimScalar (α:=α) (n:=a.n) (n':=n) h a.v
                some (propagateReluBoundsWithAlpha (α:=α) preB idB rfl aT)
              else
                some (propagateReluBounds (α:=α) preB idB rfl)
            | _ =>
              some (propagateReluBounds (α:=α) preB idB rfl)
          match localB? with
          | some localB =>
              match backwardUnaryDiag (α:=α) dir preB localB aY with
              | some (aX, cadd) =>
                let st' := addCoeff (α:=α) st p1 aX
                addConstant (α := α) dir st' cadd
              | none => st.fail
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | _ =>
      backwardNode (α:=α) dir nodes ps ibp ctx st id

/--
Run a complete backward sweep in one direction and return the objective as an affine form in the
graph input, or `none` when some node along the way could not be handled.
-/
def Internal.runBackwardObjectiveDir
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := 0 }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNode (α:=α) dir g.nodes ps ibp ctx acc i) init
    if st.failed then
      none
    else
      match st.coeffs[ctx.inputId]! with
      | some aIn =>
        if hIn : aIn.n = ctx.inputDim then
          let vIn : Tensor α [ctx.inputDim] :=
            castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
          let A : Tensor α [1, ctx.inputDim] := Tensor.dim (fun _ => vIn)
          let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
          some { A := A, c := c }
        else
          none
      | none =>
        -- Every active coefficient was consumed by input-independent nodes.
        let A : Tensor α [1, ctx.inputDim] :=
          Tensor.full (α := α) (.dim 1 (.dim ctx.inputDim .scalar)) 0
        let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
  else
    none

/-- `runBackwardObjectiveDir` with the relu slopes supplied from outside. -/
private def runBackwardObjectiveDirWithReluAlpha
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α)
  (reluAlpha : Array (Option (FlatTensor α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := 0 }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNodeWithReluAlpha (α:=α) dir g.nodes ps ibp ctx reluAlpha acc i) init
    if st.failed then
      none
    else
      match st.coeffs[ctx.inputId]! with
      | some aIn =>
        if hIn : aIn.n = ctx.inputDim then
          let vIn : Tensor α [ctx.inputDim] :=
            castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
          let A : Tensor α [1, ctx.inputDim] := Tensor.dim (fun _ => vIn)
          let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
          some { A := A, c := c }
        else
          none
      | none =>
        let A : Tensor α [1, ctx.inputDim] :=
          Tensor.full (α := α) (.dim 1 (.dim ctx.inputDim .scalar)) 0
        let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
  else
    none

/-- One step of the directed sweep, carrying interval coefficients instead of exact ones. -/
@[expose] def Internal.directedBackwardNode
    (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (ctx : AffineCtx) (st : DirectedBackwardState α) (id : Nat) :
    DirectedBackwardState α :=
  if st.failed then st else
  match st.coeffs[id]! with
  | none => st
  | some aY =>
      let node := nodes[id]!
      let consumeCurrent :=
        match ibp[id]! with
        | some By => consumeDirectedObjective (α := α) st aY By
        | none => st.fail
      match node.kind with
      | .input =>
          if node.id = ctx.inputId then
            st
          else
            consumeCurrent
      | .const _ =>
          match ps.constVals[id]? with
          | some v => consumeDirectedObjective (α := α) st aY (pointCoeffBox (α := α) v)
          | none => st.fail
      | .detach =>
          match unaryParent? node.parents with
          | some p => addDirectedCoeff (α := α) st p aY
          | none => st.fail
      | .add =>
          match binaryParents? node.parents with
          | some (p1, p2) =>
              let st := addDirectedCoeff (α := α) st p1 aY
              addDirectedCoeff (α := α) st p2 aY
          | none => st.fail
      | .sub =>
          match binaryParents? node.parents with
          | some (p1, p2) =>
              let st := addDirectedCoeff (α := α) st p1 aY
              addDirectedCoeff (α := α) st p2 (negateDirectedCoeff (α := α) aY)
          | none => st.fail
      | .linear =>
          match unaryParent? node.parents with
          | some p =>
              match ps.linearWB[id]? with
              | some config =>
                  match directedBackwardLinear (α := α) aY config.w config.b with
                  | some (aX, c) =>
                      let st := addDirectedCoeff (α := α) st p aX
                      addDirectedConstant (α := α) st c.1 c.2
                  | none => st.fail
              | none => st.fail
          | none => st.fail
      | .matmul =>
          match unaryParent? node.parents with
          | some p =>
              match ps.matmulW[id]? with
              | some config =>
                  let zeroBias := Tensor.full (α := α) (.dim config.m .scalar) 0
                  match directedBackwardLinear (α := α) aY config.w zeroBias with
                  | some (aX, c) =>
                      let st := addDirectedCoeff (α := α) st p aX
                      addDirectedConstant (α := α) st c.1 c.2
                  | none => st.fail
              | none => st.fail
          | none => consumeCurrent
      | .conv configuration =>
          match unaryParent? node.parents with
          | some p =>
              match ps.convCfg[id]?, nodes[p]? with
              | some config, some parent =>
                  match planConvTransfer? configuration config parent.outShape node.outShape with
                  | some leading =>
                      let aff := affOfConv (α := α) config leading
                      match directedBackwardLinear (α := α) aY aff.A aff.c with
                      | some (aX, c) =>
                          let st := addDirectedCoeff (α := α) st p aX
                          addDirectedConstant (α := α) st c.1 c.2
                      | none => st.fail
                  | none => st.fail
              | _, _ => st.fail
          | none => st.fail
      | .sum =>
          match unaryParent? node.parents with
          | some p =>
              match ibp[p]! with
              | some Bx =>
                  if h : aY.dim = 1 then
                    let lo : Tensor α [1] :=
                      castDimScalar (α := α) (n := aY.dim) (n' := 1) h aY.lo
                    let hi : Tensor α [1] :=
                      castDimScalar (α := α) (n := aY.dim) (n' := 1) h aY.hi
                    let aX : FlatBox α :=
                      { dim := Bx.dim
                        lo := Tensor.full (α := α) (.dim Bx.dim .scalar) (getAtOrZero lo [0])
                        hi := Tensor.full (α := α) (.dim Bx.dim .scalar) (getAtOrZero hi [0]) }
                    addDirectedCoeff (α := α) st p aX
                  else
                    st.fail
              | none => st.fail
          | none => st.fail
      | .reshape _ _ | .flatten _ =>
          match unaryParent? node.parents with
          | some p => addDirectedCoeff (α := α) st p aY
          | none => st.fail
      | .concat axis =>
          let contributions := do
            let layout ← concatBackwardLayout? (α := α) nodes ibp node axis
            let coefficients ← splitDirectedCoeff (α := α) layout aY
            pure (node.parents.zip coefficients)
          match contributions with
          | some parents =>
              parents.foldl (fun state (parent, coefficient) =>
                addDirectedCoeff (α := α) state parent coefficient) st
          | none => st.fail
      | .transpose axis₁ axis₂ =>
          match unaryParent? node.parents with
          | some p =>
              match (OpContracts.transposePerm nodes[p]!.outShape.rank axis₁ axis₂).toOption
                  >>= fun perm => directedAxisPermutation? node.outShape perm aY with
              | some aX => addDirectedCoeff st p aX
              | none => st.fail
          | none => st.fail
      | .permute perm =>
          match unaryParent? node.parents with
          | some p =>
              match directedAxisPermutation? node.outShape perm aY with
              | some aX => addDirectedCoeff st p aX
              | none => st.fail
          | none => st.fail
      | .layernorm _ =>
          if !crownNodeSemanticsSupported (α := α) nodes ps id then st.fail else consumeCurrent
      | _ => consumeCurrent

/-- Select affine endpoints for one interval coefficient, including the crossing-zero correction. -/
@[expose] def Internal.directedCoeffAffine (l u al au : α) : (α × α) × (α × α) :=
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
@[expose] def Internal.directedInputAffines
    (inputDim : Nat) (xB aIn : FlatBox α) (cLo cHi : α) :
    Option (AffineVec α inputDim 1 × AffineVec α inputDim 1) :=
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
        directedCoeffAffine l u al au
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
      let lower : AffineVec α inputDim 1 :=
        { A := Tensor.dim (fun _ => lowerRow)
          c := Tensor.dim (fun _ => Tensor.scalar lowerConstant) }
      let upper : AffineVec α inputDim 1 :=
        { A := Tensor.dim (fun _ => upperRow)
          c := Tensor.dim (fun _ => Tensor.scalar upperConstant) }
      some (lower, upper)
    else
      none
  else
    none

/--
Run the directed backward sweep and return the lower and upper affine forms of the objective.
-/
@[expose] def Internal.runDirectedBackwardObjective
    (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α) :
    Option (AffineVec α ctx.inputDim 1 × AffineVec α ctx.inputDim 1) := do
  if outputId < g.nodes.size then
    let initCoeffs :=
      (Array.replicate g.nodes.size none).set! outputId (some (pointCoeffBox (α := α) obj))
    let init : DirectedBackwardState α :=
      { coeffs := initCoeffs, cstLo := 0, cstHi := 0 }
    let st := (List.finRange g.nodes.size).reverse.foldl
      (fun acc i => directedBackwardNode (α := α) g.nodes ps ibp ctx acc i.val) init
    if st.failed then
      none
    else
      let inputBox ← ibp[ctx.inputId]?
      let inputBox ← inputBox
      let aIn := st.coeffs[ctx.inputId]!.getD
        { dim := ctx.inputDim
          lo := Tensor.full (α := α) (.dim ctx.inputDim .scalar) 0
          hi := Tensor.full (α := α) (.dim ctx.inputDim .scalar) 0 }
      directedInputAffines (α := α) ctx.inputDim inputBox aIn st.cstLo st.cstHi
  else
    none

/-- Nodewise affine bounds obtained by directed backward propagation of the coordinate objectives.

The rows share the checked node intervals. Coefficients are rounded outwards throughout each
sweep, including cancellation and sign changes; rounding only the final affine evaluation would
not enclose errors introduced while composing coefficients. A node without a directed transfer
retains its IBP enclosure as constant affine forms.

These bounds target the real-arithmetic graph described by the stored parameters, under the
backend's directed-arithmetic contract. They do not establish an error bound for a separate
floating-point execution schedule.
-/
@[expose] def directedNodeBounds? (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (outputId : Nat) :
    Option (FlatAffineBounds α) := do
  unless crownGraphSemanticsSupported g ps do failure
  let node ← g.nodes[outputId]?
  let outDim := node.outShape.size
  let rows : Option (Fin outDim → AffineVec α ctx.inputDim 1 × AffineVec α ctx.inputDim 1) :=
    Tensor.Internal.sequenceFinM fun i =>
      let objective : FlatTensor α :=
        { n := outDim, v := Tensor.ofFn fun j => if i = j then 1 else 0 }
      runDirectedBackwardObjective g ps ctx ibp outputId objective
  match rows with
  | some rows =>
    pure
      { inDim := ctx.inputDim
        outDim := outDim
        loAff :=
          { A := Tensor.matrix fun i j => Spec.get2 (rows i).1.A 0 j
            c := Tensor.ofFn fun i => Tensor.getScalar (rows i).1.c 0 }
        hiAff :=
          { A := Tensor.matrix fun i j => Spec.get2 (rows i).2.A 0 j
            c := Tensor.ofFn fun i => Tensor.getScalar (rows i).2.c 0 } }
  | none =>
    let box ← ibp[outputId]?
    let box ← box
    pure (boundsConst ctx.inputDim box.dim box.lo box.hi)

/--
Objective-dependent backward CROWN bound for a scalar objective.

Given a linear objective `objᵀ * output`, this runs a backward pass that propagates the objective
coefficients through the graph, selects the relaxation attached to each node, and returns a pair of
affine bounds on the objective with respect to `ctx.inputId`.

The returned `FlatAffineBounds` always has `outDim = 1` (a scalar objective).
-/
@[expose] def runCROWNBackwardObjective
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α) :
  Option (FlatAffineBounds α) :=
  if crownGraphSemanticsSupported (α := α) g ps then
    let bounds :=
      if BoundOps.supportsExactAffineReassociation (α := α) then
        (runBackwardObjectiveDir (α := α) .lower g ps ctx ibp outputId obj,
          runBackwardObjectiveDir (α := α) .upper g ps ctx ibp outputId obj)
      else
        match runDirectedBackwardObjective (α := α) g ps ctx ibp outputId obj with
        | some bounds => (some bounds.1, some bounds.2)
        | none =>
            (objectiveFromOutputBox (α := α) .lower ibp outputId ctx.inputDim obj,
              objectiveFromOutputBox (α := α) .upper ibp outputId ctx.inputDim obj)
    match bounds with
    | (some loAff, some hiAff) =>
        some { inDim := ctx.inputDim, outDim := 1, loAff := loAff, hiAff := hiAff }
    | _ => none
  else
    none

/-- Evaluate already-computed backward-CROWN objective bounds on an input box. -/
@[expose] def evalBackwardObjectiveBox? (bounds : FlatAffineBounds α) (xB : FlatBox α)
    (inputDim : Nat) : Except String (FlatBox α) := do
  if hIn : bounds.inDim = inputDim then
    if hXB : xB.dim = inputDim then
      if hOut : bounds.outDim = 1 then
        let outB := bounds.evalOnFlatBoxAsDim xB (by simpa [hXB] using hIn.symm) hOut
        pure { dim := 1, lo := outB.lo, hi := outB.hi }
      else
        throw s!"backward CROWN objective dimension mismatch: got {bounds.outDim}, expected 1"
    else
      throw s!"input box dimension mismatch: got {xB.dim}, expected {inputDim}"
  else
    throw s!"backward CROWN input dimension mismatch: got {bounds.inDim}, expected {inputDim}"

/--
Run objective-dependent backward CROWN and evaluate the scalar objective bounds on the input box.

The result is a `FlatBox` of dimension `1`, with `lo[0]` and `hi[0]` bounding
`objᵀ * output` over `xB`.
-/
@[expose] def backwardObjectiveBox? (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (xB : FlatBox α)
    (outputId : Nat) (obj : FlatTensor α) : Except String (FlatBox α) := do
  let some bounds := runCROWNBackwardObjective (α := α) g ps ctx ibp outputId obj
    | throw "CROWN backward objective failed"
  evalBackwardObjectiveBox? (α := α) bounds xB ctx.inputDim

/--
Backward CROWN objective lower bound with externally provided ReLU alpha slopes.

This is an integration hook for alpha-CROWN style workflows where ReLU slopes are optimized outside
TorchLean and then imported as a per-node vector in `reluAlpha`. Imported slopes currently refine
the exact scalar path. Rounded scalar backends use the directed coefficient pass; until imported
slopes carry their own rounding contract, nonlinear nodes are discharged against directed IBP
boxes.
-/
def runCROWNBackwardObjectiveLowerWithReluAlpha
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α)
  (reluAlpha : Array (Option (FlatTensor α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  if !crownGraphSemanticsSupported (α := α) g ps then
    none
  else if BoundOps.supportsExactAffineReassociation (α := α) then
    runBackwardObjectiveDirWithReluAlpha (α := α) .lower g ps ctx ibp outputId obj reluAlpha
  else
    (runDirectedBackwardObjective (α := α) g ps ctx ibp outputId obj).map (fun bounds => bounds.1)

end NN.MLTheory.CROWN.Graph
