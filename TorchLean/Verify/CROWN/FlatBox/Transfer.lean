/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Graph
public import TorchLean.Verify.CROWN.Activation
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Rep.Traversal.Sequence
public import TorchLean.Verify.Tensor

/-!
# Interval transfers on flat boxes

The interval transfer functions of the bound passes, on flat boxes (`FlatBox`) and flat tensors
(`FlatTensor`): elementwise arithmetic and activations, concatenation and slice layouts,
reductions, the softmax and layer-norm ranges, the linear layer on its parameters
(`TorchLean.SomeLinear`), the dimension casts they need, and the input subdivision of `Refinement`
(`split?` at the widest coordinate, `combine?`, `bound` under a work budget). None of them reads a
graph, and the passes over registered operators (`TorchLean.CROWN.Op`) call them.
-/

public section

open TorchLean.Shape.Coord (linearize unlinearize)
open TorchLean.Shape (size)

namespace TorchLean.CROWN.Graph

open _root_.TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
variable [BoundOps α]

open BoundOps

/--
A rank-one tensor packed with its length, used by the flat LiRPA engine.

Graph nodes carry dimensions found while lowering, so a length cannot always appear in the
surrounding static type, and `FlatTensor` keeps it in the field `n`.
-/
structure FlatTensor (α : Type) [Storage α] [Context α] where
  /-- Number of scalar entries. -/
  n : Nat
  /-- Rank-one tensor payload (shape `.dim n .scalar`). -/
  v : Tensor α [n]

/-- Geometry shared by all concat transfers, preserving parent occurrences in list order. -/
structure ConcatLayout where
  /-- Dimensions before the selected axis. -/
  leading : Shape
  /-- Dimensions after the selected axis. -/
  trailing : Shape
  /-- Selected-axis length of each parent occurrence, including zero lengths. -/
  lengths : List Nat

namespace ConcatLayout

/-- Shape of one parent occurrence. -/
@[expose]
def parentShape (layout : ConcatLayout) (parent : Fin layout.lengths.length) : Shape :=
  layout.leading ++ layout.lengths.get parent :: layout.trailing

/-- Shape of the concatenated output. -/
@[expose]
def outputShape (layout : ConcatLayout) : Shape :=
  layout.leading ++ layout.lengths.sum :: layout.trailing

/-- Bijection between parent scalar coordinates and concatenated scalar coordinates. -/
@[expose]
def flatEquiv (layout : ConcatLayout) :
    ((parent : Fin layout.lengths.length) × Fin (layout.parentShape parent).size) ≃
      Fin layout.outputShape.size :=
  (Equiv.sigmaCongrRight fun parent =>
    (Shape.Coord.equivFin (layout.parentShape parent)).symm).trans <|
    (Shape.Coord.concatEquiv
      layout.leading layout.trailing layout.lengths).trans <|
      Shape.Coord.equivFin layout.outputShape

/-- Read every output from its unique parent coordinate, without scalar arithmetic. -/
@[expose]
def concat (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) →
      Tensor α [(layout.parentShape parent).size]) :
    Tensor α [layout.outputShape.size] :=
  Tensor.ofFn fun i =>
    let source := layout.flatEquiv.symm i
    Tensor.getScalar (values source.1) source.2

/-- Pull output coefficients back to one parent occurrence using the same coordinate bijection. -/
@[expose]
def split (layout : ConcatLayout) (value : Tensor α [layout.outputShape.size])
    (parent : Fin layout.lengths.length) : Tensor α [(layout.parentShape parent).size] :=
  Tensor.ofFn fun i => Tensor.getScalar value (layout.flatEquiv ⟨parent, i⟩)

end ConcatLayout

/-- Concatenate flat values after checking every parent payload length. -/
@[expose]
def concatFlatValues? (layout : ConcatLayout) (values : Array (FlatTensor α)) :
    Option (FlatTensor α) := do
  unless values.size == layout.lengths.length do failure
  let inputs : Fin layout.lengths.length → FlatTensor α ←
    Tensor.sequenceFinM fun parent => values[parent.val]?
  if h : ∀ parent, (inputs parent).n = (layout.parentShape parent).size then
    let tensors := fun parent => h parent ▸ (inputs parent).v
    pure { n := layout.outputShape.size, v := layout.concat tensors }
  else
    none

/-- Concatenate interval endpoints through the value coordinate map, without rounding. -/
@[expose]
def concatFlatBoxes? (layout : ConcatLayout) (boxes : Array (FlatBox α)) :
    Option (FlatBox α) := do
  unless boxes.size == layout.lengths.length do failure
  let inputs : Fin layout.lengths.length → FlatBox α ←
    Tensor.sequenceFinM fun parent => boxes[parent.val]?
  if h : ∀ parent, (inputs parent).dim = (layout.parentShape parent).size then
    pure
      { dim := layout.outputShape.size
        lo := layout.concat fun parent => h parent ▸ (inputs parent).lo
        hi := layout.concat fun parent => h parent ▸ (inputs parent).hi }
  else
    none

/-- Checked geometry for a contiguous slice along one tensor axis. -/
structure SliceLayout where
  /-- The shape of the sliced tensor. -/
  sourceShape : Shape
  /-- The sliced axis. -/
  axis : Nat
  /-- The first index of the slice. -/
  start : Nat
  /-- The number of indices in the slice. -/
  length : Nat
  /-- `axis` is an axis of `sourceShape`. -/
  axisInBounds : Shape.AxisInBounds axis sourceShape
  /-- The slice ends within the axis. -/
  start_add_length_le : start + length ≤ @Shape.axisSize sourceShape axis axisInBounds

namespace SliceLayout

open Shape (Coord)

/-- Shape after replacing the selected axis by the slice length. -/
@[expose, reducible]
def outputShape (layout : SliceLayout) : Shape :=
  layout.sourceShape.set layout.axis layout.length

/-- Each output scalar reads one source scalar through the source slice coordinate map. -/
@[expose]
def flatIndex (layout : SliceLayout) (i : Fin layout.outputShape.size) :
    Fin layout.sourceShape.size :=
  Coord.linearize <|
    Tensor.sliceAxisRangeCoordinate ⟨layout.axis, layout.axisInBounds.proof⟩ layout.start
      layout.length layout.start_add_length_le (Coord.unlinearize i)

/-- Route slice coefficients back to their source coordinates, filling unused entries with zero. -/
@[expose]
def scatter {α : Type} [Storage α] [Zero α] (layout : SliceLayout)
    (value : Tensor α [layout.outputShape.size]) : Tensor α [layout.sourceShape.size] :=
  Tensor.reshape (target := [_]) (hSize := (Shape.size_vector _).symm) <|
    Tensor.sliceBackward (α := α) [⟨layout.axis, layout.axisInBounds.proof⟩] [layout.start]
      [layout.start + layout.length] [1]
      (Tensor.castShape (Tensor.reshape value layout.outputShape (Shape.size_vector _))
        (Shape.slice_singleton layout.sourceShape ⟨layout.axis, layout.axisInBounds.proof⟩
          layout.start layout.length).symm)

end SliceLayout

/-- Validate the slice axis, range, and declared output shape. -/
@[expose]
def sliceLayout? (axis start length : Nat) (sourceShape outputShape : Shape) :
    Option SliceLayout := do
  let ⟨hAxis⟩ ← Shape.axisInBounds? axis sourceShape
  letI := hAxis
  if hRange : start + length ≤ sourceShape.axisSize axis then
    if sourceShape.set axis length = outputShape then
      some
        { sourceShape, axis, start, length, axisInBounds := hAxis, start_add_length_le := hRange }
    else none
  else none

/-- Route interval objective coefficients through a slice without rounding their entries. -/
@[expose]
def scatterSliceBoxes? (layout : SliceLayout) (input : FlatBox α) : Option (FlatBox α) :=
  if h : input.dim = layout.outputShape.size then
    some
      { dim := layout.sourceShape.size
        lo := layout.scatter (h ▸ input.lo)
        hi := layout.scatter (h ▸ input.hi) }
  else none

/-- Default inhabitant for `FlatBox` (a 0-dimensional box at `0`). -/
instance : Inhabited (FlatBox α) where
  default :=
    { dim := 0
      lo := Tensor.full (α := α) (.dim 0 .scalar) 0
      hi := Tensor.full (α := α) (.dim 0 .scalar) 0 }

/-- Interval multiplication for scalar endpoints: given `[aLo,aHi]` and `[bLo,bHi]`, return bounds
  on the product. The result runs from the least to the greatest of the four corner products, each
  rounded in its own direction, so `[-1, 2]` times `[-3, 1]` gives `[-6, 3]`.
  `Graph.intervalMul_encloses` proves the enclosure at a lawful scalar. -/
@[expose] def intervalMul (aLo aHi bLo bHi : α) : α × α :=
  (Context.minimum (Context.minimum (mulDown aLo bLo) (mulDown aLo bHi))
      (Context.minimum (mulDown aHi bLo) (mulDown aHi bHi)),
    Context.maximum (Context.maximum (mulUp aLo bLo) (mulUp aLo bHi))
      (Context.maximum (mulUp aHi bLo) (mulUp aHi bHi)))

/-- Coordinatewise interval product of two flat boxes, with the four endpoint products of
`intervalMul` at each coordinate. The result is `none` when the dimensions differ. -/
@[expose] public def boxMulElem (B1 B2 : FlatBox α) : Option (FlatBox α) :=
  match B1, B2 with
  | ⟨n1, l1, u1⟩, ⟨n2, l2, u2⟩ =>
    if h : n1 = n2 then
      by
        cases h
        let lo :=
          Tensor.ofFn fun i =>
            (intervalMul (l1.getScalar i) (u1.getScalar i) (l2.getScalar i) (u2.getScalar i)).1
        let hi :=
          Tensor.ofFn fun i =>
            (intervalMul (l1.getScalar i) (u1.getScalar i) (l2.getScalar i) (u2.getScalar i)).2
        exact some { dim := n1, lo := lo, hi := hi }
    else none

/-- Convert a dependent `Box` of shape `.dim n .scalar` into a `FlatBox` with `dim := n`. -/
@[expose]
def toFlatBox (n : Nat) (B : Box α (.dim n .scalar)) : FlatBox α :=
  { dim := n, lo := B.lo, hi := B.hi }

/-- Convert a `FlatBox` to a dependent `Box` at shape `.dim B.dim .scalar`. -/
@[expose]
public def ofFlatBox (B : FlatBox α) : Box α (.dim B.dim .scalar) :=
  { lo := B.lo, hi := B.hi }

/-- Coordinatewise sum of two flat boxes, with lower endpoint sums rounded down and upper endpoint
sums rounded up. On a dimension mismatch the result is the left box. -/
@[expose]
public def boxAdd (B1 B2 : FlatBox α) : FlatBox α :=
  match B1 with
  | ⟨n1, lo1, hi1⟩ =>
    match B2 with
    | ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact
            { dim := n1
              lo := Tensor.zipWith BoundOps.addDown lo1 lo2
              hi := Tensor.zipWith BoundOps.addUp hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Coordinatewise difference of two flat boxes, rounded outward. The left box comes back
unchanged when the dimensions differ. -/
@[expose]
public def boxSub (B1 B2 : FlatBox α) : FlatBox α :=
  match B1 with
  | ⟨n1, lo1, hi1⟩ =>
    match B2 with
    | ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          -- Sound interval subtraction: [l1,u1] - [l2,u2] = [l1 - u2, u1 - l2]
          exact
            { dim := n1
              lo := Tensor.zipWith BoundOps.subDown lo1 hi2
              hi := Tensor.zipWith BoundOps.subUp hi1 lo2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Sum all coordinates of a flat box with directed accumulation. -/
@[expose]
def boxSum (B : FlatBox α) : FlatBox α :=
  let lo := (List.finRange B.dim).foldl (fun acc i =>
    BoundOps.addDown acc (B.lo.getScalar i)) 0
  let hi := (List.finRange B.dim).foldl (fun acc i =>
    BoundOps.addUp acc (B.hi.getScalar i)) 0
  { dim := 1
    lo := Tensor.full (α := α) (.dim 1 .scalar) lo
    hi := Tensor.full (α := α) (.dim 1 .scalar) hi }

/-- Apply ReLU to both endpoints of a `FlatBox` (monotone activation, so endpoints suffice). It uses
`Activation.relu`, the backend's ReLU. -/
@[expose]
public def boxReLU (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.map (fun x => Activation.relu (α := α) x) B.lo
    hi := Tensor.map (fun x => Activation.relu (α := α) x) B.hi }

/-- An upper bound of `|x|` for `x` in `[lo, hi]`: the larger of `hi` and `0 - lo` rounded up. -/
@[expose] def absBound (lo hi : α) : α :=
  Context.maximum (BoundOps.subUp 0 lo) hi

/-- Componentwise absolute value bounds using directed negation at negative endpoints. -/
@[expose] def boxAbs (B : FlatBox α) : FlatBox α :=
  let lo' := Tensor.ofFn fun i =>
    let l := B.lo.getScalar i
    let u := B.hi.getScalar i
    if l < 0 then
      if 0 < u then 0 else BoundOps.subDown 0 u
    else
      l
  let hi' := Tensor.ofFn fun i => absBound (B.lo.getScalar i) (B.hi.getScalar i)
  { dim := B.dim, lo := lo', hi := hi' }

/-- Apply a scalar interval enclosure coordinatewise to a flat box. -/
@[expose]
def boxUnaryEnclosure?
    (enclose : α → α → Option (α × α)) (B : FlatBox α) : Option (FlatBox α) := do
  let bounds ← Tensor.sequenceFinM fun i =>
    enclose (B.lo.getScalar i) (B.hi.getScalar i)
  let lower : Tensor α [B.dim] := Tensor.ofFn fun i => (bounds i).1
  let upper : Tensor α [B.dim] := Tensor.ofFn fun i => (bounds i).2
  pure { dim := B.dim, lo := lower, hi := upper }

/-- Componentwise square-root bounds using the scalar backend's domain checks.

The supplied real and native-float backends clamp negative lower endpoints to zero and reject
wholly negative intervals. A backend may also reject an operation it does not support. -/
@[expose]
def boxSqrt? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α) NonlinearBoundOps.sqrtBounds B

/-- Softplus bounds, applied independently at every tensor coordinate. -/
@[expose]
def boxSoftplus? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α) NonlinearBoundOps.softplusBounds B

/-- SafeLog bounds with one shared scalar epsilon interval.

The epsilon parent must contain one scalar. Keeping this check here makes direct graph construction
follow the same contract as the typed builder and IR shape inference. -/
@[expose]
def boxSafeLog? [NonlinearBoundOps α] (B epsilon : FlatBox α) : Option (FlatBox α) :=
  if h : epsilon.dim = 1 then
    let zero : Fin epsilon.dim := ⟨0, by omega⟩
    boxUnaryEnclosure? (α := α)
      (fun lo hi => NonlinearBoundOps.safeLogBounds lo hi
        (epsilon.lo.getScalar zero) (epsilon.hi.getScalar zero)) B
  else
    none

/-- Componentwise reciprocal bounds. The real and native-float backends give `none` when some
coordinate interval contains zero, as an endpoint or inside. -/
@[expose]
def boxInv? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi => NonlinearBoundOps.divBounds 1 1 lo hi) B

/-- The box of `-x` for `x` in `B`: `0 - B.hi` rounded down below and `0 - B.lo` rounded up above,
since `LawfulBoundOps` does not assume an exact negation (`Directed.boxNeg_encloses`). -/
@[expose] def boxNeg (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.map (fun x => BoundOps.subDown 0 x) B.hi
    hi := Tensor.map (fun x => BoundOps.subUp 0 x) B.lo }

/-- Componentwise max bounds: `max(x,y)` over interval boxes, with the bound scalar's `max`. On a
dimension mismatch the result is the left box. -/
@[expose] def boxMaxElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.zipWith max lo1 lo2
                  hi := Tensor.zipWith max hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Componentwise min bounds: `min(x,y)` over interval boxes, with the bound scalar's `min`. On a
dimension mismatch the result is the left box. -/
@[expose] def boxMinElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.zipWith min lo1 lo2
                  hi := Tensor.zipWith min hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Directed bounds on `x * x` for `x` between `l` and `u`: below, `0` when `l ≤ 0 ≤ u` and
otherwise the smaller downward-rounded endpoint square, and above, the larger upward-rounded
endpoint square. `square_encloses` proves the enclosure. -/
@[expose] def squareBounds (l u : α) : α × α :=
  (if !(decide (0 < l)) && !(decide (u < 0)) then 0
    else Context.minimum (mulDown l l) (mulDown u u),
   Context.maximum (mulUp l l) (mulUp u u))

/--
Componentwise square of an interval box: `squareBounds` at each coordinate.

The body is exposed because `boxSquare_dim` and `CROWN.Directed.boxSquare_encloses`
unfold it to prove that it keeps the dimension and encloses the square of each coordinate.
-/
@[expose] def boxSquare (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.ofFn fun i => (squareBounds (B.lo.getScalar i) (B.hi.getScalar i)).1
    hi := Tensor.ofFn fun i => (squareBounds (B.lo.getScalar i) (B.hi.getScalar i)).2 }

/-- Reinterpret a flattened tensor as shape `s` when the element counts agree. -/
@[expose] def ibpUnflatten {s : Shape} (dim : Nat) (t : Tensor α [dim]) (h : dim =
  Shape.size s) :
    Tensor α s :=
  let t' : Tensor α [Shape.size s] :=
    Tensor.castShape t (congrArg (fun k => Shape.dim k Shape.scalar) h)
  Tensor.reshape (α := α) t' s (Shape.size_vector _)

/-- Sum one axis with downward lower accumulation and upward upper accumulation. -/
@[expose]
def ibpReduceSumAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Shape.size s then
    match Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some _ =>
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceDimWith
          (fun row => Tensor.foldl addDown 0 row) _ axis xLo
        let yHi := Tensor.reduceDimWith
          (fun row => Tensor.foldl addUp 0 row) _ axis xHi
        let outS := s.eraseIdx axis
        let flatLo := Tensor.reshape (α := α) yLo [_] (Shape.size_vector _).symm
        let flatHi := Tensor.reshape (α := α) yHi [_] (Shape.size_vector _).symm
        some { dim := Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/--
Format-independent softmax enclosure on a flattened tensor.

A row of length one gets the point box at 1, and every coordinate of a longer row lies in `[0,1]`.
The rule forgoes a tighter exponential relaxation, so executable checking needs no directed
transcendental implementation from its scalar backend.
-/
@[expose]
def ibpSoftmaxRange (s : Shape) (axis dim : Nat) : FlatBox α :=
  let rowLength := s[axis]?.getD 0
  if rowLength = 1 then
    let ones := Tensor.full (α := α) (.dim dim .scalar) 1
    { dim := dim, lo := ones, hi := ones }
  else
    { dim := dim
      lo := Tensor.full (α := α) (.dim dim .scalar) 0
      hi := Tensor.full (α := α) (.dim dim .scalar) 1 }

/-!
## Hard-masked softmax IBP (last axis)

Blocked coordinates have weight zero. An allowed coordinate lies in `[0,1]`, and it has weight one
when it is the only allowed coordinate in its row. These bounds do not evaluate `exp` or division,
so they hold for executable endpoint types whose `BoundOps` instance covers only directed
arithmetic. A tighter transfer rule requires separately certified directed bounds for
transcendental operations.
-/

/-- Conservative interval bounds for hard-masked softmax along the last tensor axis. -/
@[expose]
def ibpHardMaskedSoftmax : {s : Shape} →
    Tensor α s → Tensor α s → Tensor Bool s → (Tensor α s × Tensor α s)
  | .scalar, _lo, _hi, allowed =>
      let allowed := allowed.item
      let value := if allowed then 1 else 0
      (Tensor.scalar value, Tensor.scalar value)
  | .dim n .scalar, _lo, _hi, allowed =>
      let lower := Tensor.ofFn fun i =>
        if allowed.getScalar i then
          let hasOtherAllowed := (List.finRange n).any fun j =>
            i != j && allowed.getScalar j
          if hasOtherAllowed then 0 else 1
        else
          0
      let upper := Tensor.ofFn fun i =>
        if allowed.getScalar i then 1 else 0
      (lower, upper)
  | .dim n inner, lo, hi, allowed =>
      let lower := Tensor.dim fun i : Fin n =>
        (ibpHardMaskedSoftmax (s := inner)
          (lo.unstack i) (hi.unstack i) (allowed.unstack i)).1
      let upper := Tensor.dim fun i : Fin n =>
        (ibpHardMaskedSoftmax (s := inner)
          (lo.unstack i) (hi.unstack i) (allowed.unstack i)).2
      (lower, upper)

/-!
## LayerNorm interval propagation

Normalization acts independently on each row of the configured trailing shape. Its directed
transfer bounds the mean, variance, stabilized denominator, and affine scale and bias.
-/

/--
Checks an interval before a nonlinear transfer uses it: the result is the interval itself when the
upper endpoint is not below the lower one and each endpoint `x` satisfies `x - x == 0`, and `none`
otherwise. Under IEEE 754 that test fails exactly for infinite and NaN endpoints, and over `ℝ` it
always holds.
-/
@[expose] def finiteBounds? (bounds : α × α) : Option (α × α) :=
  if !(decide (bounds.2 < bounds.1)) && bounds.1 - bounds.1 == 0 &&
      bounds.2 - bounds.2 == 0 then
    some bounds
  else
    none

omit [Storage α] [BoundOps α] in
/-- A successful finite-bounds check returns its input endpoints unchanged. -/
theorem finiteBounds?_eq_of_eq_some {input output : α × α}
    (h : finiteBounds? input = some output) : input = output := by
  unfold finiteBounds? at h
  split at h
  · exact Option.some.inj h
  · contradiction

/-- The count `n` between two endpoints: `n` ones added to zero, rounding down and rounding up. -/
@[expose] def countBounds (n : Nat) : α × α :=
  (List.range n).foldl (fun b _ => (BoundOps.addDown b.1 1, BoundOps.addUp b.2 1)) (0, 0)

/-- Directed mean bounds for one nonempty row. The denominator encloses the real row
length, including lengths that the scalar format cannot represent. -/
@[expose] def rowMean? [NonlinearBoundOps α] {n : Nat}
    (bounds : Fin n → α × α) : Option (α × α) := do
  if n = 0 then none else do
    let (sumLo, sumHi, countLo, countHi) :=
      (List.finRange n).foldl (fun (lo, hi, countLo, countHi) i =>
        (addDown lo (bounds i).1, addUp hi (bounds i).2,
         addDown countLo 1, addUp countHi 1)) (0, 0, 0, 0)
    let _ ← finiteBounds? (sumLo, sumHi)
    let _ ← finiteBounds? (countLo, countHi)
    let result ← NonlinearBoundOps.divBounds sumLo sumHi countLo countHi
    finiteBounds? result

/-- Average a nonempty flat box, enclosing both the sum and the exact coordinate count. -/
@[expose] def boxMean? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) := do
  let (lo, hi) ← rowMean? fun i => (B.lo.getScalar i, B.hi.getScalar i)
  pure { dim := 1, lo := Tensor.full [1] lo, hi := Tensor.full [1] hi }

/--
Average one axis using directed sums and division. The denominator encloses the exact axis
length, so a rounded natural-number conversion cannot silently narrow the result.
-/
@[expose]
def ibpReduceMeanAxis [NonlinearBoundOps α]
    (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) := do
  let summed ← ibpReduceSumAxis axis Xin s
  let length := s[axis]?.getD 0
  let (countLo, countHi) := countBounds length
  let _ ← finiteBounds? (countLo, countHi)
  boxUnaryEnclosure? (fun lo hi => do
    let _ ← finiteBounds? (lo, hi)
    let result ← NonlinearBoundOps.divBounds lo hi countLo countHi
    finiteBounds? result) summed

/--
Directed bounds for one LayerNorm row, given its affine parameters `gamma` and `beta`.

The order is that of `Tensor.layerNorm` without its scale: center the input, center again inside
`var`, square, average, clamp variance to zero, add epsilon, take a square root, and divide the
once-centered values by it. Over the reals, dividing each centered row by its scale first, as
`Tensor.layerNorm` does, changes nothing. Each coordinate is then multiplied by its gamma, which
swaps the interval endpoints when gamma is negative, and shifted by its beta.

Epsilon must be finite and positive, the row endpoints finite and ordered, and the affine
parameters finite. Division and square root use the selected nonlinear backend, and an invalid or
nonfinite intermediate interval gives `none`. `layerNormRow?_encloses` proves that the
bounds enclose the real row under the backend's arithmetic laws. Whether a native floating-point
LayerNorm lands in the box is a separate numerical question.

We kept each step to plain interval arithmetic, so a coordinate and the row mean are bounded
independently. With two coordinates in `[0, 1]` the centered interval is `[-1, 1]`, while
`x₁ - (x₁ + x₂) / 2` ranges over `[-1/2, 1/2]`.
-/
@[expose] def layerNormRow? [NonlinearBoundOps α] {n : Nat}
    (lo hi gamma beta : Tensor α [n]) (epsilon : α) :
    Option (Tensor α [n] × Tensor α [n]) := do
  let _ ← finiteBounds? (epsilon, epsilon)
  if !(epsilon > 0) then none else do
    let _ ← Tensor.sequenceFinM fun i : Fin n => do
      let _ ← finiteBounds? (lo.getScalar i, hi.getScalar i)
      let _ ← finiteBounds? (gamma.getScalar i, gamma.getScalar i)
      finiteBounds? (beta.getScalar i, beta.getScalar i)
    let (meanLo, meanHi) ← rowMean? fun i => (lo.getScalar i, hi.getScalar i)
    let centeredLo := Tensor.ofFn fun i => BoundOps.subDown (lo.getScalar i) meanHi
    let centeredHi := Tensor.ofFn fun i => BoundOps.subUp (hi.getScalar i) meanLo
    let _ ← Tensor.sequenceFinM fun i : Fin n =>
      finiteBounds? (centeredLo.getScalar i, centeredHi.getScalar i)
    let (centerMeanLo, centerMeanHi) ← rowMean? fun i =>
      (centeredLo.getScalar i, centeredHi.getScalar i)
    let recenteredLo :=
      Tensor.ofFn fun i => BoundOps.subDown (centeredLo.getScalar i) centerMeanHi
    let recenteredHi :=
      Tensor.ofFn fun i => BoundOps.subUp (centeredHi.getScalar i) centerMeanLo
    let _ ← Tensor.sequenceFinM fun i : Fin n =>
      finiteBounds? (recenteredLo.getScalar i, recenteredHi.getScalar i)
    let squaredLo := Tensor.ofFn fun i =>
      (squareBounds (recenteredLo.getScalar i) (recenteredHi.getScalar i)).1
    let squaredHi := Tensor.ofFn fun i =>
      (squareBounds (recenteredLo.getScalar i) (recenteredHi.getScalar i)).2
    let (varianceLo, varianceHi) ← rowMean? fun i =>
      (squaredLo.getScalar i, squaredHi.getScalar i)
    let (stabilizedLo, stabilizedHi) ← finiteBounds?
      (BoundOps.addDown (Context.maximum varianceLo 0) epsilon,
       BoundOps.addUp (Context.maximum varianceHi 0) epsilon)
    let (denominatorLo, denominatorHi) ←
      NonlinearBoundOps.sqrtBounds (Context.maximum stabilizedLo 0)
          (Context.maximum stabilizedHi 0) >>=
        finiteBounds?
    if !(denominatorLo > 0) then none else do
      let bounds ← Tensor.sequenceFinM fun i : Fin n => do
        let (lower, upper) ←
          NonlinearBoundOps.divBounds (centeredLo.getScalar i) (centeredHi.getScalar i)
            denominatorLo denominatorHi >>= finiteBounds?
        let scale := gamma.getScalar i
        let (scaledLo, scaledHi) ← finiteBounds?
          (Context.minimum (BoundOps.mulDown lower scale) (BoundOps.mulDown upper scale),
           Context.maximum (BoundOps.mulUp lower scale) (BoundOps.mulUp upper scale))
        finiteBounds?
          (BoundOps.addDown scaledLo (beta.getScalar i),
           BoundOps.addUp scaledHi (beta.getScalar i))
      pure (Tensor.ofFn fun i => (bounds i).1, Tensor.ofFn fun i => (bounds i).2)

-- Casting helpers for dependent shapes
/-- Cast the output dimension of an affine map across a proven equality. -/
@[expose]
def castAffineOut {n m m' : Nat}
  (h : m = m') (a : Linear α n m) : Linear α n m' :=
  { weights :=
      Tensor.castShape a.weights (congrArg (fun k => Shape.dim k (Shape.dim n Shape.scalar)) h)
    bias := Tensor.castShape a.bias (congrArg (fun k => Shape.dim k Shape.scalar) h) }

/--
Cast a dim-scalar tensor across an equality of dimensions.

It is an `abbrev`, so it unfolds at reducible transparency in the `simp`-based soundness proofs.
-/
abbrev castDimScalar {n n' : Nat}
    (h : n = n') (t : Tensor α [n]) : Tensor α [n'] :=
  Tensor.castShape t (congrArg (fun k => Shape.dim k Shape.scalar) h)

omit [Context α] [BoundOps α] in
/-- Casting a flat vector tensor along an equality from a dimension to itself changes no data. -/
@[simp] theorem castDimScalar_self {n : Nat}
    (h : n = n) (t : Tensor α [n]) :
    castDimScalar (α := α) h t = t := by
  exact Tensor.castShape_self t _

omit [Context α] [BoundOps α] in
/-- Reading a dimension cast reads the source at the cast index. -/
theorem getScalar_castDimScalar {n n' : Nat} (h : n = n') (t : Tensor α [n]) (i : Fin n') :
    Tensor.getScalar (castDimScalar (α := α) h t) i = Tensor.getScalar t (Fin.cast h.symm i) := by
  cases h
  simp [castDimScalar]

/-- Interval propagation through the linear layer `p` by `IBP.linear`, with the bias as a point
box. The result is `none` when the box dimension differs from `p.inDim`. -/
@[expose]
public def ibpLinearParams (p : SomeLinear α) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = p.inDim then
    let xB   : Box α (.dim p.inDim .scalar) := Xin.getScalarBox h
    let bBox : Box α (.dim p.outDim .scalar) := Box.point (α:=α) p.spec.bias
    let yB := CROWN.IBP.linear (α:=α) (m:=p.outDim) (n:=p.inDim) p.spec.weights xB bBox
    some (toFlatBox p.outDim yB)
  else none

/-- The bias of a linear map at each of `rows` rows, flattened row-major. Over `ℝ` this is the
value of a linear node whose input extent is empty (the IR adds the bias to an empty sum). -/
@[expose]
public def linearBiasRows {m : Nat} (b : Tensor α [m]) (rows : Nat) : Tensor α [rows * m] :=
  Tensor.ofFn fun k => b.getScalar (Fin.modNat k)

section Nonlinear

variable [NonlinearBoundOps α]

/-- Enclose real SiLU by multiplying the input box and its sigmoid enclosure.

The result encloses the real function, and it says nothing about the rounding error of a native
SiLU kernel.
-/
@[expose] def ibpSiLU? (input : FlatBox α) : Option (FlatBox α) := do
  let gate ← boxUnaryEnclosure? NonlinearBoundOps.sigmoidBounds input
  boxMulElem input gate

/-- The interval transfer of a reduction: a sum over every dimension or over one dimension, without
`keepdim`. Other reductions have no transfer. -/
@[expose] def ibpReduce? (op : Tensor.ReduceOp) (dims : Array Nat) (keepdim : Bool)
    (parentShape : Shape) (input : FlatBox α) : Option (FlatBox α) :=
  if op = .sum ∧ keepdim = false then
    if dims = Array.range parentShape.rank then some (boxSum input)
    else if h : dims.size = 1 then ibpReduceSumAxis dims[0] input parentShape
    else none
  else none

/-- The interval transfer of a mean over one dimension without `keepdim`. -/
@[expose] def ibpMean? (dims : Array Nat) (keepdim : Bool) (parentShape : Shape)
    (input : FlatBox α) : Option (FlatBox α) :=
  if h : dims.size = 1 ∧ keepdim = false then ibpReduceMeanAxis dims[0] input parentShape
  else none

end Nonlinear

namespace Refinement

omit [BoundOps α] in
/-- Split one coordinate at a shared boundary, leaving every other endpoint unchanged. -/
@[expose] def splitAt (box : FlatBox α) (axis : Fin box.dim) (cut : α) : FlatBox α × FlatBox α :=
  ({ box with hi := Tensor.ofFn fun i => if i = axis then cut else box.hi.getScalar i },
   { box with lo := Tensor.ofFn fun i => if i = axis then cut else box.lo.getScalar i })

end Refinement

end TorchLean.CROWN.Graph

namespace TorchLean.CROWN.Graph


variable {α : Type} [Storage α] [Context α]

namespace Refinement

open Tensor (boundsOrdered)

/-- Split the widest coordinate that has a representable interior midpoint.

Halving both endpoints before adding them keeps the midpoint `lo / 2 + hi / 2` finite whenever the
endpoints are. The strict checks `lo < cut` and `cut < hi` reject stagnation, unordered values and
infinite midpoints. A box with no such coordinate, or with reversed or unordered endpoints, gives
`none`, and no epsilon perturbation of its domain forces a split.
-/
def split? (box : FlatBox α) : Option (FlatBox α × FlatBox α) := Id.run do
  if !boundsOrdered box.lo box.hi then return none
  let mut best : Option (Fin box.dim × α × α) := none
  for i in List.finRange box.dim do
    let lo := box.lo.getScalar i
    let hi := box.hi.getScalar i
    let cut := lo / 2 + hi / 2
    if lo < cut && cut < hi then
      let width := hi - lo
      match best with
      | none => best := some (i, cut, width)
      | some (_, _, oldWidth) =>
        if oldWidth < width then best := some (i, cut, width)
  return best.map fun (i, cut, _) => splitAt box i cut

/-- Combine compatible boxes by their coordinatewise hull or intersection.

Both inputs and the result must have ordered endpoints (`Tensor.boundsOrdered`), so an empty
intersection gives `none`. These checks also keep NaN endpoints from disappearing through endpoint
selection on native floating-point backends.
-/
def combine? (hull : Bool) (a b : FlatBox α) : Option (FlatBox α) := do
  if !boundsOrdered a.lo a.hi || !boundsOrdered b.lo b.hi then none
  else if h : b.dim = a.dim then
    let blo : Tensor α [a.dim] := h ▸ b.lo
    let bhi : Tensor α [a.dim] := h ▸ b.hi
    let result : FlatBox α :=
      { dim := a.dim
        lo := Tensor.ofFn fun i =>
          if hull then Context.minimum (a.lo.getScalar i) (blo.getScalar i)
          else Context.maximum (a.lo.getScalar i) (blo.getScalar i)
        hi := Tensor.ofFn fun i =>
          if hull then Context.maximum (a.hi.getScalar i) (bhi.getScalar i)
          else Context.minimum (a.hi.getScalar i) (bhi.getScalar i) }
    if boundsOrdered result.lo result.hi then some result else none
  else none

/-- Refine any box-to-box enclosure procedure using at most `budget` binary splits.

The two halves of a split share the remaining budget, so there are at most `2 * budget + 1` calls
to `bound` and the budget counts work over the whole split tree. Zero budget returns
`enclose input`. When both halves succeed, the hull of their results is intersected with
`enclose input`, or used alone when `enclose input` is `none`. A failed half, a dimension mismatch
or an invalid hull or intersection falls back to `enclose input`, and one successful half never
stands in for the whole input domain.
-/
def bound (enclose : FlatBox α → Option (FlatBox α)) (input : FlatBox α)
    (budget : Nat) : Option (FlatBox α) :=
  let baseline := enclose input
  match budget with
  | 0 => baseline
  | remaining + 1 =>
    match split? input with
    | none => baseline
    | some (left, right) =>
      let leftBudget := remaining / 2
      let rightBudget := remaining - leftBudget
      match bound enclose left leftBudget, bound enclose right rightBudget with
      | some a, some b =>
        match combine? true a b with
        | none => baseline
        | some hull =>
          match baseline with
          | none => some hull
          | some original => (combine? false original hull).or baseline
      | _, _ => baseline
termination_by budget
decreasing_by
  all_goals have := Nat.div_le_self remaining 2
  all_goals omega

/-- With zero budget, `bound enclose input` is `enclose input`. -/
@[simp] theorem bound_zero (enclose : FlatBox α → Option (FlatBox α)) (input : FlatBox α) :
    bound enclose input 0 = enclose input := by
  rw [bound]

end Refinement

end TorchLean.CROWN.Graph
