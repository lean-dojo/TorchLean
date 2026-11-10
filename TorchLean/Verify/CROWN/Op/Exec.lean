/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.Cert

/-!
# Interval bound propagation on operator graphs

`runIBP G P seeds` computes one optional interval box per node of a graph of registered operators
(`IR.Graph`) at any scalar with endpoint operations (`BoundOps`, `NonlinearBoundOps`), node by
node in order (`tabulate`). An input node gets its seed box, and a constant node the point box of
its payload constant when that constant has numeric entries (`flatOf`: Boolean, index and integer
constants have no box). An operator node gets the rule of its operator on the boxes its references
name and the constants of its parameter references (`ibp?`). A node with a reference to an output
other than `0` has no box, and neither has a node whose rule finds no box or no constant it reads.
The Bernoulli mask's box ignores its input.

`shared?` holds the rules of `cert?`, the certificate rule, at any scalar, one for each operator
of kind `shared` in the table `ruleKind`: detach, add, sub, mul, relu, silu, softplus, safeLog,
linear, matmul, scaledMatmul, conv and cat. `directed?` holds the engine's rules for the operators
of kinds `backend` and `directed`, and an operator of kind `none` has no box. Tanh, sigmoid, sine,
cosine and exp take the backend's enclosures, while `cert?` keeps endpoint min-max and
clipped midpoint rules for the first four, the operators of kind `backend`. A backend may have no
enclosure of an operator: native `Float` has none of exp, and native `Float32` none of exp, tanh
and sigmoid. Log on a positive box, sqrt and reciprocal take the backend's enclosures too, and
abs, negation and squaring use directed endpoint differences and products, with zero below a box
across zero. Max and min, reshape and the operators that copy, select or take extrema of input
entries (permute, slice, flip, broadcast, indexAt, max and min pooling, max and min reductions)
apply the operator to each endpoint. Sum and average pooling sum each window by directed additions,
and an average divides that sum by the backend's quotient enclosure with the window's count between
directed sums of ones, on finite sums, counts and quotients.

The sums and means that `ibpReduce?` and `ibpMean?` accept go through those functions. Softmax and
last-axis masked softmax on a finite box get boxes inside `[0, 1]`. Layer norm runs
`layerNormRow?` on each row, and batch norm with running statistics of literal constants,
on any axis, runs `batchNormDenominator?` on each channel and `batchNormBounds` on
each entry. Scaling, leaky ReLU, ELU and clamping by scalars of literal constants use
`scaleBounds`, `leakyReLUBounds`, `eluBounds` and `clampBounds`
on each entry, and GELU uses `zeroHullBounds`. Sinh and cosh (`sinhBounds`,
`coshBounds`) come from the backend's enclosures of exp and of the quotient by 2,
log-softmax (`logSoftmaxEntry`) from its enclosures of exp and log on each fiber, and
cumulative sums (`cumsumBounds`) from directed additions. Select takes the hull of the
boxes of its two branches (`selectBox?`), and padding pads the endpoints with the
endpoints of the padding value (`padBox?`). Gather and embedding take at each output
coordinate the hull of the operand's box over every entry an index could name
(`gatherBox?`), without reading the index port.

The sign, floor, ceiling, rounding, truncation, fractional part, arctangent, arcsine, arccosine and
erf go through `elementwiseBox?`: the sign of each endpoint, the box widened by 1, or a
range of the map, which arcsine and arccosine get only on a finite box inside `[-1, 1]`.
`exp x - 1`, `2 ^ x`, `arsinh`, tangent, `log (1 + x)`, `log2`, `log10` and the
reciprocal square root come from the backend's enclosures of exp, log, sqrt, sine, cosine and the
quotient. Fmax and fmin take `boxMaxElem` and `boxMinElem`, while true and truncating division,
atan2 and hypot go entry by entry through `binaryBox?`. When the dividend's box is finite and the
divisor's box is finite and avoids zero (`nonzeroDivisor?`), fmod gets the zero hull of the
dividend's box and remainder gets `[-m, m]` for a bound `m` of both operands' sizes. Copysign
gets `[-m, m]` for a bound `m` of its first operand's size, and contraction (`contractBox?`) adds
outward products by directed sums. Mse loss gets the mean of the squared difference of its two
boxes, and the random operators get the range `[0, 1]`. Every other operator has no box.

The affine and monotone rules are the interval bound propagation of Gowal et al. (2018), Sec. 3,
typed in by hand operator by operator. A linear layer takes per weight the smaller and
the larger of its two endpoint products, rounded down and up, and sums them by directed additions
(`IBP.linear`). At `ℝ` that is the box of their Eq. (6), which gets it from the center and the
radius with two matrix products, where we spend four products per weight. ReLU and clamping are
nondecreasing and send endpoints to endpoints, as in their Eq. (7).

No theorem here relates the boxes at a rounded scalar to its native execution. At `ℝ`,
`TorchLean.Verify.CROWN.Op.ExecCert` relates them to the certificate. At every lawful scalar,
`Op.Directed.encloses_of_run` proves that each box of `runIBP` encloses its node's flat value in a
successful real run whose inputs lie in the seed boxes.

References:
- Gowal et al., "On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust
  Models", arXiv:1810.12715: Sec. 3, Eq. (6) and (7).
-/

@[expose] public section

open TorchLean.Shape.Coord (linearize unlinearize)
open TorchLean.Shape (size)

namespace TorchLean.CROWN.Op

open Graph
open Shape (Coord)

variable {α : Type} [Storage α] [Context α]

/-- The box of output `i` of the graph: the box of the node its reference names, at output `0`. -/
def outputBox? (G : IR.Graph) (boxes : Array (Option (FlatBox α))) (i : Nat) :
    Except String (FlatBox α) :=
  match G.outputs[i]? with
  | some r =>
      match box? boxes r with
      | some B => pure B
      | none => throw s!"output box missing at node {r.node} output {r.out}"
  | none => throw s!"graph output {i} is out of bounds for {G.outputs.size} outputs"

/-- Seed boxes by node id: the box `xs[k]` at an input node of graph input `k`. -/
def inputSeeds {β : Type} (G : IR.Graph) (xs : List β) (id : Nat) : Option β :=
  match G.nodes[id]? with
  | some (.input k _) => xs[k]?
  | _ => none

/-- Every graph input `k` of `G` has the box `inputs[k]` of its port's size, with ordered
endpoints. -/
def validInputs (G : IR.Graph) (inputs : List (FlatBox α)) : Bool :=
  G.nodes.all fun
    | .input k p =>
      match inputs[k]? with
      | some box => box.dim == p.shape.size && Tensor.boundsOrdered box.lo box.hi
      | none => false
    | _ => true

/-- `some ()` when every endpoint of `B` passes `finiteBounds?`, none otherwise. -/
def finiteBox? (B : FlatBox α) : Option Unit := do
  let _ ← Tensor.sequenceFinM fun i : Fin B.dim =>
    finiteBounds? (B.lo.getScalar i, B.hi.getScalar i)
  pure ()

/-- The box `[0, 1]` of `n` entries. -/
def unitBox (n : Nat) : FlatBox α :=
  { dim := n, lo := Tensor.full [n] 0, hi := Tensor.full [n] 1 }

/-- The box of an operator from shape `s` to shape `t` with the bounds `bound lo hi o` of each
output coordinate `o`, where `lo` and `hi` are the endpoints of `B` read at `s`. It is none unless
`B` has `s.size` entries and every output coordinate has bounds. -/
def coordwiseBox? {s t : Shape}
    (bound : Tensor α s → Tensor α s → t.Coord → Option (α × α)) (B : FlatBox α) :
    Option (FlatBox α) :=
  if h : B.dim = s.size then do
    let lo := Tensor.reshape B.lo s ((Shape.size_vector _).trans h)
    let hi := Tensor.reshape B.hi s ((Shape.size_vector _).trans h)
    let bounds ← Tensor.sequenceFinM fun j : Fin t.size =>
      bound lo hi (Coord.unlinearize j)
    pure { dim := t.size, lo := Tensor.ofFn fun j => (bounds j).1,
           hi := Tensor.ofFn fun j => (bounds j).2 }
  else none

/-- The operator on the lower and on the upper endpoint of `B`, read at its input port. On ordered
values without NaN this encloses the operator's value for an operator that computes each output
entry as the input entry at a fixed coordinate, an extremum of the input entries at a fixed set of
coordinates, or zero for a window of padding only, as for the operators `directed?` passes it. -/
def endpointsBox? [Op.Capabilities α] {s : Shape} {outs : List Port}
    (op : Op [.value s] outs) (B : FlatBox α) : Option (FlatBox α) :=
  if h : B.dim = s.size then
    let read (x : Tensor α [B.dim]) : Op.Value α :=
      ⟨.value s, Tensor.reshape x s ((Shape.size_vector _).trans h)⟩
    match op.invoke (.ok [read B.lo]), op.invoke (.ok [read B.hi]) with
    | .ok [l], .ok [u] => do
        let l ← flatOf l
        let u ← flatOf u
        if hn : u.n = l.n then some { dim := l.n, lo := l.v, hi := hn ▸ u.v } else none
    | _, _ => none
  else none

variable [BoundOps α] [NonlinearBoundOps α]

/-- The range of an entry with scale `g` and bias `b` of a layer-normalized row of width `width`:
`b ± |g| r` rounded outward, where `r` is the backend's bound `layerNormAbsBound width` on a
normalized entry (`|x̂| ≤ sqrt width`). At unit scale and zero bias it is `[-r, r]`. A row of at
most one entry gets `[b, b]`, since a single entry normalizes to zero, and a wider row gets none
from a backend without the bound. -/
def layerNormRange (width : Nat) (g b : α) : Option (α × α) :=
  if width ≤ 1 then some (b, b)
  else (NonlinearBoundOps.layerNormAbsBound (α := α) width).map fun r =>
    if g == 1 && b == 0 then (BoundOps.subDown 0 r, r)
    else
      let s := BoundOps.mulUp (Graph.absBound g g) r
      (BoundOps.subDown b s, BoundOps.addUp b s)

/-- The box of a layer norm over the last axis of `rows` rows of width `width`, with scale
`gamma`, bias `beta` and epsilon `epsilon`: each row through `layerNormRow?`, each entry
intersected with its uniform range (`Op.layerNormRange`) where the backend has one. It is
none unless the input box has `rows * width` entries and every row has bounds. -/
def layerNormBox? (rows width : Nat) (gamma beta : Tensor α [width]) (epsilon : α)
    (input : FlatBox α) : Option (FlatBox α) :=
  let matrix : Shape := .dim rows (.dim width .scalar)
  if h : input.dim = matrix.size then do
    let lo := ibpUnflatten (s := matrix) input.dim input.lo h
    let hi := ibpUnflatten (s := matrix) input.dim input.hi h
    let bounds ← Tensor.sequenceFinM fun i : Fin rows =>
      layerNormRow? (lo.unstack i) (hi.unstack i) gamma beta epsilon
    let range (j : Fin width) :=
      Op.layerNormRange width (gamma.getScalar j) (beta.getScalar j)
    let outLo : Tensor α matrix := Tensor.dim fun i => Tensor.ofFn fun j =>
      match range j with
      | some (l, _) => Context.maximum ((bounds i).1.getScalar j) l
      | none => (bounds i).1.getScalar j
    let outHi : Tensor α matrix := Tensor.dim fun i => Tensor.ofFn fun j =>
      match range j with
      | some (_, u) => Context.minimum ((bounds i).2.getScalar j) u
      | none => (bounds i).2.getScalar j
    pure { dim := matrix.size
           lo := Tensor.reshape outLo [_] (Shape.size_vector _).symm
           hi := Tensor.reshape outHi [_] (Shape.size_vector _).symm }
  else none

/-- The box of a softmax along the last axis of `s` with the Boolean mask `allowed`:
`ibpHardMaskedSoftmax` on the endpoints read at `s`. -/
def maskedSoftmaxBox? (s : Shape) (allowed : Tensor Bool s) (B : FlatBox α) :
    Option (FlatBox α) :=
  if h : B.dim = s.size then
    let (lo, hi) := ibpHardMaskedSoftmax
      (Tensor.reshape B.lo s ((Shape.size_vector _).trans h))
      (Tensor.reshape B.hi s ((Shape.size_vector _).trans h)) allowed
    some { dim := s.size, lo := lo.reshape [_] (Shape.size_vector _).symm,
           hi := hi.reshape [_] (Shape.size_vector _).symm }
  else none

/-- The window sum of a pooling at output coordinate `o` between two endpoints: the window's entries
of `lo`, and of `hi`, added to zero in window order rounding down, and rounding up, with zero for a
padding cell. -/
def poolSumBounds {s : Shape} {dims : List (Fin s.rank)}
    {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool} (lo hi : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) : α × α :=
  let window := List.finRange (Shape.size (s.poolWindow dims kernel))
  (window.foldl (fun acc i =>
      BoundOps.addDown acc (Tensor.poolInput lo o (Coord.unlinearize i))) 0,
    window.foldl (fun acc i =>
      BoundOps.addUp acc (Tensor.poolInput hi o (Coord.unlinearize i))) 0)

/-- The bounds of `sqrt (max var 0 + epsilon)` for a channel of a batch norm with mean `mean`,
variance `var`, scale `gamma`, bias `beta` and epsilon `epsilon`: the backend's square root
enclosure between directed sums. They exist only when every parameter, the sums and the root bounds
pass `finiteBounds?` and the lower root bound is positive. -/
def batchNormDenominator? (mean var gamma beta epsilon : α) : Option (α × α) := do
  let _ ← finiteBounds? (epsilon, epsilon)
  let _ ← finiteBounds? (mean, mean)
  let _ ← finiteBounds? (gamma, gamma)
  let _ ← finiteBounds? (beta, beta)
  let _ ← finiteBounds? (var, var)
  let v := Context.maximum var 0
  let r ← finiteBounds? (BoundOps.addDown v epsilon, BoundOps.addUp v epsilon)
  let d ← NonlinearBoundOps.sqrtBounds r.1 r.2 >>= finiteBounds?
  if 0 < d.1 then pure d else none

/-- Bounds of `(x - mean) / r * gamma + beta` for `x` between `lo` and `hi` and `r` between the
endpoints of `d`, by directed differences and sums, the backend's quotient enclosure and the
outward-rounded product. The endpoints and every intermediate result must pass
`finiteBounds?`, or the result is none. -/
def batchNormBounds (lo hi mean gamma beta : α) (d : α × α) : Option (α × α) := do
  let _ ← finiteBounds? (lo, hi)
  let c ← finiteBounds? (BoundOps.subDown lo mean, BoundOps.subUp hi mean)
  let q ← NonlinearBoundOps.divBounds c.1 c.2 d.1 d.2 >>= finiteBounds?
  let p ← finiteBounds? (intervalMul q.1 q.2 gamma gamma)
  finiteBounds? (BoundOps.addDown p.1 beta, BoundOps.addUp p.2 beta)

variable [Op.Capabilities α]

/-- The box of a pooling with reduction `op` from the box `B` of its input. Max and min pooling
apply the operator to each endpoint, sum pooling sums each window by `Op.poolSumBounds`, and
average pooling divides that sum by the backend's quotient enclosure with the window's count
between `Graph.countBounds`. An average is none unless the sum, the count and the quotient pass
`finiteBounds?`. -/
def poolBox? {s : Shape} (op : Tensor.PoolOp) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) (B : FlatBox α) :
    Option (FlatBox α) :=
  match op with
  | .max | .min => Op.endpointsBox?
      (Op.pool (s := s) op dims kernel stride padLow padHigh dilation ceilMode) B
  | .sum =>
      Op.coordwiseBox? (s := s) (t := s.pool dims kernel stride padLow padHigh dilation
        ceilMode) (fun lo hi o => some (Op.poolSumBounds lo hi o)) B
  | .avg countIncludePad =>
      Op.coordwiseBox? (s := s) (t := s.pool dims kernel stride padLow padHigh dilation
        ceilMode) (fun lo hi o => do
          let sum ← finiteBounds? (Op.poolSumBounds lo hi o)
          let count ← finiteBounds?
            (Graph.countBounds (Tensor.poolCount countIncludePad o))
          NonlinearBoundOps.divBounds sum.1 sum.2 count.1 count.2 >>= finiteBounds?) B

/-- The box of `x c` for `x` in `[lo, hi]`: the outward product of `[lo, hi]` and `[c, c]`. -/
def scaleBounds (c lo hi : α) : Option (α × α) := some (intervalMul lo hi c c)

/-- The box of `min (max x a) b` for `x` in `[lo, hi]`, nondecreasing in `x` for any `a` and `b`:
the map at each endpoint. -/
def clampBounds (a b lo hi : α) : Option (α × α) :=
  some (Context.minimum (Context.maximum lo a) b, Context.minimum (Context.maximum hi a) b)

/-- The box of `x` above 0 and `a x` at most 0 for `x` in `[lo, hi]`, for either sign of `a`. It
is `[lo, hi]` when `0 < lo`, the hull of `a lo` rounded outward, 0 and `hi` when `lo ≤ 0 < hi`, and
the outward product of `[lo, hi]` and `[a, a]` otherwise. -/
def leakyReLUBounds (a lo hi : α) : Option (α × α) :=
  if 0 < lo then some (lo, hi)
  else if 0 < hi then
    some (Context.minimum (BoundOps.mulDown a lo) 0, Context.maximum (BoundOps.mulUp a lo) hi)
  else some (intervalMul lo hi a a)

/-- The box of `x` above 0 and `a (exp x - 1)` at most 0 for `x` in `[lo, hi]`, for either sign of
`a`. It is `[lo, hi]` when `0 < lo`. Otherwise it is the outward product of `[a, a]` and the
backend's bounds of `exp` on `[lo, min hi 0]` less one rounded outward, joined with 0 and `hi` when
`0 < hi`. -/
def eluBounds (a lo hi : α) : Option (α × α) :=
  if 0 < lo then some (lo, hi)
  else do
    let e ← NonlinearBoundOps.expBounds lo (Context.minimum hi 0)
    let n := intervalMul (BoundOps.subDown e.1 1) (BoundOps.subUp e.2 1) a a
    if 0 < hi then some (Context.minimum n.1 0, Context.maximum n.2 hi) else some n

/-- The box of a map whose value lies between its argument and 0, for `x` in `[lo, hi]`: the hull
of `[lo, hi]` and 0. -/
def zeroHullBounds (lo hi : α) : Option (α × α) :=
  some (Context.minimum lo 0, Context.maximum hi 0)

/-- The box of `x / 2` for `x` in `[lo, hi]`: the backend's quotient enclosure by `[2, 2]`, its
endpoints `1 + 1` rounded down and up. -/
def halfBounds (lo hi : α) : Option (α × α) :=
  NonlinearBoundOps.divBounds lo hi (BoundOps.addDown 1 1) (BoundOps.addUp 1 1)

/-- The backend's bounds of `exp (-x)` for `x` in `[lo, hi]`: its bounds of `exp` on
`[0 - hi, 0 - lo]` rounded outward. -/
def expNegBounds (lo hi : α) : Option (α × α) :=
  NonlinearBoundOps.expBounds (BoundOps.subDown 0 hi) (BoundOps.subUp 0 lo)

/-- The box of `sinh x = (exp x - exp (-x)) / 2` for `x` in `[lo, hi]`: the outward difference of
the backend's bounds of `exp x` and `exp (-x)`, halved. -/
def sinhBounds (lo hi : α) : Option (α × α) := do
  let e ← NonlinearBoundOps.expBounds lo hi
  let f ← Op.expNegBounds lo hi
  Op.halfBounds (BoundOps.subDown e.1 f.2) (BoundOps.subUp e.2 f.1)

/-- Bounds of `2 cosh a = exp a + exp (-a)`: the outward sum of the backend's bounds at `a`. -/
def twoCoshAt (a : α) : Option (α × α) := do
  let e ← NonlinearBoundOps.expBounds a a
  let f ← Op.expNegBounds a a
  some (BoundOps.addDown e.1 f.1, BoundOps.addUp e.2 f.2)

/-- The box of `cosh x` for `x` in `[lo, hi]`, from the bounds of `2 cosh` (`twoCoshAt`)
halved. As `cosh` is nondecreasing in `|x|`, the box lies below the larger of its upper bounds at
`lo` and `hi`, and above its lower bound at `lo` when `0 < lo`, at `hi` when `hi < 0`, and 1
otherwise. -/
def coshBounds (lo hi : α) : Option (α × α) := do
  let l ← Op.twoCoshAt lo
  let u ← Op.twoCoshAt hi
  Op.halfBounds (if 0 < lo then l.1 else if hi < 0 then u.1 else BoundOps.addDown 1 1)
    (Context.maximum l.2 u.2)

/-- The bounds of entry `k` of the log-softmax of a vector between `lo` and `hi`, as
`-log (∑ j, if j = k then 1 else exp (x j - x k))`: the backend's bounds of `exp` on
`[lo j - hi k, hi j - lo k]` rounded outward, their directed sums, and the backend's bounds of
`log` between the sums, negated outward. With exact enclosures and arithmetic these are the least
and greatest values of the entry on the box. -/
def logSoftmaxBounds {n : Nat} (lo hi : Fin n → α) (k : Fin n) : Option (α × α) := do
  let e ← Tensor.sequenceFinM fun j : Fin n =>
    if j = k then some ((1 : α), (1 : α))
    else NonlinearBoundOps.expBounds (BoundOps.subDown (lo j) (hi k)) (BoundOps.subUp (hi j) (lo k))
  let l ← NonlinearBoundOps.logBounds
    ((List.finRange n).foldl (fun acc j => BoundOps.addDown acc (e j).1) 0)
    ((List.finRange n).foldl (fun acc j => BoundOps.addUp acc (e j).2) 0)
  pure (BoundOps.subDown 0 l.2, BoundOps.subUp 0 l.1)

/-- The bounds of the log-softmax along `dim` at coordinate `c` between the endpoint tensors `lo`
and `hi`: `Op.logSoftmaxBounds` on their fibers along `dim` through `c`, at the index of `c`
along `dim`. -/
def logSoftmaxEntry {s : Shape} (dim : Fin s.rank) (lo hi : Tensor α s) (c : s.Coord) :
    Option (α × α) :=
  Op.logSoftmaxBounds (fun j => (lo.fiber dim c).getScalar j)
    (fun j => (hi.fiber dim c).getScalar j)
    (Coord.axisEquiv dim (shape := s) c).1

/-- The bounds of the cumulative sum along `dim` at coordinate `c` between the endpoint tensors
`lo` and `hi`: the entries of their fibers through `c` with index at most that of `c`, added to
zero in index order rounding down, and rounding up. -/
def cumsumBounds {s : Shape} (dim : Fin s.rank) (lo hi : Tensor α s) (c : s.Coord) :
    Option (α × α) :=
  let src := fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
    Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i)
  some ((List.finRange _).foldl (fun acc i => BoundOps.addDown acc (lo (src i))) 0,
    (List.finRange _).foldl (fun acc i => BoundOps.addUp acc (hi (src i))) 0)

/-- The box of `select` from the boxes `B₁` and `B₂` of its two branches: at each entry the least
of their lower endpoints and the greatest of their upper endpoints, whatever the condition. Boxes
of different widths give none. -/
def selectBox? (B₁ B₂ : FlatBox α) : Option (FlatBox α) :=
  if h : B₂.dim = B₁.dim then
    some { dim := B₁.dim
           lo := Tensor.ofFn fun i =>
             Context.minimum (B₁.lo.getScalar i) (B₂.lo.getScalar (Fin.cast h.symm i))
           hi := Tensor.ofFn fun i =>
             Context.maximum (B₁.hi.getScalar i) (B₂.hi.getScalar (Fin.cast h.symm i)) }
  else none

/-- The box of `pad` from the box `Bx` of its input, read at `s`, and the box `Bv` of its padding
value: the padding of the lower endpoints with the lower endpoint of the value, and of the upper
endpoints with the upper one. It is none unless the boxes have the widths of their ports. -/
def padBox? {s : Shape} (low high interior : Fin s.rank → Nat) (Bx Bv : FlatBox α) :
    Option (FlatBox α) :=
  if hx : Bx.dim = s.size then
    if hv : Bv.dim = Shape.size [] then
      let pad := fun (x : Tensor α [Bx.dim]) (v : Tensor α [Bv.dim]) =>
        Tensor.pad (Tensor.reshape x s ((Shape.size_vector _).trans hx)) low high interior
          (Tensor.reshape v [] ((Shape.size_vector _).trans hv)).item
      some { dim := (s.pad low high interior).size
             lo := Tensor.ofFn fun j => pad Bx.lo Bv.lo (Coord.unlinearize j)
             hi := Tensor.ofFn fun j => pad Bx.hi Bv.hi (Coord.unlinearize j) }
    else none
  else none

/-- The index tuple of the coordinate `a` along the axes `dims`. -/
def axisRead {s : Shape} (dims : List (Fin s.rank)) (a : s.Coord) :
    Einops.AxisTuple s.extent dims :=
  fun k => Shape.Coord.get a (dims.get k)

/-- The coordinate `h.source p ct cr` that `gather` reads for the index tuple `p` has the tuple `p`
along the indexed axes `dims`. -/
theorem axisRead_source {s t : Shape} {dims : List (Fin s.rank)}
    {batchDims : List (Fin s.rank × Fin t.rank)} (h : Shape.GatherValid s t dims batchDims)
    (p : Einops.AxisTuple s.extent dims) (ct : t.Coord)
    (cr : (s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord) :
    Op.axisRead dims (h.source p ct cr) = p := by
  have hnd : dims.Nodup := h.nodup.sublist (List.sublist_append_left _ _)
  funext k
  have hk : dims.idxOf (dims.get k) = k.val := hnd.idxOf_getElem k.val k.isLt
  apply Fin.ext
  simp only [Op.axisRead, Shape.GatherValid.source, Shape.Coord.get_ofFun,
    List.get_mem, dite_true, Fin.val_cast]
  have key : ∀ (i j : Fin dims.length), i = j → (p i).val = (p j).val := by
    rintro i j rfl; rfl
  exact key _ _ (Fin.ext hk)

/-- The bounds of `gather` at output coordinate `o` between the endpoint tensors `lo` and `hi`:
the least lower and greatest upper endpoint over the input coordinates that `gather` can read at
`o` for some index tuple. It is none when there is no such coordinate. -/
def gatherBound {s t : Shape} {dims : List (Fin s.rank)}
    {batchDims : List (Fin s.rank × Fin t.rank)} (h : Shape.GatherValid s t dims batchDims)
    (lo hi : Tensor α s) (o : (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord) :
    Option (α × α) :=
  let parts := Coord.appendEquiv t (s.eraseAxes (dims ++ batchDims.map Prod.fst)) o
  (List.finRange s.size).foldl (fun acc j =>
    let a : s.Coord := Coord.unlinearize j
    if Coord.linearize (h.source (Op.axisRead dims a) parts.1 parts.2) = j then
      match acc with
      | none => some (lo a, hi a)
      | some (l, u) => some (Context.minimum l (lo a), Context.maximum u (hi a))
    else acc) none

/-- The box of `gather` from the box `B` of its operand, read at `s`: the `Op.gatherBound`
hull at each output coordinate, without reading the index port. It is none unless `B` has `s.size`
entries and every output coordinate has a bound. -/
def gatherBox? {s t : Shape} {dims : List (Fin s.rank)}
    {batchDims : List (Fin s.rank × Fin t.rank)} (h : Shape.GatherValid s t dims batchDims)
    (B : FlatBox α) : Option (FlatBox α) :=
  Op.coordwiseBox? (s := s) (t := t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))
    (Op.gatherBound h) B

/-- The box `[-m, n]`, its endpoints rounded outward, whatever the argument. -/
def rangeBounds (m n : Nat) (_lo _hi : α) : Option (α × α) :=
  some (BoundOps.subDown 0 (Graph.countBounds m).2, (Graph.countBounds n).2)

/-- `rangeBounds m n` at a finite argument box inside `[-1, 1]`, and none otherwise: rounded
arcsine and arccosine are NaN outside `[-1, 1]`. -/
def unitRangeBounds (m n : Nat) (lo hi : α) : Option (α × α) :=
  if (finiteBounds? (lo, hi)).isSome && !(decide (lo < -1) || decide (1 < hi)) then
    rangeBounds m n lo hi
  else none

/-- The box of a map whose value lies within 1 of its argument, for `x` in `[lo, hi]`:
`[lo - 1, hi + 1]` rounded outward. -/
def withinOneBounds (lo hi : α) : Option (α × α) :=
  some (BoundOps.subDown lo 1, BoundOps.addUp hi 1)

/-- The box of the sign of `x` for `x` in `[lo, hi]`, nondecreasing: the sign of each endpoint,
with `-1` rounded outward. -/
def signBounds (lo hi : α) : Option (α × α) :=
  some (if 0 < lo then 1 else if lo < 0 then BoundOps.subDown 0 1 else 0,
    if hi < 0 then BoundOps.subUp 0 1 else if 0 < hi then 1 else 0)

/-- The box of `exp x - 1` for `x` in `[lo, hi]`: the backend's bounds of `exp`, less one rounded
outward. -/
def expm1Bounds (lo hi : α) : Option (α × α) := do
  let e ← NonlinearBoundOps.expBounds lo hi
  some (BoundOps.subDown e.1 1, BoundOps.subUp e.2 1)

/-- The box of `log (1 + x)` for `x` in `[lo, hi]`: the backend's bounds of `log` on
`[1 + lo, 1 + hi]` rounded outward. -/
def log1pBounds (lo hi : α) : Option (α × α) :=
  NonlinearBoundOps.logBounds (BoundOps.addDown 1 lo) (BoundOps.addUp 1 hi)

/-- The box of `1 / sqrt x` for `x` in `[lo, hi]`: the backend's quotient enclosure of 1 by its
bounds of `sqrt`. -/
def rsqrtBounds (lo hi : α) : Option (α × α) := do
  let r ← NonlinearBoundOps.sqrtBounds lo hi
  NonlinearBoundOps.divBounds 1 1 r.1 r.2

/-- The box of `tan x = sin x / cos x` for `x` in `[lo, hi]`: the backend's quotient enclosure of
its bounds of `sin` by its bounds of `cos`. -/
def tanBounds (lo hi : α) : Option (α × α) := do
  let s ← NonlinearBoundOps.sinBounds lo hi
  let c ← NonlinearBoundOps.cosBounds lo hi
  NonlinearBoundOps.divBounds s.1 s.2 c.1 c.2

/-- The box of `logb b x = log x / log b` for `x` in `[lo, hi]`: the backend's bounds of `log` at
`x` and on `Graph.countBounds b`, and its quotient enclosure. -/
def logbBounds (b : Nat) (lo hi : α) : Option (α × α) := do
  let l ← NonlinearBoundOps.logBounds lo hi
  let d ← NonlinearBoundOps.logBounds (Graph.countBounds b).1 (Graph.countBounds b).2
  NonlinearBoundOps.divBounds l.1 l.2 d.1 d.2

/-- The box of `2 ^ x = exp (log 2 * x)` for `x` in `[lo, hi]`: the backend's bounds of `exp` on
the outward product of its bounds of `log 2` and `[lo, hi]`. -/
def exp2Bounds (lo hi : α) : Option (α × α) := do
  let d ← NonlinearBoundOps.logBounds (Graph.countBounds 2).1 (Graph.countBounds 2).2
  let p := intervalMul d.1 d.2 lo hi
  NonlinearBoundOps.expBounds p.1 p.2

/-- Bounds of `arsinh a = log (a + sqrt (1 + a ^ 2))`: the backend's bounds of `sqrt` and `log`
between directed sums and products. -/
def arsinhAt (a : α) : Option (α × α) := do
  let r ← NonlinearBoundOps.sqrtBounds (BoundOps.addDown 1 (BoundOps.mulDown a a))
    (BoundOps.addUp 1 (BoundOps.mulUp a a))
  NonlinearBoundOps.logBounds (BoundOps.addDown a r.1) (BoundOps.addUp a r.2)

/-- The box of `arsinh x` for `x` in `[lo, hi]`, increasing: the lower bound at `lo` and the upper
bound at `hi` of `Op.arsinhAt`. -/
def arsinhBounds (lo hi : α) : Option (α × α) := do
  let l ← Op.arsinhAt lo
  let u ← Op.arsinhAt hi
  some (l.1, u.2)

/-- The box `[-m, m]`: its lower end `0 - m` rounded down. -/
def symBounds (m : α) : Option (α × α) :=
  some (BoundOps.subDown 0 m, m)

/-- `enclose` at a finite dividend box and a finite divisor box that avoids zero, and none
otherwise: rounded fmod and remainder are NaN at a zero divisor or an infinite dividend. -/
def nonzeroDivisor? (enclose : α → α → α → α → Option (α × α)) (aLo aHi bLo bHi : α) :
    Option (α × α) :=
  if (finiteBounds? (aLo, aHi)).isSome && (finiteBounds? (bLo, bHi)).isSome &&
      NonlinearBoundOps.denominatorAvoidsZero bLo bHi then enclose aLo aHi bLo bHi
  else none

/-- The box of a binary map from the boxes `B₁` and `B₂` of its operands: each pair of entries
through `enclose`. Boxes of different widths, or a pair without bounds, give none. -/
def binaryBox? (enclose : α → α → α → α → Option (α × α)) (B₁ B₂ : FlatBox α) :
    Option (FlatBox α) :=
  if h : B₂.dim = B₁.dim then do
    let bounds ← Tensor.sequenceFinM fun i : Fin B₁.dim =>
      enclose (B₁.lo.getScalar i) (B₁.hi.getScalar i) (B₂.lo.getScalar (Fin.cast h.symm i))
        (B₂.hi.getScalar (Fin.cast h.symm i))
    pure { dim := B₁.dim, lo := Tensor.ofFn fun i => (bounds i).1,
           hi := Tensor.ofFn fun i => (bounds i).2 }
  else none

/-- The box of `hypot a b = sqrt (a * a + b * b)`: the backend's bounds of `sqrt` between the
directed sums of the outward squares, each at least 0. -/
def hypotBounds (aLo aHi bLo bHi : α) : Option (α × α) :=
  let p := intervalMul aLo aHi aLo aHi
  let q := intervalMul bLo bHi bLo bHi
  NonlinearBoundOps.sqrtBounds (BoundOps.addDown (Context.maximum 0 p.1) (Context.maximum 0 q.1))
    (BoundOps.addUp p.2 q.2)

/-- Bounds of entry `o` of the contraction of `x` and `y` between the endpoint tensors `xlo`,
`xhi` of `x` and `ylo`, `yhi` of `y`: the outward products of the entries it pairs, added to zero
in the order of `Tensor.contract`, rounding down and rounding up. -/
def contractBounds {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}
    (h : Shape.ContractValid s t dims batchDims) (xlo xhi : Tensor α s) (ylo yhi : Tensor α t)
    (o : (s.contract t dims batchDims).Coord) : α × α :=
  (List.finRange (Shape.size ((dims.map Prod.fst).map s.extent))).foldl
    (fun acc i =>
      let a := s.contractLeft t dims batchDims o (Coord.unlinearize i)
      let b := h.contractRight o (Coord.unlinearize i)
      let p := intervalMul (xlo a) (xhi a) (ylo b) (yhi b)
      (BoundOps.addDown acc.1 p.1, BoundOps.addUp acc.2 p.2)) (0, 0)

/-- The box of a contraction from the boxes `Bx` and `By` of its operands, read at `s` and `t`:
`Op.contractBounds` at each output coordinate. Boxes of other widths give none. -/
def contractBox? {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}
    (h : Shape.ContractValid s t dims batchDims) (Bx By : FlatBox α) : Option (FlatBox α) :=
  if hy : By.dim = t.size then
    Op.coordwiseBox? (s := s) (t := s.contract t dims batchDims) (fun lo hi o =>
      some (Op.contractBounds h lo hi
        (Tensor.reshape By.lo t ((Shape.size_vector _).trans hy))
        (Tensor.reshape By.hi t ((Shape.size_vector _).trans hy)) o)) Bx
  else none

/-- The box of an elementwise or binary operator from the boxes of its references. Sign
(`Op.signBounds`), floor, ceiling, rounding and truncation (`Op.withinOneBounds`),
fractional part and erf (`[-1, 1]`), arctangent and arcsine (`[-2, 2]`), arccosine (`[0, 4]`),
`exp x - 1`, `2 ^ x`, `arsinh`, tangent, `log (1 + x)`, `log2`, `log10` and the reciprocal square
root take each entry of the reference's box through their bounds, arcsine and arccosine only on a
finite box inside `[-1, 1]` (`Op.unitRangeBounds`). Fmax and fmin take `boxMaxElem`
and `boxMinElem` of the two boxes, true and truncating division, atan2, hypot, fmod, remainder and
copysign go through `Op.binaryBox?`, fmod and remainder only at a finite dividend box and a finite
divisor box that avoids zero (`Op.nonzeroDivisor?`), and contraction through `Op.contractBox?`.
Every other operator gives none. -/
@[simp] def elementwiseBox? (boxes : Array (Option (FlatBox α))) :
    {ins outs : List Port} → Op ins outs → List IR.Ref → Option (FlatBox α)
  | _, _, .sign, [r] => box? boxes r >>= boxUnaryEnclosure? Op.signBounds
  | _, _, .floor, [r] | _, _, .ceil, [r] | _, _, .round, [r] | _, _, .trunc, [r] =>
      box? boxes r >>= boxUnaryEnclosure? Op.withinOneBounds
  | _, _, .frac, [r] | _, _, .erf, [r] =>
      box? boxes r >>= boxUnaryEnclosure? (Op.rangeBounds 1 1)
  | _, _, .atan, [r] => box? boxes r >>= boxUnaryEnclosure? (Op.rangeBounds 2 2)
  | _, _, .asin, [r] => box? boxes r >>= boxUnaryEnclosure? (Op.unitRangeBounds 2 2)
  | _, _, .acos, [r] => box? boxes r >>= boxUnaryEnclosure? (Op.unitRangeBounds 0 4)
  | _, _, .expm1, [r] => box? boxes r >>= boxUnaryEnclosure? Op.expm1Bounds
  | _, _, .exp2, [r] => box? boxes r >>= boxUnaryEnclosure? Op.exp2Bounds
  | _, _, .asinh, [r] => box? boxes r >>= boxUnaryEnclosure? Op.arsinhBounds
  | _, _, .tan, [r] => box? boxes r >>= boxUnaryEnclosure? Op.tanBounds
  | _, _, .log1p, [r] => box? boxes r >>= boxUnaryEnclosure? Op.log1pBounds
  | _, _, .log2, [r] => box? boxes r >>= boxUnaryEnclosure? (Op.logbBounds 2)
  | _, _, .log10, [r] => box? boxes r >>= boxUnaryEnclosure? (Op.logbBounds 10)
  | _, _, .rsqrt, [r] => box? boxes r >>= boxUnaryEnclosure? Op.rsqrtBounds
  | _, _, .fmax, [r₁, r₂] => do
      let B₁ ← box? boxes r₁
      let B₂ ← box? boxes r₂
      if B₁.dim = B₂.dim then some (boxMaxElem B₁ B₂) else none
  | _, _, .fmin, [r₁, r₂] => do
      let B₁ ← box? boxes r₁
      let B₂ ← box? boxes r₂
      if B₁.dim = B₂.dim then some (boxMinElem B₁ B₂) else none
  | _, _, .div .none, [r₁, r₂] => do
      Op.binaryBox? NonlinearBoundOps.divBounds (← box? boxes r₁) (← box? boxes r₂)
  | _, _, .div .trunc, [r₁, r₂] => do
      boxUnaryEnclosure? Op.withinOneBounds
        (← Op.binaryBox? NonlinearBoundOps.divBounds (← box? boxes r₁) (← box? boxes r₂))
  | _, _, .atan2, [r₁, r₂] => do
      Op.binaryBox? (fun aLo aHi _ _ => Op.rangeBounds 4 4 aLo aHi) (← box? boxes r₁)
        (← box? boxes r₂)
  | _, _, .hypot, [r₁, r₂] => do
      Op.binaryBox? Op.hypotBounds (← box? boxes r₁) (← box? boxes r₂)
  | _, _, .contract _ _ h, [rx, ry] => do
      Op.contractBox? h (← box? boxes rx) (← box? boxes ry)
  | _, _, .fmod, [r₁, r₂] => do
      Op.binaryBox? (Op.nonzeroDivisor? fun aLo aHi _ _ => Op.zeroHullBounds aLo aHi)
        (← box? boxes r₁) (← box? boxes r₂)
  | _, _, .remainder, [r₁, r₂] => do
      Op.binaryBox? (Op.nonzeroDivisor? fun aLo aHi bLo bHi => Op.symBounds
          (Context.maximum (Graph.absBound aLo aHi) (Graph.absBound bLo bHi)))
        (← box? boxes r₁) (← box? boxes r₂)
  | _, _, .copysign, [r₁, r₂] => do
      Op.binaryBox? (fun aLo aHi _ _ => Op.symBounds (Graph.absBound aLo aHi))
        (← box? boxes r₁) (← box? boxes r₂)
  | _, _, _, _ => none

/-- The box of scaling, leaky ReLU, ELU and clamping, whose scalars are payload constants at frozen
scalar ports: each entry of the box of the value reference through `Op.scaleBounds`,
`Op.leakyReLUBounds`, `Op.eluBounds` and `Op.clampBounds`. GELU, sinh and cosh
go entry by entry through `Op.zeroHullBounds`, `Op.sinhBounds` and
`Op.coshBounds`, log-softmax and cumulative sums through `Op.logSoftmaxEntry` and
`Op.cumsumBounds` at each coordinate, select and padding through `Op.selectBox?` and
`Op.padBox?`, and gather and embedding through `Op.gatherBox?` on the box of their
value reference. Every other operator goes to `Op.elementwiseBox?`. -/
@[simp] def scalarParamBox? (G : IR.Graph) (P : IR.Payload α)
    (boxes : Array (Option (FlatBox α))) :
    {ins outs : List Port} → Op ins outs → List IR.Ref → Option (FlatBox α)
  | _, _, .scale, [rx, rc] => do
      let c ← constAt? G P rc (.frozen [])
      box? boxes rx >>= boxUnaryEnclosure? (Op.scaleBounds (Tensor.item c))
  | _, _, .leakyReLU, [rx, ra] => do
      let a ← constAt? G P ra (.frozen [])
      box? boxes rx >>= boxUnaryEnclosure? (Op.leakyReLUBounds (Tensor.item a))
  | _, _, .elu, [rx, ra] => do
      let a ← constAt? G P ra (.frozen [])
      box? boxes rx >>= boxUnaryEnclosure? (Op.eluBounds (Tensor.item a))
  | _, _, .clamp, [rx, ra, rb] => do
      let a ← constAt? G P ra (.frozen [])
      let b ← constAt? G P rb (.frozen [])
      box? boxes rx >>= boxUnaryEnclosure? (Op.clampBounds (Tensor.item a) (Tensor.item b))
  | _, _, .gelu _, [r] => box? boxes r >>= boxUnaryEnclosure? Op.zeroHullBounds
  | _, _, .sinh, [r] => box? boxes r >>= boxUnaryEnclosure? Op.sinhBounds
  | _, _, .cosh, [r] => box? boxes r >>= boxUnaryEnclosure? Op.coshBounds
  | _, _, .logSoftmax (s := s) dim, [r] =>
      box? boxes r >>= Op.coordwiseBox? (s := s) (t := s) (Op.logSoftmaxEntry dim)
  | _, _, .cumsum (s := s) dim, [r] =>
      box? boxes r >>= Op.coordwiseBox? (s := s) (t := s) (Op.cumsumBounds dim)
  | _, _, .select, [_, rx, ry] => do Op.selectBox? (← box? boxes rx) (← box? boxes ry)
  | _, _, .pad (s := s) low high interior, [rx, rv] => do
      Op.padBox? (s := s) low high interior (← box? boxes rx) (← box? boxes rv)
  | _, _, .gather _ _ h, [rx, _] => box? boxes rx >>= Op.gatherBox? h
  | _, _, .embedding (num := num) (dim := dim) (t := t), [rx, _] =>
      box? boxes rx >>= Op.gatherBox? (s := [num, dim]) (t := t)
        (dims := Shape.selectDims [num, dim] ⟨0, Nat.zero_lt_two⟩)
        (batchDims := Shape.noBatch [num, dim] t)
        ⟨List.nodup_singleton _, List.nodup_nil, fun _ h => nomatch h⟩
  | _, _, op, refs => Op.elementwiseBox? boxes op refs

/-- The box of operator `o` on references `refs` by the engine's rules for the operators outside
`shared?`: tanh, sigmoid, sine and cosine by the backend's enclosures, and the operators of kind
`directed` in `ruleKind`. It reads the boxes the references name and the constants of its parameter
references. -/
def directed? (G : IR.Graph) (P : IR.Payload α) (boxes : Array (Option (FlatBox α))) :
    IR.Operator → List IR.Ref → Option (FlatBox α)
  | ⟨_, _, .tanh⟩, [r] => box? boxes r >>= boxUnaryEnclosure? NonlinearBoundOps.tanhBounds
  | ⟨_, _, .sigmoid⟩, [r] => box? boxes r >>= boxUnaryEnclosure? NonlinearBoundOps.sigmoidBounds
  | ⟨_, _, .sin⟩, [r] => box? boxes r >>= boxUnaryEnclosure? NonlinearBoundOps.sinBounds
  | ⟨_, _, .cos⟩, [r] => box? boxes r >>= boxUnaryEnclosure? NonlinearBoundOps.cosBounds
  | ⟨_, _, .exp⟩, [r] => box? boxes r >>= boxUnaryEnclosure? NonlinearBoundOps.expBounds
  | ⟨_, _, .log⟩, [r] => do
      let B ← box? boxes r
      if (List.finRange B.dim).all fun i => decide (0 < B.lo.getScalar i) then
        boxUnaryEnclosure? NonlinearBoundOps.logBounds B
      else none
  | ⟨_, _, .sqrt _⟩, [r] => box? boxes r >>= boxSqrt?
  | ⟨_, _, .reciprocal⟩, [r] => box? boxes r >>= boxInv?
  | ⟨_, _, .abs⟩, [r] => (box? boxes r).map boxAbs
  | ⟨_, _, .neg⟩, [r] => (box? boxes r).map boxNeg
  | ⟨_, _, .square⟩, [r] => (box? boxes r).map boxSquare
  | ⟨_, _, .max⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some B₁, some B₂ => if B₁.dim = B₂.dim then some (boxMaxElem B₁ B₂) else none
      | _, _ => none
  | ⟨_, _, .min⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some B₁, some B₂ => if B₁.dim = B₂.dim then some (boxMinElem B₁ B₂) else none
      | _, _ => none
  | ⟨_, _, .reshape _⟩, [r] => box? boxes r
  | ⟨_, _, op@(.permute _ _ _)⟩, [r] => box? boxes r >>= endpointsBox? op
  | ⟨_, _, op@(.slice _ _ _ _ _)⟩, [r] => box? boxes r >>= endpointsBox? op
  | ⟨_, _, op@(.flip _)⟩, [r] => box? boxes r >>= endpointsBox? op
  | ⟨_, _, op@(.broadcast _ _)⟩, [r] => box? boxes r >>= endpointsBox? op
  | ⟨_, _, op@(.indexAt _ _)⟩, [r] => box? boxes r >>= endpointsBox? op
  | ⟨_, _, .pool op dims kernel stride padLow padHigh dilation ceilMode⟩, [r] =>
      box? boxes r >>= poolBox? op dims kernel stride padLow padHigh dilation ceilMode
  | ⟨_, _, op@(.reduce .max _ _ _)⟩, [r] => box? boxes r >>= endpointsBox? op
  | ⟨_, _, op@(.reduce .min _ _ _)⟩, [r] => box? boxes r >>= endpointsBox? op
  | ⟨_, _, .reduce (s := s) op dims keepdim _⟩, [r] =>
      box? boxes r >>= ibpReduce? op (dims.map Fin.val).toArray keepdim s
  | ⟨_, _, .mean (s := s) dims keepdim⟩, [r] =>
      box? boxes r >>= ibpMean? (dims.map Fin.val).toArray keepdim s
  | ⟨_, _, .softmax (s := s) dim⟩, [r] => do
      let B ← box? boxes r
      finiteBox? B
      if B.dim = s.size then some (ibpSoftmaxRange s dim.val B.dim) else none
  | ⟨_, _, .maskedSoftmax (s := s) dim⟩, [r, rm] => do
      let B ← box? boxes r
      let allowed ← constAt? G P rm (.bool s)
      finiteBox? B
      if dim.val + 1 = s.rank then maskedSoftmaxBox? s allowed B else none
  | ⟨_, _, .layerNorm (seqLen := rows) (embedDim := w) _ _ _ _⟩, [rx, rg, rb, re] => do
      let gamma ← constAt? G P rg (.value [w])
      let beta ← constAt? G P rb (.value [w])
      let epsilon ← constAt? G P re (.frozen [])
      let B ← box? boxes rx
      layerNormBox? rows w gamma beta (Tensor.item epsilon) B
  | ⟨_, _, .batchNormRunning (s := s) dim⟩, [rx, rg, rb, rm, rv, re] => do
      let gamma ← constAt? G P rg (.value [s.extent dim])
      let beta ← constAt? G P rb (.value [s.extent dim])
      let mean ← constAt? G P rm (.frozen [s.extent dim])
      let var ← constAt? G P rv (.frozen [s.extent dim])
      let epsilon ← constAt? G P re (.frozen [])
      let B ← box? boxes rx
      let d ← Tensor.sequenceFinM fun k : Fin (s.extent dim) =>
        batchNormDenominator? (mean.getScalar k) (var.getScalar k) (gamma.getScalar k)
          (beta.getScalar k) (Tensor.item epsilon)
      coordwiseBox? (s := s) (t := s) (fun lo hi c =>
        let k := Shape.Coord.get c dim
        batchNormBounds (lo c) (hi c) (mean.getScalar k) (gamma.getScalar k)
          (beta.getScalar k) (d k)) B
  | ⟨_, _, .mseLoss⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some y, some t => if y.dim = t.dim then boxMean? (boxSquare (boxSub y t)) else none
      | _, _ => none
  | ⟨_, _, .randUniform (s := s) _⟩, [] => some (unitBox s.size)
  | ⟨_, _, .bernoulliMask (s := s) _⟩, [_] => some (unitBox s.size)
  | ⟨_, _, op⟩, refs => scalarParamBox? G P boxes op refs

/-- The box of operator `o` on references `refs`: `shared?`, and `directed?` where that gives
none. -/
def ibp? (G : IR.Graph) (P : IR.Payload α) (boxes : Array (Option (FlatBox α)))
    (o : IR.Operator) (refs : List IR.Ref) : Option (FlatBox α) :=
  match shared? G P boxes o refs with
  | some B => some B
  | none => directed? G P boxes o refs

/-- The boxes of every node of the graph, computed node by node in order. -/
@[noinline, nospecialize]
def runIBP (G : IR.Graph) (P : IR.Payload α) (seeds : Nat → Option (FlatBox α)) :
    Array (Option (FlatBox α)) :=
  tabulate (step? G P (ibp? G P) seeds) G.nodes.size

end TorchLean.CROWN.Op
