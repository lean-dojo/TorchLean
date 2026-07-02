/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core
public import NN.IR.Payload

/-!
Shared definitions for the graph CROWN engine.

This file contains the flat vector representation, parameter stores, interval boxes, shape
permutation helpers, and tensor casts used by the IBP, derivative, affine, CROWN, and backward
objective passes.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/--
Flat vector pack: a tensor paired with its flattened dimension.

This is used for constant payloads and objective coefficient vectors in the flat LiRPA engine.
-/
structure FlatVec (α : Type) [Context α] where
  /-- Vector dimension. -/
  n : Nat
  /-- Vector payload (shape `.dim n .scalar`). -/
  v : Tensor α (.dim n .scalar)

-- The flat-vector engine is the canonical executable path for the current graph verifier.

/--
Parameters for a linear layer `y = W*x + b` in flattened form.

`m` is the output dimension and `n` is the input dimension.
-/
structure LinParams (α : Type) [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α (.dim m (.dim n .scalar))
  /-- Bias vector `b` (shape `m`). -/
  b : Tensor α (.dim m .scalar)

/-- Matrix parameters for bias-free matmul: y = W x. -/
structure MatParams (α : Type) [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α (.dim m (.dim n .scalar))

/-- Conv2D parameters with cached spatial dimensions for graph propagation. -/
abbrev Conv2DParams (α : Type) [Context α] : Type :=
  NN.IR.Conv2DParams α

/-- Eval-mode BatchNorm2d parameters for an `N×C×H×W` node. -/
abbrev BatchNorm2DNchwEvalParams (α : Type) [Context α] : Type :=
  NN.IR.BatchNorm2DNchwEvalParams α

/-- Channel index for a flattened `N×C×H×W` tensor in row-major order. -/
def nchwChannelOfFlat (c h w idx : Nat) : Nat :=
  if h * w = 0 then
    0
  else
    (idx / (h * w)) % c

/-- Eval BatchNorm scale for one channel. -/
def batchNorm2dNchwEvalScale (cfg : BatchNorm2DNchwEvalParams α) (ci : Fin cfg.c) : α :=
  match getAtSpec cfg.gamma ci, getAtSpec cfg.var ci with
  | .scalar gamma, .scalar var =>
      gamma / MathFunctions.sqrt (max var Numbers.zero + cfg.eps)

/-- Eval BatchNorm bias for one channel after folding running statistics into an affine map. -/
def batchNorm2dNchwEvalBias (cfg : BatchNorm2DNchwEvalParams α) (ci : Fin cfg.c) : α :=
  match getAtSpec cfg.beta ci, getAtSpec cfg.mean ci with
  | .scalar beta, .scalar mean =>
      beta - mean * batchNorm2dNchwEvalScale (α := α) cfg ci

/--
Build the exact diagonal affine form for eval-mode BatchNorm2d over an `N×C×H×W` tensor.

The IR stores the channel parameters in the node payload. The spatial dimensions come from the
checked parent shape, so malformed shapes simply do not produce a verifier transfer rule.
-/
def batchNorm2dNchwEvalLinear? (parentShape : Shape)
    (cfg : BatchNorm2DNchwEvalParams α) : Option (LinParams α) :=
  match parentShape with
  | .dim _n (.dim c (.dim h (.dim w .scalar))) =>
      if hcfg : cfg.c = 0 then
        none
      else if c = cfg.c then
        haveI : NeZero cfg.c := ⟨hcfg⟩
        let outDim := parentShape.size
        let weight : Tensor α (.dim outDim (.dim outDim .scalar)) :=
          Tensor.dim (fun oi =>
            Tensor.dim (fun ii =>
              let ch := nchwChannelOfFlat cfg.c h w oi.val
              let scale := batchNorm2dNchwEvalScale (α := α) cfg (Fin.ofNat cfg.c ch)
              Tensor.scalar (if decide (oi.val = ii.val) then scale else Numbers.zero)))
        let bias : Tensor α (.dim outDim .scalar) :=
          Tensor.dim (fun oi =>
            let ch := nchwChannelOfFlat cfg.c h w oi.val
            Tensor.scalar (batchNorm2dNchwEvalBias (α := α) cfg (Fin.ofNat cfg.c ch)))
        some { m := outDim, n := outDim, w := weight, b := bias }
      else
        none
  | _ => none

/--
Parameters keyed by node id (weights, biases, constants, and seeded input boxes).

This is kept compact: it is the graph interpreter used to run IBP/CROWN on a pure `Graph`
without pulling in a heavyweight runtime.
-/
structure ParamStore (α : Type) [Context α] where
  /-- Seed boxes for designated input nodes (`id -> FlatBox`). -/
  inputBoxes : Std.HashMap Nat (FlatBox α) := Std.HashMap.emptyWithCapacity
  /-- Constants (`id -> FlatVec`). -/
  constVals  : Std.HashMap Nat (FlatVec α) := Std.HashMap.emptyWithCapacity
  /-- Linear layer params (`id -> (W,b)`). -/
  linearWB   : Std.HashMap Nat (LinParams α) := Std.HashMap.emptyWithCapacity
  /-- Matmul params (`id -> W`) for bias-free multiplication. -/
  matmulW    : Std.HashMap Nat (MatParams α) := Std.HashMap.emptyWithCapacity
  /-- Conv2d specs (`id -> conv configuration`). -/
  conv2dCfg  : Std.HashMap Nat (Conv2DParams α) := Std.HashMap.emptyWithCapacity
  /-- Eval-mode BatchNorm2d parameters (`id -> gamma/beta/running stats`). -/
  batchNorm2dNchwEval : Std.HashMap Nat (BatchNorm2DNchwEvalParams α) :=
    Std.HashMap.emptyWithCapacity

namespace ParamStore

/-- Insert an input interval box for a graph node. -/
def seedInputBox {α : Type} [Context α]
    (ps : ParamStore α) (inputId : Nat) (xB : FlatBox α) : ParamStore α :=
  { ps with inputBoxes := ps.inputBoxes.insert inputId xB }

/-- Seed a graph input with a uniform `ℓ∞` box around a shaped tensor. -/
def seedLInfBall {α : Type} [Context α] {s : Shape}
    (ps : ParamStore α) (inputId : Nat) (center : Tensor α s) (eps : α) : ParamStore α :=
  ps.seedInputBox inputId <| FlatBox.lInfBall (α := α) center eps

end ParamStore

/-- Read a node's interval box from an IBP-style result array. -/
def outputBox? {α : Type} [Context α]
    (boxes : Array (Option (FlatBox α))) (outId : Nat) : Except String (FlatBox α) := do
  match boxes[outId]? with
  | some (some outB) => pure outB
  | some none => throw s!"output box missing at node {outId}"
  | none => throw s!"output node {outId} is out of bounds for {boxes.size} boxes"

/-- Default inhabitant for `FlatBox` (a 0-dimensional box at `0`). -/
instance : Inhabited (FlatBox α) where
  default := { dim := 0, lo := Spec.fill (α:=α) 0 (.dim 0 .scalar), hi := Spec.fill (α:=α) 0 (.dim 0
    .scalar) }

/-- Elementwise product of two FlatBoxes (interval product per component). Requires equal dims. -/
@[expose] public def box_mul_elem (B1 B2 : FlatBox α) : Option (FlatBox α) :=
  match B1, B2 with
  | ⟨n1, l1, u1⟩, ⟨n2, l2, u2⟩ =>
    if h : n1 = n2 then
      by
        cases h
        let lo :=
          match l1, u1, l2, u2 with
          | .dim l1, .dim u1, .dim l2, .dim u2 =>
            Tensor.dim (fun i =>
              match l1 i, u1 i, l2 i, u2 i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
                let p1 := BoundOps.mulDown lx ly; let p2 := BoundOps.mulDown lx uy
                let p3 := BoundOps.mulDown ux ly; let p4 := BoundOps.mulDown ux uy
                let m1 := min2 p1 p2
                let m2 := min2 p3 p4
                Tensor.scalar (min2 m1 m2))
        let hi :=
          match l1, u1, l2, u2 with
          | .dim l1, .dim u1, .dim l2, .dim u2 =>
            Tensor.dim (fun i =>
              match l1 i, u1 i, l2 i, u2 i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
                let p1 := BoundOps.mulUp lx ly; let p2 := BoundOps.mulUp lx uy
                let p3 := BoundOps.mulUp ux ly; let p4 := BoundOps.mulUp ux uy
                let m1 := max2 p1 p2
                let m2 := max2 p3 p4
                Tensor.scalar (max2 m1 m2))
        exact some { dim := n1, lo := lo, hi := hi }
    else none

/-- Derivative range for `exp` over a value box; `exp' = exp` is monotone. -/
def derivBoxExp (zB : FlatBox α) : FlatBox α :=
  { dim := zB.dim, lo := Tensor.expSpec zB.lo, hi := Tensor.expSpec zB.hi }

/-- Derivative range for the positive-domain log rule used by derivative propagation. -/
def derivBoxLog (zB : FlatBox α) : FlatBox α :=
  match zB.lo, zB.hi with
  | .dim flo, .dim fhi =>
    let lo := Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar _, .scalar u => Tensor.scalar (Numbers.one / u))
    let hi := Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar _ =>
        let l' := if l > Numbers.epsilon then l else Numbers.epsilon
        Tensor.scalar (Numbers.one / l'))
    { dim := zB.dim, lo := lo, hi := hi }

/-- Chain-rule multiplication for derivative intervals. Returns `none` on dimension mismatch. -/
def chainMul (dZ dF : FlatBox α) : Option (FlatBox α) :=
  box_mul_elem (α:=α) dZ dF

/-- Convert a dependent `Box` of shape `.dim n .scalar` into a `FlatBox` with `dim := n`. -/
@[expose]
def toFlatBox (n : Nat) (B : Box α (.dim n .scalar)) : FlatBox α :=
  { dim := n, lo := B.lo, hi := B.hi }

/-- Convert a `FlatBox` to a dependent `Box` at shape `.dim B.dim .scalar`. -/
@[expose]
public def ofFlatBox (B : FlatBox α) : Box α (.dim B.dim .scalar) :=
  { lo := B.lo, hi := B.hi }

/-- Add two flat interval boxes coordinatewise; dimension mismatches preserve the left box. -/
@[expose]
public def box_add (B1 B2 : FlatBox α) : FlatBox α :=
  match B1 with
  | ⟨n1, lo1, hi1⟩ =>
    match B2 with
    | ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact
            { dim := n1
              lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
              hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Interval subtraction on `FlatBox` endpoints (sound enclosure). -/
@[expose]
public def box_sub (B1 B2 : FlatBox α) : FlatBox α :=
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
              lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
              hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Apply ReLU to both endpoints of a `FlatBox` (monotone activation, so endpoints suffice). -/
@[expose]
public def box_relu (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.lo
    hi := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.hi }

/-- Componentwise absolute value bounds. Soundly encloses `abs` over each interval component. -/
def boxAbs (B : FlatBox α) : FlatBox α :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
      let lo' :=
        Tensor.dim (fun i =>
          match lo i, hi i with
          | .scalar l, .scalar u =>
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              let minAbs :=
                if l < Numbers.zero then
                  if Numbers.zero < u then Numbers.zero else (if al < au then al else au)
                else
                  if al < au then al else au
              Tensor.scalar minAbs)
      let hi' :=
        Tensor.dim (fun i =>
          match lo i, hi i with
          | .scalar l, .scalar u =>
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              let maxAbs := if al > au then al else au
              Tensor.scalar maxAbs)
      { dim := B.dim, lo := lo', hi := hi' }

/-- Componentwise sqrt bounds. Uses the spec semantics `sqrt(max(x,0))`, which is monotone. -/
def boxSqrt (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.sqrtSpec (α := α) B.lo
    hi := Tensor.sqrtSpec (α := α) B.hi }

/-- Componentwise reciprocal bounds using `operators.arithmetic.ibp_reciprocal`. -/
@[expose]
def boxInv (B : FlatBox α) : FlatBox α :=
  let yB := NN.MLTheory.CROWN.Operators.Arithmetic.ibpReciprocal (α := α) (n := B.dim) (ofFlatBox
    B)
  toFlatBox B.dim yB

/-- Dynamic tensor value used while reshaping and permuting flattened boxes. -/
abbrev FlatDVal (α : Type) [Context α] : Type :=
  Σ s : Shape, Tensor α s

/-- Shape projection for `FlatDVal`. -/
def flatDValShape {α : Type} [Context α] (v : FlatDVal α) : Shape := v.1

/-- Tensor projection for `FlatDVal`, preserving the dependent shape stored beside it. -/
def flatDValTensor {α : Type} [Context α] (v : FlatDVal α) :
    Tensor α (flatDValShape (α := α) v) := v.2

/-- Return the position of `x` in a list, if present. -/
def findIndex? (xs : List Nat) (x : Nat) : Option Nat :=
  let rec go (i : Nat) : List Nat → Option Nat
    | [] => none
    | y :: ys => if y = x then some i else go (i + 1) ys
  go 0 xs

/-- Swap adjacent entries `d` and `d+1` in a list of axis ids. -/
def swapAt (xs : List Nat) (d : Nat) : List Nat :=
  match xs, d with
  | [], _ => []
  | [x], _ => [x]
  | x :: y :: rest, 0 => y :: x :: rest
  | x :: rest, d + 1 => x :: swapAt rest d

/-- Decompose an axis permutation into adjacent swaps, rejecting invalid permutations. -/
def swapDepthsForPerm? (perm : List Nat) (r : Nat) : Option (List Nat) :=
  let rec bubbleLeft (cur : List Nat) (swapsRev : List Nat) (i j : Nat) : List Nat × List Nat :=
    if j ≤ i then
      (cur, swapsRev)
    else
      bubbleLeft (swapAt cur (j - 1)) ((j - 1) :: swapsRev) i (j - 1)
  if perm.length = r && perm.all (fun d => d < r) then
    let rec go (i : Nat) (targets : List Nat) (cur : List Nat) (swapsRev : List Nat) :
        Option (List Nat) :=
      match targets with
      | [] => some swapsRev.reverse
      | target :: targets' =>
          match findIndex? cur target with
          | none => none
          | some j =>
              let (cur', swapsRev') := bubbleLeft cur swapsRev i j
              go (i + 1) targets' cur' swapsRev'
    go 0 perm (List.range r) []
  else
    none

/-- Apply one adjacent-axis swap to a dynamic tensor value. -/
def applySwapDepth {α : Type} [Context α] (v : FlatDVal α) (d : Nat) : FlatDVal α :=
  match v with
  | ⟨s, t⟩ =>
      let t' : Tensor α (s.swapAdjacentAtDepth d) := Tensor.swapAtDepthHelper (tensor := t) d
      ⟨s.swapAdjacentAtDepth d, t'⟩

/-- Apply a full axis permutation to a dynamic tensor value when the permutation is valid. -/
def permuteDVal? {α : Type} [Context α] (v : FlatDVal α) (perm : List Nat) :
    Option (FlatDVal α) :=
  let sIn := flatDValShape (α := α) v
  match Spec.Shape.permute? sIn perm with
  | none => none
  | some _ =>
      match swapDepthsForPerm? perm (Shape.rank sIn) with
      | none => none
      | some swaps => some <| swaps.foldl (fun acc d => applySwapDepth (α := α) acc d) v

/-- Componentwise max bounds: `max(x,y)` over interval boxes. -/
def boxMaxElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.maxSpec (α := α) lo1 lo2
                  hi := Tensor.maxSpec (α := α) hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Componentwise min bounds: `min(x,y)` over interval boxes. -/
def boxMinElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.minSpec (α := α) lo1 lo2
                  hi := Tensor.minSpec (α := α) hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/--
Componentwise square of an interval box: for each component `[l,u]` produce `[min (l^2,u^2), max
  (l^2,u^2)]`, with `0` as the minimum when the interval crosses `0`.

The body is exposed because the proof-facing theorem module unfolds this executable rule when
proving dimension preservation and pointwise enclosure.
-/
@[expose] def boxSquare (B : FlatBox α) : FlatBox α :=
  let loF : Fin B.dim → Tensor α .scalar :=
    match B.lo with
    | .dim f => f
  let hiF : Fin B.dim → Tensor α .scalar :=
    match B.hi with
    | .dim f => f
  let lo' :=
    Tensor.dim (fun i =>
      match loF i, hiF i with
      | .scalar l, .scalar u =>
        let l2 := l * l
        let u2 := u * u
        let minSq :=
          if l < Numbers.zero then
            if Numbers.zero < u then Numbers.zero else (if l2 < u2 then l2 else u2)
          else (if l2 < u2 then l2 else u2)
        Tensor.scalar minSq)
  let hi' :=
    Tensor.dim (fun i =>
      match loF i, hiF i with
      | .scalar l, .scalar u =>
        let l2 := l * l
        let u2 := u * u
        let maxSq := if l2 > u2 then l2 else u2
        Tensor.scalar maxSq)
  { dim := B.dim, lo := lo', hi := hi' }

/-- Interval multiplication for scalar endpoints: given `[aLo,aHi]` and `[bLo,bHi]`, return bounds
  on the product. -/
def intervalMul (aLo aHi bLo bHi : α) : α × α :=
  let p1 := aLo * bLo
  let p2 := aLo * bHi
  let p3 := aHi * bLo
  let p4 := aHi * bHi
  let lo1 := if p1 < p2 then p1 else p2
  let lo2 := if p3 < p4 then p3 else p4
  let lo  := if lo1 < lo2 then lo1 else lo2
  let hi1 := if p1 > p2 then p1 else p2
  let hi2 := if p3 > p4 then p3 else p4
  let hi  := if hi1 > hi2 then hi1 else hi2
  (lo, hi)

/-- Length of the last axis of a shape; scalars are treated as length one. -/
def lastDimLen : Shape → Nat
  | .scalar => 1
  | .dim n .scalar => n
  | .dim _ rest => lastDimLen rest

/-- Runtime witness that an axis is valid for a shape. -/
def mkValidAxis? (axis : Nat) : (s : Shape) → Option (PLift (Shape.valid_axis axis s))
  | .scalar => none
  | .dim n rest =>
      match axis, n with
      | 0, Nat.succ k => some ⟨Shape.valid_axis.valid_zero (n := k) (s := rest)⟩
      | 0, 0 => none
      | Nat.succ a, Nat.succ k =>
          (mkValidAxis? a rest).map (fun h => ⟨Shape.valid_axis.valid_succ (n := k) (s := rest) (k
            := a) h.down⟩)
      | Nat.succ _, 0 => none

/-- Runtime witness that one shape can broadcast to another. -/
def mkCanBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
  | s₁, s₂ =>
    if Shape.rank s₁ < Shape.rank s₂ then
      match s₂ with
      | .scalar => none
      | .dim n₂ t₂ =>
        (mkCanBroadcastTo? s₁ t₂).map (fun tail =>
          Shape.CanBroadcastTo.expand_dims (n := n₂) (s₁ := s₁) (s₂ := t₂) tail)
    else if Shape.rank s₂ < Shape.rank s₁ then
      none
    else
      match s₁, s₂ with
      | .scalar, s₂ => some (.scalar_to_any s₂)
      | .dim n₁ t₁, .dim n₂ t₂ =>
          if hEq : n₁ = n₂ then
            (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
              hEq ▸ Shape.CanBroadcastTo.dim_eq (n := n₁) (s₁ := t₁) (s₂ := t₂) tail)
          else if h1 : n₁ = 1 then
            (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
              h1 ▸ Shape.CanBroadcastTo.dim_1_to_n (n := n₂) (s₁ := t₁) (s₂ := t₂) tail)
          else
            none
      | _, _ => none

/-- Reinterpret a flattened tensor as shape `s` when the element counts agree. -/
def ibpUnflatten {s : Shape} (dim : Nat) (t : Tensor α (.dim dim .scalar)) (h : dim =
  Shape.size s) :
    Tensor α s :=
  let t' : Tensor α (.dim (Shape.size s) .scalar) := by
    simpa [h] using t
  Tensor.unflattenSpec (α := α) s t'

/-- IBP rule for broadcasting a flattened input box to a target shape. -/
def ibpBroadcastTo (s₁ s₂ : Shape) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = Shape.size s₁ then
    match mkCanBroadcastTo? s₁ s₂ with
    | none => none
    | some cb =>
        let xLo : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.lo h
        let xHi : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.hi h
        let yLo : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xLo
        let yHi : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xHi
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Shape.size s₂, lo := flatLo, hi := flatHi }
  else
    none

/-- IBP rule for reducing a shaped box by summing along one axis. -/
def ibpReduceSumAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Shape.size s then
    match mkValidAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := Shape.proveReducibleAlong axis s hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceSum (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceSum (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/-- IBP rule for reducing a shaped box by averaging along one axis. -/
def ibpReduceMeanAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Shape.size s then
    match mkValidAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := Shape.proveReducibleAlong axis s hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceMean (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceMean (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/-!
## Softmax IBP (last axis)

For a 1D vector `x` with interval bounds `l <= x <= u`, a standard componentwise enclosure for
softmax is:

* `softmax_i(x) = exp(x_i) / sum_j exp(x_j)`
* Lower bound (worst-case denominator): `exp(l_i) / (exp(l_i) + sum_{j != i} exp(u_j))`
* Upper bound (best-case denominator): `exp(u_i) / (exp(u_i) + sum_{j != i} exp(l_j))`

This uses monotonicity of `exp` and the fact that all terms in the denominator are nonnegative.
The implementation below applies the 1D rule on the last tensor axis and recurses over leading batch
dimensions.

References:
- CROWN / DeepPoly context: Zhang et al., 2018 (CROWN): https://arxiv.org/abs/1811.00866
- auto_LiRPA: Xu et al., 2020: https://arxiv.org/abs/2002.12920
-/

/-- Interval bound propagation for `softmax`, applied on the last axis and lifted over leading dims.
  -/
def ibpSoftmaxLastTensor : {s : Shape} → Tensor α s → Tensor α s → (Tensor α s × Tensor α s)
  | .scalar, _lo, _hi => (Tensor.scalar Numbers.one, Tensor.scalar Numbers.one)
  | .dim _n .scalar, lo, hi =>
      -- Tighter IBP for softmax on a 1D vector:
      --  lower: exp(l_i) / (exp(l_i) + Σ_{j≠i} exp(u_j))
      --  upper: exp(u_i) / (exp(u_i) + Σ_{j≠i} exp(l_j))
      let exp_lo := Tensor.expSpec lo
      let exp_hi := Tensor.expSpec hi
      let total_lo := Spec.Tensor.sumSpec exp_lo
      let total_hi := Spec.Tensor.sumSpec exp_hi
      match exp_lo, exp_hi with
      | .dim elo, .dim ehi =>
        let outLo :=
          Tensor.dim (fun i =>
            match elo i, ehi i with
            | .scalar e_li, .scalar e_ui =>
              let denom := e_li + (total_hi - e_ui)
              Tensor.scalar (e_li / denom))
        let outHi :=
          Tensor.dim (fun i =>
            match elo i, ehi i with
            | .scalar e_li, .scalar e_ui =>
              let denom := e_ui + (total_lo - e_li)
              Tensor.scalar (e_ui / denom))
        (outLo, outHi)
  | .dim n inner, Tensor.dim loF, Tensor.dim hiF =>
      let outLo := Tensor.dim (fun i : Fin n => (ibpSoftmaxLastTensor (s := inner) (loF i) (hiF
        i)).1)
      let outHi := Tensor.dim (fun i : Fin n => (ibpSoftmaxLastTensor (s := inner) (loF i) (hiF
        i)).2)
      (outLo, outHi)

/-!
## LayerNorm IBP (last axis)

Layer normalization (Ba et al.) computes, per vector, something like:

`y = (x - mean(x)) / sqrt(var(x) + eps)`.

We implement a conservative enclosure by:
1. Bounding mean using sums of endpoints.
2. Bounding variance using a max-deviation upper bound.
3. Bounding the per-component ratio by checking endpoint combinations against a positive denominator
   interval.

This is intended as a simple checker-side transfer rule. It is conservative and is not an
optimized relaxation.

References:
- Ba, Kiros, Hinton, "Layer Normalization", 2016: https://arxiv.org/abs/1607.06450
- Bound propagation context: Xu et al., 2020 (auto_LiRPA): https://arxiv.org/abs/2002.12920
-/

/--
Upper bound on the variance term used by the LayerNorm interval rules.

Given endpoint bounds for a vector and bounds on its mean, each coordinate is at most
`max |x_i - μ|` away from the bounded mean interval. Squaring and summing those coordinate radii
gives the conservative variance upper bound reused by IBP, affine propagation, and derivative
interval passes.
-/
def layerNormVarianceUpper {n : Nat}
    (lo hi : Tensor α (.dim n .scalar)) (muLo muHi : α) : α :=
  if _h : n > 0 then
    match lo, hi with
    | .dim flo, .dim fhi =>
        let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let dl := MathFunctions.abs (l - muHi)
            let du := MathFunctions.abs (u - muLo)
            let a := if dl > du then dl else du
            acc + (a * a)) 0
        sumAbsSq / (n : Nat)
  else
    Numbers.zero

/-- Mean bounds for a nonempty vector whose coordinates are bounded by endpoint tensors.

For `n = 0`, the mathematical mean is undefined; this total helper returns `(0,0)` so callers do
not accidentally divide by zero while they reject or totalize the empty case.
-/
def layerNormMeanBounds {n : Nat}
    (lo hi : Tensor α (.dim n .scalar)) : α × α :=
  if _h : n > 0 then
    let nA : α := (n : Nat)
    (Spec.Tensor.sumSpec lo / nA, Spec.Tensor.sumSpec hi / nA)
  else
    (Numbers.zero, Numbers.zero)

/--
Bounds for `x - μ` when `x` is bounded coordinatewise and `μ` is bounded by an interval.

LayerNorm transfer rules repeatedly need this centered interval for the input, first derivative,
and second derivative streams. Keeping it here avoids duplicating the same endpoint arithmetic in
IBP and derivative propagation.
-/
def layerNormCenteredBounds {n : Nat}
    (lo hi : Tensor α (.dim n .scalar)) (muLo muHi : α) :
    Tensor α (.dim n .scalar) × Tensor α (.dim n .scalar) :=
  let flo := match lo with | .dim f => f
  let fhi := match hi with | .dim f => f
  let loOut :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let dl := l - muHi
        let du := u - muLo
        Tensor.scalar (if dl < du then dl else du))
  let hiOut :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let dl := l - muHi
        let du := u - muLo
        Tensor.scalar (if dl > du then dl else du))
  (loOut, hiOut)

/-- Bounds for the reciprocal LayerNorm denominator from an upper variance bound. -/
def layerNormInvStdBounds (varHi : α) : α × α :=
  let sLo := MathFunctions.sqrt Numbers.epsilon
  let sHi := MathFunctions.sqrt (varHi + Numbers.epsilon)
  (Numbers.one / (if sHi > Numbers.epsilon then sHi else Numbers.epsilon),
   Numbers.one / (if sLo > Numbers.epsilon then sLo else Numbers.epsilon))

/-- Interval bound propagation for `layernorm`, applied on the last axis and lifted over leading
  dims. -/
def ibpLayernormLastTensor : {s : Shape} → Tensor α s → Tensor α s → (Tensor α s × Tensor α s)
  | .scalar, lo, hi => (lo, hi)
  | .dim n .scalar, lo, hi =>
      if n > 0 then
        let nA : α := (n : Nat)
        let sum_lo := Spec.Tensor.sumSpec lo
        let sum_hi := Spec.Tensor.sumSpec hi
        let mu_lo := sum_lo / nA
        let mu_hi := sum_hi / nA
        let flo := match lo with | .dim f => f
        let fhi := match hi with | .dim f => f
        let var_hi := layerNormVarianceUpper (α := α) lo hi mu_lo mu_hi
        let den_lo := MathFunctions.sqrt Numbers.epsilon
        let den_hi := MathFunctions.sqrt (var_hi + Numbers.epsilon)
        let outLo :=
          Tensor.dim (fun i =>
            match flo i, fhi i with
            | .scalar l, .scalar u =>
              let dl := l - mu_hi
              let du := u - mu_lo
              -- For positive denom interval [den_lo, den_hi], bound (x/denom) by checking all
              -- endpoint ratios.
              let c1 := dl / den_lo
              let c2 := dl / den_hi
              let c3 := du / den_lo
              let c4 := du / den_hi
              let mn12 := if c1 < c2 then c1 else c2
              let mn34 := if c3 < c4 then c3 else c4
              let mn := if mn12 < mn34 then mn12 else mn34
              Tensor.scalar mn)
        let outHi :=
          Tensor.dim (fun i =>
            match flo i, fhi i with
            | .scalar l, .scalar u =>
              let dl := l - mu_hi
              let du := u - mu_lo
              let c1 := dl / den_lo
              let c2 := dl / den_hi
              let c3 := du / den_lo
              let c4 := du / den_hi
              let mx12 := if c1 > c2 then c1 else c2
              let mx34 := if c3 > c4 then c3 else c4
              let mx := if mx12 > mx34 then mx12 else mx34
              Tensor.scalar mx)
        (outLo, outHi)
      else
        -- Degenerate n=0: pass through
        (lo, hi)
  | .dim n inner, Tensor.dim loF, Tensor.dim hiF =>
      let outLo := Tensor.dim (fun i : Fin n => (ibpLayernormLastTensor (s := inner) (loF i) (hiF
        i)).1)
      let outHi := Tensor.dim (fun i : Fin n => (ibpLayernormLastTensor (s := inner) (loF i) (hiF
        i)).2)
      (outLo, outHi)

/-- For tensors known to have shape `.dim n .scalar`, extract the underlying function. -/
@[expose] public def getDimScalarFn {n : Nat} (t : Tensor α (.dim n .scalar)) : (Fin n → Tensor α
  .scalar) :=
  match t with
  | .dim f => f

-- Casting helpers for dependent shapes
/-- Cast a 1D `Box` along an equality of dimensions. -/
@[expose]
public def castBoxDim {n n' : Nat}
  (h : n = n')
  (B : Box α (.dim n .scalar)) : Box α (.dim n' .scalar) := by
  simpa [h] using B

/-- Cast a ReLU relaxation vector across a proven-equal hidden dimension. -/
def castRelax {n n' : Nat}
  (h : n = n')
  (r : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim n .scalar)) :
  Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim n' .scalar) := by
  simpa [h] using r

/-- Cast the input dimension of an affine map across a proven equality. -/
def castAffineIn {n n' m : Nat}
  (h : n = n') (a : AffineVec α n m) : AffineVec α n' m := by
  simpa [h] using a

/-- Cast the output dimension of an affine map across a proven equality. -/
@[expose]
def castAffineOut {n m m' : Nat}
  (h : m = m') (a : AffineVec α n m) : AffineVec α n m' := by
  simpa [h] using a

/--
Cast a dim-scalar tensor across an equality of dimensions.

We keep this as an `abbrev` so it unfolds aggressively in simp-based soundness proofs.
-/
abbrev castDimScalar {n n' : Nat}
  (h : n = n') (t : Tensor α (.dim n .scalar)) : Tensor α (.dim n' .scalar) := by
  simpa [h] using t

/-- IBP propagation through explicit linear parameters. -/
@[expose]
public def ibpLinearParams (p : LinParams α) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = p.n then
    let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
    let bBox : Box α (.dim p.m .scalar) := Box.point (α:=α) p.b
    let yB   := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB bBox
    -- Materialize to avoid deep closure chains in multi-layer verifier runs.
    let yB' : Box α (.dim p.m .scalar) :=
      { lo := Tensor.materialize yB.lo
        hi := Tensor.materialize yB.hi }
    some (toFlatBox p.m yB')
  else none

/-- IBP propagation for a `.linear` node using `ParamStore.linearWB`. -/
@[expose]
public def ibp_linear (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.linearWB[id]? with
  | none => none
  | some p => ibpLinearParams (α := α) p Xin

/-- IBP propagation for a `.matmul` node (bias-free) using `ParamStore.matmulW`. -/
@[expose]
public def ibp_matmul (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.matmulW[id]? with
  | none => none
  | some p =>
    if h : Xin.dim = p.n then
      let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
      let zeroB : Box α (.dim p.m .scalar) :=
        let z := Spec.fill (α:=α) 0 (.dim p.m .scalar)
        Box.point (α:=α) z
      let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
      -- Materialize to avoid deep closure chains (runtime performance).
      let yB' : Box α (.dim p.m .scalar) :=
        { lo := Tensor.materialize yB.lo
          hi := Tensor.materialize yB.hi }
      some (toFlatBox p.m yB')
    else none

/-- IBP transfer for a convolution node whose parameters are stored in `ParamStore.conv2dCfg`. -/
def ibpConv2dNode (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.conv2dCfg[id]? with
  | none => none
  | some cfg =>
    let expected := cfg.inC * cfg.inH * cfg.inW
    if _hs : cfg.stride = 0 then
      none
    else if hdim : Xin.dim = expected then
      let sFlat := Shape.dim Xin.dim Shape.scalar
      let sIn := Shape.dim cfg.inC (Shape.dim cfg.inH (Shape.dim cfg.inW Shape.scalar))
      have hsize : sFlat.size = sIn.size := by
        simp [Shape.size, sFlat, sIn, hdim, expected, Nat.mul_assoc]
      let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
      let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
      let xBox : Box α sIn := { lo := xLo, hi := xHi }
      let yBox := NN.MLTheory.CROWN.ibpConv2d (α:=α)
        (layer:=cfg.spec) (xB:=xBox)
      let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
      let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
      let outShape := Shape.dim cfg.outC (Shape.dim outH (Shape.dim outW Shape.scalar))
      let flatLo := Tensor.flattenSpec (α:=α) yBox.lo
      let flatHi := Tensor.flattenSpec (α:=α) yBox.hi
      some { dim := outShape.size, lo := flatLo, hi := flatHi }
    else none


end NN.MLTheory.CROWN.Graph
