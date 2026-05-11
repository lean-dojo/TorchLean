/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core

public section


/-
Flat LiRPA engine over flattened vectors
=======================================
This module contains the executable graph engine: flat boxes, parameter stores, forward IBP,
derivative IBP, forward affine CROWN, and objective-dependent backward CROWN.

We keep those executable passes in one implementation module because the backward objective pass
reuses the same affine-transfer helpers as the forward CROWN pass. Keeping those helpers private
is cleaner than exposing a broad internal helper API just to split the file more aggressively.
Proof-facing shape/enclosure facts live separately in `Graph/Theorems`.

The executable engine covers the verifier dialect used by TorchLean certificates, including
input/const/add/sub/relu/reshape/flatten/linear/matmul and selected convolution, normalization,
softmax, and elementwise rules. Operators outside a specific transfer rule are kept conservative:
they either return interval-only state or an explicit `none`, depending on the pass.
-/
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
structure Conv2DParams (α : Type) [Context α] where
  /-- Input channels. -/
  inC : Nat
  /-- Output channels. -/
  outC : Nat
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (shared for height/width). -/
  stride : Nat
  /-- Zero padding (shared for height/width). -/
  padding : Nat
  /-- Input height. -/
  inH : Nat
  /-- Input width. -/
  inW : Nat
  /-- Proof that `inC` is nonzero (required by the typed conv spec). -/
  hIn : inC ≠ 0
  /-- Proof that `kH` is nonzero. -/
  hKH : kH ≠ 0
  /-- Proof that `kW` is nonzero. -/
  hKW : kW ≠ 0
  /-- Typed conv specification payload. -/
  spec : Spec.Conv2DSpec inC outC kH kW stride padding α hIn hKH hKW

/--
Parameters keyed by node id (weights, biases, constants, and seeded input boxes).

This is deliberately compact: it is the graph interpreter used to run IBP/CROWN on a pure `Graph`
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

/-- Default inhabitant for `FlatBox` (a 0-dimensional box at `0`). -/
instance [Context α] : Inhabited (FlatBox α) where
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

private def derivBoxExp (zB : FlatBox α) : FlatBox α :=
  { dim := zB.dim, lo := Tensor.expSpec zB.lo, hi := Tensor.expSpec zB.hi }

private def derivBoxLog (zB : FlatBox α) : FlatBox α :=
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

private def chainMul (dZ dF : FlatBox α) : Option (FlatBox α) :=
  box_mul_elem (α:=α) dZ dF

/-- Convert a dependent `Box` of shape `.dim n .scalar` into a `FlatBox` with `dim := n`. -/
@[expose]
def toFlatBox (n : Nat) (B : Box α (.dim n .scalar)) : FlatBox α :=
  { dim := n, lo := B.lo, hi := B.hi }

/-- Convert a `FlatBox` to a dependent `Box` at shape `.dim B.dim .scalar`. -/
@[expose]
public def ofFlatBox (B : FlatBox α) : Box α (.dim B.dim .scalar) :=
  { lo := B.lo, hi := B.hi }

/-- Basic FlatBox ops -/
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
def boxInv (B : FlatBox α) : FlatBox α :=
  let yB := NN.MLTheory.CROWN.Operators.Arithmetic.ibpReciprocal (α := α) (n := B.dim) (ofFlatBox
    B)
  toFlatBox B.dim yB

private abbrev DVal (α : Type) [Context α] : Type :=
  Σ s : Shape, Tensor α s

private def dvalShape {α : Type} [Context α] (v : DVal α) : Shape := v.1

private def dvalTensor {α : Type} [Context α] (v : DVal α) : Tensor α (dvalShape (α := α) v) := v.2

private def findIndex? (xs : List Nat) (x : Nat) : Option Nat :=
  let rec go (i : Nat) : List Nat → Option Nat
    | [] => none
    | y :: ys => if y = x then some i else go (i + 1) ys
  go 0 xs

private def swapAt (xs : List Nat) (d : Nat) : List Nat :=
  match xs, d with
  | [], _ => []
  | [x], _ => [x]
  | x :: y :: rest, 0 => y :: x :: rest
  | x :: rest, d + 1 => x :: swapAt rest d

private def swapDepthsForPerm? (perm : List Nat) (r : Nat) : Option (List Nat) :=
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

private def applySwapDepth {α : Type} [Context α] (v : DVal α) (d : Nat) : DVal α :=
  match v with
  | ⟨s, t⟩ =>
      let t' : Tensor α (s.swapAdjacentAtDepth d) := Tensor.swapAtDepthHelper (tensor := t) d
      ⟨s.swapAdjacentAtDepth d, t'⟩

private def permuteDVal? {α : Type} [Context α] (v : DVal α) (perm : List Nat) : Option (DVal α) :=
  let sIn := dvalShape (α := α) v
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

private def lastDimLen : Shape → Nat
  | .scalar => 1
  | .dim n .scalar => n
  | .dim _ rest => lastDimLen rest

private def mkValidAxis? (axis : Nat) : (s : Shape) → Option (PLift (Shape.valid_axis axis s))
  | .scalar => none
  | .dim n rest =>
      match axis, n with
      | 0, Nat.succ k => some ⟨Shape.valid_axis.valid_zero (n := k) (s := rest)⟩
      | 0, 0 => none
      | Nat.succ a, Nat.succ k =>
          (mkValidAxis? a rest).map (fun h => ⟨Shape.valid_axis.valid_succ (n := k) (s := rest) (k
            := a) h.down⟩)
      | Nat.succ _, 0 => none

private def mkCanBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
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

private def ibpUnflatten {s : Shape} (dim : Nat) (t : Tensor α (.dim dim .scalar)) (h : dim =
  Shape.size s) :
    Tensor α s :=
  let t' : Tensor α (.dim (Shape.size s) .scalar) := by
    simpa [h] using t
  Tensor.unflattenSpec (α := α) s t'

private def ibpBroadcastTo (s₁ s₂ : Shape) (Xin : FlatBox α) : Option (FlatBox α) :=
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

private def ibpReduceSumAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
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

private def ibpReduceMeanAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
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
        let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let dl := MathFunctions.abs (l - mu_hi)
            let du := MathFunctions.abs (u - mu_lo)
            let a := if dl > du then dl else du
            acc + (a * a)
        ) 0
        let var_hi := sumAbsSq / nA
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

private def castRelax {n n' : Nat}
  (h : n = n')
  (r : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim n .scalar)) :
  Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim n' .scalar) := by
  simpa [h] using r

private def castAffineIn {n n' m : Nat}
  (h : n = n') (a : AffineVec α n m) : AffineVec α n' m := by
  simpa [h] using a

private def castAffineOut {n m m' : Nat}
  (h : m = m') (a : AffineVec α n m) : AffineVec α n m' := by
  simpa [h] using a

/--
Cast a dim-scalar tensor across an equality of dimensions.

We keep this as an `abbrev` so it unfolds aggressively in simp-based soundness proofs.
-/
abbrev castDimScalar {n n' : Nat}
  (h : n = n') (t : Tensor α (.dim n .scalar)) : Tensor α (.dim n' .scalar) := by
  simpa [h] using t

/-- IBP propagation for a `.linear` node using `ParamStore.linearWB`. -/
@[expose]
public def ibp_linear (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.linearWB[id]? with
  | none => none
  | some p =>
    if h : Xin.dim = p.n then
      let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
      let bBox : Box α (.dim p.m .scalar) := Box.point (α:=α) p.b
      let yB   := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB bBox
      -- Materialize to avoid deep closure chains (crucial for runtime performance on
      -- multi-layer networks when tensors are represented functionally).
      let yB' : Box α (.dim p.m .scalar) :=
        { lo := Tensor.materialize yB.lo
          hi := Tensor.materialize yB.hi }
      some (toFlatBox p.m yB')
    else none

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

private def ibpConv2dNode (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.conv2dCfg[id]? with
  | none => none
  | some cfg =>
    let expected := cfg.inC * cfg.inH * cfg.inW
    if hdim : Xin.dim = expected then
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

/-- IBP propagation for one node using `ParamStore`.

This executable function expects parents to have already been processed. The proof layer makes that
precondition explicit via `TopoSorted`; callers that execute graphs directly should use graphs whose
parents appear before their children.
-/
def propagateIBPNode (nodes : Array Node) (ps : ParamStore α) (boxes : Array (Option (FlatBox α)))
  (id : Nat) : Array (Option (FlatBox α)) :=
  let node := nodes[id]!
  let get! (pid : Nat) := (boxes[pid]!).get!
  match node.kind with
  | .input =>
    match ps.inputBoxes[id]? with
    | some B => boxes.set! id (some B)
    | none   => boxes
  | .const _ =>
    match ps.constVals[id]? with
    | some v => boxes.set! id (some { dim := v.n, lo := v.v, hi := v.v })
    | none   => boxes
  | .detach =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (get! p1))
    | _ => boxes
  | .randUniform _ | .bernoulliMask _ =>
    -- Stochastic nodes are treated as *nondeterministic-but-bounded* for verification.
    -- Sound enclosure: U[0,1) ⊆ [0,1], Bernoulli mask ⊆ [0,1].
    let d := node.outShape.size
    let lo := Spec.fill (α := α) Numbers.zero (.dim d .scalar)
    let hi := Spec.fill (α := α) Numbers.one (.dim d .scalar)
    boxes.set! id (some { dim := d, lo := lo, hi := hi })
  | .add =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (box_add (get! p1) (get! p2)))
    | _ => boxes
  | .sub =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (box_sub (get! p1) (get! p2)))
    | _ => boxes
  | .abs =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (boxAbs (α := α) (get! p1)))
    | _ => boxes
  | .sqrt =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (boxSqrt (α := α) (get! p1)))
    | _ => boxes
  | .inv =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (boxInv (α := α) (get! p1)))
    | _ => boxes
  | .maxElem =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (boxMaxElem (α := α) (get! p1) (get! p2)))
    | _ => boxes
  | .minElem =>
    match node.parents with
    | p1 :: p2 :: _ => boxes.set! id (some (boxMinElem (α := α) (get! p1) (get! p2)))
    | _ => boxes
  | .maxPool2d kH kW stride =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                  (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .maxPool2dPad kH kW stride padding =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                  (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .avgPool2d kH kW stride =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                  (h1 := hkH) (h2 := hkW) (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                (h1 := hkH) (h2 := hkW) (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .avgPool2dPad kH kW stride padding =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim inC (.dim inH (.dim inW .scalar)) =>
        let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
        let expectedInDim := sIn.size
        if hIn : Xin.dim = expectedInDim then
            if hkH : kH = 0 then
              boxes
            else if hkW : kW = 0 then
              boxes
            else if hs : stride = 0 then
              boxes
            else
              let sFlat := Shape.dim Xin.dim Shape.scalar
              let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
              have hsize : sFlat.size = sIn.size := by
                simp [sFlat, sIn, expectedInDim, Shape.size, hIn]
              let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
              let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
              let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
              let yLo : Tensor α outShape :=
                Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                  (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                  (h1 := hkH) (h2 := hkW) (layer := layer) (input := xLo)
            let yHi : Tensor α outShape :=
              Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                (h1 := hkH) (h2 := hkW) (layer := layer) (input := xHi)
            let flatLo := Tensor.flattenSpec (α := α) yLo
            let flatHi := Tensor.flattenSpec (α := α) yHi
            boxes.set! id (some { dim := outShape.size, lo := flatLo, hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .broadcastTo s₁ s₂ =>
    match node.parents with
    | p1 :: _ =>
      match ibpBroadcastTo (α := α) s₁ s₂ (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .reduceSum axis =>
    match node.parents with
    | p1 :: _ =>
      let s := nodes[p1]!.outShape
      match ibpReduceSumAxis (α := α) axis (get! p1) s with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .reduceMean axis =>
    match node.parents with
    | p1 :: _ =>
      let s := nodes[p1]!.outShape
      match ibpReduceMeanAxis (α := α) axis (get! p1) s with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .relu =>
    match node.parents with
    | p1 :: _ => boxes.set! id (some (box_relu (get! p1)))
    | _ => boxes
  | .linear =>
    match node.parents with
    | p1 :: _ =>
      match ibp_linear (α:=α) id ps (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none    => boxes
    | _ => boxes
  | .matmul =>
    match node.parents with
    | p1 :: p2 :: _ =>
      let A := get! p1
      let B := get! p2
      let sA := nodes[p1]!.outShape
      let sB := nodes[p2]!.outShape
      let dyn2D? : Option (FlatBox α) :=
        match sA, sB with
        | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
          if hk : k = k' then
            match hk with
            | rfl =>
              if hA : A.dim = m * k then
                if hB : B.dim = k * n then
                  let outDim := m * n
                  let loT : Tensor α (.dim outDim .scalar) :=
                    Tensor.dim (fun idx =>
                      let t := idx.val
                      let i := t / n
                      let j := t % n
                      let (sumLo, _sumHi) :=
                        (List.range k).foldl (fun (acc : α × α) kk =>
                          let (accLo, accHi) := acc
                          let aLo := getAtOrZero A.lo [i * k + kk]
                          let aHi := getAtOrZero A.hi [i * k + kk]
                          let bLo := getAtOrZero B.lo [kk * n + j]
                          let bHi := getAtOrZero B.hi [kk * n + j]
                          let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                          (accLo + pLo, accHi + pHi)
                        ) (0, 0)
                      Tensor.scalar sumLo)
                  let hiT : Tensor α (.dim outDim .scalar) :=
                    Tensor.dim (fun idx =>
                      let t := idx.val
                      let i := t / n
                      let j := t % n
                      let (_sumLo, sumHi) :=
                        (List.range k).foldl (fun (acc : α × α) kk =>
                          let (accLo, accHi) := acc
                          let aLo := getAtOrZero A.lo [i * k + kk]
                          let aHi := getAtOrZero A.hi [i * k + kk]
                          let bLo := getAtOrZero B.lo [kk * n + j]
                          let bHi := getAtOrZero B.hi [kk * n + j]
                          let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                          (accLo + pLo, accHi + pHi)
                        ) (0, 0)
                      Tensor.scalar sumHi)
                  some { dim := outDim, lo := loT, hi := hiT }
                else none
              else none
          else none
        | _, _ => none
      let dyn3D? : Option (FlatBox α) :=
        match sA, sB with
        | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
          if hb : b = b' then
            match hb with
            | rfl =>
              if hk : k = k' then
                match hk with
                | rfl =>
                  if hA : A.dim = b * m * k then
                    if hB : B.dim = b * k * n then
                      let outDim := b * m * n
                      let block : Nat := m * n
                      let strideA : Nat := m * k
                      let strideB : Nat := k * n
                      let loT : Tensor α (.dim outDim .scalar) :=
                        Tensor.dim (fun idx =>
                          let t := idx.val
                          let bi := t / block
                          let rem := t % block
                          let i := rem / n
                          let j := rem % n
                          let baseA := bi * strideA
                          let baseB := bi * strideB
                          let (sumLo, _sumHi) :=
                            (List.range k).foldl (fun (acc : α × α) kk =>
                              let (accLo, accHi) := acc
                              let aLo := getAtOrZero A.lo [baseA + i * k + kk]
                              let aHi := getAtOrZero A.hi [baseA + i * k + kk]
                              let bLo := getAtOrZero B.lo [baseB + kk * n + j]
                              let bHi := getAtOrZero B.hi [baseB + kk * n + j]
                              let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                              (accLo + pLo, accHi + pHi)
                            ) (0, 0)
                          Tensor.scalar sumLo)
                      let hiT : Tensor α (.dim outDim .scalar) :=
                        Tensor.dim (fun idx =>
                          let t := idx.val
                          let bi := t / block
                          let rem := t % block
                          let i := rem / n
                          let j := rem % n
                          let baseA := bi * strideA
                          let baseB := bi * strideB
                          let (_sumLo, sumHi) :=
                            (List.range k).foldl (fun (acc : α × α) kk =>
                              let (accLo, accHi) := acc
                              let aLo := getAtOrZero A.lo [baseA + i * k + kk]
                              let aHi := getAtOrZero A.hi [baseA + i * k + kk]
                              let bLo := getAtOrZero B.lo [baseB + kk * n + j]
                              let bHi := getAtOrZero B.hi [baseB + kk * n + j]
                              let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                              (accLo + pLo, accHi + pHi)
                            ) (0, 0)
                          Tensor.scalar sumHi)
                      some { dim := outDim, lo := loT, hi := hiT }
                    else none
                  else none
              else none
          else none
        | _, _ => none
      match dyn2D?, dyn3D? with
      | some yB, _ => boxes.set! id (some yB)
      | none, some yB => boxes.set! id (some yB)
      | none, none => boxes
    | p1 :: _ =>
      match ibp_matmul (α:=α) id ps (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none    => boxes
    | _ => boxes
  | .reshape _ _ =>
    match node.parents with
    | p1 :: _ => boxes.set! id (boxes[p1]!)
    | _ => boxes
  | .flatten _ =>
    match node.parents with
    | p1 :: _ => boxes.set! id (boxes[p1]!)
    | _ => boxes
  | .swap_first_two =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim m (.dim n rest) =>
        let sIn : Shape := .dim m (.dim n rest)
        if hdim : Xin.dim = sIn.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = sIn.size := by
            simp [sFlat, sIn, Shape.size, hdim]
          let xLo : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
          let xHi : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
          let yLoT : Tensor α (.dim n (.dim m rest)) := Tensor.swapFirstTwoSpec (α:=α) xLo
          let yHiT : Tensor α (.dim n (.dim m rest)) := Tensor.swapFirstTwoSpec (α:=α) xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id (some { dim := (Shape.dim n (Shape.dim m rest)).size, lo := flatLo, hi :=
            flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .transpose3dLastTwo =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match nodes[p1]!.outShape with
      | .dim a (.dim b (.dim c .scalar)) =>
        let sIn : Shape := .dim a (.dim b (.dim c .scalar))
        if hdim : Xin.dim = sIn.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = sIn.size := by
            simp [sFlat, sIn, Shape.size, hdim]
          let xLo : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
          let xHi : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
          let yLoT : Tensor α (.dim a (.dim c (.dim b .scalar))) :=
            Tensor.transpose3DLastTwoSpec (α:=α) xLo
          let yHiT : Tensor α (.dim a (.dim c (.dim b .scalar))) :=
            Tensor.transpose3DLastTwoSpec (α:=α) xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id
            (some { dim := (Shape.dim a (Shape.dim c (Shape.dim b Shape.scalar))).size,
                    lo := flatLo,
                    hi := flatHi })
        else boxes
      | _ => boxes
    | _ => boxes
  | .permute perm =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let sIn := nodes[p1]!.outShape
      if hdim : Xin.dim = sIn.size then
        let sFlat : Shape := .dim Xin.dim .scalar
        have hsize : sFlat.size = sIn.size := by
          simp [sFlat, sIn, Shape.size, hdim]
        let xLo : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
        let xHi : Tensor α sIn := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
        match permuteDVal? (α := α) (v := ⟨sIn, xLo⟩) perm, permuteDVal? (α := α) (v := ⟨sIn, xHi⟩)
          perm with
        | some yLoV, some yHiV =>
            let sOut := dvalShape (α := α) yLoV
            if hSame : dvalShape (α := α) yHiV = sOut then
              if hOut : sOut = node.outShape then
                let yLoSOut : Tensor α sOut := dvalTensor (α := α) yLoV
                let yHiSOut : Tensor α sOut := hSame ▸ dvalTensor (α := α) yHiV
                let yLoT : Tensor α node.outShape := hOut ▸ yLoSOut
                let yHiT : Tensor α node.outShape := hOut ▸ yHiSOut
                let flatLo := Tensor.flattenSpec (α:=α) yLoT
                let flatHi := Tensor.flattenSpec (α:=α) yHiT
                boxes.set! id (some { dim := node.outShape.size, lo := flatLo, hi := flatHi })
              else
                boxes
            else
              boxes
        | _, _ => boxes
      else
        boxes
    | _ => boxes
  | .mul_elem =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match box_mul_elem (α:=α) (get! p1) (get! p2) with
      | some prod => boxes.set! id (some prod)
      | none => boxes
    | _ => boxes
  | .sum =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let loVal := Spec.Tensor.sumSpec Xin.lo
      let hiVal := Spec.Tensor.sumSpec Xin.hi
      let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
      let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
      boxes.set! id (some { dim := 1, lo := loT, hi := hiT })
    | _ => boxes
  | .mseLoss =>
    match node.parents with
    | p1 :: p2 :: _ =>
      let Y := get! p1
      let T := get! p2
      if hdim : Y.dim = T.dim then
        let Thi := castDimScalar (α:=α) (n:=T.dim) (n':=Y.dim) (h:=hdim.symm) T.hi
        let Tlo := castDimScalar (α:=α) (n:=T.dim) (n':=Y.dim) (h:=hdim.symm) T.lo
        let diff : FlatBox α :=
          { dim := Y.dim
            lo := Tensor.subSpec Y.lo Thi
            hi := Tensor.subSpec Y.hi Tlo }
        let sq := boxSquare (α:=α) diff
        let n := sq.dim
        if hn : n > 0 then
          let nA : α := (n : Nat)
          let loVal := (Spec.Tensor.sumSpec sq.lo) / nA
          let hiVal := (Spec.Tensor.sumSpec sq.hi) / nA
          let loT := Spec.fill (α:=α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α:=α) hiVal (.dim 1 .scalar)
          boxes.set! id (some { dim := 1, lo := loT, hi := hiT })
        else
          boxes
      else boxes
    | _ => boxes
  | .conv2d .. =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      match ibpConv2dNode (α:=α) id ps Xin with
      | some yB => boxes.set! id (some yB)
      | none =>
        -- Fallback: allow callers to inject a flattened linear form when conv params are absent.
        match ibp_linear (α:=α) id ps Xin with
        | some yB => boxes.set! id (some yB)
        | none    => boxes
    | _ => boxes
  | .exp =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      -- exp is monotone increasing: apply to lo and hi
      let lo := Tensor.expSpec Xin.lo
      let hi := Tensor.expSpec Xin.hi
      boxes.set! id (some { dim := Xin.dim, lo := lo, hi := hi })
    | _ => boxes
  | .log =>
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      -- Ensure positivity on lower bound to avoid log of non-positive
      let flo := getDimScalarFn (α:=α) Xin.lo
      let loSafe := Tensor.dim (fun i =>
        match flo i with
        | .scalar v => Tensor.scalar (if v > Numbers.epsilon then v else Numbers.epsilon))
      let hiSafe := Xin.hi
      let lo := Tensor.logSpec loSafe
      let hi := Tensor.logSpec hiSafe
      boxes.set! id (some { dim := Xin.dim, lo := lo, hi := hi })
    | _ => boxes
  -- layernorm/concat handled in dedicated cases below
  | .concat _ =>
    -- Concatenate two flattened boxes along the vector dimension
    match node.parents with
    | p1 :: p2 :: _ =>
      let B1 := get! p1; let B2 := get! p2
      match B1, B2 with
      | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
        let f1lo := getDimScalarFn (α:=α) lo1
        let f2lo := getDimScalarFn (α:=α) lo2
        let f1hi := getDimScalarFn (α:=α) hi1
        let f2hi := getDimScalarFn (α:=α) hi2
        let lo :=
          Tensor.dim (fun i =>
            Fin.addCases (fun i1 => f1lo i1) (fun i2 => f2lo i2) i)
        let hi :=
          Tensor.dim (fun i =>
            Fin.addCases (fun i1 => f1hi i1) (fun i2 => f2hi i2) i)
        boxes.set! id (some { dim := n1 + n2, lo := lo, hi := hi })
    | _ => boxes
  | .layernorm axis =>
    -- Last-axis LayerNorm bounds (without affine gamma/beta).
    -- We only implement `axis = rank-1` (the TorchLean usage); other axes are left unsupported.
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let s := node.outShape
      if axis = Shape.rank s - 1 then
        if hdim : Xin.dim = s.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = s.size := by
            simp [sFlat, Shape.size, hdim]
          let xLo : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.lo hsize
          let xHi : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.hi hsize
          let (yLoT, yHiT) := ibpLayernormLastTensor (α:=α) (s := s) xLo xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id (some { dim := s.size, lo := flatLo, hi := flatHi })
        else boxes
      else boxes
    | _ => boxes
  | .softmax axis =>
    -- Last-axis softmax bounds. We only implement `axis = rank-1`.
    match node.parents with
    | p1 :: _ =>
      let Xin := get! p1
      let s := node.outShape
      if axis = Shape.rank s - 1 then
        if hdim : Xin.dim = s.size then
          let sFlat : Shape := .dim Xin.dim .scalar
          have hsize : sFlat.size = s.size := by
            simp [sFlat, Shape.size, hdim]
          let xLo : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.lo hsize
          let xHi : Tensor α s := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=s) Xin.hi hsize
          let (yLoT, yHiT) := ibpSoftmaxLastTensor (α:=α) (s := s) xLo xHi
          let flatLo := Tensor.flattenSpec (α:=α) yLoT
          let flatHi := Tensor.flattenSpec (α:=α) yHiT
          boxes.set! id (some { dim := s.size, lo := flatLo, hi := flatHi })
        else boxes
      else boxes
    | _ => boxes
  | .tanh =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))
  | .sigmoid =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))
  | .sin =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))
  | .cos =>
    let Xin :=
      match node.parents with
      | p1 :: _ => get! p1
      | _ => get! 0
    let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=Xin.dim) (ofFlatBox Xin)
    boxes.set! id (some (toFlatBox Xin.dim yB))

/-- Run an IBP pass over the whole graph. Caller seeds inputs via ParamStore.inputBoxes. -/
def runIBP (g : Graph) (ps : ParamStore α) : Array (Option (FlatBox α)) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl (fun acc i => propagateIBPNode (α:=α) g.nodes ps acc i) init

/-- Derivative IBP pass for 1D input: computes for each node an interval on dy/dx. Requires value
  IBP boxes for activations. -/
def runDeriv1D (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) : Array (Option
  (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (drs : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        let one := Spec.fill (α:=α) Numbers.one (.dim B.dim .scalar)
        drs.set! id (some { dim := B.dim, lo := one, hi := one })
      | none => drs
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        drs.set! id (some { dim := v.n, lo := z, hi := z })
      | none => drs
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      drs.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
      -- Not supported by the derivative-bound passes (used by PINN tooling).
      drs
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dXin =>
          let loVal := Spec.Tensor.sumSpec dXin.lo
          let hiVal := Spec.Tensor.sumSpec dXin.hi
          let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
          drs.set! id (some { dim := 1, lo := loT, hi := hiT })
        | none => drs
      | _ => drs
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.linearWB[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .matmul =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.matmulW[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .relu =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dIn =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim dIn.dim .scalar)
          let o := Spec.fill (α:=α) Numbers.one  (.dim dIn.dim .scalar)
          let dF : FlatBox α := { dim := dIn.dim, lo := z, hi := o }
          match box_mul_elem (α:=α) dIn dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .tanh =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          -- Use tighter derivative bounds: tanh'(z) = 1 - tanh(z)^2, with tanh(z) ∈ [yl, yh]
          if dZ.dim = yB.dim then
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_max := if yl2 > yh2 then yl2 else yh2
                  Tensor.scalar (Numbers.one - s_max))
            let dhi :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_min :=
                    if yl < Numbers.zero then
                      if Numbers.zero < yh then Numbers.zero else (if yl2 < yh2 then yl2 else yh2)
                    else (if yl2 < yh2 then yl2 else yh2)
                  Tensor.scalar (Numbers.one - s_min))
            let dF : FlatBox α := { dim := yB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some sB =>
          if dZ.dim = sB.dim then
            let fsLo := getDimScalarFn (α:=α) sB.lo
            let fsHi := getDimScalarFn (α:=α) sB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mn := if fa < fb then fa else fb
                  Tensor.scalar mn)
            let dhi :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mxEnds := if fa > fb then fa else fb
                  let quarter := Numbers.pointfive * (Numbers.one - Numbers.pointfive)
                  let mx :=
                    if a < Numbers.pointfive then
                      if Numbers.pointfive < b then
                        let mx' := if mxEnds < quarter then quarter else mxEnds
                        mx'
                      else mxEnds
                    else mxEnds
                  Tensor.scalar mx)
            let dF : FlatBox α := { dim := sB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .softmax _ =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          if h : dZ.dim = yB.dim then
            let n := yB.dim
            -- Cast derivative tensors to dimension n for Fin alignment
            let dLo := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.lo
            let dHi := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.hi
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let fdLo := getDimScalarFn (α:=α) dLo
            let fdHi := getDimScalarFn (α:=α) dHi
            let mulI (aLo aHi bLo bHi : α) : α × α :=
              let p1 := aLo * bLo; let p2 := aLo * bHi
              let p3 := aHi * bLo; let p4 := aHi * bHi
              let lo1 := if p1 < p2 then p1 else p2
              let lo2 := if p3 < p4 then p3 else p4
              let lo  := if lo1 < lo2 then lo1 else lo2
              let hi1 := if p1 > p2 then p1 else p2
              let hi2 := if p3 > p4 then p3 else p4
              let hi  := if hi1 > hi2 then hi1 else hi2
              (lo, hi)
            let dlo :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (sumLo, _sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumLo)
            let dhi :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (_sumLo, sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumHi)
            drs.set! id (some { dim := n, lo := dlo, hi := dhi })
          else drs
        | _, _ => drs
      | _ => drs
    | .sin =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let cB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let dF : FlatBox α := { dim := zB.dim, lo := cB.lo, hi := cB.hi }
          match box_mul_elem (α:=α) dZ dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .cos =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let sB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let negSB : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sB.hi
              hi := Tensor.mapSpec (fun x => -x) sB.lo }
          match box_mul_elem (α:=α) dZ negSB with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .exp =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxExp (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .log =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxLog (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_add (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_sub (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]!, ibp[p1]!, ibp[p2]! with
        | some dx, some dy, some xB, some yB =>
          match box_mul_elem (α:=α) dx yB, box_mul_elem (α:=α) xB dy with
          | some t1, some t2 => drs.set! id (some (box_add (α:=α) t1 t2))
          | _, _ => drs
        | _, _, _, _ => drs
      | _ => drs
    | .layernorm _ =>
      -- Derivative of layernorm y = (x - mean(x))/sqrt(var+eps): dy ≈ t*(dx - mean dx) + dt*u.
      -- We bound t in [t_lo,t_hi], bound v := (dx - mean dx) per-component, and bound |dt| via
      -- |dt| ≤ 0.5 * (var+eps)^(-3/2)_hi * (2/n) * Σ_j max|u_j| * max|v_j|; then add symmetric dt*u
      -- term.
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dXin, some Xin =>
          let n := Xin.dim
          if hn : dXin.dim = n then
            -- Compute mean bounds of x and dx
            let sum_lo := Spec.Tensor.sumSpec Xin.lo
            let sum_hi := Spec.Tensor.sumSpec Xin.hi
            let nA : α := (n : Nat)
            let mu_lo := sum_lo / nA
            let mu_hi := sum_hi / nA
            -- u_j bounds = x_j - mean(x)
            let flo := getDimScalarFn (α:=α) Xin.lo
            let fhi := getDimScalarFn (α:=α) Xin.hi
            let u_lo :=
              Tensor.dim (fun i =>
                match flo i, fhi i with
                | .scalar l, .scalar u =>
                  let dl := l - mu_hi
                  let du := u - mu_lo
                  let mn := if dl < du then dl else du
                  Tensor.scalar mn)
            let u_hi :=
              Tensor.dim (fun i =>
                match flo i, fhi i with
                | .scalar l, .scalar u =>
                  let dl := l - mu_hi
                  let du := u - mu_lo
                  let mx := if dl > du then dl else du
                  Tensor.scalar mx)
            -- Bounds on variance and denom s = sqrt(var+eps)
            let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
              match flo i, fhi i with
              | .scalar l, .scalar u =>
                let dl := MathFunctions.abs (l - mu_hi)
                let du := MathFunctions.abs (u - mu_lo)
                let a := if dl > du then dl else du
                acc + (a * a)
            ) 0
            let var_hi := sumAbsSq / nA
            let s_lo := MathFunctions.sqrt Numbers.epsilon
            let s_hi := MathFunctions.sqrt (var_hi + Numbers.epsilon)
            let t_lo := Numbers.one / (if s_hi > Numbers.epsilon then s_hi else Numbers.epsilon)
            let t_hi := Numbers.one / (if s_lo > Numbers.epsilon then s_lo else Numbers.epsilon)
            -- dx mean bounds and v_j = dx_j - mean(dx)
            let dlo := getDimScalarFn (α:=α) dXin.lo
            let dhi := getDimScalarFn (α:=α) dXin.hi
            let sum_dx_lo := Spec.Tensor.sumSpec dXin.lo
            let sum_dx_hi := Spec.Tensor.sumSpec dXin.hi
            let dmu_lo := sum_dx_lo / nA
            let dmu_hi := sum_dx_hi / nA
            let v_lo :=
              Tensor.dim (fun i =>
                match dlo i, dhi i with
                | .scalar l, .scalar u =>
                  let a := l - dmu_hi
                  let b := u - dmu_lo
                  Tensor.scalar (if a < b then a else b))
            let v_hi :=
              Tensor.dim (fun i =>
                match dlo i, dhi i with
                | .scalar l, .scalar u =>
                  let a := l - dmu_hi
                  let b := u - dmu_lo
                  Tensor.scalar (if a > b then a else b))
            -- Align v bounds to dimension n via cast
            let v_loN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_lo
            let v_hiN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_hi
            -- First term: t * v
            -- Compute base = t * v per component where t∈[t_lo,t_hi] and v_i∈[v_loN[i],v_hiN[i]]
            let vLoFn := getDimScalarFn (α:=α) v_loN
            let vHiFn := getDimScalarFn (α:=α) v_hiN
            let base_lo :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let m1 := if p1 < p2 then p1 else p2
                  let m2 := if p3 < p4 then p3 else p4
                  Tensor.scalar (if m1 < m2 then m1 else m2))
            let base_hi :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let M1 := if p1 > p2 then p1 else p2
                  let M2 := if p3 > p4 then p3 else p4
                  Tensor.scalar (if M1 > M2 then M1 else M2))
            let baseN : FlatBox α := { dim := n, lo := base_lo, hi := base_hi }
            -- Bound |dt| using t3_hi = 1/s^3 and |(2/n) Σ u_j v_j|
            let t3_hi :=
              let s_lo' := s_lo
              let s3 := s_lo' * s_lo' * s_lo'
              Numbers.one / (if s3 > Numbers.epsilon then s3 else Numbers.epsilon)
            let abs_max (l u : α) : α :=
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              if al > au then al else au
            -- compute G = Σ max|u_j| * max|v_j|
            let u_abs := getDimScalarFn (α:=α) u_lo
            let u_abs_hi := getDimScalarFn (α:=α) u_hi
            let v_abs := getDimScalarFn (α:=α) v_loN
            let v_abs_hi := getDimScalarFn (α:=α) v_hiN
            let G : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
              match u_abs i, u_abs_hi i, v_abs i, v_abs_hi i with
              | .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                let au := abs_max ul uu
                let av := abs_max vl vu
                acc + (au * av)
            ) 0
            let V := (Numbers.two * G) / nA
            let dt_abs := (Numbers.pointfive * t3_hi) * V
            -- Add symmetric dt*u term per component: ± dt_abs * max|u_i|
            let fulo := getDimScalarFn (α:=α) u_lo
            let fuhi := getDimScalarFn (α:=α) u_hi
            let bLoFn := getDimScalarFn (α:=α) baseN.lo
            let bHiFn := getDimScalarFn (α:=α) baseN.hi
            let add_lo :=
              Tensor.dim (fun i =>
                match bLoFn i, fulo i, fuhi i with
                | .scalar bi, .scalar ul, .scalar uu =>
                  let au := abs_max ul uu
                  Tensor.scalar (bi - dt_abs * au))
            let add_hi :=
              Tensor.dim (fun i =>
                match bHiFn i, fulo i, fuhi i with
                | .scalar bi, .scalar ul, .scalar uu =>
                  let au := abs_max ul uu
                  Tensor.scalar (bi + dt_abs * au))
            drs.set! id (some { dim := n, lo := add_lo, hi := add_hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .reshape _ _ | .flatten _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .permute _
      =>
      match node.parents with
      | p1 :: _ => drs.set! id (drs[p1]!)
      | _ => drs
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      drs
    | .mseLoss => drs
    | .conv2d .. => drs
  (List.finRange g.nodes.size).foldl propagate init

/-- Directional first-derivative pass: like `runDeriv1D` but seeds the derivative at the
    input node with a user-provided direction vector (as a FlatBox with lo=hi). This allows
    extracting partial derivatives for multi-dimensional inputs by choosing e_x, e_y, etc. -/
def runDerivDirectional (g : Graph) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α))) (seed : FlatBox α) : Array (Option (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (drs : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        if h : seed.dim = B.dim then
          let dlo := castDimScalar (α:=α) (n:=seed.dim) (n':=B.dim) (h:=h) seed.lo
          let dhi := castDimScalar (α:=α) (n:=seed.dim) (n':=B.dim) (h:=h) seed.hi
          drs.set! id (some { dim := B.dim, lo := dlo, hi := dhi })
        else drs
      | none => drs
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        drs.set! id (some { dim := v.n, lo := z, hi := z })
      | none => drs
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      drs.set! id (some { dim := d, lo := z, hi := z })
    | .sin =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let cB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let dF : FlatBox α := { dim := zB.dim, lo := cB.lo, hi := cB.hi }
          match box_mul_elem (α:=α) dZ dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .cos =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let sB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let negSB : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sB.hi
              hi := Tensor.mapSpec (fun x => -x) sB.lo }
          match box_mul_elem (α:=α) dZ negSB with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
      -- Not supported by the derivative-bound passes.
      drs
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dXin =>
          let loVal := Spec.Tensor.sumSpec dXin.lo
          let hiVal := Spec.Tensor.sumSpec dXin.hi
          let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
          drs.set! id (some { dim := 1, lo := loT, hi := hiT })
        | none => drs
      | _ => drs
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.linearWB[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .matmul =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.matmulW[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .relu =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dIn =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim dIn.dim .scalar)
          let o := Spec.fill (α:=α) Numbers.one  (.dim dIn.dim .scalar)
          let dF : FlatBox α := { dim := dIn.dim, lo := z, hi := o }
          match box_mul_elem (α:=α) dIn dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .tanh =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          if dZ.dim = yB.dim then
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_max := if yl2 > yh2 then yl2 else yh2
                  Tensor.scalar (Numbers.one - s_max))
            let dhi :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_min :=
                    if yl < Numbers.zero then
                      if Numbers.zero < yh then Numbers.zero else (if yl2 < yh2 then yl2 else yh2)
                    else (if yl2 < yh2 then yl2 else yh2)
                  Tensor.scalar (Numbers.one - s_min))
            let dF : FlatBox α := { dim := yB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some sB =>
          if dZ.dim = sB.dim then
            let fsLo := getDimScalarFn (α:=α) sB.lo
            let fsHi := getDimScalarFn (α:=α) sB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mn := if fa < fb then fa else fb
                  Tensor.scalar mn)
            let dhi :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mxEnds := if fa > fb then fa else fb
                  let quarter := Numbers.pointfive * (Numbers.one - Numbers.pointfive)
                  let mx :=
                    if a < Numbers.pointfive then
                      if Numbers.pointfive < b then
                        let mx' := if mxEnds < quarter then quarter else mxEnds
                        mx'
                      else mxEnds
                    else mxEnds
                  Tensor.scalar mx)
            let dF : FlatBox α := { dim := sB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .softmax _ =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          if h : dZ.dim = yB.dim then
            let n := yB.dim
            let dLo := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.lo
            let dHi := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.hi
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let fdLo := getDimScalarFn (α:=α) dLo
            let fdHi := getDimScalarFn (α:=α) dHi
            let mulI (aLo aHi bLo bHi : α) : α × α :=
              let p1 := aLo * bLo; let p2 := aLo * bHi
              let p3 := aHi * bLo; let p4 := aHi * bHi
              let lo1 := if p1 < p2 then p1 else p2
              let lo2 := if p3 < p4 then p3 else p4
              let lo  := if lo1 < lo2 then lo1 else lo2
              let hi1 := if p1 > p2 then p1 else p2
              let hi2 := if p3 > p4 then p3 else p4
              let hi  := if hi1 > hi2 then hi1 else hi2
              (lo, hi)
            let dlo :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (sumLo, _sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumLo)
            let dhi :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (_sumLo, sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumHi)
            drs.set! id (some { dim := n, lo := dlo, hi := dhi })
          else drs
        | _, _ => drs
      | _ => drs
    | .exp =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxExp (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .log =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxLog (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_add (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_sub (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]!, ibp[p1]!, ibp[p2]! with
        | some dx, some dy, some xB, some yB =>
          match box_mul_elem (α:=α) dx yB, box_mul_elem (α:=α) xB dy with
          | some t1, some t2 => drs.set! id (some (box_add (α:=α) t1 t2))
          | _, _ => drs
        | _, _, _, _ => drs
      | _ => drs
    | .layernorm _ =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dXin, some Xin =>
          let n := Xin.dim
          if hn : dXin.dim = n then
            let sum_lo := Spec.Tensor.sumSpec Xin.lo
            let sum_hi := Spec.Tensor.sumSpec Xin.hi
            let nA : α := (n : Nat)
            let mu_lo := sum_lo / nA
            let mu_hi := sum_hi / nA
            let flo := getDimScalarFn (α:=α) Xin.lo
            let fhi := getDimScalarFn (α:=α) Xin.hi
            let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
              match flo i, fhi i with
              | .scalar l, .scalar u =>
                let dl := MathFunctions.abs (l - mu_hi)
                let du := MathFunctions.abs (u - mu_lo)
                let a := if dl > du then dl else du
                acc + (a * a)
            ) 0
            let var_hi := sumAbsSq / nA
            let s_lo := MathFunctions.sqrt Numbers.epsilon
            let s_hi := MathFunctions.sqrt (var_hi + Numbers.epsilon)
            let t_lo := Numbers.one / (if s_hi > Numbers.epsilon then s_hi else Numbers.epsilon)
            let t_hi := Numbers.one / (if s_lo > Numbers.epsilon then s_lo else Numbers.epsilon)
            let dlo := getDimScalarFn (α:=α) dXin.lo
            let dhi := getDimScalarFn (α:=α) dXin.hi
            let sum_dx_lo := Spec.Tensor.sumSpec dXin.lo
            let sum_dx_hi := Spec.Tensor.sumSpec dXin.hi
            let dmu_lo := sum_dx_lo / nA
            let dmu_hi := sum_dx_hi / nA
            let v_lo :=
              Tensor.dim (fun i =>
                match dlo i, dhi i with
                | .scalar l, .scalar u =>
                  let a := l - dmu_hi
                  let b := u - dmu_lo
                  Tensor.scalar (if a < b then a else b))
            let v_hi :=
              Tensor.dim (fun i =>
                match dlo i, dhi i with
                | .scalar l, .scalar u =>
                  let a := l - dmu_hi
                  let b := u - dmu_lo
                  Tensor.scalar (if a > b then a else b))
            let v_loN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_lo
            let v_hiN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_hi
            let vLoFn := getDimScalarFn (α:=α) v_loN
            let vHiFn := getDimScalarFn (α:=α) v_hiN
            let base_lo :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let m1 := if p1 < p2 then p1 else p2
                  let m2 := if p3 < p4 then p3 else p4
                  Tensor.scalar (if m1 < m2 then m1 else m2))
            let base_hi :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let M1 := if p1 > p2 then p1 else p2
                  let M2 := if p3 > p4 then p3 else p4
                  Tensor.scalar (if M1 > M2 then M1 else M2))
            let baseN : FlatBox α := { dim := n, lo := base_lo, hi := base_hi }
            drs.set! id (some baseN)
          else drs
        | _, _ => drs
      | _ => drs
    | .reshape _ _ | .flatten _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .permute _
      =>
      match node.parents with
      | p1 :: _ => drs.set! id (drs[p1]!)
      | _ => drs
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      drs
    | .mseLoss => drs
    | .conv2d .. => drs
  (List.finRange g.nodes.size).foldl propagate init

/-- Second-derivative IBP pass for 1D input: computes per node an interval on d²y/dx².
    Requires value IBP boxes and first-derivative boxes. Covers input, linear/matmul, tanh, add/sub.
      -/
def runDeriv2D (g : Graph) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α))) (d1 : Array (Option (FlatBox α))) : Array (Option (FlatBox α))
    :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (d2s : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim B.dim .scalar)
        d2s.set! id (some { dim := B.dim, lo := z, hi := z })
      | none => d2s
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        d2s.set! id (some { dim := v.n, lo := z, hi := z })
      | none => d2s
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      d2s.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
      -- Not supported by the second-derivative bound pass.
      d2s
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match d2s[p1]!, ps.linearWB[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .matmul =>
      match node.parents with
      | p1 :: _ =>
        match d2s[p1]!, ps.matmulW[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (box_add (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (box_sub (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]!, d1[p1]!, d1[p2]!, d2s[p1]!, d2s[p2]! with
        | some xB, some yB, some dx, some dy, some d2x, some d2y =>
          -- y'' = x''⊙y + 2 x'⊙y' + x⊙y''
          match box_mul_elem (α:=α) d2x yB, box_mul_elem (α:=α) dx dy, box_mul_elem (α:=α) xB d2y
            with
          | some t1, some mid, some t3 =>
            let twoMid := box_add (α:=α) mid mid
            d2s.set! id (some (box_add (α:=α) t1 (box_add (α:=α) twoMid t3)))
          | _, _, _ => d2s
        | _, _, _, _, _, _ => d2s
      | _ => d2s
    | .relu =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some zB =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim zB.dim .scalar)
          d2s.set! id (some { dim := zB.dim, lo := z, hi := z })
        | none => d2s
      | _ => d2s
    | .tanh =>
      match node.parents with
      | p1 :: _ =>
        match ibp[id]!, d1[p1]!, d2s[p1]! with
        | some yB, some dz, some d2z =>
          if dz.dim = yB.dim then
            if d2z.dim = yB.dim then
              -- f'(z) = 1 - y^2; f''(z) = -2 y (1 - y^2)
              let fyLo := getDimScalarFn (α:=α) yB.lo
              let fyHi := getDimScalarFn (α:=α) yB.hi
              let f1_lo :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let yl2 := yl * yl
                    let yh2 := yh * yh
                    let s_max := if yl2 > yh2 then yl2 else yh2
                    Tensor.scalar (Numbers.one - s_max))
              let f1_hi :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let yl2 := yl * yl
                    let yh2 := yh * yh
                    let s_min :=
                      if yl < Numbers.zero then
                        if Numbers.zero < yh then Numbers.zero else (if yl2 < yh2 then yl2 else yh2)
                      else (if yl2 < yh2 then yl2 else yh2)
                    Tensor.scalar (Numbers.one - s_min))
              let f2_lo :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let cube (v : α) := v * v * v
                    let cand1 := (-(Numbers.two) * yl) + (Numbers.two * cube yl)
                    let cand2 := (-(Numbers.two) * yh) + (Numbers.two * cube yh)
                    let rt := MathFunctions.sqrt (Numbers.one / Numbers.three)
                    -- evaluate at +/- 1/sqrt(3) when inside interval
                    let cand3 := if (yl < rt ∧ rt < yh) then ((-(Numbers.two) * rt) + (Numbers.two *
                      cube rt)) else cand1
                    let nrt := (-rt)
                    let cand4 := if (yl < nrt ∧ nrt < yh) then ((-(Numbers.two) * nrt) +
                      (Numbers.two * cube nrt)) else cand2
                    let m1 := if cand1 < cand2 then cand1 else cand2
                    let m2 := if cand3 < cand4 then cand3 else cand4
                    Tensor.scalar (if m1 < m2 then m1 else m2))
              let f2_hi :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let cube (v : α) := v * v * v
                    let cand1 := (-(Numbers.two) * yl) + (Numbers.two * cube yl)
                    let cand2 := (-(Numbers.two) * yh) + (Numbers.two * cube yh)
                    let rt := MathFunctions.sqrt (Numbers.one / Numbers.three)
                    let cand3 := if (yl < rt ∧ rt < yh) then ((-(Numbers.two) * rt) + (Numbers.two *
                      cube rt)) else cand1
                    let nrt := (-rt)
                    let cand4 := if (yl < nrt ∧ nrt < yh) then ((-(Numbers.two) * nrt) +
                      (Numbers.two * cube nrt)) else cand2
                    let M1 := if cand1 > cand2 then cand1 else cand2
                    let M2 := if cand3 > cand4 then cand3 else cand4
                    Tensor.scalar (if M1 > M2 then M1 else M2))
              let f1B : FlatBox α := { dim := yB.dim, lo := f1_lo, hi := f1_hi }
              let f2B : FlatBox α := { dim := yB.dim, lo := f2_lo, hi := f2_hi }
              let dz2 := boxSquare (α:=α) dz
              match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
              | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
              | _, _ => d2s
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sin =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          -- y = sin(z): y'' = (-sin(z))*(z')^2 + cos(z)*z''
          let sinB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let cosB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let f1B : FlatBox α := { dim := zB.dim, lo := cosB.lo, hi := cosB.hi }
          let f2B : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sinB.hi
              hi := Tensor.mapSpec (fun x => -x) sinB.lo }
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .cos =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          -- y = cos(z): y'' = (-cos(z))*(z')^2 + (-sin(z))*z''
          let sinB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let cosB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let f1B : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sinB.hi
              hi := Tensor.mapSpec (fun x => -x) sinB.lo }
          let f2B : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) cosB.hi
              hi := Tensor.mapSpec (fun x => -x) cosB.lo }
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
        match ibp[id]!, d1[p1]!, d2s[p1]! with
        | some sB, some dz, some d2z =>
          if dz.dim = sB.dim then
            if d2z.dim = sB.dim then
              let fsLo := getDimScalarFn (α:=α) sB.lo
              let fsHi := getDimScalarFn (α:=α) sB.hi
              -- f'(z) = s(1-s)
              let f1_lo :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar a, .scalar b =>
                    let fa := a * (Numbers.one - a)
                    let fb := b * (Numbers.one - b)
                    let mn := if fa < fb then fa else fb
                    Tensor.scalar mn)
              let f1_hi :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar a, .scalar b =>
                    let fa := a * (Numbers.one - a)
                    let fb := b * (Numbers.one - b)
                    let quarter := Numbers.pointfive * (Numbers.one - Numbers.pointfive)
                    let mxEnds := if fa > fb then fa else fb
                    let mx := if a < Numbers.pointfive then (if Numbers.pointfive < b then (if
                      mxEnds < quarter then quarter else mxEnds) else mxEnds) else mxEnds
                    Tensor.scalar mx)
              -- f''(z) = f'(z) * (1 - 2s) with s in [a,b]
              let oneMinus2s_lo :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar _a, .scalar b => Tensor.scalar (Numbers.one - (Numbers.two * b)))
              let oneMinus2s_hi :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar a, .scalar _b => Tensor.scalar (Numbers.one - (Numbers.two * a)))
              let f1B : FlatBox α := { dim := sB.dim, lo := f1_lo, hi := f1_hi }
              let f2fac : FlatBox α := { dim := sB.dim, lo := oneMinus2s_lo, hi := oneMinus2s_hi }
              match box_mul_elem (α:=α) f1B f2fac with
              | some f2B =>
                let dz2 := boxSquare (α:=α) dz
                match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
                | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
                | _, _ => d2s
              | none => d2s
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .exp =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          let f1 := { dim := zB.dim, lo := Tensor.expSpec zB.lo, hi := Tensor.expSpec zB.hi }
          let f2 := f1 -- same for exp
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2 dz2, box_mul_elem (α:=α) f1 d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .log =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          let flo := getDimScalarFn (α:=α) zB.lo
          let fhi := getDimScalarFn (α:=α) zB.hi
          let f1_lo :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar _l, .scalar u => Tensor.scalar (Numbers.one / u))
          let f1_hi :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar l, .scalar _u =>
                let l' := if l > Numbers.epsilon then l else Numbers.epsilon
                Tensor.scalar (Numbers.one / l'))
          let f2_lo :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar l, .scalar _u =>
                let l' := if l > Numbers.epsilon then l else Numbers.epsilon
                -- f'' = -1/z^2 in [-(1/l'^2), -(1/u^2)] with l' ≤ u
                Tensor.scalar (-(Numbers.one / (l' * l'))))
          let f2_hi :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar _l, .scalar u =>
                -- upper bound (less negative): -(1/u^2)
                Tensor.scalar (-(Numbers.one / (u * u))))
          let f1B : FlatBox α := { dim := zB.dim, lo := f1_lo, hi := f1_hi }
          let f2B : FlatBox α := { dim := zB.dim, lo := f2_lo, hi := f2_hi }
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match d2s[p1]! with
        | some d2Xin =>
          let loVal := Spec.Tensor.sumSpec d2Xin.lo
          let hiVal := Spec.Tensor.sumSpec d2Xin.hi
          let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
          d2s.set! id (some { dim := 1, lo := loT, hi := hiT })
        | none => d2s
      | _ => d2s
    | .reshape _ _ | .flatten _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .permute _
      =>
      match node.parents with
      | p1 :: _ => d2s.set! id (d2s[p1]!)
      | _ => d2s
    | .mseLoss => d2s
    | .softmax _ =>
      -- y''_i = Σ_k J_ik d2z_k + Σ_{j,k} H_ijk dz_j dz_k, with
      -- J = diag(y) - y yᵀ and H derived from ∂J/∂z (bounded via y-bounds).
      match node.parents with
      | p1 :: _ =>
        match ibp[id]!, d1[p1]!, d2s[p1]! with
        | some yB, some dz, some d2z =>
          if h1 : dz.dim = yB.dim then
            if h2 : d2z.dim = yB.dim then
              let n := yB.dim
              -- Cast derivative tensors to dimension n for Fin alignment
              let d1Lo := castDimScalar (α:=α) (n:=dz.dim) (n':=n) (h:=h1) dz.lo
              let d1Hi := castDimScalar (α:=α) (n:=dz.dim) (n':=n) (h:=h1) dz.hi
              let d2Lo := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.lo
              let d2Hi := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.hi
              let fyLo := getDimScalarFn (α:=α) yB.lo
              let fyHi := getDimScalarFn (α:=α) yB.hi
              let fd1Lo := getDimScalarFn (α:=α) d1Lo
              let fd1Hi := getDimScalarFn (α:=α) d1Hi
              let fd2Lo := getDimScalarFn (α:=α) d2Lo
              let fd2Hi := getDimScalarFn (α:=α) d2Hi
              let mulI (aLo aHi bLo bHi : α) : α × α :=
                let p1 := aLo * bLo; let p2 := aLo * bHi
                let p3 := aHi * bLo; let p4 := aHi * bHi
                let lo1 := if p1 < p2 then p1 else p2
                let lo2 := if p3 < p4 then p3 else p4
                let lo  := if lo1 < lo2 then lo1 else lo2
                let hi1 := if p1 > p2 then p1 else p2
                let hi2 := if p3 > p4 then p3 else p4
                let hi  := if hi1 > hi2 then hi1 else hi2
                (lo, hi)
              -- Bounds for (δ_ik - y_k)
              let deltaMinus (i k : Fin n) : α × α :=
                if decide (i.val = k.val) then
                  let ykLo := match fyLo k with | .scalar v => v
                  let ykHi := match fyHi k with | .scalar v => v
                  (Numbers.one - ykHi, Numbers.one - ykLo)
                else
                  let ykLo := match fyLo k with | .scalar v => v
                  let ykHi := match fyHi k with | .scalar v => v
                  ((-ykHi), (-ykLo))
              -- J*d2z term per i
              let part1_lo :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := match fd2Lo k with | .scalar v => v
                      let d2kHi := match fd2Hi k with | .scalar v => v
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumLo)
              let part1_hi :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := match fd2Lo k with | .scalar v => v
                      let d2kHi := match fd2Hi k with | .scalar v => v
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumHi)
              -- Quadratic term Σ_{j,k} H_ijk dz_j dz_k, use interval-bounded H from y-bounds
              let part2_lo :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := match fyLo j with | .scalar v => v
                      let yjHi := match fyHi j with | .scalar v => v
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (Numbers.one -
                        yjHi, Numbers.one - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := match fyLo k with | .scalar v => v
                        let ykHi := match fyHi k with | .scalar v => v
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (Numbers.one -
                          ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        -- H_ijk = y_i (dij)(dik) - y_i y_j (δ_jk - y_k)
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (Numbers.one - ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        -- H interval = t1 - t2
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := match fd1Lo j with | .scalar v => v
                        let dzjHi := match fd1Hi j with | .scalar v => v
                        let dzkLo := match fd1Lo k with | .scalar v => v
                        let dzkHi := match fd1Hi k with | .scalar v => v
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumLo)
              let part2_hi :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := match fyLo j with | .scalar v => v
                      let yjHi := match fyHi j with | .scalar v => v
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (Numbers.one -
                        yjHi, Numbers.one - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := match fyLo k with | .scalar v => v
                        let ykHi := match fyHi k with | .scalar v => v
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (Numbers.one -
                          ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (Numbers.one - ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := match fd1Lo j with | .scalar v => v
                        let dzjHi := match fd1Hi j with | .scalar v => v
                        let dzkLo := match fd1Lo k with | .scalar v => v
                        let dzkHi := match fd1Hi k with | .scalar v => v
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumHi)
              let lo := Tensor.addSpec part1_lo part2_lo
              let hi := Tensor.addSpec part1_hi part2_hi
              d2s.set! id (some { dim := n, lo := lo, hi := hi })
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .layernorm _ =>
      -- Conservative d² for layernorm.
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some Xin, some dXin, some d2Xin =>
          let n := Xin.dim
          if hn1 : dXin.dim = n then
            if hn2 : d2Xin.dim = n then
              let nA : α := (n : Nat)
              let sum_lo := Spec.Tensor.sumSpec Xin.lo
              let sum_hi := Spec.Tensor.sumSpec Xin.hi
              let mu_lo := sum_lo / nA
              let mu_hi := sum_hi / nA
              let flo := getDimScalarFn (α:=α) Xin.lo
              let fhi := getDimScalarFn (α:=α) Xin.hi
              -- u := x - mean(x)
              let u_lo :=
                Tensor.dim (fun i => match flo i, fhi i with
                  | .scalar l, .scalar u =>
                    let dl := l - mu_hi; let du := u - mu_lo
                    Tensor.scalar (if dl < du then dl else du))
              let u_hi :=
                Tensor.dim (fun i => match flo i, fhi i with
                  | .scalar l, .scalar u =>
                    let dl := l - mu_hi; let du := u - mu_lo
                    Tensor.scalar (if dl > du then dl else du))
              -- s = sqrt(var+eps), bounds
              let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
                match flo i, fhi i with
                | .scalar l, .scalar u =>
                  let dl := MathFunctions.abs (l - mu_hi)
                  let du := MathFunctions.abs (u - mu_lo)
                  let a := if dl > du then dl else du
                  acc + (a * a)
              ) 0
              let var_hi := sumAbsSq / nA
              let s_lo := MathFunctions.sqrt Numbers.epsilon
              let s_hi := MathFunctions.sqrt (var_hi + Numbers.epsilon)
              let t_lo := Numbers.one / (if s_hi > Numbers.epsilon then s_hi else Numbers.epsilon)
              let t_hi := Numbers.one / (if s_lo > Numbers.epsilon then s_lo else Numbers.epsilon)
              -- d2v = d2x - mean(d2x)
              let d2lo := castDimScalar (α:=α) (n:=d2Xin.dim) (n':=n) (h:=hn2) d2Xin.lo
              let d2hi := castDimScalar (α:=α) (n:=d2Xin.dim) (n':=n) (h:=hn2) d2Xin.hi
              let sum_d2_lo := Spec.Tensor.sumSpec d2lo
              let sum_d2_hi := Spec.Tensor.sumSpec d2hi
              let d2mu_lo := sum_d2_lo / nA
              let d2mu_hi := sum_d2_hi / nA
              let fd2Lo := getDimScalarFn (α:=α) d2lo
              let fd2Hi := getDimScalarFn (α:=α) d2hi
              let d2v_lo :=
                Tensor.dim (fun i => match fd2Lo i, fd2Hi i with
                  | .scalar l, .scalar u =>
                    let a := l - d2mu_hi; let b := u - d2mu_lo
                    Tensor.scalar (if a < b then a else b))
              let d2v_hi :=
                Tensor.dim (fun i => match fd2Lo i, fd2Hi i with
                  | .scalar l, .scalar u =>
                    let a := l - d2mu_hi; let b := u - d2mu_lo
                    Tensor.scalar (if a > b then a else b))
              -- v = dx - mean(dx)
              let dlo := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn1) dXin.lo
              let dhi := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn1) dXin.hi
              let sum_dx_lo := Spec.Tensor.sumSpec dlo
              let sum_dx_hi := Spec.Tensor.sumSpec dhi
              let dmu_lo := sum_dx_lo / nA
              let dmu_hi := sum_dx_hi / nA
              let fd1Lo := getDimScalarFn (α:=α) dlo
              let fd1Hi := getDimScalarFn (α:=α) dhi
              let v_lo :=
                Tensor.dim (fun i => match fd1Lo i, fd1Hi i with
                  | .scalar l, .scalar u =>
                    let a := l - dmu_hi; let b := u - dmu_lo
                    Tensor.scalar (if a < b then a else b))
              let v_hi :=
                Tensor.dim (fun i => match fd1Lo i, fd1Hi i with
                  | .scalar l, .scalar u =>
                    let a := l - dmu_hi; let b := u - dmu_lo
                    Tensor.scalar (if a > b then a else b))
              -- helper: abs max per comp
              let abs_max (l u : α) : α :=
                let al := MathFunctions.abs l; let au := MathFunctions.abs u
                if al > au then al else au
              let uLoFn := getDimScalarFn (α:=α) u_lo
              let uHiFn := getDimScalarFn (α:=α) u_hi
              let vLoFn := getDimScalarFn (α:=α) v_lo
              let vHiFn := getDimScalarFn (α:=α) v_hi
              let d2vLoFn := getDimScalarFn (α:=α) d2v_lo
              let d2vHiFn := getDimScalarFn (α:=α) d2v_hi
              -- global scalars for dt, d2t bounds
              let t3_hi :=
                let s3 := s_lo * s_lo * s_lo
                Numbers.one / (if s3 > Numbers.epsilon then s3 else Numbers.epsilon)
              let t5_hi :=
                let s5 := s_lo * s_lo * s_lo * s_lo * s_lo
                Numbers.one / (if s5 > Numbers.epsilon then s5 else Numbers.epsilon)
              let G : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
                match uLoFn i, uHiFn i, vLoFn i, vHiFn i with
                | .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                  let au := abs_max ul uu; let av := abs_max vl vu
                  acc + (au * av)
              ) 0
              let H : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
                match uLoFn i, uHiFn i, vLoFn i, vHiFn i, d2vLoFn i, d2vHiFn i with
                | .scalar ul, .scalar uu, .scalar vl, .scalar vu, .scalar wl, .scalar wu =>
                  let au := abs_max ul uu; let av := abs_max vl vu; let aw := abs_max wl wu
                  acc + ((av * av) + (au * aw))
              ) 0
              let dvar_abs := (Numbers.two * G) / nA
              let d2var_abs := (Numbers.two * H) / nA
              let dt_abs := (Numbers.pointfive * t3_hi) * dvar_abs
              let d2t_abs := (Numbers.pointfive * Numbers.three * t5_hi) * (dvar_abs * dvar_abs) +
                (Numbers.pointfive * t3_hi) * d2var_abs
              -- base = t * d2v
              let base_lo :=
                Tensor.dim (fun i =>
                  match d2vLoFn i, d2vHiFn i with
                  | .scalar l, .scalar u =>
                    let p1 := t_lo * l; let p2 := t_lo * u
                    let p3 := t_hi * l; let p4 := t_hi * u
                    let m1 := if p1 < p2 then p1 else p2
                    let m2 := if p3 < p4 then p3 else p4
                    Tensor.scalar (if m1 < m2 then m1 else m2))
              let base_hi :=
                Tensor.dim (fun i =>
                  match d2vLoFn i, d2vHiFn i with
                  | .scalar l, .scalar u =>
                    let p1 := t_lo * l; let p2 := t_lo * u
                    let p3 := t_hi * l; let p4 := t_hi * u
                    let M1 := if p1 > p2 then p1 else p2
                    let M2 := if p3 > p4 then p3 else p4
                    Tensor.scalar (if M1 > M2 then M1 else M2))
              -- inflate by 2|dt||v_i| + |d2t||u_i|
              let baseLoFn := getDimScalarFn (α:=α) base_lo
              let baseHiFn := getDimScalarFn (α:=α) base_hi
              let lo :=
                Tensor.dim (fun i =>
                  match baseLoFn i, uLoFn i, uHiFn i, vLoFn i, vHiFn i with
                  | .scalar bi, .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                    let au := abs_max ul uu; let av := abs_max vl vu
                    Tensor.scalar (bi - (Numbers.two * dt_abs * av) - (d2t_abs * au)))
              let hi :=
                Tensor.dim (fun i =>
                  match baseHiFn i, uLoFn i, uHiFn i, vLoFn i, vHiFn i with
                  | .scalar bi, .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                    let au := abs_max ul uu; let av := abs_max vl vu
                    Tensor.scalar (bi + (Numbers.two * dt_abs * av) + (d2t_abs * au)))
              d2s.set! id (some { dim := n, lo := lo, hi := hi })
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      d2s
    | .conv2d .. => d2s
  (List.finRange g.nodes.size).foldl propagate init

/--
Context for affine (CROWN/DeepPoly) propagation.

Affine bounds are computed with respect to a single designated *input* node, whose flattened
dimension is `inputDim`.
-/
structure AffineCtx where
  /-- Node id treated as the input variable for affine bounds. -/
  inputId  : Nat
  /-- Flattened input dimension. -/
  inputDim : Nat

private def affIdentity (n : Nat) : AffineVec α n n :=
  let A :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then 1 else 0)))
  let c := Spec.fill (α:=α) 0 (.dim n .scalar)
  { A := A, c := c }

private def affAdd {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.addSpec a1.A a2.A, c := Tensor.addSpec a1.c a2.c }

private def affSub {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.subSpec a1.A a2.A, c := Tensor.subSpec a1.c a2.c }

private def affScale {n m : Nat} (s : α) (a : AffineVec α n m) : AffineVec α n m :=
  let A' :=
    match a.A with
    | .dim rows =>
      Tensor.dim (fun i =>
        match rows i with
        | .dim cols => Tensor.dim (fun j => match cols j with | .scalar v => Tensor.scalar (s * v)))
  let c' :=
    match a.c with
    | .dim cv => Tensor.dim (fun i => match cv i with | .scalar v => Tensor.scalar (s * v))
  { A := A', c := c' }

-- Affine helpers for linear/matmul are handled by the explicit transfer rules below.

private def reluRelaxFromBox (B : FlatBox α) : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α)
  (.dim B.dim .scalar) :=
  NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=B.dim) B.lo B.hi

private def affThroughRelu {inDim hidDim : Nat}
  (relax : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim hidDim .scalar))
  (aff : AffineVec α inDim hidDim) : AffineVec α inDim hidDim :=
  NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α) (inDim:=inDim) (hidDim:=hidDim) relax
    aff

private def affOfLinear (p : LinParams α) : AffineVec α p.n p.m :=
  AffineVec.ofLinear (α:=α) (inDim:=p.n) (outDim:=p.m) p.w p.b

private def affOfMatmul (p : MatParams α) : AffineVec α p.n p.m :=
  let zb := Spec.fill (α:=α) 0 (.dim p.m .scalar)
  AffineVec.ofLinear (α:=α) (inDim:=p.n) (outDim:=p.m) p.w zb

private def affOfConv2d (cfg : Conv2DParams α) :
  let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
  let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
  AffineVec α (cfg.inC * cfg.inH * cfg.inW) (cfg.outC * outH * outW) :=
  let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
  let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
  let inShape := Shape.dim cfg.inC (Shape.dim cfg.inH (Shape.dim cfg.inW Shape.scalar))
  let outShape := Shape.dim cfg.outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  let nIn := inShape.size
  let nOut := outShape.size
  have hIn' : nIn = cfg.inC * (cfg.inH * cfg.inW) := by
    simp [nIn, inShape, Shape.size]
  have hIn : nIn = cfg.inC * cfg.inH * cfg.inW := by
    simpa [Nat.mul_assoc] using hIn'
  have hOut' : nOut = cfg.outC * (outH * outW) := by
    simp [nOut, outShape, Shape.size, outH, outW]
  have hOut : nOut = cfg.outC * outH * outW := by
    simpa [Nat.mul_assoc] using hOut'
  let Wraw := NN.MLTheory.CROWN.conv2dLinearMatrix (α:=α)
    (inC:=cfg.inC) (outC:=cfg.outC) (kH:=cfg.kH) (kW:=cfg.kW)
    (stride:=cfg.stride) (padding:=cfg.padding)
    (inH:=cfg.inH) (inW:=cfg.inW) cfg.spec
  let bRaw := NN.MLTheory.CROWN.conv2dBiasBroadcast (α:=α)
    (outC:=cfg.outC) (inH:=cfg.inH) (inW:=cfg.inW)
    (kH:=cfg.kH) (kW:=cfg.kW)
    (stride:=cfg.stride) (padding:=cfg.padding) cfg.spec.bias
  have hShapeW : Shape.dim nOut (Shape.dim nIn Shape.scalar) =
      Shape.dim (cfg.outC * outH * outW) (Shape.dim (cfg.inC * cfg.inH * cfg.inW) Shape.scalar) :=
        by
    simp [hIn, hOut]
  have hShapeB : Shape.dim nOut Shape.scalar = Shape.dim (cfg.outC * outH * outW) Shape.scalar := by
    simp [hOut]
  let W := Spec.tensorCast
    (Shape.dim (cfg.outC * outH * outW) (Shape.dim (cfg.inC * cfg.inH * cfg.inW) Shape.scalar))
    hShapeW Wraw
  let b := Spec.tensorCast
    (Shape.dim (cfg.outC * outH * outW) Shape.scalar)
    hShapeB bRaw
  AffineVec.ofLinear (α:=α)
    (inDim:=cfg.inC * cfg.inH * cfg.inW)
    (outDim:=cfg.outC * outH * outW)
    W b

/--
Propagate a single node’s affine form (CROWN/DeepPoly style) given parent affine forms.

This updates the `affs` array at index `id` when the node kind admits an affine transfer rule.
For non-affine nodes (or missing parents/params), the array is left unchanged so downstream code
can fall back to IBP boxes.
-/
def propagateAffineNode
  (nodes : Array Node) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α)))
  (affs : Array (Option (FlatAffine α)))
  (ctx : AffineCtx) (id : Nat) : Array (Option (FlatAffine α)) :=
  let node := nodes[id]!
  let getAff (pid : Nat) := (affs[pid]!)
  match node.kind with
  | .input =>
    if node.id = ctx.inputId then
      let aff := affIdentity (α:=α) ctx.inputDim
      affs.set! id (some { inDim := ctx.inputDim, outDim := ctx.inputDim, aff := aff })
    else affs
  | .const _ =>
    -- Lift constant to an affine with zero A and constant c; use ctx.inputDim for input width
    match ps.constVals[id]? with
    | some v =>
      let zA := Spec.fill (α:=α) 0 (.dim v.n (.dim ctx.inputDim .scalar))
      let aff : AffineVec α ctx.inputDim v.n := { A := zA, c := v.v }
      affs.set! id (some { inDim := ctx.inputDim, outDim := v.n, aff := aff })
    | none => affs
  | .detach =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1 with
      | some a => affs.set! id (some a)
      | none => affs
    | _ => affs
  | .randUniform _ | .bernoulliMask _ =>
    -- Stochastic nodes are treated as non-affine; downstream passes can fall back to IBP boxes.
    affs
  | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
    -- Pooling is non-affine; downstream passes can fall back to IBP boxes.
    affs
  | .add =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getAff p1, getAff p2 with
      | some a1, some a2 =>
        if hout : a1.outDim = a2.outDim then
          if hin : a1.inDim = a2.inDim then
            let a2' := castAffineIn (α:=α) (n:=a2.inDim) (n':=a1.inDim) (m:=a2.outDim) hin.symm
              a2.aff
            let a2'' := castAffineOut (α:=α) (n:=a1.inDim) (m:=a2.outDim) (m':=a1.outDim) hout.symm
              a2'
            let outAff := affAdd (α:=α) (n:=a1.inDim) (m:=a1.outDim) a1.aff a2''
            affs.set! id (some { inDim := a1.inDim, outDim := a1.outDim, aff := outAff })
          else affs
        else affs
      | _, _ => affs
    | _ => affs
  | .sub =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getAff p1, getAff p2 with
      | some a1, some a2 =>
        if hout : a1.outDim = a2.outDim then
          if hin : a1.inDim = a2.inDim then
            let a2' := castAffineIn (α:=α) (n:=a2.inDim) (n':=a1.inDim) (m:=a2.outDim) hin.symm
              a2.aff
            let a2'' := castAffineOut (α:=α) (n:=a1.inDim) (m:=a2.outDim) (m':=a1.outDim) hout.symm
              a2'
            let outAff := affSub (α:=α) (n:=a1.inDim) (m:=a1.outDim) a1.aff a2''
            affs.set! id (some { inDim := a1.inDim, outDim := a1.outDim, aff := outAff })
          else affs
        else affs
      | _, _ => affs
    | _ => affs
  | .relu =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[p1]! with
      | some paff, some preB =>
        if hdim : preB.dim = paff.outDim then
          let relax0 := reluRelaxFromBox (α:=α) preB
          let relax := castRelax (α:=α) hdim relax0
          let outAff := affThroughRelu (α:=α) (inDim:=paff.inDim) (hidDim:=paff.outDim) relax
            paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else
          affs
      | _, _ => affs
    | _ => affs
  | .linear =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ps.linearWB[id]? with
      | some paff, some p =>
        if hdim : paff.outDim = p.n then
          let wbaff0 := affOfLinear (α:=α) p
          let wbaff  := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm wbaff0
          let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) wbaff
            paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
        else
          affs
      | _, _ => affs
    | _ => affs
  | .matmul =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ps.matmulW[id]? with
      | some paff, some p =>
        if hdim : paff.outDim = p.n then
          let waff0 := affOfMatmul (α:=α) p
          let waff  := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm waff0
          let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) waff
            paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
        else
          affs
      | _, _ => affs
    | _ => affs
  | .sum =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1 with
      | some paff =>
        let onesRow : Tensor α (.dim 1 (.dim paff.outDim .scalar)) :=
          Spec.fill (α := α) Numbers.one (.dim 1 (.dim paff.outDim .scalar))
        let outAff : AffineVec α paff.inDim 1 :=
          { A := Spec.matMulSpec onesRow paff.aff.A
            c := Spec.matVecMulSpec onesRow paff.aff.c }
        affs.set! id (some { inDim := paff.inDim, outDim := 1, aff := outAff })
      | none => affs
    | _ => affs
  | .reshape _ _ => affs
  | .flatten _ => affs
  | .swap_first_two => affs
  | .transpose3dLastTwo => affs
  | .permute _ => affs
  | .mseLoss => affs
  | .mul_elem =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getAff p1, getAff p2, ibp[p1]!, ibp[p2]! with
      | some ax, some ay, some Bx, some By =>
        -- Require matching output dims and input dims; otherwise skip
        if hout : ax.outDim = ay.outDim then
          if hin : ax.inDim = ay.inDim then
            if hbx : Bx.dim = ax.outDim then
              if hby : By.dim = ay.outDim then
                let ayOut := castAffineOut (α:=α) (n:=ay.inDim) (m:=ay.outDim) (m':=ax.outDim)
                  (h:=hout.symm) ay.aff
                let ayAligned := castAffineIn (α:=α) (n:=ay.inDim) (n':=ax.inDim) (m:=ax.outDim)
                  (h:=hin.symm) ayOut
                let bxBox := castBoxDim (α:=α) (n:=Bx.dim) (n':=ax.outDim) (h:=hbx) (ofFlatBox Bx)
                -- align By box dim to ax.outDim via ay.outDim using hout
                let hby2 : By.dim = ax.outDim := Eq.trans hby hout.symm
                let byBox := castBoxDim (α:=α) (n:=By.dim) (n':=ax.outDim) (h:=hby2) (ofFlatBox By)
                -- McCormick upper affine envelope per component i
                let A' :=
                  match ax.aff.A, ayAligned.A, bxBox.lo, bxBox.hi, byBox.lo, byBox.hi with
                  | .dim rowsX, .dim rowsY, .dim lox, .dim hix, .dim loy, .dim hiy =>
                    Tensor.dim (fun i =>
                      let rowX := rowsX i
                      let rowY := rowsY i
                      match rowX, rowY, lox i, hix i, loy i, hiy i with
                      | .dim colsX, .dim colsY,
                        .scalar lx, .scalar ux,
                        .scalar ly, .scalar uy =>
                        let cx := (lx + ux) * Numbers.pointfive
                        let cy := (ly + uy) * Numbers.pointfive
                        let u1_center := ux * cy + ly * cx - ux * ly
                        let u2_center := lx * cy + uy * cx - lx * uy
                        let sX := if u1_center < u2_center then ly else uy
                        let sY := if u1_center < u2_center then ux else lx
                        Tensor.dim (fun j =>
                          match colsX j, colsY j with
                          | .scalar aijx, .scalar aijy => Tensor.scalar (sX * aijx + sY * aijy)))
                let c' :=
                  match ax.aff.c, ayAligned.c, bxBox.lo, bxBox.hi, byBox.lo, byBox.hi with
                  | .dim cxv, .dim cyv, .dim lox, .dim hix, .dim loy, .dim hiy =>
                    Tensor.dim (fun i =>
                      match cxv i, cyv i, lox i, hix i, loy i, hiy i with
                      | .scalar cxi, .scalar cyi,
                        .scalar lx, .scalar ux,
                        .scalar ly, .scalar uy =>
                        let cx := (lx + ux) * Numbers.pointfive
                        let cy := (ly + uy) * Numbers.pointfive
                        let u1_center := ux * cy + ly * cx - ux * ly
                        let u2_center := lx * cy + uy * cx - lx * uy
                        let sX := if u1_center < u2_center then ly else uy
                        let sY := if u1_center < u2_center then ux else lx
                        let off := if u1_center < u2_center then (-(ux * ly)) else (-(lx * uy))
                        Tensor.scalar (sX * cxi + sY * cyi + off))
                let outAff : AffineVec α ax.inDim ax.outDim := { A := A', c := c' }
                affs.set! id (some { inDim := ax.inDim, outDim := ax.outDim, aff := outAff })
              else affs
            else affs
          else affs
        else affs
      | _, _, _, _ => affs
    | _ => affs
  | .conv2d .. =>
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ps.conv2dCfg[id]? with
      | some paff, some cfg =>
        let convIn := cfg.inC * cfg.inH * cfg.inW
        if hdim : paff.outDim = convIn then
          let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
          let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
          let convAff0 := affOfConv2d (α:=α) cfg
          let convAff := castAffineIn (α:=α)
            (n:=convIn) (n':=paff.outDim) (m:=cfg.outC * outH * outW)
            hdim.symm convAff0
          let composed := AffineVec.compose (α:=α)
            (n:=paff.inDim) (h:=paff.outDim) (m:=cfg.outC * outH * outW)
            convAff paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := cfg.outC * outH * outW, aff :=
            composed })
        else affs
      | some paff, none =>
        match ps.linearWB[id]? with
        | some p =>
          if hdim : paff.outDim = p.n then
            let wbaff0 := affOfLinear (α:=α) p
            let wbaff := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm wbaff0
            let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) wbaff
              paff.aff
            affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
          else affs
        | none => affs
      | _, _ => affs
    | _ => affs
  | .exp =>
    -- Use a simple linear upper envelope over [l,u]: secant line of exp
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[p1]! with
      | some paff, some preB =>
        if hdim : preB.dim = paff.outDim then
          let preB' : Box α (.dim paff.outDim .scalar) := castBoxDim (α:=α) (n:=preB.dim)
            (n':=paff.outDim) hdim (ofFlatBox preB)
          let flo := getDimScalarFn (α:=α) preB'.lo
          let fhi := getDimScalarFn (α:=α) preB'.hi
          -- Build diagonal scaling and bias to approximate y ≈ a*x + b per-component
          let A' :=
            match paff.aff.A with
            | .dim rows =>
              Tensor.dim (fun i =>
                let li := match flo i with | .scalar v => v
                let ui := match fhi i with | .scalar v => v
                let den := ui - li
                let ai :=
                  if den > Numbers.epsilon then (MathFunctions.exp ui - MathFunctions.exp li) / den
                  else MathFunctions.exp li
                match rows i with
                | .dim cols =>
                    Tensor.dim (fun j =>
                      match cols j with
                      | .scalar aij => Tensor.scalar (ai * aij)))
          let c' :=
            match paff.aff.c with
            | .dim cv =>
              Tensor.dim (fun i =>
                let li := match flo i with | .scalar v => v
                let ui := match fhi i with | .scalar v => v
                let den := ui - li
                let ai :=
                  if den > Numbers.epsilon then (MathFunctions.exp ui - MathFunctions.exp li) / den
                  else MathFunctions.exp li
                let bi := MathFunctions.exp li - ai * li
                match cv i with | .scalar ci => Tensor.scalar (ai * ci + bi))
          let outAff : AffineVec α paff.inDim paff.outDim := { A := A', c := c' }
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs
  | .log =>
    -- `log` is concave (on its positive domain), so a tangent line is a sound *upper* affine bound.
    -- This affine pass tracks a single affine form per node (not separate lower/upper forms),
    -- so we only build an upper-style linearization here.
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[p1]! with
      | some paff, some preB =>
        if hdim : preB.dim = paff.outDim then
          let preB' : Box α (.dim paff.outDim .scalar) := castBoxDim (α:=α) (n:=preB.dim)
            (n':=paff.outDim) hdim (ofFlatBox preB)
          let flo := getDimScalarFn (α:=α) preB'.lo
          -- Choose t = clamp(li, eps) per component for tangent
            let A' :=
              match paff.aff.A with
              | .dim rows =>
                Tensor.dim (fun i =>
                  let li :=
                    match flo i with
                    | .scalar v => if v > Numbers.epsilon then v else Numbers.epsilon
                  let ai := Numbers.one / li  -- derivative of log at li
                  match rows i with
                  | .dim cols =>
                      Tensor.dim (fun j =>
                        match cols j with
                        | .scalar aij => Tensor.scalar (ai * aij)))
          let c' :=
            match paff.aff.c with
            | .dim cv =>
              Tensor.dim (fun i =>
                let li := match flo i with | .scalar v => (if v > Numbers.epsilon then v else
                  Numbers.epsilon)
                let ai := (Numbers.one / li)
                let bi := MathFunctions.log li - ai * li
                match cv i with | .scalar ci => Tensor.scalar (ai * ci + bi))
          let outAff : AffineVec α paff.inDim paff.outDim := { A := A', c := c' }
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs
  | .softmax _ =>
    -- Upper affine envelope per component k:
    -- softmax_k(x) = exp(x_k) / Σ_j exp(x_j) ≤ (a_k x + b_k) / total_lo,
    -- where a_k,b_k are the secant upper of exp on [l_k,u_k], and
    -- total_lo = Σ_j exp(l_j) > 0 is a scalar lower bound on the denominator.
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[p1]! with
      | some paff, some preB =>
        if hdim : preB.dim = paff.outDim then
          let preB' : Box α (.dim paff.outDim .scalar) := castBoxDim (α:=α) (n:=preB.dim)
            (n':=paff.outDim) hdim (ofFlatBox preB)
          let flo := getDimScalarFn (α:=α) preB'.lo
          let fhi := getDimScalarFn (α:=α) preB'.hi
          -- Scalar lower bound on denominator: sum_j exp(l_j)
          let exp_lo := Tensor.expSpec preB'.lo
          let total_lo := Spec.Tensor.sumSpec exp_lo
          let invDen := Numbers.one / (if total_lo > Numbers.epsilon then total_lo else
            Numbers.epsilon)
          -- Build per-row scaled A and c for numerator upper (exp secant), then divide by denom
          -- lower
          let A' :=
            match paff.aff.A with
            | .dim rows =>
              Tensor.dim (fun i =>
                let li := match flo i with | .scalar v => v
                let ui := match fhi i with | .scalar v => v
                let den := ui - li
                let ai :=
                  if den > Numbers.epsilon then (MathFunctions.exp ui - MathFunctions.exp li) / den
                  else MathFunctions.exp li
                match rows i with
                | .dim cols =>
                    Tensor.dim (fun j =>
                      match cols j with
                      | .scalar aij => Tensor.scalar (invDen * (ai * aij))))
          let c' :=
            match paff.aff.c with
            | .dim cv =>
              Tensor.dim (fun i =>
                let li := match flo i with | .scalar v => v
                let ui := match fhi i with | .scalar v => v
                let den := ui - li
                let ai :=
                  if den > Numbers.epsilon then (MathFunctions.exp ui - MathFunctions.exp li) / den
                  else MathFunctions.exp li
                let bi := MathFunctions.exp li - ai * li
                match cv i with | .scalar ci => Tensor.scalar (invDen * (ai * ci + bi)))
          let outAff : AffineVec α paff.inDim paff.outDim := { A := A', c := c' }
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs
  | .layernorm _ =>
    -- Upper affine envelope for layernorm using decomposition:
    -- y_i = (x_i - mean(x)) * t, with t = 1 / sqrt(var(x) + eps) ∈ [t_lo, t_hi].
    -- For all t in [t_lo, t_hi], u := (x_i - mean(x)) satisfies
    -- u * t ≤ t_lo * u + (t_hi - t_lo) * ReLU(u).
    -- We compute an exact affine for u and an upper affine for ReLU(u), then combine.
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[p1]! with
      | some paff, some preB =>
        if hdim : preB.dim = paff.outDim then
          let n := paff.outDim
          -- Compute bounds for mean and centered components as in IBP
          let preB' : Box α (.dim n .scalar) := castBoxDim (α:=α) (n:=preB.dim) (n':=n) hdim
            (ofFlatBox preB)
          let sum_lo := Spec.Tensor.sumSpec preB'.lo
          let sum_hi := Spec.Tensor.sumSpec preB'.hi
          let nA : α := (n : Nat)
          let mu_lo := sum_lo / nA
          let mu_hi := sum_hi / nA
          let flo := getDimScalarFn (α:=α) preB'.lo
          let fhi := getDimScalarFn (α:=α) preB'.hi
          -- Upper bound on variance as before
          let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
            match flo i, fhi i with
            | .scalar l, .scalar u =>
              let dl := MathFunctions.abs (l - mu_hi)
              let du := MathFunctions.abs (u - mu_lo)
              let a := if dl > du then dl else du
              acc + (a * a)
          ) 0
          let var_hi := sumAbsSq / nA
          let s_lo := MathFunctions.sqrt Numbers.epsilon
          let s_hi := MathFunctions.sqrt (var_hi + Numbers.epsilon)
          let t_lo := Numbers.one / (if s_hi > Numbers.epsilon then s_hi else Numbers.epsilon)
          let t_hi := Numbers.one / (if s_lo > Numbers.epsilon then s_lo else Numbers.epsilon)
          -- Build linear centering transform S = I - (1/n) 11^T, as an AffineVec
          let S : Tensor α (.dim n (.dim n .scalar)) :=
            Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then
              (Numbers.one - (Numbers.one / nA)) else (-(Numbers.one / nA)))))
          let b0 := Spec.fill (α:=α) 0 (.dim n .scalar)
          let Saff : AffineVec α n n := { A := S, c := b0 }
          let u_aff : AffineVec α paff.inDim n :=
            AffineVec.compose (α:=α) (n:=paff.inDim) (h:=n) (m:=n) Saff paff.aff
          -- Compute bounds for u per component for ReLU relaxation
          let ulo :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar l, .scalar u =>
                let dl := l - mu_hi
                let du := u - mu_lo
                let mn := if dl < du then dl else du
                Tensor.scalar mn)
          let uhi :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar l, .scalar u =>
                let dl := l - mu_hi
                let du := u - mu_lo
                let mx := if dl > du then dl else du
                Tensor.scalar mx)
          let u_box : FlatBox α := { dim := n, lo := ulo, hi := uhi }
          let u_relax := reluRelaxFromBox (α:=α) u_box
          let relu_u_aff := affThroughRelu (α:=α) (inDim:=paff.inDim) (hidDim:=n) (relax:=u_relax)
            (aff:=u_aff)
          -- Combine: t_lo * u + (t_hi - t_lo) * ReLU(u)
          let outAff := affAdd (α:=α) (n:=paff.inDim) (m:=n) (affScale (α:=α) (n:=paff.inDim)
            (m:=n) t_lo u_aff)
                                   (affScale (α:=α) (n:=paff.inDim) (m:=n) (t_hi - t_lo)
                                     relu_u_aff)
          affs.set! id (some { inDim := paff.inDim, outDim := n, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs
  | .concat axis =>
    -- Implement concat for axis=0 in flattened (vector) space: just stack output rows.
    -- For other axes/shapes, this requires stride-aware flatten/reshape bookkeeping.
    if axis != 0 then affs
    else
      let rec collect (ps : List Nat) (acc : List (FlatAffine α)) : Option (List (FlatAffine α)) :=
        match ps with
        | [] => some acc.reverse
        | p :: ps =>
          match getAff p with
          | some a => collect ps (a :: acc)
          | none => none
      match collect node.parents [] with
      | none => affs
      | some parentsAff =>
        match parentsAff with
        | [] => affs
        | first :: rest =>
          let inDim := first.inDim
          if rest.all (fun a => a.inDim == inDim) then
            let totalOut := parentsAff.foldl (fun acc a => acc + a.outDim) 0
            if Shape.size node.outShape = totalOut then
              let rec pick (k : Nat) (l : List (FlatAffine α)) : FlatAffine α × Nat :=
                match l with
                | [] => (first, 0)
                | a :: tl =>
                  if k < a.outDim then (a, k) else pick (k - a.outDim) tl
              let A' : Tensor α (.dim totalOut (.dim inDim .scalar)) :=
                Tensor.dim (fun i =>
                  let (a, k) := pick i.val parentsAff
                  Tensor.dim (fun j => Tensor.scalar (getAtOrZero a.aff.A [k, j.val])))
              let c' : Tensor α (.dim totalOut .scalar) :=
                Tensor.dim (fun i =>
                  let (a, k) := pick i.val parentsAff
                  Tensor.scalar (getAtOrZero a.aff.c [k]))
              let outAff : AffineVec α inDim totalOut := { A := A', c := c' }
              affs.set! id (some { inDim := inDim, outDim := totalOut, aff := outAff })
            else affs
          else affs
  | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean ..
    =>
    affs
  | .tanh =>
    -- Conservative upper affine: y_i ≤ y_hi[i] as a constant (safe, improves with tighter IBP)
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[id]! with
      | some paff, some yB =>
        if hdim : yB.dim = paff.outDim then
          let yB' : Box α (.dim paff.outDim .scalar) := castBoxDim (α:=α) (n:=yB.dim)
            (n':=paff.outDim) hdim (ofFlatBox yB)
          -- Zero A; bias equals y_hi per component
          let A' :=
            match paff.aff.A with
            | .dim rows =>
              Tensor.dim (fun i =>
                match rows i with
                | .dim _cols => Tensor.dim (fun _ => Tensor.scalar 0))
          let c' :=
            match yB'.hi with
            | .dim hv => Tensor.dim (fun i => match hv i with | .scalar v => Tensor.scalar v)
          let outAff : AffineVec α paff.inDim paff.outDim := { A := A', c := c' }
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs
  | .sin =>
    -- Conservative upper affine: y_i ≤ y_hi[i] as a constant
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[id]! with
      | some paff, some yB =>
        if hdim : yB.dim = paff.outDim then
          let yB' : Box α (.dim paff.outDim .scalar) :=
            castBoxDim (α:=α) (n:=yB.dim) (n':=paff.outDim) hdim (ofFlatBox yB)
          let A' :=
            match paff.aff.A with
            | .dim rows =>
              Tensor.dim (fun i =>
                match rows i with
                | .dim _cols => Tensor.dim (fun _ => Tensor.scalar 0))
          let c' :=
            match yB'.hi with
            | .dim hv => Tensor.dim (fun i => match hv i with | .scalar v => Tensor.scalar v)
          let outAff : AffineVec α paff.inDim paff.outDim := { A := A', c := c' }
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs
  | .cos =>
    -- Conservative upper affine: y_i ≤ y_hi[i] as a constant
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[id]! with
      | some paff, some yB =>
        if hdim : yB.dim = paff.outDim then
          let yB' : Box α (.dim paff.outDim .scalar) :=
            castBoxDim (α:=α) (n:=yB.dim) (n':=paff.outDim) hdim (ofFlatBox yB)
          let A' :=
            match paff.aff.A with
            | .dim rows =>
              Tensor.dim (fun i =>
                match rows i with
                | .dim _cols => Tensor.dim (fun _ => Tensor.scalar 0))
          let c' :=
            match yB'.hi with
            | .dim hv => Tensor.dim (fun i => match hv i with | .scalar v => Tensor.scalar v)
          let outAff : AffineVec α paff.inDim paff.outDim := { A := A', c := c' }
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs
  | .sigmoid =>
    -- Conservative upper affine: y_i ≤ y_hi[i] as a constant
    match node.parents with
    | p1 :: _ =>
      match getAff p1, ibp[id]! with
      | some paff, some yB =>
        if hdim : yB.dim = paff.outDim then
          let yB' : Box α (.dim paff.outDim .scalar) := castBoxDim (α:=α) (n:=yB.dim)
            (n':=paff.outDim) hdim (ofFlatBox yB)
          let A' :=
            match paff.aff.A with
            | .dim rows =>
              Tensor.dim (fun i =>
                match rows i with
                | .dim _cols => Tensor.dim (fun _ => Tensor.scalar 0))
          let c' :=
            match yB'.hi with
            | .dim hv => Tensor.dim (fun i => match hv i with | .scalar v => Tensor.scalar v)
          let outAff : AffineVec α paff.inDim paff.outDim := { A := A', c := c' }
          affs.set! id (some { inDim := paff.inDim, outDim := paff.outDim, aff := outAff })
        else affs
      | _, _ => affs
    | _ => affs

/-- Run an affine pass; requires prior IBP to supply pre-activation bounds for ReLU. -/
def runAffine (g : Graph) (ps : ParamStore α) (ctx : AffineCtx) (ibp : Array (Option (FlatBox α))) :
  Array (Option (FlatAffine α)) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl (fun acc i => propagateAffineNode (α:=α) g.nodes ps ibp acc ctx
    i) init

/-!
Basic CROWN-style affine bounds pass (lower + upper)
===================================================

This pass computes a *pair* of affine forms per node:

  loAff(x) ≤ node(x) ≤ hiAff(x)

where both affines are with respect to a designated flattened input node `ctx.inputId`.

This pass is a practical CROWN/DeepPoly implementation:
- Linear layers use sign-splitting (`W⁺/W⁻`) to combine parent bounds.
- ReLU uses the standard triangle upper bound and a simple evidence-based lower choice (0 vs x).
- Exp/log use secant/tangent bounds (convex/concave).
- Softmax/LayerNorm have conservative last-axis affine relaxations; on unsupported axes or
  shape mismatches we fall back to *constant* affine bounds derived from the IBP box.

The pass is structured so a soundness theorem can cite per-op transfer rules for the supported
relaxations. Unsupported axes or shape mismatches deliberately fall back to constant affine bounds
derived from the IBP box.
-/

private def boundsIdentity (n : Nat) : FlatAffineBounds α :=
  { inDim := n, outDim := n, loAff := affIdentity (α:=α) n, hiAff := affIdentity (α:=α) n }

private def boundsConst (inputDim outDim : Nat) (lo hi : Tensor α (.dim outDim .scalar)) :
  FlatAffineBounds α :=
  let zA := Spec.fill (α:=α) 0 (.dim outDim (.dim inputDim .scalar))
  { inDim := inputDim
    outDim := outDim
    loAff := { A := zA, c := lo }
    hiAff := { A := zA, c := hi } }

private def matPos {m n : Nat} (W : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n
  .scalar)) :=
  match W with
  | .dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | .dim cols =>
        Tensor.dim (fun j =>
          match cols j with
          | .scalar w => Tensor.scalar (if w > 0 then w else 0)))

private def matNeg {m n : Nat} (W : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n
  .scalar)) :=
  match W with
  | .dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | .dim cols =>
        Tensor.dim (fun j =>
          match cols j with
          | .scalar w => Tensor.scalar (if w > 0 then 0 else w)))

private def materializeAffineVec {inDim outDim : Nat} (a : AffineVec α inDim outDim) : AffineVec α
  inDim outDim :=
  { A := Tensor.materialize a.A, c := Tensor.materialize a.c }

private def propagateLinearBounds
  {n m : Nat}
  (W : Tensor α (.dim m (.dim n .scalar)))
  (b : Tensor α (.dim m .scalar))
  (xB : FlatAffineBounds α)
  (hout : xB.outDim = n) : FlatAffineBounds α := by
  -- Align parent affines to outDim=n.
  let xLo : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.loAff
  let xHi : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.hiAff
  let xLo := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := n) xLo
  let xHi := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := n) xHi
  let Wpos := matPos (α:=α) (m:=m) (n:=n) W
  let Wneg := matNeg (α:=α) (m:=m) (n:=n) W
  let A_hi :=
    Tensor.materialize <|
      Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xHi.A) (Spec.matMulSpec (α:=α) Wneg xLo.A)
  let c_hi :=
    Tensor.materialize <|
      Tensor.addSpec
        (Tensor.addSpec (Spec.matVecMulSpec (α:=α) Wpos xHi.c) (Spec.matVecMulSpec (α:=α)
          Wneg xLo.c))
        b
  let A_lo :=
    Tensor.materialize <|
      Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xLo.A) (Spec.matMulSpec (α:=α) Wneg xHi.A)
  let c_lo :=
    Tensor.materialize <|
      Tensor.addSpec
        (Tensor.addSpec (Spec.matVecMulSpec (α:=α) Wpos xLo.c) (Spec.matVecMulSpec (α:=α)
          Wneg xHi.c))
        b
  exact
    { inDim := xB.inDim
      outDim := m
      loAff := { A := A_lo, c := c_lo }
      hiAff := { A := A_hi, c := c_hi } }

private def affApplyDiag {inDim outDim : Nat}
  (slopes bias : Tensor α (.dim outDim .scalar))
  (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, aff.A, aff.c with
  | .dim sF, .dim bF, .dim rows, .dim cF =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i, rows i with
        | .scalar si, .dim cols =>
          Tensor.dim (fun j =>
            match cols j with
            | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, cF i, bF i with
        | .scalar si, .scalar ci, .scalar bi => Tensor.scalar (si * ci + bi))
    { A := A', c := c' }

-- Like `aff_apply_diag`, but handles mixed-sign slopes by selecting `xLo` vs `xHi` rowwise.
private def affApplyDiagSignedUpper {inDim outDim : Nat}
  (slopes bias : Tensor α (.dim outDim .scalar))
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, xLo.A, xHi.A, xLo.c, xHi.c with
  | .dim sF, .dim bF, .dim rowsL, .dim rowsU, .dim cL, .dim cU =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i with
        | .scalar si =>
          let row := if decide (si > Numbers.zero) then rowsU i else rowsL i
          match row with
          | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, bF i with
        | .scalar si, .scalar bi =>
          let ci := if decide (si > Numbers.zero) then cU i else cL i
          match ci with
          | .scalar cv => Tensor.scalar (si * cv + bi))
    { A := A', c := c' }

private def affApplyDiagSignedLower {inDim outDim : Nat}
  (slopes bias : Tensor α (.dim outDim .scalar))
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, xLo.A, xHi.A, xLo.c, xHi.c with
  | .dim sF, .dim bF, .dim rowsL, .dim rowsU, .dim cL, .dim cU =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i with
        | .scalar si =>
          let row := if decide (si > Numbers.zero) then rowsL i else rowsU i
          match row with
          | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, bF i with
        | .scalar si, .scalar bi =>
          let ci := if decide (si > Numbers.zero) then cL i else cU i
          match ci with
          | .scalar cv => Tensor.scalar (si * cv + bi))
    { A := A', c := c' }

-- One-dimensional sigmoid linear bounds on an interval `[l,u]`.
-- Returns (a_lo, b_lo, a_hi, b_hi) such that:
--   a_lo*x + b_lo ≤ σ(x) ≤ a_hi*x + b_hi  for all x ∈ [l,u].
private def sigmoidLineBounds (l u : α) : α × α × α × α :=
  let σ (x : α) := Activation.Math.sigmoidSpec (α := α) x
  let σ' (x : α) := Activation.Math.sigmoidDerivSpec (α := α) x
  if u < Numbers.zero then
    -- Convex region: secant is an upper bound, tangent is a lower bound.
    let den := u - l
    let a_hi := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
    let b_hi := σ l - a_hi * l
    let a_lo := σ' u
    let b_lo := σ u - a_lo * u
    (a_lo, b_lo, a_hi, b_hi)
  else if l > Numbers.zero then
    -- Concave region: tangent is an upper bound, secant is a lower bound.
    let a_hi := σ' l
    let b_hi := σ l - a_hi * l
    let den := u - l
    let a_lo := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
    let b_lo := σ l - a_lo * l
    (a_lo, b_lo, a_hi, b_hi)
  else
    -- Crossing the inflection: fall back to constant bounds.
    (Numbers.zero, σ l, Numbers.zero, σ u)

private def propagateReluBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α := by
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let relaxLo0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVectorLower (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) loAff
      hiAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) hiAff }

private def propagateReluBoundsWithAlpha
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim)
  (alpha : Tensor α (.dim preB.dim .scalar)) : FlatAffineBounds α := by
  -- Upper relaxation: standard secant/tight bounds (independent of α).
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  -- Lower relaxation: for crossing bounds l < 0 < u, use a provided per-neuron α ∈ [0,1]
  -- (line y ≥ α x), which is always sound for ReLU; stable regions override α.
  let clamp01 (x : α) : α :=
    let x0 := if x > Numbers.zero then x else Numbers.zero
    if x0 > Numbers.one then Numbers.one else x0
  let relaxLo0 : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim preB.dim .scalar) :=
    match preB.lo, preB.hi, alpha with
    | .dim lF, .dim uF, .dim aF =>
      Tensor.dim (fun i =>
        match lF i, uF i, aF i with
        | .scalar l, .scalar u, .scalar a =>
          if u > Numbers.zero then
            if l > Numbers.zero then
              Tensor.scalar { slope := Numbers.one, bias := Numbers.zero }
            else
              Tensor.scalar { slope := clamp01 a, bias := Numbers.zero }
          else
            Tensor.scalar { slope := Numbers.zero, bias := Numbers.zero })
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) loAff
      hiAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) hiAff }

private def propagateExpBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  -- Per-component [l,u]
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.exp u - MathFunctions.exp l) / den
          else MathFunctions.exp l
        Tensor.scalar a)
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.exp u - MathFunctions.exp l) / den
          else MathFunctions.exp l
        let b := MathFunctions.exp l - a * l
        Tensor.scalar b)
  -- Lower: tangent at l
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar l =>
        Tensor.scalar (MathFunctions.exp l))
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar l =>
        let a := MathFunctions.exp l
        Tensor.scalar (MathFunctions.exp l - a * l))
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

private def propagateLogBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  -- Clamp to positive domain.
  let loSafe : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar v => Tensor.scalar (if v > Numbers.epsilon then v else Numbers.epsilon))
  let hiSafe : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match fhi i with
      | .scalar v => Tensor.scalar (if v > Numbers.epsilon then v else Numbers.epsilon))
  let floS := getDimScalarFn (α:=α) loSafe
  let fhiS := getDimScalarFn (α:=α) hiSafe
  -- Upper: tangent at loSafe (concave ⇒ tangent is over-approx).
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i with
      | .scalar l => Tensor.scalar (Numbers.one / l))
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i with
      | .scalar l =>
        let a := Numbers.one / l
        Tensor.scalar (MathFunctions.log l - a * l))
  -- Lower: secant on [loSafe, hiSafe] (concave ⇒ secant is under-approx).
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i, fhiS i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.log u - MathFunctions.log l) / den
          else Numbers.one / l
        Tensor.scalar a)
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i, fhiS i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.log u - MathFunctions.log l) / den
          else Numbers.one / l
        let b := MathFunctions.log l - a * l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

private def propagateSigmoidBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let σ (x : α) := Activation.Math.sigmoidSpec (α:=α) x
  let σ' (x : α) := Activation.Math.sigmoidDerivSpec (α:=α) x
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        -- Convex for x ≤ 0, concave for x ≥ 0; crossing uses constant bounds.
        let a :=
          if u < Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
          else if l > Numbers.zero then
            σ' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
            σ l - a * l
          else if l > Numbers.zero then
            let a := σ' l
            σ l - a * l
          else
            -- Crossing: constant upper bound = σ(u)
            σ u
        Tensor.scalar b)
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            σ' u
          else if l > Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let a := σ' u
            σ u - a * u
          else if l > Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
            σ l - a * l
          else
            -- Crossing: constant lower bound = σ(l)
            σ l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

private def propagateTanhBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let t (x : α) := Activation.Math.tanhSpec (α:=α) x
  let t' (x : α) := Activation.Math.tanhDerivSpec (α:=α) x
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (t u - t l) / den else t' l
          else if l > Numbers.zero then
            t' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (t u - t l) / den else t' l
            t l - a * l
          else if l > Numbers.zero then
            let a := t' l
            t l - a * l
          else
            t u
        Tensor.scalar b)
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            t' u
          else if l > Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (t u - t l) / den else t' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let a := t' u
            t u - a * u
          else if l > Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (t u - t l) / den else t' l
            t l - a * l
          else
            t l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

private def permuteAffineOut {inDim outDim : Nat}
  (perm : Fin outDim → Fin outDim) (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match aff.A, aff.c with
  | .dim rows, .dim cvec =>
    { A := Tensor.dim (fun i => rows (perm i))
      c := Tensor.dim (fun i => cvec (perm i)) }

private def propagateSoftmaxBoundsLastAxis
  (s : Shape) (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.hiAff
  let m := lastDimLen s
  if m = 0 then
    { inDim := xB.inDim, outDim := preB.dim, loAff := xLo, hiAff := xHi }
  else if m = 1 then
    -- Each last-axis slice has length 1, so softmax is identically 1.
    let ones : Tensor α (.dim preB.dim .scalar) := Spec.fill (α := α) Numbers.one (.dim preB.dim
      .scalar)
    boundsConst (α := α) (inputDim := xB.inDim) (outDim := preB.dim) ones ones
  else
    let dim := preB.dim
    if dim % m = 0 then
      let expLo : Tensor α (.dim dim .scalar) := Tensor.expSpec preB.lo
      let expHi : Tensor α (.dim dim .scalar) := Tensor.expSpec preB.hi
      let groups : Nat := dim / m
      let totalExpLo : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sum : α := (List.range m).foldl (fun acc j => acc + getAtOrZero expLo [base + j]) 0
          Tensor.scalar sum)
      let totalExpHi : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sum : α := (List.range m).foldl (fun acc j => acc + getAtOrZero expHi [base + j]) 0
          Tensor.scalar sum)
      let flo := getDimScalarFn (α := α) preB.lo
      let fhi := getDimScalarFn (α := α) preB.hi
      -- Upper bound via logistic with C = Σ_{j≠i} exp(lo_j)
      let slopes_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpLo [g]
            let eLi := getAtOrZero expLo [i.val]
            let c := tot - eLi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (_aLo, _bLo, aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar aHi
            else
              Tensor.scalar Numbers.zero)
      let bias_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpLo [g]
            let eLi := getAtOrZero expLo [i.val]
            let c := tot - eLi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (_aLo, _bLo, aHi, bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar (bHi - aHi * logC)
            else
              Tensor.scalar Numbers.one)
      -- Lower bound via logistic with C = Σ_{j≠i} exp(hi_j)
      let slopes_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpHi [g]
            let eUi := getAtOrZero expHi [i.val]
            let c := tot - eUi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (aLo, _bLo, _aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar aLo
            else
              Tensor.scalar Numbers.zero)
      let bias_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpHi [g]
            let eUi := getAtOrZero expHi [i.val]
            let c := tot - eUi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (aLo, bLo, _aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar (bLo - aLo * logC)
            else
              Tensor.scalar Numbers.zero)
      let loAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_lo bias_lo xLo
      let hiAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_hi bias_hi xHi
      { inDim := xB.inDim, outDim := dim, loAff := loAff, hiAff := hiAff }
    else
      -- Shape mismatch: fall back to trivial [0,1] bounds.
      let zeros : Tensor α (.dim dim .scalar) := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
      let ones : Tensor α (.dim dim .scalar) := Spec.fill (α := α) Numbers.one (.dim dim .scalar)
      boundsConst (α := α) (inputDim := xB.inDim) (outDim := dim) zeros ones

private def propagateLayernormBoundsLastAxis
  (s : Shape) (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.hiAff
  let m := lastDimLen s
  if m = 0 then
    { inDim := xB.inDim, outDim := preB.dim, loAff := xLo, hiAff := xHi }
  else if m = 1 then
    -- Each slice has length 1: (x - mean)/sqrt(var+eps) = 0.
    let zeros : Tensor α (.dim preB.dim .scalar) := Spec.fill (α := α) Numbers.zero (.dim preB.dim
      .scalar)
    boundsConst (α := α) (inputDim := xB.inDim) (outDim := preB.dim) zeros zeros
  else
    let dim := preB.dim
    if dim % m = 0 then
      let groups : Nat := dim / m
      let mA : α := (m : Nat)
      let denLo : α := MathFunctions.sqrt Numbers.epsilon
      let muLoG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sumLo : α := (List.range m).foldl (fun acc j => acc + getAtOrZero preB.lo [base +
            j]) 0
          Tensor.scalar (sumLo / mA))
      let muHiG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sumHi : α := (List.range m).foldl (fun acc j => acc + getAtOrZero preB.hi [base +
            j]) 0
          Tensor.scalar (sumHi / mA))
      let denHiG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let muLo := getAtOrZero muLoG [g.val]
          let muHi := getAtOrZero muHiG [g.val]
          let sumAbsSq : α :=
            (List.range m).foldl (fun acc j =>
              let l := getAtOrZero preB.lo [base + j]
              let u := getAtOrZero preB.hi [base + j]
              let dl := MathFunctions.abs (l - muHi)
              let du := MathFunctions.abs (u - muLo)
              let a := if dl > du then dl else du
              acc + (a * a)) 0
          let varHi := sumAbsSq / mA
          Tensor.scalar (MathFunctions.sqrt (varHi + Numbers.epsilon)))
      let flo := getDimScalarFn (α := α) preB.lo
      let fhi := getDimScalarFn (α := α) preB.hi
      let slopes_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muLo := getAtOrZero muLoG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let uL :=
              let num := l - muLo
              let den := if decide (l > muLo) then denLo else denHi
              num / den
            let uU :=
              let num := u - muLo
              let den := if decide (u > muLo) then denLo else denHi
              num / den
            let denx := u - l
            let a := if denx > Numbers.epsilon then (uU - uL) / denx else Numbers.zero
            Tensor.scalar a)
      let bias_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muLo := getAtOrZero muLoG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let uL :=
              let num := l - muLo
              let den := if decide (l > muLo) then denLo else denHi
              num / den
            let uU :=
              let num := u - muLo
              let den := if decide (u > muLo) then denLo else denHi
              num / den
            let denx := u - l
            if denx > Numbers.epsilon then
              let a := (uU - uL) / denx
              Tensor.scalar (uL - a * l)
            else
              Tensor.scalar (if uL > uU then uL else uU))
      let slopes_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muHi := getAtOrZero muHiG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let lL :=
              let num := l - muHi
              let den := if decide (l > muHi) then denHi else denLo
              num / den
            let lU :=
              let num := u - muHi
              let den := if decide (u > muHi) then denHi else denLo
              num / den
            let denx := u - l
            let a := if denx > Numbers.epsilon then (lU - lL) / denx else Numbers.zero
            Tensor.scalar a)
      let bias_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muHi := getAtOrZero muHiG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let lL :=
              let num := l - muHi
              let den := if decide (l > muHi) then denHi else denLo
              num / den
            let lU :=
              let num := u - muHi
              let den := if decide (u > muHi) then denHi else denLo
              num / den
            let denx := u - l
            if denx > Numbers.epsilon then
              let a := (lU - lL) / denx
              Tensor.scalar (lL - a * l)
            else
              Tensor.scalar (if lL < lU then lL else lU))
      let loAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_lo bias_lo xLo
      let hiAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_hi bias_hi xHi
      { inDim := xB.inDim, outDim := dim, loAff := loAff, hiAff := hiAff }
    else
      -- Shape mismatch: conservative constant bounds from IBP on this op.
      let (flatLo, flatHi) :=
        ibpLayernormLastTensor (α := α) (s := .dim dim .scalar) preB.lo preB.hi
      boundsConst (α := α) (inputDim := xB.inDim) (outDim := dim) flatLo flatHi

private def propagateMatmulBounds
  (sA sB : Shape) (Bx By : FlatBox α)
  (aB bB : FlatAffineBounds α) :
  Option (FlatAffineBounds α) :=
  if hin : aB.inDim = bB.inDim then
    let inDim := aB.inDim
    let bLo : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.loAff
    let bHi : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.hiAff
    let split (a : α) : α × α :=
      if a > Numbers.zero then (a, Numbers.zero) else (Numbers.zero, a)
    let dims? : Option (Nat × Nat × Nat × Nat) :=
      match sA, sB with
      | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
        if k = k' then
          some (1, m, k, n)
        else
          none
      | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
        if hb : b = b' then
          match hb with
          | rfl =>
            if k = k' then
              some (b, m, k, n)
            else
              none
        else
          none
      | _, _ => none
    match dims? with
    | none => none
    | some (batch, m, k, n) =>
      let dimA := batch * m * k
      let dimB := batch * k * n
      let outDim := batch * m * n
      if Bx.dim = dimA ∧ aB.outDim = dimA then
        if By.dim = dimB ∧ bB.outDim = dimB then
          let block : Nat := m * n
          let strideA : Nat := m * k
          let strideB : Nat := k * n

          let termUpperCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL

          let termUpperConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL + off

          let termLowerCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU

          let termLowerConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let off := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU + off

          let A_hi : Tensor α (.dim outDim (.dim inDim .scalar)) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range k).foldl (fun acc kk =>
                    acc + termUpperCoeff (baseA + i * k + kk) (baseB + kk * n + j) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_hi : Tensor α (.dim outDim .scalar) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              let coeff :=
                (List.range k).foldl (fun acc kk =>
                  acc + termUpperConst (baseA + i * k + kk) (baseB + kk * n + j)
                ) 0
              Tensor.scalar coeff)

          let A_lo : Tensor α (.dim outDim (.dim inDim .scalar)) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range k).foldl (fun acc kk =>
                    acc + termLowerCoeff (baseA + i * k + kk) (baseB + kk * n + j) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_lo : Tensor α (.dim outDim .scalar) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              let coeff :=
                (List.range k).foldl (fun acc kk =>
                  acc + termLowerConst (baseA + i * k + kk) (baseB + kk * n + j)
                ) 0
              Tensor.scalar coeff)

          some
            { inDim := inDim
              outDim := outDim
              loAff := { A := A_lo, c := c_lo }
              hiAff := { A := A_hi, c := c_hi } }
        else
          none
      else
        none
  else
    none

private def propagateMulElemBounds
  (Bx By : FlatBox α)
  (xB yB : FlatAffineBounds α)
  (houtX : xB.outDim = Bx.dim) (houtY : yB.outDim = By.dim) :
  Option (FlatAffineBounds α) :=
  -- Require equal vector lengths and equal input widths.
  if hdim : Bx.dim = By.dim then
    if hin : xB.inDim = yB.inDim then
      let n := Bx.dim
      let hyo : yB.outDim = n := Eq.trans houtY (Eq.symm hdim)
      let hBy : By.dim = n := by simpa [n] using (Eq.symm hdim)
      let ByLo : Tensor α (.dim n .scalar) := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.lo
      let ByHi : Tensor α (.dim n .scalar) := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.hi

      let xLo : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.loAff
      let xHi : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.hiAff
      let yLo0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.loAff
      let yHi0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.hiAff
        let yLo : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yLo0
        let yHi : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yHi0

        -- Helper to split a scalar coefficient into (pos, neg).
        let split (a : α) : α × α := if a > Numbers.zero then (a, Numbers.zero) else (Numbers.zero,
          a)

        -- Build row-wise A/c for upper and lower using a single selected McCormick plane per
        -- component.
        let A_hi :=
          match xLo.A, xHi.A, yLo.A, yHi.A, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim AxL, .dim AxU, .dim AyL, .dim AyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, AxL i, AxU i, AyL i, AyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .dim rowXL, .dim rowXU, .dim rowYL, .dim rowYU =>
                -- Choose min of two upper planes at the interval center.
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let u1 := ux * cy + ly * cx - ux * ly
                let u2 := lx * cy + uy * cx - lx * uy
                let aX := if u1 < u2 then ly else uy     -- coeff for x
                let aY := if u1 < u2 then ux else lx     -- coeff for y
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.dim (fun j =>
                  match rowXL j, rowXU j, rowYL j, rowYU j with
                  | .scalar xl, .scalar xu, .scalar yl, .scalar yu =>
                    Tensor.scalar (aXpos * xu + aXneg * xl + aYpos * yu + aYneg * yl)))
        let c_hi :=
          match xLo.c, xHi.c, yLo.c, yHi.c, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim cxL, .dim cxU, .dim cyL, .dim cyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, cxL i, cxU i, cyL i, cyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .scalar cxl, .scalar cxu, .scalar cyl, .scalar cyu =>
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let u1 := ux * cy + ly * cx - ux * ly
                let u2 := lx * cy + uy * cx - lx * uy
                let aX := if u1 < u2 then ly else uy
                let aY := if u1 < u2 then ux else lx
                let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.scalar (aXpos * cxu + aXneg * cxl + aYpos * cyu + aYneg * cyl + off))
        let A_lo :=
          match xLo.A, xHi.A, yLo.A, yHi.A, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim AxL, .dim AxU, .dim AyL, .dim AyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, AxL i, AxU i, AyL i, AyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .dim rowXL, .dim rowXU, .dim rowYL, .dim rowYU =>
                -- Choose max of two lower planes at the interval center.
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let l1 := ux * cy + uy * cx - ux * uy
                let l2 := lx * cy + ly * cx - lx * ly
                let aX := if l1 > l2 then uy else ly     -- coeff for x
                let aY := if l1 > l2 then ux else lx     -- coeff for y
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.dim (fun j =>
                  match rowXL j, rowXU j, rowYL j, rowYU j with
                  | .scalar xl, .scalar xu, .scalar yl, .scalar yu =>
                    -- For lower bound, negative coeffs use the *upper* input bound.
                    Tensor.scalar (aXpos * xl + aXneg * xu + aYpos * yl + aYneg * yu)))
        let c_lo :=
          match xLo.c, xHi.c, yLo.c, yHi.c, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim cxL, .dim cxU, .dim cyL, .dim cyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, cxL i, cxU i, cyL i, cyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .scalar cxl, .scalar cxu, .scalar cyl, .scalar cyu =>
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let l1 := ux * cy + uy * cx - ux * uy
                let l2 := lx * cy + ly * cx - lx * ly
                let aX := if l1 > l2 then uy else ly
                let aY := if l1 > l2 then ux else lx
                let off := if l1 > l2 then (-(ux * uy)) else (-(lx * ly))
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.scalar (aXpos * cxl + aXneg * cxu + aYpos * cyl + aYneg * cyu + off))

        some
          { inDim := xB.inDim
            outDim := n
            loAff := { A := A_lo, c := c_lo }
            hiAff := { A := A_hi, c := c_hi } }
    else
      none
  else
    none

/--
Propagate a single node’s *affine bounds* (lower/upper) given parent bounds.

This is the CROWN/DeepPoly-style transfer step used by `runCROWN`. For node kinds without a
dedicated rule, we fall back to the IBP enclosure (turned into a constant affine bound).
-/
def propagateCROWNNode
  (nodes : Array Node) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α)))
  (bounds : Array (Option (FlatAffineBounds α)))
  (ctx : AffineCtx) (id : Nat) : Array (Option (FlatAffineBounds α)) :=
  let node := nodes[id]!
  let getB (pid : Nat) := (bounds[pid]!)
  match node.kind with
  | .input =>
    if node.id = ctx.inputId then
      bounds.set! id (some (boundsIdentity (α:=α) ctx.inputDim))
    else bounds
  | .const _ =>
    match ps.constVals[id]? with
    | some v =>
      -- Exact constant bounds.
      bounds.set! id (some (boundsConst (α:=α) ctx.inputDim v.n v.v v.v))
    | none => bounds
  | .detach =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some b => bounds.set! id (some b)
      | none => bounds
    | _ => bounds
  | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .permute _ | .maxElem | .minElem | .sin |
    .cos
  | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad ..
  | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
    -- Conservative fallback: use IBP box as a constant affine bound (A = 0).
    match ibp[id]! with
    | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
    | none => bounds
  | .add =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.loAff
                hiAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.hiAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .sub =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.hiAff
                hiAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.loAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .linear =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ps.linearWB[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w p.b xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .matmul =>
    match node.parents with
    | p1 :: p2 :: _ =>
      -- General (batched) matmul: use McCormick relaxations per product term.
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some aAff, some bAff, some aBox, some bBox =>
        match propagateMatmulBounds (α:=α) (sA := nodes[p1]!.outShape) (sB := nodes[p2]!.outShape)
              aBox bBox aAff bAff with
        | some out =>
          bounds.set! id (some out)
        | none =>
          match ibp[id]! with
          | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
            Bout.hi))
          | none => bounds
      | _, _, _, _ => bounds
    | p1 :: _ =>
      match getB p1, ps.matmulW[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let zb := Spec.fill (α:=α) 0 (.dim p.m .scalar)
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w zb xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .relu =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          let out := propagateReluBounds (α:=α) preB xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .exp =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateExpBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .log =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateLogBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .inv =>
    -- Reciprocal has an asymptote at 0; we conservatively fall back to a constant affine
    -- bound derived from IBP (which widens to ±1/ε when the interval crosses 0).
    match ibp[id]! with
    | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo Bout.hi))
    | none => bounds
  | .sigmoid =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateSigmoidBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .tanh =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateTanhBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .mul_elem =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some xB, some yB, some Bx, some By =>
        if hxo : xB.outDim = Bx.dim then
          if hyo : yB.outDim = By.dim then
            match propagateMulElemBounds (α:=α) Bx By xB yB hxo hyo with
            | some out => bounds.set! id (some out)
            | none =>
              match ibp[id]! with
              | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
                Bout.hi))
              | none => bounds
          else bounds
        else bounds
      | _, _, _, _ => bounds
    | _ => bounds
  | .sum =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        let onesRow : Tensor α (.dim 1 (.dim xin.outDim .scalar)) :=
          Spec.fill (α := α) Numbers.one (.dim 1 (.dim xin.outDim .scalar))
        let loAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.loAff.A
            c := Spec.matVecMulSpec onesRow xin.loAff.c }
        let hiAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.hiAff.A
            c := Spec.matVecMulSpec onesRow xin.hiAff.c }
        bounds.set! id (some { inDim := xin.inDim, outDim := 1, loAff := loAff, hiAff := hiAff })
      | none => bounds
    | _ => bounds
  | .reshape _ _ =>
    -- Flattened representation preserves order; treat as identity.
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .flatten _ =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .concat _ =>
    -- Exact concatenation on flattened vectors.
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hin : b1.inDim = b2.inDim then
          let b2Lo : AffineVec α b1.inDim b2.outDim :=
            castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
              b2.loAff
          let b2Hi : AffineVec α b1.inDim b2.outDim :=
            castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
              b2.hiAff
          match b1.loAff.A, b1.hiAff.A, b1.loAff.c, b1.hiAff.c, b2Lo.A, b2Hi.A, b2Lo.c, b2Hi.c with
          | .dim A1L, .dim A1U, .dim c1L, .dim c1U, .dim A2L, .dim A2U, .dim c2L, .dim c2U =>
            let outDim := b1.outDim + b2.outDim
            let ALo : Tensor α (.dim outDim (.dim b1.inDim .scalar)) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => A1L i1) (fun i2 => A2L i2) i)
            let AHi : Tensor α (.dim outDim (.dim b1.inDim .scalar)) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => A1U i1) (fun i2 => A2U i2) i)
            let cLo : Tensor α (.dim outDim .scalar) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => c1L i1) (fun i2 => c2L i2) i)
            let cHi : Tensor α (.dim outDim .scalar) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => c1U i1) (fun i2 => c2U i2) i)
            bounds.set! id
              (some
                { inDim := b1.inDim
                  outDim := outDim
                  loAff := { A := ALo, c := cLo }
                  hiAff := { A := AHi, c := cHi } })
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .swap_first_two =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match nodes[p1]!.outShape with
        | .dim m (.dim n rest) =>
          let sIn : Shape := .dim m (.dim n rest)
          if xin.outDim = sIn.size then
            let restSize := Shape.size rest
            let outDim := xin.outDim
            if h0 : outDim = 0 then
              -- Empty tensor: permutation is trivial.
              bounds.set! id (some xin)
            else
              haveI : NeZero outDim := ⟨h0⟩
              let block := m * restSize
              let perm : Fin outDim → Fin outDim := fun idx =>
                let t := idx.val
                let j := t / block
                let rem := t % block
                let i := rem / restSize
                let k := rem % restSize
                let tIn := i * (n * restSize) + j * restSize + k
                Fin.ofNat outDim tIn
              let loAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.loAff
              let hiAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.hiAff
              bounds.set! id (some { inDim := xin.inDim, outDim := outDim, loAff := loAff, hiAff :=
                hiAff })
          else
            match ibp[id]! with
            | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
            | none => bounds
        | _ =>
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | none => bounds
    | _ => bounds
  | .transpose3dLastTwo =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match nodes[p1]!.outShape with
        | .dim a (.dim b (.dim c .scalar)) =>
          let sIn : Shape := .dim a (.dim b (.dim c .scalar))
          if xin.outDim = sIn.size then
            let outDim := xin.outDim
            if h0 : outDim = 0 then
              bounds.set! id (some xin)
            else
              haveI : NeZero outDim := ⟨h0⟩
              let block := c * b
              let perm : Fin outDim → Fin outDim := fun idx =>
                let t := idx.val
                let i := t / block
                let rem := t % block
                let k := rem / b
                let j := rem % b
                let tIn := i * (b * c) + j * c + k
                Fin.ofNat outDim tIn
              let loAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.loAff
              let hiAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.hiAff
              bounds.set! id (some { inDim := xin.inDim, outDim := outDim, loAff := loAff, hiAff :=
                hiAff })
          else
            match ibp[id]! with
            | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
            | none => bounds
        | _ =>
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | none => bounds
    | _ => bounds
  | .layernorm axis =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if axis = Shape.rank node.outShape - 1 then
          if hout : xin.outDim = preB.dim then
            bounds.set! id (some (propagateLayernormBoundsLastAxis (α:=α) node.outShape preB xin
              hout))
          else
            bounds
        else
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | _, _ => bounds
    | _ => bounds
  | .softmax axis =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if axis = Shape.rank node.outShape - 1 then
          if hout : xin.outDim = preB.dim then
            bounds.set! id (some (propagateSoftmaxBoundsLastAxis (α:=α) node.outShape preB xin
              hout))
          else
            bounds
        else
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | _, _ => bounds
    | _ => bounds
  | .mseLoss =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some yAff, some tAff, some yB, some tB =>
        if hout : yAff.outDim = yB.dim then
          if tAff.outDim = tB.dim then
            if hdim : yB.dim = tB.dim then
              if hout2 : yAff.outDim = tAff.outDim then
                if hin : yAff.inDim = tAff.inDim then
                  let yLo : AffineVec α tAff.inDim tAff.outDim :=
                    castAffineIn (α:=α) (n:=yAff.inDim) (n':=tAff.inDim) (m:=tAff.outDim) hin
                      (castAffineOut (α:=α) (n:=yAff.inDim) (m:=yAff.outDim) (m':=tAff.outDim) hout2
                        yAff.loAff)
                  let yHi : AffineVec α tAff.inDim tAff.outDim :=
                    castAffineIn (α:=α) (n:=yAff.inDim) (n':=tAff.inDim) (m:=tAff.outDim) hin
                      (castAffineOut (α:=α) (n:=yAff.inDim) (m:=yAff.outDim) (m':=tAff.outDim) hout2
                        yAff.hiAff)
                  let tHiVec := castDimScalar (α:=α) (n:=tB.dim) (n':=yB.dim) (h:=hdim.symm) tB.hi
                  let tLoVec := castDimScalar (α:=α) (n:=tB.dim) (n':=yB.dim) (h:=hdim.symm) tB.lo
                  let diffLoVec : Tensor α (.dim yB.dim .scalar) := Tensor.subSpec yB.lo tHiVec
                  let diffHiVec : Tensor α (.dim yB.dim .scalar) := Tensor.subSpec yB.hi tLoVec
                  let n := yB.dim
                  let hOutToN : tAff.outDim = n := Eq.trans (Eq.symm hout2) hout
                  let yLoN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN yLo
                  let yHiN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN yHi
                  let tLoN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN tAff.loAff
                  let tHiN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN tAff.hiAff
                  let diffLoAff' : AffineVec α tAff.inDim n :=
                    affSub (α:=α) (n:=tAff.inDim) (m:=n) yLoN tHiN
                  let diffHiAff' : AffineVec α tAff.inDim n :=
                    affSub (α:=α) (n:=tAff.inDim) (m:=n) yHiN tLoN
                  -- Square relaxation on each component of `diff`.
                  let flo := getDimScalarFn (α := α) diffLoVec
                  let fhi := getDimScalarFn (α := α) diffHiVec
                  let slopes_hi : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u => Tensor.scalar (u + l))
                  let bias_hi : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u => Tensor.scalar (-(u * l)))
                  let slopes_lo : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u =>
                        let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                          Numbers.zero
                        Tensor.scalar (Numbers.two * d))
                  let bias_lo : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u =>
                        let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                          Numbers.zero
                        Tensor.scalar (-(d * d)))
                  let sqLoAff :=
                    affApplyDiagSignedLower (α:=α) (inDim:=tAff.inDim) (outDim:=n) slopes_lo
                      bias_lo diffLoAff' diffHiAff'
                  let sqHiAff :=
                    affApplyDiagSignedUpper (α:=α) (inDim:=tAff.inDim) (outDim:=n) slopes_hi
                      bias_hi diffLoAff' diffHiAff'
                  if n > 0 then
                    let nA : α := (n : Nat)
                    let scale : α := Numbers.one / nA
                    let scaleRow : Tensor α (.dim 1 (.dim n .scalar)) :=
                      Spec.fill (α := α) scale (.dim 1 (.dim n .scalar))
                    let outLo : AffineVec α tAff.inDim 1 :=
                      { A := Spec.matMulSpec scaleRow sqLoAff.A
                        c := Spec.matVecMulSpec scaleRow sqLoAff.c }
                    let outHi : AffineVec α tAff.inDim 1 :=
                      { A := Spec.matMulSpec scaleRow sqHiAff.A
                        c := Spec.matVecMulSpec scaleRow sqHiAff.c }
                    bounds.set! id (some { inDim := tAff.inDim, outDim := 1, loAff := outLo, hiAff
                      := outHi })
                  else
                    let z : Tensor α (.dim 1 .scalar) := Spec.fill (α := α) Numbers.zero (.dim 1
                      .scalar)
                    bounds.set! id (some (boundsConst (α := α) ctx.inputDim 1 z z))
                else bounds
              else bounds
            else bounds
          else bounds
        else bounds
      | _, _, _, _ => bounds
    | _ => bounds
  | .conv2d .. =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match ps.conv2dCfg[id]? with
        | some cfg =>
          let convIn := cfg.inC * cfg.inH * cfg.inW
          if hout : xin.outDim = convIn then
            let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
            let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
            let convAff := affOfConv2d (α:=α) cfg
            let out := propagateLinearBounds (α:=α) (n:=convIn) (m:=cfg.outC * outH * outW)
              convAff.A convAff.c xin hout
            bounds.set! id (some out)
          else bounds
        | none =>
          match ps.linearWB[id]? with
          | some p =>
            if hout : xin.outDim = p.n then
              let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w p.b xin hout
              bounds.set! id (some out)
            else bounds
          | none => bounds
      | none => bounds
    | _ => bounds

/-- Run the basic CROWN affine-bounds pass; requires prior IBP for per-node intervals. -/
def runCROWN (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) : Array (Option (FlatAffineBounds α)) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl (fun acc i => propagateCROWNNode (α:=α) g.nodes ps ibp acc ctx
    i) init


/-!
Backward/dual CROWN (objective-dependent) propagation
----------------------------------------------------

`runCROWN` above is a forward DeepPoly-style pass: it computes nodewise affine bounds that do not
depend on an objective.

Basic CROWN additionally supports *backward* propagation for a linear objective
`cᵀ · output`, selecting per-neuron relaxations based on the sign of the downstream coefficients.

The implementation below is pragmatic and covers the IR ops we support in `runCROWN`.
-/

private inductive BackwardDir where
  | lower
  | upper

private structure BackwardState (α : Type) [Context α] where
  coeffs : Array (Option (FlatVec α)) -- per-node objective coefficients
  cst    : α                         -- accumulated constant term

private def flatvecAdd (a b : FlatVec α) : Option (FlatVec α) :=
  if h : a.n = b.n then
    let bv : Tensor α (.dim a.n .scalar) :=
      castDimScalar (α := α) (n := b.n) (n' := a.n) h.symm b.v
    some { n := a.n, v := Tensor.addSpec a.v bv }
  else
    none

private def flatvecScale (k : α) (v : FlatVec α) : FlatVec α :=
  { n := v.n, v := Tensor.scaleSpec v.v k }

private def addCoeff (st : BackwardState α) (pid : Nat) (v : FlatVec α) : BackwardState α :=
  match st.coeffs[pid]! with
  | none => { st with coeffs := st.coeffs.set! pid (some v) }
  | some w =>
    match flatvecAdd (α:=α) w v with
    | some s => { st with coeffs := st.coeffs.set! pid (some s) }
    | none   => st

private def dotFlat {n : Nat} (a b : Tensor α (.dim n .scalar)) : α :=
  Spec.Tensor.sumSpec (Tensor.mulSpec a b)

private def consumeObjectiveFromBox (dir : BackwardDir) (aY : FlatVec α) (B : FlatBox α) : Option α
  :=
  if h : aY.n = B.dim then
    let aYv : Tensor α (.dim B.dim .scalar) :=
      castDimScalar (α := α) (n := aY.n) (n' := B.dim) h aY.v
    let fa := getDimScalarFn (α := α) aYv
    let flo := getDimScalarFn (α := α) B.lo
    let fhi := getDimScalarFn (α := α) B.hi
    let chosenProd : Tensor α (.dim B.dim .scalar) :=
      Tensor.dim (fun i =>
        match fa i, flo i, fhi i with
        | .scalar ay, .scalar l, .scalar u =>
          let y :=
            if decide (ay > Numbers.zero) then
              match dir with
              | .upper => u
              | .lower => l
            else
              match dir with
              | .upper => l
              | .lower => u
          Tensor.scalar (ay * y))
    some (Spec.Tensor.sumSpec chosenProd)
  else
    none

private def diagOfMat {n : Nat} (A : Tensor α (.dim n (.dim n .scalar))) : Tensor α (.dim n
  .scalar) :=
  match A with
  | .dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | .dim cols =>
        match cols i with
        | .scalar v => Tensor.scalar v)

-- Apply a chosen diagonal relaxation y = s ⊙ x + b for a bound on a scalar objective.
private def backwardApplyDiag {n : Nat}
  (dir : BackwardDir)
  (aY : Tensor α (.dim n .scalar))
  (sLo bLo sHi bHi : Tensor α (.dim n .scalar)) :
  (Tensor α (.dim n .scalar) × α) :=
  let fa := getDimScalarFn (α := α) aY
  let fsLo := getDimScalarFn (α := α) sLo
  let fbLo := getDimScalarFn (α := α) bLo
  let fsHi := getDimScalarFn (α := α) sHi
  let fbHi := getDimScalarFn (α := α) bHi
  let sChosen : Tensor α (.dim n .scalar) :=
    Tensor.dim (fun i =>
      match fa i, fsLo i, fsHi i with
      | .scalar ay, .scalar slo, .scalar shi =>
        let s :=
          if decide (ay > Numbers.zero) then
            match dir with
            | .upper => shi
            | .lower => slo
          else
            match dir with
            | .upper => slo
            | .lower => shi
        Tensor.scalar s)
  let bChosen : Tensor α (.dim n .scalar) :=
    Tensor.dim (fun i =>
      match fa i, fbLo i, fbHi i with
      | .scalar ay, .scalar blo, .scalar bhi =>
        let b :=
          if decide (ay > Numbers.zero) then
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

-- Backward step for a unary op with diagonal relaxations
-- (relu/exp/log/sigmoid/tanh/softmax/layernorm).
private def backwardUnaryDiag
  (dir : BackwardDir) (preB : FlatBox α) (localB : FlatAffineBounds α)
  (aY : FlatVec α) : Option (FlatVec α × α) := by
  if h : aY.n = preB.dim then
    let n := preB.dim
    if hIn : localB.inDim = n then
      if hOut : localB.outDim = n then
        let aYv : Tensor α (.dim n .scalar) :=
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

private def matLeftMul {m n : Nat}
  (aY : Tensor α (.dim m .scalar)) (W : Tensor α (.dim m (.dim n .scalar))) :
  Tensor α (.dim n .scalar) :=
  match aY, W with
  | .dim aF, .dim rows =>
    Tensor.materialize <|
      Tensor.dim (fun j =>
        let s : α :=
          (List.finRange m).foldl (fun acc i =>
            match aF i, rows i with
            | .scalar ai, .dim cols =>
              match cols j with
              | .scalar wij => acc + ai * wij) Numbers.zero
        Tensor.scalar s)
  | _, _ =>
    Spec.fill (α := α) Numbers.zero (.dim n .scalar)

private def backwardLinear {m n : Nat}
  (aY : FlatVec α) (W : Tensor α (.dim m (.dim n .scalar))) (b : Tensor α (.dim m .scalar)) :
  Option (FlatVec α × α) :=
  if h : aY.n = m then
    let aYv : Tensor α (.dim m .scalar) :=
      castDimScalar (α := α) (n := aY.n) (n' := m) h aY.v
    let aX := matLeftMul (α:=α) (m:=m) (n:=n) aYv W
    let cst := dotFlat (α:=α) aYv b
    some ({ n := n, v := aX }, cst)
  else
    none

private def backwardAdd (aY : FlatVec α) : FlatVec α := aY

private def backwardSubLeft (aY : FlatVec α) : FlatVec α := aY

private def backwardSubRight (aY : FlatVec α) : FlatVec α :=
  flatvecScale (α:=α) (k := (-Numbers.one)) aY

private def backwardConcatSplit
  (aY : FlatVec α) (n1 n2 : Nat) : Option (FlatVec α × FlatVec α) :=
  if h : aY.n = n1 + n2 then
    let aYv : Tensor α (.dim (n1 + n2) .scalar) :=
      castDimScalar (α := α) (n := aY.n) (n' := n1 + n2) h aY.v
    let a1 : Tensor α (.dim n1 .scalar) :=
      Tensor.dim (fun i =>
        Tensor.scalar (getAtOrZero aYv [i.val]))
    let a2 : Tensor α (.dim n2 .scalar) :=
      Tensor.dim (fun i =>
        Tensor.scalar (getAtOrZero aYv [n1 + i.val]))
    some ({ n := n1, v := a1 }, { n := n2, v := a2 })
  else
    none

private def backwardPermuteVec {n : Nat} (perm : Fin n → Fin n) (v : Tensor α (.dim n .scalar)) :
  Tensor α (.dim n .scalar) :=
  match v with
  | .dim f => Tensor.dim (fun i => f (perm i))

private def backwardMatmul
  (dir : BackwardDir)
  (aZ : FlatVec α) (Bx By : FlatBox α)
  (sA sB : Shape) :
  Option ((FlatVec α) × (FlatVec α) × α) :=
  let dims? : Option (Nat × Nat × Nat × Nat) :=
    match sA, sB with
    | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
      if k = k' then
        some (1, m, k, n)
      else
        none
    | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
      if hb : b = b' then
        match hb with
        | rfl =>
          if k = k' then
            some (b, m, k, n)
          else
            none
      else
        none
    | _, _ => none
  match dims? with
  | none => none
  | some (batch, m, k, n) =>
    let dimA := batch * m * k
    let dimB := batch * k * n
    let outDim := batch * m * n
    if aZ.n = outDim ∧ Bx.dim = dimA ∧ By.dim = dimB then
      let (aArr, bArr, cst) : Array α × Array α × α := Id.run do
        let mut aArr : Array α := Array.replicate dimA Numbers.zero
        let mut bArr : Array α := Array.replicate dimB Numbers.zero
        let mut cst : α := Numbers.zero
        let block : Nat := m * n
        let strideA : Nat := m * k
        let strideB : Nat := k * n
        for outIdx in List.range outDim do
          let az : α := getAtOrZero aZ.v [outIdx]
          let bi := outIdx / block
          let rem := outIdx % block
          let i := rem / n
          let j := rem % n
          let baseA := bi * strideA
          let baseB := bi * strideB
          for kk in List.range k do
            let aIdx := baseA + i * k + kk
            let bIdx := baseB + kk * n + j
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive

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
              if decide (az > Numbers.zero) then
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

      let aT : Tensor α (.dim dimA .scalar) :=
        Tensor.dim (fun i => Tensor.scalar (aArr[i.val]!))
      let bT : Tensor α (.dim dimB .scalar) :=
        Tensor.dim (fun i => Tensor.scalar (bArr[i.val]!))
      some ({ n := dimA, v := aT }, { n := dimB, v := bT }, cst)
    else
      none

private def backwardMulElem
  (dir : BackwardDir)
  (aZ : FlatVec α) (Bx By : FlatBox α) :
  Option ((FlatVec α) × (FlatVec α) × α) :=
  if h : aZ.n = Bx.dim ∧ Bx.dim = By.dim then
    let n := Bx.dim
    let hZ : aZ.n = n := h.1
    let aZv : Tensor α (.dim n .scalar) :=
      castDimScalar (α := α) (n := aZ.n) (n' := n) hZ aZ.v
    let xLo := getDimScalarFn (α := α) Bx.lo
    let xHi := getDimScalarFn (α := α) Bx.hi
    let yLo := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.lo)
    let yHi := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.hi)
    let aF := getDimScalarFn (α := α) aZv
    -- Choose one McCormick plane per element using the interval midpoint.
    let axU : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let ax := if u1 < u2 then ly else uy
          Tensor.scalar ax)
    let ayU : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let ay := if u1 < u2 then ux else lx
          Tensor.scalar ay)
    let bU : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let b := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
          Tensor.scalar b)
    let axL : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let ax := if l1 > l2 then ly else uy
          Tensor.scalar ax)
    let ayL : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let ay := if l1 > l2 then lx else ux
          Tensor.scalar ay)
    let bL : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
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
    let aX : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match aF i, axUFn i, axLFn i with
        | .scalar az, .scalar axu, .scalar axl =>
          let ax :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => axu
              | .lower => axl
            else
              match dir with
              | .upper => axl
              | .lower => axu
          Tensor.scalar (az * ax))
    let aY : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match aF i, ayUFn i, ayLFn i with
        | .scalar az, .scalar ayu, .scalar ayl =>
          let ay :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => ayu
              | .lower => ayl
            else
              match dir with
              | .upper => ayl
              | .lower => ayu
          Tensor.scalar (az * ay))
    let biasProd : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match aF i, bUFn i, bLFn i with
        | .scalar az, .scalar bu, .scalar bl =>
          let b :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => bu
              | .lower => bl
            else
              match dir with
              | .upper => bl
              | .lower => bu
          Tensor.scalar (az * b))
    let cst := Spec.Tensor.sumSpec biasProd
    some ({ n := n, v := aX }, { n := n, v := aY }, cst)
  else
    none

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
          | some cadd => { st with cst := st.cst + cadd }
          | none => st
        | none => st
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        if h : aY.n = v.n then
          let aYv : Tensor α (.dim v.n .scalar) :=
            castDimScalar (α := α) (n := aY.n) (n' := v.n) h aY.v
          let add := dotFlat (α:=α) aYv v.v
          { st with cst := st.cst + add }
        else st
      | none => st
    | .detach =>
      match node.parents with
      | p1 :: _ => addCoeff (α := α) st p1 aY
      | _ => st
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        let st1 := addCoeff (α:=α) st p1 (backwardAdd (α:=α) aY)
        addCoeff (α:=α) st1 p2 (backwardAdd (α:=α) aY)
      | _ => st
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        let st1 := addCoeff (α:=α) st p1 (backwardSubLeft (α:=α) aY)
        addCoeff (α:=α) st1 p2 (backwardSubRight (α:=α) aY)
      | _ => st
    | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .sin | .cos | .permute _ | .maxElem |
      .minElem
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad ..
    | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => { st with cst := st.cst + cadd }
        | none => st
      | none => st
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match ps.linearWB[id]? with
        | some p =>
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w p.b with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            { st' with cst := st'.cst + cadd }
          | none => st
        | none => st
      | _ => st
    | .matmul =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMatmul (α:=α) (dir:=dir) aY Bx By (sA := nodes[p1]!.outShape) (sB :=
            nodes[p2]!.outShape) with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            { st2 with cst := st2.cst + cadd }
          | none => st
        | _, _ => st
      | p1 :: _ =>
        match ps.matmulW[id]? with
        | some p =>
          let zb := Spec.fill (α := α) Numbers.zero (.dim p.m .scalar)
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w zb with
          | some (aX, _cadd) =>
            addCoeff (α:=α) st p1 aX
          | none => st
        | none => st
      | _ => st
    | .conv2d .. =>
      match node.parents with
      | p1 :: _ =>
        match ps.conv2dCfg[id]? with
        | some cfg =>
          let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
          let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
          let outDim := cfg.outC * outH * outW
          let convAff := affOfConv2d (α:=α) cfg
          match backwardLinear (α:=α) (m:=outDim) (n:=cfg.inC * cfg.inH * cfg.inW) aY convAff.A
            convAff.c with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            { st' with cst := st'.cst + cadd }
          | none => st
        | none => st
      | _ => st
    | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softmax _ | .layernorm _ =>
      -- Unary ops: use local diagonal relaxations computed from the parent IBP box.
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some preB =>
          let n := preB.dim
          let idB := boundsIdentity (α:=α) n
          let localB : FlatAffineBounds α :=
            match node.kind with
            | .relu      => propagateReluBounds (α:=α) preB idB rfl
            | .exp       => propagateExpBounds (α:=α) preB idB rfl
            | .log       => propagateLogBounds (α:=α) preB idB rfl
            | .inv       =>
              let invB := boxInv (α := α) preB
              boundsConst (α:=α) n n invB.lo invB.hi
            | .sigmoid   => propagateSigmoidBounds (α:=α) preB idB rfl
            | .tanh      => propagateTanhBounds (α:=α) preB idB rfl
            | .softmax axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateSoftmaxBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n (Spec.fill (α:=α) Numbers.zero (.dim n .scalar)) (Spec.fill
                  (α:=α) Numbers.one (.dim n .scalar))
            | .layernorm axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateLayernormBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n preB.lo preB.hi
            | _ => idB
          match backwardUnaryDiag (α:=α) dir preB localB aY with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            { st' with cst := st'.cst + cadd }
          | none => st
        | none => st
      | _ => st
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMulElem (α:=α) (dir:=dir) aY Bx By with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            { st2 with cst := st2.cst + cadd }
          | none => st
        | _, _ => st
      | _ => st
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some Bx =>
          if aY.n = 1 then
            let a0 : α := getAtOrZero aY.v [0]
            let out : FlatVec α :=
              { n := Bx.dim, v := Spec.fill (α := α) a0 (.dim Bx.dim .scalar) }
            addCoeff (α:=α) st p1 out
          else st
        | none => st
      | _ => st
    | .reshape _ _ | .flatten _ =>
      match node.parents with
      | p1 :: _ => addCoeff (α:=α) st p1 aY
      | _ => st
    | .concat _ =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some B1, some B2 =>
          match backwardConcatSplit (α:=α) aY B1.dim B2.dim with
          | some (a1, a2) =>
            let st1 := addCoeff (α:=α) st p1 a1
            addCoeff (α:=α) st1 p2 a2
          | none => st
        | _, _ => st
      | _ => st
    | .swap_first_two =>
      match node.parents with
      | p1 :: _ =>
        match nodes[p1]!.outShape with
        | .dim m (.dim n rest) =>
          let outDim := aY.n
          if h0 : outDim = 0 then
            addCoeff (α:=α) st p1 aY
          else
            haveI : NeZero outDim := ⟨h0⟩
            let restSize := Shape.size rest
            let block := m * restSize
            let perm : Fin outDim → Fin outDim := fun idx =>
              let t := idx.val
              let j := t / block
              let rem := t % block
              let i := rem / restSize
              let k := rem % restSize
              let tIn := i * (n * restSize) + j * restSize + k
              Fin.ofNat outDim tIn
            let aYv : Tensor α (.dim outDim .scalar) :=
              castDimScalar (α := α) (n := aY.n) (n' := outDim) rfl aY.v
            let aXv := backwardPermuteVec (α:=α) (n:=outDim) perm aYv
            addCoeff (α:=α) st p1 { n := outDim, v := aXv }
        | _ => st
      | _ => st
    | .transpose3dLastTwo =>
      match node.parents with
      | p1 :: _ =>
        match nodes[p1]!.outShape with
        | .dim _a (.dim b (.dim c .scalar)) =>
          let outDim := aY.n
          if h0 : outDim = 0 then
            addCoeff (α:=α) st p1 aY
          else
            haveI : NeZero outDim := ⟨h0⟩
            let block := c * b
            let perm : Fin outDim → Fin outDim := fun idx =>
              let t := idx.val
              let i := t / block
              let rem := t % block
              let k := rem / b
              let j := rem % b
              let tIn := i * (b * c) + j * c + k
              Fin.ofNat outDim tIn
            let aYv : Tensor α (.dim outDim .scalar) :=
              castDimScalar (α := α) (n := aY.n) (n' := outDim) rfl aY.v
            let aXv := backwardPermuteVec (α:=α) (n:=outDim) perm aYv
            addCoeff (α:=α) st p1 { n := outDim, v := aXv }
        | _ => st
      | _ => st
    | .mseLoss =>
      -- Treat mse_loss as mean(square(y - t)) using the same square relaxation as in `runCROWN`.
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some Y, some T =>
          if hdim : Y.dim = T.dim then
            let n := Y.dim
            if n > 0 then
              if aY.n = 1 then
                let a0 : α := getAtOrZero aY.v [0]
                let nA : α := (n : Nat)
                let scale : α := a0 / nA
                -- Coefficients for each squared term are all `scale`.
                let aSq : Tensor α (.dim n .scalar) := Spec.fill (α := α) scale (.dim n .scalar)
                -- Diff interval box.
                let Thi := castDimScalar (α:=α) (n:=T.dim) (n':=n) hdim.symm T.hi
                let Tlo := castDimScalar (α:=α) (n:=T.dim) (n':=n) hdim.symm T.lo
                let diffLo : Tensor α (.dim n .scalar) := Tensor.subSpec Y.lo Thi
                let diffHi : Tensor α (.dim n .scalar) := Tensor.subSpec Y.hi Tlo
                let flo := getDimScalarFn (α := α) diffLo
                let fhi := getDimScalarFn (α := α) diffHi
                let slopes_hi : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u => Tensor.scalar (u + l))
                let bias_hi : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u => Tensor.scalar (-(u * l)))
                let slopes_lo : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u =>
                      let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                        Numbers.zero
                      Tensor.scalar (Numbers.two * d))
                let bias_lo : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u =>
                      let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                        Numbers.zero
                      Tensor.scalar (-(d * d)))
                -- Choose square plane per element using sign of `aSq` and `dir`.
                let (aDiff, cadd) :=
                  backwardApplyDiag (α := α) (n := n) dir aSq slopes_lo bias_lo
                    slopes_hi bias_hi
                let cst' := st.cst + cadd
                let st1 := addCoeff (α := α) { st with cst := cst' } p1 { n := n, v := aDiff }
                let st2 :=
                  addCoeff (α := α) st1 p2
                    (flatvecScale (α := α) (k := (-Numbers.one)) { n := n, v := aDiff })
                st2
              else st
            else st
          else st
        | _, _ => st
      | _ => st

private def backwardNodeWithReluAlpha (dir : BackwardDir)
  (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
  (ctx : AffineCtx) (reluAlpha : Array (Option (FlatVec α)))
  (st : BackwardState α) (id : Nat) : BackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
    let node := nodes[id]!
    match node.kind with
    | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softmax _ | .layernorm _ =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some preB =>
          let n := preB.dim
          let idB := boundsIdentity (α:=α) n
          let localB : FlatAffineBounds α :=
            match node.kind with
            | .relu =>
              match reluAlpha[id]? with
              | some (some a) =>
                if h : a.n = n then
                  let aT : Tensor α (.dim n .scalar) :=
                    castDimScalar (α:=α) (n:=a.n) (n':=n) h a.v
                  propagateReluBoundsWithAlpha (α:=α) preB idB rfl aT
                else
                  propagateReluBounds (α:=α) preB idB rfl
              | _ =>
                propagateReluBounds (α:=α) preB idB rfl
            | .exp       => propagateExpBounds (α:=α) preB idB rfl
            | .log       => propagateLogBounds (α:=α) preB idB rfl
            | .inv       =>
              let invB := boxInv (α := α) preB
              boundsConst (α:=α) n n invB.lo invB.hi
            | .sigmoid   => propagateSigmoidBounds (α:=α) preB idB rfl
            | .tanh      => propagateTanhBounds (α:=α) preB idB rfl
            | .softmax axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateSoftmaxBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n (Spec.fill (α:=α) Numbers.zero (.dim n .scalar)) (Spec.fill
                  (α:=α) Numbers.one (.dim n .scalar))
            | .layernorm axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateLayernormBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n preB.lo preB.hi
            | _ => idB
          match backwardUnaryDiag (α:=α) dir preB localB aY with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            { st' with cst := st'.cst + cadd }
          | none => st
        | none => st
      | _ => st
    | _ =>
      backwardNode (α:=α) dir nodes ps ibp ctx st id

private def runBackwardObjectiveDir
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := Numbers.zero }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNode (α:=α) dir g.nodes ps ibp ctx acc i) init
    match st.coeffs[ctx.inputId]! with
    | some aIn =>
      if hIn : aIn.n = ctx.inputDim then
        let vIn : Tensor α (.dim ctx.inputDim .scalar) :=
          castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
        let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) := Tensor.dim (fun _ => vIn)
        let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
      else
        none
    | none =>
      -- No dependence on the designated input: return a constant affine bound.
      let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) :=
        Spec.fill (α := α) Numbers.zero (.dim 1 (.dim ctx.inputDim .scalar))
      let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
      some { A := A, c := c }
  else
    none

private def runBackwardObjectiveDirWithReluAlpha
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α)
  (reluAlpha : Array (Option (FlatVec α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := Numbers.zero }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNodeWithReluAlpha (α:=α) dir g.nodes ps ibp ctx reluAlpha acc i) init
    match st.coeffs[ctx.inputId]! with
    | some aIn =>
      if hIn : aIn.n = ctx.inputDim then
        let vIn : Tensor α (.dim ctx.inputDim .scalar) :=
          castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
        let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) := Tensor.dim (fun _ => vIn)
        let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
      else
        none
    | none =>
      let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) :=
        Spec.fill (α := α) Numbers.zero (.dim 1 (.dim ctx.inputDim .scalar))
      let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
      some { A := A, c := c }
  else
    none

/--
Objective-dependent backward CROWN bound for a scalar objective.

Given a linear objective `objᵀ * output`, this runs a backward pass that propagates the objective
coefficients through the graph, selects the relaxation attached to each node, and returns a pair of
affine bounds on the objective with respect to `ctx.inputId`.

The returned `FlatAffineBounds` always has `outDim = 1` (a scalar objective).
-/
def runCROWNBackwardObjective
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α) :
  Option (FlatAffineBounds α) := by
  -- Upper and lower affines for the same scalar objective.
  match runBackwardObjectiveDir (α:=α) .lower g ps ctx ibp outputId obj,
        runBackwardObjectiveDir (α:=α) .upper g ps ctx ibp outputId obj with
  | some loAff, some hiAff =>
    exact some { inDim := ctx.inputDim, outDim := 1, loAff := loAff, hiAff := hiAff }
  | _, _ => exact none

/--
Backward CROWN objective lower bound with externally-provided ReLU alpha slopes.

This is an integration hook for alpha-CROWN style workflows where ReLU slopes are chosen/optimized outside
TorchLean and then imported as a per-node vector in `reluAlpha`.
-/
def runCROWNBackwardObjectiveLowerWithReluAlpha
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α)
  (reluAlpha : Array (Option (FlatVec α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  runBackwardObjectiveDirWithReluAlpha (α:=α) .lower g ps ctx ibp outputId obj reluAlpha


end NN.MLTheory.CROWN.Graph
