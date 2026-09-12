/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core
public import NN.IR.Semantics
public import NN.MLTheory.CROWN.Operators.Conv
public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.IR.Payload -- shake: keep
public import NN.Spec.Core.Shape -- shake: keep
public import NN.Spec.Core.Tensor.SomeTensor -- shake: keep

/-!
Shared definitions for the graph CROWN engine.

This file contains the rank-one tensor representation, parameter stores, interval boxes, shape
permutation helpers, and tensor casts used by the IBP, derivative, affine, CROWN, and backward
objective passes.
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

/--
An existentially sized rank-one tensor used by the flat LiRPA engine.

Graph nodes carry dimensions discovered while lowering, so the dimension cannot always appear in
the surrounding static type. The field `n` is the hidden tensor dimension, not duplicate runtime
storage.
-/
structure FlatTensor (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of scalar entries. -/
  n : Nat
  /-- Rank-one tensor payload (shape `.dim n .scalar`). -/
  v : Tensor α [n]

-- The flat-tensor engine is the canonical executable path for the current graph verifier.

/--
Parameters for a linear layer `y = W*x + b` in flattened form.

`m` is the output dimension and `n` is the input dimension.
-/
structure LinParams (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α [m, n]
  /-- Bias tensor `b` (shape `m`). -/
  b : Tensor α [m]

/-- Matrix parameters for bias-free matmul: y = W x. -/
structure MatParams (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α [m, n]

/-- Coordinate on `axis` corresponding to a row-major flat index. -/
def axisCoordinateOfFlat (shape : Shape) (axis idx : Nat) : Nat :=
  let dims := shape.toList
  let axisStride := (dims.drop (axis + 1)).prod
  if axisStride = 0 then 0 else idx / axisStride

/-- Eval BatchNorm scale for one channel. -/
def batchNormEvalScale (config : NN.IR.BatchNormEvalParams α) (ci : Fin config.c) : α :=
  let gamma := Tensor.item (get config.gamma ci)
  let var := Tensor.item (get config.var ci)
  gamma / MathFunctions.sqrt (max var 0 + config.eps)

/-- Eval-mode BatchNorm bias after folding one channel's running statistics into an affine map. -/
def batchNormEvalBias (config : NN.IR.BatchNormEvalParams α) (ci : Fin config.c) : α :=
  let beta := Tensor.item (get config.beta ci)
  let mean := Tensor.item (get config.mean ci)
  beta - mean * batchNormEvalScale (α := α) config ci

/--
Build the exact diagonal affine form for eval-mode BatchNorm on an arbitrary channel axis.

The graph records the channel axis while the payload stores one scale and bias per channel. A
malformed axis or mismatched channel extent has no verifier transfer rule.
-/
def batchNormEvalLinear? (parentShape : Shape) (channelAxis : Nat)
    (config : NN.IR.BatchNormEvalParams α) : Option (LinParams α) := do
  let channels ← parentShape.toList[channelAxis]?
  if hcfg : config.c = 0 then
    none
  else if channels = config.c then
    haveI : NeZero config.c := ⟨hcfg⟩
    let outDim := parentShape.size
    let weight : Tensor α [outDim, outDim] :=
      Tensor.dim (fun oi =>
        Tensor.dim (fun ii =>
          let ch := axisCoordinateOfFlat parentShape channelAxis oi.val
          let scale := batchNormEvalScale (α := α) config (Fin.ofNat config.c ch)
          Tensor.scalar (if decide (oi.val = ii.val) then scale else 0)))
    let bias : Tensor α [outDim] :=
      Tensor.dim (fun oi =>
        let ch := axisCoordinateOfFlat parentShape channelAxis oi.val
        Tensor.scalar (batchNormEvalBias (α := α) config (Fin.ofNat config.c ch)))
    some { m := outDim, n := outDim, w := weight, b := bias }
  else
    none

/--
Parameters keyed by node id (weights, biases, constants, and seeded input boxes).

This is kept compact: it is the graph interpreter used to run IBP/CROWN on a pure `Graph`
without pulling in a heavyweight runtime.
-/
structure ParamStore (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Seed boxes for designated input nodes (`id -> FlatBox`). -/
  inputBoxes : Std.HashMap Nat (FlatBox α) := Std.HashMap.emptyWithCapacity
  /-- Constants (`id -> FlatTensor`). -/
  constVals  : Std.HashMap Nat (FlatTensor α) := Std.HashMap.emptyWithCapacity
  /-- Linear layer params (`id -> (W,b)`). -/
  linearWB   : Std.HashMap Nat (LinParams α) := Std.HashMap.emptyWithCapacity
  /-- Matmul params (`id -> W`) for bias-free multiplication. -/
  matmulW    : Std.HashMap Nat (MatParams α) := Std.HashMap.emptyWithCapacity
  /-- Convolution specs (`id -> convolution configuration`). -/
  convCfg : Std.HashMap Nat (NN.IR.ConvParams α) := Std.HashMap.emptyWithCapacity
  /-- Eval-mode BatchNorm parameters (`id -> gamma/beta/running stats`). -/
  batchNormEval : Std.HashMap Nat (NN.IR.BatchNormEvalParams α) :=
    Std.HashMap.emptyWithCapacity
  /-- Affine LayerNorm parameters keyed by node id.

  Last-axis value bounds use the stored scale, bias, and epsilon. An absent payload selects
  unit scale, zero bias, and default epsilon. Payload derivative bounds remain
  unresolved until a rule accounts for all three parameters. -/
  layerNorm : Std.HashMap Nat (NN.IR.LayerNormParams α) := Std.HashMap.emptyWithCapacity

namespace ParamStore

/-- Insert an input interval box for a graph node. -/
def seedInputBox {α : Type} [TorchLean.Storage α] [Context α]
    (ps : ParamStore α) (inputId : Nat) (xB : FlatBox α) : ParamStore α :=
  { ps with inputBoxes := ps.inputBoxes.insert inputId xB }

/-- Seed a graph input with a uniform `ℓ∞` box around a shaped tensor. -/
def seedLInfBall {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (ps : ParamStore α) (inputId : Nat) (center : Tensor α s) (eps : α) : ParamStore α :=
  ps.seedInputBox inputId <| FlatBox.lInfBall (α := α) center eps

end ParamStore

/--
Whether the current dense convolution transfer exactly matches the corresponding IR semantics.

The flattened CROWN rule implements channel-first convolution without leading batch axes, channel
groups, dilation, or asymmetric padding. The payload must also agree with the configuration stored
in the graph node; otherwise executable IR evaluation and bound propagation would denote different
operators.
-/
def convTransferSupported
    (configuration : NN.IR.ConvConfig) (parameters : NN.IR.ConvParams α)
    (parentShape outShape : Shape) : Bool :=
  parameters.matchesConfig configuration &&
    configuration.channelAxis == 0 &&
    configuration.groups == 1 &&
    Tensor.to configuration.dilation (List Nat) ==
      List.replicate configuration.spatialRank 1 &&
    Tensor.to configuration.paddingAfter (List Nat) ==
      Tensor.to configuration.padding (List Nat) &&
    parentShape == parameters.input .scalar &&
    outShape == parameters.output .scalar

/--
Check the graph-level semantic restrictions imposed by the current CROWN engine.

Unsupported convolutions, non-leading-axis concatenation, and LayerNorm over more than the last
axis are left without bounds. LayerNorm payloads must match that axis's shape exactly. This
predicate checks the common shape contract; individual transfers also check their arithmetic and
derivative requirements.
-/
def crownNodeSemanticsSupported (nodes : Array Node) (ps : ParamStore α) (id : Nat) : Bool :=
  match nodes[id]? with
  | none => false
  | some node =>
      match node.kind with
      | .conv configuration =>
          match node.parents with
          | #[parentId] =>
              match nodes[parentId]?, ps.convCfg[id]? with
              | some parent, some parameters =>
                  convTransferSupported
                    (α := α) configuration parameters parent.outShape node.outShape
              | _, _ => false
          | _ => false
      | .concat axis =>
          if axis != 0 then
            false
          else
            match node.parents with
            | #[leftId, rightId] =>
                match nodes[leftId]?, nodes[rightId]? with
                | some left, some right =>
                    match OpContracts.inferConcatOutShape axis #[left.outShape, right.outShape] with
                    | .ok expected => expected == node.outShape
                    | .error _ => false
                | _, _ => false
            | _ => false
      | .layernorm axis =>
          match node.parents with
          | #[parentId] =>
              match nodes[parentId]? with
              | some parent =>
                  if axis != node.outShape.rank - 1 || parent.outShape != node.outShape then
                    false
                  else
                    match ps.layerNorm[id]? with
                    | none => true
                    | some parameters =>
                        match OpContracts.layerNormMatrixDims axis node.outShape with
                        | .error _ => false
                        | .ok _ =>
                            parameters.normalizedShape ==
                              Shape.ofList (node.outShape.toList.drop axis)
              | none => false
          | _ => false
      | _ => true

/-- Whether every node in a graph is interpreted exactly by the current CROWN engine. -/
def crownGraphSemanticsSupported (g : Graph) (ps : ParamStore α) : Bool :=
  (List.range g.nodes.size).all (crownNodeSemanticsSupported (α := α) g.nodes ps)

/-- Read a node's interval box from an IBP-style result array. -/
def outputBox? {α : Type} [TorchLean.Storage α] [Context α]
    (boxes : Array (Option (FlatBox α))) (outId : Nat) : Except String (FlatBox α) := do
  match boxes[outId]? with
  | some (some outB) => pure outB
  | some none => throw s!"output box missing at node {outId}"
  | none => throw s!"output node {outId} is out of bounds for {boxes.size} boxes"

/-- Default inhabitant for `FlatBox` (a 0-dimensional box at `0`). -/
instance : Inhabited (FlatBox α) where
  default :=
    { dim := 0
      lo := Tensor.full (α := α) (.dim 0 .scalar) 0
      hi := Tensor.full (α := α) (.dim 0 .scalar) 0 }

/-- Elementwise product of two FlatBoxes (interval product per component). Requires equal dims. -/
@[expose] public def boxMulElem (B1 B2 : FlatBox α) : Option (FlatBox α) :=
  match B1, B2 with
  | ⟨n1, l1, u1⟩, ⟨n2, l2, u2⟩ =>
    if h : n1 = n2 then
      by
        cases h
        let lo :=
          Tensor.ofFn fun i =>
            let lx := l1.getScalar i
            let ux := u1.getScalar i
            let ly := l2.getScalar i
            let uy := u2.getScalar i
            let p1 := BoundOps.mulDown lx ly
            let p2 := BoundOps.mulDown lx uy
            let p3 := BoundOps.mulDown ux ly
            let p4 := BoundOps.mulDown ux uy
            min2 (min2 p1 p2) (min2 p3 p4)
        let hi :=
          Tensor.ofFn fun i =>
            let lx := l1.getScalar i
            let ux := u1.getScalar i
            let ly := l2.getScalar i
            let uy := u2.getScalar i
            let p1 := BoundOps.mulUp lx ly
            let p2 := BoundOps.mulUp lx uy
            let p3 := BoundOps.mulUp ux ly
            let p4 := BoundOps.mulUp ux uy
            max2 (max2 p1 p2) (max2 p3 p4)
        exact some { dim := n1, lo := lo, hi := hi }
    else none

/-- Chain-rule multiplication for derivative intervals. Returns `none` on dimension mismatch. -/
def chainMul (dZ dF : FlatBox α) : Option (FlatBox α) :=
  boxMulElem (α:=α) dZ dF

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
              lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
              hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Interval subtraction on `FlatBox` endpoints (sound enclosure). -/
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
              lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
              hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Directed lower and upper sums of all coordinates in a flat box. -/
private def boxSumEndpoints (B : FlatBox α) : α × α :=
  let lo := (List.finRange B.dim).foldl (fun acc i =>
    BoundOps.addDown acc (B.lo.getScalar i)) 0
  let hi := (List.finRange B.dim).foldl (fun acc i =>
    BoundOps.addUp acc (B.hi.getScalar i)) 0
  (lo, hi)

/-- Sum all coordinates of a flat box with directed accumulation. -/
def boxSum (B : FlatBox α) : FlatBox α :=
  let (lo, hi) := boxSumEndpoints (α := α) B
  { dim := 1
    lo := Tensor.full (α := α) (.dim 1 .scalar) lo
    hi := Tensor.full (α := α) (.dim 1 .scalar) hi }

/-- Average all coordinates of a nonempty flat box with directed division. -/
def boxMean? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) := do
  if B.dim = 0 then
    none
  else
    let (lo, hi) := boxSumEndpoints (α := α) B
    let n : α := B.dim
    let (meanLo, meanHi) ← NonlinearBoundOps.divBounds lo hi n n
    pure
      { dim := 1
        lo := Tensor.full (α := α) (.dim 1 .scalar) meanLo
        hi := Tensor.full (α := α) (.dim 1 .scalar) meanHi }

/-- Apply ReLU to both endpoints of a `FlatBox` (monotone activation, so endpoints suffice). -/
@[expose]
public def boxRelu (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.lo
    hi := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.hi }

/-- Componentwise absolute value bounds. Soundly encloses `abs` over each interval component. -/
def boxAbs (B : FlatBox α) : FlatBox α :=
  let lo' := Tensor.ofFn fun i =>
    let l := B.lo.getScalar i
    let u := B.hi.getScalar i
    let al := MathFunctions.abs l
    let au := MathFunctions.abs u
    if l < 0 then
      if 0 < u then 0 else (if al < au then al else au)
    else
      if al < au then al else au
  let hi' := Tensor.ofFn fun i =>
    let al := MathFunctions.abs (B.lo.getScalar i)
    let au := MathFunctions.abs (B.hi.getScalar i)
    if al > au then al else au
  { dim := B.dim, lo := lo', hi := hi' }

/-!
`traverseFin` is plumbing, so it lives in `Internal` like the rest of the codebase's plumbing. This
namespace used to be called `boxUnaryEnclosure`, after the function below that is its only caller,
which was misleading twice over: that caller is actually spelled `boxUnaryEnclosure?`, and
`traverseFin` knows nothing about boxes or enclosures.
-/
namespace Internal

/-- Traverse a finite family without converting its index to an untyped list. -/
def traverseFin {β : Type} {n : Nat} (f : Fin n → Option β) : Option (Fin n → β) :=
  if h : ∀ i, (f i).isSome then
    some fun i => (f i).get (h i)
  else
    none

/-- `traverseFin` succeeds exactly when every component does, and then returns those components.

This is the only fact the box operations need about it: it lets a coordinatewise enclosure argument
be read off from the aggregate `Option` without ever mentioning the `dif` in the definition. -/
theorem traverseFin_eq_some_iff {β : Type} {n : Nat}
    {f : Fin n → Option β} {g : Fin n → β} :
    traverseFin f = some g ↔ ∀ i, f i = some (g i) := by
  unfold traverseFin
  split_ifs with h
  · constructor
    · intro hfg i
      have : (fun i => (f i).get (h i)) = g := Option.some.inj hfg
      rw [← this]
      exact (Option.some_get (h i)).symm
    · intro hfg
      congr
      funext i
      obtain ⟨_, hi⟩ := Option.eq_some_iff_get_eq.mp (hfg i)
      exact hi
  · constructor
    · simp
    · intro hfg
      exfalso
      apply h
      intro i
      simp [hfg i]

end Internal

/-- Apply a scalar interval enclosure coordinatewise to a flat box. -/
@[expose]
def boxUnaryEnclosure? [NonlinearBoundOps α]
    (enclose : α → α → Option (α × α)) (B : FlatBox α) : Option (FlatBox α) := do
  let bounds ← Internal.traverseFin fun i =>
    enclose (B.lo.getScalar i) (B.hi.getScalar i)
  let lower : Tensor α [B.dim] := Tensor.ofFn fun i => (bounds i).1
  let upper : Tensor α [B.dim] := Tensor.ofFn fun i => (bounds i).2
  pure { dim := B.dim, lo := lower, hi := upper }

/-- Componentwise square-root bounds, failing when a coordinate interval reaches below zero. -/
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

/-- Componentwise reciprocal bounds, failing when an input coordinate interval crosses zero. -/
@[expose]
def boxInv? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi => NonlinearBoundOps.divBounds 1 1 lo hi) B

/-- Derivative range for `exp`; `exp' = exp`. -/
def derivBoxExp? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α) NonlinearBoundOps.expBounds zB

/-- Derivative range for `log`; `log' x = 1/x` on a strictly positive interval. -/
def derivBoxLog? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi =>
      if lo > 0 then
        NonlinearBoundOps.divBounds 1 1 lo hi
      else
        none) zB

/-- Second-derivative range for `log`; `log'' x = -1/x²` on a positive interval. -/
def secondDerivBoxLog? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi =>
      if lo > 0 then do
        let squareLo := BoundOps.mulDown lo lo
        let squareHi := BoundOps.mulUp hi hi
        let reciprocal ←
          NonlinearBoundOps.divBounds 1 1 squareLo squareHi
        pure (-reciprocal.2, -reciprocal.1)
      else
        none) zB

/-- Negate an interval box by swapping and negating its endpoints. -/
def boxNeg (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.mapSpec (fun x => -x) B.hi
    hi := Tensor.mapSpec (fun x => -x) B.lo }

/-- Apply a full axis permutation to a shape-tagged tensor when the permutation is valid. -/
def permuteSomeTensor? {α : Type} [TorchLean.Storage α] [Context α]
    (v : Spec.SomeTensor α) (perm : Array Nat) : Option (Spec.SomeTensor α) :=
  (NN.IR.Graph.permuteSomeTensor v perm).toOption

/-- Convert a row-major flat index into coordinates for the given dimensions. -/
private def flatCoordinates (dims : Array Nat) (index : Nat) : Array Nat := Id.run do
  let mut coordinates := Array.replicate dims.size 0
  let mut remainder := index
  for axis in [0:dims.size] do
    let tailSize := (dims.extract (axis + 1) dims.size).foldl (· * ·) 1
    coordinates := coordinates.set! axis (remainder / tailSize)
    remainder := remainder % tailSize
  return coordinates

/-- Convert row-major coordinates back into a flat index. -/
private def coordinatesFlatIndex (dims coordinates : Array Nat) : Nat := Id.run do
  let mut index := 0
  for axis in [0:min dims.size coordinates.size] do
    index := index * dims[axis]! + coordinates[axis]!
  return index

/--
Return the flat-coordinate permutation induced by an axis permutation.

`perm` follows the tensor convention used by `Shape.permute?`: output axis `i` is read from input
axis `perm[i]`. The resulting function therefore maps each output flat coordinate to the input
flat coordinate from which its value is taken. Invalid permutations and inconsistent dimensions
are rejected.
-/
def flatAxisPermutation? (sourceShape : Shape) (perm : Array Nat) (n : Nat) :
    Option (Fin n → Fin n) := do
  let targetShape ← Spec.Shape.permute? sourceShape perm.toList
  if sourceShape.size != n || targetShape.size != n then
    none
  else if hn : n = 0 then
    none
  else
    let _ : NeZero n := ⟨hn⟩
    let inverse ← (NN.IR.OpContracts.inversePerm perm).toOption
    let sourceDims := sourceShape.toArray
    let targetDims := targetShape.toArray
    pure fun outputIndex =>
      let targetCoordinates := flatCoordinates targetDims outputIndex.val
      let sourceCoordinates := inverse.map (fun targetAxis => targetCoordinates.getD targetAxis 0)
      Fin.ofNat n (coordinatesFlatIndex sourceDims sourceCoordinates)

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

The body is exposed because the proof layer theorem module unfolds this executable rule when
proving dimension preservation and pointwise enclosure.
-/
@[expose] def boxSquare (B : FlatBox α) : FlatBox α :=
  let lo' :=
    Tensor.ofFn fun i =>
      let l := B.lo.getScalar i
      let u := B.hi.getScalar i
      let l2 := l * l
      let u2 := u * u
      if l < 0 then
        if 0 < u then 0 else (if l2 < u2 then l2 else u2)
      else
        if l2 < u2 then l2 else u2
  let hi' :=
    Tensor.ofFn fun i =>
      let l2 := B.lo.getScalar i * B.lo.getScalar i
      let u2 := B.hi.getScalar i * B.hi.getScalar i
      if l2 > u2 then l2 else u2
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

/-- Reinterpret a flattened tensor as shape `s` when the element counts agree. -/
def ibpUnflatten {s : Shape} (dim : Nat) (t : Tensor α [dim]) (h : dim =
  Spec.Shape.size s) :
    Tensor α s :=
  let t' : Tensor α [Spec.Shape.size s] := by
    simpa [h] using t
  Tensor.unflattenSpec (α := α) s t'

/-- IBP rule for broadcasting a flattened input box to a target shape. -/
def ibpBroadcastTo (s₁ s₂ : Shape) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s₁ then
    if cb : Shape.CanBroadcastTo s₁ s₂ then
      let xLo : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.lo h
      let xHi : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.hi h
      let yLo : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xLo
      let yHi : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xHi
      let flatLo := Tensor.flattenSpec (α := α) yLo
      let flatHi := Tensor.flattenSpec (α := α) yHi
      some { dim := Spec.Shape.size s₂, lo := flatLo, hi := flatHi }
    else
      none
  else
    none

/-- IBP rule for reducing a shaped box by summing along one axis. -/
def ibpReduceSumAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s then
    match Spec.Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceSum (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceSum (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/-- IBP rule for reducing a shaped box by averaging along one axis. -/
def ibpReduceMeanAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s then
    match Spec.Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceMean (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceMean (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/--
Format-independent softmax enclosure on a flattened tensor.

A singleton row is exactly one. Every coordinate of a longer row lies in `[0,1]`. This deliberately
forgoes the tighter exponential formula above so executable checking does not assume a directed
transcendental implementation that its scalar backend has not supplied.
-/
def ibpSoftmaxRange (s : Shape) (dim : Nat) : FlatBox α :=
  let rowLength := lastDimLen s
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
so they remain valid for executable endpoint types whose `BoundOps` instance covers only directed
arithmetic. A tighter transfer rule requires separately certified directed bounds for
transcendental operations.
-/

/-- Conservative interval bounds for hard-masked softmax along the last tensor axis. -/
def ibpHardMaskedSoftmaxLastTensor : {s : Shape} →
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
        (ibpHardMaskedSoftmaxLastTensor (s := inner)
          (lo.unstack i) (hi.unstack i) (allowed.unstack i)).1
      let upper := Tensor.dim fun i : Fin n =>
        (ibpHardMaskedSoftmaxLastTensor (s := inner)
          (lo.unstack i) (hi.unstack i) (allowed.unstack i)).2
      (lower, upper)

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
Ideal-arithmetic upper bound on the variance term used by analytic LayerNorm rules.

Given endpoint bounds for a vector and bounds on its mean, each coordinate is at most
`max |x_i - μ|` away from the bounded mean interval. Squaring and summing those coordinate radii
gives a conservative variance upper bound. The implementation uses ordinary scalar arithmetic and
is called only from exact-arithmetic branches; executable endpoint propagation uses
`ibpLayerNormRange?` instead.
-/
def idealLayerNormVarianceUpper {n : Nat}
    (lo hi : Tensor α [n]) (muLo muHi : α) : α :=
  if _h : n > 0 then
    let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
      let dl := MathFunctions.abs (lo.getScalar i - muHi)
      let du := MathFunctions.abs (hi.getScalar i - muLo)
      let a := if dl > du then dl else du
      acc + (a * a)) 0
    sumAbsSq / (n : Nat)
  else
    0

/-- Ideal-arithmetic mean bounds for a nonempty vector with bounded coordinates.

For `n = 0`, the mathematical mean is undefined; this total helper returns `(0,0)` so callers do
not accidentally divide by zero while they reject or totalize the empty case.
-/
def idealLayerNormMeanBounds {n : Nat}
    (lo hi : Tensor α [n]) : α × α :=
  if _h : n > 0 then
    let nA : α := (n : Nat)
    (TorchLean.Tensor.sumSpec lo / nA, TorchLean.Tensor.sumSpec hi / nA)
  else
    (0, 0)

/--
Ideal-arithmetic bounds for `x - μ` when `x` and `μ` are bounded by intervals.

LayerNorm transfer rules repeatedly need this centered interval for the input, first derivative,
and second derivative streams. Keeping it here avoids duplicating the same endpoint arithmetic in
IBP and derivative propagation.
-/
def idealLayerNormCenteredBounds {n : Nat}
    (lo hi : Tensor α [n]) (muLo muHi : α) :
    Tensor α [n] × Tensor α [n] :=
  let loOut :=
    Tensor.ofFn fun i =>
      let dl := lo.getScalar i - muHi
      let du := hi.getScalar i - muLo
      if dl < du then dl else du
  let hiOut :=
    Tensor.ofFn fun i =>
      let dl := lo.getScalar i - muHi
      let du := hi.getScalar i - muLo
      if dl > du then dl else du
  (loOut, hiOut)

/-- Ideal-arithmetic reciprocal-denominator bounds from an upper variance bound. -/
def idealLayerNormInvStdBounds (varHi : α) : α × α :=
  let sLo := MathFunctions.sqrt Context.defaultEpsilon
  let sHi := MathFunctions.sqrt (varHi + Context.defaultEpsilon)
  (1 / (if sHi > Context.defaultEpsilon then sHi else Context.defaultEpsilon),
   1 / (if sLo > Context.defaultEpsilon then sLo else Context.defaultEpsilon))

/-- Analytic real-arithmetic LayerNorm bounds on the last axis, lifted over leading dimensions. -/
def idealLayerNormLastTensor : {s : Shape} → Tensor α s → Tensor α s → (Tensor α s × Tensor α s)
  | .scalar, lo, hi => (lo, hi)
  | .dim n .scalar, lo, hi =>
      if n > 0 then
        let nA : α := (n : Nat)
        let sum_lo := TorchLean.Tensor.sumSpec lo
        let sum_hi := TorchLean.Tensor.sumSpec hi
        let mu_lo := sum_lo / nA
        let mu_hi := sum_hi / nA
        let var_hi := idealLayerNormVarianceUpper (α := α) lo hi mu_lo mu_hi
        let den_lo := MathFunctions.sqrt Context.defaultEpsilon
        let den_hi := MathFunctions.sqrt (var_hi + Context.defaultEpsilon)
        let outLo :=
          Tensor.ofFn fun i =>
            let dl := lo.getScalar i - mu_hi
            let du := hi.getScalar i - mu_lo
            let c1 := dl / den_lo
            let c2 := dl / den_hi
            let c3 := du / den_lo
            let c4 := du / den_hi
            let mn12 := if c1 < c2 then c1 else c2
            let mn34 := if c3 < c4 then c3 else c4
            if mn12 < mn34 then mn12 else mn34
        let outHi :=
          Tensor.ofFn fun i =>
            let dl := lo.getScalar i - mu_hi
            let du := hi.getScalar i - mu_lo
            let c1 := dl / den_lo
            let c2 := dl / den_hi
            let c3 := du / den_lo
            let c4 := du / den_hi
            let mx12 := if c1 > c2 then c1 else c2
            let mx34 := if c3 > c4 then c3 else c4
            if mx12 > mx34 then mx12 else mx34
        (outLo, outHi)
      else
        -- Degenerate n=0: pass through
        (lo, hi)
  | .dim n inner, lo, hi =>
      let outLo := Tensor.dim (fun i : Fin n =>
        (idealLayerNormLastTensor (s := inner) (lo.unstack i) (hi.unstack i)).1)
      let outHi := Tensor.dim (fun i : Fin n =>
        (idealLayerNormLastTensor (s := inner) (lo.unstack i) (hi.unstack i)).2)
      (outLo, outHi)

/--
Uniform finite enclosure for last-axis layer normalization.

For a row of length `n`, exact LayerNorm without affine scale or bias satisfies
`|yᵢ| ≤ sqrt n`; a singleton row is identically zero. Backends provide the outward-rounded bound
through `NonlinearBoundOps.layerNormAbsBound`. Returning `none` is preferable to evaluating the
normalization with unqualified host division and square root.
-/
def ibpLayerNormRange? [NonlinearBoundOps α]
    (s : Shape) (dim : Nat) : Option (FlatBox α) :=
  let rowLength := lastDimLen s
  if rowLength = 0 then
    some
      { dim := dim
        lo := Tensor.full (α := α) (.dim dim .scalar) 0
        hi := Tensor.full (α := α) (.dim dim .scalar) 0 }
  else if rowLength = 1 then
    some
      { dim := dim
        lo := Tensor.full (α := α) (.dim dim .scalar) 0
        hi := Tensor.full (α := α) (.dim dim .scalar) 0 }
  else do
    let radius ← NonlinearBoundOps.layerNormAbsBound (α := α) rowLength
    pure
      { dim := dim
        lo := Tensor.full (α := α) (.dim dim .scalar) (-radius)
        hi := Tensor.full (α := α) (.dim dim .scalar) radius }

/--
Validate an interval before using it in a nonlinear transfer. Self-subtraction rejects infinite
and NaN IEEE endpoints; finite exact-real endpoints satisfy these checks as well.
-/
private def checkedLayerNormBounds (bounds : α × α) : Option (α × α) :=
  if !(decide (bounds.2 < bounds.1)) && bounds.1 - bounds.1 == 0 &&
      bounds.2 - bounds.2 == 0 then
    some bounds
  else
    none

/-- Directed mean bounds for one nonempty normalization row. -/
private def layerNormMeanBounds? [NonlinearBoundOps α] {n : Nat}
    (lo hi : Tensor α [n]) : Option (α × α) := do
  if n = 0 then none else do
    let (sumLo, sumHi) := boxSumEndpoints (α := α) { dim := n, lo, hi }
    let count : α := n
    let bounds ← NonlinearBoundOps.divBounds sumLo sumHi count count
    checkedLayerNormBounds bounds

/--
Directed bounds for one LayerNorm row with explicit affine parameters.

The order follows `Spec.layerNorm`: center the input, center again inside `reduceVar`, square,
average, clamp variance to zero, add epsilon, take a square root, and divide the first centered
values directly. Each coordinate is then multiplied by its gamma and shifted by its beta;
negative gamma reverses the interval endpoints.

Epsilon must be finite and positive, and the row and affine parameters must be finite. Division
and square root use the selected nonlinear backend. Invalid or non-finite intermediate intervals
return `none`. An end-to-end soundness theorem for this sequence and a proof of agreement with
native arithmetic remain open obligations.
-/
def directedLayerNormRow? [NonlinearBoundOps α] {n : Nat}
    (lo hi gamma beta : Tensor α [n]) (epsilon : α) :
    Option (Tensor α [n] × Tensor α [n]) := do
  let _ ← checkedLayerNormBounds (epsilon, epsilon)
  if !(epsilon > 0) then none else do
    let _ ← Internal.traverseFin fun i : Fin n => do
      let _ ← checkedLayerNormBounds (lo.getScalar i, hi.getScalar i)
      let _ ← checkedLayerNormBounds (gamma.getScalar i, gamma.getScalar i)
      checkedLayerNormBounds (beta.getScalar i, beta.getScalar i)
    let (meanLo, meanHi) ← layerNormMeanBounds? lo hi
    let centeredLo := Tensor.ofFn fun i => BoundOps.subDown (lo.getScalar i) meanHi
    let centeredHi := Tensor.ofFn fun i => BoundOps.subUp (hi.getScalar i) meanLo
    let _ ← Internal.traverseFin fun i : Fin n =>
      checkedLayerNormBounds (centeredLo.getScalar i, centeredHi.getScalar i)
    let (centerMeanLo, centerMeanHi) ← layerNormMeanBounds? centeredLo centeredHi
    let recenteredLo :=
      Tensor.ofFn fun i => BoundOps.subDown (centeredLo.getScalar i) centerMeanHi
    let recenteredHi :=
      Tensor.ofFn fun i => BoundOps.subUp (centeredHi.getScalar i) centerMeanLo
    let _ ← Internal.traverseFin fun i : Fin n =>
      checkedLayerNormBounds (recenteredLo.getScalar i, recenteredHi.getScalar i)
    let squaredLo := Tensor.ofFn fun i =>
      let l := recenteredLo.getScalar i
      let u := recenteredHi.getScalar i
      if !(decide (0 < l)) && !(decide (u < 0)) then 0
      else min2 (BoundOps.mulDown l l) (BoundOps.mulDown u u)
    let squaredHi := Tensor.ofFn fun i =>
      let l := recenteredLo.getScalar i
      let u := recenteredHi.getScalar i
      max2 (BoundOps.mulUp l l) (BoundOps.mulUp u u)
    let (varianceLo, varianceHi) ← layerNormMeanBounds? squaredLo squaredHi
    let (stabilizedLo, stabilizedHi) ← checkedLayerNormBounds
      (BoundOps.addDown (max2 varianceLo 0) epsilon,
       BoundOps.addUp (max2 varianceHi 0) epsilon)
    let (denominatorLo, denominatorHi) ←
      NonlinearBoundOps.sqrtBounds (max2 stabilizedLo 0) (max2 stabilizedHi 0) >>=
        checkedLayerNormBounds
    if !(denominatorLo > 0) then none else do
      let bounds ← Internal.traverseFin fun i : Fin n => do
        let (lower, upper) ←
          NonlinearBoundOps.divBounds (centeredLo.getScalar i) (centeredHi.getScalar i)
            denominatorLo denominatorHi >>= checkedLayerNormBounds
        let scale := gamma.getScalar i
        let (scaledLo, scaledHi) ← checkedLayerNormBounds
          (min2 (BoundOps.mulDown lower scale) (BoundOps.mulDown upper scale),
           max2 (BoundOps.mulUp lower scale) (BoundOps.mulUp upper scale))
        checkedLayerNormBounds
          (BoundOps.addDown scaledLo (beta.getScalar i),
           BoundOps.addUp scaledHi (beta.getScalar i))
      pure (Tensor.ofFn fun i => (bounds i).1, Tensor.ofFn fun i => (bounds i).2)

/--
Input-dependent last-axis LayerNorm enclosure with unit scale, zero bias, and the default epsilon.

Each row uses `directedLayerNormRow?`, including the second centering and direct division in the
specification. Leading dimensions select independent rows.
-/
def directedLayerNormLastTensor? [NonlinearBoundOps α] :
    {s : Shape} → Tensor α s → Tensor α s → Option (Tensor α s × Tensor α s)
  | .scalar, _, _ => none
  | .dim n .scalar, lo, hi =>
      directedLayerNormRow? lo hi (Tensor.full [n] 1) (Tensor.full [n] 0)
        TorchLean.normalizationEpsilon
  | .dim n (.dim m rest), lo, hi => do
      let rows ← Internal.traverseFin fun i : Fin n =>
        directedLayerNormLastTensor? (s := .dim m rest) (lo.unstack i) (hi.unstack i)
      pure (Tensor.dim fun i => (rows i).1, Tensor.dim fun i => (rows i).2)

/--
Use the backend's uniform LayerNorm range when available; otherwise propagate the supplied input
box through the directed normalization sequence.
-/
def ibpLayerNormBox? [NonlinearBoundOps α]
    (s : Shape) (input : FlatBox α) : Option (FlatBox α) := do
  if h : input.dim = s.size then
    let _ ← Internal.traverseFin fun i : Fin input.dim =>
      checkedLayerNormBounds (input.lo.getScalar i, input.hi.getScalar i)
    match ibpLayerNormRange? (α := α) s input.dim with
    | some result => pure result
    | none =>
        let (lo, hi) ← directedLayerNormLastTensor?
          (ibpUnflatten (s := s) input.dim input.lo h)
          (ibpUnflatten (s := s) input.dim input.hi h)
        pure { dim := s.size, lo := Tensor.flattenSpec lo, hi := Tensor.flattenSpec hi }
  else
    none

/--
Enclose a last-axis LayerNorm payload using its full epsilon, gamma, and beta.

The matrix view and affine suffix are checked by the same helpers used in IR evaluation. Bounds
are propagated separately for each row, then flattened back into the graph's storage order.
Every payload, including one with default parameter values, uses the directed normalization
sequence with its stored affine parameters and epsilon.
-/
def ibpLayerNormPayloadBox? [NonlinearBoundOps α]
    (s : Shape) (axis : Nat) (parameters : NN.IR.LayerNormParams α)
    (input : FlatBox α) : Option (FlatBox α) := do
  if axis != s.rank - 1 then none else do
    let (rows, width) ← (OpContracts.layerNormMatrixDims axis s).toOption
    let payload : NN.IR.Payload α := { layerNorm? := fun _ => some parameters }
    let affine ←
      (NN.IR.Graph.resolveLayerNormAffine payload 0 axis s width).toOption
    let matrixShape : Shape := .dim rows (.dim width .scalar)
    if hInput : input.dim = s.size then
      if hMatrix : s.size = matrixShape.size then
        let lo := ibpUnflatten (s := matrixShape) input.dim input.lo (hInput.trans hMatrix)
        let hi := ibpUnflatten (s := matrixShape) input.dim input.hi (hInput.trans hMatrix)
        let bounds ← Internal.traverseFin fun i : Fin rows =>
          directedLayerNormRow? (lo.unstack i) (hi.unstack i)
            affine.gamma affine.beta affine.epsilon
        let outputLo : Tensor α matrixShape := Tensor.dim fun i => (bounds i).1
        let outputHi : Tensor α matrixShape := Tensor.dim fun i => (bounds i).2
        pure
          { dim := matrixShape.size
            lo := Tensor.flattenSpec outputLo
            hi := Tensor.flattenSpec outputHi }
      else none
    else none

/-- For tensors known to have shape `.dim n .scalar`, extract the underlying function. -/
@[expose] public def getDimScalarFn {n : Nat} (t : Tensor α [n]) : Fin n → Tensor α .scalar :=
  Tensor.unstack t

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
  (r : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n]) :
  Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n'] := by
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
    (h : n = n') (t : Tensor α [n]) : Tensor α [n'] :=
  Tensor.castShape t (congrArg (fun k => Shape.dim k Shape.scalar) h)

omit [Context α] [BoundOps α] in
/-- Casting a flat vector tensor along an equality from a dimension to itself changes no data. -/
@[simp] theorem castDimScalar_self {n : Nat}
    (h : n = n) (t : Tensor α [n]) :
    castDimScalar (α := α) h t = t := by
  exact Tensor.cast_shape_self t _

/-- IBP propagation through explicit linear parameters. -/
@[expose]
public def ibpLinearParams (p : LinParams α) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = p.n then
    let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
    let bBox : Box α (.dim p.m .scalar) := Box.point (α:=α) p.b
    let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB bBox
    some (toFlatBox p.m yB)
  else none

/-- IBP propagation for a `.linear` node using `ParamStore.linearWB`. -/
@[expose]
public def ibpLinear (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.linearWB[id]? with
  | none => none
  | some p => ibpLinearParams (α := α) p Xin

/-- IBP propagation for a `.matmul` node (bias-free) using `ParamStore.matmulW`. -/
@[expose]
public def ibpMatmul (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.matmulW[id]? with
  | none => none
  | some p =>
    if h : Xin.dim = p.n then
      let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
      let zeroB : Box α (.dim p.m .scalar) :=
        let z := Tensor.full (α:=α) (.dim p.m .scalar) 0
        Box.point (α:=α) z
      let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
      some (toFlatBox p.m yB)
    else none

/--
IBP transfer for a supported convolution node whose parameters are stored in `ParamStore.convCfg`.
-/
def ibpConvNode (configuration : NN.IR.ConvConfig) (parentShape outShape : Shape)
    (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.convCfg[id]? with
  | none => none
  | some parameters =>
    if !convTransferSupported (α := α) configuration parameters parentShape outShape then
      none
    else
      let expected := parameters.inChannels *
        (Tensor.to parameters.inputSpatial (List Nat)).prod
      if hdim : Xin.dim = expected then
        let sFlat := Shape.dim Xin.dim Shape.scalar
        let sIn := Shape.ofList
          (parameters.inChannels :: Tensor.to parameters.inputSpatial (List Nat))
        have hsize : sFlat.size = sIn.size := by
          simp [Spec.Shape.size, sFlat, sIn, hdim, expected, Spec.Shape.size_eq_prod]
        let xLo := Tensor.reshapeSpec (α:=α) (source:=sFlat) (target:=sIn) Xin.lo hsize
        let xHi := Tensor.reshapeSpec (α:=α) (source:=sFlat) (target:=sIn) Xin.hi hsize
        let xBox : Box α sIn := { lo := xLo, hi := xHi }
        let yBox := NN.MLTheory.CROWN.ibpConv
          (α := α) (layer := parameters.spec) (xB := xBox)
        let outSpatial := Spec.convOutSpatial parameters.inputSpatial parameters.kernel
          parameters.stride parameters.padding
        let flatOutShape := Shape.ofList
          (parameters.outChannels :: Tensor.to outSpatial (List Nat))
        let flatLo := Tensor.flattenSpec (α:=α) yBox.lo
        let flatHi := Tensor.flattenSpec (α:=α) yBox.hi
        some { dim := flatOutShape.size, lo := flatLo, hi := flatHi }
      else none

/-- Apply a monotone `SomeTensor` operation independently to both interval endpoints. -/
def ibpMonotoneSomeTensor?
    (parentShape outShape : Shape)
    (op : SomeTensor α → Except String (SomeTensor α))
    (input : FlatBox α) : Option (FlatBox α) :=
  if hdim : input.dim = parentShape.size then
    let flatShape := Shape.dim input.dim Shape.scalar
    have hsize : flatShape.size = parentShape.size := by
      simp [flatShape, Shape.size, hdim]
    let loInput :=
      Tensor.reshapeSpec (α := α) (source := flatShape) (target := parentShape) input.lo hsize
    let hiInput :=
      Tensor.reshapeSpec (α := α) (source := flatShape) (target := parentShape) input.hi hsize
    match op (SomeTensor.mk (α := α) parentShape loInput),
        op (SomeTensor.mk (α := α) parentShape hiInput) with
    | .ok lo, .ok hi =>
        if hlo : lo.shape = outShape then
          if hhi : hi.shape = outShape then
            let loTensor : Tensor α outShape := hlo ▸ lo.tensor
            let hiTensor : Tensor α outShape := hhi ▸ hi.tensor
            some
              { dim := outShape.size
                lo := Tensor.flattenSpec (α := α) loTensor
                hi := Tensor.flattenSpec (α := α) hiTensor }
          else
            none
        else
          none
    | _, _ => none
  else
    none


end NN.MLTheory.CROWN.Graph
