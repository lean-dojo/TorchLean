/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core
public import NN.IR.Semantics
public import NN.MLTheory.CROWN.Runtime.Ops

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
    ((Tensor.Internal.Coord.equivFin (layout.parentShape parent)).trans
      (finCongr (Shape.internalSize_eq (layout.parentShape parent)))).symm).trans <|
    (Tensor.Internal.Rep.concatenateAxesCoordinateEquiv
      layout.leading layout.trailing layout.lengths).trans <|
      (Tensor.Internal.Coord.equivFin layout.outputShape).trans
        (finCongr (Shape.internalSize_eq layout.outputShape))

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

/-- Validate concat geometry against the IR contract and its declared output shape. -/
@[expose]
def concatLayout? (axis : Nat) (parentShapes : Array Shape) (outputShape : Shape) :
    Option ConcatLayout := do
  let expected ← (OpContracts.inferConcatOutShape axis parentShapes).toOption
  unless expected == outputShape do failure
  let first ← parentShapes[0]?
  let lengths := parentShapes.toList.map fun shape => shape.toList.getD axis 0
  let layout : ConcatLayout :=
    { leading := first.toList.take axis
      trailing := first.toList.drop (axis + 1)
      lengths := lengths }
  unless layout.outputShape == outputShape do failure
  unless parentShapes.toList ==
      (List.finRange layout.lengths.length).map layout.parentShape do failure
  pure layout

/-- Validate and collect every concat parent; missing ids are rejected before reading payloads. -/
@[expose]
def concatNodeLayout? (nodes : Array Node) (node : Node) (axis : Nat) :
    Option ConcatLayout := do
  let shapes ← node.parents.mapM fun parent => (nodes[parent]?).map (·.outShape)
  concatLayout? axis shapes node.outShape

/-- Concatenate flat values after checking every parent payload length. -/
@[expose]
def concatFlatValues? (layout : ConcatLayout) (values : Array (FlatTensor α)) :
    Option (FlatTensor α) := do
  unless values.size == layout.lengths.length do failure
  let inputs : Fin layout.lengths.length → FlatTensor α ←
    Tensor.Internal.sequenceFinM fun parent => values[parent.val]?
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
    Tensor.Internal.sequenceFinM fun parent => boxes[parent.val]?
  if h : ∀ parent, (inputs parent).dim = (layout.parentShape parent).size then
    pure
      { dim := layout.outputShape.size
        lo := layout.concat fun parent => h parent ▸ (inputs parent).lo
        hi := layout.concat fun parent => h parent ▸ (inputs parent).hi }
  else
    none

/-- Shared concat rule for value, first-derivative and mixed-second-derivative interval passes. -/
@[expose]
def concatNodeBoxes? (nodes : Array Node) (boxes : Array (Option (FlatBox α)))
    (node : Node) (axis : Nat) : Option (FlatBox α) := do
  let layout ← concatNodeLayout? nodes node axis
  let inputs ← node.parents.mapM fun parent => (boxes[parent]?).join
  concatFlatBoxes? layout inputs


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

  Trailing-shape value bounds use the stored scale, bias, and epsilon. An absent payload selects
  unit scale, zero bias, and default epsilon. The derivative transfers use the same stored
  parameters and reject incompatible payload shapes. -/
  layerNorm : Std.HashMap Nat (NN.IR.LayerNormParams α) := Std.HashMap.emptyWithCapacity

namespace ParamStore

/-- Insert an input interval box for a graph node. -/
def seedInputBox {α : Type} [TorchLean.Storage α] [Context α]
    (ps : ParamStore α) (inputId : Nat) (xB : FlatBox α) : ParamStore α :=
  { ps with inputBoxes := ps.inputBoxes.insert inputId xB }

/-- Seed a graph input with a uniform `ℓ∞` box around a shaped tensor. -/
def seedLInfBall {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α] {s : Shape}
    (ps : ParamStore α) (inputId : Nat) (center : Tensor α s) (eps : α) : ParamStore α :=
  ps.seedInputBox inputId <| FlatBox.lInfBall (α := α) center eps

end ParamStore

/--
Validate convolution geometry and return its leading batch shape.

The canonical IR contract checks channels, groups, dilation, stride, padding and spatial rank.
The payload and declared output must agree with the same geometry before any CROWN transfer runs.
-/
@[expose] def planConvTransfer?
    (configuration : NN.IR.ConvConfig) (parameters : NN.IR.ConvParams α)
    (parentShape outShape : Shape) : Option Shape := do
  let inferred ← (OpContracts.inferConvConfigOutShape "conv" configuration parentShape).toOption
  if inferred != outShape || !parameters.matchesConfig configuration then none else do
    let leading := Shape.ofList (parentShape.toList.take configuration.channelAxis)
    if parentShape == parameters.input leading && outShape == parameters.output leading then
      some leading
    else
      none

/--
Check the graph-level semantic restrictions imposed by the current CROWN engine.

Convolution payloads must match the declared geometry; LayerNorm payloads must match the entire
normalized suffix. This predicate checks the common shape contract; individual transfers also
check their arithmetic and derivative requirements.
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
                  (planConvTransfer?
                    (α := α) configuration parameters parent.outShape node.outShape).isSome
              | _, _ => false
          | _ => false
      | .concat axis =>
          (concatNodeLayout? nodes node axis).isSome
      | .matmul =>
          match node.parents with
          | #[leftId, rightId] =>
              match nodes[leftId]?, nodes[rightId]? with
              | some left, some right =>
                  match OpContracts.inferMatmulOutShape left.outShape right.outShape with
                  | .ok expected => expected == node.outShape
                  | .error _ => false
              | _, _ => false
          | #[parentId] =>
              match nodes[parentId]?, ps.matmulW[id]? with
              | some parent, some parameters =>
                  parameters.n == parent.outShape.size && parameters.m == node.outShape.size
              | _, _ => false
          | _ => false
      | .layernorm axis =>
          match node.parents with
          | #[parentId] =>
              match nodes[parentId]? with
              | some parent =>
                  if parent.outShape != node.outShape then
                    false
                  else
                    match OpContracts.layerNormMatrixDims axis node.outShape with
                    | .error _ => false
                    | .ok _ =>
                        match ps.layerNorm[id]? with
                        | none => true
                        | some parameters =>
                            parameters.normalizedShape ==
                              Shape.ofList (node.outShape.toList.drop axis)
              | none => false
          | _ => false
      | .softmax axis =>
          match node.parents with
          | #[parentId] =>
              match nodes[parentId]? with
              | some parent =>
                  parent.outShape == node.outShape &&
                    (OpContracts.checkAxisValid axis node.outShape).isOk
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
Finite traversals collect coordinatewise results while retaining typed indices.
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
Componentwise square of an interval box, with downward-rounded lower endpoint squares and
upward-rounded upper endpoint squares. The minimum is `0` when the interval crosses `0`.

The body is exposed because the proof layer theorem module unfolds this executable rule when
proving dimension preservation and pointwise enclosure.
-/
@[expose] def boxSquare (B : FlatBox α) : FlatBox α :=
  let lo' :=
    Tensor.ofFn fun i =>
      let l := B.lo.getScalar i
      let u := B.hi.getScalar i
      let l2 := mulDown l l
      let u2 := mulDown u u
      if l < 0 then
        if 0 < u then 0 else (if l2 < u2 then l2 else u2)
      else
        if l2 < u2 then l2 else u2
  let hi' :=
    Tensor.ofFn fun i =>
      let l2 := mulUp (B.lo.getScalar i) (B.lo.getScalar i)
      let u2 := mulUp (B.hi.getScalar i) (B.hi.getScalar i)
      if l2 > u2 then l2 else u2
  { dim := B.dim, lo := lo', hi := hi' }

/-- Interval multiplication for scalar endpoints: given `[aLo,aHi]` and `[bLo,bHi]`, return bounds
  on the product. -/
@[expose] def intervalMul (aLo aHi bLo bHi : α) : α × α :=
  (min2 (min2 (mulDown aLo bLo) (mulDown aLo bHi))
      (min2 (mulDown aHi bLo) (mulDown aHi bHi)),
    max2 (max2 (mulUp aLo bLo) (mulUp aLo bHi))
      (max2 (mulUp aHi bLo) (mulUp aHi bHi)))

/-- Directed contraction with the same scalar indexing and sum order as `Graph.matmulFlat`. -/
@[expose] def binaryMatmulBox (dims : OpContracts.MatmulDims) (left right : FlatBox α) :
    FlatBox α :=
  let endpoints (index : Fin dims.outShape.size) : α × α :=
    (List.range dims.inner).foldl (fun acc inner =>
      let leftIndex := dims.leftIndex index.val inner
      let rightIndex := dims.rightIndex index.val inner
      let product := intervalMul (α := α)
        (getAtOrZero left.lo [leftIndex]) (getAtOrZero left.hi [leftIndex])
        (getAtOrZero right.lo [rightIndex]) (getAtOrZero right.hi [rightIndex])
      (addDown acc.1 product.1, addUp acc.2 product.2)) (0, 0)
  { dim := dims.outShape.size
    lo := Tensor.ofFn fun index => (endpoints index).1
    hi := Tensor.ofFn fun index => (endpoints index).2 }

/-- Binary matmul with batch broadcasting, vector promotion, and directed endpoint arithmetic. -/
@[expose] def ibpBinaryMatmul? (leftShape rightShape : Shape) (left right : FlatBox α) :
    Option (FlatBox α) :=
  match (OpContracts.matmulDims leftShape rightShape).toOption with
  | none => none
  | some dims =>
      if left.dim = leftShape.size ∧ right.dim = rightShape.size then
        some (binaryMatmulBox dims left right)
      else none

/-- Reinterpret a flattened tensor as shape `s` when the element counts agree. -/
@[expose] def ibpUnflatten {s : Shape} (dim : Nat) (t : Tensor α [dim]) (h : dim =
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

/-- Sum one axis with downward lower accumulation and upward upper accumulation. -/
def ibpReduceSumAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s then
    match Spec.Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some _ =>
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceDim (fun row => Tensor.foldlSpec addDown 0 row) axis xLo
        let yHi := Tensor.reduceDim (fun row => Tensor.foldlSpec addUp 0 row) axis xHi
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
def ibpSoftmaxRange (s : Shape) (axis dim : Nat) : FlatBox α :=
  let rowLength := s.toList[axis]?.getD 0
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
## LayerNorm interval propagation

Normalization acts independently on each row of the configured trailing shape. Its directed
transfer bounds the mean, variance, stabilized denominator, and affine scale and bias.
-/

/--
Uniform finite enclosure for normalization over a trailing shape.

For a row of length `n`, exact LayerNorm without affine scale or bias satisfies
`|yᵢ| ≤ sqrt n`; a singleton row is identically zero. Backends provide the outward-rounded bound
through `NonlinearBoundOps.layerNormAbsBound`. Returning `none` is preferable to evaluating the
normalization with unqualified host division and square root.
-/
def ibpLayerNormRange? [NonlinearBoundOps α]
    (s : Shape) (dim : Nat) (axis : Nat := s.rank - 1) : Option (FlatBox α) :=
  let rowLength := (s.toList.drop axis).prod
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
@[expose] def checkedFiniteBounds? (bounds : α × α) : Option (α × α) :=
  if !(decide (bounds.2 < bounds.1)) && bounds.1 - bounds.1 == 0 &&
      bounds.2 - bounds.2 == 0 then
    some bounds
  else
    none

/--
Enclose eval-mode BatchNorm without treating rounded normalization coefficients as exact.

The running statistics are fixed, but their stabilized square root and division still need
directed bounds. Coordinates use the denominator interval for their channel.
-/
def ibpBatchNormEval? [NonlinearBoundOps α] (parentShape : Shape) (channelAxis : Nat)
    (config : NN.IR.BatchNormEvalParams α) (input : FlatBox α) : Option (FlatBox α) := do
  let channels ← parentShape.toList[channelAxis]?
  if config.c = 0 || channels != config.c || input.dim != parentShape.size then none else do
    let _ ← checkedFiniteBounds? (config.eps, config.eps)
    let parameters ← Internal.traverseFin fun ci : Fin config.c => do
      let mean := config.mean.getScalar ci
      let scale := config.gamma.getScalar ci
      let bias := config.beta.getScalar ci
      let variance := config.var.getScalar ci
      let _ ← checkedFiniteBounds? (mean, mean)
      let _ ← checkedFiniteBounds? (scale, scale)
      let _ ← checkedFiniteBounds? (bias, bias)
      let _ ← checkedFiniteBounds? (variance, variance)
      let (stabilizedLo, stabilizedHi) ← checkedFiniteBounds?
        (addDown (max variance 0) config.eps, addUp (max variance 0) config.eps)
      let (denominatorLo, denominatorHi) ←
        NonlinearBoundOps.sqrtBounds stabilizedLo stabilizedHi >>= checkedFiniteBounds?
      if !(denominatorLo > 0) then none else
        pure (mean, scale, bias, denominatorLo, denominatorHi)
    let bounds ← Internal.traverseFin fun i : Fin input.dim => do
      let channel := axisCoordinateOfFlat parentShape channelAxis i.val % config.c
      if hchannel : channel < config.c then do
        let (mean, scale, bias, denominatorLo, denominatorHi) := parameters ⟨channel, hchannel⟩
        let _ ← checkedFiniteBounds? (input.lo.getScalar i, input.hi.getScalar i)
        let (centeredLo, centeredHi) ← checkedFiniteBounds?
          (subDown (input.lo.getScalar i) mean, subUp (input.hi.getScalar i) mean)
        let (normalizedLo, normalizedHi) ←
          NonlinearBoundOps.divBounds centeredLo centeredHi denominatorLo denominatorHi >>=
            checkedFiniteBounds?
        let (scaledLo, scaledHi) ← checkedFiniteBounds?
          (intervalMul normalizedLo normalizedHi scale scale)
        checkedFiniteBounds? (addDown scaledLo bias, addUp scaledHi bias)
      else none
    pure
      { dim := input.dim
        lo := Tensor.ofFn fun i => (bounds i).1
        hi := Tensor.ofFn fun i => (bounds i).2 }

/-- Directed mean bounds for one nonempty row. The denominator encloses the exact row
length, including lengths that cannot be represented exactly by the scalar format. -/
@[expose] def directedRowMean? [NonlinearBoundOps α] {n : Nat}
    (bounds : Fin n → α × α) : Option (α × α) := do
  if n = 0 then none else do
    let (sumLo, sumHi, countLo, countHi) :=
      (List.finRange n).foldl (fun (lo, hi, countLo, countHi) i =>
        (addDown lo (bounds i).1, addUp hi (bounds i).2,
         addDown countLo 1, addUp countHi 1)) (0, 0, 0, 0)
    let _ ← checkedFiniteBounds? (sumLo, sumHi)
    let _ ← checkedFiniteBounds? (countLo, countHi)
    let result ← NonlinearBoundOps.divBounds sumLo sumHi countLo countHi
    checkedFiniteBounds? result

/-- Average a nonempty flat box, enclosing both the sum and the exact coordinate count. -/
def boxMean? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) := do
  let (lo, hi) ← directedRowMean? fun i => (B.lo.getScalar i, B.hi.getScalar i)
  pure { dim := 1, lo := Tensor.full [1] lo, hi := Tensor.full [1] hi }

/--
Average one axis using directed sums and division. The denominator encloses the exact axis
length, so a rounded natural-number conversion cannot silently narrow the result.
-/
def ibpReduceMeanAxis [NonlinearBoundOps α]
    (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) := do
  let summed ← ibpReduceSumAxis axis Xin s
  let length := s.toList[axis]?.getD 0
  let (countLo, countHi) :=
    (List.range length).foldl
      (fun (lo, hi) _ => (addDown lo 1, addUp hi 1)) (0, 0)
  let _ ← checkedFiniteBounds? (countLo, countHi)
  boxUnaryEnclosure? (fun lo hi => do
    let _ ← checkedFiniteBounds? (lo, hi)
    let result ← NonlinearBoundOps.divBounds lo hi countLo countHi
    checkedFiniteBounds? result) summed

/--
Directed bounds for one LayerNorm row with explicit affine parameters.

The order follows `Spec.layerNorm`: center the input, center again inside `reduceVar`, square,
average, clamp variance to zero, add epsilon, take a square root, and divide the first centered
values directly. Each coordinate is then multiplied by its gamma and shifted by its beta;
negative gamma reverses the interval endpoints.

Epsilon must be finite and positive, and the row and affine parameters must be finite. Division
and square root use the selected nonlinear backend. Invalid or non-finite intermediate intervals
return `none`. `Proofs.LayerNormEnclosure` proves enclosure of the real row under the backend's
arithmetic laws. Agreement with native arithmetic requires its own numerical correspondence.
-/
@[expose] def directedLayerNormRow? [NonlinearBoundOps α] {n : Nat}
    (lo hi gamma beta : Tensor α [n]) (epsilon : α) :
    Option (Tensor α [n] × Tensor α [n]) := do
  let _ ← checkedFiniteBounds? (epsilon, epsilon)
  if !(epsilon > 0) then none else do
    let _ ← Internal.traverseFin fun i : Fin n => do
      let _ ← checkedFiniteBounds? (lo.getScalar i, hi.getScalar i)
      let _ ← checkedFiniteBounds? (gamma.getScalar i, gamma.getScalar i)
      checkedFiniteBounds? (beta.getScalar i, beta.getScalar i)
    let (meanLo, meanHi) ← directedRowMean? fun i => (lo.getScalar i, hi.getScalar i)
    let centeredLo := Tensor.ofFn fun i => BoundOps.subDown (lo.getScalar i) meanHi
    let centeredHi := Tensor.ofFn fun i => BoundOps.subUp (hi.getScalar i) meanLo
    let _ ← Internal.traverseFin fun i : Fin n =>
      checkedFiniteBounds? (centeredLo.getScalar i, centeredHi.getScalar i)
    let (centerMeanLo, centerMeanHi) ← directedRowMean? fun i =>
      (centeredLo.getScalar i, centeredHi.getScalar i)
    let recenteredLo :=
      Tensor.ofFn fun i => BoundOps.subDown (centeredLo.getScalar i) centerMeanHi
    let recenteredHi :=
      Tensor.ofFn fun i => BoundOps.subUp (centeredHi.getScalar i) centerMeanLo
    let _ ← Internal.traverseFin fun i : Fin n =>
      checkedFiniteBounds? (recenteredLo.getScalar i, recenteredHi.getScalar i)
    let squaredLo := Tensor.ofFn fun i =>
      let l := recenteredLo.getScalar i
      let u := recenteredHi.getScalar i
      if !(decide (0 < l)) && !(decide (u < 0)) then 0
      else min2 (BoundOps.mulDown l l) (BoundOps.mulDown u u)
    let squaredHi := Tensor.ofFn fun i =>
      let l := recenteredLo.getScalar i
      let u := recenteredHi.getScalar i
      max2 (BoundOps.mulUp l l) (BoundOps.mulUp u u)
    let (varianceLo, varianceHi) ← directedRowMean? fun i =>
      (squaredLo.getScalar i, squaredHi.getScalar i)
    let (stabilizedLo, stabilizedHi) ← checkedFiniteBounds?
      (BoundOps.addDown (max2 varianceLo 0) epsilon,
       BoundOps.addUp (max2 varianceHi 0) epsilon)
    let (denominatorLo, denominatorHi) ←
      NonlinearBoundOps.sqrtBounds (max2 stabilizedLo 0) (max2 stabilizedHi 0) >>=
        checkedFiniteBounds?
    if !(denominatorLo > 0) then none else do
      let bounds ← Internal.traverseFin fun i : Fin n => do
        let (lower, upper) ←
          NonlinearBoundOps.divBounds (centeredLo.getScalar i) (centeredHi.getScalar i)
            denominatorLo denominatorHi >>= checkedFiniteBounds?
        let scale := gamma.getScalar i
        let (scaledLo, scaledHi) ← checkedFiniteBounds?
          (min2 (BoundOps.mulDown lower scale) (BoundOps.mulDown upper scale),
           max2 (BoundOps.mulUp lower scale) (BoundOps.mulUp upper scale))
        checkedFiniteBounds?
          (BoundOps.addDown scaledLo (beta.getScalar i),
           BoundOps.addUp scaledHi (beta.getScalar i))
      pure (Tensor.ofFn fun i => (bounds i).1, Tensor.ofFn fun i => (bounds i).2)

/--
Enclose a LayerNorm payload over the trailing shape starting at `axis`.

The matrix view and affine suffix are checked by the same helpers used in IR evaluation. Bounds
are propagated separately for each row, then flattened back into the graph's storage order.
Every payload, including one with default parameter values, uses the directed normalization
sequence with its stored affine parameters and epsilon.
-/
def ibpLayerNormPayloadBox? [NonlinearBoundOps α]
    (s : Shape) (axis : Nat) (parameters : NN.IR.LayerNormParams α)
    (input : FlatBox α) : Option (FlatBox α) := do
  let (rows, width) ← (OpContracts.layerNormMatrixDims axis s).toOption
  let payload : NN.IR.Payload α := { layerNorm? := fun _ => some parameters }
  let affine ←
    (NN.IR.Graph.resolveLayerNormAffine payload 0 axis s width).toOption
  let _ ← checkedFiniteBounds? (affine.epsilon, affine.epsilon)
  let _ ← if affine.epsilon > 0 then some () else none
  let _ ← Internal.traverseFin fun i : Fin width => do
    let _ ← checkedFiniteBounds? (affine.gamma.getScalar i, affine.gamma.getScalar i)
    checkedFiniteBounds? (affine.beta.getScalar i, affine.beta.getScalar i)
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

/--
Use the backend's uniform LayerNorm range when available; otherwise propagate the supplied input
box through the directed normalization sequence over the suffix beginning at `axis`.
Omitting `axis` selects the final axis.
-/
def ibpLayerNormBox? [NonlinearBoundOps α]
    (s : Shape) (input : FlatBox α) (axis : Nat := s.rank - 1) : Option (FlatBox α) := do
  let _ ← (OpContracts.layerNormMatrixDims axis s).toOption
  if input.dim = s.size then
    let _ ← Internal.traverseFin fun i : Fin input.dim =>
      checkedFiniteBounds? (input.lo.getScalar i, input.hi.getScalar i)
    match ibpLayerNormRange? (α := α) s input.dim axis with
    | some result => pure result
    | none =>
        let normalizedShape := Shape.ofList (s.toList.drop axis)
        ibpLayerNormPayloadBox? s axis
          { normalizedShape
            gamma := Tensor.full normalizedShape 1
            beta := Tensor.full normalizedShape 0
            eps := TorchLean.normalizationEpsilon }
          input
  else
    none

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
@[expose] def ibpConvNode (configuration : NN.IR.ConvConfig) (parentShape outShape : Shape)
    (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) := do
  let parameters ← ps.convCfg[id]?
  let leading ← planConvTransfer? configuration parameters parentShape outShape
  let sIn := parameters.input leading
  if hdim : Xin.dim = sIn.size then
    let xBox : Box α sIn :=
      { lo := ibpUnflatten Xin.dim Xin.lo hdim
        hi := ibpUnflatten Xin.dim Xin.hi hdim }
    let yBox := NN.MLTheory.CROWN.ibpConv parameters.spec parameters.dilation
      parameters.paddingAfter parameters.groups leading xBox
    some
      { dim := (parameters.output leading).size
        lo := Tensor.flattenSpec yBox.lo
        hi := Tensor.flattenSpec yBox.hi }
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

private def directedAvgPoolTensor? [NonlinearBoundOps α]
    (config : WindowConfig) (spatial : Tensor Nat [config.spatialRank]) (leading : Shape)
    (lo hi : Tensor α (leading.concat (Shape.ofList spatial.data.toList))) :
    Option (Box α (leading.concat (Shape.ofList
      (poolOutSpatialPad spatial config.kernel config.stride config.padding).data.toList))) :=
  match leading with
  | .scalar => do
      let outDims :=
        (poolOutSpatialPad spatial config.kernel config.stride config.padding).data.toList
      let kernelDims := config.kernel.data.toList
      let bounds ← Internal.traverseFin fun i : Fin (Shape.ofList outDims).size =>
        let outIndices := (flatCoordinates outDims.toArray i.val).toList
        directedRowMean? (n := (Shape.ofList kernelDims).size) fun j =>
          let windowIndices := (flatCoordinates kernelDims.toArray j.val).toList
          (Pooling.Internal.getPaddedAverageInputVal lo outIndices windowIndices
              config.stride.data.toList config.padding.data.toList,
           Pooling.Internal.getPaddedAverageInputVal hi outIndices windowIndices
              config.stride.data.toList config.padding.data.toList)
      pure
        { lo := Tensor.unflattenSpec (Shape.ofList outDims)
            (Tensor.ofFn fun i => (bounds i).1)
          hi := Tensor.unflattenSpec (Shape.ofList outDims)
            (Tensor.ofFn fun i => (bounds i).2) }
  | .dim n rest => do
      let bounds ← Internal.traverseFin fun i : Fin n =>
        directedAvgPoolTensor? config spatial rest (lo.unstack i) (hi.unstack i)
      pure
        { lo := Tensor.dim fun i => (bounds i).lo
          hi := Tensor.dim fun i => (bounds i).hi }

/--
Average-pool bounds with directed window sums, counts, and division.

The shared pooling plan validates the spatial suffix and preserves every leading axis. Windows
follow the spec's row-major coordinates and padded-cell lookup; padded zeros still contribute to
the denominator. Unavailable or nonfinite arithmetic and inconsistent declared shapes return `none`.
-/
def ibpAvgPool? [NonlinearBoundOps α] (config : WindowConfig)
    (parentShape outShape : Shape) (input : FlatBox α) : Option (FlatBox α) := do
  let plan ← (OpContracts.planPool "avg_pool" config parentShape).toOption
  if plan.outShape != outShape then none else
    if hdim : input.dim = parentShape.size then
      let lo : Tensor α parentShape := ibpUnflatten input.dim input.lo hdim
      let hi : Tensor α parentShape := ibpUnflatten input.dim input.hi hdim
      let bounds ← directedAvgPoolTensor? config plan.spatial plan.leading
        (plan.concat_eq.symm ▸ lo) (plan.concat_eq.symm ▸ hi)
      pure
        { dim := plan.outShape.size
          lo := Tensor.flattenSpec bounds.lo
          hi := Tensor.flattenSpec bounds.hi }
    else none


end NN.MLTheory.CROWN.Graph
