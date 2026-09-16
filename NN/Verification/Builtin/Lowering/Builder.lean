/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.Runtime.Autograd.Model.Program -- shake: keep

/-!
# Verifier IR Lowering

This interpreter records the forward operations of a TorchLean `Program` in `NN.IR.Graph`.
Tensor constants and layer weights go into a separate `ParamStore`, keyed by node identifier.
The resulting graph can be passed to the interval and CROWN checkers, which first check whether
their transfer rules support its operations and parameter payloads.

The builder handles arithmetic, shape operations, common activations, pooling, linear layers,
and arbitrary-rank convolution. Composite operations such as multi-head attention produce several
IR nodes. Training BatchNorm, tensor-valued gather/scatter indices, and selection on an
inner axis return lowering errors.

Softplus and `safeLog` have dedicated operations that retain the scalar specification's stable
branch. SafeLog stores epsilon as a scalar constant parent, so its value, including any dual
components, reaches the evaluator unchanged. Its outer logarithm follows the source backend's
semantics; certificate transfers separately check the domain needed for a sound enclosure.

LayerNorm stores the supplied epsilon, scale, and bias together in its parameter payload. We keep
epsilon even when its scalar equality test reports the default value: for example, `Dual` equality
compares the primal part, while its tangent may still differ. The IR evaluator uses the complete
payload. Last-axis IBP and CROWN value bounds use directed arithmetic for the normalization and
affine transform. The derivative passes leave these payloads unresolved until their rules account
for the supplied parameters and the specification's variance calculation.

Leading-axis `select` receives a `Fin` index while constructing the graph. Reading coordinate zero
of a state vector therefore becomes a constant one-hot projection followed by a reshape.
`indexSelect` and `scatterAdd` receive their indices as tensor data and need separate lowering and
transfer rules.

PyTorch weight and graph import is implemented under `NN.Runtime.PyTorch.Import`.

References (informal):
- IBP: Gowal et al. (2018).
- CROWN / DeepPoly-style linear relaxations: Zhang et al. (2018).
- LiRPA unification viewpoint: Xu et al. (2020).
-/

@[expose] public section


namespace NN.Verification.Builtin

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

/-! ## IR builder -/

/--
Reference produced while lowering a TorchLean program.

A value is either already materialized as an IR node, or it is still a compile-time tensor constant
that can be inserted into the verifier `ParamStore` if a later operation needs a node parent.
-/
inductive Ref (α : Type) [TorchLean.Storage α] : Shape → Type where
  | node  {s : Shape} (id : Nat) : Ref α s
  | const {s : Shape} (t : Tensor α s) : Ref α s

/-- Mutable builder state for translating a TorchLean program into verifier IR. -/
structure BuildState (α : Type) [TorchLean.Storage α] [Context α] where
  /-- IR nodes emitted so far, in construction/topological order. -/
  nodes : Array Node := #[]
  /-- Parameter payload accumulated for constant tensors and layer weights. -/
  ps    : NN.MLTheory.CROWN.Graph.ParamStore α := {}

/-- Builder monad used by TorchLean-to-IR lowering. -/
abbrev BuildM (α : Type) [TorchLean.Storage α] [Context α] : Type → Type :=
  StateT (BuildState α) (Except String)

/-- Raise a lowering error inside `BuildM`. -/
def fail {α : Type} [TorchLean.Storage α] [Context α] {β : Type} (msg : String) : BuildM α β :=
  throw msg

/-- Run a shared IR shape contract and surface its error from lowering. -/
def requireContract {α : Type} [TorchLean.Storage α] [Context α] {β : Type} (r : Except String β) :
    BuildM α Unit := do
  match r with
  | .ok _ => pure ()
  | .error msg => fail (α := α) msg

/-- Append a freshly constructed IR node to the builder state. -/
def pushNode {α : Type} [TorchLean.Storage α] [Context α] (n : Node) : BuildM α Unit := do
  modify fun st => { st with nodes := st.nodes.push n }

/-- Return the next node identifier, which is the current node-array size. -/
def freshId {α : Type} [TorchLean.Storage α] [Context α] : BuildM α Nat := do
  pure (←get).nodes.size

/--
Ensure a `Ref` is represented by an IR node.

Compile-time constants are materialized as `.const` nodes and recorded in the verifier
`ParamStore`; existing graph nodes are returned unchanged.
-/
def ensureNode {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (r : Ref α s) : BuildM α Nat := do
  match r with
  | .node id => pure id
  | .const t =>
      let id ← freshId (α := α)
      let flat := Tensor.flattenSpec (α := α) t
      let fv : NN.MLTheory.CROWN.Graph.FlatTensor α := { n := Spec.Shape.size s, v := flat }
      let node : Node := { id := id, parents := #[], kind := .const s, outShape := s }
      modify fun st => { st with ps := { st.ps with constVals := st.ps.constVals.insert id fv } }
      pushNode (α := α) node
      pure id

/-- Emit a unary IR operation with one parent node. -/
def emitUnary {α : Type} [TorchLean.Storage α] [Context α]
    {s t : Shape} (kind : OpKind) (x : Ref α s) (outShape : Shape := t) : BuildM α (Ref α t) := do
  let pid ← ensureNode (α := α) x
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := #[pid], kind := kind, outShape := outShape }
  pushNode (α := α) node
  pure (.node id)

/-- Emit a binary IR operation whose operands have the same shape. -/
def emitBinary {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (kind : OpKind) (a b : Ref α s) : BuildM α (Ref α s) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := #[pa, pb], kind := kind, outShape := s }
  pushNode (α := α) node
  pure (.node id)

/-- Emit a matrix-multiplication IR node. -/
def emitMatmul {α : Type} [TorchLean.Storage α] [Context α]
    {sA sB sOut : Shape} (a : Ref α sA) (b : Ref α sB) (outShape : Shape := sOut) :
    BuildM α (Ref α sOut) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := #[pa, pb], kind := .matmul, outShape := outShape }
  pushNode (α := α) node
  pure (.node id)

/-- Emit the designated verifier input node.  We keep this at id `0` for bound seeding. -/
def emitInput {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} : BuildM α (Ref α s) := do
  let id ← freshId (α := α)
  if id ≠ 0 then
    -- keep the designated input id stable for verifier seeding
    fail (α := α) "TorchLean IR lowering: internal error (input node must be id 0)"
  let node : Node := { id := id, parents := #[], kind := .input, outShape := s }
  pushNode (α := α) node
  pure (.node id)

/-- Read a compile-time constant tensor, failing if the value already depends on graph input. -/
def getConst {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (r : Ref α s) :
    BuildM α (Tensor α s) :=
  match r with
  | .const t => pure t
  | .node _ => fail (α := α)
    "TorchLean IR lowering: expected a compile-time constant tensor (got a graph node)"

/-- Lower one sample of multi-head attention into the verifier IR. -/
def emitMultiHeadAttention {α : Type} [TorchLean.Storage α] [Context α]
    {n numHeads dModel headDim : Nat}
    (wq : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wk : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wv : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wo : Ref α (.dim (numHeads * headDim) (.dim dModel .scalar)))
    (x : Ref α (.dim n (.dim dModel .scalar)))
    (mask : Option (Tensor Bool [n, n])) :
    BuildM α (Ref α (.dim n (.dim dModel .scalar))) := do
  let sX : Shape := .dim n (.dim dModel .scalar)
  let sBig : Shape := .dim n (.dim (numHeads * headDim) .scalar)
  let Q : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wq) (sOut := sBig) (outShape := sBig)
  let K : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wk) (sOut := sBig) (outShape := sBig)
  let V : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wv) (sOut := sBig) (outShape := sBig)

  -- The projection coordinate is `(head, coordinate-within-head)`. Preserve that row-major
  -- interpretation by reshaping each token first, then moving the head axis outward. Directly
  -- reshaping `(n, numHeads * headDim)` to `(numHeads, n, headDim)` changes which projected
  -- features belong to each head.
  let sProjected : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  let sHeads : Shape := .dim numHeads (.dim n (.dim headDim .scalar))
  let QProjected : Ref α sProjected ←
    emitUnary (α := α) (kind := .reshape sBig sProjected) (x := Q) (t := sProjected)
      (outShape := sProjected)
  let KProjected : Ref α sProjected ←
    emitUnary (α := α) (kind := .reshape sBig sProjected) (x := K) (t := sProjected)
      (outShape := sProjected)
  let VProjected : Ref α sProjected ←
    emitUnary (α := α) (kind := .reshape sBig sProjected) (x := V) (t := sProjected)
      (outShape := sProjected)
  let Qh : Ref α sHeads ←
    emitUnary (α := α) (kind := .transpose 0 1) (x := QProjected) (t := sHeads)
      (outShape := sHeads)
  let Kh : Ref α sHeads ←
    emitUnary (α := α) (kind := .transpose 0 1) (x := KProjected) (t := sHeads)
      (outShape := sHeads)
  let Vh : Ref α sHeads ←
    emitUnary (α := α) (kind := .transpose 0 1) (x := VProjected) (t := sHeads)
      (outShape := sHeads)

  let sKt : Shape := .dim numHeads (.dim headDim (.dim n .scalar))
  let Kt : Ref α sKt ←
    emitUnary (α := α) (kind := .transpose 1 2) (x := Kh) (t := sKt)
      (outShape := sKt)
  let sScores : Shape := .dim numHeads (.dim n (.dim n .scalar))
  let scores : Ref α sScores ←
    emitMatmul (α := α) (a := Qh) (b := Kt) (sOut := sScores) (outShape := sScores)

  let invScale : α := 1 / Spec.attentionScaleDenom (α := α) headDim
  let scaleTensor : Tensor α sScores := Tensor.full (α := α) sScores invScale
  let scaledScores ←
    emitBinary (α := α) (kind := .mulElem) (a := scores) (b := .const scaleTensor)

  -- A blocked mask entry has exactly zero numerator. No finite sentinel is introduced here.
  let attn : Ref α sScores ←
    match mask with
    | none =>
        emitUnary (α := α) (kind := .softmax (axis := 2)) (x := scaledScores) (t := sScores)
          (outShape := sScores)
    | some m => do
        let mask3D : Tensor Bool sScores := Tensor.dim (fun _ => m)
        emitUnary (α := α)
          (kind := .hardMaskedSoftmax (NN.IR.HardMask.ofTensor mask3D))
          (x := scaledScores) (t := sScores) (outShape := sScores)

  let outHeads : Ref α sHeads ←
    emitMatmul (α := α) (a := attn) (b := Vh) (sOut := sHeads) (outShape := sHeads)
  let sSwap : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  let swapped : Ref α sSwap ←
    emitUnary (α := α) (kind := .transpose 0 1) (x := outHeads) (t := sSwap)
      (outShape := sSwap)
  let concat : Ref α sBig ←
    emitUnary (α := α) (kind := .reshape sSwap sBig) (x := swapped) (t := sBig)
      (outShape := sBig)
  emitMatmul (α := α) (a := concat) (b := wo) (sOut := sX) (outShape := sX)

/-- Exact leading-axis slice expressed through the verifier's affine matrix fragment. -/
def emitLeadingSlice {α : Type} [TorchLean.Storage α] [Context α]
    {n len : Nat} {s : Shape} (start : Nat) (_h : start + len ≤ n)
    (x : Ref α (.dim n s)) : BuildM α (Ref α (.dim len s)) := do
  let block : Nat := Spec.Shape.size s
  let xMat : Ref α (.dim n (.dim block .scalar)) ←
    emitUnary (α := α)
      (kind := .reshape (.dim n s) (.dim n (.dim block .scalar)))
      (x := x) (t := .dim n (.dim block .scalar))
      (outShape := .dim n (.dim block .scalar))
  let selector : Tensor α [len, n] :=
    Tensor.dim (fun row =>
      Tensor.dim (fun col =>
        Tensor.scalar (if col.val = start + row.val then (1 : α) else (0 : α))))
  let yMat : Ref α (.dim len (.dim block .scalar)) ←
    emitMatmul (α := α) (a := .const selector) (b := xMat)
      (sOut := .dim len (.dim block .scalar))
      (outShape := .dim len (.dim block .scalar))
  emitUnary (α := α)
    (kind := .reshape (.dim len (.dim block .scalar)) (.dim len s))
    (x := yMat) (t := .dim len s) (outShape := .dim len s)

/-- Emit a verifier-IR concatenation along the leading axis. -/
def emitLeadingConcat {α : Type} [TorchLean.Storage α] [Context α]
    {n m : Nat} {s : Shape} (a : Ref α (.dim n s)) (b : Ref α (.dim m s)) :
    BuildM α (Ref α (.dim (n + m) s)) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node :=
    { id := id, parents := #[pa, pb], kind := .concat 0, outShape := .dim (n + m) s }
  pushNode (α := α) node
  pure (.node id)

/--
Lower batched attention as the leading-axis map of the single-sample verifier graph.

This is intentionally a semantic lowering rather than a claim that the verifier understands a new
opaque fused kernel.
-/
def emitBatchedMultiHeadAttention {α : Type} [TorchLean.Storage α] [Context α]
    {batch n numHeads dModel headDim : Nat}
    (wq : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wk : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wv : Ref α (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wo : Ref α (.dim (numHeads * headDim) (.dim dModel .scalar)))
    (x : Ref α (.dim batch (.dim n (.dim dModel .scalar))))
    (mask : Option (Tensor Bool [n, n])) :
    BuildM α (Ref α (.dim batch (.dim n (.dim dModel .scalar)))) :=
  match batch with
  | 0 =>
      pure <| .const <| Tensor.dim (fun i : Fin 0 => Fin.elim0 i)
  | batch + 1 => do
      let head1 ← emitLeadingSlice (α := α) (n := batch + 1) (len := 1) 0 (by simp) x
      let head : Ref α (.dim n (.dim dModel .scalar)) ←
        emitUnary (α := α)
          (kind := .reshape
            (.dim 1 (.dim n (.dim dModel .scalar)))
            (.dim n (.dim dModel .scalar)))
          (x := head1) (t := .dim n (.dim dModel .scalar))
          (outShape := .dim n (.dim dModel .scalar))
      let yHead ← emitMultiHeadAttention (α := α) wq wk wv wo head mask
      let yHead1 : Ref α (.dim 1 (.dim n (.dim dModel .scalar))) ←
        emitUnary (α := α)
          (kind := .reshape
            (.dim n (.dim dModel .scalar))
            (.dim 1 (.dim n (.dim dModel .scalar))))
          (x := yHead) (t := .dim 1 (.dim n (.dim dModel .scalar)))
          (outShape := .dim 1 (.dim n (.dim dModel .scalar)))
      let tail ← emitLeadingSlice (α := α) (n := batch + 1) (len := batch) 1
        (by simp [Nat.add_comm]) x
      let yTail ← emitBatchedMultiHeadAttention (α := α) wq wk wv wo tail mask
      let y ← emitLeadingConcat (α := α)
        (n := 1) (m := batch) (s := .dim n (.dim dModel .scalar)) yHead1 yTail
      return (by simpa [Nat.one_add] using y)

instance {α : Type} [TorchLean.Storage α] [Context α] :
    Runtime.Autograd.Torch.Ops (m := BuildM α) (α := α) where
  Ref := Ref α
  DataRef := fun β _ s => Tensor β s

  const := fun {_s} t => pure (.const t)
  dataConst := fun t => t
  mapData := fun f t => f t

  add := fun {_s} a b => emitBinary (α := α) (kind := .add) (a := a) (b := b)
  sub := fun {_s} a b => emitBinary (α := α) (kind := .sub) (a := a) (b := b)
  mul := fun {_s} a b => emitBinary (α := α) (kind := .mulElem) (a := a) (b := b)

  scale := fun {s} x c => do
    -- IR has no dedicated `scale`; encode as elementwise mul with a constant tensor.
    let cT : Tensor α s := Tensor.full s c
    emitBinary (α := α) (kind := .mulElem) (a := x) (b := .const cT)

  abs := fun {s} _x =>
    emitUnary (α := α) (kind := .abs) (x := _x) (t := s) (outShape := s)
  sqrt := fun {s} x =>
    emitUnary (α := α) (kind := .sqrt) (x := x) (t := s) (outShape := s)
  clamp := fun {s} x lo hi => do
    -- Lower to `min(max(x, lo), hi)` using const-filled tensors.
    let loT : Tensor α s := Tensor.full (α := α) s lo
    let hiT : Tensor α s := Tensor.full (α := α) s hi
    let y ← emitBinary (α := α) (kind := .maxElem) (a := x) (b := .const loT)
    emitBinary (α := α) (kind := .minElem) (a := y) (b := .const hiT)
  max := fun {_s} a b =>
    emitBinary (α := α) (kind := .maxElem) (a := a) (b := b)
  min := fun {_s} a b =>
    emitBinary (α := α) (kind := .minElem) (a := a) (b := b)

  broadcastTo := fun {s₁ s₂} _cb x => do
    emitUnary (α := α) (kind := .broadcastTo s₁ s₂) (x := x) (t := s₂) (outShape := s₂)

  reshape := fun {s₁ s₂} x _h => do
    let out : Shape := s₂
    emitUnary (α := α) (kind := .reshape s₁ s₂) (x := x) (t := s₂) (outShape := out)

  swapAdjacentAtDepth := fun {s} depth x => do
    let out := s.swapAdjacentAtDepth depth
    -- Equal axis lengths preserve the shape, but their coordinates still move. In particular,
    -- transposing a square matrix must retain the exchange of its off-diagonal entries.
    if depth + 1 < s.rank then
      let kind := OpKind.transpose depth (depth + 1)
      emitUnary (α := α) (kind := kind) (x := x) (t := out) (outShape := out)
    else if h : out = s then
      -- The total source operation leaves a tensor unchanged when the depth is out of range.
      pure (Eq.mp (congrArg (fun sh => Ref α sh) h.symm) x)
    else
      fail (α := α)
        s!"TorchLean→IR: swapAdjacentAtDepth (depth={depth}) invalid for shape {repr s}"

  reduceSum := fun {s} axis _valid _wf x => do
    let out : Shape := TorchLean.Tensor.shapeAfterSum s axis
    emitUnary (α := α) (kind := .reduceSum axis) (x := x) (t := out) (outShape := out)
  reduceMean := fun {s} axis _valid _wf x => do
    let out : Shape := TorchLean.Tensor.shapeAfterSum s axis
    emitUnary (α := α) (kind := .reduceMean axis) (x := x) (t := out) (outShape := out)

  -- A `select` with a statically known index is a constant one-hot projection, not a data-dependent
  -- gather, so it does belong to the verifier fragment: we reuse the same selector-matmul encoding
  -- that `sliceLeadingAxisRange` uses and then drop the length-one axis with a reshape. Only the
  -- leading axis is handled. An inner axis would need the projection applied under the outer
  -- dimensions, which the fragment cannot express without a stack/unstack pair, so that case still
  -- reports a fragment error rather than silently lowering to something else.
  select := fun {s} axis _axisInBounds x index =>
    -- Read the index out as a plain `Nat` before matching on the shape: its `Fin` type mentions
    -- both `s` and `axis`, and a `match` on those two does not refine it.
    let idx : Nat := index.val
    match s, axis with
    | .dim n rest, 0 =>
        if h : idx + 1 ≤ n then do
          let sliced : Ref α (.dim 1 rest) ←
            emitLeadingSlice (α := α) (n := n) (len := 1) (s := rest) idx h x
          emitUnary (α := α) (kind := .reshape (.dim 1 rest) rest)
            (x := sliced) (t := rest) (outShape := rest)
        else
          -- Unreachable: `index : Fin (axisSize s 0)` already bounds `idx` by `n`.
          fail (α := α) s!"TorchLean→IR: select index {idx} out of range for leading axis {n}"
    | _, _ + 1 =>
        fail (α := α)
          "TorchLean→IR: select on a non-leading axis is outside the verifier IR fragment"
    | .scalar, _ =>
        fail (α := α) "TorchLean→IR: select needs a positive-rank input shape"
  indexSelect := fun {_s} _axis _count _axisInBounds _x _indices =>
    fail (α := α) "TorchLean→IR: indexSelect is outside the verifier IR fragment"
  scatterAdd := fun {_s} _axis _count _axisInBounds _base _source _indices =>
    fail (α := α) "TorchLean→IR: scatterAdd is outside the verifier IR fragment"

  matmul := fun {_batchA _batchB batch mDim _nDim pDim} _broadcastA _broadcastB a b => do
    let out : Shape := batch.concat [mDim, pDim]
    emitMatmul (α := α) (a := a) (b := b) (sOut := out) (outShape := out)

  concatLeadingAxis := fun {_nDim _mDim} {_s} a b =>
    emitLeadingConcat (α := α) a b

  sliceLeadingAxisRange := fun {_nDim} {_s} start _len h x =>
    emitLeadingSlice (α := α) start h x

  maxPool := fun {d C} {inSpatial kernel stride padding} x => do
    let config : WindowConfig :=
      { spatialRank := d, kernel := kernel, stride := stride, padding := padding }
    let inputShape := Shape.ofList (C :: (Tensor.to inSpatial (List Nat)))
    let outputShape :=
      Shape.ofList
        (C :: (Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPoolOutShape "max_pool" kernel stride padding inputShape
    emitUnary (α := α) (kind := .maxPool config) (x := x) (t := outputShape)
      (outShape := outputShape)

  avgPool := fun {d C} {inSpatial kernel stride padding} x => do
    let config : WindowConfig :=
      { spatialRank := d, kernel := kernel, stride := stride, padding := padding }
    let inputShape := Shape.ofList (C :: (Tensor.to inSpatial (List Nat)))
    let outputShape :=
      Shape.ofList
        (C :: (Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPoolOutShape "avg_pool" kernel stride padding inputShape
    emitUnary (α := α) (kind := .avgPool config) (x := x) (t := outputShape)
      (outShape := outputShape)

  smoothMaxPool := fun {_d _C} {_inSpatial _kernel _stride _padding}
      [_decidableEq : DecidableEq α] _x _beta =>
    fail (α := α) "TorchLean→IR: smooth_max_pool is outside the verifier IR fragment"

  relu := fun {s} x => emitUnary (α := α) (kind := .relu) (x := x) (t := s) (outShape := s)
  sigmoid := fun {s} x => emitUnary (α := α) (kind := .sigmoid) (x := x) (t := s) (outShape := s)
  tanh := fun {s} x => emitUnary (α := α) (kind := .tanh) (x := x) (t := s) (outShape := s)
  gelu := fun {s} x => do
    -- Keep GELU explicit in verifier IR until the IR itself gains a dedicated opcode. Runtime
    -- backends use one tape node; this expansion remains transparent to graph proofs.
    let c0 : α := Activation.Math.geluTanhCoeff
    let c1 : α := MathFunctions.sqrt (2 / MathFunctions.pi)
    let x2 ← emitBinary (α := α) (kind := .mulElem) (a := x) (b := x)
    let x3 ← emitBinary (α := α) (kind := .mulElem) (a := x2) (b := x)
    let c0Tensor : Tensor α s := Tensor.full s c0
    let scaledX3 ← emitBinary (α := α) (kind := .mulElem) (a := x3) (b := .const c0Tensor)
    let inner ← emitBinary (α := α) (kind := .add) (a := x) (b := scaledX3)
    let c1Tensor : Tensor α s := Tensor.full s c1
    let tanhInput ← emitBinary (α := α) (kind := .mulElem) (a := inner) (b := .const c1Tensor)
    let tanhOut ← emitUnary (α := α) (kind := .tanh) (x := tanhInput) (t := s) (outShape := s)
    let ones : Tensor α s := Tensor.full s 1
    let onePlus ← emitBinary (α := α) (kind := .add) (a := tanhOut) (b := .const ones)
    let mid ← emitBinary (α := α) (kind := .mulElem) (a := x) (b := onePlus)
    let halves : Tensor α s := Tensor.full s (1 / 2)
    emitBinary (α := α) (kind := .mulElem) (a := mid) (b := .const halves)
  softmaxLast := fun {s} x => do
    let axis :=
      match Spec.Shape.rank s with
      | 0 => 0
      | Nat.succ r => r
    emitUnary (α := α) (kind := .softmax axis) (x := x) (t := s) (outShape := s)
  logSoftmaxLast := fun {s} x => do
    -- The verifier IR represents `log_softmax` by lowering it through
    -- `softmax` followed by `log` so the semantic graph remains expressible; eager/typed graph
    -- training still uses the stable primitive from the autograd runtime.
    let axis :=
      match Spec.Shape.rank s with
      | 0 => 0
      | Nat.succ r => r
    let probs ← emitUnary (α := α) (kind := .softmax axis) (x := x) (t := s) (outShape := s)
    emitUnary (α := α) (kind := .log) (x := probs) (t := s) (outShape := s)
  softplus := fun {s} x =>
    emitUnary (α := α) (kind := .softplus) (x := x) (t := s) (outShape := s)
  exp := fun {s} x => emitUnary (α := α) (kind := .exp) (x := x) (t := s) (outShape := s)
  sin := fun {s} x => emitUnary (α := α) (kind := .sin) (x := x) (t := s) (outShape := s)
  cos := fun {s} x => emitUnary (α := α) (kind := .cos) (x := x) (t := s) (outShape := s)
  log := fun {s} x => emitUnary (α := α) (kind := .log) (x := x) (t := s) (outShape := s)
  inv := fun {s} x => emitUnary (α := α) (kind := .inv) (x := x) (t := s) (outShape := s)
  detach := fun {s} x => emitUnary (α := α) (kind := .detach) (x := x) (t := s) (outShape := s)
  safeLog := fun {s} x epsilon => do
    let px ← ensureNode (α := α) x
    let pe ← ensureNode (α := α) (.const (Tensor.scalar epsilon))
    let id ← freshId (α := α)
    pushNode (α := α) { id, parents := #[px, pe], kind := .safeLog, outShape := s }
    pure (.node id)
  sum := fun {_s} x =>
    emitUnary (α := α) (kind := .sum) (x := x) (t := Shape.scalar) (outShape := Shape.scalar)
  flatten := fun {s} x => do
    let outShape : Shape := .dim (Spec.Shape.size s) .scalar
    emitUnary (α := α) (kind := .flatten s) (x := x) (t := outShape) (outShape := outShape)

  linear := fun {inDim outDim} w b x => do
    let wT ← getConst (α := α) (s := .dim outDim (.dim inDim .scalar)) w
    let bT ← getConst (α := α) (s := .dim outDim .scalar) b
    let xId ← ensureNode (α := α) (s := .dim inDim .scalar) x
    let id ← freshId (α := α)
    let node : Node :=
      { id := id, parents := #[xId], kind := .linear, outShape := .dim outDim .scalar }
    modify fun st =>
      { st with
          ps := { st.ps with
            linearWB := st.ps.linearWB.insert id { m := outDim, n := inDim, w := wT, b := bT } } }
    pushNode (α := α) node
    pure (.node id)

  mseLoss := fun {s} yhat target => do
    let yId ← ensureNode (α := α) (s := s) yhat
    let tId ← ensureNode (α := α) (s := s) target
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := #[yId, tId], kind := .mseLoss, outShape :=
      Shape.scalar }
    pushNode (α := α) node
    pure (.node id)

  layerNorm := fun {seqLen embedDim} _hSeq _hEmb x gamma beta epsilon => do
    let sX : Shape := .dim seqLen (.dim embedDim .scalar)
    let gammaT ← getConst (α := α) (s := .dim embedDim .scalar) gamma
    let betaT ← getConst (α := α) (s := .dim embedDim .scalar) beta
    let xId ← ensureNode (α := α) x
    let id ← freshId (α := α)
    let params : LayerNormParams α :=
      { normalizedShape := [embedDim], gamma := gammaT, beta := betaT, eps := epsilon }
    modify fun st => { st with ps := { st.ps with layerNorm := st.ps.layerNorm.insert id params } }
    pushNode (α := α)
      { id := id, parents := #[xId], kind := .layernorm (axis := 1), outShape := sX }
    pure (.node id)

  batchNorm := fun {_channels _sSpatial} _hWellFormed _x _gamma _beta _epsilon => do
    fail (α := α)
      ("TorchLean→IR: batch_norm (training-style BN: stats from " ++
        "x) is outside the verifier IR fragment. For inference-time BN, " ++
        "use the eval-mode batch-normalization operation with explicit running statistics.")

  multiHeadAttention := fun {_n _numHeads _dModel _headDim} _h1 wq wk wv wo x mask =>
    emitMultiHeadAttention (α := α) wq wk wv wo x mask

  batchedMultiHeadAttention :=
    fun {_batch _n _numHeads _dModel _headDim} _hBatch _h1 wq wk wv wo x mask =>
      emitBatchedMultiHeadAttention (α := α) wq wk wv wo x mask

  conv := fun {d inC outC} {kernel stride padding} {inSpatial} w b x => do
    if hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0 then
      if hStride : ∀ i : Fin d, stride.getScalar i ≠ 0 then
        let config : ConvConfig :=
          { spatialRank := d
            kernel := kernel
            stride := stride
            padding := padding
            dilation := Tensor.full [d] 1
            paddingAfter := padding
            groups := 1
            channelAxis := 0
            inChannels := inC
            outChannels := outC }
        let inputShape := Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))
        let outputShape :=
          Shape.ofList (outC ::
            (Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat)))
        requireContract (α := α) <|
          NN.IR.OpContracts.inferConvOutShape "conv" 0 inC outC kernel stride padding inputShape
        let kernelTensor ← getConst (α := α) w
        let biasTensor ← getConst (α := α) b
        let inputId ← ensureNode (α := α) x
        let id ← freshId (α := α)
        let spec : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := kernelTensor, bias := biasTensor }
        let params : NN.IR.ConvParams α :=
          { spatialRank := d
            inChannels := inC
            outChannels := outC
            kernel := kernel
            stride := stride
            padding := padding
            dilation := Tensor.full [d] 1
            paddingAfter := padding
            groups := 1
            inputSpatial := inSpatial
            kernelNonzero := hKernel
            strideNonzero := hStride
            spec := spec }
        modify fun state =>
          { state with ps := { state.ps with convCfg := state.ps.convCfg.insert id params } }
        pushNode (α := α)
          { id := id, parents := #[inputId], kind := .conv config, outShape := outputShape }
        pure (.node id)
      else
        fail (α := α) "TorchLean→IR: conv: every stride must be nonzero"
    else
      fail (α := α) "TorchLean→IR: conv: every kernel extent must be nonzero"

  convTranspose := fun {_d _inC _outC} {_kernel _stride _padding} {_inSpatial} _w
      _b _x =>
    fail (α := α) "TorchLean→IR: conv_transpose is outside the verifier IR fragment"

  randUniform := fun {s} seed => do
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := #[], kind := .randUniform seed, outShape := s }
    pushNode (α := α) node
    pure (.node id)

  bernoulliMask := fun {s} keepProb seed => do
    let pId ← ensureNode (α := α) keepProb
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := #[pId], kind := .bernoulliMask seed, outShape := s }
    pushNode (α := α) node
    pure (.node id)

end NN.Verification.Builtin
