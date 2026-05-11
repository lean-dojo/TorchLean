/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.OpContracts
public import NN.Runtime.Context
public import NN.Runtime.Autograd.TorchLean.Random
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling

import NN.Spec.Core.Tensor.Linalg

/-!
# Semantics

Denotational semantics for `NN.IR.Graph`.

This file defines an evaluator for the current IR fragment:
- it evaluates nodes in SSA/topological order,
- each node applies the corresponding *spec-layer* tensor operation to its parents,
- parameter payloads for `const`, `linear`, and `conv2d` are supplied by an explicit `Payload`.

The evaluator is total on well-formed, well-shaped graphs and returns `Except String` on malformed
graphs or missing payloads.

Softmax and layer norm:
- `softmax axis` is interpreted as softmax along the given `axis`, but the spec primitive we have
  is last-axis softmax (`Activation.softmax_spec`). We therefore interpret non-last-axis softmax by
  *permuting* the requested axis to the last position, applying last-axis softmax, then permuting
  back. This matches the meaning of `torch.softmax(x, dim=axis)` in PyTorch.
- `layernorm axis` matches PyTorch's `F.layer_norm(x, normalized_shape=x.shape[axis:])` convention:
  `axis` selects the start of the **normalized suffix**. We implement this by reshaping the tensor
  into a 2D view `(seqLen, embedDim)`, applying the spec 2D LayerNorm (`Spec.layerNorm`), then
  reshaping back. This keeps the spec primitive small while supporting arbitrary ranks.

How this relates to PyTorch:
- `Graph.nodes` is analogous to a topologically-sorted IR like FX/TorchScript.
- `Payload` is analogous to “parameters / buffers / constants” that live outside the pure graph
  structure.
- The evaluator is a pure, denotational model of running the graph. It is designed for clarity and
  for connecting to proofs and verification passes (not for performance).

References / related systems:
- PyTorch FX: https://pytorch.org/docs/stable/fx.html
- TorchScript: https://pytorch.org/docs/stable/jit.html
- ONNX (graph + initializers): https://onnx.ai/
-/

@[expose] public section


namespace NN.IR

open _root_.Spec
open _root_.Spec.Tensor

/-!
## Parameter payloads

The IR graph stores `OpKind` and `outShape`, but it does not embed tensor values for parameters.
Instead, evaluation is parameterized by a `Payload`:

- `const? id` supplies a constant tensor for a `const` node,
- `linear? id` supplies `W,b` for a `linear` node,
- `conv2d? id` supplies the convolution spec/weights for a `conv2d` node.

This matches how most graph formats work in practice: structure is one artifact, parameters are
another.
-/

/--
Payload record for a `const` node.

Constants are stored in a “flat” (1-D) representation so backends can keep a uniform container
(e.g. an array). During evaluation we check the flat length against `Shape.size` and then
`unflatten` to the requested `Shape`.
-/
structure ConstFlat (α : Type) [Context α] where
  /-- Number of scalar entries stored in the flat constant payload. -/
  n : Nat
  /-- Constant values stored as a vector before evaluation reshapes them to the IR node shape. -/
  v : Tensor α (.dim n .scalar)

/--
Payload record for a `linear` node: weight matrix `W` and bias vector `b`.

The node's input `x` comes from the graph edge; `W,b` live in the external `Payload` (similar to
ONNX initializers or a PyTorch `state_dict`).
-/
structure LinearWB (α : Type) [Context α] where
  /-- Output dimension. -/
  outDim : Nat
  /-- Input dimension. -/
  inDim  : Nat
  /-- Weight matrix in the PyTorch convention `outDim × inDim`. -/
  W : Tensor α (.dim outDim (.dim inDim .scalar))
  /-- Bias vector added after matrix-vector multiplication. -/
  b : Tensor α (.dim outDim .scalar)

/--
Payload record for a `conv2d` node.

We store the spec-layer `Conv2DSpec` together with the dimension parameters needed to reconstruct
it. The nonzero proofs are required by the spec-layer definition and ensure the convolution is
well-formed.
-/
structure Conv2DParams (α : Type) [Context α] where
  /-- Input channels. -/
  inC : Nat
  /-- Output channels. -/
  outC : Nat
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride. -/
  stride : Nat
  /-- Padding size. -/
  padding : Nat
  /-- in H. -/
  inH : Nat
  /-- Input width. -/
  inW : Nat
  /-- Proof that the input channel count is nonzero, required by the spec convolution layer. -/
  hIn : inC ≠ 0
  /-- Proof that the kernel height is nonzero. -/
  hKH : kH ≠ 0
  /-- Proof that the kernel width is nonzero. -/
  hKW : kW ≠ 0
  /-- Spec-layer convolution package containing weights, bias, and convolution metadata. -/
  spec : Spec.Conv2DSpec inC outC kH kW stride padding α hIn hKH hKW

/--
External parameter payloads keyed by IR node id.

This is deliberately small; different backends may store parameters differently.
-/
structure Payload (α : Type) [Context α] where
  -- Each lookup is keyed by the *node id*.
  const?  : Nat → Option (ConstFlat α) := fun _ => none
  linear? : Nat → Option (LinearWB α) := fun _ => none
  conv2d? : Nat → Option (Conv2DParams α) := fun _ => none

/-!
## Dynamic (shape-tagged) values

During evaluation we keep values in a dependent pair `Σ s, Tensor α s` so we can store a
  heterogenous
table of intermediate tensors while still recovering precise shapes when we need them.
-/

/--
Dynamic (shape-tagged) tensor value used by the IR evaluator.

This is a dependent pair `Σ s, Tensor α s`, which lets us store heterogeneously-shaped intermediate
values in one table while still recovering exact shapes when needed.
-/
abbrev DVal (α : Type) [Context α] : Type :=
  Σ s : Shape, Tensor α s

namespace DVal

/-- The shape tag carried by a dynamic value. -/
@[simp] def shape {α : Type} [Context α] (v : DVal α) : Shape := v.1

/-- The underlying tensor, with its shape recovered from the dependent pair. -/
@[simp] def tensor {α : Type} [Context α] (v : DVal α) : Tensor α v.shape := v.2

/-- Construct a dynamic value from a shape and a tensor of that shape. -/
@[simp] def mk {α : Type} [Context α] (s : Shape) (t : Tensor α s) : DVal α := ⟨s, t⟩

end DVal

namespace Graph

/-!
## Small proof-helpers used by evaluation

The evaluator frequently needs evidence that an axis is valid or that a broadcast is legal so it can
call the spec-layer operations, which are typed with these preconditions. We build these witnesses
from runtime data (`Nat` axis values and shapes) using `Option`:

- returning `none` means the IR node is ill-formed for the given shapes,
- returning `some h` provides the witness needed to call the spec operator.
-/

/--
Build a witness that `axis` is a valid axis for shape `s`.

Many spec-layer ops (e.g. reductions, `softmax`, `layernorm`) are typed with a `Shape.valid_axis`
precondition. Since the IR stores axes as raw `Nat`, we reconstruct the witness at runtime.

Returns `none` when `axis` is out of bounds.
-/
def mkValidAxis? (axis : Nat) : (s : Shape) → Option (PLift (Shape.valid_axis axis s))
  | .scalar => none
  | .dim n rest =>
      match axis, n with
      | 0, Nat.succ k => some ⟨Shape.valid_axis.valid_zero (n := k) (s := rest)⟩
      | 0, 0 => none
      | Nat.succ a, Nat.succ k =>
          (mkValidAxis? a rest).map (fun h =>
            ⟨Shape.valid_axis.valid_succ (n := k) (s := rest) (k := a) h.down⟩)
      | Nat.succ _, 0 => none

/--
Build a witness that `s₁` can be broadcast to `s₂` (NumPy/PyTorch-style broadcasting).

The spec-layer broadcasting operator is typed with `Shape.CanBroadcastTo`. Since the IR stores only
runtime shapes, we reconstruct this witness on demand.

Returns `none` when broadcasting is not possible.
-/
def mkCanBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
  | .scalar, s₂ => some (.scalar_to_any s₂)
  | .dim n₁ t₁, .dim n₂ t₂ =>
      if hEq : n₁ = n₂ then
        (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
          hEq ▸ Shape.CanBroadcastTo.dim_eq (n := n₁) (s₁ := t₁) (s₂ := t₂) tail)
      else if h1 : n₁ = 1 then
        (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
          h1 ▸ Shape.CanBroadcastTo.dim_1_to_n (n := n₂) (s₁ := t₁) (s₂ := t₂) tail)
      else
        -- expand dims: broadcast s₁ to `t₂`, then add a leading dim
        (mkCanBroadcastTo? (.dim n₁ t₁) t₂).map (fun tail =>
          Shape.CanBroadcastTo.expand_dims (n := n₂) (s₁ := .dim n₁ t₁) (s₂ := t₂) tail)
  | _, _ => none

/-- Return the index of the first occurrence of `x` in `xs` (or `none` if absent). -/
def findIndex? (xs : List Nat) (x : Nat) : Option Nat :=
  let rec go (i : Nat) : List Nat → Option Nat
    | [] => none
    | y :: ys => if y = x then some i else go (i + 1) ys
  go 0 xs

/-- Safe list indexing: `listGet? xs n` returns `some xs[n]` when in bounds. -/
def listGet? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n + 1 => listGet? xs n

/-- Swap the adjacent entries at positions `d` and `d+1` (no-op when out of range). -/
def swapAt (xs : List Nat) (d : Nat) : List Nat :=
  match xs, d with
  | [], _ => []
  | [x], _ => [x]
  | x :: y :: rest, 0 => y :: x :: rest
  | x :: rest, d + 1 => x :: swapAt rest d

/--
Compute a sequence of adjacent swaps that realizes a target permutation.

This is used to implement `.permute` by repeatedly applying `swapAdjacentAtDepth`, which is already
available in the spec tensor library. If the permutation is ill-formed, this returns an error
explaining what went wrong.
-/
def swapDepthsForPerm (perm : List Nat) (r : Nat) : Except String (List Nat) := do
  let mut cur : List Nat := List.range r
  let mut swapsRev : List Nat := []
  for i in [0:r] do
    match listGet? perm i with
    | none => throw s!"permute: internal error: missing perm[{i}]"
    | some target =>
        match findIndex? cur target with
        | none => throw s!"permute: internal error: target axis {target} not in current axes {cur}"
        | some j =>
            let mut k := j
            while k > i do
              swapsRev := (k - 1) :: swapsRev
              cur := swapAt cur (k - 1)
              k := k - 1
  pure swapsRev.reverse

/--
Apply one adjacent-swap-at-depth to a dynamic tensor value.

This is the execution-level building block used to implement `.permute` in terms of repeated
adjacent swaps, reusing the spec tensor library's `swapAdjacentAtDepth`.
-/
def applySwapDepth {α : Type} [Context α] (v : DVal α) (d : Nat) : DVal α :=
  match v with
  | ⟨s, t⟩ =>
      let t' : Tensor α (s.swapAdjacentAtDepth d) := Tensor.swapAtDepthHelper (tensor := t) d
      ⟨s.swapAdjacentAtDepth d, t'⟩

/--
Permute a dynamic tensor value according to `perm`.

This checks that `perm` is a valid permutation for the input shape (using `Shape.permute?`), then
lowers it to a sequence of adjacent swaps and applies them to the tensor.
-/
def permuteDVal {α : Type} [Context α] (v : DVal α) (perm : List Nat) : Except String (DVal α) := do
  let sIn := v.shape
  match Spec.Shape.permute? sIn perm with
  | none => throw s!"permute: invalid permutation {repr perm} for shape {repr sIn}"
  | some _ =>
      let swaps ← swapDepthsForPerm perm (Shape.rank sIn)
      pure <| swaps.foldl (fun acc d => applySwapDepth (α := α) acc d) v

/-!
## Evaluation helpers

The evaluator itself (`evalAt` / `denoteAll`) is a fold over nodes. These helpers keep the fold
readable:
- `expectShape` enforces “dynamic shape agrees with declared `outShape`” at each step.
- `evalConst`/`evalLinear`/`evalConv2D` fetch and apply external payloads keyed by node id.
-/

/-- Check a dynamic value has the expected shape and return it as a statically-typed tensor. -/
def expectShape {α : Type} [Context α] [DecidableEq Shape]
    (expected : Shape) (v : DVal α) : Except String (Tensor α expected) := do
  if h : v.shape = expected then
    -- transport across the shape equality
    pure (h ▸ v.tensor)
  else
    throw s!"IR eval: shape mismatch: expected {repr expected}, got {repr v.shape}"

/-- Evaluate MSE loss on two dynamic values, checking that their runtime shapes agree. -/
def mseLossDVal {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (yVal tVal : DVal α) : Except String (DVal α) := do
  if h : yVal.shape = tVal.shape then
    let yT : Tensor α yVal.shape := yVal.tensor
    let tT : Tensor α yVal.shape := h.symm ▸ tVal.tensor
    let s := yVal.shape
    let diff := Tensor.subSpec (α := α) yT tT
    let sq := Tensor.mulSpec (α := α) diff diff
    let total : α := Tensor.sumSpec (α := α) sq
    let mean : α := total / (↑(Shape.size s) : α)
    pure (DVal.mk (α := α) Shape.scalar (Tensor.scalar mean))
  else
    throw <|
      s!"IR eval: node {i}: mse_loss expects equal shapes, got " ++
        s!"{repr yVal.shape} vs {repr tVal.shape}"

@[simp] theorem mseLossDVal_mk {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) {s : Shape} (y t : Tensor α s) :
    mseLossDVal (α := α) i (DVal.mk (α := α) s y) (DVal.mk (α := α) s t) =
      .ok (DVal.mk (α := α) Shape.scalar
        (Tensor.scalar
          (((Tensor.subSpec (α := α) y t).mulSpec (Tensor.subSpec (α := α) y t)).sumSpec /
            (↑(Shape.size s) : α)))) := by
  simp [mseLossDVal, DVal.shape, DVal.tensor, DVal.mk]
  rfl

/-- Transport a `Tensor α (dim n scalar)` across an equality `n = n'` (helper for payload casts). -/
def castDimScalar {α : Type} [Context α] {n n' : Nat}
    (h : n = n') (t : Tensor α (Shape.dim n Shape.scalar)) : Tensor α (Shape.dim n' Shape.scalar) :=
      by
  simpa [h] using t

/--
Evaluate a `const` node from the external payload.

Constants are stored “flat” (1D) for convenience, so we check the flattened length matches
`Shape.size s` and then `unflatten` to the requested shape.
-/
def evalConst {α : Type} [Context α] [Inhabited α]
    (payload : Payload α) (id : Nat) (s : Shape) : Except String (Tensor α s) := do
  match payload.const? id with
  | none => throw s!"IR eval: missing const payload for node {id}"
  | some c =>
      if h : c.n = Shape.size s then
        let v' : Tensor α (.dim (Shape.size s) .scalar) := castDimScalar (α := α) h c.v
        pure (Tensor.unflattenSpec (α := α) (s := s) v')
      else
        throw s!"IR eval: const {id}: flat length mismatch: have {c.n}, expected {Shape.size s}"

/--
Evaluate a `linear` node from the external payload.

We enforce:
- the input dynamic value has shape `(inDim)`, and
- the node's declared outShape matches `(outDim)`.

The actual math is the usual affine map: `y = W·x + b`.
-/
def evalLinear {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat) (x : DVal α) (outShape : Shape) : Except String (DVal α) := do
  match payload.linear? id with
  | none => throw s!"IR eval: missing linear payload for node {id}"
  | some p =>
      let expectedIn : Shape := Shape.dim p.inDim Shape.scalar
      let expectedOut : Shape := Shape.dim p.outDim Shape.scalar
      let xT ← expectShape (α := α) (expected := expectedIn) x
      if hOut : outShape = expectedOut then
        let y : Tensor α (Shape.dim p.outDim Shape.scalar) :=
        Tensor.addSpec (α := α) (Spec.matVecMulSpec (α := α) (m := p.outDim) (n := p.inDim) p.W
          xT) p.b
        pure (DVal.mk (α := α) outShape (hOut ▸ y))
      else
          throw <|
          s!"IR eval: linear {id}: declared outShape mismatch: {repr outShape} vs " ++
            s!"expected {repr expectedOut}"

/--
Evaluate a `conv2d` node from the external payload.

The output shape is computed with the standard (no dilation) formula:
`out = ⌊(in + 2*pad - k)/stride⌋ + 1` for each spatial dimension.
-/
def evalConv2D {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat) (x : DVal α) : Except String (DVal α) := do
  match payload.conv2d? id with
  | none => throw s!"IR eval: missing conv2d payload for node {id}"
  | some cfg =>
      let xT ← expectShape (α := α)
        (expected := Shape.dim cfg.inC (Shape.dim cfg.inH (Shape.dim cfg.inW Shape.scalar))) x
      let y := Spec.conv2dSpec (α := α)
        (layer := cfg.spec)
        (input := xT)
      let outH : Nat := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
      let outW : Nat := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
      let outShape : Shape := Shape.dim cfg.outC (Shape.dim outH (Shape.dim outW Shape.scalar))
      pure (DVal.mk (α := α) outShape y)

/-- Deterministic LayerNorm used by the IR evaluator (gamma=1, beta=0). -/
def layernormPure {α : Type} [Context α]
    (seqLen embedDim : Nat) (x : Tensor α (Shape.dim seqLen (Shape.dim embedDim Shape.scalar))) :
    Except String (Tensor α (Shape.dim seqLen (Shape.dim embedDim Shape.scalar))) := do
  if hSeq : seqLen > 0 then
    if hEmb : embedDim > 0 then
      let gamma : Tensor α (Shape.dim embedDim Shape.scalar) :=
        Spec.fill (α := α) 1 (Shape.dim embedDim Shape.scalar)
      let beta : Tensor α (Shape.dim embedDim Shape.scalar) :=
        Spec.fill (α := α) 0 (Shape.dim embedDim Shape.scalar)
      pure (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (x := x) (gamma := gamma) (beta := beta) (h_seq_pos := hSeq) (h_embed_pos := hEmb))
    else
      throw s!"layernorm: embedDim must be > 0 (got {embedDim})"
  else
    throw s!"layernorm: seqLen must be > 0 (got {seqLen})"

/--
Evaluate node `i` given already computed parent values `vals`.

This is the core “one step” of the denotational semantics:
- lookup the node,
- read its parent values from `vals` (using the topo/id invariant),
- apply the corresponding spec-layer operation,
- enforce that the produced shape matches the node’s declared `outShape`.

This function assumes the graph is structurally well-formed (ids are in bounds and parents are
strictly smaller ids). `denoteAll` performs that check up front.
-/
def evalAt
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : DVal α) (vals : Array (DVal α)) (i : Nat) :
    Except String (DVal α) := do
  let n ← g.getNode i
  let getParent (pid : Nat) : DVal α := vals[pid]!
  let v : DVal α ←
    match n.kind with
    | .input =>
        let t ← expectShape (α := α) (expected := n.outShape) input
        pure (DVal.mk (α := α) n.outShape t)
    | .const s =>
        let t ← evalConst (α := α) (payload := payload) (id := n.id) (s := s)
        pure (DVal.mk (α := α) s t)
    | .permute perm =>
        match n.parents with
        | [pId] =>
            let vOut ← permuteDVal (α := α) (v := getParent pId) perm
            if h : vOut.shape = n.outShape then
              pure (DVal.mk (α := α) n.outShape (h ▸ vOut.tensor))
            else
              throw <|
                s!"IR eval: node {i}: permute outShape mismatch: " ++
                  s!"computed={repr vOut.shape}, declared={repr n.outShape} ({n.summary})"
        | _ => throw s!"IR eval: node {i}: permute expects 1 parent ({n.summary})"
    | .detach =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape p)
        | _ => throw s!"IR eval: node {i}: detach expects 1 parent ({n.summary})"
    | .randUniform seed =>
        match n.parents with
        | [] =>
            let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
            let t : Tensor α n.outShape :=
              Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := n.outShape)
            pure (DVal.mk (α := α) n.outShape t)
        | _ => throw s!"IR eval: node {i}: rand_uniform expects 0 parents ({n.summary})"
    | .bernoulliMask seed =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            match pV.shape, pV.tensor with
            | .scalar, Tensor.scalar keepProb =>
                let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
                let t : Tensor α n.outShape :=
                  Runtime.Autograd.TorchLean.Random.mask (α := α) key keepProb (s := n.outShape)
                pure (DVal.mk (α := α) n.outShape t)
            | _, _ =>
                throw
                  s!"IR eval: node {i}: bernoulli_mask expects scalar keepProb parent ({n.summary})"
        | _ => throw s!"IR eval: node {i}: bernoulli_mask expects 1 parent ({n.summary})"
    | .add =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (getParent bId)
            pure (DVal.mk (α := α) n.outShape (Tensor.addSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: add expects 2 parents ({n.summary})"
    | .sub =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (getParent bId)
            pure (DVal.mk (α := α) n.outShape (Tensor.subSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: sub expects 2 parents ({n.summary})"
    | .mul_elem =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (getParent bId)
            pure (DVal.mk (α := α) n.outShape (Tensor.mulSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: mul_elem expects 2 parents ({n.summary})"
    | .abs =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Tensor.absSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: abs expects 1 parent ({n.summary})"
    | .sqrt =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Tensor.sqrtSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: sqrt expects 1 parent ({n.summary})"
    | .maxElem =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (getParent bId)
            pure (DVal.mk (α := α) n.outShape (Tensor.maxSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: max_elem expects 2 parents ({n.summary})"
    | .minElem =>
        match n.parents with
        | [aId, bId] =>
            let a ← expectShape (α := α) (expected := n.outShape) (getParent aId)
            let b ← expectShape (α := α) (expected := n.outShape) (getParent bId)
            pure (DVal.mk (α := α) n.outShape (Tensor.minSpec (α := α) a b))
        | _ => throw s!"IR eval: node {i}: min_elem expects 2 parents ({n.summary})"
    | .maxPool2d kH kW stride =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: max_pool2d requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: max_pool2d requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: max_pool2d requires stride ≠ 0 ({n.summary})"
                else
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                  let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                      (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (DVal.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: max_pool2d outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: max_pool2d expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: max_pool2d expects 1 parent ({n.summary})"
    | .maxPool2dPad kH kW stride padding =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: max_pool2d_pad requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: max_pool2d_pad requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: max_pool2d_pad requires stride ≠ 0 ({n.summary})"
                else
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape :=
                    Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
                  let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                      (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (DVal.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: max_pool2d_pad outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: max_pool2d_pad expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: max_pool2d_pad expects 1 parent ({n.summary})"
    | .avgPool2d kH kW stride =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d requires stride ≠ 0 ({n.summary})"
                else
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                  let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                      (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (DVal.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: avg_pool2d outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: avg_pool2d expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: avg_pool2d expects 1 parent ({n.summary})"
    | .avgPool2dPad kH kW stride padding =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            match pV.shape with
            | .dim inC (.dim inH (.dim inW .scalar)) =>
                if hkH : kH = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d_pad requires kH ≠ 0 ({n.summary})"
                else if hkW : kW = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d_pad requires kW ≠ 0 ({n.summary})"
                else if hs : stride = 0 then
                  throw s!"IR eval: node {i}: avg_pool2d_pad requires stride ≠ 0 ({n.summary})"
                else
                  let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                  let xCHW ← expectShape (α := α) (expected := sIn) pV
                  let expected : Shape :=
                    Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
                  let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
                  let y : Tensor α expected :=
                    Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                      (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
                      (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                  if h : expected = n.outShape then
                    pure (DVal.mk (α := α) n.outShape (h ▸ y))
                  else
                    throw <|
                      s!"IR eval: node {i}: avg_pool2d_pad outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"
            | _ =>
                throw s!"IR eval: node {i}: avg_pool2d_pad expects CHW parent shape ({n.summary})"
        | _ => throw s!"IR eval: node {i}: avg_pool2d_pad expects 1 parent ({n.summary})"
    | .broadcastTo s₁ s₂ =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := s₁) (getParent pId)
            match mkCanBroadcastTo? s₁ s₂ with
            | none => throw s!"IR eval: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
            | some cb =>
                let y := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb p
                pure (DVal.mk (α := α) s₂ y)
        | _ => throw s!"IR eval: node {i}: broadcastTo expects 1 parent ({n.summary})"
    | .reduceSum axis =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            let s := pV.shape
            let pT : Tensor α s := pV.tensor
            match mkValidAxis? (axis := axis) s with
            | none =>
                let msg :=
                  s!"IR eval: node {i}: reduce_sum invalid axis={axis}" ++
                    s!" for shape {repr s}"
                throw msg
            | some hAxis =>
                let hRed := Shape.proveReducibleAlong axis s hAxis.down
                let y := Tensor.reduceSum (α := α) (s := s) axis pT hRed
                pure (DVal.mk (α := α) (shapeAfterSum s axis) y)
        | _ => throw s!"IR eval: node {i}: reduce_sum expects 1 parent ({n.summary})"
    | .reduceMean axis =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            let s := pV.shape
            let pT : Tensor α s := pV.tensor
            match mkValidAxis? (axis := axis) s with
            | none =>
                let msg :=
                  s!"IR eval: node {i}: reduce_mean invalid axis={axis}" ++
                    s!" for shape {repr s}"
                throw msg
            | some hAxis =>
                let hRed := Shape.proveReducibleAlong axis s hAxis.down
                let y := Tensor.reduceMean (α := α) (s := s) axis pT hRed
                pure (DVal.mk (α := α) (shapeAfterSum s axis) y)
        | _ => throw s!"IR eval: node {i}: reduce_mean expects 1 parent ({n.summary})"
    | .sum =>
        match n.parents with
        | [pId] =>
            let p := getParent pId
            let s := p.shape
            let t : Tensor α s := p.tensor
            let v : α := Tensor.sumSpec (α := α) t
            pure (DVal.mk (α := α) .scalar (Tensor.scalar v))
        | _ => throw s!"IR eval: node {i}: sum expects 1 parent ({n.summary})"
    | .matmul =>
        match n.parents with
        | [aId, bId] =>
            let aV := getParent aId
            let bV := getParent bId
            match aV.shape, bV.shape with
            | Shape.dim m (Shape.dim n Shape.scalar), Shape.dim n' (Shape.dim p Shape.scalar) =>
                let aT ← expectShape (α := α) (expected := Shape.dim m (Shape.dim n Shape.scalar))
                  aV
                let bT ← expectShape (α := α) (expected := Shape.dim n' (Shape.dim p Shape.scalar))
                  bV
                if h : n = n' then
                  match h with
                  | rfl =>
                      let y := Spec.matMulSpec (α := α) (m := m) (n := n) (p := p) aT bT
                      pure (DVal.mk (α := α) (Shape.dim m (Shape.dim p Shape.scalar)) y)
                else
                  throw s!"IR eval: node {i}: matmul inner dims mismatch: {n} vs {n'}"
            | Shape.dim batch (Shape.dim m (Shape.dim n Shape.scalar)),
              Shape.dim batch' (Shape.dim n' (Shape.dim p Shape.scalar)) =>
                let aT ← expectShape (α := α)
                  (expected := Shape.dim batch (Shape.dim m (Shape.dim n Shape.scalar))) aV
                let bT ← expectShape (α := α)
                  (expected := Shape.dim batch' (Shape.dim n' (Shape.dim p Shape.scalar))) bV
                if hb : batch = batch' then
                  if hn : n = n' then
                    match hb, hn with
                    | rfl, rfl =>
                        let y := Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p :=
                          p) aT bT
                        pure (DVal.mk (α := α) (Shape.dim batch (Shape.dim m (Shape.dim p
                          Shape.scalar))) y)
                  else
                    throw s!"IR eval: node {i}: matmul inner dims mismatch: {n} vs {n'}"
                else
                  throw s!"IR eval: node {i}: matmul batch dims mismatch: {batch} vs {batch'}"
            | _, _ =>
                throw <|
                  s!"IR eval: node {i}: unsupported matmul shapes: {repr aV.shape} · " ++
                    s!"{repr bV.shape}"
        | _ => throw s!"IR eval: node {i}: matmul expects 2 parents ({n.summary})"
    | .linear =>
        match n.parents with
        | [pId] =>
            evalLinear (α := α) (payload := payload) (id := n.id) (x := getParent pId) (outShape :=
              n.outShape)
        | _ => throw s!"IR eval: node {i}: linear expects 1 parent ({n.summary})"
    | .conv2d .. =>
        match n.parents with
        | [pId] =>
            let y ← evalConv2D (α := α) (payload := payload) (id := n.id) (x := getParent pId)
            if y.shape != n.outShape then
              throw <|
                s!"IR eval: node {i}: conv2d outShape mismatch: computed={repr y.shape}, " ++
                  s!"declared={repr n.outShape}"
            pure y
        | _ => throw s!"IR eval: node {i}: conv2d expects 1 parent ({n.summary})"
    | .relu =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Activation.reluSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: relu expects 1 parent ({n.summary})"
    | .tanh =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Activation.tanhSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: tanh expects 1 parent ({n.summary})"
    | .sin =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Tensor.mapSpec (fun x => MathFunctions.sin x) p))
        | _ => throw s!"IR eval: node {i}: sin expects 1 parent ({n.summary})"
    | .cos =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Tensor.mapSpec (fun x => MathFunctions.cos x) p))
        | _ => throw s!"IR eval: node {i}: cos expects 1 parent ({n.summary})"
    | .sigmoid =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Activation.sigmoidSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: sigmoid expects 1 parent ({n.summary})"
    | .exp =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Tensor.expSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: exp expects 1 parent ({n.summary})"
    | .log =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            -- Domain discipline: align IR denotation with the compiled runtime backend.
            -- The raw `log` is treated as undefined on nonpositive inputs; both the compiler
            -- (`IRExec.buildFrom`) and this evaluator model that by producing a default value
            -- (in Lean, `panic!` reduces to `Inhabited.default`). Use `safeLogSpec`/`safeLogOp`
            -- in models that require epsilon protection.
            let t : Tensor α n.outShape :=
              if Tensor.allSpec (α := α) (s := n.outShape) (fun v => decide (0 < v)) p then
                Tensor.logSpec (α := α) p
              else
                panic!
                  "IR eval: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
            pure (DVal.mk (α := α) n.outShape t)
        | _ => throw s!"IR eval: node {i}: log expects 1 parent ({n.summary})"
    | .inv =>
        match n.parents with
        | [pId] =>
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            pure (DVal.mk (α := α) n.outShape (Tensor.invSpec (α := α) p))
        | _ => throw s!"IR eval: node {i}: inv expects 1 parent ({n.summary})"
    | .softmax axis => do
        match n.parents with
        | [pId] =>
            match OpContracts.checkAxisValid axis n.outShape with
            | .ok () => pure ()
            | .error msg =>
                throw s!"IR eval: node {i}: softmax: {msg} ({n.summary})"
            let p ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            -- Our spec primitive is last-axis softmax; for non-last axes we permute the chosen axis
            -- to the last position, apply softmax, then permute back.
            if axis + 1 = Shape.rank n.outShape then
              pure (DVal.mk (α := α) n.outShape (Activation.softmaxSpec (α := α) p))
            else
              let permToLast ←
                match OpContracts.permMoveAxisToLast axis n.outShape with
                | .ok perm => pure perm
                | .error msg => throw s!"IR eval: node {i}: softmax: {msg} ({n.summary})"
              let permBack ←
                match OpContracts.inversePerm permToLast with
                | .ok perm => pure perm
                | .error msg => throw s!"IR eval: node {i}: softmax: {msg} ({n.summary})"
              let x0 : DVal α := DVal.mk (α := α) n.outShape p
              let xLast ←
                match permuteDVal (α := α) x0 permToLast with
                | .ok v => pure v
                | .error msg => throw s!"IR eval: node {i}: softmax: {msg} ({n.summary})"
              let yLast : DVal α :=
                match xLast with
                | ⟨sLast, tLast⟩ =>
                    ⟨sLast, Activation.softmaxSpec (α := α) (s := sLast) tLast⟩
              let y0 ←
                match permuteDVal (α := α) yLast permBack with
                | .ok v => pure v
                | .error msg => throw s!"IR eval: node {i}: softmax: {msg} ({n.summary})"
              let y ← expectShape (α := α) (expected := n.outShape) y0
              pure (DVal.mk (α := α) n.outShape y)
        | _ =>
            throw s!"IR eval: node {i}: softmax expects 1 parent ({n.summary})"
    | .layernorm axis =>
        match n.parents with
        | [pId] => do
            let x ← expectShape (α := α) (expected := n.outShape) (getParent pId)
            let (seqLen, embedDim) ←
              match OpContracts.layerNorm2DParams axis n.outShape with
              | .ok p => pure p
              | .error msg => throw s!"IR eval: node {i}: layernorm: {msg} ({n.summary})"
            let view2D : Shape := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
            if hNumel : Shape.size n.outShape = Shape.size view2D then
              let x2D : Tensor α view2D :=
                Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ := view2D) x hNumel
              let y2D ← layernormPure (α := α) (seqLen := seqLen) (embedDim := embedDim) x2D
              let y : Tensor α n.outShape :=
                Tensor.reshapeSpec (α := α) (s₁ := view2D) (s₂ := n.outShape) y2D hNumel.symm
              pure (DVal.mk (α := α) n.outShape y)
            else
              throw <|
                s!"IR eval: node {i}: layernorm internal error: bad reshape sizes " ++
                  s!"({Shape.size n.outShape} vs {Shape.size view2D}) ({n.summary})"
        | _ =>
            throw s!"IR eval: node {i}: layernorm expects 1 parent ({n.summary})"
    | .reshape inS outS =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            let pT ← expectShape (α := α) (expected := inS) pV
            if h : Shape.size inS = Shape.size outS then
              let y := Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) pT h
              pure (DVal.mk (α := α) outS y)
            else
              throw
                s!"IR eval: node {i}: reshape numel mismatch: {Shape.size inS} vs {Shape.size outS}"
        | _ => throw s!"IR eval: node {i}: reshape expects 1 parent ({n.summary})"
    | .flatten s =>
        match n.parents with
        | [pId] =>
            let pV := getParent pId
            let pT ← expectShape (α := α) (expected := s) pV
            let y := Tensor.flattenSpec (α := α) (s := s) pT
            pure (DVal.mk (α := α) (.dim (Shape.size s) .scalar) y)
        | _ => throw s!"IR eval: node {i}: flatten expects 1 parent ({n.summary})"
    | .concat axis => do
        let parents := n.parents.map getParent
        let expected ←
          match OpContracts.inferConcatOutShape axis (parents.map (fun pv => pv.shape)) with
          | .ok s => pure s
          | .error msg => throw s!"IR eval: node {i}: {msg} ({n.summary})"
        if expected != n.outShape then
          throw <|
            s!"IR eval: node {i}: concat outShape mismatch: " ++
              s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"

        -- Interpret `concat axis` by permuting `axis` to the front (axis 0), concatenating along
        -- axis 0
        -- (using the spec primitive `concat_dim0_spec`), then permuting back.
        let permFront ←
          match OpContracts.permMoveAxisToFront axis n.outShape with
          | .ok perm => pure perm
          | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
        let permBack ←
          match OpContracts.inversePerm permFront with
          | .ok perm => pure perm
          | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
        let outPermShape ←
          match Spec.Shape.permute? n.outShape permFront with
          | some s => pure s
          | none =>
              throw <|
                s!"IR eval: node {i}: concat: internal error (invalid permutation for " ++
                  s!"outShape) ({n.summary})"
        let parentsPerm : List (DVal α) ←
          parents.mapM (fun pv => do
            match permuteDVal (α := α) pv permFront with
            | .ok v => pure v
            | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})")
        match outPermShape with
        | Shape.dim nOut rest =>
            let toSigma (pv : DVal α) : Except String (Sigma fun n => Tensor α (Shape.dim n rest))
              := do
              match pv.shape, pv.tensor with
              | Shape.dim nP restP, t =>
                  if hRest : restP = rest then
                    let t' : Tensor α (Shape.dim nP rest) := by
                      simpa [hRest] using t
                    pure ⟨nP, t'⟩
                  else
                    throw <|
                      s!"IR eval: node {i}: concat: permuted tail mismatch: {repr restP} vs " ++
                        s!"{repr rest}"
              | _, _ =>
                  throw s!"IR eval: node {i}: concat expects rank≥1 parents, got {repr pv.shape}"
            let sigs ← parentsPerm.mapM toSigma
            match sigs with
            | [] =>
                throw s!"IR eval: node {i}: concat internal error"
            | s0 :: srest =>
                let outSigma :=
                  srest.foldl
                    (fun acc nxt =>
                      match acc, nxt with
                      | ⟨n1, t1⟩, ⟨n2, t2⟩ =>
                          ⟨n1 + n2, Tensor.concatDim0Spec (α := α) (n := n1) (m := n2) (s := rest)
                            t1 t2⟩)
                    s0
                match outSigma with
                | ⟨nSum, tSum⟩ =>
                    if h : nSum = nOut then
                      let yPerm : Tensor α (Shape.dim nOut rest) := by
                        simpa [h] using tSum
                      let outPerm : DVal α := DVal.mk (α := α) (Shape.dim nOut rest) yPerm
                      let out0 ←
                        match permuteDVal (α := α) outPerm permBack with
                        | .ok v => pure v
                        | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
                      let y ← expectShape (α := α) (expected := n.outShape) out0
                      pure (DVal.mk (α := α) n.outShape y)
                    else
                      throw <|
                        s!"IR eval: node {i}: concat out dim mismatch: declared {nOut}, " ++
                          s!"computed {nSum}"
        | _ =>
            throw s!"IR eval: node {i}: concat expects rank≥1 outShape, got {repr n.outShape}"
      | .swap_first_two =>
          match n.parents with
          | [pId] =>
              match n.outShape with
              | Shape.dim nDim (Shape.dim m rest) =>
                  let p ←
                    expectShape (α := α) (expected := Shape.dim m (Shape.dim nDim rest)) (getParent
                      pId)
                  let y := Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest) p
                  pure (DVal.mk (α := α) (Shape.dim nDim (Shape.dim m rest)) y)
              | _ =>
                  throw s!"IR eval: node {i}: swap_first_two expects rank≥2 outShape ({n.summary})"
          | _ => throw s!"IR eval: node {i}: swap_first_two expects 1 parent ({n.summary})"
      | .transpose3dLastTwo =>
          match n.parents with
          | [pId] =>
              match n.outShape with
              | Shape.dim a (Shape.dim c (Shape.dim b Shape.scalar)) =>
                  let p ←
                    expectShape (α := α) (expected := Shape.dim a (Shape.dim b (Shape.dim c
                      Shape.scalar)))
                      (getParent pId)
                  let y := Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) p
                  pure (DVal.mk (α := α) (Shape.dim a (Shape.dim c (Shape.dim b Shape.scalar))) y)
              | _ =>
                  throw <|
                    s!"IR eval: node {i}: transpose3d_last_two expects rank=3 with scalar " ++
                      s!"base outShape ({n.summary})"
          | _ => throw s!"IR eval: node {i}: transpose3d_last_two expects 1 parent ({n.summary})"
      | .mseLoss =>
          match n.parents with
          | [yId, tId] =>
              mseLossDVal (α := α) i (getParent yId) (getParent tId)
          | _ => throw s!"IR eval: node {i}: mse_loss expects 2 parents ({n.summary})"
  if h : v.shape = n.outShape then
    -- Normalize the returned value’s shape tag to the node’s declared `outShape`.
    pure (DVal.mk (α := α) n.outShape (h ▸ v.tensor))
  else
    throw <|
      s!"IR eval: node {i}: produced shape mismatch: produced={repr v.shape}, " ++
        s!"declared={repr n.outShape} ({n.summary})"

/--
Evaluate nodes `i, i+1, ...` given already computed prefix values `vals`.

This is written as a structurally recursive function so it is easy to reason about in proofs
(evaluation is “a simple loop over node ids”).
-/
def denoteAllFrom
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : DVal α) (i : Nat) (vals : Array (DVal α)) :
    Except String (Array (DVal α)) := do
  if h : i < g.nodes.size then
    let v ← evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i := i)
    denoteAllFrom (α := α) (g := g) (payload := payload) (input := input) (i := i + 1) (vals :=
      vals.push v)
  else
    pure vals
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

/--
Evaluate a graph to a table of node values.

This returns an array `vals` of length `g.size` where `vals[i]` is the value of node `i`.

We do a structural well-formedness check once up front (ids/arity/topology). For compiler-produced
graphs, the boolean `Graph.wellFormed` check is a fast path; if it fails we fall back to the
exception-producing `Graph.checkWellFormed` so callers get a readable error message.

The evaluator is total in the sense that it always returns either:
- `.ok vals` (all nodes evaluated successfully), or
- `.error msg` describing the first failure (malformed IR, missing payload, or a local shape error).
-/
def denoteAll
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : DVal α) : Except String (Array (DVal α)) := do
  -- Fast path: compiler-produced graphs typically satisfy the boolean `wellFormed` discipline.
  if g.wellFormed then
    pure ()
  else
    g.checkWellFormed
  denoteAllFrom (α := α) (g := g) (payload := payload) (input := input) (i := 0)
    (vals := #[])

/-! ## Scoped notation -/

/--
Scoped notation for evaluating a graph to all node values.

Use with:

```lean
open scoped IR
g⟦payload, input⟧
```
-/
scoped[IR] notation g "⟦" payload ", " input "⟧" =>
  _root_.NN.IR.Graph.denoteAll g payload input

/-- ASCII alternative to `g⟦payload, input⟧`. -/
scoped[IR] notation g "[[" payload ", " input "]]" =>
  _root_.NN.IR.Graph.denoteAll g payload input

/-- Evaluate the graph and return the value at `outputId`. -/
def denote
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : DVal α) (outputId : Nat) : Except String (DVal α) :=
      do
  let vals ← denoteAll (α := α) (g := g) (payload := payload) (input := input)
  match vals[outputId]? with
  | none => throw s!"IR eval: outputId out of bounds: {outputId}"
  | some v => pure v

end Graph

end NN.IR
