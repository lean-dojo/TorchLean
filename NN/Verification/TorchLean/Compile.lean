/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.MLTheory.CROWN.Graph
public import NN.Runtime.Autograd.TorchLean.Backend

/-!
# Compile

TorchLean → verifier bridge.

This module lets you take a TorchLean `Program` (written once over the `TorchLean.Ops` interface)
and compile it into the op-tagged IR (`NN.IR.Graph`) plus a CROWN/LiRPA-style `ParamStore`.

Why this exists:
- TorchLean programs can be executed (eager/compiled) for training, but the same computation graph
  is also useful as a *verification artifact*.
- By compiling to an explicit DAG IR, we can run bound propagation (IBP, CROWN/DeepPoly variants)
  and certificate checkers on the exact same model definition.

Scope:
- targets the forward-only TorchLean fragment used by bound-propagation checkers;
- supports a curated operator set (arithmetic, shape ops, common nonlinearities, pooling, `linear`,
  `conv2d`) plus a few composite ops lowered to IR subgraphs (e.g. `layer_norm`,
  `multi_head_attention`);
- ops outside the verifier fragment throw with an explicit error (notably: general Nat-indexed gather/scatter, and
  training-style BatchNorm).

Related tooling:
- PyTorch interop lives under `NN.Runtime.PyTorch.Import.*` and is used by many examples to import
  weights or certificates produced by Python scripts.

References (informal):
- IBP: Gowal et al. (2018).
- CROWN / DeepPoly-style linear relaxations: Zhang et al. (2018).
- LiRPA unification viewpoint: Xu et al. (2020).
-/

@[expose] public section


namespace NN.Verification.TorchLean

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

/-! ## IR builder backend -/

/--
Reference produced while compiling a TorchLean program.

A value is either already materialized as an IR node, or it is still a compile-time tensor constant
that can be inserted into the verifier `ParamStore` if a later operation needs a node parent.
-/
inductive Ref (α : Type) : Shape → Type where
  | node  {s : Shape} (id : Nat) : Ref α s
  | const {s : Shape} (t : Tensor α s) : Ref α s

/-- Mutable builder state for translating a TorchLean program into verifier IR. -/
structure BuildState (α : Type) [Context α] where
  /-- IR nodes emitted so far, in construction/topological order. -/
  nodes : Array Node := #[]
  /-- Parameter payload accumulated for constant tensors and layer weights. -/
  ps    : NN.MLTheory.CROWN.Graph.ParamStore α := {}

/-- Builder monad used by the TorchLean→IR compiler. -/
abbrev BuildM (α : Type) [Context α] : Type → Type :=
  StateT (BuildState α) (Except String)

/-- Raise a compiler error inside `BuildM`. -/
def fail {α : Type} [Context α] {β : Type} (msg : String) : BuildM α β :=
  throw msg

/-- Run a shared IR shape contract and surface its error from the compiler path. -/
def requireContract {α : Type} [Context α] {β : Type} (r : Except String β) : BuildM α Unit := do
  match r with
  | .ok _ => pure ()
  | .error msg => fail (α := α) msg

/-- Append a freshly constructed IR node to the builder state. -/
def pushNode {α : Type} [Context α] (n : Node) : BuildM α Unit := do
  modify fun st => { st with nodes := st.nodes.push n }

/-- Return the next node identifier, which is the current node-array size. -/
def freshId {α : Type} [Context α] : BuildM α Nat := do
  pure (←get).nodes.size

/--
Ensure a `Ref` is represented by an IR node.

Compile-time constants are materialized as `.const` nodes and recorded in the verifier
`ParamStore`; existing graph nodes are returned unchanged.
-/
def ensureNode {α : Type} [Context α]
    {s : Shape} (r : Ref α s) : BuildM α Nat := do
  match r with
  | .node id => pure id
  | .const t =>
      let id ← freshId (α := α)
      let flat := Tensor.flattenSpec (α := α) t
      let fv : NN.MLTheory.CROWN.Graph.FlatVec α := { n := Shape.size s, v := flat }
      let node : Node := { id := id, parents := [], kind := .const s, outShape := s }
      modify fun st => { st with ps := { st.ps with constVals := st.ps.constVals.insert id fv } }
      pushNode (α := α) node
      pure id

/-- Emit a unary IR operation with one parent node. -/
def emitUnary {α : Type} [Context α]
    {s t : Shape} (kind : OpKind) (x : Ref α s) (outShape : Shape := t) : BuildM α (Ref α t) := do
  let pid ← ensureNode (α := α) x
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := [pid], kind := kind, outShape := outShape }
  pushNode (α := α) node
  pure (.node id)

/-- Emit a binary IR operation whose operands have the same shape. -/
def emitBinary {α : Type} [Context α]
    {s : Shape} (kind : OpKind) (a b : Ref α s) : BuildM α (Ref α s) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := [pa, pb], kind := kind, outShape := s }
  pushNode (α := α) node
  pure (.node id)

/-- Emit a matrix-multiplication IR node. -/
def emitMatmul {α : Type} [Context α]
    {sA sB sOut : Shape} (a : Ref α sA) (b : Ref α sB) (outShape : Shape := sOut) :
    BuildM α (Ref α sOut) := do
  let pa ← ensureNode (α := α) a
  let pb ← ensureNode (α := α) b
  let id ← freshId (α := α)
  let node : Node := { id := id, parents := [pa, pb], kind := .matmul, outShape := outShape }
  pushNode (α := α) node
  pure (.node id)

/-- Emit the designated verifier input node.  We keep this at id `0` for bound seeding. -/
def emitInput {α : Type} [Context α] {s : Shape} : BuildM α (Ref α s) := do
  let id ← freshId (α := α)
  if id ≠ 0 then
    -- keep the designated input id stable for verifier seeding
    fail (α := α) "TorchLean IR compile: internal error (input node must be id 0)"
  let node : Node := { id := id, parents := [], kind := .input, outShape := s }
  pushNode (α := α) node
  pure (.node id)

/-- Read a compile-time constant tensor, failing if the value already depends on graph input. -/
def getConst {α : Type} [Context α] {s : Shape} (r : Ref α s) : BuildM α (Tensor α s) :=
  match r with
  | .const t => pure t
  | .node _ => fail (α := α)
    "TorchLean IR compile: expected a compile-time constant tensor (got a graph node)"

private theorem vector2_toList {α : Type} (v : Vector α 2) :
    v.toList = [v.get ⟨0, by decide⟩, v.get ⟨1, by decide⟩] := by
  -- Reduce to the underlying array.
  simp [Vector.toList, Vector.get]
  apply List.ext_getElem
  · simp
  · intro i hi
    have hi2 : i < 2 := by
      simpa using hi
    cases i with
    | zero =>
        simp [List.getElem_cons, Array.getElem_toList]
    | succ i =>
        cases i with
        | zero =>
            simp [List.getElem_cons, Array.getElem_toList]
        | succ i =>
            have : 2 ≤ Nat.succ (Nat.succ i) :=
              Nat.succ_le_succ (Nat.succ_le_succ (Nat.zero_le i))
            exact (False.elim ((Nat.not_lt_of_ge this) hi2))

instance {α : Type} [Context α] [DecidableEq Shape] :
    Runtime.Autograd.Torch.Ops (m := BuildM α) (α := α) where
  Ref := Ref α

  const := fun {_s} t => pure (.const t)

  add := fun {_s} a b => emitBinary (α := α) (kind := .add) (a := a) (b := b)
  sub := fun {_s} a b => emitBinary (α := α) (kind := .sub) (a := a) (b := b)
  mul := fun {_s} a b => emitBinary (α := α) (kind := .mul_elem) (a := a) (b := b)

  scale := fun {s} x c => do
    -- IR has no dedicated `scale`; encode as elementwise mul with a constant tensor.
    let cT : Tensor α s := Spec.fill c s
    emitBinary (α := α) (kind := .mul_elem) (a := x) (b := .const cT)

  abs := fun {s} _x =>
    emitUnary (α := α) (kind := .abs) (x := _x) (t := s) (outShape := s)
  sqrt := fun {s} x =>
    emitUnary (α := α) (kind := .sqrt) (x := x) (t := s) (outShape := s)
  clamp := fun {s} x lo hi => do
    -- Lower to `min(max(x, lo), hi)` using const-filled tensors.
    let loT : Tensor α s := Spec.fill (α := α) lo s
    let hiT : Tensor α s := Spec.fill (α := α) hi s
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

  transpose2d := fun {mDim nDim} x => do
    let out : Shape := .dim nDim (.dim mDim .scalar)
    emitUnary (α := α) (kind := .swap_first_two) (x := x) (t := out) (outShape := out)

  transpose3dFirstToLast := fun {a b c} x => do
    -- (a,b,c) -> (b,c,a) = swap_first_two then transpose3d_last_two
    let s1 : Shape := .dim b (.dim a (.dim c .scalar))
    let tmp : Ref α s1 ←
      emitUnary (α := α) (kind := .swap_first_two) (x := x) (t := s1) (outShape := s1)
    let out : Shape := .dim b (.dim c (.dim a .scalar))
    emitUnary (α := α) (kind := .transpose3dLastTwo) (x := tmp) (t := out) (outShape := out)
  transpose3dLastToFirst := fun {a b c} x => do
    -- (a,b,c) -> (c,a,b) = transpose3d_last_two then swap_first_two
    let s1 : Shape := .dim a (.dim c (.dim b .scalar))
    let tmp : Ref α s1 ←
      emitUnary (α := α) (kind := .transpose3dLastTwo) (x := x) (t := s1) (outShape := s1)
    let out : Shape := .dim c (.dim a (.dim b .scalar))
    emitUnary (α := α) (kind := .swap_first_two) (x := tmp) (t := out) (outShape := out)
  transpose3dLastTwo := fun {a b c} x => do
    let out : Shape := .dim a (.dim c (.dim b .scalar))
    emitUnary (α := α) (kind := .transpose3dLastTwo) (x := x) (t := out) (outShape := out)

  swapAdjacentAtDepth := fun {s} depth x => do
    -- We support only the cases already present in the verifier IR:
    -- - depth=0 (swap first two dims): `.swap_first_two`
    -- - depth=1 on rank-3 scalar-base tensors: `.transpose3d_last_two`
    -- In all other cases, if the swap is a no-op on shape, return `x` unchanged; otherwise fail.
    match depth, s with
    | 0, .dim m (.dim n rest) =>
        let out : Shape := .dim n (.dim m rest)
        emitUnary (α := α) (kind := .swap_first_two) (x := x) (t := out) (outShape := out)
    | 1, .dim a (.dim b (.dim c .scalar)) =>
        let out : Shape := .dim a (.dim c (.dim b .scalar))
        emitUnary (α := α) (kind := .transpose3dLastTwo) (x := x) (t := out) (outShape := out)
    | depth', s' =>
        if h : s'.swapAdjacentAtDepth depth' = s' then
          pure (Eq.mp (congrArg (fun sh => Ref α sh) (Eq.symm h)) x)
        else
          -- General adjacent swap is representable as a `permute` op.
          let r := Shape.rank s'
          if depth' + 1 < r then
            let perm : List Nat :=
              (List.range r).map (fun i =>
                if i = depth' then depth' + 1 else if i = depth' + 1 then depth' else i)
            let out : Shape := s'.swapAdjacentAtDepth depth'
            emitUnary (α := α) (kind := .permute perm) (x := x) (t := out) (outShape := out)
          else
            -- This should be unreachable because `swapAdjacentAtDepth` changed the shape, but keep
            -- it defensive.
            fail (α := α)
              s!"TorchLean→IR: swapAdjacentAtDepth (depth={depth}) invalid for shape {repr s'}"

  reduceSum := fun {s} axis _valid _wf x => do
    let out : Shape := Spec.Tensor.shapeAfterSum s axis
    emitUnary (α := α) (kind := .reduceSum axis) (x := x) (t := out) (outShape := out)
  reduceMean := fun {s} axis _valid _wf x => do
    let out : Shape := Spec.Tensor.shapeAfterSum s axis
    emitUnary (α := α) (kind := .reduceMean axis) (x := x) (t := out) (outShape := out)

  gatherScalar := fun {n} x i => do
    -- Lower `gather_scalar` to IR ops:
    --   (1×n) one-hot row  @  reshape(x, n×1)   → (1×1) → scalar
    let sel : Tensor α (.dim 1 (.dim n .scalar)) :=
      Tensor.dim (fun _ =>
        Tensor.dim (fun j => Tensor.scalar (if j = i then (1 : α) else (0 : α))))
    let xCol : Ref α (.dim n (.dim 1 .scalar)) ←
      emitUnary (α := α)
        (kind := .reshape (.dim n .scalar) (.dim n (.dim 1 .scalar)))
        (x := x) (t := .dim n (.dim 1 .scalar)) (outShape := .dim n (.dim 1 .scalar))
    let y11 : Ref α (.dim 1 (.dim 1 .scalar)) ←
      emitMatmul (α := α) (a := .const sel) (b := xCol)
        (sOut := .dim 1 (.dim 1 .scalar)) (outShape := .dim 1 (.dim 1 .scalar))
    emitUnary (α := α)
      (kind := .reshape (.dim 1 (.dim 1 .scalar)) .scalar)
      (x := y11) (t := Shape.scalar) (outShape := Shape.scalar)

  gatherRow := fun {rows cols} x i => do
    -- Lower `gather_row` to IR ops:
    --   (1×rows) one-hot row  @  x(rows×cols)  → (1×cols) → (cols)
    let sel : Tensor α (.dim 1 (.dim rows .scalar)) :=
      Tensor.dim (fun _ =>
        Tensor.dim (fun j => Tensor.scalar (if j = i then (1 : α) else (0 : α))))
    let y1c : Ref α (.dim 1 (.dim cols .scalar)) ←
      emitMatmul (α := α) (a := .const sel) (b := x)
        (sOut := .dim 1 (.dim cols .scalar)) (outShape := .dim 1 (.dim cols .scalar))
    emitUnary (α := α)
      (kind := .reshape (.dim 1 (.dim cols .scalar)) (.dim cols .scalar))
      (x := y1c) (t := .dim cols .scalar) (outShape := .dim cols .scalar)

  gatherScalarNat := fun {_n} _x _i =>
    fail (α := α) "TorchLean→IR: gather is outside the verifier IR fragment"
  gatherVecNat := fun {_n _k} _x _idx =>
    fail (α := α) "TorchLean→IR: gather is outside the verifier IR fragment"
  gatherRowsNat := fun {_rows _cols _k} _x _idx =>
    fail (α := α) "TorchLean→IR: gather is outside the verifier IR fragment"
  floatVecToNatTensor := fun {_k} _x =>
    fail (α := α) "TorchLean→IR: float_vec_to_nat_tensor is outside the verifier IR fragment"

  scatterAddVec := fun {_n} _x _val _i =>
    fail (α := α) "TorchLean→IR: scatter is outside the verifier IR fragment"
  scatterAddRow := fun {_rows _cols} _x _row _i =>
    fail (α := α) "TorchLean→IR: scatter is outside the verifier IR fragment"

  matmul := fun {mDim _nDim pDim} a b => do
    let out : Shape := .dim mDim (.dim pDim .scalar)
    emitMatmul (α := α) (a := a) (b := b) (sOut := out) (outShape := out)

  bmm := fun {batch mDim _nDim pDim} a b => do
    let out : Shape := .dim batch (.dim mDim (.dim pDim .scalar))
    emitMatmul (α := α) (a := a) (b := b) (sOut := out) (outShape := out)

  concatVectors := fun {nDim mDim} a b => do
    let pa ← ensureNode (α := α) a
    let pb ← ensureNode (α := α) b
    let id ← freshId (α := α)
    let out : Shape := .dim (nDim + mDim) .scalar
    let node : Node := { id := id, parents := [pa, pb], kind := .concat 0, outShape := out }
    pushNode (α := α) node
    pure (.node id)
  concatDim0 := fun {nDim mDim} {s} a b => do
    let pa ← ensureNode (α := α) a
    let pb ← ensureNode (α := α) b
    let id ← freshId (α := α)
    let out : Shape := .dim (nDim + mDim) s
    let node : Node := { id := id, parents := [pa, pb], kind := .concat 0, outShape := out }
    pushNode (α := α) node
    pure (.node id)

  sliceRange0 := fun {nDim} {s} start len _h x => do
    -- Lower leading-axis slicing to the existing linear verifier fragment:
    --
    --   x : (nDim, s)
    --   reshape x                         : (nDim, block)
    --   oneHot[start:start+len] @ reshape : (len, block)
    --   reshape                            : (len, s)
    --
    -- This is exact for a contiguous slice along dimension 0. It avoids adding a separate IR
    -- primitive while still giving IBP/CROWN the same affine operation they already understand.
    let block : Nat := Shape.size s
    let xMat : Ref α (.dim nDim (.dim block .scalar)) ←
      emitUnary (α := α)
        (kind := .reshape (.dim nDim s) (.dim nDim (.dim block .scalar)))
        (x := x) (t := .dim nDim (.dim block .scalar))
        (outShape := .dim nDim (.dim block .scalar))
    let selector : Tensor α (.dim len (.dim nDim .scalar)) :=
      Tensor.dim (fun row =>
        Tensor.dim (fun col =>
          Tensor.scalar (if col.val = start + row.val then (1 : α) else (0 : α))))
    let yMat : Ref α (.dim len (.dim block .scalar)) ←
      emitMatmul (α := α) (a := .const selector) (b := xMat)
        (sOut := .dim len (.dim block .scalar))
        (outShape := .dim len (.dim block .scalar))
    have hsize :
        Shape.size (.dim len (.dim block .scalar)) = Shape.size (.dim len s) := by
      change len * (Shape.size s * 1) = len * Shape.size s
      rw [Nat.mul_one]
    emitUnary (α := α)
      (kind := .reshape (.dim len (.dim block .scalar)) (.dim len s))
      (x := yMat) (t := .dim len s) (outShape := .dim len s)

  -- ---------------------------------------------------------------------------
  -- ND pooling/conv wrappers (verifier IR supports CHW 2D ops)
  -- ---------------------------------------------------------------------------

  maxPool := fun {d C} {inSpatial kernel stride padding} {_hKernel} x => do
    -- The verifier IR only supports CHW max_pool2d(_pad) with symmetric stride/padding.
    match d with
    | 2 =>
        let kH : Nat := kernel.get ⟨0, by decide⟩
        let kW : Nat := kernel.get ⟨1, by decide⟩
        let sH : Nat := stride.get ⟨0, by decide⟩
        let sW : Nat := stride.get ⟨1, by decide⟩
        let pH : Nat := padding.get ⟨0, by decide⟩
        let pW : Nat := padding.get ⟨1, by decide⟩
        let inH : Nat := inSpatial.get ⟨0, by decide⟩
        let inW : Nat := inSpatial.get ⟨1, by decide⟩

        -- Cast input to explicit CHW so the verifier IR checker can pattern-match on it.
        have hinSpatial : inSpatial.toList = [inH, inW] := by
          simpa [inH, inW] using (vector2_toList (v := inSpatial))
        have hx :
            Shape.ofList (C :: inSpatial.toList) = .dim C (.dim inH (.dim inW .scalar)) := by
          simp [Shape.ofList, hinSpatial]
        let xCHW : Ref α (.dim C (.dim inH (.dim inW .scalar))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hx) x

        if hs : sH = sW then
          if hp : pH = pW then
            requireContract (α := α) <|
              NN.IR.OpContracts.inferPool2dCHWOutShapePad "max_pool2d_pad" kH kW sH pH
                (.dim C (.dim inH (.dim inW .scalar)))
            let xId ← ensureNode (α := α) xCHW
            let id ← freshId (α := α)
            let outShape : Shape := Spec.pool2dMultiOutShapePad C inH inW kH kW sH pH
            let node : Node :=
              { id := id
                parents := [xId]
                kind := .maxPool2dPad kH kW sH pH
                outShape := outShape }
            pushNode (α := α) node
            -- Cast to the generic output shape expected by `Ops.max_pool`.
            have hout :
                outShape =
                  Shape.ofList
                    (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList) := by
              have houtList :
                  (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList =
                    [ (Spec.poolOutSpatialPad inSpatial kernel stride padding).get ⟨0, by decide⟩
                    , (Spec.poolOutSpatialPad inSpatial kernel stride padding).get ⟨1, by decide⟩
                    ] := vector2_toList (v := Spec.poolOutSpatialPad inSpatial kernel stride padding)
              -- Expose the two output axes, then reduce the `Vector.ofFn` gets.
              rw [houtList]
              simp [Shape.ofList]
              simp [Spec.poolOutSpatialPad, Vector.ofFn, Vector.get]
              have hs' : stride[1] = stride[0] := by
                change stride.get ⟨1, by decide⟩ = stride.get ⟨0, by decide⟩
                exact hs.symm
              have hp' : padding[1] = padding[0] := by
                change padding.get ⟨1, by decide⟩ = padding.get ⟨0, by decide⟩
                exact hp.symm
              simp [hs', hp']
              simp [outShape, Spec.pool2dMultiOutShapePad, inH, inW, kH, kW, sH, pH]
              simp [Vector.get]
            pure (Eq.mp (congrArg (fun sh => Ref α sh) hout) (.node id))
          else
            fail (α := α) "TorchLean→IR: max_pool: verifier IR requires uniform padding"
        else
          fail (α := α) "TorchLean→IR: max_pool: verifier IR requires uniform stride"
    | _ =>
          fail (α := α) "TorchLean→IR: max_pool: verifier IR accepts d=2"

  avgPool := fun {d C} {inSpatial kernel stride padding} _hKernel x => do
    -- The verifier IR only supports CHW avg_pool2d(_pad) with symmetric stride/padding.
    match d with
    | 2 =>
        let kH : Nat := kernel.get ⟨0, by decide⟩
        let kW : Nat := kernel.get ⟨1, by decide⟩
        let sH : Nat := stride.get ⟨0, by decide⟩
        let sW : Nat := stride.get ⟨1, by decide⟩
        let pH : Nat := padding.get ⟨0, by decide⟩
        let pW : Nat := padding.get ⟨1, by decide⟩
        let inH : Nat := inSpatial.get ⟨0, by decide⟩
        let inW : Nat := inSpatial.get ⟨1, by decide⟩

        -- Cast input to explicit CHW so the verifier IR checker can pattern-match on it.
        have hinSpatial : inSpatial.toList = [inH, inW] := by
          simpa [inH, inW] using (vector2_toList (v := inSpatial))
        have hx :
            Shape.ofList (C :: inSpatial.toList) = .dim C (.dim inH (.dim inW .scalar)) := by
          simp [Shape.ofList, hinSpatial]
        let xCHW : Ref α (.dim C (.dim inH (.dim inW .scalar))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hx) x

        if hs : sH = sW then
          if hp : pH = pW then
            requireContract (α := α) <|
              NN.IR.OpContracts.inferPool2dCHWOutShapePad "avg_pool2d_pad" kH kW sH pH
                (.dim C (.dim inH (.dim inW .scalar)))
            let xId ← ensureNode (α := α) xCHW
            let id ← freshId (α := α)
            let outShape : Shape := Spec.pool2dMultiOutShapePad C inH inW kH kW sH pH
            let node : Node :=
              { id := id
                parents := [xId]
                kind := .avgPool2dPad kH kW sH pH
                outShape := outShape }
            pushNode (α := α) node
            -- Cast to the generic output shape expected by `Ops.avg_pool`.
            have hout :
                outShape =
                  Shape.ofList
                    (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList) := by
              have houtList :
                  (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList =
                    [ (Spec.poolOutSpatialPad inSpatial kernel stride padding).get ⟨0, by decide⟩
                    , (Spec.poolOutSpatialPad inSpatial kernel stride padding).get ⟨1, by decide⟩
                    ] := vector2_toList (v := Spec.poolOutSpatialPad inSpatial kernel stride padding)
              rw [houtList]
              simp [Shape.ofList]
              simp [Spec.poolOutSpatialPad, Vector.ofFn, Vector.get]
              have hs' : stride[1] = stride[0] := by
                change stride.get ⟨1, by decide⟩ = stride.get ⟨0, by decide⟩
                exact hs.symm
              have hp' : padding[1] = padding[0] := by
                change padding.get ⟨1, by decide⟩ = padding.get ⟨0, by decide⟩
                exact hp.symm
              simp [hs', hp']
              simp [outShape, Spec.pool2dMultiOutShapePad, inH, inW, kH, kW, sH, pH]
              simp [Vector.get]
            pure (Eq.mp (congrArg (fun sh => Ref α sh) hout) (.node id))
          else
            fail (α := α) "TorchLean→IR: avg_pool: verifier IR requires uniform padding"
        else
          fail (α := α) "TorchLean→IR: avg_pool: verifier IR requires uniform stride"
    | _ =>
        fail (α := α) "TorchLean→IR: avg_pool: only d=2 is supported by the verifier IR right now"

  smoothMaxPool := fun {_d _C} {_inSpatial _kernel _stride _padding} {_hKernel} _x _temp =>
    fail (α := α) "TorchLean→IR: smooth_max_pool is outside the verifier IR fragment"

  maxPool2d := fun {kH kW inH inW inC stride} {_h1 : kH ≠ 0} {_h2 : kW ≠ 0} x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dCHWOutShape "max_pool2d" kH kW stride
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
    let node : Node :=
      { id := id, parents := [xId], kind := .maxPool2d kH kW stride, outShape := outShape }
    pushNode (α := α) node
    pure (.node id)
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {_h1 : kH ≠ 0} {_h2 : kW ≠ 0} x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dCHWOutShapePad "max_pool2d_pad" kH kW stride padding
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
    let node : Node :=
      { id := id
        parents := [xId]
        kind := .maxPool2dPad kH kW stride padding
        outShape := outShape }
    pushNode (α := α) node
    pure (.node id)
  smoothMaxPool2d := fun {_kH _kW _inH _inW _inC _stride} {_h1} {_h2} _x _temp =>
    fail (α := α) "TorchLean→IR: smooth_max_pool2d is outside the verifier IR fragment"
  avgPool2d := fun {kH kW inH inW inC stride} (_h1 : kH ≠ 0) (_h2 : kW ≠ 0) x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dCHWOutShape "avg_pool2d" kH kW stride
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
    let node : Node :=
      { id := id, parents := [xId], kind := .avgPool2d kH kW stride, outShape := outShape }
    pushNode (α := α) node
    pure (.node id)
  avgPool2dPad := fun {kH kW inH inW inC stride padding} (_h1 : kH ≠ 0) (_h2 : kW ≠ 0) x => do
    requireContract (α := α) <|
      NN.IR.OpContracts.inferPool2dCHWOutShapePad "avg_pool2d_pad" kH kW stride padding
        (.dim inC (.dim inH (.dim inW .scalar)))
    let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) x
    let id ← freshId (α := α)
    let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding
    let node : Node :=
      { id := id
        parents := [xId]
        kind := .avgPool2dPad kH kW stride padding
        outShape := outShape }
    pushNode (α := α) node
    pure (.node id)

  relu := fun {s} x => emitUnary (α := α) (kind := .relu) (x := x) (t := s) (outShape := s)
  sigmoid := fun {s} x => emitUnary (α := α) (kind := .sigmoid) (x := x) (t := s) (outShape := s)
  tanh := fun {s} x => emitUnary (α := α) (kind := .tanh) (x := x) (t := s) (outShape := s)
  softmax := fun {s} x => do
    let axis :=
      match Shape.rank s with
      | 0 => 0
      | Nat.succ r => r
    emitUnary (α := α) (kind := .softmax axis) (x := x) (t := s) (outShape := s)
  logSoftmax := fun {s} x => do
    -- The verifier IR represents `log_softmax` by lowering it through
    -- `softmax` followed by `log` so the semantic graph remains expressible; runtime eager/compiled
    -- training still uses the stable primitive added in the autograd stack.
    let axis :=
      match Shape.rank s with
      | 0 => 0
      | Nat.succ r => r
    let probs ← emitUnary (α := α) (kind := .softmax axis) (x := x) (t := s) (outShape := s)
    emitUnary (α := α) (kind := .log) (x := probs) (t := s) (outShape := s)
  softplus := fun {s} x => do
    let oneT : Tensor α s := Spec.fill (α := α) Numbers.one s
    let ex ← emitUnary (α := α) (kind := .exp) (x := x) (t := s) (outShape := s)
    let sum ← emitBinary (α := α) (kind := .add) (a := ex) (b := .const oneT)
    emitUnary (α := α) (kind := .log) (x := sum) (t := s) (outShape := s)
  exp := fun {s} x => emitUnary (α := α) (kind := .exp) (x := x) (t := s) (outShape := s)
  log := fun {s} x => emitUnary (α := α) (kind := .log) (x := x) (t := s) (outShape := s)
  inv := fun {s} x => emitUnary (α := α) (kind := .inv) (x := x) (t := s) (outShape := s)
  detach := fun {s} x => emitUnary (α := α) (kind := .detach) (x := x) (t := s) (outShape := s)
  safeLog := fun {s} x ε => do
    let epsT : Tensor α s := Spec.fill (α := α) ε s
    let shifted ← emitBinary (α := α) (kind := .add) (a := x) (b := .const epsT)
    emitUnary (α := α) (kind := .log) (x := shifted) (t := s) (outShape := s)
  sum := fun {_s} x =>
    emitUnary (α := α) (kind := .sum) (x := x) (t := Shape.scalar) (outShape := Shape.scalar)
  flatten := fun {s} x => do
    let outShape : Shape := .dim (Shape.size s) .scalar
    emitUnary (α := α) (kind := .flatten s) (x := x) (t := outShape) (outShape := outShape)

  linear := fun {inDim outDim} w b x => do
    let wT ← getConst (α := α) (s := .dim outDim (.dim inDim .scalar)) w
    let bT ← getConst (α := α) (s := .dim outDim .scalar) b
    let xId ← ensureNode (α := α) (s := .dim inDim .scalar) x
    let id ← freshId (α := α)
    let node : Node :=
      { id := id, parents := [xId], kind := .linear, outShape := .dim outDim .scalar }
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
    let node : Node := { id := id, parents := [yId, tId], kind := .mseLoss, outShape :=
      Shape.scalar }
    pushNode (α := α) node
    pure (.node id)

  layerNorm := fun {seqLen embedDim} _hSeq _hEmb x gamma beta => do
    let sX : Shape := .dim seqLen (.dim embedDim .scalar)
    let xNorm : Ref α sX ←
      emitUnary (α := α) (kind := .layernorm (axis := 1)) (x := x) (t := sX) (outShape := sX)
    let gammaT ← getConst (α := α) (s := .dim embedDim .scalar) gamma
    let betaT ← getConst (α := α) (s := .dim embedDim .scalar) beta
    let gammaB : Tensor α sX := Tensor.dim (fun _ => gammaT)
    let betaB : Tensor α sX := Tensor.dim (fun _ => betaT)
    let scaled ← emitBinary (α := α) (kind := .mul_elem) (a := xNorm) (b := .const gammaB)
    emitBinary (α := α) (kind := .add) (a := scaled) (b := .const betaB)

  batchnormChannelFirst := fun {_channels _height _width} _hC _hH _hW _x _gamma _beta => do
    fail (α := α)
      ("TorchLean→IR: batchnorm_channel_first (training-style BN: stats from " ++
        "x) is outside the verifier IR fragment. For inference-time BN, " ++
        "use TorchLean.Norm.batch_norm2d_chw_eval / batch_norm2d_nchw_eval (or " ++
        "NN.batchnorm_channel_first_eval).")

  multiHeadAttention := fun {n numHeads dModel headDim} _h1 wq wk wv wo x mask => do
    let sX : Shape := .dim n (.dim dModel .scalar)
    let sBig : Shape := .dim n (.dim (numHeads * headDim) .scalar)
    let Q : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wq) (sOut := sBig) (outShape := sBig)
    let K : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wk) (sOut := sBig) (outShape := sBig)
    let V : Ref α sBig ← emitMatmul (α := α) (a := x) (b := wv) (sOut := sBig) (outShape := sBig)

    -- split_heads_spec: reshape (n, numHeads*headDim) → (numHeads, n, headDim)
    let sHeads : Shape := .dim numHeads (.dim n (.dim headDim .scalar))
    let Qh : Ref α sHeads ← emitUnary (α := α) (kind := .reshape sBig sHeads) (x := Q) (t := sHeads)
      (outShape := sHeads)
    let Kh : Ref α sHeads ← emitUnary (α := α) (kind := .reshape sBig sHeads) (x := K) (t := sHeads)
      (outShape := sHeads)
    let Vh : Ref α sHeads ← emitUnary (α := α) (kind := .reshape sBig sHeads) (x := V) (t := sHeads)
      (outShape := sHeads)

    -- scores = Q · Kᵀ per head: (numHeads,n,headDim) × (numHeads,headDim,n) → (numHeads,n,n)
    let sKt : Shape := .dim numHeads (.dim headDim (.dim n .scalar))
    let Kt : Ref α sKt ← emitUnary (α := α) (kind := .transpose3dLastTwo) (x := Kh) (t := sKt)
      (outShape := sKt)
    let sScores : Shape := .dim numHeads (.dim n (.dim n .scalar))
    let scores : Ref α sScores ← emitMatmul (α := α) (a := Qh) (b := Kt) (sOut := sScores) (outShape
      := sScores)

    -- scale by 1 / sqrt(headDim)
    let invScale : α := Numbers.one / MathFunctions.sqrt (headDim : α)
    let scaleT : Tensor α sScores := Spec.fill (α := α) invScale sScores
    let scaledScores ← emitBinary (α := α) (kind := .mul_elem) (a := scores) (b := .const scaleT)

    -- Attention weights. Boolean masks use hard-mask semantics: blocked entries contribute
    -- literal zero numerator, not a finite additive sentinel.
    let attn : Ref α sScores ←
      match mask with
      | none =>
          -- last-axis softmax (axis = 2 for rank-3 tensor)
          emitUnary (α := α) (kind := .softmax (axis := 2)) (x := scaledScores) (t := sScores)
            (outShape := sScores)
      | some m => do
          let rec to01 : {s : Shape} → Tensor Bool s → Tensor α s
            | .scalar, .scalar b => Tensor.scalar (if b then Numbers.one else Numbers.zero)
            | .dim _ s, .dim f => Tensor.dim (fun i => to01 (s := s) (f i))
          let mask2D : Tensor α (.dim n (.dim n .scalar)) :=
            to01 (s := .dim n (.dim n .scalar)) m
          let mask3D : Tensor α sScores := Tensor.dim (fun _ => mask2D)
          let maskR : Ref α sScores := .const mask3D
          let numeratorsRaw ←
            emitUnary (α := α) (kind := .exp) (x := scaledScores) (t := sScores)
              (outShape := sScores)
          let numerators ← emitBinary (α := α) (kind := .mul_elem) (a := numeratorsRaw) (b := maskR)
          let sDenom2D : Shape := .dim numHeads (.dim n .scalar)
          let denom2D : Ref α sDenom2D ←
            emitUnary (α := α) (kind := .reduceSum 2) (x := numerators) (t := sDenom2D)
              (outShape := sDenom2D)
          let sDenom3D : Shape := .dim numHeads (.dim n (.dim 1 .scalar))
          let denom3D : Ref α sDenom3D ←
            emitUnary (α := α) (kind := .reshape sDenom2D sDenom3D) (x := denom2D)
              (t := sDenom3D) (outShape := sDenom3D)
          let denomB : Ref α sScores ←
            emitUnary (α := α) (kind := .broadcastTo sDenom3D sScores) (x := denom3D)
              (t := sScores) (outShape := sScores)
          let invDenom ← emitUnary (α := α) (kind := .inv) (x := denomB) (t := sScores)
            (outShape := sScores)
          emitBinary (α := α) (kind := .mul_elem) (a := numerators) (b := invDenom)

    -- apply attention weights to values: (numHeads,n,n) × (numHeads,n,headDim) →
    -- (numHeads,n,headDim)
    let outHeads : Ref α sHeads ← emitMatmul (α := α) (a := attn) (b := Vh) (sOut := sHeads)
      (outShape := sHeads)

    -- combine_heads_spec: swap first two dims then reshape back to (n, numHeads*headDim)
    let sSwap : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
    let swapped : Ref α sSwap ← emitUnary (α := α) (kind := .swap_first_two) (x := outHeads) (t :=
      sSwap) (outShape := sSwap)
    let concat : Ref α sBig ← emitUnary (α := α) (kind := .reshape sSwap sBig) (x := swapped) (t :=
      sBig) (outShape := sBig)

    -- output projection: (n, numHeads*headDim) × (numHeads*headDim, dModel) → (n, dModel)
    let out : Ref α sX ← emitMatmul (α := α) (a := concat) (b := wo) (sOut := sX) (outShape := sX)
    pure out

  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC} {hKernel} w b x => do
    -- The verifier IR only supports CHW conv2d with symmetric stride/padding.
    match d with
    | 2 =>
        let kH : Nat := kernel.get ⟨0, by decide⟩
        let kW : Nat := kernel.get ⟨1, by decide⟩
        let sH : Nat := stride.get ⟨0, by decide⟩
        let sW : Nat := stride.get ⟨1, by decide⟩
        let pH : Nat := padding.get ⟨0, by decide⟩
        let pW : Nat := padding.get ⟨1, by decide⟩
        let inH : Nat := inSpatial.get ⟨0, by decide⟩
        let inW : Nat := inSpatial.get ⟨1, by decide⟩

        -- Cast weights/input to explicit CHW / (outC,inC,kH,kW) so the verifier IR checker can
        -- pattern-match on them.
        have hKernelList : kernel.toList = [kH, kW] := by
          simpa [kH, kW] using (vector2_toList (v := kernel))
        have hw :
            Shape.ofList (outC :: inC :: kernel.toList) =
              .dim outC (.dim inC (.dim kH (.dim kW .scalar))) := by
          simp [Shape.ofList, hKernelList]
        let w4 : Ref α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hw) w

        have hinSpatial : inSpatial.toList = [inH, inW] := by
          simpa [inH, inW] using (vector2_toList (v := inSpatial))
        have hx :
            Shape.ofList (inC :: inSpatial.toList) = .dim inC (.dim inH (.dim inW .scalar)) := by
          simp [Shape.ofList, hinSpatial]
        let xCHW : Ref α (.dim inC (.dim inH (.dim inW .scalar))) :=
          Eq.mp (congrArg (fun sh => Ref α sh) hx) x

        if hs : sH = sW then
          if hp : pH = pW then
            if hStride : sH = 0 then
              fail (α := α) "TorchLean→IR: conv: stride must be nonzero"
            else
              requireContract (α := α) <|
                NN.IR.OpContracts.inferConv2dCHWOutShape inC outC kH kW sH pH
                  (.dim inC (.dim inH (.dim inW .scalar)))
              let kT ← getConst (α := α) (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar)))) w4
              let bT ← getConst (α := α) (s := .dim outC .scalar) b
              let xId ← ensureNode (α := α) xCHW
              let id ← freshId (α := α)
              let outH : Nat := (inH + 2 * pH - kH) / sH + 1
              let outW : Nat := (inW + 2 * pH - kW) / sH + 1
              let outShape : Shape := .dim outC (.dim outH (.dim outW .scalar))
              let node : Node :=
                { id := id
                  parents := [xId]
                  kind := .conv2d inC outC kH kW sH pH
                  outShape := outShape }
              have hkH : kH ≠ 0 := hKernel ⟨0, by decide⟩
              have hkW : kW ≠ 0 := hKernel ⟨1, by decide⟩
              let spec : Spec.Conv2DSpec inC outC kH kW sH pH α hInC hkH hkW :=
                { kernel := kT, bias := bT }
              let cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α :=
                { inC := inC, outC := outC, kH := kH, kW := kW
                  stride := sH, padding := pH
                  inH := inH, inW := inW
                  hIn := hInC, hKH := hkH, hKW := hkW, hStride := hStride,
                  spec := spec }
              modify fun st =>
                { st with ps := { st.ps with conv2dCfg := st.ps.conv2dCfg.insert id cfg } }
              pushNode (α := α) node
              -- Cast to the generic output shape expected by `Ops.conv`.
              have hout :
                  outShape =
                    Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)
                  := by
                have houtList :
                    (Spec.convOutSpatial inSpatial kernel stride padding).toList =
                      [ (Spec.convOutSpatial inSpatial kernel stride padding).get ⟨0, by decide⟩
                      , (Spec.convOutSpatial inSpatial kernel stride padding).get ⟨1, by decide⟩
                      ] := vector2_toList (v := Spec.convOutSpatial inSpatial kernel stride padding)
                rw [houtList]
                simp [Shape.ofList]
                simp [Spec.convOutSpatial, Spec.convOutDim, Vector.ofFn, Vector.get]
                have hs' : stride[1] = stride[0] := by
                  change stride.get ⟨1, by decide⟩ = stride.get ⟨0, by decide⟩
                  exact hs.symm
                have hp' : padding[1] = padding[0] := by
                  change padding.get ⟨1, by decide⟩ = padding.get ⟨0, by decide⟩
                  exact hp.symm
                simp [hs', hp']
                simp [outShape, outH, outW, inH, inW, kH, kW, sH, pH]
                simp [Vector.get]
              pure (Eq.mp (congrArg (fun sh => Ref α sh) hout) (.node id))
          else
            fail (α := α) "TorchLean→IR: conv: verifier IR requires uniform padding"
        else
          fail (α := α) "TorchLean→IR: conv: verifier IR requires uniform stride"
    | _ =>
          fail (α := α) "TorchLean→IR: conv: verifier IR accepts d=2"

  convTranspose := fun {_d _inC _outC} {_kernel _stride _padding} {_inSpatial} {_hInC} {_hKernel} _w
      _b _x =>
    fail (α := α) "TorchLean→IR: conv_transpose is outside the verifier IR fragment"

  conv2d := fun {inC outC kH kW stride padding inH inW} {h1} {h2} {h3} kernel bias input => do
    if hStride : stride = 0 then
      fail (α := α) "TorchLean→IR: conv2d: stride must be nonzero"
    else
      requireContract (α := α) <|
        NN.IR.OpContracts.inferConv2dCHWOutShape inC outC kH kW stride padding
          (.dim inC (.dim inH (.dim inW .scalar)))
      let kT ← getConst (α := α) (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar)))) kernel
      let bT ← getConst (α := α) (s := .dim outC .scalar) bias
      let xId ← ensureNode (α := α) (s := .dim inC (.dim inH (.dim inW .scalar))) input
      let id ← freshId (α := α)
      let outH : Nat := (inH + 2 * padding - kH) / stride + 1
      let outW : Nat := (inW + 2 * padding - kW) / stride + 1
      let outShape : Shape := .dim outC (.dim outH (.dim outW .scalar))
      let node : Node :=
        { id := id
          parents := [xId]
          kind := .conv2d inC outC kH kW stride padding
          outShape := outShape }
      let spec : Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3 :=
        { kernel := kT, bias := bT }
      let cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α :=
        { inC := inC, outC := outC, kH := kH, kW := kW
          stride := stride, padding := padding
          inH := inH, inW := inW
          hIn := h1, hKH := h2, hKW := h3, hStride := hStride,
          spec := spec }
      modify fun st =>
        { st with ps := { st.ps with conv2dCfg := st.ps.conv2dCfg.insert id cfg } }
      pushNode (α := α) node
      pure (.node id)

  convTranspose2d := fun {_inC _outC _kH _kW _stride _padding _inH _inW} {_h1} {_h2} {_h3} _kernel _bias
      _input =>
    fail (α := α) "TorchLean→IR: conv_transpose2d is outside the verifier IR fragment"

  randUniform := fun {s} seed => do
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := [], kind := .randUniform seed, outShape := s }
    pushNode (α := α) node
    pure (.node id)

  bernoulliMask := fun {s} keepProb seed => do
    let pId ← ensureNode (α := α) keepProb
    let id ← freshId (α := α)
    let node : Node := { id := id, parents := [pId], kind := .bernoulliMask seed, outShape := s }
    pushNode (α := α) node
    pure (.node id)

/-! ## Public compile entrypoints -/

/--
Result of compiling a TorchLean forward model to verifier IR.

This bundles:
- the produced IR graph (`NN.IR.Graph`),
- a CROWN/LiRPA-style `ParamStore` containing constants and layer parameters, and
- the distinguished input/output node ids (used by bound propagation and certificate checkers).
-/
structure CompiledIR (α : Type) [Context α] where
  /-- Compiled IR graph. -/
  graph    : Graph
  /-- Parameters/constants for verifier algorithms (IBP, CROWN, etc.). -/
  ps       : NN.MLTheory.CROWN.Graph.ParamStore α
  /-- Distinguished input node id (kept stable as `0`). -/
  inputId  : Nat
  /-- Output node id. -/
  outputId : Nat

/-- Seed the distinguished verifier input with an explicit flat input box. -/
def CompiledIR.seedInputBox {α : Type} [Context α]
    (compiled : CompiledIR α) (xB : NN.MLTheory.CROWN.FlatBox α) :
    NN.MLTheory.CROWN.Graph.ParamStore α :=
  compiled.ps.seedInputBox compiled.inputId xB

/-- Flatten a shaped center/radius pair into the verifier input-box representation. -/
def lInfBox {α : Type} [Context α] {s : Shape}
    (center radius : Tensor α s) : NN.MLTheory.CROWN.FlatBox α :=
  NN.MLTheory.CROWN.FlatBox.lInfBox (α := α) center radius

/-- Uniform `ℓ∞` box around a shaped TorchLean input tensor. -/
def lInfBall {α : Type} [Context α] {s : Shape}
    (center : Tensor α s) (eps : α) : NN.MLTheory.CROWN.FlatBox α :=
  NN.MLTheory.CROWN.FlatBox.lInfBall (α := α) center eps

/-- Seed the distinguished verifier input with a uniform `ℓ∞` ball. -/
def CompiledIR.seedLInfBall {α : Type} [Context α] {s : Shape}
    (compiled : CompiledIR α) (center : Tensor α s) (eps : α) :
    NN.MLTheory.CROWN.Graph.ParamStore α :=
  compiled.ps.seedLInfBall compiled.inputId center eps

/-- Shape of the distinguished verifier input node. -/
def CompiledIR.inputShape? {α : Type} [Context α] (compiled : CompiledIR α) :
    Except String Shape := do
  match compiled.graph.nodes[compiled.inputId]? with
  | some node => pure node.outShape
  | none =>
      throw s!"compiled verifier input node {compiled.inputId} is out of bounds for {compiled.graph.nodes.size} graph nodes"

/-- Flattened dimension of the distinguished verifier input node. -/
def CompiledIR.inputDim? {α : Type} [Context α] (compiled : CompiledIR α) :
    Except String Nat := do
  pure (Shape.size (← compiled.inputShape?))

/-- Checked affine/CROWN context for the distinguished verifier input. -/
def CompiledIR.affineCtx? {α : Type} [Context α] (compiled : CompiledIR α) :
    Except String NN.MLTheory.CROWN.Graph.AffineCtx := do
  pure { inputId := compiled.inputId, inputDim := ← compiled.inputDim? }

/--
Compatibility affine/CROWN context for existing callers.

New verifier paths should prefer `affineCtx?`, which reports malformed compiled graphs through
`Except`. This pure helper avoids panic-style indexing but cannot surface an error.
-/
def CompiledIR.affineCtx {α : Type} [Context α] (compiled : CompiledIR α) :
    NN.MLTheory.CROWN.Graph.AffineCtx :=
  { inputId := compiled.inputId
    inputDim :=
      match compiled.inputDim? with
      | .ok n => n
      | .error _ => 0 }

/-- Run IBP on a compiled verifier graph. -/
def CompiledIR.runIBP {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    Array (Option (NN.MLTheory.CROWN.FlatBox α)) :=
  NN.MLTheory.CROWN.Graph.runIBP (α := α) compiled.graph ps

/-- Read the verifier output box from an IBP result array. -/
def CompiledIR.outputBox? {α : Type} [Context α]
    (compiled : CompiledIR α)
    (boxes : Array (Option (NN.MLTheory.CROWN.FlatBox α))) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  NN.MLTheory.CROWN.Graph.outputBox? boxes compiled.outputId

/-- Read the compiled verifier output box, throwing an `IO.userError` if it is missing. -/
def CompiledIR.outputBoxOrThrow {α : Type} [Context α]
    (compiled : CompiledIR α)
    (boxes : Array (Option (NN.MLTheory.CROWN.FlatBox α))) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match compiled.outputBox? boxes with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Read the verifier output affine form from a forward affine result array. -/
def CompiledIR.outputAffine? {α : Type} [Context α]
    (compiled : CompiledIR α)
    (affs : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffine α))) :
    Except String (NN.MLTheory.CROWN.Graph.FlatAffine α) := do
  match affs[compiled.outputId]? with
  | some (some outAff) => pure outAff
  | some none => throw s!"verification output affine missing at node {compiled.outputId}"
  | none =>
      throw s!"verification output node {compiled.outputId} is out of bounds for {affs.size} affine entries"

/-- Read the verifier output CROWN bounds from a forward CROWN result array. -/
def CompiledIR.outputCROWN? {α : Type} [Context α]
    (compiled : CompiledIR α)
    (bounds : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds α))) :
    Except String (NN.MLTheory.CROWN.Graph.FlatAffineBounds α) := do
  match bounds[compiled.outputId]? with
  | some (some outB) => pure outB
  | some none => throw s!"verification CROWN output missing at node {compiled.outputId}"
  | none =>
      throw s!"verification output node {compiled.outputId} is out of bounds for {bounds.size} CROWN entries"

/-- Run forward CROWN and evaluate the compiled verifier output on a selected input box. -/
def CompiledIR.outputBoxCROWN? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (xB : NN.MLTheory.CROWN.FlatBox α) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  let inputDim ← compiled.inputDim?
  NN.MLTheory.CROWN.Graph.outputBoxCROWN? (α := α) compiled.graph ps xB
    compiled.inputId compiled.outputId inputDim

/-- Run forward CROWN for a compiled verifier graph, throwing an `IO.userError` on failure. -/
def CompiledIR.outputBoxCROWNOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (xB : NN.MLTheory.CROWN.FlatBox α) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match compiled.outputBoxCROWN? ps xB with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Run objective-dependent backward CROWN and evaluate the scalar objective on the input box. -/
def CompiledIR.backwardObjectiveBox? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox α)))
    (xB : NN.MLTheory.CROWN.FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  let ctx ← compiled.affineCtx?
  NN.MLTheory.CROWN.Graph.backwardObjectiveBox? (α := α) compiled.graph ps ctx
    ibp xB compiled.outputId obj

/-- `IO` wrapper around `CompiledIR.backwardObjectiveBox?`. -/
def CompiledIR.backwardObjectiveBoxOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox α)))
    (xB : NN.MLTheory.CROWN.FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match compiled.backwardObjectiveBox? ps ibp xB obj with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Convert a parameter `TList` into a `RefList` of compile-time constants. -/
def refListConstOfTList {α : Type} [Context α] :
    {ss : List Shape} → Runtime.Autograd.Torch.TList α ss → Runtime.Autograd.Torch.RefList (Ref α)
      ss
  | [], .nil => .nil
  | _s :: ss, .cons t ts => .cons (.const t) (refListConstOfTList (ss := ss) ts)

/-- Compile a TorchLean forward model with a single distinguished input (the last argument). -/
def compileForward1
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Runtime.Autograd.TorchLean.Program α (paramShapes ++ [inShape]) outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    Except String (CompiledIR α) :=
  let build : BuildM α Nat := do
    let x : Ref α inShape ← emitInput (α := α)
    let psRefs : Runtime.Autograd.Torch.RefList (Ref α) paramShapes :=
      refListConstOfTList (α := α) (ss := paramShapes) params
    let allRefs : Runtime.Autograd.Torch.RefList (Ref α) (paramShapes ++ [inShape]) :=
      Runtime.Autograd.Torch.RefList.append (ss₁ := paramShapes) (ss₂ := [inShape]) psRefs (.cons x
        .nil)
    let outRef ← Runtime.Autograd.Torch.CurriedRef.uncurry
      (Ref := fun s => Ref α s) (ss := paramShapes ++ [inShape]) (model (m := BuildM α)) allRefs
    ensureNode (α := α) outRef
  match StateT.run build { nodes := #[], ps := {} } with
  | Except.error e => Except.error e
  | Except.ok (outId, st) =>
      let g : Graph := { nodes := st.nodes }
      match (g.checkWellFormed *> g.checkShapes) with
      | Except.error e => Except.error s!"TorchLean→IR: produced an ill-formed graph: {e}"
      | Except.ok _ =>
          Except.ok { graph := g, ps := st.ps, inputId := 0, outputId := outId }

end NN.Verification.TorchLean
