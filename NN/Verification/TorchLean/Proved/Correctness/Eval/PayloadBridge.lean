/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.BatchNorm
public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadOps

/-!
# ParamStore to IR Payload Bridge

The verifier pipeline stores constants and layer parameters in `ParamStore`; the executable IR
semantics reads them through `Payload`.  These lemmas make that boundary explicit for every
payload-backed evaluator path.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open NN.IR

namespace Correctness

namespace IRStep

open NN.Verification.TorchLean

/-- Convert a verifier flat vector into the IR constant payload format. -/
def irConstOfFlatVec {α : Type} [Context α]
    (c : NN.MLTheory.CROWN.Graph.FlatVec α) : ConstFlat α :=
  { n := c.n, v := c.v }

/-- Convert verifier linear parameters into the IR linear payload format. -/
def irLinearOfLinParams {α : Type} [Context α]
    (p : NN.MLTheory.CROWN.Graph.LinParams α) : LinearWB α :=
  { outDim := p.m, inDim := p.n, W := p.w, b := p.b }

/-- Convert verifier convolution parameters into the IR convolution payload format. -/
def irConv2DOfGraphParams {α : Type} [Context α]
    (cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α) : NN.IR.Conv2DParams α :=
  { inC := cfg.inC, outC := cfg.outC, kH := cfg.kH, kW := cfg.kW
    stride := cfg.stride, padding := cfg.padding, inH := cfg.inH, inW := cfg.inW
    hIn := cfg.hIn, hKH := cfg.hKH, hKW := cfg.hKW, hStride := cfg.hStride,
    spec := cfg.spec }

/-- Convert verifier BatchNorm parameters into the IR BatchNorm payload format. -/
def irBatchNorm2DNchwEvalOfGraphParams {α : Type} [Context α]
    (p : NN.MLTheory.CROWN.Graph.BatchNorm2DNchwEvalParams α) :
    NN.IR.BatchNorm2DNchwEvalParams α :=
  { c := p.c, gamma := p.gamma, beta := p.beta, mean := p.mean, var := p.var, eps := p.eps }

/-! ## Payload bridge coverage -/

/-- IR constructor families whose evaluator path reads from `ParamStore` through `Payload`. -/
def payloadBridgeCoverageWitnesses : List OpKind :=
  [
    .const .scalar,
    .linear,
    .conv2d 1 1 1 1 1 0,
    .batchNorm2dNchwEval 1
  ]

/-- Tags for constructor families covered by the ParamStore-to-Payload bridge lemmas. -/
def payloadBridgeCoverageTags : List String :=
  payloadBridgeCoverageWitnesses.map OpKind.tag

/-- Whether an IR constructor family has a payload-backed evaluator path. -/
def opKindUsesPayloadBridge : OpKind → Bool
  | .const .. => true
  | .linear => true
  | .conv2d .. => true
  | .batchNorm2dNchwEval .. => true
  | _ => false

/-- Current payload-backed IR constructor tags. -/
theorem payloadBridgeCoverageTags_eq :
    payloadBridgeCoverageTags =
      ["const", "linear", "conv2d", "batch_norm2d_nchw_eval"] := by
  rfl

/-- The payload bridge checklist records each payload-backed constructor tag once. -/
theorem payloadBridgeCoverageTags_nodup :
    payloadBridgeCoverageTags.Nodup := by
  decide

/-- Every current payload-backed IR constructor family appears in the bridge checklist. -/
theorem payloadBridgeCoverageTags_complete
    (kind : OpKind) (h : opKindUsesPayloadBridge kind = true) :
    kind.tag ∈ payloadBridgeCoverageTags := by
  cases kind <;> simp [opKindUsesPayloadBridge, payloadBridgeCoverageTags,
    payloadBridgeCoverageWitnesses, OpKind.tag] at h ⊢

/-- The payload bridge checklist is exactly the set of current payload-backed constructor tags. -/
theorem payloadBridgeCoverageTags_iff (tag : String) :
    tag ∈ payloadBridgeCoverageTags ↔
      ∃ kind : OpKind, opKindUsesPayloadBridge kind = true ∧ kind.tag = tag := by
  constructor
  · intro h
    simp [payloadBridgeCoverageTags, payloadBridgeCoverageWitnesses, OpKind.tag] at h
    rcases h with h | h | h | h
    · subst tag
      exact ⟨.const .scalar, by simp [opKindUsesPayloadBridge, OpKind.tag]⟩
    · subst tag
      exact ⟨.linear, by simp [opKindUsesPayloadBridge, OpKind.tag]⟩
    · subst tag
      exact ⟨.conv2d 1 1 1 1 1 0, by simp [opKindUsesPayloadBridge, OpKind.tag]⟩
    · subst tag
      exact ⟨.batchNorm2dNchwEval 1, by simp [opKindUsesPayloadBridge, OpKind.tag]⟩
  · rintro ⟨kind, hPayload, rfl⟩
    exact payloadBridgeCoverageTags_complete kind hPayload

/-- Constants are forwarded from `ParamStore.constVals` to `Payload.const?` without changing data. -/
theorem payloadOfParamStore_const?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).const? id =
      (ps.constVals.get? id).map (irConstOfFlatVec (α := α)) := by
  rfl

/-- A present constant entry becomes the matching IR constant payload. -/
theorem payloadOfParamStore_const?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (c : NN.MLTheory.CROWN.Graph.FlatVec α)
    (h : ps.constVals.get? id = some c) :
    (payloadOfParamStore (α := α) ps).const? id =
      some (irConstOfFlatVec (α := α) c) := by
  rw [payloadOfParamStore_const?_eq, h]
  rfl

/-- Missing constant entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_const?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.constVals.get? id = none) :
    (payloadOfParamStore (α := α) ps).const? id = none := by
  rw [payloadOfParamStore_const?_eq, h]
  rfl

/-- Linear weights are forwarded from `ParamStore.linearWB` to `Payload.linear?`. -/
theorem payloadOfParamStore_linear?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).linear? id =
      (ps.linearWB.get? id).map (irLinearOfLinParams (α := α)) := by
  rfl

/-- A present linear entry becomes the matching IR linear payload. -/
theorem payloadOfParamStore_linear?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (p : NN.MLTheory.CROWN.Graph.LinParams α)
    (h : ps.linearWB.get? id = some p) :
    (payloadOfParamStore (α := α) ps).linear? id =
      some (irLinearOfLinParams (α := α) p) := by
  rw [payloadOfParamStore_linear?_eq, h]
  rfl

/-- Missing linear entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_linear?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.linearWB.get? id = none) :
    (payloadOfParamStore (α := α) ps).linear? id = none := by
  rw [payloadOfParamStore_linear?_eq, h]
  rfl

/-- Convolution parameters are forwarded from `ParamStore.conv2dCfg` to `Payload.conv2d?`. -/
theorem payloadOfParamStore_conv2d?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).conv2d? id =
      (ps.conv2dCfg.get? id).map (irConv2DOfGraphParams (α := α)) := by
  rfl

/-- A present convolution entry becomes the matching IR convolution payload. -/
theorem payloadOfParamStore_conv2d?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α)
    (h : ps.conv2dCfg.get? id = some cfg) :
    (payloadOfParamStore (α := α) ps).conv2d? id =
      some (irConv2DOfGraphParams (α := α) cfg) := by
  rw [payloadOfParamStore_conv2d?_eq, h]
  rfl

/-- Missing convolution entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_conv2d?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.conv2dCfg.get? id = none) :
    (payloadOfParamStore (α := α) ps).conv2d? id = none := by
  rw [payloadOfParamStore_conv2d?_eq, h]
  rfl

/-- BatchNorm parameters are forwarded from `ParamStore.batchNorm2dNchwEval` to the IR payload. -/
theorem payloadOfParamStore_batchNorm2dNchwEval?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).batchNorm2dNchwEval? id =
      (ps.batchNorm2dNchwEval.get? id).map
        (irBatchNorm2DNchwEvalOfGraphParams (α := α)) := by
  rfl

/-- A present BatchNorm entry becomes the matching IR BatchNorm payload. -/
theorem payloadOfParamStore_batchNorm2dNchwEval?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (p : NN.MLTheory.CROWN.Graph.BatchNorm2DNchwEvalParams α)
    (h : ps.batchNorm2dNchwEval.get? id = some p) :
    (payloadOfParamStore (α := α) ps).batchNorm2dNchwEval? id =
      some (irBatchNorm2DNchwEvalOfGraphParams (α := α) p) := by
  rw [payloadOfParamStore_batchNorm2dNchwEval?_eq, h]
  rfl

/-- Missing BatchNorm entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_batchNorm2dNchwEval?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.batchNorm2dNchwEval.get? id = none) :
    (payloadOfParamStore (α := α) ps).batchNorm2dNchwEval? id = none := by
  rw [payloadOfParamStore_batchNorm2dNchwEval?_eq, h]
  rfl

/-! ## Evaluator facts for ParamStore-backed payloads -/

/-- `Graph.evalConst` reads flat constants through the `ParamStore` bridge at any node id. -/
theorem evalConst_from_paramStore
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat) (s : Shape)
    (v : Tensor α (.dim (Shape.size s) .scalar))
    (hStore :
      ps.constVals.get? id =
        some ({ n := Shape.size s, v := v } : NN.MLTheory.CROWN.Graph.FlatVec α)) :
    Graph.evalConst (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (s := s)
      =
      Except.ok (Tensor.unflattenSpec (α := α) (s := s) v) := by
  simp [Graph.evalConst,
    payloadOfParamStore_const?_some (ps := ps) (id := id)
      ({ n := Shape.size s, v := v } : NN.MLTheory.CROWN.Graph.FlatVec α) hStore,
    irConstOfFlatVec, Graph.castDimScalar, Pure.pure, Except.pure]

/-- Missing `ParamStore.constVals` entries are rejected by `Graph.evalConst` at any node id. -/
theorem evalConst_missing_from_paramStore
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat) (s : Shape)
    (hMissing : ps.constVals.get? id = none) :
    Graph.evalConst (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (s := s)
      =
      Except.error s!"IR eval: missing const payload for node {id}" := by
  simp [Graph.evalConst, payloadOfParamStore_const?_none (ps := ps) (id := id) hMissing]
  rfl

/-- `Graph.evalLinear` reads affine parameters through the `ParamStore` bridge at any node id. -/
theorem evalLinear_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id outDim inDim : Nat)
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar))
    (hStore :
      ps.linearWB.get? id =
        some ({ m := outDim, n := inDim, w := W, b := b } :
          NN.MLTheory.CROWN.Graph.LinParams α)) :
    Graph.evalLinear (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := DVal.mk (α := α) (.dim inDim .scalar) x)
        (outShape := .dim outDim .scalar)
      =
      Except.ok
        (DVal.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalLinear,
    payloadOfParamStore_linear?_some (ps := ps) (id := id)
      ({ m := outDim, n := inDim, w := W, b := b } :
        NN.MLTheory.CROWN.Graph.LinParams α) hStore,
    irLinearOfLinParams, Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.linearWB` entries are rejected by `Graph.evalLinear` at any node id. -/
theorem evalLinear_missing_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id outDim inDim : Nat)
    (x : Tensor α (.dim inDim .scalar))
    (hMissing : ps.linearWB.get? id = none) :
    Graph.evalLinear (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := DVal.mk (α := α) (.dim inDim .scalar) x)
        (outShape := .dim outDim .scalar)
      =
      Except.error s!"IR eval: missing linear payload for node {id}" := by
  simp [Graph.evalLinear,
    payloadOfParamStore_linear?_none (ps := ps) (id := id) hMissing]
  rfl

/-- `Graph.evalConv2D` reads convolution parameters through the `ParamStore` bridge at any node id. -/
theorem evalConv2D_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat)
    (cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α)
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (hStore : ps.conv2dCfg.get? id = some cfg)
    (hHeight : OpContracts.checkWindowFits "conv2d" "height" cfg.inH cfg.kH cfg.padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "conv2d" "width" cfg.inW cfg.kW cfg.padding = .ok ()) :
    let outShape : Shape :=
      .dim cfg.outC
        (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
          (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))
    Graph.evalConv2D (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := DVal.mk (α := α) (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) x)
      =
      Except.ok
        (DVal.mk (α := α) outShape
          (Spec.conv2dSpec (α := α) (layer := cfg.spec) (input := x))) := by
  have hInfer :
      OpContracts.inferConv2dCHWOutShape cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride
          cfg.padding (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) =
        Except.ok
          (.dim cfg.outC
            (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
              (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))) := by
    simp [OpContracts.inferConv2dCHWOutShape, OpContracts.checkPositive,
      OpContracts.conv2dCHWOutShape, OpContracts.slideOutPad, cfg.hIn, cfg.hKH,
      cfg.hKW, cfg.hStride, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
      Except.pure]
  simp [Graph.evalConv2D,
    payloadOfParamStore_conv2d?_some (ps := ps) (id := id) cfg hStore,
    irConv2DOfGraphParams, Graph.expectShape, hInfer,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.conv2dCfg` entries are rejected by `Graph.evalConv2D` at any node id. -/
theorem evalConv2D_missing_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat)
    (cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α)
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (hMissing : ps.conv2dCfg.get? id = none) :
    Graph.evalConv2D (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := DVal.mk (α := α) (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) x)
      =
      Except.error s!"IR eval: missing conv2d payload for node {id}" := by
  simp [Graph.evalConv2D,
    payloadOfParamStore_conv2d?_none (ps := ps) (id := id) hMissing]
  rfl

/--
`Graph.evalBatchNorm2DNchwEval` reads eval-mode BatchNorm parameters through the `ParamStore`
bridge at any node id.
-/
theorem evalBatchNorm2DNchwEval_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id n c h w : Nat)
    (gamma beta mean var : Tensor α (.dim c .scalar))
    (eps : α)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar)))))
    (hStore :
      ps.batchNorm2dNchwEval.get? id =
        some ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
          NN.MLTheory.CROWN.Graph.BatchNorm2DNchwEvalParams α)) :
    Graph.evalBatchNorm2DNchwEval (α := α) (payload := payloadOfParamStore (α := α) ps)
        (id := id)
        (x := DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar)))) x)
      =
      Except.ok
        (DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))
          (batchNorm2dNchwEvalTensor (α := α) gamma beta mean var eps x)) := by
  simp [Graph.evalBatchNorm2DNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_some (ps := ps) (id := id)
      ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
        NN.MLTheory.CROWN.Graph.BatchNorm2DNchwEvalParams α) hStore,
    irBatchNorm2DNchwEvalOfGraphParams, Graph.expectShape,
    batchNorm2dNchwEvalTensor, batchNorm2dNchwEvalScalar, Bind.bind, Except.bind,
    Pure.pure, Except.pure]
  rfl

/-- Missing `ParamStore.batchNorm2dNchwEval` entries are rejected by BatchNorm evaluation. -/
theorem evalBatchNorm2DNchwEval_missing_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id n c h w : Nat)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar)))))
    (hMissing : ps.batchNorm2dNchwEval.get? id = none) :
    Graph.evalBatchNorm2DNchwEval (α := α) (payload := payloadOfParamStore (α := α) ps)
        (id := id)
        (x := DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar)))) x)
      =
      Except.error s!"IR eval: missing batch_norm2d_nchw_eval payload for node {id}" := by
  simp [Graph.evalBatchNorm2DNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_none (ps := ps) (id := id) hMissing]
  rfl

/-! ## One-step graph facts for ParamStore-backed payload nodes -/

/-- A `const` node in any graph reads its value from the matching `ParamStore.constVals` entry. -/
theorem evalAt_const_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id : Nat) (s inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (v : Tensor α (.dim (Shape.size s) .scalar))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [] (NN.IR.OpKind.const s) s))
    (hStore :
      ps.constVals.get? id =
        some ({ n := Shape.size s, v := v } : NN.MLTheory.CROWN.Graph.FlatVec α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.unflattenSpec (α := α) (s := s) v)) := by
  simp [Graph.evalAt, hNode,
    evalConst_from_paramStore (ps := ps) (id := id) (s := s) (v := v) hStore,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.constVals` entries are rejected at any `const` node id. -/
theorem evalAt_const_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id : Nat) (s inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [] (NN.IR.OpKind.const s) s))
    (hMissing : ps.constVals.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing const payload for node {id}" := by
  simp [Graph.evalAt, hNode,
    evalConst_missing_from_paramStore (ps := ps) (id := id) (s := s) hMissing,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- A `linear` node in any graph reads weights and bias from its `ParamStore.linearWB` entry. -/
theorem evalAt_linear_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId outDim inDim : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] NN.IR.OpKind.linear
          (Shape.dim outDim Shape.scalar)))
    (hParent :
      Graph.expectShape (α := α) (expected := Shape.dim inDim Shape.scalar) vals[pId]! =
        Except.ok x)
    (hStore :
      ps.linearWB.get? id =
        some ({ m := outDim, n := inDim, w := W, b := b } :
          NN.MLTheory.CROWN.Graph.LinParams α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok
        (DVal.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalAt, hNode, Graph.evalLinear,
    payloadOfParamStore_linear?_some (ps := ps) (id := id)
      ({ m := outDim, n := inDim, w := W, b := b } :
        NN.MLTheory.CROWN.Graph.LinParams α) hStore,
    irLinearOfLinParams, hParent,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.linearWB` entries are rejected at any `linear` node id. -/
theorem evalAt_linear_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId outDim : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] NN.IR.OpKind.linear
          (Shape.dim outDim Shape.scalar)))
    (hMissing : ps.linearWB.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing linear payload for node {id}" := by
  simp [Graph.evalAt, hNode, Graph.evalLinear,
    payloadOfParamStore_linear?_none (ps := ps) (id := id) hMissing,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

/-- A `conv2d` node in any graph reads its convolution payload from `ParamStore.conv2dCfg`. -/
theorem evalAt_conv2d_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId : Nat)
    (cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (hNode :
      let outShape : Shape :=
        .dim cfg.outC
          (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
            (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId]
          (NN.IR.OpKind.conv2d cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding)
          outShape))
    (hParent :
      Graph.expectShape (α := α)
          (expected := .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) vals[pId]! =
        Except.ok x)
    (hStore : ps.conv2dCfg.get? id = some cfg)
    (hHeight : OpContracts.checkWindowFits "conv2d" "height" cfg.inH cfg.kH cfg.padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "conv2d" "width" cfg.inW cfg.kW cfg.padding = .ok ()) :
    let outShape : Shape :=
      .dim cfg.outC
        (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
          (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok
        (DVal.mk (α := α) outShape
          (Spec.conv2dSpec (α := α) (layer := cfg.spec) (input := x))) := by
  have hParentShape :
      vals[pId]!.shape = .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar)) := by
    by_cases hEq : vals[pId]!.fst = .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))
    · simpa [DVal.shape] using hEq
    · exfalso
      simp [Graph.expectShape, DVal.shape, hEq] at hParent
  have hInfer :
      OpContracts.inferConv2dCHWOutShape cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride
          cfg.padding vals[pId]!.fst =
        Except.ok
          (.dim cfg.outC
            (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
              (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))) := by
    change OpContracts.inferConv2dCHWOutShape cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride
        cfg.padding vals[pId]!.shape =
      Except.ok
        (.dim cfg.outC
          (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
            (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar)))
    rw [hParentShape]
    simp [OpContracts.inferConv2dCHWOutShape, OpContracts.checkPositive,
      OpContracts.conv2dCHWOutShape, OpContracts.slideOutPad, cfg.hIn, cfg.hKH,
      cfg.hKW, cfg.hStride, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
      Except.pure]
  simp [Graph.evalAt, hNode, Graph.evalConv2D,
    payloadOfParamStore_conv2d?_some (ps := ps) (id := id) cfg hStore,
    irConv2DOfGraphParams, hParent, hInfer, shapeBNe_refl,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.conv2dCfg` entries are rejected at any `conv2d` node id. -/
theorem evalAt_conv2d_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId : Nat)
    (cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (hNode :
      let outShape : Shape :=
        .dim cfg.outC
          (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
            (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId]
          (NN.IR.OpKind.conv2d cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding)
          outShape))
    (hMissing : ps.conv2dCfg.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing conv2d payload for node {id}" := by
  simp [Graph.evalAt, hNode, Graph.evalConv2D,
    payloadOfParamStore_conv2d?_none (ps := ps) (id := id) hMissing,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

/-- A BatchNorm node in any graph reads eval-mode NCHW parameters from its ParamStore entry. -/
theorem evalAt_batchNorm2dNchwEval_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId n c h w : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (gamma beta mean var : Tensor α (.dim c .scalar))
    (eps : α)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar)))))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] (NN.IR.OpKind.batchNorm2dNchwEval c)
          (Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))))))
    (hParentShape :
      vals[pId]!.shape = Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))))
    (hParent :
      Graph.expectShape (α := α)
          (expected := Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))))
          vals[pId]! =
        Except.ok x)
    (hStore :
      ps.batchNorm2dNchwEval.get? id =
        some ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
          NN.MLTheory.CROWN.Graph.BatchNorm2DNchwEvalParams α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok
        (DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))
          (batchNorm2dNchwEvalTensor (α := α) gamma beta mean var eps x)) := by
  have hShapeFst :
      vals[pId]!.fst = Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))) := by
    simpa [DVal.shape] using hParentShape
  simp [Graph.evalAt, hNode, Graph.evalBatchNorm2DNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_some (ps := ps) (id := id)
      ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
        NN.MLTheory.CROWN.Graph.BatchNorm2DNchwEvalParams α) hStore,
    irBatchNorm2DNchwEvalOfGraphParams, hShapeFst, hParent,
    batchNorm2dNchwEvalTensor, batchNorm2dNchwEvalScalar, shapeBNe_refl,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  funext ni
  rfl

/-- Missing BatchNorm ParamStore entries are rejected at any eval-mode NCHW BatchNorm node id. -/
theorem evalAt_batchNorm2dNchwEval_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId n c h w : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (DVal α))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] (NN.IR.OpKind.batchNorm2dNchwEval c)
          (Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))))))
    (hMissing : ps.batchNorm2dNchwEval.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := DVal.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing batch_norm2d_nchw_eval payload for node {id}" := by
  simp [Graph.evalAt, hNode, Graph.evalBatchNorm2DNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_none (ps := ps) (id := id) hMissing,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
