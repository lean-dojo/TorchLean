/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.LinearAlgebra

/-!
# Payload-Backed IR Evaluation

`linear` and `conv2d` nodes read weights from the external IR payload.  These lemmas state the
local contract at that boundary: when the expected payload is present and the shape preconditions
are met, the IR evaluator returns the corresponding spec-layer operation.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- A lookup table with one defined entry. -/
def singletonAt {β : Type} (id : Nat) (x : β) : Nat → Option β :=
  fun j => if j = id then some x else none

@[simp]
theorem singletonAt_self {β : Type} (id : Nat) (x : β) :
    singletonAt id x id = some x := by
  simp [singletonAt]

/-- A payload containing one flat constant at `id`. -/
def singletonConstPayload {α : Type} [Context α] (id : Nat) (c : ConstFlat α) : Payload α :=
  { const? := singletonAt id c }

/-- A payload containing one linear layer at `id`. -/
def singletonLinearPayload {α : Type} [Context α] (id : Nat) (p : LinearWB α) : Payload α :=
  { linear? := singletonAt id p }

/-- A payload containing one convolution layer at `id`. -/
def singletonConv2DPayload {α : Type} [Context α] (id : Nat) (p : Conv2DParams α) : Payload α :=
  { conv2d? := singletonAt id p }

/-- A payload containing one eval-mode NCHW BatchNorm layer at `id`. -/
def singletonBatchNorm2DNchwEvalPayload {α : Type} [Context α]
    (id : Nat) (p : BatchNorm2DNchwEvalParams α) : Payload α :=
  { batchNorm2dNchwEval? := singletonAt id p }

/-- A graph containing a zero-parent `const` node. -/
def constGraph (s : Shape) : Graph :=
  { nodes := #[{ id := 0, parents := [], kind := .const s, outShape := s }] }

/-- Local IR semantics for a payload-backed flat `const` node. -/
theorem evalConst_eq_unflatten
    {α : Type} [Context α]
    (id : Nat) (s : Shape)
    (v : Tensor α (.dim (Shape.size s) .scalar)) :
    let c : ConstFlat α := { n := Shape.size s, v := v }
    Graph.evalConst (α := α) (payload := singletonConstPayload (α := α) id c) (id := id) (s := s)
      =
      Except.ok (Tensor.unflattenSpec (α := α) (s := s) v) := by
  simp [Graph.evalConst, singletonConstPayload, Graph.castDimScalar, Pure.pure, Except.pure]

/-- Local IR semantics for a payload-backed flat `const` node. -/
theorem evalAt_const_eq_unflatten
    {α : Type} [Context α] [DecidableEq Shape]
    (s : Shape)
    (v : Tensor α (.dim (Shape.size s) .scalar)) :
    let c : ConstFlat α := { n := Shape.size s, v := v }
    Graph.evalAt (α := α) (g := constGraph s)
        (payload := singletonConstPayload (α := α) 0 c)
        (input := DVal.mk (α := α) s (Tensor.default (α := α) (s := s)))
        (vals := #[]) (i := 0)
      =
      Except.ok
        (DVal.mk (α := α) s (Tensor.unflattenSpec (α := α) (s := s) v)) := by
  simp [Graph.evalAt, constGraph, Graph.getNode, Graph.getNode?, Graph.evalConst,
    singletonConstPayload, Graph.castDimScalar, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing constant payloads are rejected before unflattening. -/
theorem evalConst_missing_payload
    {α : Type} [Context α]
    (payload : Payload α) (id : Nat) (s : Shape)
    (hMissing : payload.const? id = none) :
    Graph.evalConst (α := α) (payload := payload) (id := id) (s := s)
      =
      Except.error s!"IR eval: missing const payload for node {id}" := by
  simp [Graph.evalConst, hMissing]
  rfl

/-- Local IR semantics for a payload-backed `linear` node. -/
theorem evalLinear_eq_affine
    {α : Type} [Context α] [DecidableEq Shape]
    (id outDim inDim : Nat)
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar)) :
    let p : LinearWB α := { outDim := outDim, inDim := inDim, W := W, b := b }
    Graph.evalLinear (α := α) (payload := singletonLinearPayload (α := α) id p) (id := id)
        (x := DVal.mk (α := α) (.dim inDim .scalar) x)
        (outShape := .dim outDim .scalar)
      =
      Except.ok
        (DVal.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalLinear, singletonLinearPayload, Graph.expectShape, Bind.bind, Except.bind,
    Pure.pure, Except.pure]

/-- Local IR semantics for a payload-backed `linear` node. -/
theorem evalAt_linear_eq_affine
    {α : Type} [Context α] [DecidableEq Shape]
    (outDim inDim : Nat)
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar)) :
    let p : LinearWB α := { outDim := outDim, inDim := inDim, W := W, b := b }
    Graph.evalAt (α := α)
        (g := unaryGraphOut .linear (.dim inDim .scalar) (.dim outDim .scalar))
        (payload := singletonLinearPayload (α := α) 1 p)
        (input := DVal.mk (α := α) (.dim inDim .scalar) x)
        (vals := #[DVal.mk (α := α) (.dim inDim .scalar) x])
        (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.evalLinear, singletonLinearPayload, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing linear payloads are rejected before the affine operation is evaluated. -/
theorem evalLinear_missing_payload
    {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat)
    (hMissing : payload.linear? id = none)
    (x : DVal α) (outShape : Shape) :
    Graph.evalLinear (α := α) (payload := payload) (id := id) (x := x) (outShape := outShape)
      =
      Except.error s!"IR eval: missing linear payload for node {id}" := by
  simp [Graph.evalLinear, hMissing]
  rfl

/--
Local IR semantics for a payload-backed `conv2d` node.

The window-fit hypotheses are the same runtime checks used by `Graph.evalConv2D`; keeping them in
the statement makes the no-dilation shape contract explicit.
-/
theorem evalConv2D_eq_spec
    {α : Type} [Context α] [DecidableEq Shape]
    (id : Nat)
    (cfg : Conv2DParams α)
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (hHeight : OpContracts.checkWindowFits "conv2d" "height" cfg.inH cfg.kH cfg.padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "conv2d" "width" cfg.inW cfg.kW cfg.padding = .ok ()) :
    Graph.evalConv2D (α := α) (payload := singletonConv2DPayload (α := α) id cfg) (id := id)
        (x := DVal.mk (α := α) (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) x)
      =
      Except.ok
        (DVal.mk (α := α)
          (.dim cfg.outC
            (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
              (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar)))
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
  simp [Graph.evalConv2D, singletonConv2DPayload, Graph.expectShape, hInfer,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for a payload-backed no-dilation `conv2d` node. -/
theorem evalAt_conv2d_eq_spec
    {α : Type} [Context α] [DecidableEq Shape]
    (cfg : Conv2DParams α)
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (hHeight : OpContracts.checkWindowFits "conv2d" "height" cfg.inH cfg.kH cfg.padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "conv2d" "width" cfg.inW cfg.kW cfg.padding = .ok ()) :
    let outShape : Shape :=
      .dim cfg.outC
        (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
          (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.conv2d cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding)
          (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) outShape)
        (payload := singletonConv2DPayload (α := α) 1 cfg)
        (input := DVal.mk (α := α) (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) x)
        (vals := #[DVal.mk (α := α) (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) x])
        (i := 1)
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
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.evalConv2D, singletonConv2DPayload, Graph.expectShape, hInfer, shapeBNe_refl,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing convolution payloads are rejected before convolution is evaluated. -/
theorem evalConv2D_missing_payload
    {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat)
    (hMissing : payload.conv2d? id = none)
    (x : DVal α) :
    Graph.evalConv2D (α := α) (payload := payload) (id := id) (x := x)
      =
      Except.error s!"IR eval: missing conv2d payload for node {id}" := by
  simp [Graph.evalConv2D, hMissing]
  rfl

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
