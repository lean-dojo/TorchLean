/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common

/-!
# Activation Operators

Semantic preservation for activation lowering. Pointwise activations reuse the unary validation
and graph-tail proof in `Correctness.Common`. Softmax additionally checks the axis before looking
up its parent; hard-masked softmax looks up the parent before decoding the mask. Their proofs keep
those different guards explicit.

The masked operation normalizes the innermost dimension and returns zero for an all-blocked row.
These are forward specification equalities; they do not establish native execution or gradient
correctness.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal
open Proofs (Idx getIdx)

/-- Semantic-preservation lemma for `.relu` lowering. -/
theorem buildFrom_denoteAllFrom_relu
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .relu) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  refine buildFrom_denoteAllFrom_unary g payload gd i st' x n hN hi
    "relu" (Activation.reluSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerRelu]
  · intro pId value hp hGet
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.tanh` lowering. -/
theorem buildFrom_denoteAllFrom_tanh
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .tanh) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  refine buildFrom_denoteAllFrom_unary g payload gd i st' x n hN hi
    "tanh" (Activation.tanhSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerTanh]
  · intro pId value hp hGet
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.sigmoid` lowering. -/
theorem buildFrom_denoteAllFrom_sigmoid
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sigmoid) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  refine buildFrom_denoteAllFrom_unary g payload gd i st' x n hN hi
    "sigmoid" (Activation.sigmoidSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerSigmoid]
  · intro pId value hp hGet
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.softplus` lowering. -/
theorem buildFrom_denoteAllFrom_softplus
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .softplus) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  refine buildFrom_denoteAllFrom_unary g payload gd i st' x n hN hi
    "softplus" (Activation.softplusSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerSoftplus]
  · intro pId value hp hGet
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.exp` lowering. -/
theorem buildFrom_denoteAllFrom_exp
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .exp) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  refine buildFrom_denoteAllFrom_unary g payload gd i st' x n hN hi
    "exp" (Tensor.expSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerExp]
  · intro pId value hp hGet
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.sin` lowering. -/
theorem buildFrom_denoteAllFrom_sin
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sin) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  refine buildFrom_denoteAllFrom_unary g payload gd i st' x n hN hi
    "sin" (Tensor.map (fun value => MathFunctions.sin value)) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerSin]
  · intro pId value hp hGet
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, Tensor.sinSpec, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.cos` lowering. -/
theorem buildFrom_denoteAllFrom_cos
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .cos) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  refine buildFrom_denoteAllFrom_unary g payload gd i st' x n hN hi
    "cos" (Tensor.map (fun value => MathFunctions.cos value)) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerCos]
  · intro pId value hp hGet
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, Tensor.cosSpec, throw_eq_error, Pure.pure, Except.pure]

/--
Semantic-preservation lemma for `.softmax axis` lowering.

The lowering accepts every axis that names a dimension of the output shape. The resulting typed
node uses the same axis-indexed specification as the denotational IR semantics.
-/
theorem buildFrom_denoteAllFrom_softmax
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .softmax axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerSoftmax] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild
      try cases hBuild
  | some pId =>
          cases hAxis : Spec.Shape.axisInBounds? axis n.outShape with
          | none =>
              simp [hp, hAxis] at hBuild
              try cases hBuild
          | some h =>
              simp (config := { failIfUnchanged := false }) [hp, hAxis] at hBuild
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
              | error msg =>
                  simp [hIdx] at hBuild
                  try cases hBuild
              | ok ip =>
                  simp [hIdx] at hBuild
                  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                      @Activation.softmaxSpec α _ _ n.outShape axis h.down
                        (ctx.read ip))
                  let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                  have hRec :
                      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                        (i := i + 1) st1 = .ok st' := by
                    simpa [st1, nodeData] using hBuild
                  have hGet :
                      vals0[pId]? =
                        some (Spec.SomeTensor.mk (α := α) n.outShape
                          (getIdx (α := α) (xs := ctx) ip)) := by
                    simpa [vals0, ctx] using
                      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                        (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
                  have hEval :
                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                          (input := input) (vals := vals0) (i := i) =
                        .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                      NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hAxis, hGet, throw_eq_error,
                      Pure.pure, Except.pure, nodeData, mkForwardNode]
                  have hTail := ih st1 hRec
                  exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                    (payload := payload)
                    (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                    (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic preservation for stable last-axis softmax with a hard Boolean mask. -/
theorem buildFrom_denoteAllFrom_hardMaskedSoftmax
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (mask : NN.IR.HardMask)
    (hN : g.getNode i = .ok n) (hk : n.kind = .hardMaskedSoftmax mask)
    (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerHardMaskedSoftmax] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp, throw_eq_error] at hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
          | ok ip =>
              cases hMask : NN.IR.HardMask.toTensorAs? mask n.outShape with
              | error msg =>
                  simp [hp, hIdx, hMask, throw_eq_error] at hBuild
              | ok allowed =>
                  simp (config := { failIfUnchanged := false }) [hp, hIdx, hMask] at hBuild
                  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                      Spec.hardMaskedSoftmaxSpec
                        (ctx.read ip) allowed)
                  let st1 : State α inShape :=
                    ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                  have hRec :
                      buildFrom (α := α) (g := g) (payload := payload)
                          (inShape := inShape) (i := i + 1) st1 = .ok st' := by
                    simpa [st1, nodeData] using hBuild
                  have hGet :
                      vals0[pId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                          (getIdx (α := α) (xs := ctx) ip)) := by
                    simpa [vals0, ctx] using
                      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                        (gd := gd) (x := x) (pid := pId) (s := n.outShape)
                        (idx := ip) hIdx)
                  have hEval :
                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                          (input := input) (vals := vals0) (i := i) =
                        .ok
                          (Spec.SomeTensor.mk (α := α) n.outShape
                            (nodeData.eval ctx)) := by
                    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                      NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hMask,
                      throw_eq_error, Pure.pure, Except.pure, nodeData, mkForwardNode]
                  have hTail := ih st1 hRec
                  exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                    (payload := payload)
                    (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                    (τ := n.outShape) (nodeData := nodeData) hTail hEval

end IRExec
end Autograd
end Runtime
