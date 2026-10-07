/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.Core
public import NN.Runtime.Autograd.Torch.Core.ParameterGroups

/-!
# Typed Graph Session: Differentiation and Backpropagation
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open TorchLean TorchLean.Tensor

namespace Internal

namespace TypedGraphSession

/-! ## Saved backward execution and explicit raw Tape compatibility -/

/-- Apply the names and gradient mask recorded for typed-graph leaves to a lowered tape. -/
def applyLeafMetadata {α : Type} [TorchLean.Storage α] (metadata : Array LeafMetadata)
    (tape : Runtime.Autograd.Tape α) : Runtime.Autograd.Tape α :=
  { nodes := tape.nodes.mapIdx fun id node =>
      match metadata[id]? with
      | some leaf => { node with name := leaf.name, requiresGrad := leaf.requiresGrad }
      | none => node }

/--
Lower the recorded executable graph into a runtime tape and restore its leaf metadata.

This is an explicit compatibility adapter. Normal session backward uses saved programs;
the raw Tape engine still materializes a dense contribution for each node.

The shape-indexed graph deliberately contains only mathematical leaf values. Names and
`requiresGrad` are runtime concerns, so the session stores them alongside the graph and attaches
them here. A malformed internal snapshot is rejected instead of silently changing gradient
behavior.
-/
def lowerTape {α : Type} [TorchLean.Storage α]
    (st : TypedGraphSessionState α) : Runtime.Autograd.Result (Runtime.Autograd.Tape α) := do
  if st.leafMetadata.size != st.Γ.length then
    throw "typed graph session: leaf metadata is not aligned with the typed leaf context"
  let (tape, _) ← Runtime.Autograd.TypedGraph.lowerToTapeChecked st.g st.x st.nat
  pure (applyLeafMetadata st.leafMetadata tape)

/--
Run reverse-mode backprop for the whole recorded context and return a dense gradient array.

`seed` is the upstream gradient for `out` (same convention as PyTorch's
  `loss.backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α) [Add α] [Zero α]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Spec.SomeTensor α)) := do
  s.validateTensorRef out
  let st0 ← s.state.get
  let output ← IO.ofExcept (index (Γ := st0.Γ) (ss := st0.ss) out.id sh)
  if st0.leafMetadata.size != st0.Γ.length then
    throw <| IO.userError
      "typed graph session: leaf metadata is not aligned with the typed leaf context"
  let compiled ← IO.ofExcept <|
    Runtime.Autograd.TypedGraph.compileChecked st0.g st0.x st0.nat
  let gradients := compiled.backwardDenseAllFrom output seed
  -- Frozen leaves have no outgoing reverse program. Suppressing their incoming contributions
  -- therefore changes only their final gradient. The engine retains an explicit seed even on
  -- a frozen leaf, so restore that seed rather than unconditionally replacing it with zero.
  pure <| gradients.mapIdx fun id gradient =>
    match st0.leafMetadata[id]? with
    | some leaf =>
        if leaf.requiresGrad then gradient
        else if id == output.i.val then Spec.SomeTensor.ofTensor seed
        else Spec.SomeTensor.ofTensor (Tensor.zeros gradient.shape)
    | none => gradient

/--
Extract the gradient tensor for a particular `TensorRef` from a dense gradient array.

This is the typed analogue of looking up `grads[x.id]` and casting it to the expected shape.
-/
def grad {α : Type} [TorchLean.Storage α] {sh : Shape}
  (grads : Array (Spec.SomeTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch(TypedGraphSession): gradient array out of bounds"
    if h : gAny.shape = sh then
      pure (gAny.cast h)
    else
      throw <| IO.userError <|
        s!"torch(TypedGraphSession): grad shape mismatch (expected {Shape.pretty sh}, got "
          ++ s!"{Shape.pretty gAny.shape})"

/-! ## Forward-mode JVP over the typed graph -/

/-- Check an index into the leaf context `Γ`, excluding recorded operation nodes. -/
def leafIndex {Γ : List Shape} (id : Nat) (s : Shape) :
    Runtime.Autograd.Result (Proofs.Idx Γ s) := by
    if h : id < Γ.length then
      let fin : Fin Γ.length := ⟨id, h⟩
      let got : Shape := Γ.get fin
      if hg : got = s then
        exact .ok ⟨fin, hg⟩
      else
        exact .error <|
          s!"torch(TypedGraphSession): leaf shape mismatch at id={id}: "
            ++ s!"expected {Shape.pretty s}, got {Shape.pretty got}"
  else
    exact .error s!"torch(TypedGraphSession): invalid leaf id={id} for leafLen={Γ.length}"

/-- Evaluate a JVP from a tangent pack aligned with the session's typed leaf context. -/
def jvp {α : Type} [TorchLean.Storage α] (st : TypedGraphSessionState α)
    {sh : Shape} (out : TensorRef α sh) (dx : TorchLean.TensorPack α st.Γ) :
    IO (Tensor α sh) := do
  let (_, dctx) ← IO.ofExcept <|
    Runtime.Autograd.TypedGraph.jvpChecked st.g st.x dx st.nat
  let idx ← IO.ofExcept (index (Γ := st.Γ) (ss := st.ss) out.id sh)
  pure (Proofs.getIdx (α := α) (xs := dctx) idx)

/--
Jacobian-vector product for the current session snapshot.

`dxs` is a dense array of tangents for leaf tensors, aligned with leaf creation order.
-/
def jvpDenseAll {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
    {sh : Shape} (out : TensorRef α sh) (dxs : Array (Spec.SomeTensor α)) :
  IO (Tensor α sh) := do
  s.validateTensorRef out
  let st0 ← s.state.get
  let dx ←
    if dxs.size = st0.Γ.length then
      IO.ofExcept (TorchLean.TensorPack.ofShapeErasedArray (α := α) dxs (shapes := st0.Γ))
    else
      throw <| IO.userError
        s!"torch(TypedGraphSession): dx array size mismatch \
          (expected {st0.Γ.length}, got {dxs.size})"
  jvp (α := α) st0 out dx

/-- JVP for a single leaf: tangent is nonzero only at `x`. -/
def jvpLeaf {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α) [Zero α]
    {shOut shX : Shape}
    (out : TensorRef α shOut) (x : TensorRef α shX) (dx : Tensor α shX) :
    IO (Tensor α shOut) := do
  s.validateTensorRef out
  s.validateTensorRef x
  let st0 ← s.state.get
  let idxX ← IO.ofExcept (leafIndex (Γ := st0.Γ) x.id shX)
  let dxAll : TorchLean.TensorPack α st0.Γ :=
    Proofs.Autograd.Algebra.TensorPack.single (α := α) (Γ := st0.Γ) (s := shX) idxX dx
  jvp (α := α) st0 out dxAll

/--
Apply an SGD update to all parameters recorded via `use`.

`gradients` is expected to be the dense gradient array returned by `backwardDenseAll`.
Only entries corresponding to parameters (leaves that were produced by
`use`) are used to update `Param.value`. Repeated uses of the same storage contribute additively
in leaf recording order. All gradients are validated before each storage is updated once.
PyTorch comparison: like iterating `params` and doing `p.data -= lr * p.grad`.
-/
def sgdStepAll {α : Type} [TorchLean.Storage α] (s : TypedGraphSession α)
  [Sub α] [Mul α] [Add α] [Zero α]
  (learningRate : α) (gradients : Array (Spec.SomeTensor α)) : IO Unit := do
  let parameters ← s.parametersByLeaf.get
  let groups ← EagerSession.groupParameterRegistrations parameters (← s.parameterStorageByLeaf.get)
  -- Keep the typed-session diagnostics while checking every contribution before mutation.
  for group in groups do
    for id in group.leaves do
      let gradient ← match gradients[id]? with
        | some value => pure value
        | none =>
          throw <| IO.userError "torch(TypedGraphSession): gradient array out of bounds during SGD"
      unless gradient.shape == group.parameter.s do
        throw <| IO.userError "torch(TypedGraphSession): internal grad shape mismatch during SGD"
  let prepared ← groups.mapM fun group => do
    pure (group.parameter, ← EagerSession.denseGroupGradient group gradients)
  for (parameter, gradient) in prepared do
    if hs : gradient.shape = parameter.s then
      let parameterValue ← parameter.get
      if hp : parameterValue.shape = parameter.s then
        let parameterTensor : Tensor α parameter.s := parameterValue.cast hp
        let gradientTensor : Tensor α parameter.s := gradient.cast hs
        let updated : Tensor α parameter.s :=
          subSpec parameterTensor
            (scaleSpec (α := α) (s := parameter.s) gradientTensor learningRate)
        parameter.set (Spec.SomeTensor.ofTensor updated)
      else
        throw <| IO.userError "torch(TypedGraphSession): internal param shape mismatch"
    else
      throw <| IO.userError "torch(TypedGraphSession): internal grad shape mismatch during SGD"

/-! ## Lowered-tape equivalence -/

/--
Running the runtime reverse-mode loop on the lowered tape equals `GraphData` backpropagation.

`lowerGraphDataToTape` produces a tape, and `Tape.backwardDenseFrom` is equal to
`GraphData.backpropAllCtx` up to the `TorchLean.TensorPack.toShapeErasedArray` representation
change. This theorem proves the lowering faithful to the stored VJP program. It does not prove that
the VJP is the derivative of the stored forward function; that stronger statement requires
proof-carrying `Node`s.
-/
theorem backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx
    {α : Type} [TorchLean.Storage α] [Add α]
    (st : TypedGraphSessionState α) (seed : TorchLean.TensorPack α (st.Γ ++ st.ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (Proofs.Autograd.Algebra.Graph.lowerGraphDataToTape (α := α) (Δ := NatEnv) (Γ := st.Γ)
          (ss := st.ss) st.g st.x st.nat).1)
        (grads0 := TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := st.Γ ++ st.ss)
          seed)
      =
      .ok
        (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := st.Γ ++ st.ss)
          (Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := NatEnv) (Γ :=
            st.Γ) (ss := st.ss) st.g st.x st.nat seed)) :=
  Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx
    (α := α) (Δ := NatEnv) (Γ := st.Γ) (ss := st.ss) st.g st.x st.nat seed

end TypedGraphSession

end Internal
end Torch
end Autograd
end Runtime
