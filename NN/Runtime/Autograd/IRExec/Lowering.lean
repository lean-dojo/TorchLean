/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Basic
public import NN.Runtime.Autograd.IRExec.Lowering.Common
public import NN.Runtime.Autograd.IRExec.Lowering.ConvolutionNormalization
public import NN.Runtime.Autograd.IRExec.Lowering.Elementwise
public import NN.Runtime.Autograd.IRExec.Lowering.LinearAlgebra
public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Reductions
public import NN.Runtime.Autograd.IRExec.Lowering.Shape
public import NN.Runtime.Autograd.IRExec.Lowering.ShapeArray

/-!
# IR Node Lowering

The canonical checked lowering loop and exhaustive operation-family dispatch from IR nodes to
executable SSA nodes.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
open NN.IR
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

namespace Internal

/--
Exhaustive operation-family dispatch, shared by the logical and array lowering loops.

The two compiler attributes keep the context index erased across this boundary. `nospecialize`
prevents copies with unused runtime shape parameters. `inline_if_reduce` permits inlining only
when the operation tag is known, and keeps the generic body from specializing its callees.
The lowering loop has a runtime operation tag, so it calls the generic dispatcher without
materializing a chronological shape prefix.
-/
@[simp, inline_if_reduce, nospecialize] def lowerNode {α : Type} [Storage α] [Context α]
    {Γ : List Shape}
    (lowering : NodeLoweringContext α Γ) : NodeLoweringResult lowering :=
  match lowering.node.kind with
  | .input => lowerBasic lowering (.input)
  | .const s => lowerBasic lowering (.const s)
  | .detach => lowerBasic lowering (.detach)
  | .randUniform seed => lowerBasic lowering (.randUniform seed)
  | .bernoulliMask seed => lowerBasic lowering (.bernoulliMask seed)
  | .add => lowerElementwise lowering (.add)
  | .sub => lowerElementwise lowering (.sub)
  | .mulElem => lowerElementwise lowering (.mulElem)
  | .abs => lowerElementwise lowering (.abs)
  | .sqrt => lowerElementwise lowering (.sqrt)
  | .inv => lowerElementwise lowering (.inv)
  | .maxElem => lowerElementwise lowering (.maxElem)
  | .minElem => lowerElementwise lowering (.minElem)
  | .relu => lowerElementwise lowering (.relu)
  | .tanh => lowerElementwise lowering (.tanh)
  | .sigmoid => lowerElementwise lowering (.sigmoid)
  | .softplus => lowerElementwise lowering (.softplus)
  | .safeLog => lowerElementwise lowering (.safeLog)
  | .exp => lowerElementwise lowering (.exp)
  | .log => lowerElementwise lowering (.log)
  | .sin => lowerElementwise lowering (.sin)
  | .cos => lowerElementwise lowering (.cos)
  | .softmax axis => lowerElementwise lowering (.softmax axis)
  | .hardMaskedSoftmax mask => lowerElementwise lowering (.hardMaskedSoftmax mask)
  | .broadcastTo s₁ s₂ => lowerReduction lowering (.broadcastTo s₁ s₂)
  | .reduceSum axis => lowerReduction lowering (.reduceSum axis)
  | .reduceMean axis => lowerReduction lowering (.reduceMean axis)
  | .sum => lowerReduction lowering (.sum)
  | .mseLoss => lowerReduction lowering (.mseLoss)
  | .matmul => lowerLinearAlgebra lowering (.matmul)
  | .linear => lowerLinearAlgebra lowering (.linear)
  | .maxPool config => lowerConvolutionNormalization lowering (.maxPool config)
  | .avgPool config => lowerConvolutionNormalization lowering (.avgPool config)
  | .conv config => lowerConvolutionNormalization lowering (.conv config)
  | .batchNormEval channelAxis channels =>
      lowerConvolutionNormalization lowering (.batchNormEval channelAxis channels)
  | .layernorm axis => lowerConvolutionNormalization lowering (.layernorm axis)
  | .permute perm => lowerShape lowering (.permute perm)
  | .reshape inS outS => lowerShape lowering (.reshape inS outS)
  | .flatten s => lowerShape lowering (.flatten s)
  | .concat axis => lowerShape lowering (.concat axis)
  | .transpose axis₁ axis₂ => lowerShape lowering (.transpose axis₁ axis₂)

/--
Lower the IR graph starting at node index `i`, extending the current SSA `State`.

This is the main lowering loop:
- it checks `i < g.nodes.size`,
- lowers node `i` into a `ForwardNode` closure (rejecting unsupported ops/shapes), and
- appends the resulting node to the accumulating `ForwardData`.

The public entrypoint `lowerToForwardGraph` handles node 0 and calls `buildFrom` starting at
`i = 1`.

Operationally, `buildFrom` is a checked lowering pass:
- success means every visited node had well-typed parents and a supported lowering case,
- failure returns a concrete error explaining the first unsupported/malformed node.
-/
def buildFrom
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape)
    (i : Nat) (st : State α inShape) : Except String (State α inShape) := do
  let ⟨ss, gd⟩ := st
  if h : i < g.nodes.size then
    let n ← g.getNode i
    let τ : Shape := n.outShape

    -- Helper: build a typed parent index expecting a specific shape.
    let parentIdx (pid : Nat) (s : Shape) : Except String (Idx ([inShape] ++ ss) s) :=
      mkIdx (inShape := inShape) (ss := ss) pid s

    let lowering : NodeLoweringContext α ([inShape] ++ ss) :=
      { graph := g, payload := payload, index := i, node := n, parentIdx := parentIdx }

    let nodeData : ForwardNode α ([inShape] ++ ss) τ ←
      lowerNode lowering
    let st' : State α inShape :=
      ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩
    buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st'
  else
    pure st
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

/-- Materialize the chronological shape list once, when lowering returns its final state. -/
def stateOfReverseData {α : Type} [Storage α] {inShape : Shape} {rev : List Shape}
    (gd : ReverseData α [inShape] rev) : State α inShape :=
  ⟨rev.reverse, ⟨rev, gd, rfl⟩⟩

/-- A reversed internal step represents exactly the public chronological `snoc` state. -/
theorem stateOfReverseData_snoc {α : Type} [Storage α] {inShape τ : Shape}
    {rev : List Shape} (gd : ReverseData α [inShape] rev)
    (node : ForwardNode α ([inShape] ++ rev.reverse) τ) :
    stateOfReverseData (.snoc gd node) =
      ⟨rev.reverse ++ [τ], ForwardData.snoc ⟨rev, gd, rfl⟩ node⟩ := by
  apply Sigma.ext (by simp [stateOfReverseData])
  unfold ForwardData.snoc
  have heq {ss ss' : List Shape} (h : (τ :: rev).reverse = ss)
      (h' : (τ :: rev).reverse = ss') :
      HEq (⟨τ :: rev, .snoc gd node, h⟩ : ForwardData α [inShape] ss)
        (⟨τ :: rev, .snoc gd node, h'⟩ : ForwardData α [inShape] ss') := by
    cases h
    cases h'
    rfl
  exact heq _ _

/--
Lower with shared reversed prefixes and an array of parent shapes.

Each successful step conses one shape onto the graph's reversed prefix and pushes one shape into
the uniquely owned array. The recursive index itself is reversed: even a compiler that retains
that index as a runtime argument only constructs `τ :: rev`, never `ss ++ [τ]`.

The compilation boundary at `lowerNode` prevents downstream specialization from materializing
the chronological context argument of the operation-family dispatcher.
-/
def buildFromArray {α : Type} [Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape) (i : Nat)
    {rev : List Shape} (gd : ReverseData α [inShape] rev)
    (shapes : ShapeArray ([inShape] ++ rev.reverse)) : Except String (State α inShape) := do
  if h : i < g.nodes.size then
    let n ← g.getNode i
    let nodeData ← lowerNode
      { graph := g, payload := payload, index := i, node := n
        parentIdx := fun pid shape => shapes.mkIdx pid shape }
    buildFromArray g payload inShape (i + 1) (.snoc gd nodeData)
      ((shapes.push n.outShape).cast (by simp [List.reverse_cons]))
  else
    pure (stateOfReverseData gd)
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

/-- Array lowering preserves every node closure and every failure of the logical lowering loop. -/
theorem buildFromArray_eq {α : Type} [Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape) (i : Nat)
    {rev : List Shape} (gd : ReverseData α [inShape] rev)
    (shapes : ShapeArray ([inShape] ++ rev.reverse)) :
    buildFromArray g payload inShape i gd shapes =
      buildFrom g payload inShape i (stateOfReverseData gd) := by
  have hIdx : shapes.mkIdx = mkIdx inShape rev.reverse := by
    funext id shape
    exact shapes.mkIdx_eq inShape rev.reverse id shape
  change buildFromArray g payload inShape i gd shapes =
    buildFrom g payload inShape i ⟨rev.reverse, ⟨rev, gd, rfl⟩⟩
  rw [buildFromArray, buildFrom]
  split
  next hi =>
    congr 1
    funext n
    congr 1
    · rw [hIdx]
    · funext node
      exact (buildFromArray_eq g payload inShape (i + 1) (.snoc gd node)
        ((shapes.push n.outShape).cast (by simp [List.reverse_cons]))).trans
          (congrArg (buildFrom g payload inShape (i + 1)) (stateOfReverseData_snoc gd node))
  next hi =>
    rfl
termination_by g.nodes.size - i
decreasing_by
  omega

/-- Initialize the array context once before lowering an arbitrary existing prefix state. -/
def buildFromWithArray {α : Type} [Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape) (i : Nat)
    (st : State α inShape) : Except String (State α inShape) :=
  buildFromArray g payload inShape i st.2.body
    ((ShapeArray.ofList ([inShape] ++ st.1)).cast
      (congrArg ([inShape] ++ ·) st.2.shape_eq.symm))

/-- Compile the original public lowering loop using its extensionally equal array implementation. -/
@[csimp] theorem buildFrom_eq_buildFromWithArray : @buildFrom = @buildFromWithArray := by
  funext α storage context g payload inShape i st
  rcases st with ⟨ss, ⟨rev, gd, rfl⟩⟩
  exact (buildFromArray_eq g payload inShape i gd
    (ShapeArray.ofList ([inShape] ++ rev.reverse))).symm

end Internal
end IRExec
end Autograd
end Runtime
