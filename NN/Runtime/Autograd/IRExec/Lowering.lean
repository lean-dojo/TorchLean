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
      match n.kind with
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
    let st' : State α inShape :=
      ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩
    buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st'
  else
    pure st
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

end Internal
end IRExec
end Autograd
end Runtime
