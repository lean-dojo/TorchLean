/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.WellFormed
public import Std.Data.HashMap.Lemmas

/-!
# Lowered Forward Evaluation: Prefix Preservation

Lowering a forward let-chain only appends IR nodes and only inserts payload at fresh node ids.
These lemmas record that every node and payload entry below the starting graph size is unchanged
by lowering the rest of the chain.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin

/--
Generic prefix-preservation argument for `ParamStore` lookups.

The forward lowering pass appends exactly one fresh IR node at each let-binding. Any payload
lookup that is preserved by a single `lowerNode` step for keys below the fresh id is therefore
preserved by the whole lowered suffix.
-/
private theorem lowerForwardLetChain_ps_lookup_get?_lt
    {α β : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (read : NN.MLTheory.CROWN.Graph.ParamStore α → Nat → Option β)
    (hStep :
      ∀ {ss₀ : List Shape} {mid₀ : Shape} {node : Node α paramShapes inShape ss₀ mid₀}
        (id k : Nat) (params : TorchLean.TensorPack α paramShapes)
        (ps : NN.MLTheory.CROWN.Graph.ParamStore α),
        k < id →
        read
            (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := ss₀) (out := mid₀) id node params ps).2 k =
          read ps k)
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    read
        (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g params c).ps k =
      read c.ps k := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [lowerForwardLetChain]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      have hk' : k < id := by simpa [id] using hk
      have hk_succ : k < id + 1 := Nat.lt_succ_of_lt hk'
      let res :=
        lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
          (out := mid₀) id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.Builtin.LoweredIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hps' : read ps' k = read c.ps k := by
        simpa [res, ps'] using hStep (id := id) (k := k) (params := params) (ps := c.ps) hk'
      have hIH :=
        ih (c := c') (hk := by simpa [c', Array.size_push, id] using hk_succ)
      have : read
          (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params c').ps k =
          read c.ps k := by
        calc
          read
              (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params c').ps k
              =
            read c'.ps k := hIH
          _ = read c.ps k := by simpa [c'] using hps'
      simpa [lowerForwardLetChain, c', id, res] using this

/--
Lowering a let-chain does not change `ps.constVals` entries for keys `< c.graph.nodes.size`.
Lowering only inserts payload at the fresh node id, so older keys are unchanged.
-/
theorem lowerForwardLetChain_ps_constVals_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.constVals.get? k = c.ps.constVals.get? k := by
  classical
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.FlatTensor α)
      (read := fun ps k => ps.constVals.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
Lowering a let-chain does not change `ps.linearWB` entries for keys `< c.graph.nodes.size`.
Lowering only inserts linear payload at the fresh node id, so older keys are unchanged.
-/
theorem lowerForwardLetChain_ps_linearWB_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.linearWB.get? k = c.ps.linearWB.get? k := by
  classical
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.LinParams α)
      (read := fun ps k => ps.linearWB.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
Lowering a let-chain does not change `ps.convCfg` entries for keys `< c.graph.nodes.size`.
Lowering only inserts convolution payloads at fresh node ids, so older keys are unchanged.
-/
theorem lowerForwardLetChain_ps_convCfg_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.convCfg.get? k = c.ps.convCfg.get? k := by
  classical
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.ConvParams α)
      (read := fun ps k => ps.convCfg.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
Lowering a let-chain does not change `ps.batchNormEval` entries for keys below the
starting graph size. Eval-mode BatchNorm payloads enter through the broader IR/import bridge,
not through this proved first-order fragment.
-/
theorem lowerForwardLetChain_ps_batchNormEval_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.batchNormEval.get? k =
      c.ps.batchNormEval.get? k := by
  classical
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.BatchNormEvalParams α)
      (read := fun ps k => ps.batchNormEval.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        cases node <;> simp [lowerNode])
      g params c hk

/--
Lowering a let-chain preserves LayerNorm payloads below the starting graph size.

A payload-free LayerNorm step erases only its own fresh id, preventing stale future entries from
changing the source fragment's unit-affine semantics.
-/
theorem lowerForwardLetChain_ps_layerNorm_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
      (ss := ss) (out := out) g params c).ps.layerNorm.get? k = c.ps.layerNorm.get? k := by
  classical
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.LayerNormParams α)
      (read := fun ps k => ps.layerNorm.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := (ne_comm).1 hk.ne
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_erase, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
`lowerForwardLetChain` does not change existing nodes at indices `< c.graph.nodes.size`.
Lowering only appends nodes, so `getNode` agrees on the prefix.
-/
theorem lowerForwardLetChain_getNode_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {i : Nat} (hi : i < c.graph.nodes.size) :
    (NN.IR.Graph.getNode
      (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss) (out := out) g params c).graph) i)
      =
    NN.IR.Graph.getNode (g := c.graph) i := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [lowerForwardLetChain]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
          (out := mid₀) id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.Builtin.LoweredIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hi' : i < c'.graph.nodes.size := by
        simpa [c', Array.size_push] using Nat.lt_succ_of_lt hi
      have hNext :
          NN.IR.Graph.getNode
              (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes)
                (inShape := inShape) (ss := ss₀ ++ [mid₀]) (out := out₀)
                gNext params c').graph) i
            =
          NN.IR.Graph.getNode (g := c'.graph) i :=
        ih (c := c') (hi := hi')
      have hPush :
          NN.IR.Graph.getNode (g := c'.graph) i = NN.IR.Graph.getNode (g := c.graph) i := by
        simpa [c', res, id] using getNode_push_lt (g := c.graph) (n := n) (hi := hi)
      simpa [lowerForwardLetChain, c', id, res] using Eq.trans hNext hPush

/-- `lowerForwardLetChain` is monotone in `graph.nodes.size` (it only appends nodes). -/
theorem lowerForwardLetChain_nodesSize_le
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α) :
    c.graph.nodes.size ≤
      (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := out) g params c).graph.nodes.size := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [lowerForwardLetChain]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
          (out := mid₀) id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.Builtin.LoweredIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have h1 : c.graph.nodes.size ≤ c'.graph.nodes.size := by
        simp [c', Array.size_push]
      have h2 : c'.graph.nodes.size ≤
          (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params c').graph.nodes.size := by
        exact ih (c := c')
      simpa [lowerForwardLetChain, c', id, res] using Nat.le_trans h1 h2

end Correctness

end NN.Verification.Builtin.Proved
