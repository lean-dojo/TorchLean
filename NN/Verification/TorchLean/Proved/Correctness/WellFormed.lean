/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Compile

/-!
# Verified Forward Fragment: Graph Structure

The structural part of compiler correctness: every graph produced by `compileVerifiedForward1`
satisfies the verifier IR well-formedness checks.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean

  /-- Extract the list of shapes from an array of dynamic values. -/
  def shapesOfVals {α : Type} [Context α] (vals : Array (DVal α)) : List Shape :=
    vals.toList.map (fun v => v.1)

/-- `shapesOfVals` commutes with pushing an element onto the value array. -/
theorem shapesOfVals_push {α : Type} [Context α] (vals : Array (DVal α)) (v : DVal α) :
    shapesOfVals (α := α) (vals.push v) = shapesOfVals (α := α) vals ++ [v.1] := by
  simp [shapesOfVals, Array.push, List.concat_eq_append]

/-- Pushing a node onto `g.nodes` does not affect `getNode` for earlier indices. -/
theorem getNode_push_lt (g : NN.IR.Graph) (n : NN.IR.Node) {i : Nat} (hi : i < g.nodes.size)
  :
    (NN.IR.Graph.getNode (g := { nodes := g.nodes.push n }) i) = NN.IR.Graph.getNode (g := g) i :=
      by
  simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, Array.getElem?_push, hi, Nat.ne_of_lt hi]

/--
Preservation of `Graph.wellFormed` under pushing a new node with the right id, arity, and parent
discipline.
-/
theorem wellFormed_push
    (g : NN.IR.Graph) (n : NN.IR.Node)
    (hWF : g.wellFormed = true)
    (hId : n.id = g.nodes.size)
    (hArity : n.hasValidArity = true)
    (hParentsBelow : n.parentsBelow = true) :
    ({ nodes := g.nodes.push n } : NN.IR.Graph).wellFormed = true := by
    classical
  -- Work with the underlying boolean predicates (and avoid `simp` rewriting `List.all = true`
  -- into a `∀` too early).
  let pOld : Fin g.nodes.size → Bool := fun i =>
    match g.nodes[i]? with
    | none => false
    | some nd => (nd.id = i) && nd.hasValidArity && nd.parentsBelow

  let pNew : Fin (g.nodes.size + 1) → Bool := fun i =>
    match (g.nodes.push n)[i]? with
    | none => false
    | some nd => (nd.id = i) && nd.hasValidArity && nd.parentsBelow

  have hWF' : (List.finRange g.nodes.size).all pOld = true := by
    have hWF'' := hWF
    unfold NN.IR.Graph.wellFormed at hWF''
    simpa [pOld] using hWF''

  -- Unfold the new `wellFormed` and split the `finRange (k+1)` check into:
  -- - the old indices (via `castSucc`)
  -- - the new last index.
  unfold NN.IR.Graph.wellFormed
  rw [Array.size_push]
  rw [List.finRange_succ_last]
  rw [List.all_append]
  rw [List.all_map]
  rw [List.all_cons, List.all_nil]
  rw [Bool.and_true]

  -- Reduce `A && B = true` to `A = true ∧ B = true`.
  rw [Bool.and_eq_true]
  constructor
  · -- The old part: show `pNew (castSucc i) = pOld i`.
    change (List.finRange g.nodes.size).all (fun i : Fin g.nodes.size => pNew (Fin.castSucc i)) =
      true
    have hPred : ∀ i : Fin g.nodes.size, pNew (Fin.castSucc i) = pOld i := by
      intro i
      have hGetOld : g.nodes[i]? = some g.nodes[i] := by
        simp
      have hOldTrue : pOld i = true := by
        have hAll : ∀ x ∈ List.finRange g.nodes.size, pOld x = true := by
          simpa using (List.all_eq_true.mp hWF')
        exact hAll i (List.mem_finRange i)
      have hOldFull :
          (g.nodes[i].id = (i : Nat) ∧ g.nodes[i].hasValidArity = true) ∧ g.nodes[i].parentsBelow =
            true := by
        simpa [pOld, hGetOld] using hOldTrue
      have hNewTrue : pNew (Fin.castSucc i) = true := by
        have hEqNode : (g.nodes.push n)[i.castSucc] = g.nodes[i] := by
          simpa using (Array.getElem_push_lt (xs := g.nodes) (x := n) (i := i.1) i.2)
        have hNewFull :
            ((g.nodes.push n)[i.castSucc].id = (i : Nat) ∧
              (g.nodes.push n)[i.castSucc].hasValidArity = true) ∧
              (g.nodes.push n)[i.castSucc].parentsBelow = true := by
          constructor
          · constructor
            · simpa [hEqNode] using hOldFull.1.1
            · simpa [hEqNode] using hOldFull.1.2
          · simpa [hEqNode] using hOldFull.2
        simpa [pNew, hEqNode, hNewFull]
      simpa [hOldTrue] using hNewTrue
    simpa [hPred] using hWF'
  · -- The last part: it is exactly the pushed node.
    simp [Fin.val_last, hId, hArity, hParentsBelow]

  /-- Any typed index `Idx Γ s` points to a position strictly below `Γ.length`. -/
  theorem idx_id_lt_length {Γ : List Shape} {s : Shape} (x : Idx Γ s) : x.id < Γ.length :=
    by
    simp [Idx.id]

  /-- Specialized bound for indices into `Ctx inShape ss = inShape :: ss`. -/
  theorem idx_id_lt_ctxLen {inShape : Shape} {ss : List Shape} {s : Shape}
      (x : Idx (Ctx inShape ss) s) : x.id < ss.length + 1 := by
    simpa [Ctx] using (idx_id_lt_length (x := x))

  /--
  Compiled nodes always satisfy the IR arity check.
  -/
  theorem compileNode_hasValidArity
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (id : Nat)
    (node : Node α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := out)
        id node params ps).1.hasValidArity = true := by
    cases node <;>
      simp [compileNode, NN.IR.Node.hasValidArity, NN.IR.OpKind.minParents,
        NN.IR.OpKind.maxParents?]

  /--
  Compiled nodes satisfy `parentsBelow` when compiled at the next fresh id. Typed parent indices
  ensure parent ids are below the id of the newly-pushed node.
  -/
  theorem compileNode_parentsBelow
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat)
    (hId : id = (Ctx inShape ss).length)
    (node : Node α paramShapes inShape ss out) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := out)
        id node params ps).1.parentsBelow = true := by
    -- All parent indices come from typed `Idx`s into the context; hence they are < `id`.
    subst hId
    cases node with
    | const =>
      simp [compileNode, NN.IR.Node.parentsBelow]
    | paramConst =>
      simp [compileNode, NN.IR.Node.parentsBelow]
    | add a b =>
      have ha : a.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) a
      have hb : b.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) b
      have : a.id ≤ ss.length ∧ b.id ≤ ss.length :=
        ⟨Nat.lt_succ_iff.mp ha, Nat.lt_succ_iff.mp hb⟩
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, ha, hb] using this
    | sub a b =>
      have ha : a.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) a
      have hb : b.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) b
      have : a.id ≤ ss.length ∧ b.id ≤ ss.length :=
        ⟨Nat.lt_succ_iff.mp ha, Nat.lt_succ_iff.mp hb⟩
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, ha, hb] using this
    | mulElem a b =>
      have ha : a.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) a
      have hb : b.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) b
      have : a.id ≤ ss.length ∧ b.id ≤ ss.length :=
        ⟨Nat.lt_succ_iff.mp ha, Nat.lt_succ_iff.mp hb⟩
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, ha, hb] using this
    | relu x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | exp x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | log x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | inv x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | matmul2d _m _n _p a b =>
      have ha : a.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) a
      have hb : b.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) b
      have : a.id ≤ ss.length ∧ b.id ≤ ss.length :=
        ⟨Nat.lt_succ_iff.mp ha, Nat.lt_succ_iff.mp hb⟩
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, ha, hb] using this
    | bmm _batch _m _n _p a b =>
      have ha : a.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) a
      have hb : b.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) b
      have : a.id ≤ ss.length ∧ b.id ≤ ss.length :=
        ⟨Nat.lt_succ_iff.mp ha, Nat.lt_succ_iff.mp hb⟩
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, ha, hb] using this
    | reshape _inS _outS _h x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | swap_first_two _m _n _rest x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | transpose3dLastTwo _a _b _c x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | softmaxLast _hRank x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | layernorm2d _seqLen _embedDim _hSeq _hEmb x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | linear _inDim _outDim _w _b x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | conv2d _inC _outC _kH _kW _stride _padding _inH _inW _hIn _hKH _hKW _hStride _hHeight _hWidth _kernel
        _bias x =>
      have hx : x.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) x
      have : x.id ≤ ss.length := Nat.lt_succ_iff.mp hx
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hx] using this
    | mseLoss yhat target =>
      have hy : yhat.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) yhat
      have ht : target.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) target
      have : yhat.id ≤ ss.length ∧ target.id ≤ ss.length :=
        ⟨Nat.lt_succ_iff.mp hy, Nat.lt_succ_iff.mp ht⟩
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hy, ht] using this

  /--
  Compilation preserves `Graph.wellFormed` while threading the compiler accumulator through a
  forward let-chain.
  -/
  theorem compileFGraph_wellFormed
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    (hSize : c.graph.nodes.size = (Ctx inShape ss).length)
    (hWF : c.graph.wellFormed = true) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).graph.wellFormed = true := by
    classical
    induction g generalizing c with
    | ret y =>
      simpa [compileFGraph] using hWF
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      have hArity : n.hasValidArity = true := by
        simpa [n, res] using
          compileNode_hasValidArity (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss
            := ss₀) (out := mid₀)
            id node params c.ps
      have hParentsBelow : n.parentsBelow = true := by
        have hIdCtx : id = (Ctx inShape ss₀).length := by
          simpa [id] using hSize
        simpa [n, res] using
          compileNode_parentsBelow (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
            ss₀) (out := mid₀)
            (params := params) (ps := c.ps) (id := id) (hId := hIdCtx) node
      have hIdDisc : n.id = c.graph.nodes.size := by
        cases node <;> simp [compileNode, n, res, id]
      have hWF' : ({ nodes := c.graph.nodes.push n } : NN.IR.Graph).wellFormed = true := by
        exact wellFormed_push (g := c.graph) (n := n) hWF hIdDisc hArity hParentsBelow
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hWFc' : c'.graph.wellFormed = true := by
        simpa [c', id, n] using hWF'
      have hSize' : c'.graph.nodes.size = (Ctx inShape (ss₀ ++ [mid₀])).length := by
        simp [c', Ctx, Array.size_push, hSize]
      have hNext :
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
              gNext params c').graph.wellFormed = true :=
        ih (c := c') (hSize := hSize') (hWF := hWFc')
      simpa [compileFGraph, c', id, n, ps', res] using hNext

/--
Graphs produced by `compileVerifiedForward1` satisfy the IR structural discipline (`Graph.wellFormed =
  true`).
-/
theorem compileVerifiedForward1_wellFormed
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape outShape : Shape}
      (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
      outShape)
        p params).graph.wellFormed = true := by
    classical
  let input : NN.IR.Node := { id := 0, parents := [], kind := .input, outShape := inShape }
  let c0 : NN.Verification.TorchLean.CompiledIR α :=
    { graph := { nodes := #[input] }, ps := {}, inputId := 0, outputId := 0 }
  have hWF0 : c0.graph.wellFormed = true := by
    simp [c0, NN.IR.Graph.wellFormed, input, NN.IR.Node.hasValidArity, NN.IR.Node.parentsBelow,
      NN.IR.OpKind.minParents, NN.IR.OpKind.maxParents?]
  have hSize0 : c0.graph.nodes.size = (Ctx inShape []).length := by
    simp [c0, Ctx]
  simpa [compileVerifiedForward1, c0, input] using
      compileFGraph_wellFormed (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := [])
        (out := outShape)
        (g := p) (params := params) (c := c0) hSize0 hWF0

end Correctness

end NN.Verification.TorchLean.Proved
