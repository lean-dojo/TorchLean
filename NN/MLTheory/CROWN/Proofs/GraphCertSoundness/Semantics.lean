/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Theorems
public import NN.Spec.Core.Context.Real
public import NN.Spec.Layers.Linear

/-!
# Graph Certificate Semantics

Value semantics for the verifier graph dialect over `ℝ`, together with the local semantic
consistency predicate used by certificate soundness proofs.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## Basic types and predicates

We work over `ℝ` because it has the right order structure for “true” soundness theorems.
The runtime checkers operate over `Float` (fast, executable), and can be used to connect a
Python-produced floating certificate to the *same* computations in Lean.
-/

/-- A value flowing along a graph edge in the semantics: a real flat tensor. -/
abbrev Val := FlatTensor ℝ

/-- Componentwise enclosure predicate for a tensor point inside a `FlatBox`. -/
abbrev encloses (B : FlatBox ℝ) (x : Tensor ℝ [B.dim]) : Prop :=
  NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses (α := ℝ) B x

/-- A flat box encloses a flat tensor when their dimensions agree and every coordinate is bounded.
-/
def EnclosesBox {α : Type} [TorchLean.Storage α] [Context α]
    (B : FlatBox α) (v : FlatTensor α) : Prop :=
  ∃ h : B.dim = v.n,
    NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses B
      (castDimScalar h.symm v.v)

/-!
## Denotational (value) semantics for the verifier graph dialect

The semantics is defined as a *safe* `Option` evaluator:

* If required parameters are missing, it returns `none`.
* If parents are missing (not yet evaluated) or dimensions mismatch, it returns `none`.
-/

/-- Safe lookup of a previously computed parent value. -/
def getVal? (vals : Array (Option Val)) (pid : Nat) : Option Val :=
  if _h : pid < vals.size then vals[pid]! else none

/-- Binary matmul uses the runtime's broadcast layout and ordered scalar contraction. -/
def evalBinaryMatmul? (leftShape rightShape : Shape) (left right : Val) : Option Val :=
  match (NN.IR.OpContracts.matmulDims leftShape rightShape).toOption with
  | none => none
  | some dims =>
      if left.n = leftShape.size ∧ right.n = rightShape.size then
        some ⟨dims.outShape.size, NN.IR.Graph.matmulFlat dims left.v right.v⟩
      else none

/-- Reconstruct a shaped value, run a generic unary tensor operation, and flatten its result. -/
def evalSomeTensorUnary? (shape : Shape)
    (op : SomeTensor ℝ → Except String (SomeTensor ℝ)) (x : Val) : Option Val :=
  if h : x.n = shape.size then
    let flatShape : Shape := .dim x.n .scalar
    have hsize : flatShape.size = shape.size := by
      simpa [flatShape, Spec.Shape.size] using h
    let input : Tensor ℝ shape := Tensor.reshapeSpec (α := ℝ) x.v hsize
    match op ⟨shape, input⟩ with
    | .ok output =>
      some { n := output.shape.size, v := Tensor.flattenSpec output.tensor }
    | .error _ => none
  else
    none

/-- Value evaluation of the same checked grouped convolution used by the interval transfer. -/
def evalConvNode? (configuration : NN.IR.ConvConfig) (parentShape outShape : Shape)
    (id : Nat) (ps : ParamStore ℝ) (input : Val) : Option Val := do
  let parameters ← ps.convCfg[id]?
  let leading ← planConvTransfer? configuration parameters parentShape outShape
  if hdim : input.n = (parameters.input leading).size then
    let shaped := ibpUnflatten input.n input.v hdim
    some ⟨(parameters.output leading).size,
      flattenSpec (Tensor.mapLeading leading
        (groupedConvSpec (inSpatial := parameters.inputSpatial)
          (stride := parameters.stride) (dilation := parameters.dilation)
          (paddingBefore := parameters.padding) (paddingAfter := parameters.paddingAfter)
          parameters.groups parameters.spec.kernel parameters.spec.bias) shaped)⟩
  else none

/-- Value semantics for a single node in the supported dialect (over `ℝ`). -/
def evalNode? (nodes : Array Node) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val)) (id : Nat) : Option Val :=
  let node := nodes[id]!
  match node.kind with
  | .input =>
      inputs[id]?
  | .const _ =>
      ps.constVals[id]?
  | .detach =>
      match NN.IR.unaryParent? node.parents with
      | some p1 => getVal? vals p1
      | _ => none
  | .add =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              if h : x.n = y.n then
                -- Use an explicit cast rather than `by simpa [h]` to keep later proofs stable.
                let yv : Tensor ℝ [x.n] :=
                  castDimScalar (α := ℝ) (Eq.symm h) y.v
                some { n := x.n, v := Tensor.addSpec (α := ℝ) x.v yv }
              else
                none
          | _, _ => none
      | _ => none
  | .sub =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              if h : x.n = y.n then
                let yv : Tensor ℝ [x.n] :=
                  castDimScalar (α := ℝ) (Eq.symm h) y.v
                some { n := x.n, v := Tensor.subSpec (α := ℝ) x.v yv }
              else
                none
          | _, _ => none
      | _ => none
  | .mulElem =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              if h : x.n = y.n then
                let yv : Tensor ℝ [x.n] :=
                  castDimScalar (α := ℝ) (Eq.symm h) y.v
                some { n := x.n, v := Tensor.mulSpec (α := ℝ) x.v yv }
              else
                none
          | _, _ => none
      | _ => none
  | .maxPool config =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x => evalSomeTensorUnary? nodes[p1]!.outShape (NN.IR.Graph.evalMaxPool config) x
          | none => none
      | _ => none
  | .avgPool config =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x => evalSomeTensorUnary? nodes[p1]!.outShape (NN.IR.Graph.evalAvgPool config) x
          | none => none
      | _ => none
  | .relu =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.reluSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .tanh =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.tanhSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .sigmoid =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.sigmoidSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .softplus =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.softplusSpec (α := ℝ) x.v }
          | none => none
      | none => none
  | .safeLog =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some epsilon =>
              if h : epsilon.n = 1 then
                let scalar := castDimScalar (α := ℝ) h epsilon.v
                some { n := x.n, v := Activation.safeLogSpec x.v (scalar.getScalar 0) }
              else none
          | _, _ => none
      | none => none
  | .sin =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x =>
              some
                { n := x.n
                  v := Tensor.map (α := ℝ) (shape := .dim x.n .scalar) (fun z => Real.sin z) x.v }
          | none => none
      | _ => none
  | .cos =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x =>
              some
                { n := x.n
                  v := Tensor.map (α := ℝ) (shape := .dim x.n .scalar) (fun z => Real.cos z) x.v }
          | none => none
      | _ => none
  | .linear =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1, ps.linearWB[id]? with
          | some x, some p =>
              if h : x.n = p.n then
                let xv : Tensor ℝ [p.n] := castDimScalar (α := ℝ) h x.v
                let yv : Tensor ℝ [p.m] :=
                  Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv
                some { n := p.m, v := yv }
              else
                none
          | _, _ => none
      | _ => none
  | .matmul =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1, ps.matmulW[id]? with
          | some x, some p =>
              if h : x.n = p.n then
                let xv : Tensor ℝ [p.n] := castDimScalar (α := ℝ) h x.v
                let z : Tensor ℝ [p.m] := Tensor.full (α := ℝ) (.dim p.m .scalar) 0
                let yv : Tensor ℝ [p.m] :=
                  Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv
                some { n := p.m, v := yv }
              else
                none
          | _, _ => none
      | none =>
          match NN.IR.binaryParents? node.parents with
          | some (p1, p2) =>
              match getVal? vals p1, getVal? vals p2 with
              | some left, some right =>
                  evalBinaryMatmul? nodes[p1]!.outShape nodes[p2]!.outShape left right
              | _, _ => none
          | none => none
  | .sum =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x =>
              let onesRow : Tensor ℝ [1, x.n] :=
                Tensor.full (α := ℝ) (.dim 1 (.dim x.n .scalar)) 1
              let y : Tensor ℝ [1] := Spec.matVecMulSpec (α := ℝ) onesRow x.v
              some { n := 1, v := y }
          | none => none
      | _ => none
  | .reshape _ _ | .flatten _ =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getVal? vals p1 with
          | some x =>
              if h : x.n = node.outShape.size then
                let xv : Tensor ℝ [node.outShape.size] :=
                  castDimScalar (α := ℝ) h x.v
                some { n := node.outShape.size, v := xv }
              else
                none
          | none => none
      | _ => none
  | .conv configuration => do
      let parent ← NN.IR.unaryParent? node.parents
      let parentNode ← nodes[parent]?
      let input ← getVal? vals parent
      evalConvNode? configuration parentNode.outShape node.outShape id ps input
  | .concat axis => do
      let layout ← concatNodeLayout? nodes node axis
      let parents ← node.parents.mapM fun parent => getVal? vals parent
      concatFlatValues? layout parents
  | _ =>
      none

/-- Optional traversal depends only on the callback values at array members. -/
theorem array_mapM_congr_of_mem {α β : Type} {f g : α → Option β} (xs : Array α)
    (h : ∀ x ∈ xs, f x = g x) : xs.mapM f = xs.mapM g := by
  have hlist : ∀ ys : List α, (∀ x ∈ ys, f x = g x) → ys.mapM f = ys.mapM g := by
    intro ys
    induction ys with
    | nil => intro _; rfl
    | cons y ys ih =>
        intro hys
        simp only [List.mapM_cons, hys y (by simp),
          ih fun x hx => hys x (by simp [hx])]
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList,
    hlist xs.toList fun x hx => h x (Array.mem_toList_iff.mp hx)]

/-- `evalNode?` reads the value table only at the node's parents, through `getVal?`. Every arm
decodes its parents with `unaryParent?`, `binaryParents?`, or a traversal of `parents`. -/
theorem evalNode?_congr (nodes : Array Node) (ps : ParamStore ℝ)
    (inputs : Std.HashMap Nat Val) (vals₁ vals₂ : Array (Option Val)) (id : Nat)
    (h : ∀ p ∈ (nodes[id]!).parents, getVal? vals₁ p = getVal? vals₂ p) :
    evalNode? nodes ps inputs vals₁ id = evalNode? nodes ps inputs vals₂ id := by
  have hu : ∀ {p}, NN.IR.unaryParent? (nodes[id]!).parents = some p →
      getVal? vals₁ p = getVal? vals₂ p :=
    fun hp => h _ (NN.IR.mem_of_unaryParent?_eq_some hp)
  have hb : ∀ {pq : Nat × Nat}, NN.IR.binaryParents? (nodes[id]!).parents = some pq →
      getVal? vals₁ pq.1 = getVal? vals₂ pq.1 ∧ getVal? vals₁ pq.2 = getVal? vals₂ pq.2 :=
    fun hp => ⟨h _ (NN.IR.fst_mem_of_binaryParents?_eq_some hp),
      h _ (NN.IR.snd_mem_of_binaryParents?_eq_some hp)⟩
  cases hk : (nodes[id]!).kind
  case concat axis =>
    simp only [evalNode?, hk, array_mapM_congr_of_mem (nodes[id]!).parents h]
  all_goals
    cases hp : NN.IR.unaryParent? (nodes[id]!).parents <;>
    cases hq : NN.IR.binaryParents? (nodes[id]!).parents with
    | none => simp_all [evalNode?]
    | some pq =>
        obtain ⟨h₁, h₂⟩ := hb hq
        simp_all [evalNode?]

/-!
The main soundness theorem does not fix a particular graph evaluator. It is stated for *any* array
`vals` that is a local model of the semantics step: each node's value must equal `evalNode?`
computed from its parents' values. Concrete total evaluators (`evalGraphRec` in
`NN.MLTheory.CROWN.Proofs.GraphRunibpEndToEnd`) are then shown to satisfy this predicate.

This separates semantic consistency from the particular implementation of the evaluator.
-/

/-!
## Local semantic consistency (`SemLocalOK`)

`SemLocalOK g ps inputs vals` means:

* `vals` has the correct length, and
* each entry `vals[id]` equals `evalNode?` computed from the full array `vals`.

For a DAG (and only for a DAG), this is exactly the property that `vals` is a valid interpretation
of the graph semantics.

Existence and uniqueness of `vals` are evaluator-correctness facts. This file proves the certificate
theorem in the reusable form: for any semantic interpretation `vals`, a locally-correct certificate
encloses it.
-/

/-- `vals` is a full semantic interpretation of `g`: one entry per node, each equal to what the
node's own evaluation rule produces from its parents' entries.

This is stated as a local condition rather than built by a recursive evaluator so the soundness
theorem applies to any interpretation, however it was obtained. -/
def SemLocalOK (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val)) : Prop :=
  vals.size = g.nodes.size ∧
  ∀ id : Nat, id < g.nodes.size → vals[id]! = evalNode? g.nodes ps inputs vals id

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
