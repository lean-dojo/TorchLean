/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.Array.Lemmas
public import Init.Data.List.FinRange
public import Init.Data.List.Lemmas
public import Init.Data.Range.Lemmas
public import NN.Spec.Core.TensorReductionShape
public import NN.Verification.TorchLean.Compile
public import NN.Verification.TorchLean.Correctness
public import Std.Data.HashMap.Lemmas

/-!
# Verified

Verified forward fragment (first-order).

This file defines a first-order SSA/DAG language for forward models together with a compiler into
the verifier IR (`NN.IR.Graph` plus a CROWN-style `ParamStore`).

**Main theorem (verified here):**
- `Correctness.evalCompiledForward1_eq_evalForward1`:
  evaluating the compiled IR graph agrees with the forward-fragment evaluator.

**Also verified here:**
This module contains the implementation proof. Downstream code should normally import
`NN.Verification.TorchLean.Verified`, which re-exports the concise public names.

- `compileForward1`:
  shorter compilation name for `compileVerifiedForward1`.
- `compileForward1_wellFormed`:
  alias of the graph-well-formedness theorem.
- `evalCompiledForward1_eq_evalForward1`:
  alias of the main correctness theorem.
- `evalCompiledForward1_forward1_eq`:
  alias with a shorter proof label.

`compileVerifiedForward1_wellFormed`:
  graphs produced by `compileVerifiedForward1` satisfy the IR structural discipline
  (`Graph.wellFormed = true`).

This fragment isolates a first-order subset of the higher-order runtime program language, so that
compiler correctness can be proved once and then extended op-by-op.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

-- Make projection out of dynamic values definitional for `simp` in correctness proofs.
@[simp] theorem dval_tensor_mk
    {α : Type} [Context α] {s : Shape} (t : Tensor α s) :
    DVal.tensor (α := α) (⟨s, t⟩ : DVal α) = t := rfl

@[simp] theorem graph_expectShape_mk
    {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (t : Tensor α s) :
    Graph.expectShape (α := α) (expected := s) (DVal.mk (α := α) s t) = .ok t := by
  simp [Graph.expectShape, DVal.shape, DVal.tensor, DVal.mk]
  rfl

/-! ## Typed indices -/

/-- An index into a shape context `Γ`, carrying a proof that it has shape `s`. -/
structure Idx (Γ : List Shape) (s : Shape) where
  /-- i. -/
  i : Fin Γ.length
  /-- h. -/
  h : Γ.get i = s

namespace Idx

/-- Eta rule for `Idx`: rebuilding from projections gives the same index. -/
@[simp] theorem mk_eta {Γ : List Shape} {s : Shape} (x : Idx Γ s) : Idx.mk x.i x.h = x := by
  cases x
  rfl

/--
The underlying numeric index of an `Idx`.

This is convenient when we store context values in arrays (indexed by `Nat`) rather than in
dependent lists.
-/
def id {Γ : List Shape} {s : Shape} (x : Idx Γ s) : Nat :=
  x.i.1

end Idx

/-! ## Parameter access -/

/--
Fetch a tensor from a runtime `TList` by a plain `Fin` index.

This is the low-level accessor used by `getParam`; the public shape guarantee comes from the
dependent index carried by the input list itself.
-/
def tlistGet {α : Type} : {ss : List Shape} → Runtime.Autograd.Torch.TList α ss →
    (i : Fin ss.length) → Tensor α (ss.get i)
  | [], .nil, i => nomatch i
  | _s :: _ss, .cons x _xs, ⟨0, _⟩ => x
  | _s :: ss, .cons _x xs, ⟨Nat.succ j, hj⟩ =>
      tlistGet (ss := ss) xs ⟨j, Nat.lt_of_succ_lt_succ hj⟩

/--
Fetch a parameter tensor from a runtime `TList`, using a typed index `Idx`.

This is the bridge between the parameter context `paramShapes` and the strongly-typed tensor value
returned at shape `s`.
-/
def getParam {α : Type} {paramShapes : List Shape} {s : Shape}
    (params : Runtime.Autograd.Torch.TList α paramShapes) (idx : Idx paramShapes s) : Tensor α s :=
  Tensor.castShape (tlistGet (α := α) (ss := paramShapes) params idx.i) idx.h

/-! ## First-order SSA nodes -/

/- We index runtime values by the context `inShape :: ss` where `ss` are the already-produced
node output shapes. Input is always index 0. -/

/--
Evaluation context shape list.

We always treat the distinguished input as index `0`, then append the shapes of previously-produced
SSA node outputs (`ss`).
-/
abbrev Ctx (inShape : Shape) (ss : List Shape) : List Shape :=
  inShape :: ss

/--
A well-typed SSA node in the verified forward fragment.

Each `Node` can only reference earlier values (via `Idx (Ctx inShape ss) _`), ensuring the DAG/SSA
discipline by construction.

The constructors match the operator subset for which this file proves compiler correctness into the
verifier IR (`NN.IR.Graph`).  Adding a new operator means extending both this syntax and the
correctness proof, which keeps the trusted fragment explicit.
-/
inductive Node
    (α : Type) (paramShapes : List Shape) (inShape : Shape) (ss : List Shape) :
    Shape → Type where
  | const {s : Shape} (wf : Shape.WellFormed s) (t : Tensor α s) :
      Node α paramShapes inShape ss s
  | paramConst {s : Shape} (wf : Shape.WellFormed s) (p : Idx paramShapes s) :
      Node α paramShapes inShape ss s
  | add {s : Shape} (a b : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | sub {s : Shape} (a b : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | mulElem {s : Shape} (a b : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | relu {s : Shape} (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | exp {s : Shape} (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | log {s : Shape} (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | inv {s : Shape} (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | matmul2d (m n p : Nat)
      (a : Idx (Ctx inShape ss) (.dim m (.dim n .scalar)))
      (b : Idx (Ctx inShape ss) (.dim n (.dim p .scalar))) :
      Node α paramShapes inShape ss (.dim m (.dim p .scalar))
  | bmm (batch m n p : Nat)
      (a : Idx (Ctx inShape ss) (.dim batch (.dim m (.dim n .scalar))))
      (b : Idx (Ctx inShape ss) (.dim batch (.dim n (.dim p .scalar)))) :
      Node α paramShapes inShape ss (.dim batch (.dim m (.dim p .scalar)))
  | reshape (inS outS : Shape) (h : Shape.size inS = Shape.size outS)
      (x : Idx (Ctx inShape ss) inS) :
      Node α paramShapes inShape ss outS
  | swap_first_two (m n : Nat) (rest : Shape)
      (x : Idx (Ctx inShape ss) (.dim m (.dim n rest))) :
      Node α paramShapes inShape ss (.dim n (.dim m rest))
  | transpose3dLastTwo (a b c : Nat)
      (x : Idx (Ctx inShape ss) (.dim a (.dim b (.dim c .scalar)))) :
      Node α paramShapes inShape ss (.dim a (.dim c (.dim b .scalar)))
  | softmaxLast {s : Shape} (hRank : 0 < Shape.rank s) (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | layernorm2d (seqLen embedDim : Nat) (hSeq : 0 < seqLen) (hEmb : 0 < embedDim)
      (x : Idx (Ctx inShape ss) (.dim seqLen (.dim embedDim .scalar))) :
      Node α paramShapes inShape ss (.dim seqLen (.dim embedDim .scalar))
  | linear (inDim outDim : Nat)
      (w : Idx paramShapes (.dim outDim (.dim inDim .scalar)))
      (b : Idx paramShapes (.dim outDim .scalar))
      (x : Idx (Ctx inShape ss) (.dim inDim .scalar)) :
      Node α paramShapes inShape ss (.dim outDim .scalar)
  | mseLoss {s : Shape} (yhat target : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss .scalar

/-! ## Programs (forward let-chains) -/

/--
Well-typed first-order programs, represented as a forward “let chain”.

The type parameter `ss` tracks the list of already-produced node output shapes, so every node
can only reference earlier values (including the distinguished input at index `0`).
-/
inductive FGraph (α : Type) (paramShapes : List Shape) (inShape : Shape) :
    List Shape → Shape → Type where
  | ret {ss : List Shape} {out : Shape} (y : Idx (Ctx inShape ss) out) :
      FGraph α paramShapes inShape ss out
  | let1 {ss : List Shape} {mid out : Shape} :
      Node α paramShapes inShape ss mid →
      FGraph α paramShapes inShape (ss ++ [mid]) out →
      FGraph α paramShapes inShape ss out

/-- A closed forward program from input `inShape` to output `outShape`. -/
abbrev Program (α : Type) (paramShapes : List Shape) (inShape outShape : Shape) : Type :=
  FGraph α paramShapes inShape [] outShape

/-! ## Evaluation -/

/--
Read a previously computed dynamic value and cast it back to the statically expected shape.

The verified fragment constructs only well-scoped indices, but the executable evaluator stores values
in an array, so this check gives a clear error if an implementation bug ever violates the shape
discipline.
-/
def getVal {α : Type} [Context α] [DecidableEq Shape]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (DVal α)) (idx : Idx (Ctx inShape ss) s) : Except String (Tensor α s) := do
  let v : DVal α := vals[idx.id]!
  if h : v.shape = s then
    pure (h ▸ v.tensor)
  else
    throw s!"TorchLeanVerified: expected shape {repr s}, got {repr v.shape}"

/--
Evaluate a single SSA node, given the parameter environment and current value context.

This mirrors the IR denotation for the supported operator subset.
-/
def evalNode
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (node : Node α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (vals : Array (DVal α)) : Except String (DVal α) := do
  match node with
  | .const (s := s) _wf t =>
      pure <| DVal.mk (α := α) s t
  | .paramConst (s := s) _wf p =>
      pure <| DVal.mk (α := α) s (getParam (α := α) (paramShapes := paramShapes) params p)
  | .add (s := s) a b =>
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure <| DVal.mk (α := α) s (Tensor.addSpec (α := α) ta tb)
  | .sub (s := s) a b =>
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure <| DVal.mk (α := α) s (Tensor.subSpec (α := α) ta tb)
  | .mulElem (s := s) a b =>
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure <| DVal.mk (α := α) s (Tensor.mulSpec (α := α) ta tb)
  | .relu (s := s) x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| DVal.mk (α := α) s (Activation.reluSpec (α := α) tx)
  | .exp (s := s) x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| DVal.mk (α := α) s (Tensor.expSpec (α := α) tx)
  | .log (s := s) x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      -- Domain discipline: align the verified execution model with the IR semantics and compiled
      -- runtime backend. The raw `log` is treated as undefined on nonpositive inputs; use
      -- `safe_log` in models that require epsilon protection.
      let y : Tensor α s :=
        if Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v)) tx then
          Tensor.logSpec (α := α) tx
        else
          panic!
            "TorchLeanVerified: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
      pure <| DVal.mk (α := α) s y
  | .inv (s := s) x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| DVal.mk (α := α) s (Tensor.invSpec (α := α) tx)
  | .matmul2d m n p a b =>
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim m (.dim n .scalar)) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim n (.dim p .scalar)) vals b
      pure <| DVal.mk (α := α) (.dim m (.dim p .scalar))
        (Tensor.matMulSpec (α := α) (m := m) (n := n) (p := p) ta tb)
  | .bmm batch m n p a b =>
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim batch (.dim m (.dim n .scalar))) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim batch (.dim n (.dim p .scalar))) vals b
      pure <| DVal.mk (α := α) (.dim batch (.dim m (.dim p .scalar)))
        (Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) ta tb)
  | .reshape inS outS h x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := inS) vals x
      pure <| DVal.mk (α := α) outS (Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) tx h)
  | .swap_first_two m n rest x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := .dim m (.dim n rest)) vals x
      pure <| DVal.mk (α := α) (.dim n (.dim m rest))
        (Tensor.swapFirstTwoSpec (α := α) (m := m) (n := n) (s := rest) tx)
  | .transpose3dLastTwo a b c x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim a (.dim b (.dim c .scalar))) vals x
      pure <| DVal.mk (α := α) (.dim a (.dim c (.dim b .scalar)))
        (Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) tx)
  | .softmaxLast (s := s) _hRank x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| DVal.mk (α := α) s (Activation.softmaxSpec (α := α) tx)
  | .layernorm2d seqLen embedDim hSeq hEmb x =>
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim seqLen (.dim embedDim .scalar)) vals x
      let y := Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (x := tx)
        (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
        (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
        (h_seq_pos := hSeq) (h_embed_pos := hEmb)
      pure <| DVal.mk (α := α) (.dim seqLen (.dim embedDim .scalar)) y
  | .linear inDim outDim w b x =>
      let wT := getParam (α := α) (paramShapes := paramShapes) params w
      let bT := getParam (α := α) (paramShapes := paramShapes) params b
      let xT ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim inDim .scalar) vals x
      let y := Tensor.addSpec (α := α)
        (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT
      pure <| DVal.mk (α := α) (.dim outDim .scalar) y
  | .mseLoss (s := _s) yhat target =>
      -- Mirror the IR semantics: `mse_loss` is dynamically shape-checked (both parents must have
      -- equal shape),
      -- then reduces to a scalar by averaging the squared error.
      let yV : DVal α := vals[yhat.id]!
      let tV : DVal α := vals[target.id]!
      if h : yV.shape = tV.shape then
        let yT : Tensor α yV.shape := yV.tensor
        let tT : Tensor α yV.shape := h.symm ▸ tV.tensor
        let s := yV.shape
        let diff := Tensor.subSpec (α := α) yT tT
        let sq := Tensor.mulSpec (α := α) diff diff
        let total : α := Tensor.sumSpec (α := α) sq
        let mean : α := total / (↑(Shape.size s) : α)
        pure <| DVal.mk (α := α) .scalar (Tensor.scalar mean)
      else
        throw
          s!"TorchLeanVerified: mse_loss expects equal shapes, got {repr yV.shape} vs {repr tV.shape}"

/--
Evaluate a forward let-chain program, threading an array of dynamic values.

The `vals` array stores the input and all previously-computed node outputs, so that node evaluation
can do simple array lookups by `Idx.id`.
-/
def evalFGraph
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (vals : Array (DVal α)) : Except String (Tensor α out) := do
  match g with
  | .ret y =>
      let v : DVal α := vals[y.id]!
      if h : v.shape = out then
        pure (h ▸ v.tensor)
      else
        throw s!"TorchLeanVerified: expected shape {repr out}, got {repr v.shape}"
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext =>
      let vOut ←
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := mid)
          node params vals
      evalFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss ++ [mid]) (out
        := out)
        gNext params (vals.push vOut)

/--
Evaluate a verified forward fragment program.

This is the top-level evaluator for `Program`: it initializes the context with the input value and
then interprets the SSA let-chain.
-/
def evalForward1
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) : Except String (Tensor α outShape) := do
  let vals0 : Array (DVal α) := #[DVal.mk (α := α) inShape x]
  evalFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out := outShape)
    p params vals0

/-! ## Compilation to verifier IR -/

/-- Flatten a well-formed tensor into the `FlatVec` payload format used by CROWN/LiRPA IR nodes. -/
def flatOfTensor {α : Type} [Context α] {s : Shape}
    (_wf : Shape.WellFormed s)
    (t : Tensor α s) : NN.MLTheory.CROWN.Graph.FlatVec α :=
  { n := Shape.size s, v := Tensor.flattenSpec (α := α) (s := s) t }

/--
Compile a single forward-fragment node into the verifier IR.

Returns the corresponding `NN.IR.Node` together with an updated CROWN `ParamStore` that contains any
payload required by `.const`/`.linear` nodes.
-/
def compileNode
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (id : Nat)
    (node : Node α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    NN.IR.Node × NN.MLTheory.CROWN.Graph.ParamStore α :=
  match node with
  | .const (s := s) wf t =>
      let n : NN.IR.Node := { id := id, parents := [], kind := .const s, outShape := s }
      let ps' := { ps with constVals := ps.constVals.insert id (flatOfTensor (α := α) (s := s) wf t)
        }
      (n, ps')
  | .paramConst (s := s) wf p =>
      let t := getParam (α := α) (paramShapes := paramShapes) params p
      let n : NN.IR.Node := { id := id, parents := [], kind := .const s, outShape := s }
      let ps' := { ps with constVals := ps.constVals.insert id (flatOfTensor (α := α) (s := s) wf t)
        }
      (n, ps')
  | .add (s := s) a b =>
      ({ id := id, parents := [a.id, b.id], kind := .add, outShape := s }, ps)
  | .sub (s := s) a b =>
      ({ id := id, parents := [a.id, b.id], kind := .sub, outShape := s }, ps)
  | .mulElem (s := s) a b =>
      ({ id := id, parents := [a.id, b.id], kind := .mul_elem, outShape := s }, ps)
  | .relu (s := s) x =>
      ({ id := id, parents := [x.id], kind := .relu, outShape := s }, ps)
  | .exp (s := s) x =>
      ({ id := id, parents := [x.id], kind := .exp, outShape := s }, ps)
  | .log (s := s) x =>
      ({ id := id, parents := [x.id], kind := .log, outShape := s }, ps)
  | .inv (s := s) x =>
      ({ id := id, parents := [x.id], kind := .inv, outShape := s }, ps)
  | .matmul2d m _n p a b =>
      ({ id := id
         parents := [a.id, b.id]
         kind := .matmul
         outShape := .dim m (.dim p .scalar) }, ps)
  | .bmm batch m _n p a b =>
      ({ id := id
         parents := [a.id, b.id]
         kind := .matmul
         outShape := .dim batch (.dim m (.dim p .scalar)) }, ps)
  | .reshape inS outS _h x =>
      ({ id := id, parents := [x.id], kind := .reshape inS outS, outShape := outS }, ps)
  | .swap_first_two m n rest x =>
      ({ id := id, parents := [x.id], kind := .swap_first_two, outShape := .dim n (.dim m rest) },
        ps)
  | .transpose3dLastTwo _a _b _c x =>
      ({ id := id, parents := [x.id], kind := .transpose3dLastTwo, outShape := out }, ps)
  | .softmaxLast (s := s) _hRank x =>
      let axis := (Shape.rank s) - 1
      ({ id := id, parents := [x.id], kind := .softmax axis, outShape := s }, ps)
  | .layernorm2d seqLen embedDim _hSeq _hEmb x =>
      ({ id := id
         parents := [x.id]
         kind := .layernorm (axis := 1)
         outShape := .dim seqLen (.dim embedDim .scalar) }, ps)
  | .linear inDim outDim w b x =>
      let wT := getParam (α := α) (paramShapes := paramShapes) params w
      let bT := getParam (α := α) (paramShapes := paramShapes) params b
      let n : NN.IR.Node :=
        { id := id, parents := [x.id], kind := .linear, outShape := .dim outDim .scalar }
      let ps' :=
        { ps with
            linearWB := ps.linearWB.insert id { m := outDim, n := inDim, w := wT, b := bT } }
      (n, ps')
  | .mseLoss (s := _s) yhat target =>
      ({ id := id, parents := [yhat.id, target.id], kind := .mseLoss, outShape := .scalar }, ps)

/--
Compile a forward let-chain into a `CompiledIR` graph.

This threads an accumulator `CompiledIR` that contains:
- the growing `NN.IR.Graph`,
- the payload store (`ParamStore`),
- and the current output id.
-/
def compileFGraph
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α) :
    NN.Verification.TorchLean.CompiledIR α :=
  match g with
  | .ret y =>
      { c with outputId := y.id }
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext =>
      let id := c.graph.nodes.size
      let (n, ps') :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          mid)
          id node params c.ps
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss ++ [mid])
        (out := out)
        gNext params c'

/--
Compile a proved forward fragment program into the verifier IR.

The resulting `CompiledIR` can be executed by the IR evaluator, and we prove (in this file) that
its denotation agrees with `evalForward1`.
-/
def compileVerifiedForward1
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    NN.Verification.TorchLean.CompiledIR α :=
  let input : NN.IR.Node := { id := 0, parents := [], kind := .input, outShape := inShape }
  let c0 : NN.Verification.TorchLean.CompiledIR α :=
    { graph := { nodes := #[input] }, ps := {}, inputId := 0, outputId := 0 }
  compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
    outShape)
    p params c0

/-! ## Correctness theorem (IR evaluation matches DSL evaluation) -/

namespace Correctness

open NN.Verification.TorchLean

  /-- Extract the list of shapes from an array of dynamic values. -/
  private def shapesOfVals {α : Type} [Context α] (vals : Array (DVal α)) : List Shape :=
    vals.toList.map (fun v => v.1)

/-- `shapesOfVals` commutes with pushing an element onto the value array. -/
private theorem shapesOfVals_push {α : Type} [Context α] (vals : Array (DVal α)) (v : DVal α) :
    shapesOfVals (α := α) (vals.push v) = shapesOfVals (α := α) vals ++ [v.1] := by
  simp [shapesOfVals, Array.push, List.concat_eq_append]

/-- Pushing a node onto `g.nodes` does not affect `getNode` for earlier indices. -/
private theorem getNode_push_lt (g : NN.IR.Graph) (n : NN.IR.Node) {i : Nat} (hi : i < g.nodes.size)
  :
    (NN.IR.Graph.getNode (g := { nodes := g.nodes.push n }) i) = NN.IR.Graph.getNode (g := g) i :=
      by
  simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, Array.getElem?_push, hi, Nat.ne_of_lt hi]

/--
Preservation of `Graph.wellFormed` under pushing a new node.

Informal: if `g` is well-formed and `n` has the right id, arity, and parent discipline, then the
extended graph `{nodes := g.nodes.push n}` is well-formed.
-/
private theorem wellFormed_push
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
  private theorem idx_id_lt_length {Γ : List Shape} {s : Shape} (x : Idx Γ s) : x.id < Γ.length :=
    by
    simp [Idx.id]

  /-- Specialized bound for indices into `Ctx inShape ss = inShape :: ss`. -/
  private theorem idx_id_lt_ctxLen {inShape : Shape} {ss : List Shape} {s : Shape}
      (x : Idx (Ctx inShape ss) s) : x.id < ss.length + 1 := by
    simpa [Ctx] using (idx_id_lt_length (x := x))

  /--
  Compiled nodes always satisfy the IR arity check.

  Informal: `compileNode` never produces an operator with the wrong number of parents.
  -/
  private theorem compileNode_hasValidArity
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
  Compiled nodes satisfy `parentsBelow` when compiled at the next fresh id.

  Informal: typed parent indices (`Idx (Ctx inShape ss) _`) ensure parent ids are below the id of
  the newly-pushed node.
  -/
  private theorem compileNode_parentsBelow
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
  | mseLoss yhat target =>
      have hy : yhat.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) yhat
      have ht : target.id < ss.length + 1 := idx_id_lt_ctxLen (inShape := inShape) (ss := ss) target
      have : yhat.id ≤ ss.length ∧ target.id ≤ ss.length :=
        ⟨Nat.lt_succ_iff.mp hy, Nat.lt_succ_iff.mp ht⟩
      simpa [compileNode, NN.IR.Node.parentsBelow, List.all, hy, ht] using this

  /--
  Compilation preserves `Graph.wellFormed`.

  Informal: if the accumulator `c.graph` is well-formed and its size matches the current context
  length, then compiling one more let-chain step yields a well-formed graph.
  -/
  private theorem compileFGraph_wellFormed
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

/-! ### Compiler correctness (forward fragment) -/

private def evalFGraphVals
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (vals : Array (DVal α)) : Except String (Array (DVal α)) := do
  match g with
  | .ret _y =>
      pure vals
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext =>
      let vOut ←
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := mid)
          node params vals
      evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss ++ [mid])
        (out := out)
        gNext params (vals.push vOut)

  /--
  Compiling a let-chain does not change `ps.constVals` entries for keys `< c.graph.nodes.size`.

  Informal: compilation only inserts payload at the fresh node id, so older keys are unchanged.
  -/
  private theorem compileFGraph_ps_constVals_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.constVals.get? k = c.ps.constVals.get? k := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [compileFGraph]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      -- One compilation step pushes a fresh node at `id = c.graph.nodes.size` and only inserts
      -- payload at that id.
      let id := c.graph.nodes.size
      have hk' : k < id := by simpa [id] using hk
      have hk_succ : k < id + 1 := Nat.lt_succ_of_lt hk'
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hps' : ps'.constVals.get? k = c.ps.constVals.get? k := by
        -- `compileNode` only inserts into `constVals` at key `id`, and `k < id`.
        have hidk : id ≠ k := (ne_comm).1 hk'.ne
        cases node <;>
          simp [compileNode, res, ps', Std.HashMap.getElem?_insert,
            beq_eq_false_iff_ne.mpr hidk]
      -- Apply IH to the suffix compilation: keys < `c'.graph.nodes.size` are preserved.
      have hIH :=
        ih (c := c') (hk := by simpa [c', Array.size_push, id] using hk_succ)
      have : (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
        [mid₀]) (out := out₀)
          gNext params c').ps.constVals.get? k = c.ps.constVals.get? k := by
        calc
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
              gNext params c').ps.constVals.get? k
              =
            c'.ps.constVals.get? k := hIH
          _ = c.ps.constVals.get? k := by simpa [c'] using hps'
      simpa [compileFGraph, c', id, res] using this

  /--
  Compiling a let-chain does not change `ps.linearWB` entries for keys `< c.graph.nodes.size`.

  Informal: compilation only inserts linear payload at the fresh node id, so older keys are
    unchanged.
  -/
  private theorem compileFGraph_ps_linearWB_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.linearWB.get? k = c.ps.linearWB.get? k := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [compileFGraph]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      have hk' : k < id := by simpa [id] using hk
      have hk_succ : k < id + 1 := Nat.lt_succ_of_lt hk'
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hps' : ps'.linearWB.get? k = c.ps.linearWB.get? k := by
        -- `compileNode` only inserts into `linearWB` at key `id`, and `k < id`.
        have hidk : id ≠ k := (ne_comm).1 hk'.ne
        cases node <;>
          simp [compileNode, res, ps', Std.HashMap.getElem?_insert,
            beq_eq_false_iff_ne.mpr hidk]
      have hIH :=
        ih (c := c') (hk := by simpa [c', Array.size_push, id] using hk_succ)
      have : (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
        [mid₀]) (out := out₀)
          gNext params c').ps.linearWB.get? k = c.ps.linearWB.get? k := by
        calc
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
              gNext params c').ps.linearWB.get? k
              =
            c'.ps.linearWB.get? k := hIH
          _ = c.ps.linearWB.get? k := by simpa [c'] using hps'
      simpa [compileFGraph, c', id, res] using this

  /--
  `compileFGraph` does not change existing nodes at indices `< c.graph.nodes.size`.

  Informal: compilation only appends nodes, so `getNode` agrees on the prefix.
  -/
  private theorem compileFGraph_getNode_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {i : Nat} (hi : i < c.graph.nodes.size) :
    (NN.IR.Graph.getNode
        (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out)
          g params c).graph) i)
      =
    NN.IR.Graph.getNode (g := c.graph) i := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [compileFGraph]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hi' : i < c'.graph.nodes.size := by
        simpa [c', Array.size_push] using Nat.lt_succ_of_lt hi
      have hNext :
          NN.IR.Graph.getNode
              (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
                ss₀ ++ [mid₀]) (out := out₀)
                gNext params c').graph) i
            =
          NN.IR.Graph.getNode (g := c'.graph) i :=
        ih (c := c') (hi := hi')
      have hPush :
          NN.IR.Graph.getNode (g := c'.graph) i = NN.IR.Graph.getNode (g := c.graph) i := by
        simpa [c', res, id] using getNode_push_lt (g := c.graph) (n := n) (hi := hi)
      simpa [compileFGraph, c', id, res] using Eq.trans hNext hPush

    /-- `compileFGraph` is monotone in `graph.nodes.size` (it only appends nodes). -/
    private theorem compileFGraph_nodesSize_le
        {α : Type} [Context α]
        {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (c : NN.Verification.TorchLean.CompiledIR α) :
      c.graph.nodes.size ≤
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          out)
            g params c).graph.nodes.size := by
    classical
    induction g generalizing c with
    | ret y =>
        simp [compileFGraph]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have h1 : c.graph.nodes.size ≤ c'.graph.nodes.size := by
        simp [c', Array.size_push]
      have h2 : c'.graph.nodes.size ≤
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
            gNext params c').graph.nodes.size := by
        exact ih (c := c')
      simpa [compileFGraph, c', id, res] using Nat.le_trans h1 h2

    /-- Array/list compatibility lemma for looking up shapes produced by `shapesOfVals`. -/
    private lemma shapesOfVals_get?_eq
        {α : Type} [Context α] (vals : Array (DVal α)) (i : Nat) :
        (shapesOfVals (α := α) vals)[i]? = (vals[i]?).map (fun v => v.1) := by
    -- Avoid `simp` loops on `Array.getElem?_eq_toList_get?'`.
    have hToList : vals.toList[i]? = vals[i]? := by
      simp
    -- `List.getElem?_map` reduces the `map` and then we rewrite the list lookup to the array
    -- lookup.
    simp [shapesOfVals, List.getElem?_map, hToList]

  @[simp] private lemma shapesOfVals_length {α : Type} [Context α] (vals : Array (DVal α)) :
      (shapesOfVals (α := α) vals).length = vals.size := by
    simp [shapesOfVals]

  @[simp] private theorem shape_of_vals_of_hShapes
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (DVal α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      (vals[idx.id]!).shape = s := by
    classical
    have hLen : vals.size = (Ctx inShape ss).length := by
      simpa [shapesOfVals_length] using congrArg List.length hShapes
    have hiΓ : idx.id < (Ctx inShape ss).length := idx_id_lt_length (x := idx)
    have hiVals : idx.id < vals.size := by simpa [hLen] using hiΓ
    have hFin : (⟨idx.id, hiΓ⟩ : Fin (Ctx inShape ss).length) = idx.i := by
      apply Fin.ext
      rfl
    have hGetElem : (Ctx inShape ss)[idx.id]'hiΓ = s := by
      -- `l[i]'h` is definitional `l.get ⟨i,h⟩`.
      simpa [Idx.id, List.get, hFin] using idx.h
    have hΓOpt : (Ctx inShape ss)[idx.id]? = some s := by
      have hSome : (Ctx inShape ss)[idx.id]? = some ((Ctx inShape ss)[idx.id]'hiΓ) := by
        simp
      simp [hSome, hGetElem]
    have hShapesAt :
        (shapesOfVals (α := α) vals)[idx.id]? = some s := by
      have hEq : (shapesOfVals (α := α) vals)[idx.id]? = (Ctx inShape ss)[idx.id]? :=
        congrArg (fun l => l[idx.id]?) hShapes
      exact Eq.trans hEq hΓOpt
    -- Convert to an Option statement about the Array lookup.
    have hAt :
        (vals[idx.id]?).map (fun v => v.1) = some s := by
      simpa [shapesOfVals_get?_eq] using hShapesAt
    -- `idx.id < vals.size`, so the Array lookup is `some (vals[idx.id]!)`.
    have hSome : vals[idx.id]? = some (vals[idx.id]!) := by
      simp [getElem?_pos, hiVals]
    -- Extract the shape from the mapped option.
    simpa [hSome] using hAt

    /--
    If node evaluation succeeds under a consistent `shapesOfVals` invariant, the resulting dynamic
      value
    has the expected output shape.

    This is a small “shape preservation” lemma used in the main compiler-correctness proof.
    -/
        private theorem evalNode_ok_shape_of_hShapes
            {α : Type} [Context α] [DecidableEq Shape]
            {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (node : Node α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (vals : Array (DVal α))
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      ∀ {v : DVal α}, evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := out)
            node params vals = Except.ok v → v.1 = out := by
        intro v hv
        classical
        -- We only need the *shape tag* of the produced `DVal`. Avoid unfolding `getVal`: its
        -- internal dependent cast can make `cases hv` brittle. Instead, split on the `getVal`
        -- results (without unfolding it) and reduce the `do`-blocks with `simp`.
        cases node
        case const wf t =>
            simp [evalNode] at hv
            cases hv
            simp
        case paramConst wf p =>
            simp [evalNode] at hv
            cases hv
            simp
        case add a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (DVal.mk (α := α) out (Tensor.addSpec (α := α) ta tb))
                          : Except String (DVal α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case sub a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (DVal.mk (α := α) out (Tensor.subSpec (α := α) ta tb))
                          : Except String (DVal α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case mulElem a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (DVal.mk (α := α) out (Tensor.mulSpec (α := α) ta tb))
                          : Except String (DVal α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case relu x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Activation.reluSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case exp x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Tensor.expSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case log x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                -- Domain discipline: `log` is undefined outside the positive domain. In Lean's
                -- logic, `panic!` reduces to the default inhabitant, so `evalNode` returns
                -- `if allSpec (0 < ·) tx then logSpec tx else default`.
                simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hv
                simp
        case inv x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Tensor.invSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case matmul2d m n p a b =>
            cases hta :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim m (.dim n .scalar)) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb :
                    getVal (α := α) (inShape := inShape) (ss := ss)
                      (s := .dim n (.dim p .scalar)) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    simp [evalNode, hta, htb] at hv
                    cases hv
                    simp
        case bmm batch m n p a b =>
            cases hta :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim batch (.dim m (.dim n .scalar))) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb :
                    getVal (α := α) (inShape := inShape) (ss := ss)
                      (s := .dim batch (.dim n (.dim p .scalar))) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    simp [evalNode, hta, htb] at hv
                    cases hv
                    simp
        -- `Node.reshape` has arguments `(inS outS : Shape) (hSize : size inS = size outS) (x : Idx … inS)`.
        -- Here `outS` is forced to be the branch output `out`, so `cases` introduces `(inS, x, hSize)`.
        case reshape inS x hSize =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := inS) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok
                        (DVal.mk (α := α) out
                          (Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := out) tx hSize))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case swap_first_two m n rest x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := .dim m (.dim n rest)) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok
                        (DVal.mk (α := α) (.dim n (.dim m rest))
                          (Tensor.swapFirstTwoSpec (α := α) (m := m) (n := n) (s := rest) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case transpose3dLastTwo a b c x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim a (.dim b (.dim c .scalar))) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok
                        (DVal.mk (α := α) (.dim a (.dim c (.dim b .scalar)))
                          (Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case softmaxLast _hRank x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Activation.softmaxSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case layernorm2d seqLen embedDim _hSeq _hEmb x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim seqLen (.dim embedDim .scalar)) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq := hv
                simp [evalNode, hx, Bind.bind, Except.bind] at hEq
                cases hEq
                simp
        case linear inDim outDim w b x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := .dim inDim .scalar) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok xT =>
                have hEq := hv
                simp [evalNode, hx, Bind.bind, Except.bind] at hEq
                cases hEq
                simp
        case mseLoss yhat target =>
            -- `mseLoss` is statically typed with a shared parent shape `s`, but `evalNode`
            -- Mirrors the IR semantics and shape-checks dynamically.
            --
            -- Under `hShapes`, both parent `DVal`s must have shape `s`, so the dynamic check is
            -- provably always true. We reduce the dependent `if` using that proof, rather than
            -- eliminating an unreduced `Decidable.rec`.
            rename_i s
            let yV : DVal α := vals[yhat.id]!
            let tV : DVal α := vals[target.id]!
            have hy : yV.shape = s := by
              simpa [yV] using
                shape_of_vals_of_hShapes (α := α) (inShape := inShape) (ss := ss) (s := s)
                  (vals := vals) (idx := yhat) (hShapes := hShapes)
            have ht : tV.shape = s := by
              simpa [tV] using
                shape_of_vals_of_hShapes (α := α) (inShape := inShape) (ss := ss) (s := s)
                  (vals := vals) (idx := target) (hShapes := hShapes)
            -- After unfolding `DVal.shape`, the `evalNode` condition is stated in terms of `.fst`.
            have hCond : yV.fst = tV.fst := by
              -- Bridge through the `shape` equalities coming from `hShapes`.
              simpa [DVal.shape] using (hy.trans ht.symm)
            have hEq := hv
            simp (config := { zeta := true })
              [evalNode, yV, tV, hCond, Except.pure, Pure.pure] at hEq
            cases hEq
            simp

  set_option maxHeartbeats 2000000

    /--
    `denoteAllFrom` for the compiled IR agrees with the forward-fragment evaluator that returns all
    intermediate values.

    Informal: compilation preserves not just the final output, but the entire vector of SSA
    intermediate values (up to the current compilation point).
    -/
    private theorem denoteAllFrom_compileFGraph_eq_evalFGraphVals
        {α : Type} [Context α] [DecidableEq Shape]
        {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    (x : Tensor α inShape)
    (vals : Array (DVal α))
    (hSize : vals.size = c.graph.nodes.size)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    NN.IR.Graph.denoteAllFrom (α := α)
        (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out)
          g params c).graph)
        (payload := payloadOfParamStore (α := α)
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out
            := out)
            g params c).ps)
          (input := DVal.mk (α := α) inShape x)
          (i := c.graph.nodes.size) (vals := vals)
        =
      evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
        out) g params vals := by
    classical
    induction g generalizing c vals with
    | ret y =>
        -- No more nodes: the compiled graph doesn't add nodes, so `denoteAllFrom` returns `vals`.
        simp [compileFGraph, evalFGraphVals, NN.IR.Graph.denoteAllFrom]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      let cOut :=
        compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
          [mid₀]) (out := out₀)
          gNext params c'
      have hLt : id < cOut.graph.nodes.size := by
        have hmono :=
          compileFGraph_nodesSize_le (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c')
        have : id + 1 ≤ cOut.graph.nodes.size := by
          simpa [cOut, c', id, Array.size_push] using hmono
        exact Nat.lt_of_lt_of_le (Nat.lt_succ_self id) this
      -- Rewrite the goal to the one-step expansion and apply IH to the suffix.
      -- The only nontrivial work is showing `evalAt` matches `evalNode` at the fresh id.
      have hConst :
          cOut.ps.constVals.get? id = c'.ps.constVals.get? id :=
        compileFGraph_ps_constVals_get?_lt (α := α) (paramShapes := paramShapes) (inShape :=
          inShape)
          (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      have hLin :
          cOut.ps.linearWB.get? id = c'.ps.linearWB.get? id :=
        compileFGraph_ps_linearWB_get?_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      -- `getNode` at the fresh index is exactly the freshly pushed node.
      have hnId : n.id = id := by
        cases node <;> simp [n, res, compileNode, id]
      have hGetNode : NN.IR.Graph.getNode (g := cOut.graph) id = pure n := by
        have hPres :=
          compileFGraph_getNode_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c')
            (i := id) (hi := by simp [c', id, Array.size_push])
        have hAtPush : NN.IR.Graph.getNode (g := c'.graph) id = pure n := by
          simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, c', id, hnId]
        simpa [cOut] using Eq.trans hPres hAtPush
      -- One-step correctness: IR `evalAt` at the fresh id matches `evalNode`.
      have hStep :
          NN.IR.Graph.evalAt (α := α)
              (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (vals := vals) (i := id)
            =
          evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
            mid₀)
              node params vals := by
        classical
        -- reduce to the freshly pushed IR node, then finish by cases on the source node
        -- (doing this case-by-case avoids `simp` timeouts on the full IR evaluator).
        have expectShape_eq_ok {expected : Shape} (v : DVal α) (h : v.shape = expected) :
            NN.IR.Graph.expectShape (α := α) (expected := expected) v = Except.ok (h ▸ v.tensor) :=
              by
          cases h
          simp [NN.IR.Graph.expectShape, DVal.shape, DVal.tensor]
          rfl
        have getVal_eq_ok {expected : Shape} (idx : Idx (Ctx inShape ss₀) expected)
            (h : (vals[idx.id]!).1 = expected) :
            getVal (α := α) (inShape := inShape) (ss := ss₀) (s := expected) vals idx =
              Except.ok (h ▸ (vals[idx.id]!).snd) := by
          unfold getVal
          simp [DVal.shape]
          rw [dif_pos h]
          rfl
        cases node with
        | const wf t =>
            letI : Shape.WellFormed mid₀ := wf
            -- Show the const payload is present and evaluates back to `t`.
            let flat : NN.MLTheory.CROWN.Graph.FlatVec α :=
              flatOfTensor (α := α) (s := mid₀) wf t
            have hnKind : n.kind = .const mid₀ := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            have hGet' : c'.ps.constVals.get? id = some flat := by
              have hIns : (c.ps.constVals.insert id flat).get? id = some flat := by
                -- Use the `m[k]?` lemma; it is definitionaly `m.get? k`.
                simp
              -- `c'.ps.constVals = c.ps.constVals.insert id flat`.
              simp [c', res, compileNode, ps', flat]
            have hGet : cOut.ps.constVals.get? id = some flat :=
              hConst.trans hGet'
            have hConstPayload :
                (payloadOfParamStore (α := α) cOut.ps).const? id =
                  some { n := flat.n, v := flat.v } := by
              dsimp [payloadOfParamStore]
              -- rewrite the HashMap lookup before `simp` unfolds it
              have hGetElem : cOut.ps.constVals[id]? = some flat := by
                simpa using hGet
              rw [hGetElem]
              simp [flat]
            have hEvalConst :
                NN.IR.Graph.evalConst (α := α)
                    (payload := payloadOfParamStore (α := α) cOut.ps) (id := id) (s := mid₀)
                  =
                Except.ok t := by
              have hUF : unflattenSpec mid₀ t.flattenSpec = t := by
                simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t := t))
              rw [NN.IR.Graph.evalConst, hConstPayload]
              simp [flat, flatOfTensor, NN.IR.Graph.castDimScalar, hUF]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ t) := by
                have hn :
                    n = { id := id, parents := [], kind := .const mid₀, outShape := mid₀ } := by
                  simp [n, res, compileNode]
                have hGetNodeConst :
                    NN.IR.Graph.getNode (g := cOut.graph) id =
                      pure ({ id := id, parents := [], kind := .const mid₀, outShape := mid₀ } :
                        NN.IR.Node) := by
                  simp [hGetNode, hn]
                -- Let `simp` reduce the `do`-block without unfolding `pure`/`bind` (unfolding
                -- `pure`
                -- would block simp’s monad-law reductions).
                simp [NN.IR.Graph.evalAt, hGetNodeConst, hEvalConst, DVal.shape, DVal.tensor,
                  DVal.mk,
                  throw, throwThe, MonadExceptOf.throw]
            have hEvalNode :
                evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out
                  := mid₀)
                    (Node.const wf t) params vals
                  =
                Except.ok (DVal.mk (α := α) mid₀ t) := by
              rfl
            simp [hEvalNode]
            exact hEvalAt
        | paramConst wf p =>
            letI : Shape.WellFormed mid₀ := wf
            -- Same as `const`, but the stored constant comes from `params`.
            let tp : Tensor α mid₀ := getParam (α := α) (paramShapes := paramShapes) params p
            let flat : NN.MLTheory.CROWN.Graph.FlatVec α :=
              flatOfTensor (α := α) (s := mid₀) wf tp
            have hGet' : c'.ps.constVals.get? id = some flat := by
              simp [c', res, compileNode, ps', flat, tp]
            have hGet : cOut.ps.constVals.get? id = some flat :=
              hConst.trans hGet'
            have hConstPayload :
                (payloadOfParamStore (α := α) cOut.ps).const? id =
                  some { n := flat.n, v := flat.v } := by
              dsimp [payloadOfParamStore]
              have hGetElem : cOut.ps.constVals[id]? = some flat := by
                simpa using hGet
              rw [hGetElem]
              rfl
            have hEvalConst :
                NN.IR.Graph.evalConst (α := α)
                    (payload := payloadOfParamStore (α := α) cOut.ps) (id := id) (s := mid₀)
                  =
                Except.ok tp := by
              have hUF : unflattenSpec mid₀ tp.flattenSpec = tp := by
                simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t :=
                  tp))
              rw [NN.IR.Graph.evalConst, hConstPayload]
              simp [flat, flatOfTensor, NN.IR.Graph.castDimScalar, hUF]
              rfl
            have hnKind : n.kind = .const mid₀ := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ tp) := by
                have hn :
                    n = { id := id, parents := [], kind := .const mid₀, outShape := mid₀ } := by
                  simp [n, res, compileNode]
                have hGetNodeConst :
                    NN.IR.Graph.getNode (g := cOut.graph) id =
                      pure ({ id := id, parents := [], kind := .const mid₀, outShape := mid₀ } :
                        NN.IR.Node) := by
                  simp [hGetNode, hn]
                simp [NN.IR.Graph.evalAt, hGetNodeConst, hEvalConst, DVal.shape, DVal.tensor,
                  DVal.mk,
                  throw, throwThe, MonadExceptOf.throw]
            have hEvalNode :
                evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out
                  := mid₀)
                    (Node.paramConst wf p) params vals
                  =
                Except.ok (DVal.mk (α := α) mid₀ tp) := by
              rfl
            simpa [hEvalNode] using hEvalAt
        | add a b =>
            have ha : (vals[a.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := a) (s
                  := mid₀)
            have hb : (vals[b.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := b) (s
                  := mid₀)
            have hnKind : n.kind = .add := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let ta : Tensor α mid₀ := ha ▸ (vals[a.id]!).snd
            let tb : Tensor α mid₀ := hb ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[a.id]!) = Except.ok ta :=
                  by
              simpa [ta] using expectShape_eq_ok (expected := mid₀) (v := vals[a.id]!) ha
            have hExpectB :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[b.id]!) = Except.ok tb :=
                  by
              simpa [tb] using expectShape_eq_ok (expected := mid₀) (v := vals[b.id]!) hb
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals a = Except.ok ta
                  := by
              simpa [ta] using getVal_eq_ok (expected := mid₀) (idx := a) ha
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals b = Except.ok tb
                  := by
              simpa [tb] using getVal_eq_ok (expected := mid₀) (idx := b) hb
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.addSpec (α := α) ta tb)) := by
              have hExpectA' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[a.id]!) =
                    Except.ok ta := by
                simpa [hnOut] using hExpectA
              have hExpectB' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[b.id]!) =
                    Except.ok tb := by
                simpa [hnOut] using hExpectB
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectA', hExpectB']
              cases hnOut
              rfl
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb] using hEvalAt
        | sub a b =>
            have ha : (vals[a.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := a) (s
                  := mid₀)
            have hb : (vals[b.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := b) (s
                  := mid₀)
            have hnKind : n.kind = .sub := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let ta : Tensor α mid₀ := ha ▸ (vals[a.id]!).snd
            let tb : Tensor α mid₀ := hb ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[a.id]!) = Except.ok ta :=
                  by
              simpa [ta] using expectShape_eq_ok (expected := mid₀) (v := vals[a.id]!) ha
            have hExpectB :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[b.id]!) = Except.ok tb :=
                  by
              simpa [tb] using expectShape_eq_ok (expected := mid₀) (v := vals[b.id]!) hb
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals a = Except.ok ta
                  := by
              simpa [ta] using getVal_eq_ok (expected := mid₀) (idx := a) ha
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals b = Except.ok tb
                  := by
              simpa [tb] using getVal_eq_ok (expected := mid₀) (idx := b) hb
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.subSpec (α := α) ta tb)) := by
              have hExpectA' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[a.id]!) =
                    Except.ok ta := by
                simpa [hnOut] using hExpectA
              have hExpectB' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[b.id]!) =
                    Except.ok tb := by
                simpa [hnOut] using hExpectB
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectA', hExpectB']
              cases hnOut
              rfl
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb] using hEvalAt
        | mulElem a b =>
            have ha : (vals[a.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := a) (s
                  := mid₀)
            have hb : (vals[b.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := b) (s
                  := mid₀)
            have hnKind : n.kind = .mul_elem := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let ta : Tensor α mid₀ := ha ▸ (vals[a.id]!).snd
            let tb : Tensor α mid₀ := hb ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[a.id]!) = Except.ok ta :=
                  by
              simpa [ta] using expectShape_eq_ok (expected := mid₀) (v := vals[a.id]!) ha
            have hExpectB :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[b.id]!) = Except.ok tb :=
                  by
              simpa [tb] using expectShape_eq_ok (expected := mid₀) (v := vals[b.id]!) hb
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals a = Except.ok ta
                  := by
              simpa [ta] using getVal_eq_ok (expected := mid₀) (idx := a) ha
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals b = Except.ok tb
                  := by
              simpa [tb] using getVal_eq_ok (expected := mid₀) (idx := b) hb
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.mulSpec (α := α) ta tb)) := by
              have hExpectA' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[a.id]!) =
                    Except.ok ta := by
                simpa [hnOut] using hExpectA
              have hExpectB' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[b.id]!) =
                    Except.ok tb := by
                simpa [hnOut] using hExpectB
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectA', hExpectB']
              cases hnOut
              rfl
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb] using hEvalAt
        | relu xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .relu := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Activation.reluSpec (α := α) tx)) := by
              have hExpectX' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[xIdx.id]!) =
                    Except.ok tx := by
                simpa [hnOut] using hExpectX
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX']
              cases hnOut
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx] using hEvalAt
        | exp xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .exp := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.expSpec (α := α) tx)) := by
              have hExpectX' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[xIdx.id]!) =
                    Except.ok tx := by
                simpa [hnOut] using hExpectX
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX']
              cases hnOut
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx] using hEvalAt
        | log xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .log := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) mid₀
                    (if Tensor.allSpec (α := α) (s := mid₀) (fun v => decide (0 < v)) tx then
                      Tensor.logSpec (α := α) tx
                    else
                      panic!
                        ("IR eval: log: input contains values <= 0 (or NaN); " ++
                          "use `safe_log` if you want epsilon protection"))) := by
              have hExpectX' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[xIdx.id]!) =
                    Except.ok tx := by
                simpa [hnOut] using hExpectX
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX']
              cases hnOut
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx] using hEvalAt
        | inv xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .inv := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.invSpec (α := α) tx)) := by
              have hExpectX' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[xIdx.id]!) =
                    Except.ok tx := by
                simpa [hnOut] using hExpectX
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX']
              cases hnOut
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx] using hEvalAt
        | matmul2d m nDim p a b =>
            have ha : (vals[a.id]!).1 = .dim m (.dim nDim .scalar) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := a) (s := .dim m (.dim nDim .scalar))
            have hb : (vals[b.id]!).1 = .dim nDim (.dim p .scalar) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := b) (s := .dim nDim (.dim p .scalar))
            have haF : (vals[a.id]!).fst = .dim m (.dim nDim .scalar) := by
              simpa using ha
            have hbF : (vals[b.id]!).fst = .dim nDim (.dim p .scalar) := by
              simpa using hb
            have hnKind : n.kind = .matmul := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim m (.dim p .scalar) := by
              simp [compileNode, res, n]
            let ta : Tensor α (.dim m (.dim nDim .scalar)) := haF ▸ (vals[a.id]!).snd
            let tb : Tensor α (.dim nDim (.dim p .scalar)) := hbF ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim m (.dim nDim .scalar)) (vals[a.id]!) =
                  Except.ok ta := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos haF]
              rfl
            have hExpectB :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim nDim (.dim p .scalar)) (vals[b.id]!) =
                  Except.ok tb := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hbF]
              rfl
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim m (.dim nDim .scalar)) vals a =
                  Except.ok ta := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos haF]
              rfl
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim nDim (.dim p .scalar)) vals b =
                  Except.ok tb := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hbF]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim m (.dim p .scalar))
                    (Tensor.matMulSpec (α := α) (m := m) (n := nDim) (p := p) ta tb)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, haF, hbF, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectA, hExpectB]
              split_ifs with hOut
              · cases hOut
                rfl
              · exfalso
                exact hOut (by simp [hnOut])
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb] using hEvalAt
        | bmm batch m nDim p a b =>
            have ha : (vals[a.id]!).1 = .dim batch (.dim m (.dim nDim .scalar)) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := a) (s := .dim batch (.dim m (.dim nDim .scalar)))
            have hb : (vals[b.id]!).1 = .dim batch (.dim nDim (.dim p .scalar)) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := b) (s := .dim batch (.dim nDim (.dim p .scalar)))
            have haF : (vals[a.id]!).fst = .dim batch (.dim m (.dim nDim .scalar)) := by
              simpa using ha
            have hbF : (vals[b.id]!).fst = .dim batch (.dim nDim (.dim p .scalar)) := by
              simpa using hb
            have hnKind : n.kind = .matmul := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim batch (.dim m (.dim p .scalar)) := by
              simp [compileNode, res, n]
            let ta : Tensor α (.dim batch (.dim m (.dim nDim .scalar))) := haF ▸ (vals[a.id]!).snd
            let tb : Tensor α (.dim batch (.dim nDim (.dim p .scalar))) := hbF ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim batch (.dim m (.dim nDim .scalar))) (vals[a.id]!) =
                  Except.ok ta := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos haF]
              rfl
            have hExpectB :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim batch (.dim nDim (.dim p .scalar))) (vals[b.id]!) =
                  Except.ok tb := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hbF]
              rfl
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim batch (.dim m (.dim nDim .scalar))) vals a =
                  Except.ok ta := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos haF]
              rfl
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim batch (.dim nDim (.dim p .scalar))) vals b =
                  Except.ok tb := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hbF]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim batch (.dim m (.dim p .scalar)))
                    (Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := nDim) (p := p) ta tb))
                      := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, haF, hbF, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectA, hExpectB]
              split_ifs with hOut
              · cases hOut
                rfl
              · exfalso
                exact hOut (by simp [hnOut])
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb] using hEvalAt
        | reshape inS mid₀ h xIdx =>
            have hx : (vals[xIdx.id]!).1 = inS := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := inS)
            have hxF : (vals[xIdx.id]!).fst = inS := by
              simpa using hx
            have hnKind : n.kind = .reshape inS mid₀ := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α inS := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := inS) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := inS) vals xIdx = Except.ok tx
                  := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) mid₀
                    (Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := mid₀) tx h)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX]
              simp [h]
              split_ifs with hOut
              · cases hOut
                rfl
              · exfalso
                exact hOut (by simp [hnOut])
            simpa [evalNode, hGetValX, DVal.mk, h, tx] using hEvalAt
        | swap_first_two m nDim rest xIdx =>
            have hx : (vals[xIdx.id]!).1 = .dim m (.dim nDim rest) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim m (.dim nDim rest))
            have hxF : (vals[xIdx.id]!).fst = .dim m (.dim nDim rest) := by
              simpa using hx
            have hnKind : n.kind = .swap_first_two := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim nDim (.dim m rest) := by
              simp [compileNode, res, n]
            let tx : Tensor α (.dim m (.dim nDim rest)) := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := .dim m (.dim nDim rest))
                  (vals[xIdx.id]!) =
                  Except.ok tx := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := .dim m (.dim nDim rest)) vals
                  xIdx =
                  Except.ok tx := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim nDim (.dim m rest))
                    (Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest) tx)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, hnOut, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX]
              simp [hnOut]
            simpa [evalNode, hGetValX, DVal.mk, tx, hnOut] using hEvalAt
        | transpose3dLastTwo a b c xIdx =>
            have hx : (vals[xIdx.id]!).1 = .dim a (.dim b (.dim c .scalar)) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim a (.dim b (.dim c .scalar)))
            have hxF : (vals[xIdx.id]!).fst = .dim a (.dim b (.dim c .scalar)) := by
              simpa using hx
            have hnKind : n.kind = .transpose3dLastTwo := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim a (.dim c (.dim b .scalar)) := by
              simp [compileNode, res, n]
            let tx : Tensor α (.dim a (.dim b (.dim c .scalar))) := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim a (.dim b (.dim c .scalar))) (vals[xIdx.id]!) =
                  Except.ok tx := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim a (.dim b (.dim c .scalar))) vals xIdx =
                  Except.ok tx := by
              unfold getVal
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim a (.dim c (.dim b .scalar)))
                    (Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) tx)) :=
                      by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, hnOut, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX]
              simp [hnOut]
            simpa [evalNode, hGetValX, DVal.mk, tx, hnOut] using hEvalAt
        | softmaxLast hRank xIdx =>
            have hAxis : (Shape.rank mid₀ - 1) + 1 = Shape.rank mid₀ := by
              exact Nat.sub_add_cancel (Nat.succ_le_of_lt hRank)
            have hAxisValid : OpContracts.checkAxisValid (Shape.rank mid₀ - 1) mid₀ = .ok () := by
              unfold OpContracts.checkAxisValid
              have hLt : Shape.rank mid₀ - 1 < Shape.rank mid₀ := by
                cases hR : Shape.rank mid₀ with
                | zero =>
                    simp [hR] at hRank
                | succ r =>
                    simp
              simp [hLt]
              rfl
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hxF : (vals[xIdx.id]!).fst = mid₀ := by
              simpa using hx
            have hnKind : n.kind = .softmax (Shape.rank mid₀ - 1) := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              simpa [tx] using expectShape_eq_ok (expected := mid₀) (v := vals[xIdx.id]!) hxF
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using getVal_eq_ok (expected := mid₀) (idx := xIdx) hxF
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) mid₀
                    (Activation.softmaxSpec (α := α) tx)) := by
              have hAxisValid' :
                  OpContracts.checkAxisValid (Shape.rank mid₀ - 1) n.outShape = Except.ok () := by
                simpa [hnOut] using hAxisValid
              have hExpectX' :
                  NN.IR.Graph.expectShape (α := α) (expected := n.outShape) (vals[xIdx.id]!) =
                    Except.ok tx := by
                simpa [hnOut] using hExpectX
              have hAxis' : (Shape.rank mid₀ - 1) + 1 = Shape.rank n.outShape := by
                simpa [hnOut] using hAxis
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hAxisValid', hExpectX']
              simp [hAxis']
              cases hnOut
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx] using hEvalAt
        | layernorm2d seqLen embedDim hSeq hEmb xIdx =>
            have hParams :
                OpContracts.layerNorm2DParams 1
                    (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) =
                  .ok (seqLen, embedDim) := by
              -- For a `(seqLen × embedDim)` tensor, `axis=1` normalizes the last dimension.
              simp [OpContracts.layerNorm2DParams, OpContracts.checkAxisValid, Shape.toList]
              rfl
            have hx : (vals[xIdx.id]!).1 = .dim seqLen (.dim embedDim .scalar) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim seqLen (.dim embedDim .scalar))
            have hxF : (vals[xIdx.id]!).fst = .dim seqLen (.dim embedDim .scalar) := by
              simpa using hx
            have hnKind : n.kind = .layernorm 1 := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim seqLen (.dim embedDim .scalar) := by
              simp [compileNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .layernorm 1
                     outShape := .dim seqLen (.dim embedDim .scalar) } : NN.IR.Node) := by
              simp [n, res, compileNode]
            have hGetNodeLN :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .layernorm 1
                       outShape := .dim seqLen (.dim embedDim .scalar) } : NN.IR.Node) := by
              simp [hGetNode, hn]
            cases hnOut
            let tx : Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
              hxF ▸ (vals[xIdx.id]!).snd
            have hExpect :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim seqLen (.dim embedDim .scalar)) (vals[xIdx.id]!) =
                  Except.ok tx := by
              simpa [tx] using
                expectShape_eq_ok (expected := .dim seqLen (.dim embedDim .scalar)) (v :=
                  vals[xIdx.id]!) hxF
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim seqLen (.dim embedDim .scalar)) vals xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok (expected := .dim seqLen (.dim embedDim .scalar)) (idx
                := xIdx) hxF
            have hLN :
                NN.IR.Graph.layernormPure (α := α) (seqLen := seqLen) (embedDim := embedDim)
                    (Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      tx rfl)
                  =
                Except.ok
                  (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                    (x := Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      tx rfl)
                    (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
                    (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
                    (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
              simp [NN.IR.Graph.layernormPure, hSeq, hEmb]
              rfl
            have hNumel :
                Shape.size (.dim seqLen (.dim embedDim .scalar)) =
                  Shape.size (.dim seqLen (.dim embedDim .scalar)) := rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim seqLen (.dim embedDim .scalar))
                    (Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                        (x := Tensor.reshapeSpec (α := α)
                          (s₁ := .dim seqLen (.dim embedDim .scalar))
                          (s₂ := .dim seqLen (.dim embedDim .scalar))
                          tx rfl)
                        (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
                        (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
                        (h_seq_pos := hSeq) (h_embed_pos := hEmb))
                      rfl)) := by
                simp [NN.IR.Graph.evalAt, hGetNodeLN, hExpect, hParams,
                  DVal.shape, DVal.tensor, DVal.mk,
                  throw, throwThe, MonadExceptOf.throw]
                simpa [hExpect, hParams, hNumel, DVal.mk] using
                  congrArg
                    (fun e =>
                      (fun a : Tensor α (.dim seqLen (.dim embedDim .scalar)) =>
                        DVal.mk (α := α) (.dim seqLen (.dim embedDim .scalar))
                          (Tensor.reshapeSpec (α := α)
                            (s₁ := .dim seqLen (.dim embedDim .scalar))
                            (s₂ := .dim seqLen (.dim embedDim .scalar))
                            a rfl)) <$> e)
                    hLN
            simpa [evalNode, getVal, DVal.shape, DVal.tensor, hxF, DVal.mk,
              tx, Tensor.reshapeSpec, Tensor.flatten_unflatten_inverse] using hEvalAt
        | linear inDim outDim w b xIdx =>
            let wT : Tensor α (.dim outDim (.dim inDim .scalar)) :=
              getParam (α := α) (paramShapes := paramShapes) params w
            let bT : Tensor α (.dim outDim .scalar) :=
              getParam (α := α) (paramShapes := paramShapes) params b
            have hx : (vals[xIdx.id]!).1 = .dim inDim .scalar := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim inDim .scalar)
            have hxF : (vals[xIdx.id]!).fst = .dim inDim .scalar := by
              simpa using hx
            have hLin' : cOut.ps.linearWB[n.id]? = c'.ps.linearWB[n.id]? := by
              simpa [hnId] using hLin
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .linear
                     outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [n, res, compileNode]
            have hGetNodeLinear :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .linear
                       outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [hGetNode, hn]
            let xT : Tensor α (.dim inDim .scalar) :=
              hxF ▸ (vals[xIdx.id]!).snd
            have hExpectIn :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim inDim .scalar) (vals[xIdx.id]!) =
                  Except.ok xT := by
              simpa [xT] using expectShape_eq_ok (expected := .dim inDim .scalar) (v :=
                vals[xIdx.id]!) hxF
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim inDim .scalar) vals xIdx =
                  Except.ok xT := by
              simpa [xT] using getVal_eq_ok (expected := .dim inDim .scalar) (idx := xIdx) hxF
            have hLinearPayload :
                (payloadOfParamStore (α := α) cOut.ps).linear? id =
                  some { outDim := outDim, inDim := inDim, W := wT, b := bT } := by
                dsimp [payloadOfParamStore]
                rw [show cOut.ps.linearWB[id]? = c'.ps.linearWB[id]? by simpa [hnId] using hLin']
                simp [c', res, compileNode, ps', wT, bT]
            have hEvalLinear :
                NN.IR.Graph.evalLinear (α := α)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (id := id)
                    (x := vals[xIdx.id]!)
                    (outShape := .dim outDim .scalar)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim outDim .scalar)
                    (Tensor.addSpec (α := α)
                      (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT)) := by
              rw [NN.IR.Graph.evalLinear, hLinearPayload]
              simp [wT, bT]
              rw [hExpectIn]
              simp
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim outDim .scalar)
                    (Tensor.addSpec (α := α)
                      (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT)) := by
              -- `evalAt` runs the op-specific evaluator (`evalLinear` here) and then normalizes the
              -- returned value's shape tag to the node's declared `outShape`.
              simp [NN.IR.Graph.evalAt, hGetNodeLinear, hEvalLinear, DVal.shape, DVal.tensor,
                DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              -- `simp` does not reduce `Except` do-notation; specialize the `ok`-bind manually,
              -- then simplify the `if` by reflexivity.
              change (if h : (⟨Shape.dim outDim Shape.scalar,
                        Tensor.addSpec (α := α)
                          (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT⟩ :
                            DVal α).fst =
                      Shape.dim outDim Shape.scalar then _ else _) = _
              simp
              rfl
            simpa [evalNode, getVal, DVal.shape, DVal.tensor, DVal.mk, hxF, xT, wT, bT] using
              hEvalAt
        | mseLoss yhat target =>
            rename_i s
            have hy : (vals[yhat.id]!).1 = s := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := yhat)
                  (s := s)
            have ht : (vals[target.id]!).1 = s := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx :=
                  target) (s := s)
            have hEq : vals[yhat.id]!.fst = vals[target.id]!.fst := by
              simp [hy, ht]
            have hnKind : n.kind = .mseLoss := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [yhat.id, target.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .scalar := by
              simp [compileNode, res, n]
            -- `evalAt` and `evalNode` do the same dynamic check (`yhat.shape = target.shape`) and
            -- then compute
            -- the same scalar MSE.
            simp [NN.IR.Graph.evalAt, NN.IR.Graph.mseLossDVal, hGetNode, hnKind, hnParents, hnOut,
              evalNode, hy, ht, DVal.shape, DVal.tensor, DVal.mk,
              throw, throwThe, MonadExceptOf.throw]
            -- After unfolding, the only difference is the IR normalizer's `v.shape = n.outShape`
            -- cast.
            cases hnOut
            rfl
      -- Unfold both evaluators one step, then dispatch by cases on the shared `evalNode`.
      have hStart :
          NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let v ← NN.IR.Graph.evalAt (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (vals := vals) (i := id)
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push v)) := by
        -- Unfold `denoteAllFrom` once at the top-level; don't simp-recursively unfold the recursive
        -- call.
        rw [NN.IR.Graph.denoteAllFrom.eq_1]
        simp [hLt]
      -- Rewrite the goal using `hStart` and the one-step lemma `hStep`.
      have hStart' :
          NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
              ss₀) (out := mid₀)
              node params vals
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push vOut)) := by
        -- Rewrite `denoteAllFrom` once, then replace `evalAt` with the already-verified `hStep`.
        -- Doing this explicitly avoids `simp` unfolding `DVal.mk` too early, which can prevent
        -- matching on the `hStart` rewrite.
        rw [hStart]
        have hStep' :
            NN.IR.Graph.evalAt (α := α) (g := cOut.graph)
                (payload := payloadOfParamStore (α := α) cOut.ps)
                (input := ⟨inShape, x⟩)
                (vals := vals) (i := id)
              =
            evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out := mid₀)
              node params vals := by
          simpa [DVal.mk] using hStep
        simp [hStep']
      -- Now split on the result of `evalNode`.
      cases hEval : evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
        (out := mid₀)
          node params vals with
        | error e =>
            -- If the next DSL node fails, both evaluators stop immediately with the same error.
            -- First unfold compilation/evaluation one step so the goal is stated in terms of
            -- `cOut`.
            simp [compileFGraph, evalFGraphVals]
            -- The simp step above rewrites the compiled graph to `cOut.graph`, but may unfold
            -- `DVal.mk` to `⟨_,_⟩`. Normalize before rewriting with `hStart'`.
            have hStart'' :
                cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) id vals
                  =
                (do
                  let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                    (ss := ss₀) (out := mid₀) node params vals
                  cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) (id + 1) (vals.push vOut)) := by
              simpa [DVal.mk] using hStart'
            rw [hStart'']
            simp [hEval]
            rfl
        | ok vOut =>
          have hSize' : (vals.push vOut).size = c'.graph.nodes.size := by
            simp [c', hSize, Array.size_push]
          have hvOutShape : vOut.1 = mid₀ :=
            evalNode_ok_shape_of_hShapes (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := ss₀) (out := mid₀) node params vals hShapes (v := vOut) (by simp [hEval])
          have hShapes' : shapesOfVals (α := α) (vals.push vOut) = Ctx inShape (ss₀ ++ [mid₀]) := by
            calc
              shapesOfVals (α := α) (vals.push vOut)
                  = shapesOfVals (α := α) vals ++ [vOut.1] := shapesOfVals_push (α := α) (vals :=
                    vals) (v := vOut)
              _ = Ctx inShape ss₀ ++ [vOut.1] := by simp [hShapes]
              _ = (inShape :: ss₀) ++ [mid₀] := by simp [Ctx, hvOutShape]
              _ = Ctx inShape (ss₀ ++ [mid₀]) := by simp [Ctx, List.cons_append]
          have hIH :=
            ih (c := c') (vals := vals.push vOut) (hSize := hSize') (hShapes := hShapes')
          -- Rewrite the overall goal to the suffix goal (start at `id+1`), then discharge with IH.
          simp [compileFGraph, evalFGraphVals]
          have hStart'' :
              cOut.graph.denoteAllFrom
                  (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) id vals
                =
              (do
                let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                  (ss := ss₀) (out := mid₀) node params vals
                cOut.graph.denoteAllFrom
                  (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) (id + 1) (vals.push vOut)) := by
            simpa [DVal.mk] using hStart'
          rw [hStart'']
          simp [hEval]
          -- now the goal is exactly the suffix IH (start index is `id+1 = c'.graph.nodes.size`).
          -- `DVal.mk` is definitional `⟨_,_⟩`, but the pretty-printer may choose either form; normalize
          -- before applying the IH.
          simpa [c', id, DVal.mk] using hIH

  set_option maxHeartbeats 200000

    /-- `Array` indexing is proof-irrelevant: the bounds proof does not affect the element returned.
      -/
    private theorem array_getElem_proof_irrel
        {β : Type} (a : Array β) (i : Nat) (h₁ h₂ : i < a.size) :
        a[i]'h₁ = a[i]'h₂ := by
      -- The bounds proof lives in `Prop`, so it is proof-irrelevant.
      have : h₁ = h₂ := Subsingleton.elim _ _
      cases this
      rfl

    /-- Casting along an equality is proof-irrelevant: the proof does not affect the result. -/
    private theorem cast_proof_irrel
        {β : Sort _} {a b : β} {P : β → Sort _}
        (h₁ h₂ : a = b) (x : P a) :
        (h₁ ▸ x) = (h₂ ▸ x) := by
      cases h₁
      cases h₂
      rfl

  /-!
  Helper functions for the final "compiled forward = DSL forward" theorem.

  - `finalSs g` is the list of available value shapes at the point where `g` returns.
    (It is the `ss` parameter of the `.ret` constructor reached by running through `.let1`.)
  - `outIdx g` is the return index of `g`, but expressed at the `finalSs g` context.
  -/

  private def finalSs
      {α : Type} {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape} :
      FGraph α paramShapes inShape ss out → List Shape
    | .ret _y => ss
    | .let1 _node gNext => finalSs gNext

  /--
  Return index of a forward let-chain, expressed in the *final* context.

  As we traverse `.let1` nodes, the local context `ss` grows; this function returns the output index
  at the end of the chain (`finalSs g`), so it can be used with the final `vals` array produced by
  `evalFGraphVals`.
  -/
  private def outIdx
      {α : Type} {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape} :
      (g : FGraph α paramShapes inShape ss out) → Idx (Ctx inShape (finalSs g)) out
    | .ret y => y
    | .let1 _node gNext => outIdx gNext

  /--
  The compiled graph's `outputId` agrees with the return index `outIdx` of the source let-chain.

  Informal: `compileFGraph` records as output exactly the node index returned by the `.ret` case
  after threading through the `.let1` chain.
  -/
  private theorem compileFGraph_outputId_eq_outIdx_id
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (c : NN.Verification.TorchLean.CompiledIR α) :
      (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
        out)
          g params c).outputId
        =
      (outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := out)
        g).id := by
    classical
    induction g generalizing c with
    | ret y =>
        simp [compileFGraph, outIdx]
        rfl
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        simp [compileFGraph, outIdx, ih]
        rfl

  /--
  `evalFGraph` is `evalFGraphVals` followed by selecting the return index `outIdx`.

  This isolates “evaluate all SSA values” from “pick the output tensor”, which is useful in the
    final
  correctness statement.
  -/
  private theorem evalFGraph_eq_evalFGraphVals_outIdx
      {α : Type} [Context α] [DecidableEq Shape]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (vals : Array (DVal α)) :
      evalFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := out)
        g params vals
        =
      (do
        let vals' ←
          evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out
            := out)
            g params vals
        let v : DVal α := vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g).id]!
        if h : v.shape = out then
          pure (h ▸ v.tensor)
        else
          throw s!"TorchLeanVerified: expected shape {repr out}, got {repr v.shape}") := by
    classical
    induction g generalizing vals with
    | ret y =>
        -- Definitional: `evalFGraphVals (.ret _) = pure vals`.
        rfl
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        cases hNode :
            evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
              mid₀)
              node params vals with
        | error e =>
            -- Short-circuiting on `Except.error` makes both sides definitional.
            simp [evalFGraph, evalFGraphVals, outIdx, hNode]
            rfl
        | ok vOut =>
            have hIH := ih (vals := vals.push vOut)
            -- Reduce the outer `evalNode` bind and then apply the IH on the extended `vals`.
            simpa [evalFGraph, evalFGraphVals, outIdx, hNode, Pure.pure, Except.pure, Except.bind,
              Except.instMonad]
              using hIH

  /--
  Shape-invariant for `evalFGraphVals`.

  If the input value array has shapes `Ctx inShape ss`, then the result array has shapes
  `Ctx inShape (finalSs g)` at the return point.
  -/
  private theorem evalFGraphVals_shapes_of_hShapes
      {α : Type} [Context α] [DecidableEq Shape]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (vals vals' : Array (DVal α))
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
      (hOk :
        evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          out) g params vals =
          Except.ok vals') :
      shapesOfVals (α := α) vals' = Ctx inShape (finalSs g) := by
    classical
    induction g generalizing vals vals' with
    | ret y =>
        simp [evalFGraphVals] at hOk
        cases hOk
        simpa [finalSs]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        -- Unfold once and split on `evalNode`.
        cases hNode :
            evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
              mid₀)
              node params vals with
        | error e =>
            -- impossible: `hOk` claims the whole computation returned `ok`.
            simp [evalFGraphVals, hNode] at hOk
            cases hOk
        | ok vOut =>
            have hvOutShape : vOut.1 = mid₀ :=
              evalNode_ok_shape_of_hShapes (α := α) (paramShapes := paramShapes) (inShape :=
                inShape)
                (ss := ss₀) (out := mid₀) node params vals hShapes (v := vOut) (by simp [hNode])
            have hShapes' : shapesOfVals (α := α) (vals.push vOut) = Ctx inShape (ss₀ ++ [mid₀]) :=
              by
              calc
                shapesOfVals (α := α) (vals.push vOut)
                    = shapesOfVals (α := α) vals ++ [vOut.1] :=
                      shapesOfVals_push (α := α) (vals := vals) (v := vOut)
                _ = Ctx inShape ss₀ ++ [vOut.1] := by simp [hShapes]
                _ = Ctx inShape (ss₀ ++ [mid₀]) := by simp [Ctx, hvOutShape, List.cons_append]
            have hOk' :
                evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀
                  ++ [mid₀]) (out := out₀)
                    gNext params (vals.push vOut)
                  =
                Except.ok vals' := by
              simpa [evalFGraphVals, hNode] using hOk
            -- Apply IH to the suffix.
            simpa [finalSs, evalFGraphVals, hNode] using
              ih (vals := vals.push vOut) (vals' := vals') (hShapes := hShapes') (hOk := hOk')

  /--
  **Main compiler correctness theorem (verified forward fragment).**

  In words: compiling a first-order forward program `p` into the verifier IR and then
  evaluating the compiled graph yields the same output as directly evaluating `p` with
  `evalForward1`.
  -/
  theorem evalCompiledForward1_compileVerifiedForward1_eq_evalForward1
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) :
    evalCompiledForward1 (α := α) (inShape := inShape) (outShape := outShape)
        (c := compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape
          := outShape) p params)
        x
      =
    evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
      params x := by
  classical
  -- Evaluate `compileVerifiedForward1` via the IR semantics, and rewrite it to the DSL evaluator.
  let inputVal : DVal α := DVal.mk (α := α) inShape x
  let inputNode : NN.IR.Node := { id := 0, parents := [], kind := .input, outShape := inShape }
  let c0 : NN.Verification.TorchLean.CompiledIR α :=
    { graph := { nodes := #[inputNode] }, ps := {}, inputId := 0, outputId := 0 }
  have hWF :
      (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
        outShape) p params).graph.wellFormed = true :=
    compileVerifiedForward1_wellFormed (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape
      := outShape) p params
  -- Unfold the compiled evaluator down to the IR `denoteAllFrom` suffix, then apply the correctness
  -- lemma.
  -- The input node is always id=0, so we start the suffix evaluation at `i=1` with
  -- `vals=[inputVal]`.
  have hDenote :
      (NN.IR.Graph.denoteAllFrom (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape) p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := inputVal)
          (i := 1) (vals := #[inputVal]))
        =
      evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
        outShape) p params #[inputVal] := by
    -- `compileVerifiedForward1` is `compileFGraph` starting from `c0`, so apply the lemma at `c=c0` and
    -- `vals=[inputVal]`.
    have hSize0 : (#[inputVal] : Array (DVal α)).size = c0.graph.nodes.size := by
      simp [c0]
    have hShapes0 : shapesOfVals (α := α) (#[inputVal] : Array (DVal α)) = Ctx inShape [] := by
      simp [shapesOfVals, Ctx, inputVal, DVal.mk]
    simpa [compileVerifiedForward1, c0, inputNode] using
      denoteAllFrom_compileFGraph_eq_evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape
        := inShape) (ss := []) (out := outShape)
        (g := p) (params := params) (c := c0) (x := x) (vals := #[inputVal]) hSize0 hShapes0
  -- Unfold both front-end evaluators, then rewrite both sides to a shared `evalFGraphVals`
  -- computation.
  have hOutId :
      (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
        outShape) p params).outputId
        =
      (outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
        outShape) p).id := by
    simpa [compileVerifiedForward1] using
      compileFGraph_outputId_eq_outIdx_id (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := []) (out := outShape) p params c0

  -- Rewrite `Graph.denoteAll` to start at `i=1` with the already-evaluated input.
  have hDenoteAll0 :
      NN.IR.Graph.denoteAll (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape) p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := inputVal)
        =
      NN.IR.Graph.denoteAllFrom (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape) p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := inputVal) (i := 1) (vals := #[inputVal]) := by
    -- `denoteAll` runs `denoteAllFrom` from `i=0` with `vals=[]`.
    simp (config := { zeta := false }) [NN.IR.Graph.denoteAll, hWF]
    -- Unfold one step at `i=0`: the `input` node deterministically yields `inputVal`.
    have h0 :
        (0 : Nat) <
          (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
            outShape)
              p params).graph.nodes.size := by
      have h0c : (0 : Nat) < c0.graph.nodes.size := by
        simp [c0]
      have hLe :
          c0.graph.nodes.size ≤
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape)
                p params).graph.nodes.size := by
        simpa [compileVerifiedForward1] using
          compileFGraph_nodesSize_le (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := []) (out := outShape) (g := p) (params := params) (c := c0)
      exact Nat.lt_of_lt_of_le h0c hLe
    have hGet0 :
        NN.IR.Graph.getNode
            (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape)
                p params).graph)
            0
          =
        pure inputNode := by
      have hi : (0 : Nat) < c0.graph.nodes.size := by
        simp [c0]
      -- `compileVerifiedForward1` is `compileFGraph` starting from `c0`; indices < `c0`'s size are
      -- preserved.
      simpa [compileVerifiedForward1, c0, inputNode] using
        compileFGraph_getNode_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) (g := p) (params := params) (c := c0) (i := 0) (hi := hi)
    have hEval0 :
        NN.IR.Graph.evalAt (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape)
              p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := inputVal) (vals := #[]) (i := 0)
        =
        Except.ok inputVal := by
      -- `getNode 0` returns the input node, and the `.input` branch deterministically returns
      -- `inputVal`.
      simp [NN.IR.Graph.evalAt, hGet0, inputNode, inputVal, NN.IR.Graph.expectShape,
        DVal.shape, DVal.mk, DVal.tensor,
        Bind.bind, Pure.pure, Except.pure, Except.bind]
    rw [NN.IR.Graph.denoteAllFrom.eq_1, dif_pos h0]
    rw [hEval0]
    simp
    have hPushEq : (#[].push inputVal : Array (DVal α)) = #[inputVal] := by
      rfl
    exact congrArg
      (fun vals =>
        NN.IR.Graph.denoteAllFrom (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape) p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := inputVal) (i := 1) (vals := vals))
      hPushEq

  -- Now both `evalCompiledForward1` and `evalForward1` can be expressed via `evalFGraphVals`.
  -- We finish by case-splitting on `evalFGraphVals` and simplifying the shared `if`/lookup logic.
  -- (The "out of bounds" / "shape mismatch" branches are unreachable under the well-typedness
  -- invariants we establish.)
  have hEvalForward :
      evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape)
        p params x
        =
      (do
        let vals' ←
          evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
            := outShape) p params #[inputVal]
        let v : DVal α := vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) p).id]!
        if h : v.shape = outShape then
          pure (h ▸ v.tensor)
        else
          throw s!"TorchLeanVerified: expected shape {repr outShape}, got {repr v.shape}") := by
    simpa [evalForward1, inputVal] using
      (evalFGraph_eq_evalFGraphVals_outIdx (α := α) (paramShapes := paramShapes) (inShape :=
        inShape)
        (ss := []) (out := outShape) (g := p) (params := params) (vals := #[inputVal]))

  -- Rewrite the RHS, and unfold the compiled evaluator down to the same `evalFGraphVals`.
  rw [hEvalForward]
  rw [NN.Verification.TorchLean.evalCompiledForward1, NN.IR.Graph.denote, hOutId]

  -- The remaining LHS still mentions IR `denoteAll`; rewrite it using the established equalities.
  have hDenoteAll0' :
      NN.IR.Graph.denoteAll (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape) p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := DVal.mk (α := α) inShape x)
        =
      NN.IR.Graph.denoteAllFrom (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape) p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := DVal.mk (α := α) inShape x) (i := 1) (vals := #[DVal.mk (α := α) inShape x]) :=
            by
    simpa [inputVal] using hDenoteAll0
  have hDenote' :
      NN.IR.Graph.denoteAllFrom (α := α)
          (g := (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (outShape := outShape) p params).graph)
          (payload := payloadOfParamStore (α := α)
            (compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape) p params).ps)
          (input := DVal.mk (α := α) inShape x) (i := 1) (vals := #[DVal.mk (α := α) inShape x])
        =
      evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
        outShape) p params
        #[DVal.mk (α := α) inShape x] := by
    simpa [inputVal, Except.bind, Except.pure, Pure.pure] using hDenote
  -- Rewrite `Graph.denoteAll` → `denoteAllFrom` → `evalFGraphVals` on the compiled side.
  rw [hDenoteAll0', hDenote']

  -- Now both sides bind the same `evalFGraphVals`; split on that result.
  cases hVals :
      evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
        outShape) p params #[inputVal] with
  | error e =>
      -- Both sides are `Except.error e` because the first monadic bind fails.
      simp [Bind.bind, Except.bind]
  | ok vals' =>
      -- Establish that the output index is in-bounds and has the expected shape.
      have hShapes' :
          shapesOfVals (α := α) vals' = Ctx inShape (finalSs (α := α) (paramShapes := paramShapes)
            (inShape := inShape)
            (ss := []) (out := outShape) p) := by
        exact evalFGraphVals_shapes_of_hShapes (α := α) (paramShapes := paramShapes) (inShape :=
          inShape)
          (ss := []) (out := outShape) (g := p) (params := params) (vals := #[inputVal]) (vals' :=
            vals')
          (hShapes := by simp [shapesOfVals, Ctx, inputVal, DVal.mk]) (hOk := by simp [hVals])
      have hOutShape' :
          (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
            := outShape) p).id]!).shape =
            outShape := by
        simpa [DVal.shape] using
          shape_of_vals_of_hShapes (α := α) (vals := vals')
            (idx := outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := [])
              (out := outShape) p)
            (hShapes := hShapes')
      have hOutLt :
          (outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
            outShape) p).id < vals'.size := by
        have hLen : vals'.size = (Ctx inShape (finalSs (α := α) (paramShapes := paramShapes)
          (inShape := inShape)
              (ss := []) (out := outShape) p)).length := by
          have := congrArg List.length hShapes'
          simpa [shapesOfVals_length] using this
        have hIdx :
            (outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
              outShape) p).id <
              (Ctx inShape (finalSs (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss
                := []) (out := outShape) p)).length :=
          idx_id_lt_length (x := outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := []) (out := outShape) p)
        simpa [hLen] using hIdx
      -- In-bounds array lookup: `get?` returns `some (get!)`.
      have hOutSome :
          vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
            := outShape) p).id]? =
            some (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
              []) (out := outShape) p).id]!) := by
        simp [getElem?_pos, hOutLt]
      have hCond :
          (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
            := outShape) p).id]!).fst =
            outShape := by
        simpa [DVal.shape] using hOutShape'
      -- Rewrite the output lookup to the known in-bounds value and eliminate the dead
      -- shape-mismatch branch.
      cases hGet :
          vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
            := outShape) p).id]? with
      | none =>
          have hEq :
              (none : Option (DVal α)) =
                some (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
                  []) (out := outShape) p).id]!) := by
            simp [hGet] at hOutSome
          cases hEq
      | some out =>
          have hOutEq :
              out =
                vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := [])
                  (out := outShape) p).id]! := by
            simp [hGet] at hOutSome
            exact hOutSome
          subst out
          -- Both sides take the successful `get?` branch and the successful shape check under
          -- `hCond`.
          simp [hGet, hCond, DVal.shape, DVal.tensor, Bind.bind, Pure.pure, Except.pure, Except.bind]

end Correctness

/-- Compile a verified single-input forward program into the verifier IR. -/
abbrev compileForward1
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    NN.Verification.TorchLean.CompiledIR α :=
  compileVerifiedForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params

/-- Graph structural safety for the concise compiler name. -/
theorem compileForward1_wellFormed
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    (compileForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params).graph.wellFormed = true :=
  Correctness.compileVerifiedForward1_wellFormed (α := α) (paramShapes := paramShapes) (inShape := inShape)
    (outShape := outShape) p params

/-- Main end-to-end compiler correctness using the short name. -/
theorem evalCompiledForward1_eq_evalForward1
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) :
    NN.Verification.TorchLean.evalCompiledForward1 (α := α) (inShape := inShape) (outShape := outShape)
        (c := compileForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
          params)
        x
      =
    NN.Verification.TorchLean.Proved.evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
      (outShape := outShape) p params x :=
  Correctness.evalCompiledForward1_compileVerifiedForward1_eq_evalForward1 (α := α)
    (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p params x

/-- Convenience spelling for the same short correctness lemma. -/
theorem evalCompiledForward1_forward1_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) :
    NN.Verification.TorchLean.evalCompiledForward1 (α := α) (inShape := inShape) (outShape := outShape)
        (c := compileForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
          params)
        x
      =
    NN.Verification.TorchLean.Proved.evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
      (outShape := outShape) p params x :=
  evalCompiledForward1_eq_evalForward1 (α := α) (paramShapes := paramShapes) (inShape := inShape)
    (outShape := outShape) p params x

end NN.Verification.TorchLean.Proved
