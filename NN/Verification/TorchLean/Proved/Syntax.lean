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
# Verified Forward Fragment: Syntax And Evaluation

The first-order forward language used by the TorchLean verifier bridge, together with its direct
value evaluator over the scalar semantics selected by the caller.
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
  /-- Position in the context. -/
  i : Fin Γ.length
  /-- Proof that the context entry at `i` has shape `s`. -/
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
  | conv2d (inC outC kH kW stride padding inH inW : Nat)
      (hIn : inC ≠ 0) (hKH : kH ≠ 0) (hKW : kW ≠ 0)
      (hStride : stride ≠ 0)
      (hHeight : OpContracts.checkWindowFits "conv2d" "height" inH kH padding = .ok ())
      (hWidth : OpContracts.checkWindowFits "conv2d" "width" inW kW padding = .ok ())
      (kernel : Idx paramShapes (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
      (bias : Idx paramShapes (.dim outC .scalar))
      (x : Idx (Ctx inShape ss) (.dim inC (.dim inH (.dim inW .scalar)))) :
      Node α paramShapes inShape ss
        (.dim outC
          (.dim ((inH + 2 * padding - kH) / stride + 1)
            (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))
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

/-- Read a dynamic value from the executable context with a user-facing bounds error. -/
def getDVal? {α : Type} [Context α] (vals : Array (DVal α)) (idx : Nat) :
    Except String (DVal α) :=
  match vals[idx]? with
  | some v => .ok v
  | none =>
      .error s!"TorchLeanVerified: value index {idx} out of bounds for context of size {vals.size}"

/--
Read a previously computed dynamic value and cast it back to the statically expected shape.

The verified fragment constructs only well-scoped indices, but the executable evaluator stores values
in an array, so this check gives a clear error if an implementation bug ever violates the shape
discipline.
-/
def getVal {α : Type} [Context α] [DecidableEq Shape]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (DVal α)) (idx : Idx (Ctx inShape ss) s) : Except String (Tensor α s) := do
  let v : DVal α ← getDVal? vals idx.id
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
  | .conv2d inC outC kH kW stride padding inH inW hIn hKH hKW _hStride _hHeight _hWidth kernel bias x =>
      let kT := getParam (α := α) (paramShapes := paramShapes) params kernel
      let bT := getParam (α := α) (paramShapes := paramShapes) params bias
      let xT ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim inC (.dim inH (.dim inW .scalar))) vals x
      let spec : Spec.Conv2DSpec inC outC kH kW stride padding α hIn hKH hKW :=
        { kernel := kT, bias := bT }
      let y := Spec.conv2dSpec (α := α) (layer := spec) (input := xT)
      pure <| DVal.mk (α := α)
        (.dim outC
          (.dim ((inH + 2 * padding - kH) / stride + 1)
            (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))) y
  | .mseLoss (s := _s) yhat target =>
      -- Mirror the IR semantics: `mse_loss` is dynamically shape-checked (both parents must have
      -- equal shape),
      -- then reduces to a scalar by averaging the squared error.
      let yV : DVal α ← getDVal? vals yhat.id
      let tV : DVal α ← getDVal? vals target.id
      if h : yV.shape = tV.shape then
        let yT : Tensor α yV.shape := yV.tensor
        let tT : Tensor α yV.shape := h.symm ▸ tV.tensor
        let s := yV.shape
        let diff := Tensor.subSpec (α := α) yT tT
        let sq := Tensor.mulSpec (α := α) diff diff
        let total : α := Tensor.sumSpec (α := α) sq
        let mean : α := total / (↑(NN.IR.Graph.meanDenom s) : α)
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
      let v : DVal α ← getDVal? vals y.id
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

end NN.Verification.TorchLean.Proved
