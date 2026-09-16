/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Lowering -- shake: keep
public import NN.Verification.Builtin.Correctness -- shake: keep

/-!
# Verified Forward Fragment: Syntax And Evaluation

The first-order forward language used by the TorchLean verifier bridge, together with its direct
value evaluator over the scalar semantics selected by the caller.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

-- Make projection out of dynamic values definitional for `simp` in correctness proofs.
/-- Projecting the tensor from a freshly constructed dynamic value is definitionally exact. -/
@[simp] theorem dval_tensor_mk
    {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Spec.SomeTensor.tensor (α := α) (⟨s, t⟩ : Spec.SomeTensor α) = t := rfl

/-- Shape checking succeeds for a dynamic value constructed with the expected shape tag. -/
@[simp] theorem graph_expectShape_mk
    {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Graph.expectShape (α := α) (expected := s) (Spec.SomeTensor.mk (α := α) s t) = .ok t := by
  simp [Graph.expectShape, Pure.pure, Except.pure]

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
Fetch a parameter tensor from a `TensorPack`, using a typed index `Idx`.

This is the bridge between the parameter context `paramShapes` and the strongly-typed tensor value
returned at shape `s`.
-/
def getParam {α : Type} [TorchLean.Storage α] {paramShapes : List Shape} {s : Shape}
    (params : TorchLean.TensorPack α paramShapes) (idx : Idx paramShapes s) : Tensor α s :=
  Tensor.castShape (_root_.TorchLean.TensorPack.get params idx.i) idx.h

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

/-- Typed evidence for matrix multiplication over an arbitrary shared leading shape. -/
inductive MatmulOperation : Shape → Shape → Shape → Type where
  /-- Multiply the final two axes independently at every index of `leading`. -/
  | leading (leading : List Nat) (m n p : Nat) :
      MatmulOperation ((Shape.ofList leading).concat [m, n])
        ((Shape.ofList leading).concat [n, p]) ((Shape.ofList leading).concat [m, p])

/-- Typed denotation of a supported matrix multiplication operation. -/
def MatmulOperation.denote
    {α : Type} [TorchLean.Storage α] [Context α] {leftShape rightShape outShape : Shape}
  (op : MatmulOperation leftShape rightShape outShape)
    (left : Tensor α leftShape) (right : Tensor α rightShape) : Tensor α outShape :=
  match op with
  | .leading batchAxes m n p =>
      NN.IR.Graph.matmulLeading (α := α) (Shape.ofList batchAxes)
        (m := m) (n := n) (p := p) left right

/--
Typed evidence for LayerNorm over a suffix of an arbitrary tensor shape.

`rows × width` is an evaluation view obtained by flattening the dimensions before and after
`axis`; it is not a restriction on the rank or layout of the input tensor.
-/
structure LayerNormOperation (s : Shape) where
  /-- First axis included in the normalized suffix. -/
  axis : Nat
  /-- Number of independent rows in the flattened evaluation view. -/
  rows : Nat
  /-- Number of entries normalized in each row. -/
  width : Nat
  /-- The generic IR shape contract computes the recorded matrix dimensions. -/
  matrixDims : OpContracts.layerNormMatrixDims axis s = .ok (rows, width)
  /-- Reshaping to the matrix view preserves the number of elements. -/
  size_eq : Shape.size s = Shape.size (.dim rows (.dim width .scalar))
  /-- Every evaluation view has at least one row. -/
  rows_pos : 0 < rows
  /-- The normalized suffix is nonempty. -/
  width_pos : 0 < width

/--
A well-typed SSA node in the verified forward fragment.

Each `Node` can only reference earlier values (via `Idx (Ctx inShape ss) _`), ensuring the DAG/SSA
discipline by construction.

The constructors match the operator subset for which this file proves lowering correctness into the
verifier IR (`NN.IR.Graph`).  Adding a new operator means extending both this syntax and the
correctness proof, which keeps the trusted fragment explicit.
-/
inductive Node
    (α : Type) [TorchLean.Storage α]
    (paramShapes : List Shape) (inShape : Shape) (ss : List Shape) :
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
  | matmul {leftShape rightShape outShape : Shape}
      (op : MatmulOperation leftShape rightShape outShape)
      (a : Idx (Ctx inShape ss) leftShape)
      (b : Idx (Ctx inShape ss) rightShape) :
      Node α paramShapes inShape ss outShape
  | reshape (inS outS : Shape) (h : Spec.Shape.size inS = Spec.Shape.size outS)
      (x : Idx (Ctx inShape ss) inS) :
      Node α paramShapes inShape ss outS
  | transpose {s out : Shape} (axis₁ axis₂ : Nat)
      (hOut : OpContracts.inferTransposeOutShape axis₁ axis₂ s = .ok out)
      (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss out
  | softmax {s : Shape} (axis : Nat) (hAxis : Shape.AxisInBounds axis s)
      (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | layerNorm {s : Shape} (op : LayerNormOperation s) (x : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss s
  | linear (inDim outDim : Nat)
      (w : Idx paramShapes (.dim outDim (.dim inDim .scalar)))
      (b : Idx paramShapes (.dim outDim .scalar))
      (x : Idx (Ctx inShape ss) (.dim inDim .scalar)) :
      Node α paramShapes inShape ss (.dim outDim .scalar)
  | conv {d : Nat} (inC outC : Nat)
      (kernelShape stride padding inSpatial : TorchLean.Tensor Nat [d])
      (hIn : inC ≠ 0)
      (hKernel : ∀ i : Fin d, kernelShape.getScalar i ≠ 0)
      (hStride : ∀ i : Fin d, stride.getScalar i ≠ 0)
      (hInfer : OpContracts.inferConvOutShape "conv" 0 inC outC
        kernelShape stride padding (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))) =
          .ok (Shape.ofList
            (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernelShape stride padding)
              (List Nat))))
      (kernel : Idx paramShapes
        (Shape.ofList (outC :: inC :: (Tensor.to kernelShape (List Nat)))))
      (bias : Idx paramShapes (.dim outC .scalar))
      (x : Idx (Ctx inShape ss) (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))) :
      Node α paramShapes inShape ss
        (Shape.ofList
          (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernelShape stride padding)
            (List Nat)))
  | mseLoss {s : Shape} (yhat target : Idx (Ctx inShape ss) s) :
      Node α paramShapes inShape ss .scalar

/-! ## Forward programs (closed let-chains) -/

/--
Well-typed first-order programs, represented as a forward “let chain”.

The type parameter `ss` tracks the list of already-produced node output shapes, so every node
can only reference earlier values (including the distinguished input at index `0`).
-/
inductive ForwardLetChain (α : Type) [TorchLean.Storage α]
    (paramShapes : List Shape) (inShape : Shape) :
    List Shape → Shape → Type where
  | ret {ss : List Shape} {out : Shape} (y : Idx (Ctx inShape ss) out) :
      ForwardLetChain α paramShapes inShape ss out
  | let1 {ss : List Shape} {mid out : Shape} :
      Node α paramShapes inShape ss mid →
      ForwardLetChain α paramShapes inShape (ss ++ [mid]) out →
      ForwardLetChain α paramShapes inShape ss out

/-- A closed program in the proved forward fragment, from `inShape` to `outShape`. -/
abbrev ForwardProgram (α : Type) [TorchLean.Storage α]
    (paramShapes : List Shape) (inShape outShape : Shape) : Type :=
  ForwardLetChain α paramShapes inShape [] outShape

/-! ## Evaluation -/

/-- Read a dynamic value from the executable context with a user-facing bounds error. -/
def getValue? {α : Type} [TorchLean.Storage α] [Context α] (vals : Array (Spec.SomeTensor α))
    (idx : Nat) : Except String (Spec.SomeTensor α) :=
  match vals[idx]? with
  | some v => .ok v
  | none =>
      .error s!"TorchLeanVerified: value index {idx} out of bounds for context of size {vals.size}"

/--
Read a previously computed dynamic value and cast it back to the statically expected shape.

The verified fragment constructs only well-scoped indices, but the executable evaluator stores
values in an array, so this check gives a clear error if an implementation bug ever violates the
shape discipline.
-/
def getVal {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {s : Shape}
    (vals : Array (Spec.SomeTensor α)) (idx : Idx (Ctx inShape ss) s) :
    Except String (Tensor α s) := do
  let v : Spec.SomeTensor α ← getValue? vals idx.id
  if h : v.shape = s then
    pure (h ▸ v.tensor)
  else
    throw s!"TorchLeanVerified: expected shape {repr s}, got {repr v.shape}"

/--
Evaluate a single SSA node, given the parameter environment and current value context.

This mirrors the IR denotation for the supported operator subset.
-/
def evalNode
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (node : Node α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (vals : Array (Spec.SomeTensor α)) : Except String (Spec.SomeTensor α) :=
  match node with
  | .const (s := s) _wf t => do
      pure <| Spec.SomeTensor.mk (α := α) s t
  | .paramConst (s := s) _wf p => do
      pure <| Spec.SomeTensor.mk (α := α) s
        (getParam (α := α) (paramShapes := paramShapes) params p)
  | .add (s := s) a b => do
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure <| Spec.SomeTensor.mk (α := α) s (Tensor.addSpec (α := α) ta tb)
  | .sub (s := s) a b => do
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure <| Spec.SomeTensor.mk (α := α) s (Tensor.subSpec (α := α) ta tb)
  | .mulElem (s := s) a b => do
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure <| Spec.SomeTensor.mk (α := α) s (Tensor.mulSpec (α := α) ta tb)
  | .relu (s := s) x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| Spec.SomeTensor.mk (α := α) s (Activation.reluSpec (α := α) tx)
  | .exp (s := s) x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| Spec.SomeTensor.mk (α := α) s (Tensor.expSpec (α := α) tx)
  | .log (s := s) x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      -- Domain discipline: align the verified execution model with the IR semantics. The raw
      -- `log` is treated as undefined on nonpositive inputs; use `safe_log` in models that require
      -- epsilon protection.
      if Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v)) tx then
        pure <| Spec.SomeTensor.mk (α := α) s (Tensor.logSpec (α := α) tx)
      else
        throw
          ("IR eval: log: input contains values <= 0 (or NaN); " ++
            "use `safe_log` if you want epsilon protection")
  | .inv (s := s) x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| Spec.SomeTensor.mk (α := α) s (Tensor.invSpec (α := α) tx)
  | .matmul (leftShape := leftShape) (rightShape := rightShape) (outShape := outShape) op a b => do
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := leftShape) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := rightShape) vals b
      pure <| Spec.SomeTensor.mk (α := α) outShape (op.denote ta tb)
  | .reshape inS outS h x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := inS) vals x
      pure <| Spec.SomeTensor.mk (α := α) outS
        (Tensor.reshapeSpec (α := α) (source := inS) (target := outS) tx h)
  | .transpose (s := s) (out := out) axis₁ axis₂ _hOut x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      let perm ← OpContracts.transposePerm s.rank axis₁ axis₂
      let y ← Graph.permuteSomeTensor (α := α)
        (Spec.SomeTensor.mk (α := α) s tx) perm
      let ty ← Graph.expectShape (α := α) (expected := out) y
      pure <| Spec.SomeTensor.mk (α := α) out ty
  | .softmax (s := s) axis _hAxis x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure <| Spec.SomeTensor.mk (α := α) s
        (Activation.softmaxSpec (α := α) (s := s) axis tx)
  | .layerNorm (s := s) op x => do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      let matrixShape : Shape := .dim op.rows (.dim op.width .scalar)
      let xMatrix : Tensor α matrixShape :=
        Tensor.reshapeSpec (α := α) (source := s) (target := matrixShape) tx op.size_eq
      let yMatrix := Spec.layerNorm (α := α) (seqLen := op.rows) (embedDim := op.width)
        (x := xMatrix)
        (gamma := Tensor.full (α := α) (.dim op.width .scalar) 1)
        (beta := Tensor.full (α := α) (.dim op.width .scalar) 0)
        (h_seq_pos := op.rows_pos) (h_embed_pos := op.width_pos)
      let y := Tensor.reshapeSpec (α := α) (source := matrixShape) (target := s)
        yMatrix op.size_eq.symm
      pure <| Spec.SomeTensor.mk (α := α) s y
  | .linear inDim outDim w b x => do
      let wT := getParam (α := α) (paramShapes := paramShapes) params w
      let bT := getParam (α := α) (paramShapes := paramShapes) params b
      let xT ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := .dim inDim .scalar) vals x
      let y := Tensor.addSpec (α := α)
        (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT
      pure <| Spec.SomeTensor.mk (α := α) (.dim outDim .scalar) y
  | .conv (d := d) inC outC kernelShape stride padding inSpatial _hIn _hKernel _hStride _hInfer
      kernel bias x => do
      let kT := getParam (α := α) (paramShapes := paramShapes) params kernel
      let bT := getParam (α := α) (paramShapes := paramShapes) params bias
      let xT ← getVal (α := α) (inShape := inShape) (ss := ss)
        (s := Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))) vals x
      let spec : Spec.ConvSpec d inC outC kernelShape stride padding α :=
        { kernel := kT, bias := bT }
      let y := Spec.convSpec (α := α) (layer := spec) (input := xT)
      pure <| Spec.SomeTensor.mk (α := α)
        (Shape.ofList (outC :: Tensor.to
          (Spec.convOutSpatial inSpatial kernelShape stride padding) (List Nat))) y
  | .mseLoss (s := _s) yhat target => do
      -- Mirror the IR semantics: `mse_loss` is dynamically shape-checked (both parents must have
      -- equal shape),
      -- then reduces to a scalar by averaging the squared error.
      let yV : Spec.SomeTensor α ← getValue? vals yhat.id
      let tV : Spec.SomeTensor α ← getValue? vals target.id
      if h : yV.shape = tV.shape then
        let yT : Tensor α yV.shape := yV.tensor
        let tT : Tensor α yV.shape := h.symm ▸ tV.tensor
        let s := yV.shape
        let diff := Tensor.subSpec (α := α) yT tT
        let sq := Tensor.mulSpec (α := α) diff diff
        let total : α := Tensor.sumSpec (α := α) sq
        let mean : α := total / (↑(TorchLean.Tensor.meanDenominator s) : α)
        pure <| Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar mean)
      else
        throw
          (s!"TorchLeanVerified: mse_loss expects equal shapes, " ++
            s!"got {repr yV.shape} vs {repr tV.shape}")

/--
Evaluate a forward let-chain program, threading an array of dynamic values.

The `vals` array stores the input and all previously-computed node outputs, so that node evaluation
can do simple array lookups by `Idx.id`.
-/
def evalForwardLetChain
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (vals : Array (Spec.SomeTensor α)) : Except String (Tensor α out) :=
  match g with
  | .ret y => do
      let v : Spec.SomeTensor α ← getValue? vals y.id
      if h : v.shape = out then
        pure (h ▸ v.tensor)
      else
        throw s!"TorchLeanVerified: expected shape {repr out}, got {repr v.shape}"
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext => do
      let vOut ←
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := mid)
          node params vals
      evalForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss ++ [mid]) (out := out)
        gNext params (vals.push vOut)

/--
Evaluate a verified forward fragment program.

This is the top-level evaluator for `ForwardProgram`: it initializes the context with the input
value and then interprets the SSA let-chain.
-/
def evalForward
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : ForwardProgram α paramShapes inShape outShape)
    (params : TorchLean.TensorPack α paramShapes)
    (x : Tensor α inShape) : Except String (Tensor α outShape) := do
  let vals0 : Array (Spec.SomeTensor α) := #[Spec.SomeTensor.mk (α := α) inShape x]
  evalForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := [])
    (out := outShape) p params vals0

end NN.Verification.Builtin.Proved
