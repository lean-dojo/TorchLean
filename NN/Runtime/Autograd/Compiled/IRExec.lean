/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Proofs.Autograd.Runtime.Link

/-!
# IRExec

IR → executable SSA graph bridge.

This module lets us *run* an op-tagged `NN.IR.Graph` by compiling it into an executable
`Proofs.Autograd.Algebra.GraphData` (the SSA/DAG representation used by the proof-compiled runtime).

Why this exists:
- Verification tooling already targets `NN.IR.Graph` (an op-tagged DAG with external payloads).
- The runtime `.compiled` path executes `GraphData` (closures for each node).
- To enforce a single shared IR contract, we provide a checked translation `IR.Graph → GraphData`.
  The forward-correctness theorem connecting `GraphData.eval` to the IR denotation
  (`NN.IR.Graph.denote*`) lives in `NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalence`
  (split out so routine runtime imports do not pull in the full semantic proof).

Important:
- The produced `GraphData` is **forward-correct** by construction.
- Today `jvp`/`vjp` are forward-only sentinels; this bridge is intended for forward execution and
  for closing the shared-IR semantics gap, not for training-style gradient computation.

### PyTorch intuition

If you’re coming from PyTorch:
- This is closer in spirit to *compiled graph execution* (TorchScript / `torch.compile`) than to
  eager mode.
- `NN.IR.Graph` is the "shared IR" we want verifiers and runtimes to agree on.
- `GraphData` is the executable SSA/DAG form: each node becomes a closure that reads parent values
  from a typed runtime context.
- PyTorch’s autograd engine computes gradients by recording an eager tape; this bridge is about
  running the forward pass of an IR graph with a proof that it matches the IR semantics.

## Reading map

- `ExecGraphData` packages a compiled graph with its input shape.
- `IRExec.dValsOfCtx` converts typed runtime contexts back into IR-style value arrays.
- `IRExec.buildFrom` is the compiler from `NN.IR.Graph` to executable graph data.
- `IRExec.execGraphOfIR` is the main user-facing bridge entry point.

## Main definitions

- `ExecGraphData`: compiled executable graph package.
- `IRExec.mkIdx`: checked parent-id to typed-index bridge.
- `IRExec.mkFwdNode`: forward-only node constructor used during lowering.
- `IRExec.buildFrom`: recursive compiler from IR graph to executable SSA graph.
- `IRExec.execGraphOfIR`: user-facing compile entrypoint.

## Implementation notes

- We intentionally focus this bridge on forward semantics first; this keeps the shared IR contract
  clear between verification and runtime execution.
- We set `jvp`/`vjp` sentinels in this layer because gradient compilation is a separate concern
  from proving forward semantic equivalence.
- Lowering untyped numeric ids goes through typed indices (`Idx`) and explicit shape checks.

## References

- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)
- [PyTorch graph execution intuition](https://pytorch.org/docs/stable/generated/torch.compile.html)

## Tags

ir, compiler, runtime, graphdata, forward-semantics
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

/--
`simp` rule for `Except`-`do` chains: binding an `.ok` value is just function application.

This keeps proof scripts around `buildFrom` readable when large `do` blocks are unfolded.
-/
@[simp] theorem Except.ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) :
    (Except.ok a >>= f) = f a := by
  simp [Bind.bind, Except.bind]

/--
`simp` rule for `Except`-`do` chains: binding an `.error` short-circuits.

Used heavily when discharging impossible branches in compilation correctness proofs.
-/
@[simp] theorem Except.error_bind {ε α β : Type} (e : ε) (f : α → Except ε β) :
    (Except.error e >>= f) = Except.error e := by
  simp [Bind.bind, Except.bind]

/-- Definitional simplification for `Except.bind` on `.ok`. -/
@[simp] theorem Except.bind_ok {ε α β : Type} (a : α) (f : α → Except ε β) :
    Except.bind (Except.ok a) f = f a := by
  rfl

/-- Definitional simplification for `Except.bind` on `.error`. -/
@[simp] theorem Except.bind_error {ε α β : Type} (e : ε) (f : α → Except ε β) :
    Except.bind (Except.error e) f = Except.error e := by
  rfl

/--
A forward-executable SSA graph derived from an `NN.IR.Graph`.

The compiled graph stores:
- one distinguished input shape (`inShape`),
- one shape per compiled node (`ss`, corresponding to IR node ids `1..n-1`),
- and executable node closures (`g`) consumed by `GraphData.eval`.
-/
structure ExecGraphData (α : Type) where
  /-- The distinguished IR input node’s shape (node id 0). -/
  inShape : Shape
  /-- Shapes of the IR nodes 1..(n-1) (one per executable SSA node). -/
  ss : List Shape
  /-- Executable SSA/DAG graph for nodes 1..(n-1); inputs live in `Γ := [inShape]`. -/
  g : GraphData α Unit [inShape] ss

namespace ExecGraphData

variable {α : Type}

/--
Evaluate the compiled executable SSA graph on a concrete input tensor.

The result is the full typed runtime context `[inShape] ++ ss`, i.e. input followed by every
compiled node value in topological order.
-/
def eval (e : ExecGraphData α) (x : Tensor α e.inShape) : TList α ([e.inShape] ++ e.ss) :=
  GraphData.eval (α := α) (Δ := Unit) (Γ := [e.inShape]) (ss := e.ss) e.g (.cons x .nil) ()

end ExecGraphData

/-!
## Denotation Table Helper

`ExecGraphData.eval` produces a typed runtime context `TList α ([inShape] ++ ss)`.

For debugging and for the forward-correctness development in
`NN.Runtime.Autograd.Compiled.IRExec.Correctness`,
we provide a helper that erases this context
into an IR-style value table `Array (NN.IR.DVal α)` in node-id order.
-/

namespace IRExec

/-- Convert a runtime `AnyTensor` (shape carried as a field) into an IR denotation value `DVal`. -/
def dValOfAny {α : Type} [Context α] (v : Runtime.AnyTensor α) : NN.IR.DVal α :=
  ⟨v.s, v.t⟩

/--
Convert a typed runtime context `TList α ss` into an IR-style value table.

This is phrased in terms of `Array (DVal α)` because the IR denotation functions (`denoteAll*`)
are array-based, while the compiled runtime evaluates into a typed context (`TList`).
-/
def dValsOfCtx {α : Type} [Context α] {ss : List Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) : Array (NN.IR.DVal α) :=
  (Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := ss) ctx).map (dValOfAny (α := α))

end IRExec

namespace ExecGraphData

variable {α : Type} [Context α]

/--
Convert the full evaluated context into an IR-style value table (one `DVal` per node id).

This is the concrete bridge used in semantic equivalence statements that compare compiled evaluation against
`NN.IR.Graph.denoteAll*`.
-/
def denoteAll (e : Runtime.Autograd.Compiled.ExecGraphData α)
    (x : Tensor α e.inShape) : Array (NN.IR.DVal α) :=
  IRExec.dValsOfCtx (α := α) (ss := [e.inShape] ++ e.ss)
    (Runtime.Autograd.Compiled.ExecGraphData.eval e x)

end ExecGraphData

namespace IRExec

/--
Internal compilation state used by `buildFrom`.

It is a dependent pair of:
- `ss`: shapes of already-compiled IR nodes,
- `GraphData α Unit [inShape] ss`: executable closures for exactly that shape list.
-/
abbrev State (α : Type) (inShape : Shape) : Type :=
  Σ ss : List Shape, GraphData α Unit [inShape] ss

/--
Build a typed runtime index (`Idx`) for a numeric IR parent id.

The compiled runtime context is typed by a list of shapes `[inShape] ++ ss`. `mkIdx` checks that:
- `id` is in bounds, and
- the context shape at that position matches the expected shape `s`.

On failure, this returns a descriptive error string used directly by `buildFrom`.
-/
def mkIdx [DecidableEq Shape]
    (inShape : Shape) (ss : List Shape) (id : Nat) (s : Shape) :
    Except String (Idx ([inShape] ++ ss) s) := by
  let ctxShapes : List Shape := [inShape] ++ ss
  if h : id < ctxShapes.length then
    let fin : Fin ctxShapes.length := ⟨id, h⟩
    let got : Shape := ctxShapes.get fin
    if hg : got = s then
      exact .ok ⟨fin, hg⟩
    else
      exact .error
        s!"IRExec: shape mismatch at id={id}: expected {Shape.pretty s}, got {Shape.pretty got}"
  else
    exact .error s!"IRExec: invalid id={id} for ctxLen={ctxShapes.length}"

/--
Construct a `NodeData` for forward execution only.

The compiled runtime `GraphData` expects each node to supply `forward`, `jvp`, and `vjp`.
For this IR bridge we only care about forward correctness, so `jvp`/`vjp` are populated with
forward-only sentinels that `panic!` if called.

This is intentional: `IRExec` closes the forward semantics gap; full gradient behavior is handled
by other runtime/autograd layers. Using `panic!` here is a safety measure: it prevents silently
wrong gradients if someone accidentally routes differentiation through an `IRExec`-compiled graph.
-/
def mkFwdNode {α : Type} [Context α] {Γ : List Shape} {τ : Shape}
    (forward : TList α Γ → Tensor α τ) : NodeData α Unit Γ τ :=
  -- `panic!` requires an `Inhabited` instance for the return type; provide one for contexts.
  let rec defaultTList : {Γ : List Shape} → TList α Γ
    | [] => .nil
    | s :: Γ => .cons (Tensor.default (α := α) (s := s)) (defaultTList (Γ := Γ))
  let _inst : Inhabited (TList α Γ) := ⟨defaultTList (Γ := Γ)⟩
  { forward := fun ctx _d => forward ctx
    jvp := fun _ctx _dctx _d =>
      panic! "IRExec: JVP requested for a forward-only node (use an autograd-capable backend)."
    vjp := fun _ctx _d _δ =>
      panic! "IRExec: VJP requested for a forward-only node (use an autograd-capable backend)." }

/--
Forward projection for `mkFwdNode`.

The JVP/VJP fields are sentinels in this bridge, but the forward field is exactly the function
passed to the constructor. This small simp lemma is used by the IR semantic-equivalence proof.
-/
@[simp] theorem mkFwdNode_forward {α : Type} [Context α] {Γ : List Shape} {τ : Shape}
    (f : TList α Γ → Tensor α τ) (ctx : TList α Γ) (d : Unit) :
    (mkFwdNode (α := α) (Γ := Γ) (τ := τ) f).forward ctx d = f ctx := by
  rfl

/--
Apply a list of adjacent swaps (specified by swap depths) to a shape.

This is the shape-level companion of `applySwapsTensor`, and mirrors IR permutation lowering.
-/
def swapShapeBySwaps (s : Shape) : List Nat → Shape
  | [] => s
  | d :: ds => swapShapeBySwaps (s.swapAdjacentAtDepth d) ds

/--
Apply the same swap sequence as `swapShapeBySwaps`, but to a tensor value.

This uses `Tensor.swap_at_depth_helper` repeatedly; it is the runtime companion of the IR-side
`swapDepthsForPerm` lowering used by `.permute`.
-/
def applySwapsTensor {α : Type} [Context α] :
    {s : Shape} → (swaps : List Nat) → Tensor α s → Tensor α (swapShapeBySwaps s swaps)
  | _s, [], t => t
  | s, d :: ds, t =>
      let t' : Tensor α (s.swapAdjacentAtDepth d) := Tensor.swapAtDepthHelper (tensor := t) d
      applySwapsTensor (s := s.swapAdjacentAtDepth d) (swaps := ds) t'

/--
Concatenate a list of tensors (all with shape `.dim nP rest`) along dimension 0.

The input list is expressed as typed indices into the runtime context `Γ`; the result tracks the
total concatenated size as a sigma.

This helper supports lowering of IR concat-style operators while preserving shape information.
-/
def concatDim0FromInfos
    {α : Type} [Context α] {Γ : List Shape} {rest : Shape} (ctx : TList α Γ) :
    (infos : List (Sigma fun nP => Idx Γ (.dim nP rest))) →
      Sigma fun nSum => Tensor α (.dim nSum rest)
  | [] =>
      ⟨0, Spec.fill (α := α) 0 (.dim 0 rest)⟩
  | info :: infos =>
      let s0 : Sigma fun n => Tensor α (.dim n rest) :=
        ⟨info.1, getIdx (α := α) (xs := ctx) info.2⟩
      infos.foldl
        (fun acc nxt =>
          match acc, nxt with
  | ⟨n1, t1⟩, ⟨n2, idx2⟩ =>
              let t2 := getIdx (α := α) (xs := ctx) idx2
              ⟨n1 + n2, Tensor.concatDim0Spec (α := α) (n := n1) (m := n2) (s := rest) t1 t2⟩)
        s0

/--
The concatenated size reported by `concatDim0FromInfos` is the sum of the input sizes.

This theorem is used to justify the output-shape side conditions in concat lowering branches.
-/
theorem concatDim0FromInfos_fst_eq_sum
    {α : Type} [Context α] {Γ : List Shape} {rest : Shape}
    (ctx : TList α Γ) (infos : List (Sigma fun nP => Idx Γ (.dim nP rest))) :
    (concatDim0FromInfos (α := α) (Γ := Γ) (rest := rest) ctx infos).1 =
      infos.foldl (fun acc info => acc + info.1) 0 := by
  cases infos with
  | nil =>
      simp [concatDim0FromInfos]
  | cons info infosTail =>
      -- `concatDim0FromInfos` is a foldl over `infosTail` starting from a sigma whose `.1` is
      -- `info.1`.
      -- Its `.1` component is therefore the `Nat` foldl over the same list of `nP`s.
      let f :
          (Sigma fun n => Tensor α (.dim n rest)) →
            (Sigma fun nP => Idx Γ (.dim nP rest)) →
              (Sigma fun n => Tensor α (.dim n rest)) :=
        fun acc nxt =>
          match acc, nxt with
          | ⟨n1, t1⟩, ⟨n2, idx2⟩ =>
              let t2 := getIdx (α := α) (xs := ctx) idx2
              ⟨n1 + n2, Tensor.concatDim0Spec (α := α) (n := n1) (m := n2) (s := rest) t1 t2⟩
      have hfold :
          ∀ acc0 : Sigma fun n => Tensor α (.dim n rest),
            (infosTail.foldl f acc0).1 = infosTail.foldl (fun acc nxt => acc + nxt.1) acc0.1 := by
        intro acc0
        induction infosTail generalizing acc0 with
        | nil =>
            simp
        | cons nxt infos ih =>
            simp [List.foldl, f, ih]
      -- Now rewrite the outer fold (starting at 0) and finish.
      simpa [concatDim0FromInfos, List.foldl] using
        (hfold ⟨info.1, getIdx (α := α) (xs := ctx) info.2⟩)

/--
Compile the IR graph starting at node index `i`, extending the current SSA `State`.

This is the main compiler loop:
- it checks `i < g.nodes.size`,
- compiles node `i` into a `NodeData.forward` closure (rejecting unsupported ops/shapes), and
- `snoc`s the resulting node into the accumulating `GraphData`.

The public entrypoint `execGraphOfIR` handles node 0 and calls `buildFrom` starting at `i = 1`.

Operationally, `buildFrom` is a checked compiler:
- success means every visited node had well-typed parents and a supported lowering case,
- failure returns a concrete error explaining the first unsupported/malformed node.
-/
def buildFrom
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape)
    (i : Nat) (st : State α inShape) : Except String (State α inShape) := do
  let ⟨ss, gd⟩ := st
  if h : i < g.nodes.size then
    let n ← g.getNode i
    let τ : Shape := n.outShape

    -- Helper: build a typed parent index expecting a specific shape.
    let parentIdx (pid : Nat) (s : Shape) : Except String (Idx ([inShape] ++ ss) s) :=
      mkIdx (inShape := inShape) (ss := ss) pid s

    let nodeData : NodeData α Unit ([inShape] ++ ss) τ ←
      match n.kind with
      | .input =>
          throw s!"IRExec: internal error (handled above)"
      | .const s =>
          let t ← NN.IR.Graph.evalConst (α := α) (payload := payload) (id := n.id) (s := s)
          if hOut : s = τ then
            -- Cast so the node is typed at the declared outShape.
            pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) (fun _ctx => hOut ▸ t)
          else
            throw s!"IRExec: const node {i}: outShape mismatch: kind={repr s}, declared={repr τ}"
      | .permute perm =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let sIn := pNode.outShape
              let ip ← parentIdx pId sIn
              match Spec.Shape.permute? sIn perm with
              | none =>
                  throw s!"IRExec: node {i}: invalid permutation {repr perm} for shape {repr sIn}"
              | some expected =>
                  let swaps ← NN.IR.Graph.swapDepthsForPerm perm (Shape.rank sIn)
                  let sFinal : Shape := swapShapeBySwaps sIn swaps
                  if hFinal : sFinal = expected then
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let x := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α sFinal := applySwapsTensor (α := α) (s := sIn) (swaps :=
                          swaps) x
                        let yExpected : Tensor α expected := Tensor.castShape y hFinal
                        Tensor.castShape yExpected hOut
                      pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                    else
                      throw <|
                        s!"IRExec: node {i}: permute outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ}"
                  else
                    throw <|
                      s!"IRExec: node {i}: permute shape mismatch: computed={repr sFinal}, " ++
                        s!"expected={repr expected} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: permute expects 1 parent ({n.summary})"
      | .detach =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              if hOut : s = τ then
                let forward := fun ctx : TList α ([inShape] ++ ss) =>
                  hOut ▸ (getIdx (α := α) (xs := ctx) ip)
                pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
              else
                throw s!"IRExec: node {i}: detach expects outShape=parent.outShape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: detach expects 1 parent ({n.summary})"
      | .randUniform seed =>
          match n.parents with
          | [] =>
              let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
              let t : Tensor α τ := Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := τ)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) (fun _ctx => t)
          | _ => throw s!"IRExec: node {i}: rand_uniform expects 0 parents ({n.summary})"
      | .bernoulliMask seed =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId Shape.scalar
              let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                let kpT := getIdx (α := α) (xs := ctx) ip
                let kp : α :=
                  match kpT with
                  | Tensor.scalar v => v
                Runtime.Autograd.TorchLean.Random.mask (α := α) key kp (s := τ)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: bernoulli_mask expects 1 parent ({n.summary})"
      | .add =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.addSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: add expects 2 parents ({n.summary})"
      | .sub =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.subSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: sub expects 2 parents ({n.summary})"
      | .mul_elem =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.mulSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: mul_elem expects 2 parents ({n.summary})"
      | .abs =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.absSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: abs expects 1 parent ({n.summary})"
      | .sqrt =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.sqrtSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: sqrt expects 1 parent ({n.summary})"
      | .inv =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.invSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: inv expects 1 parent ({n.summary})"
      | .maxElem =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.maxSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: max_elem expects 2 parents ({n.summary})"
      | .minElem =>
          match n.parents with
          | [aId, bId] =>
              let ia ← parentIdx aId τ
              let ib ← parentIdx bId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.minSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
                  ctx) ib)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: min_elem expects 2 parents ({n.summary})"
      | .maxPool2d kH kW stride =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: max_pool2d requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: max_pool2d requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: max_pool2d requires stride ≠ 0 ({n.summary})"
                  else
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                    let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                            (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                    else
                      throw <|
                        s!"IRExec: node {i}: max_pool2d outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: max_pool2d expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: max_pool2d expects 1 parent ({n.summary})"
      | .maxPool2dPad kH kW stride padding =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: max_pool2d_pad requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: max_pool2d_pad requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: max_pool2d_pad requires stride ≠ 0 ({n.summary})"
                  else
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride
                      padding
                    let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                              padding)
                            (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                    else
                      throw <|
                        s!"IRExec: node {i}: max_pool2d_pad outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: max_pool2d_pad expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: max_pool2d_pad expects 1 parent ({n.summary})"
      | .avgPool2d kH kW stride =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d requires stride ≠ 0 ({n.summary})"
                  else
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                    let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                            (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                    else
                      throw <|
                        s!"IRExec: node {i}: avg_pool2d outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: avg_pool2d expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: avg_pool2d expects 1 parent ({n.summary})"
      | .avgPool2dPad kH kW stride padding =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              match pNode.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  if hkH : kH = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d_pad requires kH ≠ 0 ({n.summary})"
                  else if hkW : kW = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d_pad requires kW ≠ 0 ({n.summary})"
                  else if hs : stride = 0 then
                    throw s!"IRExec: node {i}: avg_pool2d_pad requires stride ≠ 0 ({n.summary})"
                  else
                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                    let ip ← parentIdx pId sIn
                    let expected : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW stride
                      padding
                    let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hs := {}
                    if hOut : expected = τ then
                      let forward := fun ctx : TList α ([inShape] ++ ss) =>
                        let xCHW := getIdx (α := α) (xs := ctx) ip
                        let y : Tensor α expected :=
                          Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
                            (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                              padding)
                            (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                        hOut ▸ y
                      pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                    else
                      throw <|
                        s!"IRExec: node {i}: avg_pool2d_pad outShape mismatch: " ++
                          s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              | _ =>
                  throw s!"IRExec: node {i}: avg_pool2d_pad expects CHW parent shape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: avg_pool2d_pad expects 1 parent ({n.summary})"
      | .broadcastTo s₁ s₂ =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId s₁
              match NN.IR.Graph.mkCanBroadcastTo? s₁ s₂ with
              | none =>
                  throw s!"IRExec: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
              | some cb =>
                  if hOut : s₂ = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      hOut ▸ Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                  else
                    throw <|
                      s!"IRExec: node {i}: broadcastTo outShape mismatch: kind={repr s₂}, " ++
                        s!"declared={repr τ}"
          | _ => throw s!"IRExec: node {i}: broadcastTo expects 1 parent ({n.summary})"
      | .reduceSum axis =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              match NN.IR.Graph.mkValidAxis? (axis := axis) s with
              | none =>
                  throw s!"IRExec: node {i}: reduce_sum invalid axis={axis} for shape {repr s}"
              | some hAxis =>
                  let hRed := Shape.proveReducibleAlong axis s hAxis.down
                  let expected : Shape := Spec.Tensor.shapeAfterSum s axis
                  if hOut : expected = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      let y : Tensor α expected := Tensor.reduceSum (α := α) (s := s) axis x hRed
                      hOut ▸ y
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                  else
                    throw <|
                      s!"IRExec: node {i}: reduce_sum outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: reduce_sum expects 1 parent ({n.summary})"
      | .reduceMean axis =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              match NN.IR.Graph.mkValidAxis? (axis := axis) s with
              | none =>
                  throw s!"IRExec: node {i}: reduce_mean invalid axis={axis} for shape {repr s}"
              | some hAxis =>
                  let hRed := Shape.proveReducibleAlong axis s hAxis.down
                  let expected : Shape := Spec.Tensor.shapeAfterSum s axis
                  if hOut : expected = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      let y : Tensor α expected := Tensor.reduceMean (α := α) (s := s) axis x hRed
                      hOut ▸ y
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                  else
                    throw <|
                      s!"IRExec: node {i}: reduce_mean outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: reduce_mean expects 1 parent ({n.summary})"
      | .sum =>
          match n.parents with
          | [pId] =>
              let pNode ← g.getNode pId
              let s := pNode.outShape
              let ip ← parentIdx pId s
              if hOut : Shape.scalar = τ then
                let forward := fun ctx : TList α ([inShape] ++ ss) =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  hOut ▸ Tensor.scalar (Tensor.sumSpec (α := α) x)
                pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
              else
                throw s!"IRExec: node {i}: sum expects scalar outShape ({n.summary})"
          | _ => throw s!"IRExec: node {i}: sum expects 1 parent ({n.summary})"
      | .matmul =>
          match n.parents with
          | [aId, bId] =>
              let aNode ← g.getNode aId
              let bNode ← g.getNode bId
              match aNode.outShape, bNode.outShape with
              | .dim m (.dim nDim Shape.scalar), .dim n' (.dim p Shape.scalar) =>
                  if hn : nDim = n' then
                    match hn with
                    | rfl =>
                        let ia ← parentIdx aId (.dim m (.dim nDim .scalar))
                        let ib ← parentIdx bId (.dim nDim (.dim p .scalar))
                        let expected : Shape := .dim m (.dim p .scalar)
                        if hOut : expected = τ then
                          let forward := fun ctx : TList α ([inShape] ++ ss) =>
                            let aT := getIdx (α := α) (xs := ctx) ia
                            let bT := getIdx (α := α) (xs := ctx) ib
                            let y : Tensor α expected := Spec.matMulSpec (α := α) (m := m) (n :=
                              nDim) (p := p) aT bT
                            hOut ▸ y
                          pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                        else
                          throw <|
                            s!"IRExec: node {i}: matmul outShape mismatch: " ++
                              s!"expected={repr expected}, declared={repr τ} ({n.summary})"
                  else
                    throw s!"IRExec: node {i}: matmul inner dims mismatch: {nDim} vs {n'}"
              | .dim batch (.dim m (.dim nDim Shape.scalar)), .dim batch' (.dim n' (.dim p
                Shape.scalar)) =>
                  if hb : batch = batch' then
                    if hn : nDim = n' then
                      match hb, hn with
                      | rfl, rfl =>
                          let ia ← parentIdx aId (.dim batch (.dim m (.dim nDim .scalar)))
                          let ib ← parentIdx bId (.dim batch (.dim nDim (.dim p .scalar)))
                          let expected : Shape := .dim batch (.dim m (.dim p .scalar))
                          if hOut : expected = τ then
                            let forward := fun ctx : TList α ([inShape] ++ ss) =>
                              let aT := getIdx (α := α) (xs := ctx) ia
                              let bT := getIdx (α := α) (xs := ctx) ib
                              let y : Tensor α expected :=
                                Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := nDim) (p :=
                                  p) aT bT
                              hOut ▸ y
                            pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                          else
                            throw <|
                              s!"IRExec: node {i}: bmm outShape mismatch: " ++
                                s!"expected={repr expected}, declared={repr τ} ({n.summary})"
                    else
                      throw s!"IRExec: node {i}: matmul inner dims mismatch: {nDim} vs {n'}"
                  else
                    throw s!"IRExec: node {i}: matmul batch dims mismatch: {batch} vs {batch'}"
              | _, _ =>
                  throw <|
                    s!"IRExec: node {i}: unsupported matmul shapes: {repr aNode.outShape} · " ++
                      s!"{repr bNode.outShape}"
          | _ => throw s!"IRExec: node {i}: matmul expects 2 parents ({n.summary})"
      | .linear =>
          match n.parents with
          | [xId] =>
              match payload.linear? n.id with
              | none => throw s!"IRExec: missing linear payload for node {n.id}"
              | some p =>
                  let expectedIn : Shape := .dim p.inDim .scalar
                  let expectedOut : Shape := .dim p.outDim .scalar
                  let ix ← parentIdx xId expectedIn
                  if hOut : expectedOut = τ then
                    let W := p.W
                    let b := p.b
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ix
                      let y : Tensor α expectedOut :=
                        Tensor.addSpec (α := α)
                          (Spec.matVecMulSpec (α := α) (m := p.outDim) (n := p.inDim) W x) b
                      hOut ▸ y
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                  else
                    throw <|
                      s!"IRExec: linear {n.id}: declared outShape mismatch: {repr τ} vs " ++
                        s!"expected {repr expectedOut}"
          | _ => throw s!"IRExec: node {i}: linear expects 1 parent ({n.summary})"
      | .conv2d inC outC kH kW stride padding =>
          match n.parents with
          | [xId] =>
              match payload.conv2d? n.id with
              | none => throw s!"IRExec: missing conv2d payload for node {n.id}"
              | some cfg =>
                  -- The payload stores the Conv2d dimensions used to rebuild the layer. `parentIdx`
                  -- checks that the parent has exactly that input shape, and `hOut` checks the
                  -- declared output shape below.
                  let expectedIn : Shape := .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))
                  let ix ← parentIdx xId expectedIn
                  let outH : Nat := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
                  let outW : Nat := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
                  let expected : Shape := .dim cfg.outC (.dim outH (.dim outW .scalar))
                  if hOut : expected = τ then
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x := getIdx (α := α) (xs := ctx) ix
                      let y : Tensor α expected := Spec.conv2dSpec (α := α) (layer := cfg.spec)
                        (input := x)
                      hOut ▸ y
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                  else
                    throw <|
                      s!"IRExec: node {i}: conv2d outShape mismatch: expected={repr expected}, " ++
                        s!"declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: conv2d expects 1 parent ({n.summary})"
      | .relu =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: relu expects 1 parent ({n.summary})"
      | .tanh =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: tanh expects 1 parent ({n.summary})"
      | .sigmoid =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: sigmoid expects 1 parent ({n.summary})"
      | .exp =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.expSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: exp expects 1 parent ({n.summary})"
      | .log =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                let x := getIdx (α := α) (xs := ctx) ip
                -- Keep runtime behavior consistent with the eager autograd engine:
                -- `log` rejects non-positive inputs; use `safe_log` for epsilon protection.
                if Tensor.allSpec (α := α) (s := τ) (fun v => decide (v > (0 : α))) x then
                  Tensor.logSpec (α := α) x
                else
                  panic! "IRExec: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: log expects 1 parent ({n.summary})"
      | .sin =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.sin x) (getIdx (α := α)
                  (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: sin expects 1 parent ({n.summary})"
      | .cos =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.cos x) (getIdx (α := α)
                  (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: cos expects 1 parent ({n.summary})"
      | .softmax axis =>
          match n.parents with
          | [pId] => do
              -- The runtime primitive is last-axis softmax. We keep the compiler disciplined and
              -- reject any request for a non-last axis (callers can insert an explicit `.permute`
              -- node if they want to model a different axis).
              OpContracts.checkLastAxis "softmax" axis τ
              let ip ← parentIdx pId τ
              let forward := fun ctx : TList α ([inShape] ++ ss) =>
                Activation.softmaxSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
              pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
          | _ => throw s!"IRExec: node {i}: softmax expects 1 parent ({n.summary})"
      | .layernorm axis =>
          match n.parents with
          | [pId] => do
              let (seqLen, embedDim) ←
                match OpContracts.layerNorm2DParams axis τ with
                | .ok p => pure p
                | .error msg => throw s!"IRExec: node {i}: layernorm: {msg} ({n.summary})"
              let view2D : Shape := .dim seqLen (.dim embedDim .scalar)
              if hNumel : Shape.size τ = Shape.size view2D then
                if hSeq : seqLen > 0 then
                  if hEmb : embedDim > 0 then
                    let ip ← parentIdx pId τ
                    let gamma : Tensor α (.dim embedDim .scalar) :=
                      Spec.fill (α := α) 1 (.dim embedDim .scalar)
                    let beta : Tensor α (.dim embedDim .scalar) :=
                      Spec.fill (α := α) 0 (.dim embedDim .scalar)
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let x : Tensor α τ := getIdx (α := α) (xs := ctx) ip
                      let x2D : Tensor α view2D :=
                        Tensor.reshapeSpec (α := α) (s₁ := τ) (s₂ := view2D) x hNumel
                      let y2D : Tensor α view2D :=
                        Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                          (x := x2D) (gamma := gamma) (beta := beta)
                          (h_seq_pos := hSeq) (h_embed_pos := hEmb)
                      Tensor.reshapeSpec (α := α) (s₁ := view2D) (s₂ := τ) y2D hNumel.symm
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                  else
                    throw s!"IRExec: node {i}: layernorm embedDim must be > 0 (got {embedDim})"
                else
                  throw s!"IRExec: node {i}: layernorm seqLen must be > 0 (got {seqLen})"
              else
                throw <|
                  s!"IRExec: node {i}: layernorm internal error: bad reshape sizes " ++
                    s!"({Shape.size τ} vs {Shape.size view2D}) ({n.summary})"
          | _ =>
              throw s!"IRExec: node {i}: layernorm expects 1 parent ({n.summary})"
      | .reshape inS outS =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId inS
              if hNumel : Shape.size inS = Shape.size outS then
                if hOut : outS = τ then
                  let forward := fun ctx : TList α ([inShape] ++ ss) =>
                    let x := getIdx (α := α) (xs := ctx) ip
                    hOut ▸ Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) x hNumel
                  pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                else
                  throw <|
                    s!"IRExec: node {i}: reshape outShape mismatch: kind={repr outS}, " ++
                      s!"declared={repr τ}"
              else
                throw <|
                  s!"IRExec: node {i}: reshape numel mismatch: {Shape.size inS} vs " ++
                    s!"{Shape.size outS}"
          | _ => throw s!"IRExec: node {i}: reshape expects 1 parent ({n.summary})"
      | .flatten s =>
          match n.parents with
          | [pId] =>
              let ip ← parentIdx pId s
              let expected : Shape := .dim (Shape.size s) .scalar
              if hOut : expected = τ then
                let forward := fun ctx : TList α ([inShape] ++ ss) =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  let y : Tensor α expected := Tensor.flattenSpec (α := α) (s := s) x
                  hOut ▸ y
                pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
              else
                throw <|
                  s!"IRExec: node {i}: flatten outShape mismatch: " ++
                    s!"expected={repr expected}, declared={repr τ} ({n.summary})"
          | _ => throw s!"IRExec: node {i}: flatten expects 1 parent ({n.summary})"
      | .concat axis =>
          let parents := n.parents
          if parents.length < 2 then
            throw s!"IRExec: node {i}: concat expects at least 2 parents"

          let parentShapes : List Shape ← parents.mapM (fun pid => do
            let pNode ← g.getNode pid
            pure pNode.outShape)
          let expected ←
            match OpContracts.inferConcatOutShape axis parentShapes with
            | .ok s => pure s
            | .error msg => throw s!"IRExec: node {i}: {msg} ({n.summary})"
          if expected != τ then
            throw <|
              s!"IRExec: node {i}: concat outShape mismatch: expected={repr expected}, " ++
                s!"declared={repr τ} ({n.summary})"

          if axis = 0 then
            match hτ : τ with
            | .dim nOut rest =>
                -- Precompute typed indices for each parent and check that tails match.
                let infos : List (Sigma fun nP => Idx ([inShape] ++ ss) (.dim nP rest)) ←
                  parents.mapM (fun pid => do
                    let pNode ← g.getNode pid
                    match pNode.outShape with
                    | .dim nP restP =>
                        if hRest : restP = rest then
                          let ip ← parentIdx pid (.dim nP rest)
                          pure ⟨nP, ip⟩
                        else
                          throw <|
                            s!"IRExec: node {i}: concat axis=0 tail mismatch: {repr restP} vs " ++
                              s!"{repr rest}"
                    | _ =>
                        throw <|
                          s!"IRExec: node {i}: concat axis=0 expects rank≥1 parents, got " ++
                            s!"{repr pNode.outShape}")
                let nSum : Nat := infos.foldl (fun acc info => acc + info.1) 0
                if hSum : nSum = nOut then
                  let forward := fun ctx : TList α ([inShape] ++ ss) =>
                    let outSigma :=
                      concatDim0FromInfos (α := α) (Γ := [inShape] ++ ss) (rest := rest) ctx infos
                    let nSum' : Nat := outSigma.1
                    let tSum : Tensor α (.dim nSum' rest) := outSigma.2
                    have hn : nSum' = nSum := by
                      -- `nSum'` is the first component of the same fold used to compute `nSum`.
                      simpa [nSum', nSum] using
                        (concatDim0FromInfos_fst_eq_sum (α := α) (Γ := [inShape] ++ ss) (rest :=
                          rest) ctx infos)
                    let tSum' : Tensor α (.dim nSum rest) :=
                      Tensor.castShape tSum (by simp [hn])
                    have hOutShape : Shape.dim nSum rest = τ := by
                      have hDim : Shape.dim nSum rest = Shape.dim nOut rest := by
                        simpa using congrArg (fun k => Shape.dim k rest) hSum
                      exact hDim.trans hτ.symm
                    Tensor.castShape tSum' hOutShape
                  pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                else
                  throw <|
                    s!"IRExec: node {i}: concat out dim mismatch: declared {nOut}, computed " ++
                      s!"{nSum} ({n.summary})"
            | _ =>
                throw s!"IRExec: node {i}: concat axis=0 expects rank≥1 outShape, got {repr τ}"
          else
            -- General axis concat: permute `axis` to the front (axis 0), concatenate along axis 0,
            -- then permute back.
            let permFront ←
              match OpContracts.permMoveAxisToFront axis τ with
              | .ok perm => pure perm
              | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
            let permBack ←
              match OpContracts.inversePerm permFront with
              | .ok perm => pure perm
              | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
            match Spec.Shape.permute? τ permFront with
            | none =>
                throw <|
                  s!"IRExec: node {i}: concat: invalid permutation {repr permFront} for " ++
                    s!"shape {repr τ}"
            | some outFrontExpected =>
                match hOutFrontExpected : outFrontExpected with
                | .dim nOutFront restFront =>
                    let swapsFront ← NN.IR.Graph.swapDepthsForPerm permFront (Shape.rank τ)
                    let τFrontFinal : Shape := swapShapeBySwaps τ swapsFront
                    if hOutFrontFinal : τFrontFinal = outFrontExpected then
                      let swapsBack ← NN.IR.Graph.swapDepthsForPerm permBack (Shape.rank
                        outFrontExpected)
                      let τBackFinal : Shape := swapShapeBySwaps outFrontExpected swapsBack
                      if hOutBackFinal : τBackFinal = τ then
                        let getters :
                            List (Sigma fun nP => TList α ([inShape] ++ ss) → Tensor α (.dim nP
                              restFront)) ←
                          parents.mapM (fun pid => do
                            let pNode ← g.getNode pid
                            let sIn := pNode.outShape
                            let ip ← parentIdx pid sIn
                            match Spec.Shape.permute? sIn permFront with
                            | none =>
                                throw <|
                                  s!"IRExec: node {i}: concat: invalid permutation " ++
                                    s!"{repr permFront} for parent shape {repr sIn}"
                            | some (.dim nP restP) =>
                                let sFrontExpected : Shape := .dim nP restP
                                if hRest : restP = restFront then
                                  let sFrontFinal : Shape := swapShapeBySwaps sIn swapsFront
                                  if hFinal : sFrontFinal = sFrontExpected then
                                    let getT := fun ctx : TList α ([inShape] ++ ss) =>
                                      let x : Tensor α sIn := getIdx (α := α) (xs := ctx) ip
                                      let yFinal : Tensor α sFrontFinal :=
                                        applySwapsTensor (α := α) (s := sIn) (swaps := swapsFront) x
                                      let yExpected : Tensor α sFrontExpected :=
                                        Tensor.castShape yFinal hFinal
                                      -- `yExpected` has tail `restP`; cast to the shared
                                      -- `restFront`.
                                      let yExpected' : Tensor α (.dim nP restP) := by
                                        simpa [sFrontExpected] using yExpected
                                      (by
                                        simpa [hRest] using yExpected' : Tensor α (.dim nP
                                          restFront))
                                    pure ⟨nP, getT⟩
                                  else
                                    throw <|
                                      s!"IRExec: node {i}: concat permute shape mismatch: " ++
                                      s!"computed={repr sFrontFinal}, " ++
                                        s!"expected={repr sFrontExpected} ({n.summary})"
                                else
                                  throw <|
                                    s!"IRExec: node {i}: concat: permuted tail mismatch: " ++
                                      s!"{repr restP} vs {repr restFront} ({n.summary})"
                            | some _ =>
                                throw <|
                                  s!"IRExec: node {i}: concat expects rank≥1 parents, got " ++
                                    s!"{repr sIn}"
                          )

                        let nSum : Nat := getters.foldl (fun acc info => acc + info.1) 0
                        if hSum : nSum = nOutFront then
                          let forward := fun ctx : TList α ([inShape] ++ ss) =>
                            let empty : Tensor α (.dim 0 restFront) :=
                              Spec.fill (α := α) 0 (.dim 0 restFront)
                            let outSigma :
                                Sigma fun n => Tensor α (.dim n restFront) :=
                              getters.foldl
                                (fun acc nxt =>
                                  match acc, nxt with
                                  | ⟨n1, t1⟩, ⟨n2, get2⟩ =>
                                      let t2 : Tensor α (.dim n2 restFront) := get2 ctx
                                      ⟨n1 + n2, Tensor.concatDim0Spec (α := α) (n := n1) (m := n2)
                                        (s := restFront) t1 t2⟩)
                                ⟨0, empty⟩
                            let tSum : Tensor α (.dim nSum restFront) :=
                              Tensor.castShape outSigma.2 (by
                                -- The fold's nat component is the sum of the input sizes.
                                have hFold :
                                    outSigma.1 =
                                      getters.foldl (fun acc info => acc + info.1) 0 := by
                                  -- General lemma: the `.1` component of this fold is just a nat
                                  -- fold.
                                  have hGen :
                                      ∀ (xs : List (Sigma fun nP => TList α ([inShape] ++ ss) →
                                        Tensor α (.dim nP restFront)))
                                        (n0 : Nat) (t0 : Tensor α (.dim n0 restFront)),
                                        (xs.foldl
                                            (fun acc nxt =>
                                              match acc, nxt with
                                              | ⟨n1, t1⟩, ⟨n2, get2⟩ =>
                                                  let _t2 : Tensor α (.dim n2 restFront) := get2 ctx
                                                  ⟨n1 + n2, Tensor.concatDim0Spec (α := α) (n :=
                                                    n1) (m := n2) (s := restFront) t1 _t2⟩)
                                            (⟨n0, t0⟩ : Sigma fun n => Tensor α (.dim n
                                              restFront))).1 =
                                          xs.foldl (fun acc info => acc + info.1) n0 := by
                                    intro xs n0 t0
                                    induction xs generalizing n0 t0 with
                                    | nil =>
                                        simp
                                    | cons x xs ih =>
                                        -- Unfold both folds one step and apply the IH to the
                                        -- updated accumulator.
                                        simp [List.foldl] at *
                                        -- After unfolding, the goal is exactly the IH instantiated
                                        -- at `n0 + x.1`.
                                        simpa using
                                          (ih (n0 := n0 + x.1)
                                            (t0 := Tensor.concatDim0Spec (α := α) (n := n0) (m :=
                                              x.1)
                                              (s := restFront) t0 (x.2 ctx)))
                                  simpa [outSigma] using (hGen getters 0 empty)
                                have hn : outSigma.1 = nSum := by
                                  simpa [nSum] using hFold
                                simp [hn])
                            have hOutFront : Shape.dim nSum restFront = outFrontExpected := by
                              have hDim : Shape.dim nSum restFront = Shape.dim nOutFront restFront
                                := by
                                simpa using congrArg (fun k => Shape.dim k restFront) hSum
                              simpa [hOutFrontExpected] using hDim
                            let tFront : Tensor α outFrontExpected := Tensor.castShape tSum
                              hOutFront
                            let tBack : Tensor α τBackFinal :=
                              applySwapsTensor (α := α) (s := outFrontExpected) (swaps := swapsBack)
                                tFront
                            Tensor.castShape tBack hOutBackFinal
                          pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                        else
                          throw <|
                            s!"IRExec: node {i}: concat out dim mismatch: declared {nOutFront}, " ++
                              s!"computed {nSum} ({n.summary})"
                      else
                        throw <|
                          s!"IRExec: node {i}: concat permute-back shape mismatch: " ++
                            s!"computed={repr τBackFinal}, expected={repr τ} ({n.summary})"
                    else
                      throw <|
                        s!"IRExec: node {i}: concat permute-to-front shape mismatch: " ++
                        s!"computed={repr τFrontFinal}, expected={repr outFrontExpected} " ++
                          s!"({n.summary})"
                | _ =>
                    throw s!"IRExec: node {i}: concat expects rank≥1 outShape, got {repr τ}"
        | .swap_first_two =>
            match n.parents with
            | [pId] =>
                match hτ : τ with
                | .dim nDim (.dim m rest) =>
                    let expectedIn : Shape := .dim m (.dim nDim rest)
                    let ip ← parentIdx pId expectedIn
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let y : Tensor α (.dim nDim (.dim m rest)) :=
                        Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest)
                          (getIdx (α := α) (xs := ctx) ip)
                      Tensor.castShape y hτ.symm
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                | _ =>
                    throw s!"IRExec: node {i}: swap_first_two expects rank≥2 outShape ({n.summary})"
            | _ =>
                throw s!"IRExec: node {i}: swap_first_two expects 1 parent ({n.summary})"
        | .transpose3dLastTwo =>
            match n.parents with
            | [pId] =>
                match hτ : τ with
                | .dim a (.dim c (.dim b .scalar)) =>
                    let expectedIn : Shape := .dim a (.dim b (.dim c .scalar))
                    let ip ← parentIdx pId expectedIn
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let y : Tensor α (.dim a (.dim c (.dim b .scalar))) :=
                        Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
                          (getIdx (α := α) (xs := ctx) ip)
                      Tensor.castShape y hτ.symm
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                | _ =>
                    throw <|
                      s!"IRExec: node {i}: transpose3d_last_two expects rank=3 with scalar " ++
                        s!"base outShape ({n.summary})"
            | _ =>
                throw s!"IRExec: node {i}: transpose3d_last_two expects 1 parent ({n.summary})"
        | .mseLoss =>
            match n.parents with
            | [yId, tId] =>
                let yNode ← g.getNode yId
                let tNode ← g.getNode tId
                if hShape : yNode.outShape = tNode.outShape then
                  if hOut : Shape.scalar = τ then
                    let s := yNode.outShape
                    let iy ← parentIdx yId s
                    let it ← parentIdx tId s
                    let forward := fun ctx : TList α ([inShape] ++ ss) =>
                      let yhat := getIdx (α := α) (xs := ctx) iy
                      let target := getIdx (α := α) (xs := ctx) it
                      let diff := Tensor.subSpec (α := α) yhat target
                      let sq := Tensor.mulSpec (α := α) diff diff
                      let total : α := Tensor.sumSpec (α := α) sq
                      let y0 : Tensor α Shape.scalar :=
                        Tensor.scalar (total / (↑(Shape.size s) : α))
                      Tensor.castShape y0 hOut
                    pure <| mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := τ) forward
                  else
                    throw s!"IRExec: node {i}: mse_loss expects scalar outShape ({n.summary})"
                else
                  throw <|
                    s!"IRExec: node {i}: mse_loss expects equal shapes, got " ++
                      s!"{repr yNode.outShape} vs {repr tNode.outShape}"
            | _ => throw s!"IRExec: node {i}: mse_loss expects 2 parents ({n.summary})"

    let st' : State α inShape :=
      ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩
    buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st'
  else
    pure st
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

end IRExec

/--
Compile an op-tagged IR graph into an executable SSA graph (`GraphData`) for forward evaluation.

Requirements:
- Node id 0 must be `.input`.
- The graph must satisfy `Graph.checkWellFormed`.
- The external payload must contain entries for every `.const`/`.linear`/`.conv2d` node id.

This returns an `ExecGraphData` whose `eval` computes all node values in topo order.

This is the main API consumed by runtime callers that want executable evaluation while remaining
aligned with the shared `NN.IR.Graph` semantics.
-/
def execGraphOfIR
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) : Except String (ExecGraphData α) := do
  g.checkWellFormed
  let n0 ← g.getNode 0
  match n0.kind with
  | .input =>
      let inShape := n0.outShape
      let stFinal ← IRExec.buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := 1) (st := (⟨[], .nil⟩ : IRExec.State α inShape))
      let ⟨ss, gd⟩ := stFinal
      pure { inShape := inShape, ss := ss, g := gd }
  | _ =>
      throw s!"IRExec: node 0 is not `.input` (got {n0.kind.tag})"
