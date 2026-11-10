/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Algebra
public import TorchLean.NN.TypedGraph.Data
public import TorchLean.NN.Tape.TensorContext

/-!
# Algebraic tape soundness

Reverse-mode soundness for SSA tapes over any commutative semiring. A `Graph` is an executable
`GraphData` with the proof `GraphData.Correct` that each node's VJP is the adjoint of its JVP
(`NodeData.Correct`), and `Graph.snoc` appends a `LawfulNode` in topological order.
`Graph.jvpCtx_backpropCtx_adjoint` shows that the reverse pass `GraphData.backpropCtx` of a `Graph`
is then the adjoint of `GraphData.jvpCtx` for the context dot product `TensorPack.dotList`.
Executable graphs carry no proofs, and their reverse pass needs only `Add`. `NodeData` and
`GraphData` are defined with their evaluators in `TorchLean.NN.TypedGraph.Data`.

The analytic proofs of `TorchLean.NN.Tape.RealGraph.FDeriv` use this graph at `ℝ`, and
the runtime link `TorchLean.NN.Tape.Lowering` builds on it. No step uses real
analysis, so the theorem holds as well at exact backends such as `ℚ`.

## PyTorch correspondence / citations
PyTorch's reverse-mode engine has the same structure: the forward pass records a tape of nodes,
and the reverse pass adds each node's VJP into the gradients of its parents.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open Rep (dot)

noncomputable section

namespace TensorPack

variable {α : Type} [Storage α]
variable {ss : List Shape}

section

variable [CommSemiring α]

/-- The pairing of two contexts: the sum of the pairings `dot` of corresponding entries. -/
def dotList : {ss : List Shape} → TensorPack α ss → TensorPack α ss → α
  | [], .nil, .nil => 0
  | _ :: ss, .cons a as, .cons b bs => Rep.dot (α := α) a b + dotList (ss := ss) as bs

/-- Casting the left context along `h` gives the same dot product as casting the right one back
along `h.symm`. -/
theorem dotList_cast_left {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (x : TensorPack α ss₁)
    (y : TensorPack α ss₂) :
    dotList (α := α) (Pack.cast h x) y =
      dotList (α := α) x (Pack.cast h.symm y) := by
  cases h
  rfl

/-- `dotList` is additive in its right argument, for the pack sum `TorchLean.TensorPack.add`. -/
theorem dotList_add_right {ss : List Shape} (x y z : TensorPack α ss) :
    dotList (α := α) x (TensorPack.add (α := α) y z) =
      dotList (α := α) x y + dotList (α := α) x z := by
  induction ss with
  | nil =>
    cases x; cases y; cases z; simp [dotList, TensorPack.add]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        cases z with
        | cons zh zt =>
          change Rep.dot xh (yh + zh) + dotList xt (TensorPack.add yt zt) = _
          rw [Tensor.dot_add_right, ih]
          simp only [dotList]
          ac_rfl

/-- The dot product of two `snoc` contexts is the dot product of their prefixes plus the dot
product of their last entries. -/
theorem dotList_snoc {ss : List Shape} {τ : Shape} (x y : TensorPack α ss)
    (a b : Tensor α τ) :
    dotList (α := α) (Pack.snoc (ss := ss) x a)
        (Pack.snoc (ss := ss) y b) =
      dotList (α := α) x y + Rep.dot (α := α) a b := by
  revert x y
  induction ss with
  | nil =>
    intro x y
    cases x; cases y
    simp [Pack.snoc, dotList]
  | cons s ss ih =>
    intro x y
    cases x with
    | cons xh xt =>
      cases y with
      | cons yh yt =>
        simp [Pack.snoc, dotList, ih, add_assoc]

/-- Dotting with the all-zero context on the right yields `0`. -/
theorem dotList_zero_right {ss : List Shape} (x : TensorPack α ss) :
    dotList (α := α) x (TensorPack.zero (α := α) (ss := ss)) = 0 := by
  induction ss with
  | nil =>
    cases x
    simp [dotList, TensorPack.zero]
  | cons s ss ih =>
    cases x with
    | cons xh xt =>
      change Rep.dot xh (Tensor.full s 0) + dotList xt TensorPack.zero = 0
      rw [Tensor.dot_full_zero_right, ih, zero_add]

end

end TensorPack

namespace TensorPack

variable {α : Type} [Storage α]

section

variable [CommSemiring α]

/--
`single` is adjoint to `getIdx` for `dotList`: `⟪dx, single idx v⟫ = ⟪getIdx dx idx, v⟫`.
-/
theorem dotList_single {Γ : List Shape} {s : Shape}
    (dx : TensorPack α Γ) (idx : Idx Γ s) (v : Tensor α s) :
    TensorPack.dotList (α := α) dx (single idx v) =
      Rep.dot (α := α) (getIdx (α := α) dx idx) v := by
  obtain ⟨i, rfl⟩ := idx
  induction Γ with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
    cases dx with
    | cons first rest =>
      obtain ⟨i, hi⟩ := i
      cases i with
      | zero =>
        simp [single, dotList, getIdx, Tensor.castShape, dotList_zero_right]
        rfl
      | succ i =>
        simpa [single, dotList, getIdx, Tensor.castShape,
          Tensor.dot_full_zero_right] using
          ih rest ⟨i, Nat.lt_of_succ_lt_succ hi⟩ v

end

end TensorPack

namespace Tape.NodeData

/-- The local adjoint law: the node's JVP and VJP satisfy the dot-product identity. -/
def Correct {α Δ : Type} [Storage α] [CommSemiring α] {Γ : List Shape} {τ : Shape}
    (node : NodeData α Δ Γ τ) : Prop :=
  ∀ x dx d (δ : Tensor α τ),
    Rep.dot (α := α) (node.jvp x dx d) δ = TensorPack.dotList (α := α) dx (node.vjp x d δ)

end Tape.NodeData

/--
Proof-carrying node: `NodeData` with its local adjointness law, the algebraic form of the
inner-product identity between JVP and VJP.
-/
structure Tape.LawfulNode {α : Type} [Storage α] [CommSemiring α]
    (Δ : Type) (Γ : List Shape) (τ : Shape)
    extends NodeData α Δ Γ τ where
  /-- The node's JVP and VJP satisfy the local dot-product adjoint identity. -/
  jvp_vjp_adjoint : toNodeData.Correct

namespace Tape.GraphData

variable {α : Type} [Storage α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Every node of the graph satisfies its local adjoint law `NodeData.Correct`. -/
def Correct [CommSemiring α] : {ss : List Shape} → GraphData α Δ Γ ss → Prop
  | _, .nil => True
  | _, .snoc g node => Correct g ∧ node.Correct

end Tape.GraphData

/--
A proof-carrying tape/SSA graph: an executable graph whose nodes all satisfy the local adjoint law.

Nodes are appended in topological order and may reference any value computed earlier (fan-out
and sharing are allowed). This mirrors the structure of PyTorch’s dynamic autograd graph, but with
shape-typed contexts.
-/
structure Tape.Graph {α : Type} [Storage α] [CommSemiring α]
    (Δ : Type) (Γ : List Shape) (ss : List Shape) where
  /-- The executable graph. -/
  toData : GraphData α Δ Γ ss
  /-- Every node satisfies its local adjoint law. -/
  correct : toData.Correct

namespace Tape.Graph

variable {α : Type} [Storage α] [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- The graph with no computed nodes, whose context consists only of the inputs `Γ`. -/
abbrev nil : Graph (α := α) Δ Γ [] := ⟨.nil, trivial⟩

/-- Append one locally correct node that may use the inputs and all preceding results. -/
abbrev snoc {ss : List Shape} {τ : Shape} (g : Graph (α := α) Δ Γ ss)
    (node : LawfulNode (α := α) Δ (Γ ++ ss) τ) : Graph (α := α) Δ Γ (ss ++ [τ]) :=
  ⟨.snoc g.toData node.toNodeData, g.correct, node.jvp_vjp_adjoint⟩

/--
Global tape soundness: `GraphData.backpropCtx` of `g.toData` is the adjoint of `GraphData.jvpCtx`
for `TensorPack.dotList`, at every input, tangent, environment and seed. The proof uses only the
local laws `NodeData.Correct` of the nodes, which `g.correct` supplies, and the commutative semiring
laws.
-/
theorem jvpCtx_backpropCtx_adjoint {ss : List Shape} (g : Graph (α := α) Δ Γ ss) :
    ∀ x dx d seed,
      TensorPack.dotList (α := α) (g.toData.jvpCtx x dx d) seed =
        TensorPack.dotList (α := α) dx (g.toData.backpropCtx x d seed) := by
  obtain ⟨g, hg⟩ := g
  dsimp only
  induction g with
  | nil =>
    intro x dx d seed
    simpa [GraphData.jvpCtx, GraphData.backpropCtx] using
      (TensorPack.dotList_cast_left (α := α) (h := (List.append_nil Γ).symm) (x := dx) (y := seed))
  | snoc g node ih =>
    intro x dx d seed
    rename_i ss τ
    obtain ⟨hg, hnode⟩ := hg
    let ctx := g.eval x d
    let dctx := g.jvpCtx x dx d
    let dy := node.jvp ctx dctx d
    let assoc : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]) := List.append_assoc Γ ss [τ]
    let seed' : TensorPack α ((Γ ++ ss) ++ [τ]) :=
      Pack.cast (h := assoc.symm) seed
    let seedPrev : TensorPack α (Γ ++ ss) :=
      (Pack.unsnoc (ss := Γ ++ ss) (τ := τ) seed').1
    let seedOut : Tensor α τ :=
      (Pack.unsnoc (ss := Γ ++ ss) (τ := τ) seed').2
    have hseed :
        Pack.snoc (ss := Γ ++ ss) (τ := τ) seedPrev seedOut = seed' := by
      change Pack.snoc
        (Pack.unsnoc seed').1
        (Pack.unsnoc seed').2 = seed'
      exact Pack.snoc_unsnoc seed'
    have hjvp :
        TensorPack.dotList (α := α) ((GraphData.snoc g node).jvpCtx x dx d) seed =
          TensorPack.dotList (α := α) dctx seedPrev + Rep.dot (α := α) dy seedOut := by
      have :
          TensorPack.dotList (α := α)
              (Pack.snoc (ss := Γ ++ ss) (τ := τ) dctx dy) seed' =
            TensorPack.dotList (α := α) dctx seedPrev + Rep.dot (α := α) dy seedOut := by
        simpa [hseed] using
          (TensorPack.dotList_snoc (α := α) (ss := Γ ++ ss) (τ := τ) (x := dctx) (y := seedPrev)
            (a := dy) (b := seedOut))
      simpa [GraphData.jvpCtx, ctx, dctx, dy, seed', assoc] using
        (TensorPack.dotList_cast_left (α := α) (h := assoc)
          (x := Pack.snoc (ss := Γ ++ ss) (τ := τ) dctx dy) (y := seed)
          |>.trans this)
    have hlocal :
        Rep.dot (α := α) dy seedOut =
        TensorPack.dotList (α := α) dctx (node.vjp ctx d seedOut) := by
      simpa [dy] using (hnode ctx dctx d seedOut)
    have hadd :
        TensorPack.dotList (α := α) dctx seedPrev +
            TensorPack.dotList (α := α) dctx (node.vjp ctx d seedOut) =
          TensorPack.dotList (α := α) dctx
            (TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
              (node.vjp ctx d seedOut)) := by
      simpa using
        (TensorPack.dotList_add_right (α := α) (x := dctx) (y := seedPrev) (z := node.vjp ctx d
          seedOut)).symm
    calc
      TensorPack.dotList (α := α) ((GraphData.snoc g node).jvpCtx x dx d) seed
          = TensorPack.dotList (α := α) dctx seedPrev + Rep.dot (α := α) dy seedOut := hjvp
      _ = TensorPack.dotList (α := α) dctx seedPrev +
            TensorPack.dotList (α := α) dctx (node.vjp ctx d seedOut) := by
            simp [hlocal]
      _ = TensorPack.dotList (α := α) dctx
            (TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
              (node.vjp ctx d seedOut)) := by
            simp [hadd]
      _ = TensorPack.dotList (α := α) dx
            (g.backpropCtx x d
              (TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
                (node.vjp ctx d seedOut))) := by
            simpa [dctx] using
              (ih hg x dx d
                (TensorPack.add (α := α) (ss := Γ ++ ss) seedPrev
                  (node.vjp ctx d seedOut)))
      _ = TensorPack.dotList (α := α) dx
            ((GraphData.snoc g node).backpropCtx x d seed) := by
            simp [GraphData.backpropCtx, ctx, seed', seedPrev, seedOut]

end Tape.Graph

end

end TorchLean
