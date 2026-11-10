/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.TensorVectorization
public import TorchLean.NN.Tape.AdjointOp
public import TorchLean.NN.Tape.Graph
public import TorchLean.Tensor.Euclidean
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Tape nodes from operator arms

We build the tape node of an operator arm with one value output once, for every arm.
`Tape.LawfulNode.ofArm` reads the arm's inputs from the context through `ArmArgs` (a context entry
for each value port, a fixed tensor for each other port), runs the arm's `forward` and `backward`,
and takes as JVP any map that a proof `adj` shows adjoint to the arm's pullback. The converters in
`Arm` produce `adj` from the arm's `op_cert` certificates: a derivative certificate gives its
derivative as the JVP, an algebraic one the arm's own forward-mode rule `jvp?`.

`Tape.RealNode.FDerivCorrect.ofArm₁` and `Tape.RealNode.FDerivCorrect.ofArm₂`, with their
`Tape.RealNode.FDerivCorrectAt` forms, turn a derivative certificate of the arm into the Fréchet
derivative of its node, everywhere or at one point, for one value port with the other ports fixed
and for two value ports. `Tape.NodeData.forwardVec_of_map` and `Tape.NodeData.forwardVec_of_zipWith`
give the vectorized forward map of any node that acts entrywise on one or two context entries.
-/

@[expose] public section

namespace TorchLean

open Tensor.Rep (dot)

noncomputable section

open TensorPack (getIdx)

/-- What feeds one input port of an arm in a tape context: a context entry for a value port, a
fixed tensor for every other port. -/
abbrev ArmArg (Γ : List Shape) : Port → Type
  | ⟨.value, s⟩ => Idx Γ s
  | ⟨.frozen, s⟩ => Tensor ℝ s
  | ⟨.detached, s⟩ => Tensor ℝ s
  | ⟨.index bounds, s⟩ => Tensor (Port.IndexTuple bounds) s
  | ⟨.bool, s⟩ => Tensor Bool s
  | ⟨.integer _ width, s⟩ => Tensor (BitVec width) s

/-- One `ArmArg` per input port, as nested pairs ending in `Unit`. -/
abbrev ArmArgs (Γ : List Shape) : List Port → Type
  | [] => Unit
  | p :: ps => ArmArg Γ p × ArmArgs Γ ps

namespace ArmArgs

variable {Γ : List Shape}

/-- The input values of an arm, read from a context. -/
def values : {ins : List Port} → ArmArgs Γ ins → TensorPack ℝ Γ → Values ℝ ins
  | [], (), _ => ()
  | ⟨.value, _⟩ :: _, (i, r), ctx => (getIdx ctx i, values r ctx)
  | ⟨.frozen, _⟩ :: _, (c, r), ctx => (c, values r ctx)
  | ⟨.detached, _⟩ :: _, (c, r), ctx => (c, values r ctx)
  | ⟨.index _, _⟩ :: _, (c, r), ctx => (c, values r ctx)
  | ⟨.bool, _⟩ :: _, (c, r), ctx => (c, values r ctx)
  | ⟨.integer _ _, _⟩ :: _, (c, r), ctx => (c, values r ctx)

/-- The input tangents of an arm, read from a context tangent. -/
def tangents : {ins : List Port} → ArmArgs Γ ins → TensorPack ℝ Γ → Tangents ℝ ins
  | [], (), _ => ()
  | ⟨.value, _⟩ :: _, (i, r), dctx => (getIdx dctx i, tangents r dctx)
  | ⟨.frozen, _⟩ :: _, (_, r), dctx => ((), tangents r dctx)
  | ⟨.detached, _⟩ :: _, (_, r), dctx => ((), tangents r dctx)
  | ⟨.index _, _⟩ :: _, (_, r), dctx => ((), tangents r dctx)
  | ⟨.bool, _⟩ :: _, (_, r), dctx => ((), tangents r dctx)
  | ⟨.integer _ _, _⟩ :: _, (_, r), dctx => ((), tangents r dctx)

/-- The input cotangents of an arm, added into a context at the entries of its value ports. -/
def scatter : {ins : List Port} → ArmArgs Γ ins → Tangents ℝ ins → TensorPack ℝ Γ
  | [], (), () => TensorPack.zero
  | ⟨.value, _⟩ :: _, (i, r), (g, gs) =>
      TensorPack.add (TensorPack.single i g) (scatter r gs)
  | ⟨.frozen, _⟩ :: _, (_, r), (_, gs) => scatter r gs
  | ⟨.detached, _⟩ :: _, (_, r), (_, gs) => scatter r gs
  | ⟨.index _, _⟩ :: _, (_, r), (_, gs) => scatter r gs
  | ⟨.bool, _⟩ :: _, (_, r), (_, gs) => scatter r gs
  | ⟨.integer _ _, _⟩ :: _, (_, r), (_, gs) => scatter r gs

end ArmArgs

namespace Arm

/-- The pairing of two tangents of an arm's inputs: the sum of `dot` over its value ports. -/
def tangentsDot : (ins : List Port) → Tangents ℝ ins → Tangents ℝ ins → ℝ
  | [], _, _ => 0
  | ⟨.value, _⟩ :: ps, (a, as), (b, bs) => dot a b + tangentsDot ps as bs
  | ⟨.frozen, _⟩ :: ps, (_, as), (_, bs) => tangentsDot ps as bs
  | ⟨.detached, _⟩ :: ps, (_, as), (_, bs) => tangentsDot ps as bs
  | ⟨.index _, _⟩ :: ps, (_, as), (_, bs) => tangentsDot ps as bs
  | ⟨.bool, _⟩ :: ps, (_, as), (_, bs) => tangentsDot ps as bs
  | ⟨.integer _ _, _⟩ :: ps, (_, as), (_, bs) => tangentsDot ps as bs

end Arm

/-- Adding cotangents into a context is adjoint to reading tangents from it. -/
theorem ArmArgs.dotList_scatter {Γ : List Shape} :
    {ins : List Port} → (r : ArmArgs Γ ins) → (dctx : TensorPack ℝ Γ) → (g : Tangents ℝ ins) →
    TensorPack.dotList dctx (r.scatter g) = Arm.tangentsDot ins (r.tangents dctx) g
  | [], (), dctx, () => TensorPack.dotList_zero_right dctx
  | ⟨.value, _⟩ :: _, (i, r), dctx, (g, gs) => by
      rw [scatter, TensorPack.dotList_add_right, TensorPack.dotList_single,
        dotList_scatter r dctx gs]
      rfl
  | ⟨.frozen, _⟩ :: _, (_, r), dctx, (_, gs) => dotList_scatter r dctx gs
  | ⟨.detached, _⟩ :: _, (_, r), dctx, (_, gs) => dotList_scatter r dctx gs
  | ⟨.index _, _⟩ :: _, (_, r), dctx, (_, gs) => dotList_scatter r dctx gs
  | ⟨.bool, _⟩ :: _, (_, r), dctx, (_, gs) => dotList_scatter r dctx gs
  | ⟨.integer _ _, _⟩ :: _, (_, r), dctx, (_, gs) => dotList_scatter r dctx gs

namespace Tape.LawfulNode

/-- The tape node of an operator arm with one value output, reading its inputs through `args`:
its forward map and VJP are the arm's `forward` and `backward`, and `adj` gives its JVP. -/
def ofArm {Δ : Type} {Γ : List Shape} {ins : List Port} {τ : Shape}
    (op : Op ins [.value τ]) (h : (op.rule? (α := ℝ)).isSome = true) (args : ArmArgs Γ ins)
    {jvp : Values ℝ ins → Tangents ℝ ins → Tensor ℝ τ}
    (adj : ∀ xs dxs (δ : Tensor ℝ τ), dot (jvp xs dxs) δ =
      Arm.tangentsDot ins dxs (((op.rule? (α := ℝ)).get h).backward xs (δ, ()))) :
    Tape.LawfulNode (α := ℝ) Δ Γ τ where
  forward ctx _ := (((op.rule? (α := ℝ)).get h).forward (args.values ctx)).1
  jvp ctx dctx _ := jvp (args.values ctx) (args.tangents dctx)
  vjp ctx _ δ := args.scatter (((op.rule? (α := ℝ)).get h).backward (args.values ctx) (δ, ()))
  jvp_vjp_adjoint := by
    intro ctx dctx _ δ
    show dot _ δ = _
    rw [ArmArgs.dotList_scatter]
    exact adj _ _ δ

/-- The vectorized forward map of an arm's node is the arm's forward map on the inputs read from
the context. -/
@[simp] theorem forwardVec_ofArm {Δ : Type} {Γ : List Shape} {ins : List Port} {τ : Shape}
    {op : Op ins [.value τ]} {h : (op.rule? (α := ℝ)).isSome = true} {args : ArmArgs Γ ins}
    {jvp : Values ℝ ins → Tangents ℝ ins → Tensor ℝ τ}
    {adj : ∀ xs dxs (δ : Tensor ℝ τ), dot (jvp xs dxs) δ =
      Arm.tangentsDot ins dxs (((op.rule? (α := ℝ)).get h).backward xs (δ, ()))}
    (d : Δ) (xV : CtxVec Γ) :
    (ofArm (Δ := Δ) op h args adj).forwardVec d xV =
      Tensor.toVec (((op.rule? (α := ℝ)).get h).forward (args.values (TensorPack.ofVec xV))).1 :=
  rfl

end Tape.LawfulNode

namespace Tape.NodeData

/-- A node whose forward map applies `f` to every entry of the context entry `idx`: its vectorized
forward map applies `f` to every coordinate of the block of `idx`. -/
theorem forwardVec_of_map {Δ : Type} {Γ : List Shape} {s : Shape}
    {node : Tape.NodeData ℝ Δ Γ s} (idx : Idx Γ s) (f : ℝ → ℝ)
    (hF : ∀ ctx d, node.forward ctx d = Tensor.map f (getIdx ctx idx) := by intros; rfl)
    (d : Δ) (xV : CtxVec Γ) :
    node.forwardVec d xV = WithLp.toLp 2 fun i => f (CtxVec.get idx xV i) := by
  apply PiLp.ext
  intro i
  dsimp only
  rw [← CtxVec.toVec_getTensorCLM, CtxVec.getTensorCLM_apply]
  show Tensor.toVec (node.forward (TensorPack.ofVec xV) d) i = _
  rw [hF, toVec_apply, toVec_apply]
  exact Tensor.map_apply f _ _

/-- A node whose forward map combines the context entries `a` and `b` entrywise by `f`: its
vectorized forward map combines the coordinates of their blocks by `f`. -/
theorem forwardVec_of_zipWith {Δ : Type} {Γ : List Shape} {s : Shape}
    {node : Tape.NodeData ℝ Δ Γ s} (a b : Idx Γ s) (f : ℝ → ℝ → ℝ)
    (hF : ∀ ctx d, node.forward ctx d = Tensor.zipWith f (getIdx ctx a) (getIdx ctx b) := by
      intros; rfl)
    (d : Δ) (xV : CtxVec Γ) :
    node.forwardVec d xV = WithLp.toLp 2 fun i => f (CtxVec.get a xV i) (CtxVec.get b xV i) := by
  apply PiLp.ext
  intro i
  dsimp only
  rw [← CtxVec.toVec_getTensorCLM a, ← CtxVec.toVec_getTensorCLM b,
    CtxVec.getTensorCLM_apply, CtxVec.getTensorCLM_apply]
  show Tensor.toVec (node.forward (TensorPack.ofVec xV) d) i = _
  rw [hF, toVec_apply, toVec_apply, toVec_apply]
  exact Tensor.zipWith_apply f _ _ _

end Tape.NodeData

namespace Arm

/-- One value port: an unconditional derivative certificate gives the adjointness of an arm, with
the certificate's derivative as JVP. -/
theorem adj_of_cert₁ {s τ : Shape} {rest : List Port}
    {rule : Op.Rule ℝ (.value s :: rest) [.value τ]}
    {D : Values ℝ (.value s :: rest) → Tensor ℝ s →L[ℝ] Tensor ℝ τ}
    (cert : ∀ xs (δ : Tensor ℝ τ),
      HasFDerivAt (fun y => (rule.forward (y, xs.2)).1) (D xs) xs.1 ∧
        (rule.backward xs (δ, ())).1 =
          (fderiv ℝ (fun y => (rule.forward (y, xs.2)).1) xs.1).adjoint δ)
    (hrest : ∀ a b, tangentsDot rest a b = 0 := by intros; rfl) :
    ∀ xs dxs δ, dot (D xs dxs.1) δ = tangentsDot _ dxs (rule.backward xs (δ, ())) := by
  intro xs dxs δ
  obtain ⟨hD, hB⟩ := cert xs δ
  show _ = dot dxs.1 (rule.backward xs (δ, ())).1 + tangentsDot rest _ _
  rw [hrest, add_zero, Tensor.dot_eq_inner, Tensor.dot_eq_inner, hB, hD.fderiv,
    ContinuousLinearMap.adjoint_inner_right]

/-- Two value ports: an unconditional derivative certificate on the L2 product of their tensors
gives the adjointness of an arm, with the certificate's derivative as JVP. -/
theorem adj_of_cert₂ {s t τ : Shape} {rest : List Port}
    {rule : Op.Rule ℝ (.value s :: .value t :: rest) [.value τ]}
    {D : Values ℝ (.value s :: .value t :: rest) →
      WithLp 2 (Tensor ℝ s × Tensor ℝ t) →L[ℝ] Tensor ℝ τ}
    (cert : ∀ xs (δ : Tensor ℝ τ),
      HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
          (rule.forward (p.fst, p.snd, xs.2.2)).1) (D xs) (WithLp.toLp 2 (xs.1, xs.2.1)) ∧
        WithLp.toLp 2 ((rule.backward xs (δ, ())).1, (rule.backward xs (δ, ())).2.1) =
          (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
            (rule.forward (p.fst, p.snd, xs.2.2)).1) (WithLp.toLp 2 (xs.1, xs.2.1))).adjoint δ)
    (hrest : ∀ a b, tangentsDot rest a b = 0 := by intros; rfl) :
    ∀ xs dxs δ, dot (D xs (WithLp.toLp 2 (dxs.1, dxs.2.1))) δ =
      tangentsDot _ dxs (rule.backward xs (δ, ())) := by
  intro xs dxs δ
  obtain ⟨hD, hB⟩ := cert xs δ
  rw [hD.fderiv] at hB
  have h := (D xs).adjoint_inner_right (WithLp.toLp 2 (dxs.1, dxs.2.1)) δ
  rw [← hB, WithLp.prod_inner_apply] at h
  show _ = dot dxs.1 (rule.backward xs (δ, ())).1 +
    (dot dxs.2.1 (rule.backward xs (δ, ())).2.1 + tangentsDot rest _ _)
  rw [hrest, add_zero, Tensor.dot_eq_inner, Tensor.dot_eq_inner, Tensor.dot_eq_inner]
  exact h.symm

/-- One value port: an algebraic certificate whose operation is the arm's forward map and pullback
gives the adjointness of the arm, with the certificate's tangent map as JVP. -/
theorem adj_of_opSpecCorrect {s τ : Shape} {rule : Op.Rule ℝ [.value s] [.value τ]}
    (C : Tape.AdjointOp ℝ s τ)
    (htie : C.op = ⟨fun x => (rule.forward (x, ())).1,
      fun x g => (rule.backward (x, ()) (g, ())).1⟩) :
    ∀ xs dxs δ, dot (C.jvp xs.1 dxs.1) δ = tangentsDot _ dxs (rule.backward xs (δ, ())) := by
  intro xs dxs δ
  have hc := C.isAdjointPair xs.1 dxs.1 δ
  rw [htie] at hc
  show _ = dot dxs.1 (rule.backward xs (δ, ())).1 + 0
  rw [add_zero]
  exact hc

/-- One value port: the adjointness of an arm's own forward-mode rule `jvp?` and its pullback gives
the adjointness of the arm, with that rule as JVP. -/
theorem adj_of_jvp₁ {s τ : Shape} {rest : List Port}
    {rule : Op.Rule ℝ (.value s :: rest) [.value τ]} (hj : rule.jvp?.isSome = true)
    (h : ∀ xs dxs (δ : Tensor ℝ τ),
      dot ((rule.jvp?.get hj) xs dxs).1 δ = dot dxs.1 (rule.backward xs (δ, ())).1)
    (hrest : ∀ a b, tangentsDot rest a b = 0 := by intros; rfl) :
    ∀ xs dxs δ, dot ((rule.jvp?.get hj) xs dxs).1 δ =
      tangentsDot _ dxs (rule.backward xs (δ, ())) := by
  intro xs dxs δ
  show _ = dot dxs.1 (rule.backward xs (δ, ())).1 + tangentsDot rest _ _
  rw [hrest, add_zero]
  exact h xs dxs δ

/-- Two value ports: the adjointness of an arm's own forward-mode rule `jvp?` and its pair of
pullbacks gives the adjointness of the arm, with that rule as JVP. -/
theorem adj_of_jvp₂ {s t τ : Shape} {rest : List Port}
    {rule : Op.Rule ℝ (.value s :: .value t :: rest) [.value τ]} (hj : rule.jvp?.isSome = true)
    (h : ∀ xs dxs (δ : Tensor ℝ τ), dot ((rule.jvp?.get hj) xs dxs).1 δ =
      dot dxs.1 (rule.backward xs (δ, ())).1 + dot dxs.2.1 (rule.backward xs (δ, ())).2.1)
    (hrest : ∀ a b, tangentsDot rest a b = 0 := by intros; rfl) :
    ∀ xs dxs δ, dot ((rule.jvp?.get hj) xs dxs).1 δ =
      tangentsDot _ dxs (rule.backward xs (δ, ())) := by
  intro xs dxs δ
  show _ = dot dxs.1 (rule.backward xs (δ, ())).1 +
    (dot dxs.2.1 (rule.backward xs (δ, ())).2.1 + tangentsDot rest _ _)
  rw [hrest, add_zero]
  exact h xs dxs δ

end Arm

/-! ### Derivatives of `ofArm` nodes -/

/-- A node whose vectorized forward map is `Tensor.toVec ∘ F ∘ L` has the derivative
`Tensor.toVec ∘ D ∘ L` at `xV` when `F` has derivative `D` at `L xV` with adjoint `B`, and the
node's JVP pairs with every cotangent `δ` as `L` pairs with `B δ`. -/
def Tape.RealNode.FDerivCorrectAt.ofAdjoint {Δ : Type} {Γ : List Shape} {τ : Shape}
    {V : Type} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]
    {node : NodeData ℝ Δ Γ τ} {d : Δ} {xV : CtxVec Γ}
    (L : CtxVec Γ →L[ℝ] V) {F : V → Tensor ℝ τ} {D : V →L[ℝ] Tensor ℝ τ}
    {B : Tensor ℝ τ → V}
    (hforward : ∀ yV, node.forwardVec d yV = Tensor.toVec (F (L yV)))
    (hjvp : ∀ dxV δ, inner ℝ (node.jvpVec d xV dxV) (Tensor.toVec δ) = inner ℝ (L dxV) (B δ))
    (cert : ∀ δ, HasFDerivAt F D (L xV) ∧ B δ = (fderiv ℝ F (L xV)).adjoint δ) :
    FDerivCorrectAt node d xV where
  deriv := (Tensor.toVecₗᵢ τ).toContinuousLinearEquiv.toContinuousLinearMap ∘L D ∘L L
  hasFDerivAt := by
    rw [show node.forwardVec d = fun yV => Tensor.toVec (F (L yV)) from funext hforward]
    exact (Tensor.toVecₗᵢ τ).toContinuousLinearEquiv.hasFDerivAt.comp xV
      ((cert 0).1.comp xV L.hasFDerivAt)
  jvp_eq dxV := by
    apply ext_inner_right ℝ
    intro w
    obtain ⟨δ, rfl⟩ := (Tensor.toVecₗᵢ τ).surjective w
    obtain ⟨hD, hB⟩ := cert δ
    rw [Tensor.toVecₗᵢ_apply, hjvp, hB, hD.fderiv, ContinuousLinearMap.adjoint_inner_right]
    exact ((Tensor.toVecₗᵢ τ).inner_map_map (D (L dxV)) δ).symm

/-- One value port `idx`, fixed values `v` on the other ports: the derivative certificate of the
arm at the input read from `xV` gives the derivative of its node at `xV`. -/
def Tape.RealNode.FDerivCorrectAt.ofArm₁ {Δ : Type} {Γ : List Shape} {s τ : Shape}
    {rest : List Port}
    {op : Op (.value s :: rest) [.value τ]} {h : (op.rule? (α := ℝ)).isSome = true}
    {idx : Idx Γ s} {r : ArmArgs Γ rest}
    {jvp : Values ℝ (.value s :: rest) → Tangents ℝ (.value s :: rest) → Tensor ℝ τ}
    {adj : ∀ xs dxs (δ : Tensor ℝ τ), dot (jvp xs dxs) δ =
      Arm.tangentsDot _ dxs (((op.rule? (α := ℝ)).get h).backward xs (δ, ()))}
    {d : Δ} {xV : CtxVec Γ} {v : Values ℝ rest} {D : Tensor ℝ s →L[ℝ] Tensor ℝ τ}
    (cert : ∀ δ : Tensor ℝ τ,
      HasFDerivAt (fun y => (((op.rule? (α := ℝ)).get h).forward (y, v)).1) D
          (CtxVec.getTensorCLM idx xV) ∧
        (((op.rule? (α := ℝ)).get h).backward (CtxVec.getTensorCLM idx xV, v) (δ, ())).1 =
          (fderiv ℝ (fun y => (((op.rule? (α := ℝ)).get h).forward (y, v)).1)
            (CtxVec.getTensorCLM idx xV)).adjoint δ)
    (hv : ∀ ctx, r.values ctx = v := by intros; rfl)
    (hrest : ∀ a b, Arm.tangentsDot rest a b = 0 := by intros; rfl) :
    FDerivCorrectAt (LawfulNode.ofArm (Δ := Δ) op h (idx, r) adj).toNodeData d xV :=
  FDerivCorrectAt.ofAdjoint (CtxVec.getTensorCLM idx)
    (fun yV => by
      show Tensor.toVec (((op.rule? (α := ℝ)).get h).forward
        (getIdx (TensorPack.ofVec yV) idx, r.values (TensorPack.ofVec yV))).1 = _
      rw [hv, CtxVec.getTensorCLM_apply])
    (fun dxV δ => by
      show inner ℝ (Tensor.toVec
        (jvp (getIdx (TensorPack.ofVec xV) idx, r.values (TensorPack.ofVec xV))
        (getIdx (TensorPack.ofVec dxV) idx, r.tangents (TensorPack.ofVec dxV))))
          (Tensor.toVec δ) = _
      rw [← Tensor.dot_eq_inner_toVec]
      erw [adj]
      show dot _ _ + Arm.tangentsDot rest _ _ = _
      rw [hrest, add_zero, Tensor.dot_eq_inner, hv, CtxVec.getTensorCLM_apply,
        CtxVec.getTensorCLM_apply])
    cert

/-- One value port `idx`, fixed values `v` on the other ports: a derivative certificate of the arm
at every input gives the derivative of its node everywhere. -/
def Tape.RealNode.FDerivCorrect.ofArm₁ {Δ : Type} {Γ : List Shape} {s τ : Shape} {rest : List Port}
    {op : Op (.value s :: rest) [.value τ]} {h : (op.rule? (α := ℝ)).isSome = true}
    {idx : Idx Γ s} {r : ArmArgs Γ rest}
    {jvp : Values ℝ (.value s :: rest) → Tangents ℝ (.value s :: rest) → Tensor ℝ τ}
    {adj : ∀ xs dxs (δ : Tensor ℝ τ), dot (jvp xs dxs) δ =
      Arm.tangentsDot _ dxs (((op.rule? (α := ℝ)).get h).backward xs (δ, ()))}
    {d : Δ} {v : Values ℝ rest} {D : Tensor ℝ s → Tensor ℝ s →L[ℝ] Tensor ℝ τ}
    (cert : ∀ (x : Tensor ℝ s) (δ : Tensor ℝ τ),
      HasFDerivAt (fun y => (((op.rule? (α := ℝ)).get h).forward (y, v)).1) (D x) x ∧
        (((op.rule? (α := ℝ)).get h).backward (x, v) (δ, ())).1 =
          (fderiv ℝ (fun y => (((op.rule? (α := ℝ)).get h).forward (y, v)).1) x).adjoint δ)
    (hv : ∀ ctx, r.values ctx = v := by intros; rfl)
    (hrest : ∀ a b, Arm.tangentsDot rest a b = 0 := by intros; rfl) :
    FDerivCorrect (LawfulNode.ofArm (Δ := Δ) op h (idx, r) adj).toNodeData d where
  deriv xV := (FDerivCorrectAt.ofArm₁ (xV := xV) (cert _) hv hrest).deriv
  hasFDerivAt xV := (FDerivCorrectAt.ofArm₁ (xV := xV) (cert _) hv hrest).hasFDerivAt
  jvp_eq xV := (FDerivCorrectAt.ofArm₁ (xV := xV) (cert _) hv hrest).jvp_eq

/-- Two value ports `a` and `b`: the derivative certificate of the arm on the L2 product of its
inputs, at the inputs read from `xV`, gives the derivative of its node at `xV`. -/
def Tape.RealNode.FDerivCorrectAt.ofArm₂ {Δ : Type} {Γ : List Shape} {s t τ : Shape}
    {op : Op [.value s, .value t] [.value τ]} {h : (op.rule? (α := ℝ)).isSome = true}
    {a : Idx Γ s} {b : Idx Γ t}
    {jvp : Values ℝ [.value s, .value t] → Tangents ℝ [.value s, .value t] → Tensor ℝ τ}
    {adj : ∀ xs dxs (δ : Tensor ℝ τ), dot (jvp xs dxs) δ =
      Arm.tangentsDot _ dxs (((op.rule? (α := ℝ)).get h).backward xs (δ, ()))}
    {d : Δ} {xV : CtxVec Γ} {D : WithLp 2 (Tensor ℝ s × Tensor ℝ t) →L[ℝ] Tensor ℝ τ}
    (cert : ∀ δ : Tensor ℝ τ,
      HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
          (((op.rule? (α := ℝ)).get h).forward (p.fst, p.snd, ())).1) D
          (WithLp.toLp 2 (CtxVec.getTensorCLM a xV, CtxVec.getTensorCLM b xV)) ∧
        WithLp.toLp 2
            ((((op.rule? (α := ℝ)).get h).backward
              (CtxVec.getTensorCLM a xV, CtxVec.getTensorCLM b xV, ()) (δ, ())).1,
             (((op.rule? (α := ℝ)).get h).backward
              (CtxVec.getTensorCLM a xV, CtxVec.getTensorCLM b xV, ()) (δ, ())).2.1) =
          (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
            (((op.rule? (α := ℝ)).get h).forward (p.fst, p.snd, ())).1)
            (WithLp.toLp 2 (CtxVec.getTensorCLM a xV, CtxVec.getTensorCLM b xV))).adjoint δ) :
    FDerivCorrectAt (LawfulNode.ofArm (Δ := Δ) op h (a, b, ()) adj).toNodeData d xV :=
  FDerivCorrectAt.ofAdjoint
    ((WithLp.prodContinuousLinearEquiv 2 ℝ (Tensor ℝ s) (Tensor ℝ t)).symm.toContinuousLinearMap ∘L
      (CtxVec.getTensorCLM a).prod (CtxVec.getTensorCLM b))
    (fun yV => by
      show Tensor.toVec (((op.rule? (α := ℝ)).get h).forward
          (getIdx (TensorPack.ofVec yV) a, getIdx (TensorPack.ofVec yV) b, ())).1 =
        Tensor.toVec (((op.rule? (α := ℝ)).get h).forward
          (CtxVec.getTensorCLM a yV, CtxVec.getTensorCLM b yV, ())).1
      rw [CtxVec.getTensorCLM_apply, CtxVec.getTensorCLM_apply])
    (fun dxV δ => by
      show inner ℝ (Tensor.toVec
        (jvp (getIdx (TensorPack.ofVec xV) a, getIdx (TensorPack.ofVec xV) b, ())
          (getIdx (TensorPack.ofVec dxV) a, getIdx (TensorPack.ofVec dxV) b, ())))
            (Tensor.toVec δ) =
        inner ℝ (CtxVec.getTensorCLM a dxV) (((op.rule? (α := ℝ)).get h).backward
            (CtxVec.getTensorCLM a xV, CtxVec.getTensorCLM b xV, ()) (δ, ())).1 +
          inner ℝ (CtxVec.getTensorCLM b dxV) (((op.rule? (α := ℝ)).get h).backward
            (CtxVec.getTensorCLM a xV, CtxVec.getTensorCLM b xV, ()) (δ, ())).2.1
      rw [← Tensor.dot_eq_inner_toVec]
      erw [adj]
      show dot _ _ + (dot _ _ + 0) = _
      simp only [add_zero, Tensor.dot_eq_inner, CtxVec.getTensorCLM_apply])
    cert

/-- Two value ports `a` and `b`: a derivative certificate of the arm at every pair of inputs
gives the derivative of its node everywhere. -/
def Tape.RealNode.FDerivCorrect.ofArm₂ {Δ : Type} {Γ : List Shape} {s t τ : Shape}
    {op : Op [.value s, .value t] [.value τ]} {h : (op.rule? (α := ℝ)).isSome = true}
    {a : Idx Γ s} {b : Idx Γ t}
    {jvp : Values ℝ [.value s, .value t] → Tangents ℝ [.value s, .value t] → Tensor ℝ τ}
    {adj : ∀ xs dxs (δ : Tensor ℝ τ), dot (jvp xs dxs) δ =
      Arm.tangentsDot _ dxs (((op.rule? (α := ℝ)).get h).backward xs (δ, ()))}
    {d : Δ} {D : Tensor ℝ s → Tensor ℝ t → WithLp 2 (Tensor ℝ s × Tensor ℝ t) →L[ℝ] Tensor ℝ τ}
    (cert : ∀ (x : Tensor ℝ s) (y : Tensor ℝ t) (δ : Tensor ℝ τ),
      HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
          (((op.rule? (α := ℝ)).get h).forward (p.fst, p.snd, ())).1) (D x y)
          (WithLp.toLp 2 (x, y)) ∧
        WithLp.toLp 2 ((((op.rule? (α := ℝ)).get h).backward (x, y, ()) (δ, ())).1,
            (((op.rule? (α := ℝ)).get h).backward (x, y, ()) (δ, ())).2.1) =
          (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ t) =>
            (((op.rule? (α := ℝ)).get h).forward (p.fst, p.snd, ())).1)
            (WithLp.toLp 2 (x, y))).adjoint δ) :
    FDerivCorrect (LawfulNode.ofArm (Δ := Δ) op h (a, b, ()) adj).toNodeData d where
  deriv xV := (FDerivCorrectAt.ofArm₂ (xV := xV) (cert _ _)).deriv
  hasFDerivAt xV := (FDerivCorrectAt.ofArm₂ (xV := xV) (cert _ _)).hasFDerivAt
  jvp_eq xV := (FDerivCorrectAt.ofArm₂ (xV := xV) (cert _ _)).jvp_eq

end

end TorchLean
