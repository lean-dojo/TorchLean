/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Lowering.HigherOrder

/-!
# Higher derivatives through reverse accumulation

A reverse pass reads saved activations, computes local pullbacks, and adds their contributions.
The jet relation tracks all three operations, with arbitrary tensor shapes and finite derivative
order. Seeds can depend smoothly on the inputs, as they do when differentiating a composed loss.

The theorems concern `GraphData.backpropCtx` and identify its nested-dual execution with
derivatives of its real execution. Identifying the real execution with the adjoint of the forward
derivative requires the separate first-order correctness certificate.
-/

@[expose] public section

namespace TorchLean.TensorPack.JetRelated

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}

/-- Zero-initialized gradient buffers contain no nonzero derivative coefficients. -/
theorem zero {shapes : List Shape} (directions : Fin n → E) (x : E) :
    JetRelated directions x (fun _ => TensorPack.zero (ss := shapes))
      (TensorPack.zero (α := Dual.Nested ℝ n)) := by
  induction shapes with
  | nil => trivial
  | cons shape shapes ih =>
      refine ⟨⟨contDiff_const, ?_⟩, ih⟩
      exact (DualTensor.jet_full_zero directions x).symm

/-- Placing a `C^n` tensor family at one typed input, with zeros in the other gradient buffers,
preserves its full jet. -/
theorem single {shapes : List Shape} {shape : Shape} (index : Idx shapes shape)
    (directions : Fin n → E) (x : E) {f : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) :
    JetRelated directions x (fun y => TensorPack.single index (f y))
      (TensorPack.single index (DualTensor.jet directions f x)) := by
  rcases index with ⟨i, rfl⟩
  induction shapes with
  | nil => exact Fin.elim0 i
  | cons shape shapes ih =>
      cases i using Fin.cases with
      | zero => exact ⟨⟨hf, rfl⟩, zero directions x⟩
      | succ i =>
          refine ⟨⟨contDiff_const, ?_⟩, ih i hf⟩
          exact (DualTensor.jet_full_zero directions x).symm

/-- Gradient accumulation preserves mixed derivatives, including shared-parent contributions. -/
theorem add {shapes : List Shape} {directions : Fin n → E} {x : E}
    {f g : E → TensorPack ℝ shapes} {fv gv : TensorPack (Dual.Nested ℝ n) shapes}
    (hf : JetRelated directions x f fv) (hg : JetRelated directions x g gv) :
    JetRelated directions x (fun y => TensorPack.add (f y) (g y)) (TensorPack.add fv gv) := by
  induction shapes with
  | nil => cases fv; cases gv; trivial
  | cons shape shapes ih =>
      rw [Pack.eta_cons f] at hf ⊢
      rw [Pack.eta_cons g] at hg ⊢
      cases fv with
      | cons a rest =>
        cases gv with
        | cons b tail =>
          refine ⟨⟨?_, ?_⟩, ih hf.2 hg.2⟩
          · exact Tensor.contDiff_zipWith (by fun_prop) hf.1.1 hg.1.1
          · rw [hf.1.2, hg.1.2]
            apply Tensor.Rep.ext
            intro i
            simp only [DualTensor.jet_apply, TensorPack.add, Pack.zipWith, Pack.head,
              Tensor.add_def, Tensor.zipWith_apply]
            exact (Dual.jet_add directions (x := x)
              (Tensor.contDiff_coordinate hf.1.1 i).contDiffAt
              (Tensor.contDiff_coordinate hg.1.1 i).contDiffAt).symm

/-- Splitting off a node's cotangent preserves both parts of the derivative information. -/
theorem unsnoc {shapes : List Shape} {shape : Shape} {directions : Fin n → E} {x : E}
    {f : E → TensorPack ℝ (shapes ++ [shape])}
    {values : TensorPack (Dual.Nested ℝ n) (shapes ++ [shape])}
    (h : JetRelated directions x f values) :
    JetRelated directions x (fun y => (Pack.unsnoc (f y)).1)
      (Pack.unsnoc values).1 ∧
    ContDiff ℝ n (fun y => (Pack.unsnoc (f y)).2) ∧
      (Pack.unsnoc values).2 =
        DualTensor.jet directions (fun y => (Pack.unsnoc (f y)).2) x := by
  induction shapes with
  | nil =>
      have hf : f = fun y => .cons (f y).head .nil := by
        funext y
        cases f y with
        | cons value rest => cases rest; rfl
      rw [hf] at h ⊢
      cases values with
      | cons value rest => cases rest; exact ⟨trivial, h.1⟩
  | cons shape shapes ih =>
      rw [Pack.eta_cons f] at h ⊢
      cases values with
      | cons value rest =>
        obtain ⟨hp, hl⟩ := ih h.2
        exact ⟨⟨h.1, hp⟩, hl⟩

end TorchLean.TensorPack.JetRelated

namespace TorchLean.Tape

/-- A real and a nested-dual pullback preserve jets when, for jet-related inputs and every `C^n`
seed family, the nested pullback of the seed's jet is jet related to the real pullbacks. The law
concerns differentiation of the pullback, and its first-order adjoint law is separate. -/
def NodeData.PreservesPullbackJet (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E]
    (n : Nat) {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : TensorPack ℝ Γ → Δ → Tensor ℝ shape → TensorPack ℝ Γ)
    (nested : TensorPack (Dual.Nested ℝ n) Γ → Δ →
      Tensor (Dual.Nested ℝ n) shape → TensorPack (Dual.Nested ℝ n) Γ) : Prop :=
  ∀ (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (seed : E → Tensor ℝ shape),
    TensorPack.JetRelated directions x inputs values → ContDiff ℝ n seed →
      TensorPack.JetRelated directions x (fun y => real (inputs y) data (seed y))
        (nested values data (DualTensor.jet directions seed x))

namespace NodeData.PreservesPullbackJet

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
  {Δ : Type} {Γ : List Shape} {shape : Shape}

/-- Routing the seed unchanged to one input preserves its full jet. -/
theorem single (index : Idx Γ shape) :
    NodeData.PreservesPullbackJet E n (Δ := Δ)
      (fun _ _ seed => TensorPack.single index seed)
      (fun _ _ seed => TensorPack.single index seed) := by
  intro directions x inputs values data seed hinputs hseed
  exact TensorPack.JetRelated.single index directions x hseed

/-- The sum of two jet-preserving pullbacks preserves jets, even when both target the same input.
-/
theorem add
    {realLeft realRight : TensorPack ℝ Γ → Δ → Tensor ℝ shape →
      TensorPack ℝ Γ}
    {nestedLeft nestedRight : TensorPack (Dual.Nested ℝ n) Γ → Δ →
      Tensor (Dual.Nested ℝ n) shape → TensorPack (Dual.Nested ℝ n) Γ}
    (hleft : NodeData.PreservesPullbackJet E n realLeft nestedLeft)
    (hright : NodeData.PreservesPullbackJet E n realRight nestedRight) :
    NodeData.PreservesPullbackJet E n
      (fun ctx data seed => (realLeft ctx data seed).add (realRight ctx data seed))
      (fun ctx data seed => (nestedLeft ctx data seed).add (nestedRight ctx data seed)) := by
  intro directions x inputs values data seed hinputs hseed
  exact (hleft directions x inputs values data seed hinputs hseed).add
    (hright directions x inputs values data seed hinputs hseed)

/-- Scatter `seed * coefficient` to one typed input, preserving all mixed derivatives. -/
theorem mul_left (index : Idx Γ shape)
    {real : TensorPack ℝ Γ → Δ → Tensor ℝ shape}
    {nested : TensorPack (Dual.Nested ℝ n) Γ → Δ → Tensor (Dual.Nested ℝ n) shape}
    (h : NodeData.PreservesJet E n real nested) :
    NodeData.PreservesPullbackJet E n
      (fun ctx data seed => TensorPack.single index (seed * real ctx data))
      (fun ctx data seed => TensorPack.single index (seed * nested ctx data)) := by
  intro directions x inputs values data seed hinputs hseed
  obtain ⟨hc, hv⟩ := h directions x inputs values data hinputs
  dsimp only
  rw [hv, ← DualTensor.jet_mul directions (x := x) hseed.contDiffAt hc.contDiffAt]
  exact TensorPack.JetRelated.single index directions x
    (Tensor.contDiff_zipWith (by fun_prop) hseed hc)

/-- Scatter `coefficient * seed` to one typed input, preserving all mixed derivatives. -/
theorem mul_right (index : Idx Γ shape)
    {real : TensorPack ℝ Γ → Δ → Tensor ℝ shape}
    {nested : TensorPack (Dual.Nested ℝ n) Γ → Δ → Tensor (Dual.Nested ℝ n) shape}
    (h : NodeData.PreservesJet E n real nested) :
    NodeData.PreservesPullbackJet E n
      (fun ctx data seed => TensorPack.single index (real ctx data * seed))
      (fun ctx data seed => TensorPack.single index (nested ctx data * seed)) := by
  intro directions x inputs values data seed hinputs hseed
  obtain ⟨hc, hv⟩ := h directions x inputs values data hinputs
  dsimp only
  rw [hv, ← DualTensor.jet_mul directions (x := x) hc.contDiffAt hseed.contDiffAt]
  exact TensorPack.JetRelated.single index directions x
    (Tensor.contDiff_zipWith (by fun_prop) hc hseed)

end NodeData.PreservesPullbackJet

/-- Local jet laws for both saved forward values and the pullbacks in an executable graph. -/
inductive GraphData.PreservesPullbackJet (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E]
    (n : Nat) {Δ : Type} {Γ : List Shape} : {shapes : List Shape} →
    GraphData ℝ Δ Γ shapes → GraphData (Dual.Nested ℝ n) Δ Γ shapes → Prop where
  /-- The empty reverse pass returns its supplied cotangents. -/
  | nil : PreservesPullbackJet E n .nil .nil
  /-- Extend a graph with independent forward and pullback jet laws. -/
  | snoc {shapes : List Shape} {shape : Shape}
      {real : GraphData ℝ Δ Γ shapes} {nested : GraphData (Dual.Nested ℝ n) Δ Γ shapes}
      {realNode : NodeData ℝ Δ (Γ ++ shapes) shape}
      {nestedNode : NodeData (Dual.Nested ℝ n) Δ (Γ ++ shapes) shape}
      (previous : PreservesPullbackJet E n real nested)
      (forward : NodeData.PreservesJet E n realNode.forward nestedNode.forward)
      (pullback : NodeData.PreservesPullbackJet E n realNode.vjp nestedNode.vjp) :
      PreservesPullbackJet E n (.snoc real realNode) (.snoc nested nestedNode)

namespace GraphData.PreservesPullbackJet

open TensorPack (getIdx)

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
  {Δ : Type} {Γ shapes : List Shape}
  {real : GraphData ℝ Δ Γ shapes} {nested : GraphData (Dual.Nested ℝ n) Δ Γ shapes}

/-- Reverse-mode certificates also certify the saved activations used by each pullback. -/
theorem forward (h : GraphData.PreservesPullbackJet E n real nested) :
    GraphData.PreservesJet E n real nested := by
  induction h with
  | nil => exact .nil
  | snoc _ hforward _ ih => exact .snoc ih hforward

/-- Reverse accumulation preserves all derivative coefficients of every input cotangent. -/
theorem backprop (h : GraphData.PreservesPullbackJet E n real nested)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → TensorPack ℝ (Γ ++ shapes))
    (seedValues : TensorPack (Dual.Nested ℝ n) (Γ ++ shapes))
    (hseed : TensorPack.JetRelated directions x seed seedValues) :
    TensorPack.JetRelated directions x
      (fun y => real.backpropCtx (inputs y) data (seed y))
      (nested.backpropCtx values data seedValues) := by
  induction h with
  | nil => exact hseed.cast (List.append_nil Γ)
  | @snoc shapes shape real nested realNode nestedNode previous _ pullback ih =>
      have hctx := previous.forward.eval directions x inputs values data hinputs
      obtain ⟨hp, hs, hv⟩ := (hseed.cast (List.append_assoc Γ shapes [shape]).symm).unsnoc
      have hc := pullback directions x _ _ data _ hctx hs
      apply ih
      simpa only [← hv] using hp.add hc

/-- For jet-related inputs and seeds, `tangentTensor` of each input gradient of the nested reverse
pass is the `n`-th iterated derivative of the real pullback `backpropCtx` along `directions`. The
seed may vary together with the inputs, and a fixed loss cotangent is the special case of a
constant seed. -/
theorem tangent_backprop (h : GraphData.PreservesPullbackJet E n real nested)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → TensorPack ℝ (Γ ++ shapes))
    (seedValues : TensorPack (Dual.Nested ℝ n) (Γ ++ shapes))
    (hseed : TensorPack.JetRelated directions x seed seedValues)
    {shape : Shape} (input : Idx Γ shape) :
    Dual.Nested.tangentTensor (getIdx (nested.backpropCtx values data seedValues) input) =
      iteratedFDeriv ℝ n (fun y => getIdx (real.backpropCtx (inputs y) data (seed y)) input)
        x directions := by
  obtain ⟨hsmooth, hvalue⟩ := (h.backprop directions x inputs values data hinputs
    seed seedValues hseed).get input
  rw [hvalue, DualTensor.tangent_jet directions (x := x) hsmooth.contDiffAt]

end GraphData.PreservesPullbackJet
end TorchLean.Tape
