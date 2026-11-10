/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.DualTensor
public import TorchLean.NN.Tape.Graph

/-!
# Higher-order graph evaluation

Real and nested-dual executions share the `GraphData` representation. A local jet-preservation
law for each pair of nodes propagates through the graph, saved intermediates included, and
selecting any output then gives mathlib's iterated Fréchet derivative.

The auxiliary environment is held fixed. These results concern exact-real forward evaluation:
jet preservation constrains only the forward maps, so the nodes' JVP and VJP fields need their
own correctness proofs.
-/

@[expose] public section

namespace TorchLean.TensorPack

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}

/-- A family of empty packs is constantly `nil`. -/
private theorem eq_nil_fun {ι : Type*} (f : ι → TensorPack ℝ []) : f = fun _ => .nil := by
  funext y
  cases f y
  rfl

/-- A real context family and a nested-dual context are jet related at `x` when each entry of the
family is `C^n` and the matching nested-dual entry is its jet along `directions`. -/
def JetRelated (directions : Fin n → E) (x : E) :
    {shapes : List Shape} → (E → TensorPack ℝ shapes) →
      TensorPack (Dual.Nested ℝ n) shapes → Prop
  | [], _, .nil => True
  | _ :: _, f, .cons value rest =>
      (ContDiff ℝ n (fun y => (f y).head) ∧
        value = DualTensor.jet directions (fun y => (f y).head) x) ∧
      JetRelated directions x (fun y => (f y).tail) rest

namespace JetRelated

/-- A constant context family is jet related to its `ofPrimal` embedding: fixed model state has
zero mixed derivatives. -/
theorem const {shapes : List Shape} (directions : Fin n → E) (x : E)
    (state : TensorPack ℝ shapes) :
    JetRelated directions x (fun _ => state)
      (state.map (Tensor.map (Dual.Nested.ofPrimal n))) := by
  induction state with
  | nil => trivial
  | cons value rest ih => exact ⟨⟨contDiff_const, (DualTensor.jet_const _ _ _).symm⟩, ih⟩

/-- At every derivative order, the identity family on one input tensor is jet related at `input`
to `seedTensor directions input`. -/
theorem singleton_seed {shape : Shape} (directions : Fin n → Tensor ℝ shape)
    (input : Tensor ℝ shape) :
    JetRelated directions input (fun x => Pack.singleton x)
      (Pack.singleton (Dual.Nested.seedTensor directions input)) :=
  ⟨⟨contDiff_id, (DualTensor.jet_id directions input).symm⟩, trivial⟩

/-- Join separately certified input families, for example fixed state and varying model inputs. -/
theorem append {left right : List Shape} {directions : Fin n → E} {x : E}
    {f : E → TensorPack ℝ left} {g : E → TensorPack ℝ right}
    {fv : TensorPack (Dual.Nested ℝ n) left} {gv : TensorPack (Dual.Nested ℝ n) right}
    (hf : JetRelated directions x f fv) (hg : JetRelated directions x g gv) :
    JetRelated directions x (fun y => Pack.append (f y) (g y)) (Pack.append fv gv) := by
  induction left with
  | nil =>
      have hfun := eq_nil_fun f
      subst f
      cases fv
      exact hg
  | cons shape shapes ih =>
      rw [Pack.eta_cons f] at hf ⊢
      cases fv with
      | cons value rest => exact ⟨hf.1, ih hf.2⟩

/-- Shape-list transport changes neither smoothness nor stored derivative coefficients. -/
theorem cast {shapes target : List Shape} {directions : Fin n → E} {x : E}
    {f : E → TensorPack ℝ shapes} {values : TensorPack (Dual.Nested ℝ n) shapes}
    (h : JetRelated directions x f values) (hs : shapes = target) :
    JetRelated directions x (fun y => Pack.cast hs (f y)) (Pack.cast hs values) := by
  cases hs
  exact h

/-- Appending a `C^n` tensor family and its jet extends the relation to the longer context. -/
theorem snoc {shapes : List Shape} {shape : Shape} {directions : Fin n → E} {x : E}
    {f : E → TensorPack ℝ shapes} {values : TensorPack (Dual.Nested ℝ n) shapes}
    (h : JetRelated directions x f values) {g : E → Tensor ℝ shape}
    (hg : ContDiff ℝ n g) :
    JetRelated directions x (fun y => Pack.snoc (f y) (g y))
      (Pack.snoc values (DualTensor.jet directions g x)) := by
  induction shapes with
  | nil =>
      have hf := eq_nil_fun f
      subst f
      cases values
      exact ⟨⟨hg, rfl⟩, trivial⟩
  | cons shape shapes ih =>
      rw [Pack.eta_cons f] at h ⊢
      cases values with
      | cons value rest => exact ⟨h.1, ih h.2⟩

/-- A typed lookup in jet-related contexts returns a `C^n` tensor family and its jet. -/
theorem get {shapes : List Shape} {shape : Shape} {directions : Fin n → E} {x : E}
    {f : E → TensorPack ℝ shapes} {values : TensorPack (Dual.Nested ℝ n) shapes}
    (h : JetRelated directions x f values) (index : Idx shapes shape) :
    ContDiff ℝ n (fun y => getIdx (f y) index) ∧
      getIdx values index =
        DualTensor.jet directions (fun y => getIdx (f y) index) x := by
  rcases index with ⟨i, rfl⟩
  induction shapes with
  | nil => exact Fin.elim0 i
  | cons shape shapes ih =>
      rw [Pack.eta_cons f] at h ⊢
      cases values with
      | cons value rest =>
          cases i using Fin.cases with
          | zero =>
              convert h.1 using 1 <;> rfl
          | succ i =>
              convert ih h.2 i using 1 <;> rfl

/-- Contexts whose typed lookups all satisfy the conclusion of `get` are jet related. -/
theorem of_get {shapes : List Shape} {directions : Fin n → E} {x : E}
    {f : E → TensorPack ℝ shapes} {values : TensorPack (Dual.Nested ℝ n) shapes}
    (h : ∀ {shape : Shape} (index : Idx shapes shape),
      ContDiff ℝ n (fun y => getIdx (f y) index) ∧
        getIdx values index =
          DualTensor.jet directions (fun y => getIdx (f y) index) x) :
    JetRelated directions x f values := by
  induction shapes with
  | nil => cases values; trivial
  | cons shape shapes ih =>
    rw [Pack.eta_cons f] at h ⊢
    cases values with
    | cons value rest =>
      refine ⟨?_, ih ?_⟩
      · exact h ⟨0, rfl⟩
      · intro s index
        exact h ⟨index.i.succ, index.h⟩

end JetRelated

end TorchLean.TensorPack

namespace TorchLean.Tape

open TensorPack (getIdx)

/-- A real and a nested-dual forward map preserve jets when every jet-related input family has a
`C^n` real output whose jet is the nested output. Reverse-mode correctness is a separate
obligation. -/
def NodeData.PreservesJet (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E]
    (n : Nat) {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : TensorPack ℝ Γ → Δ → Tensor ℝ shape)
    (nested : TensorPack (Dual.Nested ℝ n) Γ → Δ → Tensor (Dual.Nested ℝ n) shape) :
    Prop :=
  ∀ (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ),
    TensorPack.JetRelated directions x inputs values →
      ContDiff ℝ n (fun y => real (inputs y) data) ∧
      nested values data = DualTensor.jet directions (fun y => real (inputs y) data) x

namespace NodeData.PreservesJet

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
  {Δ : Type} {Γ : List Shape} {shape : Shape}

/-- Reading an existing context entry preserves its smoothness and all derivative coefficients. -/
theorem get (index : Idx Γ shape) :
    PreservesJet E n (Δ := Δ)
      (fun ctx _ => getIdx ctx index) (fun ctx _ => getIdx ctx index) := by
  intro directions x inputs values data hinputs
  exact hinputs.get index

/-- Lift a scalar jet rule through an elementwise graph operation reading a saved tensor. -/
theorem map (index : Idx Γ shape) (f : ℝ → ℝ)
    (lifted : Dual.Nested ℝ n → Dual.Nested ℝ n) (hf : ContDiff ℝ n f)
    (hjet : ∀ (directions : Fin n → E) (g : E → ℝ), ContDiff ℝ n g → ∀ x,
      Dual.jet directions (fun y => f (g y)) x = lifted (Dual.jet directions g x)) :
    PreservesJet E n (Δ := Δ)
      (fun inputs _ => Tensor.map f (getIdx inputs index))
      (fun values _ => Tensor.map lifted (getIdx values index)) := by
  intro directions x inputs values data hinputs
  obtain ⟨hg, hvalue⟩ := hinputs.get index
  constructor
  · exact Tensor.contDiff_map hf hg
  · dsimp only
    rw [hvalue]
    apply Tensor.Rep.ext
    intro i
    simp only [Tensor.map_apply, DualTensor.jet_apply]
    exact (hjet directions _ (Tensor.contDiff_coordinate hg i) x).symm

/-- Lift a binary scalar jet rule through any pair of saved tensors of the same shape. -/
theorem zipWith (left right : Idx Γ shape) (f : ℝ → ℝ → ℝ)
    (lifted : Dual.Nested ℝ n → Dual.Nested ℝ n → Dual.Nested ℝ n)
    (hf : ContDiff ℝ n (fun p : ℝ × ℝ => f p.1 p.2))
    (hjet : ∀ (directions : Fin n → E) (g h : E → ℝ),
      ContDiff ℝ n g → ContDiff ℝ n h → ∀ x,
      Dual.jet directions (fun y => f (g y) (h y)) x =
        lifted (Dual.jet directions g x) (Dual.jet directions h x)) :
    PreservesJet E n (Δ := Δ)
      (fun inputs _ => Tensor.zipWith f (getIdx inputs left) (getIdx inputs right))
      (fun values _ => Tensor.zipWith lifted (getIdx values left) (getIdx values right)) := by
  intro directions x inputs values data hinputs
  obtain ⟨hg, hleft⟩ := hinputs.get left
  obtain ⟨hh, hright⟩ := hinputs.get right
  constructor
  · exact Tensor.contDiff_zipWith hf hg hh
  · dsimp only
    rw [hleft, hright]
    apply Tensor.Rep.ext
    intro i
    simp only [Tensor.zipWith_apply, DualTensor.jet_apply]
    exact (hjet directions _ _ (Tensor.contDiff_coordinate hg i)
      (Tensor.contDiff_coordinate hh i) x).symm

end NodeData.PreservesJet

/-- Node-local jet laws for a real and a nested-dual executable graph, node by node in their
recorded order. -/
inductive GraphData.PreservesJet (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E]
    (n : Nat) {Δ : Type} {Γ : List Shape} : {shapes : List Shape} →
    GraphData ℝ Δ Γ shapes → GraphData (Dual.Nested ℝ n) Δ Γ shapes → Prop where
  /-- An empty graph preserves the supplied input jets. -/
  | nil : PreservesJet E n .nil .nil
  /-- Appending a pair of nodes whose forward maps preserve jets extends the law. -/
  | snoc {shapes : List Shape} {shape : Shape}
      {real : GraphData ℝ Δ Γ shapes} {nested : GraphData (Dual.Nested ℝ n) Δ Γ shapes}
      {realNode : NodeData ℝ Δ (Γ ++ shapes) shape}
      {nestedNode : NodeData (Dual.Nested ℝ n) Δ (Γ ++ shapes) shape}
      (previous : PreservesJet E n real nested)
      (node : NodeData.PreservesJet E n realNode.forward nestedNode.forward) :
      PreservesJet E n (.snoc real realNode) (.snoc nested nestedNode)

namespace GraphData.PreservesJet

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
  {Δ : Type} {Γ shapes : List Shape}
  {real : GraphData ℝ Δ Γ shapes} {nested : GraphData (Dual.Nested ℝ n) Δ Γ shapes}

/-- Graph evaluation preserves the jet relation for the entire saved context. -/
theorem eval (h : GraphData.PreservesJet E n real nested) (directions : Fin n → E)
    (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values) :
    TensorPack.JetRelated directions x (fun y => real.eval (inputs y) data)
      (nested.eval values data) := by
  induction h with
  | nil => exact hinputs.cast (List.append_nil Γ).symm
  | @snoc shapes shape real nested realNode nestedNode _ hnode ih =>
      obtain ⟨hsmooth, hvalue⟩ := hnode directions x _ _ data ih
      simpa only [GraphData.eval, hvalue] using
        (ih.snoc hsmooth).cast (List.append_assoc Γ shapes [shape])

/-- For jet-related inputs, `tangentTensor` of the nested evaluation at any graph output is the
`n`-th iterated derivative of the real evaluation along `directions`. -/
theorem tangent_eval (h : GraphData.PreservesJet E n real nested) (directions : Fin n → E)
    (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    {shape : Shape} (output : Idx (Γ ++ shapes) shape) :
    Dual.Nested.tangentTensor (getIdx (nested.eval values data) output) =
      iteratedFDeriv ℝ n (fun y => getIdx (real.eval (inputs y) data) output) x directions := by
  obtain ⟨hsmooth, hvalue⟩ := (h.eval directions x inputs values data hinputs).get output
  rw [hvalue, DualTensor.tangent_jet directions (x := x) hsmooth.contDiffAt]

end GraphData.PreservesJet

end TorchLean.Tape
