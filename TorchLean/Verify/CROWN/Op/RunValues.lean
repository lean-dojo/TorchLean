/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op
public import TorchLean.Verify.CROWN.FlatBox.Flatten
public import TorchLean.Verify.CROWN.FlatBox.Concat
import TorchLean.Tensor.ShapeChange.Flatten

/-!
# Runs of operator graphs at `ℝ`

The facts about a successful run of an operator graph (`TorchLean.IR.Node.eval`,
`TorchLean.IR.Graph.run`) that the enclosure proofs read. `runVals` is the table of flat values of
output `0` of each node, `box?_runVals` reads it through a reference and `join_runVals` reads it at
a node (`runVal`). `invoke_one`, `invoke_two` and `invoke_three` take a successful invocation
apart, and `const_eval` and `input_eval` give the outputs of a constant node and of an input node.
`flat_matmul` (through `IR.MatmulDims.matmulFlat_read`), `flat_scaledMatmul` and `flat_cat` give
the flat values of the matrix product, the scaled product and the concatenation, and at `ℝ` the
scaled product is the product times the scale (`scaledMatmul_eq_scale`).
-/

@[expose] public section

open TorchLean.Shape (rank_eq_length size)

namespace TorchLean.CROWN.Op

open Tensor
open Graph

noncomputable section

/-- The flat values of output `0` of each node of a run. -/
def runVals (values : Array (List (Op.Value ℝ))) : Array (Option (FlatTensor ℝ)) :=
  values.map fun outs => outs[0]? >>= flatOf

/-- A reference to output `0` that reads `v` names the flat value of `v`. -/
theorem box?_runVals {earlier : Array (List (Op.Value ℝ))} {r : IR.Ref}
    {v : Op.Value ℝ} (h0 : r.out = 0) (h : r.read earlier = .ok v) :
    box? (runVals earlier) r = flatOf v := by
  unfold IR.Ref.read at h
  unfold box? runVals
  rw [h0] at h
  simp only [h0, ↓reduceIte, Array.getElem?_map]
  cases he : earlier[r.node]? with
  | none => simp [he, throw, throwThe, MonadExceptOf.throw] at h
  | some outs =>
    cases ho : outs[0]? with
    | none => simp [he, ho, throw, throwThe, MonadExceptOf.throw] at h
    | some w =>
      simp only [he, ho, Option.bind_eq_bind, Option.bind_some, pure, Except.pure,
        Except.ok.injEq] at h
      subst h
      simp [ho]

/-- The node a reference names holds, in the table, the value the reference names. -/
theorem join_of_box? {β : Type} {vals : Array (Option β)} {r : IR.Ref} {t : β}
    (h : box? vals r = some t) : (vals[r.node]?).join = some t := by
  unfold box? at h
  split at h
  · exact h
  · cases h

/-- The table of a run holds at each node the flat value of its output `0`. -/
theorem join_runVals (values : Array (List (Op.Value ℝ))) (id : Nat) :
    ((runVals values)[id]?).join = runVal values id := by
  simp only [runVals, runVal, Array.getElem?_map]
  cases values[id]? <;> rfl

/-- The flat value of node `id` of a run is the flat value of the first output of its entry. -/
theorem runVal_eq {values : Array (List (Op.Value ℝ))} {id : Nat}
    {outs : List (Op.Value ℝ)} (h : values[id]? = some outs) :
    runVal values id = outs[0]? >>= flatOf := by
  simp [runVal, h]

/-- A value read at a port of the scalar flattens as the tensor it reads. -/
theorem flatOf_of_read {p : Port} {v : Op.Value ℝ} {x : p.Carrier ℝ}
    (hp : p.kind = .value ∨ p.kind = .frozen ∨ p.kind = .detached) (h : p.read v = .ok x) :
    flatOf v = flatOf ⟨p, x⟩ := by
  obtain ⟨d, s⟩ := p
  obtain ⟨⟨e, t⟩, y⟩ := v
  simp only at hp
  rcases hp with rfl | rfl | rfl <;> cases e <;>
    simp only [Port.read, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h <;>
    first
    | (split at h
       · rename_i hst
         cases hst
         cases h
         rfl
       · cases h)
    | (cases h)

/-- Reading one port's values from a list succeeds only on a singleton. -/
theorem ofErased_singleton {p : Port} {values : List (Op.Value ℝ)}
    {xs : Values ℝ [p]} (h : Values.ofErased [p] values = .ok xs) :
    ∃ v, values = [v] ∧ p.read v = .ok xs.1 := by
  match values, h with
  | [], h => simp [Values.ofErased, throw, throwThe, MonadExceptOf.throw] at h
  | [v], h =>
    refine ⟨v, rfl, ?_⟩
    simp only [Values.ofErased, bind, Except.bind, pure, Except.pure] at h
    cases hr : p.read v with
    | error e => simp [hr] at h
    | ok y => simp only [hr] at h; cases h; rfl
  | v :: w :: rest, h =>
    simp only [Values.ofErased, bind, Except.bind] at h
    cases hr : p.read v with
    | error e => simp [hr] at h
    | ok y => simp [hr, throw, throwThe, MonadExceptOf.throw] at h

/-- Reading two ports' values from a list succeeds only on a pair. -/
theorem ofErased_pair {p q : Port} {values : List (Op.Value ℝ)}
    {xs : Values ℝ [p, q]} (h : Values.ofErased [p, q] values = .ok xs) :
    ∃ v w, values = [v, w] ∧ p.read v = .ok xs.1 ∧ q.read w = .ok xs.2.1 := by
  match values, h with
  | [], h => simp [Values.ofErased, throw, throwThe, MonadExceptOf.throw] at h
  | v :: rest, h =>
    simp only [Values.ofErased, bind, Except.bind] at h
    cases hr : p.read v with
    | error e => simp [hr] at h
    | ok y =>
      simp only [hr] at h
      cases hrest : Values.ofErased [q] rest with
      | error e => simp [hrest] at h
      | ok ys =>
        simp only [hrest, pure, Except.pure] at h
        cases h
        obtain ⟨w, rfl, hw⟩ := ofErased_singleton hrest
        exact ⟨v, w, rfl, hr, hw⟩

/-- If reading a list of references gives one value, the list is one reference that reads it. -/
theorem mapM_read_singleton {refs : List IR.Ref} {earlier : Array (List (Op.Value ℝ))}
    {v : Op.Value ℝ} (h : refs.mapM (·.read earlier) = .ok [v]) :
    ∃ r, refs = [r] ∧ r.read earlier = .ok v := by
  match refs, h with
  | [], h => simp [pure, Except.pure] at h
  | [r], h =>
    simp only [List.mapM_cons, List.mapM_nil, bind, Except.bind] at h
    cases hr : r.read earlier with
    | error e => simp [hr] at h
    | ok y => simp [hr, pure, Except.pure] at h; exact ⟨r, rfl, hr ▸ congrArg Except.ok h⟩
  | r :: s :: rest, h =>
    simp only [List.mapM_cons, bind, Except.bind] at h
    cases hr : r.read earlier with
    | error e => simp [hr] at h
    | ok y =>
      cases hs : s.read earlier with
      | error e => simp [hr, hs] at h
      | ok z =>
        cases hm : rest.mapM (·.read earlier) with
        | error e => simp [hr, hs, hm] at h
        | ok l => simp [hr, hs, hm, pure, Except.pure] at h

/-- If reading a list of references gives two values, the list is two references that read them
in order. -/
theorem mapM_read_pair {refs : List IR.Ref} {earlier : Array (List (Op.Value ℝ))}
    {v w : Op.Value ℝ} (h : refs.mapM (·.read earlier) = .ok [v, w]) :
    ∃ r₁ r₂, refs = [r₁, r₂] ∧ r₁.read earlier = .ok v ∧ r₂.read earlier = .ok w := by
  match refs, h with
  | [], h => simp [pure, Except.pure] at h
  | r :: rest, h =>
    simp only [List.mapM_cons, bind, Except.bind] at h
    cases hr : r.read earlier with
    | error e => simp [hr] at h
    | ok y =>
      cases hm : rest.mapM (·.read earlier) with
      | error e => simp [hr, hm] at h
      | ok l =>
        simp only [hr, hm, pure, Except.pure, Except.ok.injEq, List.cons.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨s, rfl, hs⟩ := mapM_read_singleton hm
        exact ⟨r, s, rfl, hr, hs⟩

/-- A successful invocation of an operator with one input port and one output port has one
reference, which reads a value that the port reads as `x`, and outputs the rule's forward value on
`x`. -/
theorem invoke_one {p q : Port} {op : Op [p] [q]}
    {rule : Op.Rule ℝ [p] [q]} (hs : op.rule? (α := ℝ) = some rule)
    {refs : Array IR.Ref} {earlier : Array (List (Op.Value ℝ))}
    {outs : List (Op.Value ℝ)}
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs) :
    ∃ r v x, refs.toList = [r] ∧ r.read earlier = .ok v ∧ p.read v = .ok x ∧
      outs = [⟨q, (rule.forward (x, ())).1⟩] := by
  obtain ⟨values, xs, rule', -, hin, hx, hs', -, rfl⟩ := Op.invoke_eq_ok h
  rw [hs] at hs'
  cases hs'
  obtain ⟨v, rfl, hv⟩ := ofErased_singleton hx
  obtain ⟨r, hr, hread⟩ := mapM_read_singleton hin
  obtain ⟨x, ⟨⟩⟩ := xs
  exact ⟨r, v, x, hr, hread, hv, rfl⟩

/-- A successful invocation of an operator with two input ports and one output port has two
references, which read values that the ports read as `x₁` and `x₂`, and outputs the rule's forward
value on them. -/
theorem invoke_two {p₁ p₂ q : Port} {op : Op [p₁, p₂] [q]}
    {rule : Op.Rule ℝ [p₁, p₂] [q]} (hs : op.rule? (α := ℝ) = some rule)
    {refs : Array IR.Ref} {earlier : Array (List (Op.Value ℝ))}
    {outs : List (Op.Value ℝ)}
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs) :
    ∃ r₁ r₂ v₁ v₂ x₁ x₂, refs.toList = [r₁, r₂] ∧ r₁.read earlier = .ok v₁ ∧
      r₂.read earlier = .ok v₂ ∧ p₁.read v₁ = .ok x₁ ∧ p₂.read v₂ = .ok x₂ ∧
      outs = [⟨q, (rule.forward (x₁, x₂, ())).1⟩] := by
  obtain ⟨values, xs, rule', -, hin, hx, hs', -, rfl⟩ := Op.invoke_eq_ok h
  rw [hs] at hs'
  cases hs'
  obtain ⟨v₁, v₂, rfl, h₁, h₂⟩ := ofErased_pair hx
  obtain ⟨r₁, r₂, hr, hread₁, hread₂⟩ := mapM_read_pair hin
  obtain ⟨x₁, x₂, ⟨⟩⟩ := xs
  exact ⟨r₁, r₂, v₁, v₂, x₁, x₂, hr, hread₁, hread₂, h₁, h₂, rfl⟩

/-- A successful invocation of an operator with three input ports and one output port has three
references, which read values that the ports read as `x₁`, `x₂` and `x₃`, and outputs the rule's
forward value on them. -/
theorem invoke_three {p₁ p₂ p₃ q : Port} {op : Op [p₁, p₂, p₃] [q]}
    {rule : Op.Rule ℝ [p₁, p₂, p₃] [q]} (hs : op.rule? (α := ℝ) = some rule)
    {refs : Array IR.Ref} {earlier : Array (List (Op.Value ℝ))}
    {outs : List (Op.Value ℝ)}
    (h : op.invoke (refs.toList.mapM (·.read earlier)) = .ok outs) :
    ∃ r₁ r₂ r₃ v₁ v₂ v₃ x₁ x₂ x₃, refs.toList = [r₁, r₂, r₃] ∧ r₁.read earlier = .ok v₁ ∧
      r₂.read earlier = .ok v₂ ∧ r₃.read earlier = .ok v₃ ∧ p₁.read v₁ = .ok x₁ ∧
      p₂.read v₂ = .ok x₂ ∧ p₃.read v₃ = .ok x₃ ∧
      outs = [⟨q, (rule.forward (x₁, x₂, x₃, ())).1⟩] := by
  obtain ⟨values, xs, rule', -, hin, hx, hs', -, rfl⟩ := Op.invoke_eq_ok h
  rw [hs] at hs'
  cases hs'
  match values, hx with
  | [], hx => simp [Values.ofErased, throw, throwThe, MonadExceptOf.throw] at hx
  | v₁ :: rest, hx =>
    simp only [Values.ofErased, bind, Except.bind] at hx
    cases h₁ : p₁.read v₁ with
    | error e => simp [h₁] at hx
    | ok y =>
      simp only [h₁] at hx
      cases hrest : Values.ofErased [p₂, p₃] rest with
      | error e => simp [hrest] at hx
      | ok ys =>
        simp only [hrest, pure, Except.pure] at hx
        cases hx
        obtain ⟨v₂, v₃, rfl, h₂, h₃⟩ := ofErased_pair hrest
        match refs.toList, hin with
        | [], hin => simp [pure, Except.pure] at hin
        | r₁ :: rs, hin =>
          simp only [List.mapM_cons, bind, Except.bind] at hin
          cases hr₁ : r₁.read earlier with
          | error e => simp [hr₁] at hin
          | ok z =>
            cases hm : rs.mapM (·.read earlier) with
            | error e => simp [hr₁, hm] at hin
            | ok l =>
              simp only [hr₁, hm, pure, Except.pure, Except.ok.injEq, List.cons.injEq] at hin
              obtain ⟨rfl, rfl⟩ := hin
              obtain ⟨r₂, r₃, rfl, hr₂, hr₃⟩ := mapM_read_pair hm
              obtain ⟨y₂, y₃, ⟨⟩⟩ := ys
              exact ⟨r₁, r₂, r₃, z, v₂, v₃, y, y₂, y₃, rfl, hr₁, hr₂, hr₃, h₁, h₂, h₃, rfl⟩

/-- The flat value of a value at a value port. -/
theorem flatOf_value {s : Shape} (x : Tensor ℝ s) :
    flatOf (⟨.value s, x⟩ : Op.Value ℝ) = some (flatOfSome ⟨s, x⟩) := rfl

/-- A dimension cast keeps every zero-extended read. -/
theorem getAtOrZero_castDimScalar {n n' : Nat} (h : n = n') (t : Tensor ℝ [n]) (i : Nat) :
    getAtOrZero (castDimScalar (α := ℝ) h t) [i] = getAtOrZero t [i] := by
  cases h
  rw [castDimScalar_self]

/-- The flat value of the registered matrix product of two operands is the flat contraction over
the operator's layout (`matmulLayout`) of their flat values. -/
theorem flat_matmul {batchA batchB batch : Shape} {m n p : Nat} {lv rv : Bool}
    (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (hA : Shape.CanBroadcastTo batchA batch) (hB : Shape.CanBroadcastTo batchB batch)
    (x : Tensor ℝ (Shape.matmulLeft lv batchA m n))
    (y : Tensor ℝ (Shape.matmulRight rv batchB n p)) :
    flatOfSome ⟨Shape.matmulOut lv rv batch m p,
      Tensor.reshape (Tensor.batchMatmul hA hB
          (Tensor.reshape x _ (Shape.size_matmulLeft hl _ _))
          (Tensor.reshape y _ (Shape.size_matmulRight hr _ _)))
        _ (Shape.size_matmulOut hl hr _).symm⟩ =
      { n := (matmulLayout lv rv batchA batchB batch m n p).outShape.size
        v := IR.MatmulDims.matmulFlat (matmulLayout lv rv batchA batchB batch m n p)
          (flatOfSome ⟨_, x⟩).v (flatOfSome ⟨_, y⟩).v } := by
  have hout : (matmulLayout lv rv batchA batchB batch m n p).outShape =
      Shape.matmulOut lv rv batch m p := by
    cases lv <;> cases rv <;> rfl
  have hsz : (matmulLayout lv rv batchA batchB batch m n p).outShape.size =
      (batch ++ [m, p]).size := by
    rw [hout]
    exact Shape.size_matmulOut hl hr batch
  have hread := IR.MatmulDims.matmulFlat_read (matmulLayout lv rv batchA batchB batch m n p) hA hB
    (Tensor.reshape x _ (Shape.size_matmulLeft hl _ _))
    (Tensor.reshape y _ (Shape.size_matmulRight hr _ _)) hsz
  have hcongr := IR.MatmulDims.matmulFlat_congr (matmulLayout lv rv batchA batchB batch m n p)
    (a₁ := (flatOfSome ⟨_, x⟩).v) (b₁ := (flatOfSome ⟨_, y⟩).v)
    (fun i => (Tensor.flat_read_reshape x (Shape.size_matmulLeft hl batchA n) i).symm)
    (fun i => (Tensor.flat_read_reshape y (Shape.size_matmulRight hr batchB n) i).symm)
  symm
  refine flatTensor_eq_of_cast (congrArg Shape.size hout) _ _ ?_
  have key := Tensor.reshape_eq_reshape_of_flat_read (Shape.size_matmulOut hl hr batch).symm
    (castDimScalar (α := ℝ) (congrArg Shape.size hout)
      (IR.MatmulDims.matmulFlat (matmulLayout lv rv batchA batchB batch m n p)
        (flatOfSome ⟨_, x⟩).v (flatOfSome ⟨_, y⟩).v))
    (Tensor.batchMatmul hA hB (Tensor.reshape x _ (Shape.size_matmulLeft hl _ _))
      (Tensor.reshape y _ (Shape.size_matmulRight hr _ _))) fun c => by
      rw [getAtOrZero_castDimScalar, hcongr]
      exact hread c
  have hflat := congrArg (fun t => Tensor.reshape t [_] (Shape.size_vector _).symm) key
  simp only [Tensor.reshape_reshape_self] at hflat
  exact hflat

/-- Reading values at a nonempty list of ports reads the first at the first port. -/
theorem ofErased_cons {p : Port} {ps : List Port} {v : Op.Value ℝ}
    {vs : List (Op.Value ℝ)} {xs : Values ℝ (p :: ps)}
    (h : Values.ofErased (p :: ps) (v :: vs) = .ok xs) :
    ∃ x xs', p.read v = .ok x ∧ Values.ofErased ps vs = .ok xs' ∧ xs = (x, xs') := by
  simp only [Values.ofErased, bind, Except.bind] at h
  cases hv : p.read v with
  | error e => simp [hv] at h
  | ok x =>
    cases hvs : Values.ofErased ps vs with
    | error e => simp [hv, hvs] at h
    | ok xs' =>
      simp only [hv, hvs, pure, Except.pure] at h
      exact ⟨x, xs', rfl, rfl, (Except.ok.inj h).symm⟩

/-- At `ℝ` the scaled matrix product is the matrix product times the scale: the reals have no
quotient replay (`QuotientArithmetic.supported` is false). -/
theorem scaledMatmul_eq_scale {batchA batchB batch : Shape} {m n p : Nat}
    (hA : Shape.CanBroadcastTo batchA batch) (hB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor ℝ (batchA ++ [m, n])) (B : Tensor ℝ (batchB ++ [n, p])) (s : ℝ) :
    Attention.scaledMatmul hA hB A B s =
      Tensor.scale (Tensor.batchMatmul hA hB A B) s := by
  have hcommon : ∀ {L : Shape} (X : Tensor ℝ (L ++ [m, n])) (Y : Tensor ℝ (L ++ [n, p])),
      Tensor.matmulCommonBatch X Y =
        Tensor.zipEach L [m, p] Tensor.matmul X Y := by
    intro L X Y
    induction L with
    | scalar => rfl
    | dim d rest ih =>
        show Tensor.dim _ = Tensor.dim _
        congr 1
        funext j
        exact ih _ _
  apply Tensor.Rep.ext
  intro c
  rw [Attention.scaledMatmul, Attention.scaledProducts_apply]
  simp only [Tensor.scale, Tensor.batchMatmul, hcommon]
  rw [show ∀ {t : Shape} (f : ℝ → ℝ) (x : Tensor ℝ t) (i : Shape.Coord t),
      Tensor.Rep.get (Tensor.map f x) i = f (x i) from fun f x i =>
        Tensor.map_apply f x i]
  rw [Tensor.zipEach_apply]
  simp only [Attention.scaledProductsSlice_one
    Numeric.QuotientArithmetic.supported_real, Bool.false_eq_true, ↓reduceIte]
  exact Tensor.map_apply _ _ _

/-- The flat value of the registered scaled matrix product of two operands and a scale is the flat
contraction over the operator's layout times the scale. -/
theorem flat_scaledMatmul {batchA batchB batch : Shape} {m n p : Nat} {lv rv : Bool}
    (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (hA : Shape.CanBroadcastTo batchA batch) (hB : Shape.CanBroadcastTo batchB batch)
    (x : Tensor ℝ (Shape.matmulLeft lv batchA m n))
    (y : Tensor ℝ (Shape.matmulRight rv batchB n p)) (s : Tensor ℝ []) :
    flatOfSome ⟨Shape.matmulOut lv rv batch m p,
      Tensor.reshape (Attention.scaledMatmul hA hB
          (Tensor.reshape x _ (Shape.size_matmulLeft hl _ _))
          (Tensor.reshape y _ (Shape.size_matmulRight hr _ _)) s.item)
        _ (Shape.size_matmulOut hl hr _).symm⟩ =
      { n := (matmulLayout lv rv batchA batchB batch m n p).outShape.size
        v := Tensor.map (fun z => z * getAtOrZero (flatOfSome ⟨[], s⟩).v [0])
          (IR.MatmulDims.matmulFlat (matmulLayout lv rv batchA batchB batch m n p)
            (flatOfSome ⟨_, x⟩).v (flatOfSome ⟨_, y⟩).v) } := by
  have hs : getAtOrZero (flatOfSome ⟨[], s⟩).v [0] = s.item :=
    (getAtOrZero_eq_getScalar (flatOfSome ⟨[], s⟩).v ⟨0, Nat.one_pos⟩).trans
      (getScalar_reshape_toFin s PUnit.unit)
  rw [scaledMatmul_eq_scale, hs]
  have hm := flat_matmul hl hr hA hB x y
  have hmap : ∀ {a b : FlatTensor ℝ}, a = b →
      (⟨a.n, Tensor.map (fun z => z * s.item) a.v⟩ : FlatTensor ℝ) =
      ⟨b.n, Tensor.map (fun z => z * s.item) b.v⟩ := fun h => by subst h; rfl
  refine Eq.trans ?_ (hmap hm)
  have hreshape : ∀ {t u : Shape} (f : ℝ → ℝ) (z : Tensor ℝ t) (h : t.size = u.size),
      Tensor.reshape (Tensor.map f z) u h = Tensor.map f (Tensor.reshape z u h) := by
    intro t u f z h
    unfold Tensor.reshape
    exact (Tensor.Rep.map_reshape _ _ _).symm
  simp only [flatOfSome, Tensor.scale]
  rw [hreshape, hreshape]

/-- Over references whose reads hold their flat values in `vals`, the references of a concatenation
name the flat values of its inputs read at its ports. -/
theorem mapM_box?_cat {s : Shape} (dim : Fin s.rank) {earlier : Array (List (Op.Value ℝ))}
    {vals : Array (Option (FlatTensor ℝ))} :
    ∀ (sizes : List Nat) {refs : List IR.Ref} {values : List (Op.Value ℝ)}
      {xs : Values ℝ (sizes.map fun n => .value (s.set dim n))},
      refs.mapM (·.read earlier) = .ok values → Values.ofErased _ values = .ok xs →
      (∀ r ∈ refs, ∀ v, r.read earlier = .ok v → box? vals r = flatOf v) →
      refs.mapM (box? vals) =
        some (List.ofFn fun i =>
          flatOfSome ⟨_, Op.catElementParts (e := .scalar) dim sizes xs i⟩)
  | [], refs, values, xs, hin, hx, hvals => by
      cases values with
      | cons v vs => simp [Values.ofErased, throw, throwThe, MonadExceptOf.throw] at hx
      | nil =>
        cases refs with
        | nil => rfl
        | cons r rs =>
          simp only [List.mapM_cons, bind, Except.bind] at hin
          cases hr : r.read earlier with
          | error e => simp [hr] at hin
          | ok y =>
            cases hm : rs.mapM (·.read earlier) with
            | error e => simp [hr, hm] at hin
            | ok l => simp [hr, hm, pure, Except.pure] at hin
  | n :: rest, refs, values, xs, hin, hx, hvals => by
      cases values with
      | nil => simp [Values.ofErased, throw, throwThe, MonadExceptOf.throw] at hx
      | cons v vs =>
        obtain ⟨x, xs', hv, hvs, rfl⟩ := ofErased_cons (p := .value (s.set dim n))
          (ps := rest.map fun n => .value (s.set dim n)) hx
        cases refs with
        | nil => simp [pure, Except.pure] at hin
        | cons r rs =>
          simp only [List.mapM_cons, bind, Except.bind] at hin
          cases hr : r.read earlier with
          | error e => simp [hr] at hin
          | ok y =>
            cases hm : rs.mapM (·.read earlier) with
            | error e => simp [hr, hm] at hin
            | ok l =>
              simp only [hr, hm, pure, Except.pure, Except.ok.injEq, List.cons.injEq] at hin
              obtain ⟨rfl, rfl⟩ := hin
              have ih := mapM_box?_cat dim rest hm hvs fun r' hr' =>
                hvals r' (List.mem_cons_of_mem _ hr')
              rw [List.mapM_cons, hvals r (by simp) _ hr, flatOf_of_read (Or.inl rfl) hv,
                flatOf_value, ih, List.ofFn_succ]
              rfl

/-- The flat value of a concatenation is the checked flat concatenation over its layout
(`catLayout`) of the flat values of its inputs. -/
theorem flat_cat {s : Shape} (dim : Fin s.rank) (sizes : List Nat)
    (xs : (i : Fin sizes.length) → Tensor ℝ (s.set dim (sizes.get i))) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes.get i)) :
    concatFlatValues? (α := ℝ) (catLayout s dim sizes)
        (List.ofFn fun i => flatOfSome ⟨_, xs i⟩).toArray =
      some (flatOfSome ⟨out, Tensor.cat dim sizes.get xs out hout⟩) := by
  have hdim : dim.val < s.length := by rw [← Shape.rank_eq_length]; exact dim.isLt
  have h' : ∀ i : Fin sizes.length,
      s.set dim (sizes.get i) = (catLayout s dim sizes).parentShape i :=
    fun i => List.set_eq_take_cons_drop _ hdim
  rw [cat_eq_concatenateAxes dim sizes.get xs out hout (s.take dim) (s.drop (dim + 1)) sizes rfl
    (fun _ => rfl) (List.set_eq_take_cons_drop · hdim)
    (by simp [List.length_take, Nat.min_eq_left hdim.le]), List.toArray_ofFn]
  have hfam : (fun i => flatOfSome ⟨_, xs i⟩) = fun i =>
      ({ n := ((catLayout s dim sizes).parentShape i).size
         v := Tensor.reshape (Tensor.castShape (xs i) (h' i)) [_] (Shape.size_vector _).symm } :
        FlatTensor ℝ) := by
    funext i
    exact congrArg flatOfSome (SomeTensor.ofTensor_castShape (xs i) (h' i)).symm
  rw [hfam]
  exact (concatFlatValues?_flatten (catLayout s dim sizes) fun i =>
    Tensor.castShape (xs i) (h' i)).trans rfl

/-- A reference reads the same value from any table that agrees below its node. -/
theorem read_of_agree {pre values : Array (List (Op.Value ℝ))} {j : Nat}
    (hagree : ∀ k, k < j → pre[k]? = values[k]?) (hsize : pre.size = j) {r : IR.Ref}
    {v : Op.Value ℝ} (h : r.read pre = .ok v) : r.read values = .ok v := by
  have hlt := IR.Ref.read_lt h
  unfold IR.Ref.read at h ⊢
  rwa [← hagree r.node (hsize ▸ hlt)]

/-- A constant node's outputs are the value `constValue?` gives. -/
theorem const_eval {G : IR.Graph} {P : IR.Payload ℝ} {inputs : List (Op.Value ℝ)}
    {earlier : Array (List (Op.Value ℝ))} {c : Nat} {p : Port}
    {outs : List (Op.Value ℝ)} (hn : G.nodes[c]? = some (.const p))
    (h : (IR.Node.const p).eval P inputs earlier c = .ok outs) :
    ∃ v, constValue? G P c = some v ∧ outs = [v] := by
  simp only [IR.Node.eval] at h
  cases hP : P.const? c with
  | none => simp [hP, throw, throwThe, MonadExceptOf.throw] at h
  | some w =>
    simp only [hP, bind, Except.bind] at h
    cases hr : p.read w with
    | error e => simp [hr] at h
    | ok x =>
      simp only [hr, pure, Except.pure, Except.ok.injEq] at h
      refine ⟨⟨p, x⟩, ?_, h.symm⟩
      simp [constValue?, hn, hP, hr, Except.toOption]

/-- Each element of a list whose `mapM` succeeds has a successful image under any function that
fails whenever the element's read fails. -/
theorem read_of_mapM {β : Type} {earlier : Array (List (Op.Value ℝ))}
    {f : IR.Ref → Except String β}
    (hf : ∀ r y, f r = .ok y → ∃ v, r.read earlier = .ok v) :
    ∀ {l : List IR.Ref} {vs : List β}, l.mapM f = .ok vs →
      ∀ r ∈ l, ∃ v, r.read earlier = .ok v
  | [], _, _, r, hr => by simp at hr
  | x :: xs, vs, h, r, hr => by
      simp only [List.mapM_cons, bind, Except.bind] at h
      cases hx : f x with
      | error e => simp [hx] at h
      | ok y =>
        cases hxs : xs.mapM f with
        | error e => simp [hx, hxs] at h
        | ok ys =>
          rcases List.mem_cons.mp hr with rfl | hr
          · exact hf _ y hx
          · exact read_of_mapM hf hxs r hr

/-- An input node's outputs are its graph input read at its port. -/
theorem input_eval {P : IR.Payload ℝ} {inputs : List (Op.Value ℝ)}
    {earlier : Array (List (Op.Value ℝ))} {id k : Nat} {p : Port}
    {outs : List (Op.Value ℝ)}
    (h : (IR.Node.input k p).eval P inputs earlier id = .ok outs) :
    ∃ x y, inputs[k]? = some x ∧ p.read x = .ok y ∧ outs = [⟨p, y⟩] := by
  simp only [IR.Node.eval] at h
  cases hx : inputs[k]? with
  | none => simp [hx, throw, throwThe, MonadExceptOf.throw] at h
  | some x =>
    simp only [hx, bind, Except.bind] at h
    cases hr : p.read x with
    | error e => simp [hr] at h
    | ok y =>
      simp only [hr, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨x, y, rfl, hr, h.symm⟩

/-- An entry of a successful `mapM` over references is the image of the reference at its
position. -/
theorem mapM_getElem? {β : Type} {f : IR.Ref → Option β} :
    ∀ {refs : List IR.Ref} {xs : List β}, refs.mapM f = some xs →
      ∀ (i : Nat) (y : β), xs[i]? = some y → ∃ r, refs[i]? = some r ∧ f r = some y
  | [], xs, h, i, y, hy => by
      simp only [List.mapM_nil, pure, Option.some.injEq] at h
      subst h
      simp at hy
  | r :: rs, xs, h, i, y, hy => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, pure,
        Option.some.injEq] at h
      obtain ⟨y₀, hy₀, ys, hys, rfl⟩ := h
      cases i with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hy
        exact ⟨r, rfl, hy ▸ hy₀⟩
      | succ i =>
        simp only [List.getElem?_cons_succ] at hy
        obtain ⟨r', hr', hfr'⟩ := mapM_getElem? hys i y hy
        exact ⟨r', by simpa using hr', hfr'⟩

/-- A present entry `box?` returns is the array's entry at the reference's node. -/
theorem getElem!_of_box? {β : Type} {a : Array (Option β)} {r : IR.Ref} {x : β}
    (h : box? a r = some x) : a[r.node]! = some x := by
  unfold box? at h
  split at h
  · cases ha : a[r.node]? with
    | none => simp [ha] at h
    | some y =>
      simp only [ha, Option.join_some] at h
      subst h
      rw [getElem!_pos a r.node (Array.getElem?_eq_some_iff.mp ha).1,
        (Array.getElem?_eq_some_iff.mp ha).2]
  · cases h

end

end TorchLean.CROWN.Op
