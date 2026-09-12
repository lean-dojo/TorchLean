/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.IR.Infer

/-!
# Shape Soundness

Shape inference (`Infer.nodeOutShape`, `Graph.checkShapes`) and the reference semantics
(`Graph.evalNodeRaw`, `Graph.denoteAll`) are two matches over `OpKind`. This file proves that they
agree: on a graph accepted by `checkShapes`, the value the evaluator computes for every node already
has the shape inference assigns to it, so the declared-shape normalization performed by `evalNode`
never rejects a node.

The results are organized as one lemma per operator family, followed by an induction over the node
array that runs `Graph.inferShapesFrom` and `Graph.denoteAllFrom` in lockstep.
-/

@[expose] public section

namespace NN.IR

open _root_.Spec _root_.TorchLean

namespace Graph

/-! ## Except helpers -/

/-- `pure a` in the `Except` monad is `.ok a`. -/
theorem pure_eq_ok {ε β : Type} (a : β) : (pure a : Except ε β) = .ok a := rfl

/-- A bind in the `Except` monad succeeds exactly when both stages succeed. -/
theorem bind_ok_iff {ε β γ : Type} {x : Except ε β} {f : β → Except ε γ} {v : γ} :
    (x >>= f) = .ok v ↔ ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x <;> simp [Bind.bind, Except.bind]

/-- `Except.map` succeeds exactly when its argument succeeds. -/
theorem map_ok_iff {ε β γ : Type} {f : β → γ} {x : Except ε β} {v : γ} :
    Except.map f x = .ok v ↔ ∃ a, x = .ok a ∧ f a = v := by
  cases x <;> simp [Except.map]

/-- Structural Boolean shape equality agrees with propositional equality. -/
theorem shape_areEqual_eq_true_iff : ∀ {a b : Shape}, Shape.areEqual a b = true ↔ a = b
  | .scalar, .scalar => by simp [Shape.areEqual]
  | .scalar, .dim _ _ => by simp [Shape.areEqual]
  | .dim _ _, .scalar => by simp [Shape.areEqual]
  | .dim n₁ s₁, .dim n₂ s₂ => by
      simp [Shape.areEqual, shape_areEqual_eq_true_iff (a := s₁) (b := s₂)]

/-- `==` on shapes is propositional equality. -/
theorem shape_beq_eq_true_iff {a b : Shape} : (a == b) = true ↔ a = b :=
  beq_iff_eq

/-- `!=` on shapes is propositional disequality. -/
theorem shape_bne_eq_true_iff {a b : Shape} : (a != b) = true ↔ a ≠ b := by
  constructor
  · intro h hab
    subst hab
    simp [bne] at h
  · intro hne
    have hfalse : (a == b) = false := by
      cases hbeq : (a == b)
      · rfl
      · exact absurd (shape_beq_eq_true_iff.1 hbeq) hne
    simp [bne, hfalse]

/-- A bind in the `Option` monad succeeds exactly when both stages succeed. -/
theorem option_bind_some_iff {β γ : Type} {x : Option β} {f : β → Option γ} {v : γ} :
    (x >>= f) = some v ↔ ∃ a, x = some a ∧ f a = some v := by
  cases x <;> simp

/-- Binding a success value applies the continuation. -/
theorem ok_bind {ε β γ : Type} (a : β) (f : β → Except ε γ) : (Except.ok a >>= f) = f a := rfl

/-- Binding an error propagates the error. -/
theorem error_bind {ε β γ : Type} (e : ε) (f : β → Except ε γ) :
    (Except.error e >>= f : Except ε γ) = Except.error e := rfl

/-- Every success value of an `Except` computation has shape `s`. -/
def OkShape {α : Type} [TorchLean.Storage α] (s : Shape) (e : Except String (Spec.SomeTensor α)) :
    Prop :=
  ∀ w, e = .ok w → w.shape = s

section OkShape

variable {α : Type} [TorchLean.Storage α]

/-- Errors have no success value. -/
theorem okShape_error (s : Shape) (msg : String) :
    OkShape (α := α) s (.error msg) := fun _ h => by cases h

/-- A success value tagged with `s` has shape `s`. -/
theorem okShape_ok (s : Shape) (t : Tensor α s) : OkShape s (.ok ⟨s, t⟩) := fun _ h => by
  cases h
  rfl

/-- Binding preserves the shape invariant of the continuation. -/
theorem okShape_bind {β : Type} (s : Shape) (x : Except String β)
    (f : β → Except String (Spec.SomeTensor α)) (hf : ∀ a, OkShape s (f a)) :
    OkShape s (x >>= f) := by
  intro w hw
  rw [bind_ok_iff] at hw
  obtain ⟨a, _, ha⟩ := hw
  exact hf a w ha

/-- A dependent conditional preserves the shape invariant of both branches. -/
theorem okShape_dite (s : Shape) (c : Prop) [Decidable c]
    (t : c → Except String (Spec.SomeTensor α)) (e : ¬c → Except String (Spec.SomeTensor α))
    (ht : ∀ h, OkShape s (t h)) (he : ∀ h, OkShape s (e h)) :
    OkShape s (if h : c then t h else e h) := by
  intro w hw
  split at hw
  · exact ht _ w hw
  · exact he _ w hw

/-- A conditional preserves the shape invariant of both branches. -/
theorem okShape_ite (s : Shape) (c : Prop) [Decidable c]
    (t e : Except String (Spec.SomeTensor α)) (ht : c → OkShape s t) (he : ¬c → OkShape s e) :
    OkShape s (if c then t else e) := by
  intro w hw
  split at hw
  · exact ht ‹_› w hw
  · exact he ‹_› w hw

end OkShape

/-- Prove `OkShape s e` for a monadic `Except` computation all of whose success leaves are tagged
with `s`, by walking binds, conditionals, and matches. -/
macro "ok_shape" : tactic =>
  `(tactic| repeat (first
      | exact okShape_error _ _
      | exact okShape_ok _ _
      | (rw [ok_bind])
      | (rw [error_bind])
      | (rw [pure_eq_ok])
      | (rw [throw_eq_error])
      | (apply okShape_bind; intro _)
      | (apply okShape_dite <;> intro _)
      | (apply okShape_ite <;> intro _)
      | split))

/-- Rewrite a monadic `Except` success hypothesis into the conjunction of its stages, discharging
every branch that would have produced an error. Residual `match` expressions on opaque scrutinees
are left for `split`. -/
macro "peel_ok " h:ident : tactic =>
  `(tactic| simp only [bind_ok_iff, option_bind_some_iff, pure_eq_ok, throw_eq_error, ok_bind,
      error_bind, dite_eq_iff, ite_eq_iff, exists_false, false_or, or_false, and_false, false_and,
      Except.ok.injEq,
      reduceCtorEq, not_false_eq_true, true_and, and_true, shape_bne_eq_true_iff,
      shape_beq_eq_true_iff, bne_iff_ne, ne_eq, not_not, Bool.not_eq_true, decide_eq_true_eq,
      Option.some.injEq, map_ok_iff, Option.pure_def] at $h:ident)

/-! ## Decoder characterizations -/

/-- A successful `expectShape` certifies the stored shape tag. -/
theorem shape_eq_of_expectShape_ok {α : Type} [TorchLean.Storage α] [Context α]
    {expected : Shape} {v : Spec.SomeTensor α} {t : Tensor α expected}
    (h : expectShape (α := α) (expected := expected) v = .ok t) : v.shape = expected := by
  unfold expectShape at h
  split at h
  · assumption
  · cases h

/-- A successful parent read is an in-bounds array lookup. -/
theorem getParentValue_ok {α : Type} [TorchLean.Storage α]
    {vals : Array (Spec.SomeTensor α)} {i : Nat} {n : Node} {pid : Nat} {pv : Spec.SomeTensor α}
    (h : getParentValue vals i n pid = .ok pv) : vals[pid]? = some pv := by
  unfold getParentValue at h
  split at h
  · cases h; assumption
  · cases h

/-- The unary decoder succeeds exactly on a one-element parent array. -/
theorem unaryParentId_ok {i : Nat} {n : Node} {pid : Nat} (h : unaryParentId i n = .ok pid) :
    n.parents.size = 1 ∧ n.parents[0]? = some pid := by
  unfold unaryParentId unaryParent? at h
  split at h
  · rename_i hSome
    split at hSome
    · exact ⟨by assumption, by cases h; assumption⟩
    · cases hSome
  · cases h

/-- The binary decoder succeeds exactly on a two-element parent array. -/
theorem binaryParentIds_ok {i : Nat} {n : Node} {a b : Nat}
    (h : binaryParentIds i n = .ok (a, b)) :
    n.parents.size = 2 ∧ n.parents[0]? = some a ∧ n.parents[1]? = some b := by
  unfold binaryParentIds binaryParents? at h
  split at h
  · rename_i hSome
    split at hSome
    · rename_i hSize
      have hp := Option.some.inj hSome
      cases h
      have h0 : 0 < n.parents.size := by simp [hSize]
      have h1 : 1 < n.parents.size := by simp [hSize]
      refine ⟨hSize, ?_, ?_⟩
      · rw [Array.getElem?_eq_getElem h0, ← getElem!_pos n.parents 0 h0]
        exact congrArg (fun p => some p.1) hp
      · rw [Array.getElem?_eq_getElem h1, ← getElem!_pos n.parents 1 h1]
        exact congrArg (fun p => some p.2) hp
    · cases hSome
  · cases h

/-- The unary inference decoder returns the sole parent shape. -/
theorem expectUnaryParent_ok {tag : String} {ps : Array Shape} {s : Shape}
    (h : Infer.expectUnaryParent tag ps = .ok s) : ps.size = 1 ∧ ps[0]? = some s := by
  unfold Infer.expectUnaryParent at h
  split at h
  · rename_i hSize
    cases h
    exact ⟨hSize, Array.getElem?_eq_getElem (by simp [hSize])⟩
  · cases h

/-- The binary inference decoder returns both parent shapes. -/
theorem expectBinaryParents_ok {tag : String} {ps : Array Shape} {s₁ s₂ : Shape}
    (h : Infer.expectBinaryParents tag ps = .ok (s₁, s₂)) :
    ps.size = 2 ∧ ps[0]? = some s₁ ∧ ps[1]? = some s₂ := by
  unfold Infer.expectBinaryParents at h
  split at h
  · rename_i hSize
    cases h
    exact ⟨hSize, Array.getElem?_eq_getElem (by simp [hSize]),
      Array.getElem?_eq_getElem (by simp [hSize])⟩
  · cases h

/-! ## Parent-shape correspondence -/

/--
`parentShapes` lists the shapes of the already evaluated values at a node's parent ids.

This is the interface between the shape table maintained by `Graph.inferShapesFrom` and the value
table maintained by `Graph.denoteAllFrom`: whenever the evaluator can read parent `k`, inference
saw exactly that parent's shape at position `k`.
-/
def ParentShapesOf {α : Type} [TorchLean.Storage α]
    (vals : Array (Spec.SomeTensor α)) (parents : Array Nat) (parentShapes : Array Shape) : Prop :=
  parentShapes.size = parents.size ∧
    ∀ (k : Nat) (hk : k < parents.size) (pv : Spec.SomeTensor α),
      vals[parents[k]]? = some pv → parentShapes[k]? = some pv.shape

/-- The unary parent shape read by inference is the shape of the value read by the evaluator. -/
theorem unary_parent_shape {α : Type} [TorchLean.Storage α]
    {vals : Array (Spec.SomeTensor α)} {i : Nat} {n : Node} {parentShapes : Array Shape}
    (hParents : ParentShapesOf vals n.parents parentShapes)
    {pid : Nat} (hPid : unaryParentId i n = .ok pid)
    {pv : Spec.SomeTensor α} (hVal : getParentValue vals i n pid = .ok pv)
    {tag : String} {s : Shape} (hInfer : Infer.expectUnaryParent tag parentShapes = .ok s) :
    s = pv.shape := by
  obtain ⟨hSize, hGet⟩ := unaryParentId_ok hPid
  obtain ⟨_, hShape⟩ := expectUnaryParent_ok hInfer
  have h0 : 0 < n.parents.size := by simp [hSize]
  have hEq : n.parents[0] = pid := by
    rw [Array.getElem?_eq_getElem h0] at hGet
    exact Option.some.inj hGet
  have := hParents.2 0 h0 pv (by rw [hEq]; exact getParentValue_ok hVal)
  rw [hShape] at this
  exact Option.some.inj this

/-- The binary parent shapes read by inference are the shapes of the values read by the
evaluator. -/
theorem binary_parent_shapes {α : Type} [TorchLean.Storage α]
    {vals : Array (Spec.SomeTensor α)} {i : Nat} {n : Node} {parentShapes : Array Shape}
    (hParents : ParentShapesOf vals n.parents parentShapes)
    {a b : Nat} (hIds : binaryParentIds i n = .ok (a, b))
    {av bv : Spec.SomeTensor α} (hA : getParentValue vals i n a = .ok av)
    (hB : getParentValue vals i n b = .ok bv)
    {tag : String} {s₁ s₂ : Shape}
    (hInfer : Infer.expectBinaryParents tag parentShapes = .ok (s₁, s₂)) :
    s₁ = av.shape ∧ s₂ = bv.shape := by
  obtain ⟨hSize, hGetA, hGetB⟩ := binaryParentIds_ok hIds
  obtain ⟨_, hShapeA, hShapeB⟩ := expectBinaryParents_ok hInfer
  have h0 : 0 < n.parents.size := by simp [hSize]
  have h1 : 1 < n.parents.size := by simp [hSize]
  have hEqA : n.parents[0] = a := by
    rw [Array.getElem?_eq_getElem h0] at hGetA
    exact Option.some.inj hGetA
  have hEqB : n.parents[1] = b := by
    rw [Array.getElem?_eq_getElem h1] at hGetB
    exact Option.some.inj hGetB
  have hsA := hParents.2 0 h0 av (by rw [hEqA]; exact getParentValue_ok hA)
  have hsB := hParents.2 1 h1 bv (by rw [hEqB]; exact getParentValue_ok hB)
  rw [hShapeA] at hsA
  rw [hShapeB] at hsB
  exact ⟨Option.some.inj hsA, Option.some.inj hsB⟩

/-! ## Operator families

Each lemma below fixes one operator (or family) and shows that the raw evaluator value has the
inferred shape. The hypotheses are the same throughout: the parent-shape correspondence, a
successful inference, and a successful raw evaluation. `permute`, `transpose`, and `conv` are the
exception (see `evalNodeRaw_shape_declared`).
-/

/-- The eval-mode BatchNorm contract returns the parent shape. -/
theorem inferBatchNormEvalOutShape_ok {channelAxis channels : Nat} {s r : Shape}
    (h : OpContracts.inferBatchNormEvalOutShape channelAxis channels s = .ok r) : r = s := by
  simp only [OpContracts.inferBatchNormEvalOutShape] at h
  peel_ok h
  obtain ⟨_, _, _, _, h⟩ := h
  split at h <;> peel_ok h
  exact h.2.symm

/-- Eval-mode BatchNorm preserves the shape of its input. -/
theorem evalBatchNorm_ok_shape {α : Type} [TorchLean.Storage α] [Context α]
    {payload : Payload α} {id channelAxis channels : Nat} {x y : Spec.SomeTensor α}
    (h : evalBatchNorm payload id channelAxis channels x = .ok y) : y.shape = x.shape := by
  simp only [evalBatchNorm] at h
  peel_ok h
  obtain ⟨_, _, h⟩ := h
  rcases Option.eq_none_or_eq_some (payload.batchNormEval? id) with hp | ⟨params, hp⟩
  · rw [hp] at h
    peel_ok h
  · rw [hp] at h
    peel_ok h
    obtain ⟨_, h⟩ := h
    by_cases hx : x.shape =
      (Shape.ofList (x.shape.toList.take channelAxis)).concat
        (Shape.dim params.c (Shape.ofList (x.shape.toList.drop (channelAxis + 1))))
    · rw [show @decEq Shape inferInstance x.shape _ = isTrue hx from Subsingleton.elim _ _] at h
      peel_ok h
      subst h
      exact hx.symm
    · rw [show @decEq Shape inferInstance x.shape _ = isFalse hx from Subsingleton.elim _ _] at h
      peel_ok h

/-- Leading-axis concat produces the declared leading extent over the shared tail. -/
theorem evalConcatLeadingAxisFold_ok_shape {α : Type} [TorchLean.Storage α] [Context α]
    {i nOut : Nat} {rest : Shape} {parents : Array (Spec.SomeTensor α)} {v : Spec.SomeTensor α}
    (h : evalConcatLeadingAxisFold i nOut rest parents = .ok v) : v.shape = .dim nOut rest := by
  simp only [evalConcatLeadingAxisFold] at h
  peel_ok h
  obtain ⟨_, _, h⟩ := h
  split at h <;> peel_ok h
  obtain ⟨_, hv⟩ := h
  subst hv
  rfl

/-- A successful list `mapM` in `Except` succeeds pointwise. -/
theorem list_mapM_ok {β γ : Type} {f : β → Except String γ} :
    ∀ {l : List β} {r : List γ}, l.mapM f = .ok r →
      r.length = l.length ∧
        ∀ (k : Nat) (hk : k < l.length) (hr : k < r.length), f l[k] = .ok r[k]
  | [], r, h => by
      rw [List.mapM_nil] at h
      peel_ok h
      subst h
      exact ⟨rfl, fun k hk => absurd hk (Nat.not_lt_zero k)⟩
  | b :: l, r, h => by
      rw [List.mapM_cons] at h
      peel_ok h
      obtain ⟨c, hc, cs, hcs, hr⟩ := h
      subst hr
      obtain ⟨hlen, hget⟩ := list_mapM_ok hcs
      refine ⟨by simp [hlen], fun k hk hr => ?_⟩
      cases k with
      | zero =>
          simp only [List.getElem_cons_zero]
          exact hc
      | succ k =>
          simp only [List.getElem_cons_succ]
          exact hget k (by simpa using hk) (by simpa using hr)

/-- A successful array `mapM` in `Except` succeeds pointwise. -/
theorem array_mapM_ok {β γ : Type} {f : β → Except String γ} {as : Array β} {bs : Array γ}
    (h : as.mapM f = .ok bs) :
    bs.size = as.size ∧ ∀ (k : Nat) (hk : k < as.size) (hb : k < bs.size), f as[k] = .ok bs[k] := by
  rw [Array.mapM_eq_mapM_toList] at h
  change Except.map List.toArray (as.toList.mapM f) = .ok bs at h
  rw [map_ok_iff] at h
  obtain ⟨l, hl, hbs⟩ := h
  subst hbs
  obtain ⟨hlen, hget⟩ := list_mapM_ok hl
  refine ⟨by simp [hlen], fun k hk hb => ?_⟩
  have := hget k (by simpa using hk) (by simpa using hb)
  simpa using this

section Families

variable {α : Type} [TorchLean.Storage α] [Context α]
variable {payload : Payload α} {input : Spec.SomeTensor α} {vals : Array (Spec.SomeTensor α)}
variable {i : Nat} {n : Node} {parentShapes : Array Shape} {inferred : Shape}
variable {v : Spec.SomeTensor α}

/-- Binary elementwise operations (`add`, `sub`, `mulElem`, `maxElem`, `minElem`). -/
theorem evalNodeRaw_shape_binary_elementwise
    (hk : n.kind ∈ [OpKind.add, .sub, .mulElem, .maxElem, .minElem])
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk
  rcases hk with hk | hk | hk | hk | hk <;>
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨⟨s₁, s₂⟩, hShapes, _, hInferred⟩ := hInfer
    obtain ⟨⟨a, b⟩, hIds, av, hA, aT, hEA, bv, hB, bT, hEB, hv⟩ := hEval
    obtain ⟨hs₁, _⟩ := binary_parent_shapes hParents hIds hA hB hShapes
    subst hv hInferred
    simp only at hs₁ ⊢
    rw [hs₁, shape_eq_of_expectShape_ok hEA]

/-- Source nodes (`input`, `const`, `randUniform`, `bernoulliMask`): inference returns the shape the
evaluator tags its result with. -/
theorem evalNodeRaw_shape_source
    (hk : n.kind = .input ∨ (∃ s, n.kind = .const s) ∨ (∃ seed, n.kind = .randUniform seed) ∨
      ∃ seed, n.kind = .bernoulliMask seed)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  rcases hk with hk | ⟨s, hk⟩ | ⟨seed, hk⟩ | ⟨seed, hk⟩
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, hv⟩ := hEval
    subst hv hInfer
    rfl
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, hv⟩ := hEval
    subst hv hInfer
    rfl
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, hv⟩ := hEval
    subst hv
    exact hInfer.2
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨s, _, hInfer⟩ := hInfer
    obtain ⟨pId, _, pV, _, hEval⟩ := hEval
    split at hInfer <;> peel_ok hInfer
    split at hEval <;> peel_ok hEval
    subst hEval
    exact hInfer

/-- Unary shape-preserving operations that only read one parent at the declared shape. -/
theorem evalNodeRaw_shape_unary_elementwise
    (hk : n.kind ∈
      [OpKind.abs, .sqrt, .inv, .relu, .tanh, .sigmoid, .softplus, .exp, .sin, .cos, .detach])
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk
  rcases hk with hk | hk | hk | hk | hk | hk | hk | hk | hk | hk | hk <;>
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hEval
    obtain ⟨pId, hPid, p0, hP, p, hE, hv⟩ := hEval
    subst hv
    rw [unary_parent_shape hParents hPid hP hInfer, shape_eq_of_expectShape_ok hE]

/-- `log`: like the other unary operations, plus the data-dependent positivity check. -/
theorem evalNodeRaw_shape_log (hk : n.kind = .log)
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  simp only [Infer.nodeOutShape, hk] at hInfer
  simp only [evalNodeRaw, hk] at hEval
  peel_ok hEval
  obtain ⟨pId, hPid, p0, hP, p, hE, _, hv⟩ := hEval
  subst hv
  rw [unary_parent_shape hParents hPid hP hInfer, shape_eq_of_expectShape_ok hE]

/-- `softmax`, `hardMaskedSoftmax`, and `layernorm`: shape preserving after their axis, mask, or
normalized-suffix validation. -/
theorem evalNodeRaw_shape_normalization
    (hk : (∃ axis, n.kind = .softmax axis) ∨ (∃ mask, n.kind = .hardMaskedSoftmax mask) ∨
      ∃ axis, n.kind = .layernorm axis)
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  rcases hk with ⟨axis, hk⟩ | ⟨mask, hk⟩ | ⟨axis, hk⟩
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨s, hs, _, _, hInferred⟩ := hInfer
    obtain ⟨pId, hPid, hEval⟩ := hEval
    split at hEval <;> peel_ok hEval
    obtain ⟨p0, hP, p, hE, hv⟩ := hEval
    subst hv hInferred
    rw [unary_parent_shape hParents hPid hP hs, shape_eq_of_expectShape_ok hE]
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨s, hs, _, _, hInferred⟩ := hInfer
    obtain ⟨pId, hPid, p0, hP, p, hE, hEval⟩ := hEval
    have hv : v.shape = n.outShape := (show OkShape n.outShape _ from by ok_shape) v hEval
    rw [hv, ← hInferred, unary_parent_shape hParents hPid hP hs, shape_eq_of_expectShape_ok hE]
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨s, hs, _, _, hInferred⟩ := hInfer
    obtain ⟨pId, hPid, p0, hP, p, hE, hEval⟩ := hEval
    have hv : v.shape = n.outShape := (show OkShape n.outShape _ from by ok_shape) v hEval
    rw [hv, ← hInferred, unary_parent_shape hParents hPid hP hs, shape_eq_of_expectShape_ok hE]

/-- Axis reductions (`reduceSum`, `reduceMean`) and the full reduction `sum`. -/
theorem evalNodeRaw_shape_reduction
    (hk : (∃ axis, n.kind = .reduceSum axis) ∨ (∃ axis, n.kind = .reduceMean axis) ∨ n.kind = .sum)
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  rcases hk with ⟨axis, hk⟩ | ⟨axis, hk⟩ | hk
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨s, hs, _, _, hInferred⟩ := hInfer
    obtain ⟨pId, hPid, pV, hP, hEval⟩ := hEval
    split at hEval <;> peel_ok hEval
    subst hEval hInferred
    rw [unary_parent_shape hParents hPid hP hs]
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨s, hs, _, _, hInferred⟩ := hInfer
    obtain ⟨pId, hPid, pV, hP, hEval⟩ := hEval
    split at hEval <;> peel_ok hEval
    subst hEval hInferred
    rw [unary_parent_shape hParents hPid hP hs]
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, hInferred⟩ := hInfer
    obtain ⟨_, _, _, _, hv⟩ := hEval
    subst hv hInferred
    rfl

/-- Pure shape operations (`broadcastTo`, `reshape`, `flatten`) tag their result with the shape
written in the operation. -/
theorem evalNodeRaw_shape_shape_op
    (hk : (∃ s₁ s₂, n.kind = .broadcastTo s₁ s₂) ∨ (∃ inS outS, n.kind = .reshape inS outS) ∨
      ∃ s, n.kind = .flatten s)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  rcases hk with ⟨s₁, s₂, hk⟩ | ⟨inS, outS, hk⟩ | ⟨s, hk⟩
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, _, _, hInfer⟩ := hInfer
    obtain ⟨_, _, _, _, _, _, _, hEval⟩ := hEval
    subst hEval hInfer
    rfl
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, _, _, hInferred⟩ := hInfer
    obtain ⟨_, _, _, _, _, _, _, hv⟩ := hEval
    subst hv hInferred
    rfl
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, _, hInferred⟩ := hInfer
    obtain ⟨_, _, _, _, _, _, hv⟩ := hEval
    subst hv hInferred
    rfl

/-- `linear` and `mseLoss`: the evaluator tags its result with the declared shape, respectively the
scalar shape, and inference returns the same. -/
theorem evalNodeRaw_shape_linear_loss (hk : n.kind = .linear ∨ n.kind = .mseLoss)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  rcases hk with hk | hk
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk, evalLinear] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, hInfer⟩ := hInfer
    obtain ⟨_, _, _, _, hEval⟩ := hEval
    split at hInfer <;> peel_ok hInfer
    split at hEval <;> peel_ok hEval
    obtain ⟨_, _, _, hv⟩ := hEval
    obtain ⟨_, hInferred⟩ := hInfer
    subst hv hInferred
    rfl
  · simp only [Infer.nodeOutShape, hk] at hInfer
    simp only [evalNodeRaw, hk, mseLossSomeTensor] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨_, _, _, hInferred⟩ := hInfer
    obtain ⟨_, _, _, _, _, _, _, hv⟩ := hEval
    subst hv hInferred
    rfl

/-- `matmul`: both passes decompose the operand shapes with `OpContracts.matmulDims`. -/
theorem evalNodeRaw_shape_matmul (hk : n.kind = .matmul)
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  simp only [Infer.nodeOutShape, hk, OpContracts.inferMatmulOutShape] at hInfer
  simp only [evalNodeRaw, hk] at hEval
  peel_ok hInfer
  peel_ok hEval
  obtain ⟨⟨s₁, s₂⟩, hShapes, dims', hDims', hInferred⟩ := hInfer
  obtain ⟨⟨a, b⟩, hIds, aV, hA, bV, hB, hEval⟩ := hEval
  split at hEval <;> peel_ok hEval
  rename_i dims hDims
  obtain ⟨aT, _, bT, _, hv⟩ := hEval
  obtain ⟨hs₁, hs₂⟩ := binary_parent_shapes hParents hIds hA hB hShapes
  simp only at hs₁ hs₂ hDims'
  subst hs₁ hs₂
  rw [hDims] at hDims'
  cases hDims'
  subst hv hInferred
  rfl

/-- `maxPool` and `avgPool`: both passes plan the pooled suffix with `OpContracts.planPool`. -/
theorem evalNodeRaw_shape_pool
    (hk : (∃ config, n.kind = .maxPool config) ∨ ∃ config, n.kind = .avgPool config)
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  rcases hk with ⟨config, hk⟩ | ⟨config, hk⟩ <;>
  · simp only [Infer.nodeOutShape, hk, OpContracts.inferWindowOutShape] at hInfer
    simp only [evalNodeRaw, hk, evalMaxPool, evalAvgPool] at hEval
    peel_ok hInfer
    peel_ok hEval
    obtain ⟨s, hs, plan', hPlan', hInferred⟩ := hInfer
    obtain ⟨pId, hPid, x, hX, plan, hPlan, hv⟩ := hEval
    have hsx := unary_parent_shape hParents hPid hX hs
    subst hsx
    rw [hPlan] at hPlan'
    cases hPlan'
    subst hv hInferred
    rfl

/-- `batchNormEval`: the payload-backed evaluator preserves the parent shape, which is also what
the contract returns. -/
theorem evalNodeRaw_shape_batchNorm
    (hk : ∃ channelAxis channels, n.kind = .batchNormEval channelAxis channels)
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  obtain ⟨channelAxis, channels, hk⟩ := hk
  simp only [Infer.nodeOutShape, hk] at hInfer
  simp only [evalNodeRaw, hk] at hEval
  peel_ok hInfer
  peel_ok hEval
  obtain ⟨s, hs, hInfer⟩ := hInfer
  obtain ⟨pId, hPid, x, hX, y, hY, _, hv⟩ := hEval
  subst hv
  rw [evalBatchNorm_ok_shape hY, inferBatchNormEvalOutShape_ok hInfer,
    unary_parent_shape hParents hPid hX hs]

/-- `concat`: the evaluator recomputes `OpContracts.inferConcatOutShape` on the parent values and
insists that it equals the declared shape, which is what it produces. -/
theorem evalNodeRaw_shape_concat (hk : ∃ axis, n.kind = .concat axis)
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = inferred := by
  obtain ⟨axis, hk⟩ := hk
  simp only [Infer.nodeOutShape, hk] at hInfer
  simp only [evalNodeRaw, hk, evalConcat] at hEval
  peel_ok hEval
  obtain ⟨parents, hMap, hEval⟩ := hEval
  rcases hd : OpContracts.inferConcatOutShape axis (parents.map (fun pv => pv.shape)) with
    _ | expected <;> rw [hd] at hEval <;> peel_ok hEval
  obtain ⟨hExpected, hEval⟩ := hEval
  subst hExpected
  -- Inference saw exactly the parent value shapes.
  have hPS : parentShapes = parents.map (fun pv => pv.shape) := by
    obtain ⟨hlen, hget⟩ := array_mapM_ok hMap
    have hSize := hParents.1
    apply Array.ext
    · simp only [Array.size_map, hlen, hSize]
    · intro k hk₁ hk₂
      simp only [Array.size_map] at hk₂
      have hpv := getParentValue_ok (hget k (hlen ▸ hk₂) hk₂)
      have hshape := hParents.2 k (hlen ▸ hk₂) _ hpv
      rw [Array.getElem?_eq_getElem hk₁] at hshape
      simpa using Option.some.inj hshape
  rw [hPS, hd] at hInfer
  cases hInfer
  -- The produced value has the declared shape on both concat paths.
  rcases hEval with ⟨_, hEval⟩ | ⟨_, hEval⟩
  · revert hEval
    split
    · intro hEval
      rw [evalConcatLeadingAxisFold_ok_shape hEval]
      exact Eq.symm ‹n.outShape = _›
    · intro hEval
      peel_ok hEval
  · exact (show OkShape n.outShape _ from by ok_shape) v hEval

/-- `permute`, `transpose`, and `conv` compare the shape they realize against the declared
`outShape`, so a successful raw evaluation has the declared shape. Relating that realized shape to
the inference rule (`Shape.permute?`, respectively the convolution contract) is not needed for
soundness because `checkShapes` separately forces the inferred shape to be the declared one. -/
theorem evalNodeRaw_shape_declared
    (hk : (∃ perm, n.kind = .permute perm) ∨ (∃ a₁ a₂, n.kind = .transpose a₁ a₂) ∨
      ∃ config, n.kind = .conv config)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = n.outShape := by
  rcases hk with ⟨perm, hk⟩ | ⟨a₁, a₂, hk⟩ | ⟨config, hk⟩
  · simp only [evalNodeRaw, hk] at hEval
    peel_ok hEval
    obtain ⟨_, _, _, _, _, _, _, hv⟩ := hEval
    subst hv
    rfl
  · simp only [evalNodeRaw, hk] at hEval
    peel_ok hEval
    obtain ⟨_, _, _, _, _, _, _, _, _, _, hv⟩ := hEval
    subst hv
    rfl
  · simp only [evalNodeRaw, hk] at hEval
    peel_ok hEval
    obtain ⟨_, _, _, _, y, _, hShape, hv⟩ := hEval
    subst hv
    exact hShape

/--
Node-level soundness of shape inference.

If inference assigns the declared shape to a node (as `Graph.checkShapes` requires) and the node's
parents carry the shapes inference saw, then the raw evaluator value already has the declared shape.
-/
theorem evalNodeRaw_shape_of_infer
    (hParents : ParentShapesOf vals n.parents parentShapes)
    (hInfer : Infer.nodeOutShape n parentShapes = .ok inferred)
    (hDecl : inferred = n.outShape)
    (hEval : evalNodeRaw payload input vals i n = .ok v) : v.shape = n.outShape := by
  cases hk : n.kind with
  | input => exact hDecl ▸ evalNodeRaw_shape_source (Or.inl hk) hInfer hEval
  | const s => exact hDecl ▸ evalNodeRaw_shape_source (Or.inr (Or.inl ⟨s, hk⟩)) hInfer hEval
  | permute perm => exact evalNodeRaw_shape_declared (Or.inl ⟨perm, hk⟩) hEval
  | transpose a₁ a₂ => exact evalNodeRaw_shape_declared (Or.inr (Or.inl ⟨a₁, a₂, hk⟩)) hEval
  | detach => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | randUniform seed =>
      exact hDecl ▸ evalNodeRaw_shape_source (Or.inr (Or.inr (Or.inl ⟨seed, hk⟩))) hInfer hEval
  | bernoulliMask seed =>
      exact hDecl ▸ evalNodeRaw_shape_source (Or.inr (Or.inr (Or.inr ⟨seed, hk⟩))) hInfer hEval
  | add => exact hDecl ▸ evalNodeRaw_shape_binary_elementwise (by simp [hk]) hParents hInfer hEval
  | sub => exact hDecl ▸ evalNodeRaw_shape_binary_elementwise (by simp [hk]) hParents hInfer hEval
  | mulElem =>
      exact hDecl ▸ evalNodeRaw_shape_binary_elementwise (by simp [hk]) hParents hInfer hEval
  | abs => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | sqrt => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | inv => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | maxElem =>
      exact hDecl ▸ evalNodeRaw_shape_binary_elementwise (by simp [hk]) hParents hInfer hEval
  | minElem =>
      exact hDecl ▸ evalNodeRaw_shape_binary_elementwise (by simp [hk]) hParents hInfer hEval
  | maxPool config =>
      exact hDecl ▸ evalNodeRaw_shape_pool (Or.inl ⟨config, hk⟩) hParents hInfer hEval
  | avgPool config =>
      exact hDecl ▸ evalNodeRaw_shape_pool (Or.inr ⟨config, hk⟩) hParents hInfer hEval
  | broadcastTo s₁ s₂ =>
      exact hDecl ▸ evalNodeRaw_shape_shape_op (Or.inl ⟨s₁, s₂, hk⟩) hInfer hEval
  | reduceSum axis =>
      exact hDecl ▸ evalNodeRaw_shape_reduction (Or.inl ⟨axis, hk⟩) hParents hInfer hEval
  | reduceMean axis =>
      exact hDecl ▸ evalNodeRaw_shape_reduction (Or.inr (Or.inl ⟨axis, hk⟩)) hParents hInfer hEval
  | sum => exact hDecl ▸ evalNodeRaw_shape_reduction (Or.inr (Or.inr hk)) hParents hInfer hEval
  | matmul => exact hDecl ▸ evalNodeRaw_shape_matmul hk hParents hInfer hEval
  | linear => exact hDecl ▸ evalNodeRaw_shape_linear_loss (Or.inl hk) hInfer hEval
  | conv config => exact evalNodeRaw_shape_declared (Or.inr (Or.inr ⟨config, hk⟩)) hEval
  | batchNormEval channelAxis channels =>
      exact hDecl ▸ evalNodeRaw_shape_batchNorm ⟨channelAxis, channels, hk⟩ hParents hInfer hEval
  | relu => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | tanh => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | sigmoid =>
      exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | softplus =>
      exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | safeLog =>
      simp only [evalNodeRaw, hk] at hEval
      exact (show OkShape n.outShape _ from by ok_shape) v hEval
  | exp => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | log => exact hDecl ▸ evalNodeRaw_shape_log hk hParents hInfer hEval
  | sin => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | cos => exact hDecl ▸ evalNodeRaw_shape_unary_elementwise (by simp [hk]) hParents hInfer hEval
  | softmax axis =>
      exact hDecl ▸ evalNodeRaw_shape_normalization (Or.inl ⟨axis, hk⟩) hParents hInfer hEval
  | hardMaskedSoftmax mask =>
      exact hDecl ▸ evalNodeRaw_shape_normalization (Or.inr (Or.inl ⟨mask, hk⟩)) hParents hInfer
        hEval
  | layernorm axis =>
      exact hDecl ▸ evalNodeRaw_shape_normalization (Or.inr (Or.inr ⟨axis, hk⟩)) hParents hInfer
        hEval
  | reshape inS outS =>
      exact hDecl ▸ evalNodeRaw_shape_shape_op (Or.inr (Or.inl ⟨inS, outS, hk⟩)) hInfer hEval
  | flatten s => exact hDecl ▸ evalNodeRaw_shape_shape_op (Or.inr (Or.inr ⟨s, hk⟩)) hInfer hEval
  | concat axis => exact hDecl ▸ evalNodeRaw_shape_concat ⟨axis, hk⟩ hParents hInfer hEval
  | mseLoss => exact hDecl ▸ evalNodeRaw_shape_linear_loss (Or.inr hk) hInfer hEval

end Families

/-! ## Graph-level soundness -/

section GraphLevel

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Case analysis on an `Except` value stated as a disjunction of equations. -/
theorem except_cases {ε β : Type} (x : Except ε β) : (∃ e, x = .error e) ∨ ∃ a, x = .ok a := by
  cases x <;> simp

/-- Sequencing a computation before `y` does not change a successful result of `y`. -/
theorem seq_ok {β γ : Type} {x : Except String β} {y : Except String γ} {v : γ}
    (h : (x >>= fun _ => y) = .ok v) : y = .ok v := by
  rw [bind_ok_iff] at h
  obtain ⟨_, _, h⟩ := h
  exact h

/-- A successful `denoteAll` is a successful `denoteAllFrom` started at node `0`. -/
theorem denoteAll_ok_from {g : Graph} {payload : Payload α} {input : Spec.SomeTensor α}
    {vals : Array (Spec.SomeTensor α)} (h : denoteAll g payload input = .ok vals) :
    denoteAllFrom g payload input 0 #[] = .ok vals := by
  unfold denoteAll at h
  split at h <;> (try simp only at h) <;> first | exact h | exact seq_ok h

/-- A successful checked lookup returns the node stored at the requested index. -/
theorem getNode_ok {g : Graph} {i : Nat} {n : Node} (h : g.getNode i = .ok n) :
    g.nodes[i]? = some n := by
  unfold getNode getNode? at h
  rcases Option.eq_none_or_eq_some g.nodes[i]? with hnone | ⟨n', hsome⟩
  · rw [hnone] at h
    peel_ok h
  · rw [hsome] at h
    peel_ok h
    obtain ⟨_, hn⟩ := h
    subst hn
    exact hsome

/-- The normalized value of a node has the declared shape. -/
theorem normalizeNodeOutput_ok_shape {i : Nat} {n : Node} {v w : Spec.SomeTensor α}
    (h : normalizeNodeOutput i n v = .ok w) : w.shape = n.outShape := by
  unfold normalizeNodeOutput at h
  split at h
  · peel_ok h
    subst h
    rfl
  · peel_ok h

/-- Normalization is the identity on a value that already has the declared shape. -/
theorem normalizeNodeOutput_eq_ok_self {i : Nat} {n : Node} {v : Spec.SomeTensor α}
    (hv : v.shape = n.outShape) : normalizeNodeOutput i n v = .ok v := by
  obtain ⟨s, t⟩ := v
  simp only at hv
  subst hv
  exact normalizeNodeOutput_declared i n t

/-- Every value produced by `evalNode` has the declared shape of its node. -/
theorem evalNode_ok_shape {payload : Payload α} {input : Spec.SomeTensor α}
    {vals : Array (Spec.SomeTensor α)} {i : Nat} {n : Node} {v : Spec.SomeTensor α}
    (h : evalNode payload input vals i n = .ok v) : v.shape = n.outShape := by
  unfold evalNode at h
  peel_ok h
  obtain ⟨_, _, h⟩ := h
  exact normalizeNodeOutput_ok_shape h

/-- A successful parent lookup returns one already inferred shape per parent id. -/
theorem lookupParentShapes_ok {inferred : Array Shape} :
    ∀ {pids : List Nat} {shapes : List Shape}, lookupParentShapes inferred pids = some shapes →
      shapes.length = pids.length ∧
        ∀ (k : Nat) (hk : k < pids.length) (hs : k < shapes.length),
          inferred[pids[k]]? = some shapes[k]
  | [], shapes, h => by
      simp only [lookupParentShapes, Option.some.injEq] at h
      subst h
      exact ⟨rfl, fun k hk => absurd hk (Nat.not_lt_zero k)⟩
  | pid :: rest, shapes, h => by
      simp only [lookupParentShapes] at h
      peel_ok h
      obtain ⟨shape, hshape, shapes', hrest, hs⟩ := h
      subst hs
      obtain ⟨hlen, hget⟩ := lookupParentShapes_ok hrest
      refine ⟨by simp [hlen], fun k hk hs => ?_⟩
      cases k with
      | zero =>
          simp only [List.getElem_cons_zero]
          exact hshape
      | succ k =>
          simp only [List.getElem_cons_succ]
          exact hget k (by simpa using hk) (by simpa using hs)

omit [Context α] in
/-- The shapes looked up by inference are the shapes of the corresponding evaluated values. -/
theorem parentShapesOf_of_lookup {vals : Array (Spec.SomeTensor α)} {inferred : Array Shape}
    (hSize : vals.size = inferred.size)
    (hShapes : ∀ (j : Nat) (hj : j < inferred.size), (vals[j]'(hSize ▸ hj)).shape = inferred[j])
    {parents : Array Nat} {shapes : List Shape}
    (hLookup : lookupParentShapes inferred parents.toList = some shapes) :
    ParentShapesOf vals parents shapes.toArray := by
  obtain ⟨hlen, hget⟩ := lookupParentShapes_ok hLookup
  refine ⟨by simpa using hlen, fun k hk pv hpv => ?_⟩
  have hk' : k < parents.toList.length := by simpa using hk
  have hks : k < shapes.length := hlen ▸ hk'
  have hinf := hget k hk' hks
  rw [Array.getElem_toList] at hinf
  have hpid : parents[k] < inferred.size := by
    by_contra hcon
    rw [Array.getElem?_eq_none (Nat.le_of_not_lt hcon)] at hinf
    cases hinf
  rw [Array.getElem?_eq_getElem hpid] at hinf
  have hpv' : vals[parents[k]]'(hSize ▸ hpid) = pv := by
    rw [Array.getElem?_eq_getElem (hSize ▸ hpid)] at hpv
    exact Option.some.inj hpv
  have hshape := hShapes parents[k] hpid
  rw [hpv'] at hshape
  rw [List.getElem?_toArray, List.getElem?_eq_getElem hks, hshape]
  exact hinf.symm

/--
Evaluate nodes `i, i+1, ...` with `evalNodeRaw`, that is, without the per-node declared-shape
normalization performed by `evalNode`.

This is a proof-only reference evaluator: `denoteAllRawFrom_eq_denoteAllFrom` shows that on a
graph accepted by `checkShapes` it computes the same table as `denoteAllFrom`.
-/
def denoteAllRawFrom (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) (i : Nat)
    (vals : Array (Spec.SomeTensor α)) : Except String (Array (Spec.SomeTensor α)) := do
  if _h : i < g.nodes.size then
    let n ← g.getNode i
    let v ← evalNodeRaw payload input vals i n
    denoteAllRawFrom g payload input (i + 1) (vals.push v)
  else
    pure vals
termination_by g.nodes.size - i
decreasing_by
  exact Nat.sub_succ_lt_self _ _ _h

/-- `denoteAll` without the per-node declared-shape normalization. -/
def denoteAllRaw (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) :
    Except String (Array (Spec.SomeTensor α)) := do
  if g.wellFormed then
    pure ()
  else
    g.checkWellFormed
  denoteAllRawFrom g payload input 0 #[]

/--
Lockstep induction: if shape inference accepts the nodes from `i` on, starting from a table of
inferred shapes that matches the evaluated prefix, then raw and normalized evaluation agree from
`i` on.
-/
theorem denoteAllRawFrom_eq_denoteAllFrom (g : Graph) (payload : Payload α)
    (input : Spec.SomeTensor α) (i : Nat) (vals : Array (Spec.SomeTensor α))
    (inferred inferredAll : Array Shape)
    (hInfer : g.inferShapesFrom i inferred = .ok inferredAll)
    (hVals : vals.size = i) (hInferred : inferred.size = i)
    (hShapes : ∀ (j : Nat) (hj : j < inferred.size),
      (vals[j]'(by rw [hVals, ← hInferred]; exact hj)).shape = inferred[j]) :
    denoteAllRawFrom g payload input i vals = denoteAllFrom g payload input i vals := by
  unfold denoteAllRawFrom denoteAllFrom
  by_cases hi : i < g.nodes.size
  · rw [dif_pos hi, dif_pos hi]
    unfold inferShapesFrom at hInfer
    rw [dif_pos hi] at hInfer
    peel_ok hInfer
    obtain ⟨n, hN, hInfer⟩ := hInfer
    rcases Option.eq_none_or_eq_some (lookupParentShapes inferred n.parents.toList) with
      hL | ⟨shapes, hL⟩
    · rw [hL] at hInfer
      peel_ok hInfer
    · rw [hL] at hInfer
      peel_ok hInfer
      obtain ⟨out, hOut, hDecl, hRec⟩ := hInfer
      unfold evalAt evalNode
      rw [hN, ok_bind, ok_bind]
      rcases except_cases (evalNodeRaw payload input vals i n) with ⟨e, hE⟩ | ⟨v, hV⟩
      · rw [hE, error_bind, error_bind, error_bind]
      · have hvShape : v.shape = n.outShape :=
          evalNodeRaw_shape_of_infer
            (parentShapesOf_of_lookup (hVals.trans hInferred.symm) hShapes hL) hOut hDecl hV
        rw [hV, ok_bind, ok_bind, normalizeNodeOutput_eq_ok_self hvShape, ok_bind]
        refine denoteAllRawFrom_eq_denoteAllFrom g payload input (i + 1) (vals.push v)
          (inferred.push out) inferredAll hRec (by simp [hVals]) (by simp [hInferred]) ?_
        intro j hj
        simp only [Array.size_push] at hj
        by_cases hlt : j < i
        · rw [Array.getElem_push_lt (by rw [hVals]; exact hlt),
            Array.getElem_push_lt (by rw [hInferred]; exact hlt)]
          exact hShapes j (by rw [hInferred]; exact hlt)
        · have hj' : j < i + 1 := by rw [hInferred] at hj; exact hj
          have hji : j = i := Nat.le_antisymm (Nat.lt_succ_iff.mp hj') (Nat.le_of_not_lt hlt)
          subst hji
          simp only [Array.getElem_push, hVals, hInferred, lt_irrefl, dite_false]
          rw [hvShape, hDecl]
  · rw [dif_neg hi, dif_neg hi]
termination_by g.nodes.size - i
decreasing_by
  exact Nat.sub_succ_lt_self _ _ hi

/-- On a graph accepted by `checkShapes`, the per-node declared-shape normalization is redundant:
evaluating with `evalNodeRaw` and with `evalNode` produce the same result. -/
theorem denoteAllRaw_eq_denoteAll (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (hShapes : g.checkShapes = .ok ()) :
    denoteAllRaw g payload input = denoteAll g payload input := by
  rcases except_cases (g.inferShapesFrom 0 #[]) with ⟨e, hE⟩ | ⟨arr, hArr⟩
  · exfalso
    unfold checkShapes inferShapes at hShapes
    rw [hE] at hShapes
    peel_ok hShapes
  · have h := denoteAllRawFrom_eq_denoteAllFrom g payload input 0 #[] #[] arr hArr rfl rfl
      (fun j hj => absurd hj (by simp))
    unfold denoteAllRaw denoteAll
    simp only [h]

/-- Values already in the table are untouched by evaluating further nodes. -/
theorem denoteAllFrom_prefix (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (i : Nat) (vals out : Array (Spec.SomeTensor α))
    (h : denoteAllFrom g payload input i vals = .ok out) :
    ∀ (j : Nat) (hj : j < vals.size), out[j]? = some vals[j] := by
  unfold denoteAllFrom at h
  by_cases hi : i < g.nodes.size
  · rw [dif_pos hi] at h
    peel_ok h
    obtain ⟨v, _, hRec⟩ := h
    intro j hj
    have := denoteAllFrom_prefix g payload input (i + 1) (vals.push v) out hRec j
      (by simp only [Array.size_push]; exact Nat.lt_succ_of_lt hj)
    rw [this, Array.getElem_push_lt hj]
  · rw [dif_neg hi] at h
    peel_ok h
    subst h
    intro j hj
    exact Array.getElem?_eq_getElem hj
termination_by g.nodes.size - i
decreasing_by
  exact Nat.sub_succ_lt_self _ _ hi

/-- Evaluation from `i` fills exactly the remaining nodes, each with its declared shape. -/
theorem denoteAllFrom_ok_shapes (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (i : Nat) (vals out : Array (Spec.SomeTensor α))
    (h : denoteAllFrom g payload input i vals = .ok out) (hSize : vals.size = i)
    (hi : i ≤ g.nodes.size) :
    out.size = g.nodes.size ∧
      ∀ (j : Nat) (hj : j < g.nodes.size) (hjo : j < out.size), i ≤ j →
        out[j].shape = g.nodes[j].outShape := by
  unfold denoteAllFrom at h
  by_cases hlt : i < g.nodes.size
  · rw [dif_pos hlt] at h
    unfold evalAt at h
    peel_ok h
    obtain ⟨v, ⟨n, hN, hV⟩, hRec⟩ := h
    obtain ⟨hOutSize, hTail⟩ := denoteAllFrom_ok_shapes g payload input (i + 1) (vals.push v) out
      hRec (by simp [hSize]) hlt
    refine ⟨hOutSize, fun j hj hjo hij => ?_⟩
    by_cases hji : i + 1 ≤ j
    · exact hTail j hj hjo hji
    · have hji' : j = i :=
        Nat.le_antisymm (Nat.lt_succ_iff.mp (Nat.lt_of_not_le hji)) hij
      subst hji'
      have hprefix := denoteAllFrom_prefix g payload input (j + 1) (vals.push v) out hRec j
        (by simp [hSize])
      have hnode : g.nodes[j]? = some n := getNode_ok hN
      rw [Array.getElem?_eq_getElem hj] at hnode
      rw [Array.getElem?_eq_getElem hjo] at hprefix
      rw [Option.some.inj hprefix, Option.some.inj hnode]
      simp only [Array.getElem_push, hSize, lt_irrefl, dite_false]
      exact evalNode_ok_shape hV
  · rw [dif_neg hlt] at h
    peel_ok h
    subst h
    refine ⟨hSize.trans (Nat.le_antisymm hi (Nat.le_of_not_lt hlt)), fun j hj hjo hij => ?_⟩
    exact absurd (Nat.lt_of_le_of_lt hij hj) hlt

/--
The literal soundness statement: on a well-shaped graph, every evaluated node value has its
declared shape.

Note that `hShapes` is not needed for the conclusion, because `evalNode` normalizes each value to
the declared shape (`denoteAll_shape`); its role is documented by `denoteAllRaw_eq_denoteAll`, which
shows that on a `checkShapes`-accepted graph the normalization never changes anything.
-/
theorem denoteAll_shape (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (h : denoteAll g payload input = .ok vals) :
    vals.size = g.nodes.size ∧
      ∀ (i : Nat) (hi : i < g.nodes.size) (hiv : i < vals.size),
        vals[i].shape = g.nodes[i].outShape := by
  obtain ⟨hSize, hAll⟩ :=
    denoteAllFrom_ok_shapes g payload input 0 #[] vals (denoteAll_ok_from h) rfl (Nat.zero_le _)
  exact ⟨hSize, fun i hi hiv => hAll i hi hiv (Nat.zero_le i)⟩

/-- Shape inference is sound for the reference semantics: if `checkShapes` accepts a graph and the
graph evaluates, every node value has the shape inference assigned to it, namely the declared
one. -/
theorem checkShapes_sound (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (hShapes : g.checkShapes = .ok ())
    (hEval : denoteAll g payload input = .ok vals) :
    ∀ (i : Nat) (hi : i < g.nodes.size) (hiv : i < vals.size),
      vals[i].shape = g.nodes[i].outShape := by
  have _ := hShapes
  exact (denoteAll_shape g payload input vals hEval).2

end GraphLevel

end Graph

end NN.IR
