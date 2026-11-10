/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Backward

/-!
# Executed Backward Pass versus Proved Backward Pass

The eager trainer executes `Tape.backwardDenseAll`, which runs `Tape.backwardDense`: an
`Option`-valued reverse sweep that runs a node's VJP only when the node has received a cotangent,
then zero-fills the unreached slots. The link theorems in `BackwardGraph` and `FDeriv` are about
`Tape.backwardDenseFrom`, which starts from a total gradient array and runs the VJP of every node
that needs a gradient.

This file closes that gap at the tape level, over a scalar whose zero is an additive identity
(`AddZeroClass`). The hypothesis is `ZeroPreserving`: the VJP of every node that needs a gradient
sends the node's zero cotangent to zero contributions of the parents' shapes. Under it,

`backwardDenseAll_eq_backwardDenseFrom :
  Tape.backwardDenseAll t outId seed = Tape.backwardDenseFrom t (oneHotGrads t outId seed)`

holds for any zero-preserving tape, any valid output id and any seed of the output's shape,
including the error cases. The proof is a step-by-step simulation: `Tape.totalize` maps the
optional accumulator of `backwardDense` to the total accumulator of `backwardDenseFrom`, and each
reverse step commutes with it. A skipped (unreached) node on the executed side corresponds to a
VJP call on the zero cotangent on the proved side, which by `ZeroPreserving` adds only zeros.
Then `0 + g = g` and `g + 0 = g` make the two accumulators agree. IEEE floating point, whose zero
here is `+0`, fails both at `g = -0`, since `+0 + -0 = +0` under round-to-nearest
(IEEE 754-2019 6.3), so on `Float` the two sweeps can return opposite signs of zero even on a
zero-preserving tape.

`BackwardDenseGraph` proves that the tape `GraphData.toTape` produces from the data `g.toData` of
a proof-carrying `Tape.Graph` over a commutative semiring is `ZeroPreserving` and derives the
corollaries for `backwardDenseAll` on those tapes. The eager trainer sweeps its session tape, not
a lowered tape. On CPU that tape is linked node by node in `TorchLean.NN.Tape.Apply.Correctness`.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α]

/-! ### Zero cotangents and totalization -/

/-- The zero cotangent of a runtime node: an all-zero tensor of the node's value shape. -/
def Tape.Node.zeroCotangent [Zero α] (node : Node (SomeTensor α)) : SomeTensor α :=
  SomeTensor.ofTensor (Tensor.full node.value.shape (0 : α))

/-- A zero cotangent has the same shape as the node's value, by construction. -/
@[simp] theorem Tape.Node.shape_zeroCotangent [Zero α] (node : Node (SomeTensor α)) :
    (zeroCotangent node).shape = node.value.shape := rfl

/-- The initial total gradient array of `backwardDenseFrom`: `seed` at `outId`, zeros elsewhere. -/
def Tape.oneHotGrads [Zero α] (t : Tape (SomeTensor α)) (outId : Nat) (seed : SomeTensor α) :
    Array (SomeTensor α) :=
  t.nodes.mapIdx fun id node => if id = outId then seed else Node.zeroCotangent node

/-- `backwardDenseAll` is `backwardDense` followed by `Tape.totalize`. -/
theorem Tape.backwardDenseAll_eq_map_totalize [Add α] [Zero α] (t : Tape (SomeTensor α))
    (outId : Nat)
    (seed : SomeTensor α) :
    Tape.backwardDenseAll (t := t) outId seed =
      Except.map (Tape.totalize t) (Tape.backwardDense (t := t) outId seed) := by
  unfold Tape.backwardDenseAll
  cases Tape.backwardDense (t := t) outId seed <;> rfl

/-- Totalization produces exactly one entry per tape node. -/
@[simp] theorem Tape.size_totalize [Zero α] (t : Tape (SomeTensor α))
    (grads : Array (Option (SomeTensor α))) :
    (Tape.totalize t grads).size = t.nodes.size := by
  simp [Tape.totalize]

/-- The one-hot seed array also has one entry per tape node, so both arrays can be indexed by node
id without bounds checks in the invariants below. -/
@[simp] theorem Tape.size_oneHotGrads [Zero α] (t : Tape (SomeTensor α)) (outId : Nat) (seed
    : SomeTensor α) :
    (oneHotGrads t outId seed).size = t.nodes.size := by
  simp [oneHotGrads]

/-- Entry `id` of the totalized array, expressed through `getNode?`. -/
theorem Tape.getElem?_totalize [Zero α] (t : Tape (SomeTensor α))
    (grads : Array (Option (SomeTensor α))) (id : Nat) :
    (Tape.totalize t grads)[id]? =
      (t.getNode? id).map fun node =>
        match grads[id]? with
        | some (some grad) => grad
        | _ => Node.zeroCotangent node := by
  simp only [Tape.totalize, Array.getElem?_mapIdx, Tape.getNode?]
  congr 1
  funext node
  cases grads[id]? with
  | none => rfl
  | some o => cases o <;> rfl

/-- Entry `id` of the one-hot array, expressed through `getNode?`. -/
theorem Tape.getElem?_oneHotGrads [Zero α] (t : Tape (SomeTensor α)) (outId : Nat) (seed
    : SomeTensor α)
    (id : Nat) :
    (oneHotGrads t outId seed)[id]? =
      (t.getNode? id).map fun node => if id = outId then seed else Node.zeroCotangent node := by
  simp [oneHotGrads, Array.getElem?_mapIdx, Tape.getNode?]

/-- A node that the backward pass reached keeps its computed gradient. -/
theorem Tape.getElem?_totalize_of_some [Zero α] {t : Tape (SomeTensor α)}
    {grads : Array (Option (SomeTensor α))} {id : Nat} {node : Node
        (SomeTensor α)}
    {g : SomeTensor α} (hnode : t.getNode? id = some node) (hg : grads[id]? = some (some g)) :
    (Tape.totalize t grads)[id]? = some g := by
  simp [getElem?_totalize, hnode, hg]

/-- A node the backward pass never reached gets its zero cotangent.

With `Tape.getElem?_totalize_of_some`, it shows that totalization loses no information: a `none`
slot belongs to a node that received no contribution, and zero is the sum of no contributions. -/
theorem Tape.getElem?_totalize_of_none [Zero α] {t : Tape (SomeTensor α)}
    {grads : Array (Option (SomeTensor α))} {id : Nat} {node : Node
        (SomeTensor α)}
    (hnode : t.getNode? id = some node) (hg : grads[id]? = some none) :
    (Tape.totalize t grads)[id]? = some (Node.zeroCotangent node) := by
  simp [getElem?_totalize, hnode, hg]

/-- Totalization commutes with writing a present gradient. -/
theorem Tape.totalize_set [Zero α] (t : Tape (SomeTensor α)) (grads : Array (Option
    (SomeTensor α)))
    (id : Nat) (g : SomeTensor α) (hid : id < grads.size) (hidt : id < t.nodes.size) :
    Tape.totalize t (grads.set id (some g) hid) =
      (Tape.totalize t grads).set id g (by simpa using hidt) := by
  apply Array.ext_getElem?
  intro j
  rw [Array.getElem?_set, getElem?_totalize, getElem?_totalize, Array.getElem?_set]
  by_cases hj : id = j
  · subst hj
    have : t.getNode? id = some (t.nodes[id]'hidt) := by
      simp [Tape.getNode?, Array.getElem?_eq_getElem hidt]
    simp [this]
  · simp [hj]

/-! ### Invariants -/

/--
Well-formedness of the optional accumulator used by `backwardDense`: one slot per tape node, and
every present gradient has its node's value shape.
-/
structure Tape.OptGradsOk (t : Tape (SomeTensor α)) (grads : Array (Option (SomeTensor α)))
    : Prop where
  /-- One slot per tape node. -/
  size_eq : grads.size = t.nodes.size
  /-- Present gradients have the shape of their node's value. -/
  shape : ∀ id g, grads[id]? = some (some g) →
    ∃ node : Node
        (SomeTensor α), t.getNode? id = some node ∧ g.shape = node.value.shape

/--
Well-formedness of the total accumulator used by `backwardDenseFrom`: one slot per tape node, and
every slot has its node's value shape.
-/
structure Tape.TotGradsOk (t : Tape (SomeTensor α)) (grads : Array (SomeTensor α)) : Prop where
  /-- One slot per tape node. -/
  size_eq : grads.size = t.nodes.size
  /-- Every slot has the shape of its node's value. -/
  shape : ∀ id (node : Node (SomeTensor α)), t.getNode? id = some node →
    ∃ g, grads[id]? = some g ∧ g.shape = node.value.shape

/--
Zero preservation of a tape: the VJP of every node that needs a gradient succeeds on the node's
zero cotangent and returns contributions, each the zero cotangent of an existing parent node.

Over an `AddZeroClass` scalar it is a sufficient, not a necessary, condition for skipping
unreached nodes (`backwardDense`) and running every node that needs a gradient
(`backwardDenseFrom`) to produce the same gradients (`backwardDenseAll_eq_backwardDenseFrom`). A
tape whose VJPs are linear over a semiring and return the parents' shapes satisfies it, in
particular the tape `GraphData.toTape` produces from a proof-carrying `Tape.Graph`
(`Tape.Graph.zeroPreserving_toTape`). A
`Float` VJP that multiplies by an infinite weight does not, since `0 * inf` is `NaN`
(IEEE 754-2019 7.2).
-/
structure Tape.ZeroPreserving [Zero α] (t : Tape (SomeTensor α)) : Prop where
  /-- The VJP at the zero cotangent succeeds and emits only parent zero cotangents. -/
  backward_zero : ∀ (id : Nat) (node : Node (SomeTensor α)),
    t.getNode? id = some node → node.requiresGrad = true →
      ∃ contribs : Array (Nat × SomeTensor α),
        node.backward (Node.zeroCotangent node) = .ok contribs ∧
        ∀ pid pg, (pid, pg) ∈ contribs →
          ∃ pnode : Node
              (SomeTensor α), t.getNode? pid = some pnode ∧ pg = Node.zeroCotangent pnode

/-- A successful `getNode?` witnesses that the id is in range. -/
theorem Tape.getNode?_lt_size {t : Tape (SomeTensor α)} {id : Nat} {node : Node
    (SomeTensor α)}
    (h : t.getNode? id = some node) : id < t.nodes.size := by
  simp only [Tape.getNode?] at h
  exact (Array.getElem?_eq_some_iff.1 h).1

/-- Conversely, any in-range id resolves to a node. The pair lets the invariants below be phrased in
terms of `getNode?` alone, without carrying array bounds around. -/
theorem Tape.getNode?_of_lt_size {t : Tape (SomeTensor α)} {id : Nat} (h : id < t.nodes.size) :
    ∃ node, t.getNode? id = some node :=
  ⟨t.nodes[id], by simp [Tape.getNode?, Array.getElem?_eq_getElem h]⟩

/-- Totalizing a well-formed optional accumulator yields a well-formed total accumulator. -/
theorem Tape.totGradsOk_totalize [Zero α] {t : Tape (SomeTensor α)}
    {grads : Array (Option (SomeTensor α))} (hok : OptGradsOk t grads) :
    TotGradsOk t (Tape.totalize t grads) := by
  refine ⟨by simp, ?_⟩
  intro id node hnode
  rw [getElem?_totalize, hnode]
  cases hg : grads[id]? with
  | none => exact ⟨_, rfl, rfl⟩
  | some og =>
    cases og with
    | none => exact ⟨_, rfl, rfl⟩
    | some g =>
      obtain ⟨node', hnode', hshape⟩ := hok.shape id g hg
      rw [hnode] at hnode'
      cases hnode'
      exact ⟨g, rfl, hshape⟩

/-- Writing a correctly shaped gradient preserves `OptGradsOk`. -/
theorem Tape.optGradsOk_set {t : Tape (SomeTensor α)} {grads : Array (Option (SomeTensor α))}
    (hok : OptGradsOk t grads) {id : Nat} {node : Node (SomeTensor α)}
    (hnode : t.getNode? id = some node) {g : SomeTensor α} (hg : g.shape = node.value.shape)
    (hid : id < grads.size) : OptGradsOk t (grads.set id (some g) hid) := by
  refine ⟨by simpa using hok.size_eq, ?_⟩
  intro j g' hj
  rw [Array.getElem?_set] at hj
  by_cases hij : id = j
  · subst hij
    simp only [ite_true, Option.some.injEq] at hj
    cases hj
    exact ⟨node, hnode, hg⟩
  · rw [ite_eq_right hij] at hj
    exact hok.shape j g' hj

/-! ### Adding a zero contribution is a no-op -/

/-- Accumulating onto a node's zero cotangent returns the contribution unchanged. -/
theorem Tape.Node.zeroCotangent_add [AddZeroClass α] (node : Node
    (SomeTensor α))
    (g : SomeTensor α) (h : g.shape = node.value.shape) :
    SomeTensor.add (zeroCotangent node) g = .ok g := by
  obtain ⟨s, gt⟩ := g
  simp only at h
  subst h
  simp [SomeTensor.add, zeroCotangent, SomeTensor.cast]
  exact full_zero_add gt

/-- Accumulating a node's zero cotangent onto an existing gradient leaves it unchanged. -/
theorem Tape.Node.add_zeroCotangent [AddZeroClass α] (node : Node
    (SomeTensor α))
    (g : SomeTensor α) (h : g.shape = node.value.shape) :
    SomeTensor.add g (zeroCotangent node) = .ok g := by
  obtain ⟨s, gt⟩ := g
  simp only at h
  subst h
  simp [SomeTensor.add, zeroCotangent, SomeTensor.cast,
    add_full_zero]

/-- `addGradAll` with a parent's zero cotangent is the identity on a well-formed accumulator. -/
theorem Tape.addGradAll_zeroCotangent [AddZeroClass α] {t : Tape (SomeTensor α)} {grads : Array
    (SomeTensor α)}
    (hok : TotGradsOk t grads) {pid : Nat} {pnode : Node (SomeTensor α)}
    (hp : t.getNode? pid = some pnode) :
    Tape.addGradAll (t := t) grads pid (Node.zeroCotangent pnode) = .ok grads := by
  obtain ⟨existing, hex, hexs⟩ := hok.shape pid pnode hp
  have hid : pid < grads.size := (Array.getElem?_eq_some_iff.1 hex).1
  have hset : grads.set pid existing hid = grads := by
    apply Array.ext_getElem?
    intro j
    rw [Array.getElem?_set]
    by_cases hj : pid = j
    · subst hj
      simp [hex]
    · simp [hj]
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      GradValue.add_someTensor, GradValue.drop_someTensor, Tape.addGradAll, hp, Bind.bind,
      Except.bind, Pure.pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
  by_cases hreq : pnode.requiresGrad = false
  · simp [hreq]
  · have hreq' : pnode.requiresGrad = true := by simpa using hreq
    simp only [hreq', Bool.true_eq_false, ite_false,
      dite_eq_left (Node.shape_zeroCotangent pnode), SomeTensor.ofTensor_cast]
    split
    · rename_i hget
      simp [hex] at hget
    · rename_i value hget
      have hvalue : value = existing := Option.some.inj (hget.symm.trans hex)
      subst value
      simp only [dite_eq_left hexs,
        Node.add_zeroCotangent pnode existing hexs]
      show Except.ok (grads.set pid existing hid) = Except.ok grads
      rw [hset]

/-- Folding zero contributions through `addGradAll` is the identity on a well-formed accumulator. -/
theorem Tape.foldlM_addGradAll_zero [AddZeroClass α] {t : Tape (SomeTensor α)} {grads : Array
    (SomeTensor α)}
    (hok : TotGradsOk t grads) (contribs : List (Nat × SomeTensor α))
    (hz : ∀ pid pg, (pid, pg) ∈ contribs →
      ∃ pnode : Node
          (SomeTensor α), t.getNode? pid = some pnode ∧ pg = Node.zeroCotangent pnode) :
    contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid pg) grads =
      .ok grads := by
  induction contribs with
  | nil => rfl
  | cons head tail ih =>
    obtain ⟨pid, pg⟩ := head
    obtain ⟨pnode, hp, hpg⟩ := hz pid pg (List.mem_cons_self ..)
    rw [List.foldlM_cons]
    simp only [hpg, addGradAll_zeroCotangent hok hp, Bind.bind, Except.bind]
    exact ih fun pid' pg' hmem => hz pid' pg' (List.mem_cons_of_mem _ hmem)

/-! ### One contribution: `addGradDense` simulates `addGradAll` -/

/-- `addGradDense` preserves the optional accumulator invariant. -/
theorem Tape.addGradDense_optGradsOk [Add α] {t : Tape (SomeTensor α)} {grads : Array (Option
    (SomeTensor α))}
    (hok : OptGradsOk t grads) {pid : Nat} {g : SomeTensor α}
    {grads' : Array (Option (SomeTensor α))}
    (h : Tape.addGradDense (t := t) grads pid g = .ok grads') : OptGradsOk t grads' := by
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      GradValue.add_someTensor, GradValue.drop_someTensor,
      Tape.addGradDense, Bind.bind, Except.bind, Pure.pure, Except.pure, throw, throwThe,
      MonadExceptOf.throw] at h
  cases hnode : t.getNode? pid with
  | none => simp [hnode] at h
  | some node =>
    simp only [hnode] at h
    by_cases hreq : node.requiresGrad = false
    · simp only [hreq, ite_true, Except.ok.injEq] at h
      exact h ▸ hok
    · have hreq' : node.requiresGrad = true := by simpa using hreq
      simp only [hreq', Bool.true_eq_false, ite_false] at h
      by_cases hg : g.shape = node.value.shape
      · simp only [dite_eq_left hg] at h
        by_cases hid : pid < grads.size
        · simp only [dite_eq_left hid] at h
          cases hcur : grads[pid]'hid with
          | none =>
            simp only [hcur, Except.ok.injEq] at h
            subst h
            exact optGradsOk_set hok hnode (by simp) hid
          | some existing =>
            simp only [hcur] at h
            split at h
            · exact absurd h (by simp)
            · rename_i summed hsum
              simp only [Except.ok.injEq] at h
              subst h
              have hshape : summed.shape = node.value.shape := by
                obtain ⟨sE, tE⟩ := existing
                have hE : sE = node.value.shape := by
                  obtain ⟨node', hnode', hs⟩ :=
                    hok.shape pid ⟨sE, tE⟩ (by rw [Array.getElem?_eq_getElem hid, hcur])
                  rw [hnode] at hnode'
                  cases hnode'
                  exact hs
                subst hE
                simp [SomeTensor.add] at hsum
                rw [← hsum]
              exact optGradsOk_set hok hnode hshape hid
        · simp [hid] at h
      · simp [hg] at h

/-- One `addGradDense` step, mapped through `Tape.totalize`, is the matching `addGradAll` step. -/
theorem Tape.addGradDense_map_totalize [AddZeroClass α] {t : Tape (SomeTensor α)}
    {grads : Array (Option (SomeTensor α))} (hok : OptGradsOk t grads) (pid : Nat)
    (g : SomeTensor α) :
    Except.map (Tape.totalize t) (Tape.addGradDense (t := t) grads pid g) =
      Tape.addGradAll (t := t) (Tape.totalize t grads) pid g := by
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      GradValue.add_someTensor, GradValue.drop_someTensor,
      Tape.addGradDense, Tape.addGradAll, Bind.bind, Except.bind, Pure.pure, Except.pure, throw,
      throwThe, MonadExceptOf.throw]
  cases hnode : t.getNode? pid with
  | none => rfl
  | some node =>
    by_cases hreq : node.requiresGrad = false
    · simp [hreq, Except.map]
    · have hreq' : node.requiresGrad = true := by simpa using hreq
      simp only [hreq', Bool.true_eq_false, ite_false]
      by_cases hg : g.shape = node.value.shape
      · simp only [dite_eq_left hg]
        have hid : pid < grads.size := hok.size_eq ▸ getNode?_lt_size hnode
        have hidt : pid < t.nodes.size := getNode?_lt_size hnode
        simp only [dite_eq_left hid]
        cases hcur : grads[pid]'hid with
        | none =>
          have hcur? : grads[pid]? = some none := by rw [Array.getElem?_eq_getElem hid, hcur]
          have htotal := getElem?_totalize_of_none hnode hcur?
          simp only [SomeTensor.ofTensor_cast, Except.map,
            totalize_set t grads pid g hid hidt]
          split
          · rename_i hget
            simp [htotal] at hget
          · rename_i existing hget
            have heq : existing = Node.zeroCotangent node :=
              Option.some.inj (hget.symm.trans htotal)
            subst existing
            simp only [dite_eq_left (Node.shape_zeroCotangent node),
              Node.zeroCotangent_add node g hg]
        | some existing =>
          have hcur? : grads[pid]? = some (some existing) := by
            rw [Array.getElem?_eq_getElem hid, hcur]
          obtain ⟨node', hnode', hexs⟩ := hok.shape pid existing hcur?
          rw [hnode] at hnode'
          cases hnode'
          have htotal := getElem?_totalize_of_some hnode hcur?
          simp only [SomeTensor.ofTensor_cast]
          cases hadd : SomeTensor.add existing g with
          | error e =>
            simp only [Except.map]
            split
            · rename_i hget
              simp [htotal] at hget
            · rename_i value hget
              have heq : value = existing := Option.some.inj (hget.symm.trans htotal)
              subst value
              simp only [dite_eq_left hexs, hadd]
          | ok summed =>
            simp only [Except.map, totalize_set t grads pid summed hid hidt]
            split
            · rename_i hget
              simp [htotal] at hget
            · rename_i value hget
              have heq : value = existing := Option.some.inj (hget.symm.trans htotal)
              subst value
              simp only [dite_eq_left hexs, hadd]

      · simp [hg, Except.map]

/-! ### Folding contributions -/

/--
`Except.map f` commutes with `bind` when it commutes with the first action and, on success, with
the continuation. Both simulation folds below, over contributions and over node ids, compose their
steps with it.
-/
theorem Result.map_bind {β γ : Type} (f : β → γ) {x : Result β} {y : Result γ}
    {k : β → Result β} {k' : γ → Result γ} (hx : Except.map f x = y)
    (hk : ∀ b, x = .ok b → Except.map f (k b) = k' (f b)) :
    Except.map f (x >>= k) = y >>= k' := by
  cases x with
  | error e =>
    subst hx
    rfl
  | ok b =>
    subst hx
    exact hk b rfl

/-- Folding `addGradDense` preserves the optional accumulator invariant. -/
theorem Tape.foldlM_addGradDense_optGradsOk [Add α] {t : Tape (SomeTensor α)}
    (contribs : List (Nat × SomeTensor α)) :
    ∀ {grads grads' : Array (Option (SomeTensor α))}, OptGradsOk t grads →
      contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradDense (t := t) acc2 pid pg) grads =
        .ok grads' →
      OptGradsOk t grads' := by
  induction contribs with
  | nil =>
    intro grads grads' hok h
    simp only [List.foldlM_nil, Pure.pure, Except.pure, Except.ok.injEq] at h
    exact h ▸ hok
  | cons head tail ih =>
    intro grads grads' hok h
    obtain ⟨pid, pg⟩ := head
    rw [List.foldlM_cons] at h
    cases h1 : Tape.addGradDense (t := t) grads pid pg with
    | error e => simp [h1, Bind.bind, Except.bind] at h
    | ok grads1 =>
      simp only [h1, Bind.bind, Except.bind] at h
      exact ih (addGradDense_optGradsOk hok h1) h

/-- Folding `addGradDense`, mapped through `Tape.totalize`, is folding `addGradAll`. -/
theorem Tape.foldlM_addGradDense_map_totalize [AddZeroClass α] {t : Tape (SomeTensor α)}
    (contribs : List (Nat × SomeTensor α)) :
    ∀ {grads : Array (Option (SomeTensor α))}, OptGradsOk t grads →
      Except.map (Tape.totalize t)
          (contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradDense (t := t) acc2 pid pg) grads) =
        contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid pg)
          (Tape.totalize t grads) := by
  induction contribs with
  | nil =>
    intro grads _
    rfl
  | cons head tail ih =>
    intro grads hok
    obtain ⟨pid, pg⟩ := head
    rw [List.foldlM_cons, List.foldlM_cons]
    exact Result.map_bind (Tape.totalize t) (addGradDense_map_totalize hok pid pg)
      fun _ h1 => ih (addGradDense_optGradsOk hok h1)

/-! ### One node: the executed step simulates the proved step -/

open Tape (backwardDenseStep)

/-- The executed step preserves the optional accumulator invariant. -/
theorem Tape.backwardDenseStep_optGradsOk [Add α] {t : Tape (SomeTensor α)}
    {grads grads' : Array (Option (SomeTensor α))} (hok : OptGradsOk t grads) {id : Nat}
    (h : backwardDenseStep t grads id = .ok grads') : OptGradsOk t grads' := by
  simp only [backwardDenseStep, Bind.bind, Except.bind, Pure.pure, Except.pure, throw, throwThe,
    MonadExceptOf.throw] at h
  cases hcur : grads[id]? with
  | none => simp [hcur] at h
  | some og =>
    cases og with
    | none =>
      simp only [hcur, Except.ok.injEq] at h
      exact h ▸ hok
    | some dLdy =>
      simp only [hcur] at h
      cases hnode : t.getNode? id with
      | none => simp [hnode] at h
      | some node =>
        simp only [hnode] at h
        by_cases hreq : node.requiresGrad = false
        · simp only [hreq, ite_true, Except.ok.injEq] at h
          exact h ▸ hok
        · have hreq' : node.requiresGrad = true := by simpa using hreq
          simp only [hreq', Bool.true_eq_false, ite_false] at h
          cases hback : node.backward dLdy with
          | error e => simp [hback] at h
          | ok contribs =>
            simp only [hback] at h
            rw [← Array.foldlM_toList] at h
            exact foldlM_addGradDense_optGradsOk contribs.toList hok h

/--
The executed step at a node, mapped through `Tape.totalize`, is the proved step
`backwardDenseFromStep` at the same node, provided the tape is zero preserving.

Both sides skip a node that needs no gradient. For a reached node they run the same VJP on the
same cotangent. For an unreached node the executed side does nothing and the proved side runs the
VJP on the zero cotangent, whose contributions are all zeros and therefore accumulate to nothing.
-/
theorem Tape.backwardDenseStep_map_totalize [AddZeroClass α] {t : Tape (SomeTensor α)}
    (hzp : ZeroPreserving t) {grads : Array (Option (SomeTensor α))}
    (hok : OptGradsOk t grads) {id : Nat} (hid : id < t.nodes.size) :
    Except.map (Tape.totalize t) (backwardDenseStep t grads id) =
      Tape.backwardDenseFromStep (t := t) (Tape.totalize t grads) id := by
  have htot := totGradsOk_totalize hok
  obtain ⟨node, hnode⟩ := getNode?_of_lt_size hid
  have hidg : id < grads.size := hok.size_eq ▸ hid
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      backwardDenseStep, Tape.backwardDenseFromStep, Bind.bind, Except.bind, Pure.pure, Except.pure,
      throw, throwThe, MonadExceptOf.throw, hnode]
  cases hcur : grads[id]? with
  | none => exact absurd (Array.getElem?_eq_none_iff.1 hcur) (Nat.not_le.2 hidg)
  | some og =>
    by_cases hreq : node.requiresGrad = false
    · cases og <;> simp [hreq, Except.map]
    · have hreq' : node.requiresGrad = true := by simpa using hreq
      simp only [hreq', Bool.true_eq_false, ite_false]
      cases og with
      | none =>
        rw [getElem?_totalize_of_none hnode hcur]
        obtain ⟨contribs, hback, hz⟩ := hzp.backward_zero id node hnode hreq'
        simp only [Except.map, dite_eq_left (Node.shape_zeroCotangent node),
          SomeTensor.ofTensor_cast, hback]
        rw [← Array.foldlM_toList]
        exact (foldlM_addGradAll_zero htot contribs.toList
          fun pid pg hmem => hz pid pg (Array.mem_toList_iff.1 hmem)).symm
      | some dLdy =>
        obtain ⟨node', hnode', hshape⟩ := hok.shape id dLdy hcur
        rw [hnode] at hnode'
        cases hnode'
        rw [getElem?_totalize_of_some hnode hcur]
        simp only [dite_eq_left hshape, SomeTensor.ofTensor_cast]
        cases node.backward dLdy with
        | error e => rfl
        | ok contribs =>
          show Except.map (Tape.totalize t)
              (Array.foldlM (fun acc2 (pid, pg) => Tape.addGradDense (t := t) acc2 pid pg) grads
                contribs) =
            Array.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid pg)
              (Tape.totalize t grads) contribs
          rw [← Array.foldlM_toList, ← Array.foldlM_toList]
          exact foldlM_addGradDense_map_totalize contribs.toList hok

/-! ### The reverse loop -/

/-- Peel the first (largest) id off a reverse-range fold. -/
theorem Tape.foldlM_range_reverse_succ {m : Type → Type} [Monad m] {β : Type} (f : β → Nat → m β)
    (b : β) (n : Nat) :
    (List.range (n + 1)).reverse.foldlM f b =
      f b n >>= fun b' => (List.range n).reverse.foldlM f b' := by
  rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append,
    List.foldlM_cons]

/--
The executed reverse loop over the first `n` node ids, mapped through `Tape.totalize`, is the
proved loop `backwardDenseFromLoop` over the same ids.
-/
theorem Tape.foldlM_backwardDenseStep_map_totalize [AddZeroClass α] {t : Tape (SomeTensor α)}
    (hzp : ZeroPreserving t) :
    ∀ (n : Nat), n ≤ t.nodes.size →
      ∀ {grads : Array (Option (SomeTensor α))}, OptGradsOk t grads →
        Except.map (Tape.totalize t) ((List.range n).reverse.foldlM (backwardDenseStep t) grads) =
          Tape.backwardDenseFromLoop (t := t) n (Tape.totalize t grads) := by
  intro n
  induction n with
  | zero =>
    intro _ grads _
    rfl
  | succ n ih =>
    intro hle grads hok
    have hlt : n < t.nodes.size := Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hle
    rw [foldlM_range_reverse_succ]
    simp only [Tape.backwardDenseFromLoop]
    exact Result.map_bind (Tape.totalize t) (backwardDenseStep_map_totalize hzp hok hlt)
      fun _ h1 => ih (Nat.le_of_lt hlt) (backwardDenseStep_optGradsOk hok h1)

/-! ### Main theorem -/

/-- The seeded optional accumulator that `backwardDense` starts from satisfies `OptGradsOk`. -/
theorem Tape.optGradsOk_seed {t : Tape (SomeTensor α)} {outId : Nat} {outNode
    : Node (SomeTensor α)}
    (hout : t.getNode? outId = some outNode) {seed : SomeTensor α}
    (hseed : seed.shape = outNode.value.shape) (hlt : outId < t.nodes.size) :
    OptGradsOk t ((Array.replicate t.nodes.size (none : Option (SomeTensor α))).set outId
      (some seed) (by simpa using hlt)) := by
  refine optGradsOk_set ⟨by simp, ?_⟩ hout hseed (by simpa using hlt)
  intro id g h
  rw [Array.getElem?_replicate] at h
  split at h <;> simp at h

/-- Totalizing the seeded optional accumulator gives the one-hot total accumulator. -/
theorem Tape.totalize_seed [Zero α] (t : Tape (SomeTensor α)) (outId : Nat) (seed
    : SomeTensor α)
    (hlt : outId < t.nodes.size) :
    Tape.totalize t ((Array.replicate t.nodes.size (none : Option (SomeTensor α))).set outId
        (some seed) (by simpa using hlt)) =
      oneHotGrads t outId seed := by
  apply Array.ext_getElem?
  intro j
  rw [getElem?_totalize, getElem?_oneHotGrads, Array.getElem?_set, Array.getElem?_replicate]
  cases hnode : t.getNode? j with
  | none => rfl
  | some node =>
    have hj : j < t.nodes.size := getNode?_lt_size hnode
    by_cases hjo : outId = j
    · subst hjo
      simp
    · have hjo' : j ≠ outId := fun h => hjo h.symm
      simp [hjo, hjo', hj]

/-- `backwardDense` is the reverse fold of `backwardDenseStep` from the seeded accumulator. -/
theorem Tape.backwardDense_eq_foldlM [Add α] {t : Tape (SomeTensor α)} {outId : Nat}
    {outNode : Node (SomeTensor α)} (hout : t.getNode? outId = some outNode)
    {seed : SomeTensor α} (hseed : seed.shape = outNode.value.shape) :
    Tape.backwardDense (t := t) outId seed =
      (List.range t.nodes.size).reverse.foldlM (backwardDenseStep t)
        ((Array.replicate t.nodes.size (none : Option (SomeTensor α))).set outId (some seed)
          (by simpa using getNode?_lt_size hout)) := by
  have hlt : outId < t.nodes.size := getNode?_lt_size hout
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      Tape.backwardDense, hout, Bind.bind, Except.bind, Pure.pure, Except.pure, throw, throwThe,
      MonadExceptOf.throw, dite_eq_left hseed, Array.size_replicate, dite_eq_left hlt,
      SomeTensor.ofTensor_cast]

/--
**Executed backward pass = proved backward pass.** Over an `AddZeroClass` scalar, on a
zero-preserving tape, the totalized executed sweep `backwardDenseAll` (which skips unreached nodes)
returns exactly what the proved sweep `backwardDenseFrom` returns when started from the one-hot
seed array, including agreement of the error cases. The hypotheses on `outId` and `seed` are the
checks `backwardDense` performs before traversing. Without them `backwardDense` fails while
`backwardDenseFrom` may not.
-/
theorem Tape.backwardDenseAll_eq_backwardDenseFrom [AddZeroClass α] {t : Tape (SomeTensor α)} (hzp
    : ZeroPreserving t)
    {outId : Nat} {outNode : Node (SomeTensor α)} (hout
        : t.getNode? outId = some outNode)
    {seed : SomeTensor α} (hseed : seed.shape = outNode.value.shape) :
    Tape.backwardDenseAll (t := t) outId seed =
      Tape.backwardDenseFrom (t := t) (oneHotGrads t outId seed) := by
  have hlt : outId < t.nodes.size := getNode?_lt_size hout
  rw [backwardDenseAll_eq_map_totalize, backwardDense_eq_foldlM hout hseed,
    foldlM_backwardDenseStep_map_totalize hzp t.nodes.size (Nat.le_refl _)
      (optGradsOk_seed hout hseed hlt),
    totalize_seed t outId seed hlt]
  simp [Tape.backwardDenseFrom]

end TorchLean
