/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.Backward

/-!
# Executed Backward Pass versus Proved Backward Pass

The eager trainer executes `Tape.backwardDenseAll`, which runs `Tape.backwardDense`: an
`Option`-valued reverse sweep that runs a node's VJP only when the node has received a cotangent,
then zero-fills the unreached slots. The link theorems in `BackwardGraph` and `FDeriv` are about
`Tape.backwardDenseFrom`, which starts from a total gradient array and runs every VJP.

This file closes that gap at the tape level. The hypothesis is `ZeroPreserving`: every VJP on the
tape sends the node's zero cotangent to zero contributions of the parents' shapes. Under it,

`backwardDenseAll_eq_backwardDenseFrom :
  Tape.backwardDenseAll t outId seed = Tape.backwardDenseFrom t (oneHotGrads t outId seed)`

holds for any tape, any valid output id and any seed of the output's shape, including the error
cases. The proof is a step-by-step simulation: `totalizeGrads` maps the optional accumulator of
`backwardDense` to the total accumulator of `backwardDenseFrom`, and each reverse step commutes
with it. A skipped (unreached) node on the executed side corresponds to a VJP call on the zero
cotangent on the proved side, which by `ZeroPreserving` adds only zeros.

`BackwardDenseGraph` proves that every tape produced by `lowerGraphToTape` is `ZeroPreserving`
and derives the corollaries for the executed backward pass.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Graph

open Runtime
open Runtime.Autograd

variable {α : Type} [TorchLean.Storage α]

/-! ### Zero cotangents and totalization -/

/-- The zero cotangent of a runtime node: an all-zero tensor of the node's value shape. -/
def zeroCotangent [Zero α] (node : Runtime.Autograd.Node α) : Spec.SomeTensor α :=
  Spec.SomeTensor.ofTensor (Tensor.full node.value.shape (0 : α))

/-- A zero cotangent has the same shape as the node's value, by construction. -/
@[simp] theorem shape_zeroCotangent [Zero α] (node : Runtime.Autograd.Node α) :
    (zeroCotangent node).shape = node.value.shape := rfl

/--
Totalize the optional gradient array of `backwardDense`: reached nodes keep their gradient, every
other node gets its zero cotangent. This is exactly the post-processing `backwardDenseAll` does.
-/
def totalizeGrads [Zero α] (t : Tape α) (grads : Array (Option (Spec.SomeTensor α))) :
    Array (Spec.SomeTensor α) :=
  t.nodes.mapIdx fun id node =>
    match grads[id]? with
    | some (some grad) => grad
    | _ => zeroCotangent node

/-- The initial total gradient array of `backwardDenseFrom`: `seed` at `outId`, zeros elsewhere. -/
def oneHotGrads [Zero α] (t : Tape α) (outId : Nat) (seed : Spec.SomeTensor α) :
    Array (Spec.SomeTensor α) :=
  t.nodes.mapIdx fun id node => if id = outId then seed else zeroCotangent node

/-- `backwardDenseAll` is `backwardDense` followed by `totalizeGrads`. -/
theorem backwardDenseAll_eq_map_totalizeGrads [Add α] [Zero α] (t : Tape α) (outId : Nat)
    (seed : Spec.SomeTensor α) :
    Tape.backwardDenseAll (t := t) outId seed =
      Except.map (totalizeGrads t) (Tape.backwardDense (t := t) outId seed) := by
  unfold Tape.backwardDenseAll totalizeGrads zeroCotangent
  cases Tape.backwardDense (t := t) outId seed <;> rfl

/-- Totalization produces exactly one entry per tape node. -/
@[simp] theorem size_totalizeGrads [Zero α] (t : Tape α)
    (grads : Array (Option (Spec.SomeTensor α))) :
    (totalizeGrads t grads).size = t.nodes.size := by
  simp [totalizeGrads]

/-- So does the one-hot seed array, which is why both can be indexed by node id without bounds
checks in the invariants below. -/
@[simp] theorem size_oneHotGrads [Zero α] (t : Tape α) (outId : Nat) (seed : Spec.SomeTensor α) :
    (oneHotGrads t outId seed).size = t.nodes.size := by
  simp [oneHotGrads]

/-- Entry `id` of the totalized array, expressed through `getNode?`. -/
theorem getElem?_totalizeGrads [Zero α] (t : Tape α)
    (grads : Array (Option (Spec.SomeTensor α))) (id : Nat) :
    (totalizeGrads t grads)[id]? =
      (t.getNode? id).map fun node =>
        match grads[id]? with
        | some (some grad) => grad
        | _ => zeroCotangent node := by
  simp [totalizeGrads, Array.getElem?_mapIdx, Tape.getNode?]

/-- Entry `id` of the one-hot array, expressed through `getNode?`. -/
theorem getElem?_oneHotGrads [Zero α] (t : Tape α) (outId : Nat) (seed : Spec.SomeTensor α)
    (id : Nat) :
    (oneHotGrads t outId seed)[id]? =
      (t.getNode? id).map fun node => if id = outId then seed else zeroCotangent node := by
  simp [oneHotGrads, Array.getElem?_mapIdx, Tape.getNode?]

/-- A node that the backward pass reached keeps its computed gradient. -/
theorem getElem?_totalizeGrads_of_some [Zero α] {t : Tape α}
    {grads : Array (Option (Spec.SomeTensor α))} {id : Nat} {node : Runtime.Autograd.Node α}
    {g : Spec.SomeTensor α} (hnode : t.getNode? id = some node) (hg : grads[id]? = some (some g)) :
    (totalizeGrads t grads)[id]? = some g := by
  simp [getElem?_totalizeGrads, hnode, hg]

/-- A node the backward pass never reached gets its zero cotangent.

Taken together, these two lemmas say totalization loses no information: `none` in the sparse array
means the node genuinely received no contribution, so filling in zero is not an approximation. -/
theorem getElem?_totalizeGrads_of_none [Zero α] {t : Tape α}
    {grads : Array (Option (Spec.SomeTensor α))} {id : Nat} {node : Runtime.Autograd.Node α}
    (hnode : t.getNode? id = some node) (hg : grads[id]? = some none) :
    (totalizeGrads t grads)[id]? = some (zeroCotangent node) := by
  simp [getElem?_totalizeGrads, hnode, hg]

/-- Totalization commutes with writing a present gradient. -/
theorem totalizeGrads_set [Zero α] (t : Tape α) (grads : Array (Option (Spec.SomeTensor α)))
    (id : Nat) (g : Spec.SomeTensor α) (hid : id < grads.size) (hidt : id < t.nodes.size) :
    totalizeGrads t (grads.set id (some g) hid) =
      (totalizeGrads t grads).set id g (by simpa using hidt) := by
  apply Array.ext_getElem?
  intro j
  rw [Array.getElem?_set, getElem?_totalizeGrads, getElem?_totalizeGrads, Array.getElem?_set]
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
structure OptGradsOk (t : Tape α) (grads : Array (Option (Spec.SomeTensor α))) : Prop where
  /-- One slot per tape node. -/
  size_eq : grads.size = t.nodes.size
  /-- Present gradients have the shape of their node's value. -/
  shape : ∀ id g, grads[id]? = some (some g) →
    ∃ node : Runtime.Autograd.Node α, t.getNode? id = some node ∧ g.shape = node.value.shape

/--
Well-formedness of the total accumulator used by `backwardDenseFrom`: one slot per tape node, and
every slot has its node's value shape.
-/
structure TotGradsOk (t : Tape α) (grads : Array (Spec.SomeTensor α)) : Prop where
  /-- One slot per tape node. -/
  size_eq : grads.size = t.nodes.size
  /-- Every slot has the shape of its node's value. -/
  shape : ∀ id (node : Runtime.Autograd.Node α), t.getNode? id = some node →
    ∃ g, grads[id]? = some g ∧ g.shape = node.value.shape

/--
Zero preservation of a tape: every node's VJP sends the node's zero cotangent to an array of
contributions each of which is the zero cotangent of an existing parent node.

This is the exact condition under which skipping unreached nodes (`backwardDense`) and running
every node (`backwardDenseFrom`) produce the same gradients. It holds for linear VJPs of correct
parent shapes, in particular for every tape produced by `lowerGraphToTape`.
-/
structure ZeroPreserving [Zero α] (t : Tape α) : Prop where
  /-- The VJP at the zero cotangent succeeds and emits only parent zero cotangents. -/
  backward_zero : ∀ (id : Nat) (node : Runtime.Autograd.Node α),
    t.getNode? id = some node → node.requiresGrad = true →
      ∃ contribs : Array (Nat × Spec.SomeTensor α),
        node.backward (zeroCotangent node) = .ok contribs ∧
        ∀ pid pg, (pid, pg) ∈ contribs →
          ∃ pnode : Runtime.Autograd.Node α, t.getNode? pid = some pnode ∧ pg = zeroCotangent pnode

/-- A successful `getNode?` witnesses that the id is in range. -/
theorem getNode?_lt_size {t : Tape α} {id : Nat} {node : Runtime.Autograd.Node α}
    (h : t.getNode? id = some node) : id < t.nodes.size := by
  simp only [Tape.getNode?] at h
  exact (Array.getElem?_eq_some_iff.1 h).1

/-- Conversely, any in-range id resolves to a node. The pair lets the invariants below be phrased in
terms of `getNode?` alone, without carrying array bounds around. -/
theorem getNode?_of_lt_size {t : Tape α} {id : Nat} (h : id < t.nodes.size) :
    ∃ node, t.getNode? id = some node :=
  ⟨t.nodes[id], by simp [Tape.getNode?, Array.getElem?_eq_getElem h]⟩

/-- Totalizing a well-formed optional accumulator yields a well-formed total accumulator. -/
theorem totGradsOk_totalizeGrads [Zero α] {t : Tape α}
    {grads : Array (Option (Spec.SomeTensor α))} (hok : OptGradsOk t grads) :
    TotGradsOk t (totalizeGrads t grads) := by
  refine ⟨by simp, ?_⟩
  intro id node hnode
  rw [getElem?_totalizeGrads, hnode]
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
theorem optGradsOk_set {t : Tape α} {grads : Array (Option (Spec.SomeTensor α))}
    (hok : OptGradsOk t grads) {id : Nat} {node : Runtime.Autograd.Node α}
    (hnode : t.getNode? id = some node) {g : Spec.SomeTensor α} (hg : g.shape = node.value.shape)
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

/-- Adding the all-zero tensor on the left is the identity. -/
theorem addSpec_full_zero_left [AddZeroClass α] {s : Shape} (x : Tensor α s) :
    addSpec (Tensor.full s (0 : α)) x = x := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [addSpec, map2Spec, Tensor.full]

/-- Adding the all-zero tensor on the right is the identity. -/
theorem addSpec_full_zero_right [AddZeroClass α] {s : Shape} (x : Tensor α s) :
    addSpec x (Tensor.full s (0 : α)) = x := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [addSpec, map2Spec, Tensor.full]

/-- Accumulating onto a node's zero cotangent returns the contribution unchanged. -/
theorem someTensor_add_zeroCotangent_left [AddZeroClass α] (node : Runtime.Autograd.Node α)
    (g : Spec.SomeTensor α) (h : g.shape = node.value.shape) :
    Runtime.Autograd.SomeTensor.add (zeroCotangent node) g = .ok g := by
  obtain ⟨s, gt⟩ := g
  simp only at h
  subst h
  simp [Runtime.Autograd.SomeTensor.add, zeroCotangent, Spec.SomeTensor.cast,
    addSpec_full_zero_left]

/-- Accumulating a node's zero cotangent onto an existing gradient leaves it unchanged. -/
theorem someTensor_add_zeroCotangent_right [AddZeroClass α] (node : Runtime.Autograd.Node α)
    (g : Spec.SomeTensor α) (h : g.shape = node.value.shape) :
    Runtime.Autograd.SomeTensor.add g (zeroCotangent node) = .ok g := by
  obtain ⟨s, gt⟩ := g
  simp only at h
  subst h
  simp [Runtime.Autograd.SomeTensor.add, zeroCotangent, Spec.SomeTensor.cast,
    addSpec_full_zero_right]

/-- `addGradAll` with a parent's zero cotangent is the identity on a well-formed accumulator. -/
theorem addGradAll_zeroCotangent [AddZeroClass α] {t : Tape α} {grads : Array (Spec.SomeTensor α)}
    (hok : TotGradsOk t grads) {pid : Nat} {pnode : Runtime.Autograd.Node α}
    (hp : t.getNode? pid = some pnode) :
    Tape.addGradAll (t := t) grads pid (zeroCotangent pnode) = .ok grads := by
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
  simp only [Tape.addGradAll, hp, Bind.bind, Except.bind, Pure.pure, Except.pure, throw, throwThe,
    MonadExceptOf.throw]
  by_cases hreq : pnode.requiresGrad = false
  · simp [hreq]
  · have hreq' : pnode.requiresGrad = true := by simpa using hreq
    simp only [hreq', Bool.true_eq_false, ite_false, hex, dite_eq_left hexs, dite_eq_left hid]
    rw [dite_eq_left (shape_zeroCotangent pnode), Spec.SomeTensor.ofTensor_cast,
      Spec.SomeTensor.ofTensor_cast, someTensor_add_zeroCotangent_right pnode existing hexs]
    show Except.ok (grads.set pid existing hid) = Except.ok grads
    rw [hset]

/-- Folding zero contributions through `addGradAll` is the identity on a well-formed accumulator. -/
theorem foldlM_addGradAll_zero [AddZeroClass α] {t : Tape α} {grads : Array (Spec.SomeTensor α)}
    (hok : TotGradsOk t grads) (contribs : List (Nat × Spec.SomeTensor α))
    (hz : ∀ pid pg, (pid, pg) ∈ contribs →
      ∃ pnode : Runtime.Autograd.Node α, t.getNode? pid = some pnode ∧ pg = zeroCotangent pnode) :
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
theorem addGradDense_optGradsOk [Add α] {t : Tape α} {grads : Array (Option (Spec.SomeTensor α))}
    (hok : OptGradsOk t grads) {pid : Nat} {g : Spec.SomeTensor α}
    {grads' : Array (Option (Spec.SomeTensor α))}
    (h : Tape.addGradDense (t := t) grads pid g = .ok grads') : OptGradsOk t grads' := by
  simp only [Tape.addGradDense, Bind.bind, Except.bind, Pure.pure, Except.pure, throw, throwThe,
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
                simp [Runtime.Autograd.SomeTensor.add] at hsum
                rw [← hsum]
              exact optGradsOk_set hok hnode hshape hid
        · simp [hid] at h
      · simp [hg] at h

/-- One `addGradDense` step, mapped through `totalizeGrads`, is the matching `addGradAll` step. -/
theorem addGradDense_map_totalizeGrads [AddZeroClass α] {t : Tape α}
    {grads : Array (Option (Spec.SomeTensor α))} (hok : OptGradsOk t grads) (pid : Nat)
    (g : Spec.SomeTensor α) :
    Except.map (totalizeGrads t) (Tape.addGradDense (t := t) grads pid g) =
      Tape.addGradAll (t := t) (totalizeGrads t grads) pid g := by
  simp only [Tape.addGradDense, Tape.addGradAll, Bind.bind, Except.bind, Pure.pure, Except.pure,
    throw, throwThe, MonadExceptOf.throw]
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
        have hidTot : pid < (totalizeGrads t grads).size := by simpa using hidt
        simp only [dite_eq_left hid]
        cases hcur : grads[pid]'hid with
        | none =>
          have hcur? : grads[pid]? = some none := by rw [Array.getElem?_eq_getElem hid, hcur]
          rw [getElem?_totalizeGrads_of_none hnode hcur?]
          simp only [Spec.SomeTensor.ofTensor_cast, dite_eq_left hidTot, Except.map,
            totalizeGrads_set t grads pid g hid hidt, dite_eq_left (shape_zeroCotangent node),
            someTensor_add_zeroCotangent_left node g hg]
        | some existing =>
          have hcur? : grads[pid]? = some (some existing) := by
            rw [Array.getElem?_eq_getElem hid, hcur]
          obtain ⟨node', hnode', hexs⟩ := hok.shape pid existing hcur?
          rw [hnode] at hnode'
          cases hnode'
          rw [getElem?_totalizeGrads_of_some hnode hcur?]
          simp only [dite_eq_left hexs, Spec.SomeTensor.ofTensor_cast, dite_eq_left hidTot]
          cases Runtime.Autograd.SomeTensor.add existing g with
          | error e => rfl
          | ok summed =>
            simp only [Except.map, totalizeGrads_set t grads pid summed hid hidt]
      · simp [hg, Except.map]

/-! ### Folding contributions -/

/--
`Except.map` commutes with `bind` when it commutes with the first action and, on success, with
the continuation. This is the single lemma behind every "executed step simulates proved step"
composition below.
-/
theorem except_map_bind {β γ : Type} (f : β → γ) {x : Result β} {y : Result γ}
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
theorem foldlM_addGradDense_optGradsOk [Add α] {t : Tape α}
    (contribs : List (Nat × Spec.SomeTensor α)) :
    ∀ {grads grads' : Array (Option (Spec.SomeTensor α))}, OptGradsOk t grads →
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

/-- Folding `addGradDense`, mapped through `totalizeGrads`, is folding `addGradAll`. -/
theorem foldlM_addGradDense_map_totalizeGrads [AddZeroClass α] {t : Tape α}
    (contribs : List (Nat × Spec.SomeTensor α)) :
    ∀ {grads : Array (Option (Spec.SomeTensor α))}, OptGradsOk t grads →
      Except.map (totalizeGrads t)
          (contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradDense (t := t) acc2 pid pg) grads) =
        contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid pg)
          (totalizeGrads t grads) := by
  induction contribs with
  | nil =>
    intro grads _
    rfl
  | cons head tail ih =>
    intro grads hok
    obtain ⟨pid, pg⟩ := head
    rw [List.foldlM_cons, List.foldlM_cons]
    exact except_map_bind (totalizeGrads t) (addGradDense_map_totalizeGrads hok pid pg)
      fun _ h1 => ih (addGradDense_optGradsOk hok h1)

/-! ### One node: the executed step simulates the proved step -/

/--
The per-node step of `Tape.backwardDense`, written out as a function.

`backwardDense` folds this step over node ids in reverse order (`backwardDense_eq_foldlM`).
Unreached nodes (`acc[id] = some none`) are skipped without running their VJP.
-/
def backwardDenseStep [Add α] (t : Tape α) (acc : Array (Option (Spec.SomeTensor α)))
    (id : Nat) : Result (Array (Option (Spec.SomeTensor α))) := do
  match acc[id]? with
  | none => throw "autograd: internal error (gradient array out of bounds)"
  | some none => pure acc
  | some (some dLdy) =>
    let node ← match t.getNode? id with
      | some n => pure n
      | none => throw "autograd: internal error (node missing)"
    if node.requiresGrad = false then
      pure acc
    else
      let contribs ← node.backward dLdy
      contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradDense (t := t) acc2 pid pg) acc

/-- The executed step preserves the optional accumulator invariant. -/
theorem backwardDenseStep_optGradsOk [Add α] {t : Tape α}
    {grads grads' : Array (Option (Spec.SomeTensor α))} (hok : OptGradsOk t grads) {id : Nat}
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
The executed step at a node, mapped through `totalizeGrads`, is the proved step
`backwardDenseFromStep` at the same node, provided the tape is zero preserving.

When the node was reached both sides run the same VJP on the same cotangent. When it was not,
the executed side does nothing and the proved side runs the VJP on the zero cotangent, whose
contributions are all zeros and therefore accumulate to nothing.
-/
theorem backwardDenseStep_map_totalizeGrads [AddZeroClass α] {t : Tape α}
    (hzp : ZeroPreserving t) {grads : Array (Option (Spec.SomeTensor α))}
    (hok : OptGradsOk t grads) {id : Nat} (hid : id < t.nodes.size) :
    Except.map (totalizeGrads t) (backwardDenseStep t grads id) =
      Tape.backwardDenseFromStep (t := t) (totalizeGrads t grads) id := by
  have htot := totGradsOk_totalizeGrads hok
  obtain ⟨node, hnode⟩ := getNode?_of_lt_size hid
  have hidg : id < grads.size := hok.size_eq ▸ hid
  simp only [backwardDenseStep, Tape.backwardDenseFromStep, Bind.bind, Except.bind, Pure.pure,
    Except.pure, throw, throwThe, MonadExceptOf.throw, hnode]
  cases hcur : grads[id]? with
  | none => exact absurd (Array.getElem?_eq_none_iff.1 hcur) (Nat.not_le.2 hidg)
  | some og =>
    by_cases hreq : node.requiresGrad = false
    · cases og <;> simp [hreq, Except.map]
    · have hreq' : node.requiresGrad = true := by simpa using hreq
      simp only [hreq', Bool.true_eq_false, ite_false]
      cases og with
      | none =>
        rw [getElem?_totalizeGrads_of_none hnode hcur]
        obtain ⟨contribs, hback, hz⟩ := hzp.backward_zero id node hnode hreq'
        simp only [Except.map, dite_eq_left (shape_zeroCotangent node),
          Spec.SomeTensor.ofTensor_cast, hback]
        rw [← Array.foldlM_toList]
        exact (foldlM_addGradAll_zero htot contribs.toList
          fun pid pg hmem => hz pid pg (Array.mem_toList_iff.1 hmem)).symm
      | some dLdy =>
        obtain ⟨node', hnode', hshape⟩ := hok.shape id dLdy hcur
        rw [hnode] at hnode'
        cases hnode'
        rw [getElem?_totalizeGrads_of_some hnode hcur]
        simp only [dite_eq_left hshape, Spec.SomeTensor.ofTensor_cast]
        cases node.backward dLdy with
        | error e => rfl
        | ok contribs =>
          show Except.map (totalizeGrads t)
              (Array.foldlM (fun acc2 (pid, pg) => Tape.addGradDense (t := t) acc2 pid pg) grads
                contribs) =
            Array.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid pg)
              (totalizeGrads t grads) contribs
          rw [← Array.foldlM_toList, ← Array.foldlM_toList]
          exact foldlM_addGradDense_map_totalizeGrads contribs.toList hok

/-! ### The reverse loop -/

/-- Peel the first (largest) id off a reverse-range fold. -/
theorem foldlM_range_reverse_succ {m : Type → Type} [Monad m] {β : Type} (f : β → Nat → m β)
    (b : β) (n : Nat) :
    (List.range (n + 1)).reverse.foldlM f b =
      f b n >>= fun b' => (List.range n).reverse.foldlM f b' := by
  rw [List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append,
    List.foldlM_cons]

/--
The executed reverse loop over the first `n` node ids, mapped through `totalizeGrads`, is the
proved loop `backwardDenseFromLoop` over the same ids.
-/
theorem foldlM_backwardDenseStep_map_totalizeGrads [AddZeroClass α] {t : Tape α}
    (hzp : ZeroPreserving t) :
    ∀ (n : Nat), n ≤ t.nodes.size →
      ∀ {grads : Array (Option (Spec.SomeTensor α))}, OptGradsOk t grads →
        Except.map (totalizeGrads t) ((List.range n).reverse.foldlM (backwardDenseStep t) grads) =
          Tape.backwardDenseFromLoop (t := t) n (totalizeGrads t grads) := by
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
    exact except_map_bind (totalizeGrads t) (backwardDenseStep_map_totalizeGrads hzp hok hlt)
      fun _ h1 => ih (Nat.le_of_lt hlt) (backwardDenseStep_optGradsOk hok h1)

/-! ### Main theorem -/

/-- The seeded optional accumulator `backwardDense` starts from. -/
theorem optGradsOk_seed {t : Tape α} {outId : Nat} {outNode : Runtime.Autograd.Node α}
    (hout : t.getNode? outId = some outNode) {seed : Spec.SomeTensor α}
    (hseed : seed.shape = outNode.value.shape) (hlt : outId < t.nodes.size) :
    OptGradsOk t ((Array.replicate t.nodes.size (none : Option (Spec.SomeTensor α))).set outId
      (some seed) (by simpa using hlt)) := by
  refine optGradsOk_set ⟨by simp, ?_⟩ hout hseed (by simpa using hlt)
  intro id g h
  rw [Array.getElem?_replicate] at h
  split at h <;> simp at h

/-- Totalizing the seeded optional accumulator gives the one-hot total accumulator. -/
theorem totalizeGrads_seed [Zero α] (t : Tape α) (outId : Nat) (seed : Spec.SomeTensor α)
    (hlt : outId < t.nodes.size) :
    totalizeGrads t ((Array.replicate t.nodes.size (none : Option (Spec.SomeTensor α))).set outId
        (some seed) (by simpa using hlt)) =
      oneHotGrads t outId seed := by
  apply Array.ext_getElem?
  intro j
  rw [getElem?_totalizeGrads, getElem?_oneHotGrads, Array.getElem?_set, Array.getElem?_replicate]
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
theorem backwardDense_eq_foldlM [Add α] {t : Tape α} {outId : Nat}
    {outNode : Runtime.Autograd.Node α} (hout : t.getNode? outId = some outNode)
    {seed : Spec.SomeTensor α} (hseed : seed.shape = outNode.value.shape) :
    Tape.backwardDense (t := t) outId seed =
      (List.range t.nodes.size).reverse.foldlM (backwardDenseStep t)
        ((Array.replicate t.nodes.size (none : Option (Spec.SomeTensor α))).set outId (some seed)
          (by simpa using getNode?_lt_size hout)) := by
  have hlt : outId < t.nodes.size := getNode?_lt_size hout
  simp only [Tape.backwardDense, hout, Bind.bind, Except.bind, Pure.pure, Except.pure, throw,
    throwThe, MonadExceptOf.throw, dite_eq_left hseed, Array.size_replicate, dite_eq_left hlt,
    Spec.SomeTensor.ofTensor_cast]
  rfl

/--
**Executed backward pass = proved backward pass.** On a zero-preserving tape, the totalized
executed sweep `backwardDenseAll` (which skips unreached nodes) returns exactly what the proved
sweep `backwardDenseFrom` returns when started from the one-hot seed array, including agreement
of the error cases. The hypotheses on `outId` and `seed` are the checks `backwardDense` performs
before traversing; without them `backwardDense` fails while `backwardDenseFrom` may not.
-/
theorem backwardDenseAll_eq_backwardDenseFrom [AddZeroClass α] {t : Tape α} (hzp : ZeroPreserving t)
    {outId : Nat} {outNode : Runtime.Autograd.Node α} (hout : t.getNode? outId = some outNode)
    {seed : Spec.SomeTensor α} (hseed : seed.shape = outNode.value.shape) :
    Tape.backwardDenseAll (t := t) outId seed =
      Tape.backwardDenseFrom (t := t) (oneHotGrads t outId seed) := by
  have hlt : outId < t.nodes.size := getNode?_lt_size hout
  rw [backwardDenseAll_eq_map_totalizeGrads, backwardDense_eq_foldlM hout hseed,
    foldlM_backwardDenseStep_map_totalizeGrads hzp t.nodes.size (Nat.le_refl _)
      (optGradsOk_seed hout hseed hlt),
    totalizeGrads_seed t outId seed hlt]
  simp [Tape.backwardDenseFrom]

end Graph

end Algebra
end Autograd
end Proofs
