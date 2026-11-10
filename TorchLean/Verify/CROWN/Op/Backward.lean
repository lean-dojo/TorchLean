/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Affine.Real
public import TorchLean.Verify.CROWN.FlatBox.Backward.Real
public import TorchLean.Verify.CROWN.Op.Directed
public import TorchLean.Verify.CROWN.Op.ExecCROWN

/-!
# Soundness of the backward sweep on operator graphs

At a lawful scalar, a reverse sweep of `Op.ExecCROWN` bounds the objective `obj · v out` at a real
point of the graph (`GraphPoint`). `sweep_bound` carries the directional invariant `DirInvariant`
from the output down to the input when the rule of every node is sound in that direction
(`RuleSound`), and `sweep_encloses` takes both directions. `directedRule_sound` proves the shared
rules sound at every operator, given the real equation of each node (`Equation`). At a
convolution the transposed convolution `convBack?`, rounded outward, encloses the adjoint
of the real one (`convBack?_sound`). `directedSweep_encloses` is the resulting bound for
`directedStep`. The meet lemmas (`meetBox_encloses`, `meetAt_encloses`, `meetWith_encloses`) cover
the intersection with the box of a node.

We proved `sweep_bound` by hand, following the proof of Theorem 1 in Xu et al. (2020,
arXiv:2002.12920, App. B.1). The sweep keeps their Eq. (3) over the pending nodes (`Represents`:
real coefficients inside the stored intervals and a real constant) from the objective down to the
input. Their proof numbers the nodes in topological order, and `GraphPoint` asks the same (`topo`).
That premise is what lets a sweep in decreasing index order stand in for their BFS.
-/

@[expose] public section

open TorchLean.Shape.Coord (linearize linearize_unlinearize unlinearize)
open TorchLean.Shape (size_append)
open TorchLean.Shape.Coord (unlinearize_linearize)
open TorchLean.Shape (size size_cons)
open TorchLean.Shape (size_nil)

namespace TorchLean.CROWN.Op

open Conv Tensor
open Graph
open CROWN.Directed

section Lawful

open Shape (Coord)

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The meet of two boxes of one width that enclose real coordinates encloses them. -/
theorem meetBox_encloses {A B : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (hA : RowEncloses A n f) (hB : RowEncloses B n f) : RowEncloses (meetBox A B) n f := by
  unfold meetBox
  rw [ite_eq_left (hA.1.trans hB.1.symm)]
  refine ⟨hB.1, fun i => ?_⟩
  have hi : i.val < B.dim := hB.1 ▸ i.isLt
  obtain ⟨ha1, ha2⟩ := hA.2 i
  obtain ⟨hb1, hb2⟩ := hB.2 i
  rw [getAtOrZero_eq_getScalar _ ⟨i.val, hi⟩] at hb1 hb2
  dsimp only
  rw [getAtOrZero_eq_getScalar _ ⟨i.val, hi⟩, getAtOrZero_eq_getScalar _ ⟨i.val, hi⟩,
    getScalar_ofFn, getScalar_ofFn]
  constructor
  · split <;> assumption
  · split <;> assumption

/-- A box met with the box of node `id` encloses what both enclose. -/
theorem meetAt_encloses {ibp : Array (Option (FlatBox α))} {id : Nat} {B : FlatBox α} {n : Nat}
    {f : Nat → ℝ} (hB : RowEncloses B n f)
    (hI : ∀ I, (ibp[id]?).join = some I → RowEncloses I n f) :
    RowEncloses (meetAt ibp id B) n f := by
  unfold meetAt
  split
  · next I hI' => exact meetBox_encloses hB (hI I hI')
  · exact hB

/-- A box that `meetWith` returns encloses what both its argument and the node's box enclose. -/
theorem meetWith_encloses {ibp : Array (Option (FlatBox α))} {id : Nat} {B B' : FlatBox α}
    {n : Nat} {f : Nat → ℝ} (h : meetWith ibp id B = .ok B') (hB : RowEncloses B n f)
    (hI : ∀ I, (ibp[id]?).join = some I → RowEncloses I n f) : RowEncloses B' n f := by
  unfold meetWith at h
  split at h
  · next I hI' =>
    split at h
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact meetBox_encloses hB (hI I hI')
    · simp [throw, throwThe, MonadExceptOf.throw] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-! ## The reverse sweep -/

/-- A real point of an operator graph for the backward sweep, with widths `dims` and coordinates `v`
per node. The designated input is an input node, and `dims` gives it the width `ctx.inputDim` of
the affine forms, which need not be the size of its declared port shape. Every reference names an
earlier node, every box of `ibp` encloses its node's coordinates, and at every constant node whose
rounded value has a flat tensor the point has that tensor's width and the reals of its entries. A
constant whose value has no flat tensor gets no such equation, but its box in `ibp`, if it has one,
encloses its coordinates like every other box. The real equations of the operator nodes are
separate premises (`Equation`, `ProductEquation`). -/
structure GraphPoint (G : IR.Graph) (P : IR.Payload α) (ibp : Array (Option (FlatBox α)))
    (ctx : AffineCtx) (dims : Nat → Nat) (v : Nat → Nat → ℝ) : Prop where
  /-- The designated input is an input node. -/
  input_node : ∃ k p, G.nodes[ctx.inputId]? = some (.input k p)
  /-- `dims` gives the designated input the width of the affine forms. -/
  input_dim : dims ctx.inputId = ctx.inputDim
  /-- Every reference names an earlier node. -/
  topo : TopoSorted G
  /-- Every box of the table encloses its node's coordinates. -/
  ibp_encloses : ∀ id B, (ibp[id]?).join = some B → RowEncloses B (dims id) (v id)
  /-- At a constant node whose rounded value has a flat tensor `c`, the point has the width of `c`
  and the reals of its entries. -/
  const_eq : ∀ id (c : FlatTensor α), (constValue? G P id >>= flatOf) = some c →
    dims id = c.n ∧ ∀ i : Fin c.n, v id i.val = value (getAtOrZero c.v [i.val])

/-- `z'` bounds `z` in direction `dir`: from below at `.lower`, from above at `.upper`. -/
def DirBound : BackwardDir → ℝ → ℝ → Prop
  | .lower, z', z => z' ≤ z
  | .upper, z', z => z ≤ z'

/-- Every value bounds itself, in either direction. -/
theorem DirBound.refl (dir : BackwardDir) (z : ℝ) : DirBound dir z z := by
  cases dir <;> exact le_rfl

/-- If `a` bounds `b` and `b` bounds `c` in direction `dir`, then `a` bounds `c`. -/
theorem DirBound.trans {dir : BackwardDir} {a b c : ℝ} (hab : DirBound dir a b)
    (hbc : DirBound dir b c) : DirBound dir a c := by
  cases dir
  · exact le_trans (α := ℝ) hab hbc
  · exact le_trans (α := ℝ) hbc hab

/-- Adding `d` on the right of both values keeps a bound in direction `dir`. -/
theorem DirBound.add_right {dir : BackwardDir} {a b : ℝ} (h : DirBound dir a b) (d : ℝ) :
    DirBound dir (a + d) (b + d) := by
  cases dir <;> simp only [DirBound] at h ⊢ <;> linarith

/-- The sweep invariant in direction `dir`: the state fails, or it has a table of size `size` and
represents a value that bounds `z` in direction `dir` over the pending nodes. The pending nodes
(`pending`) are those below `k` and the designated input `input`, even when its index is at least
`k`. -/
def DirInvariant (dir : BackwardDir) (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input size k : Nat)
    (z : ℝ) (st : State α) : Prop :=
  ∃ z', DirBound dir z' z ∧ SweepInvariant dims v input size k z' st

/-- The sweep invariant at `z` gives the directed invariant at `z`, in either direction. -/
theorem DirInvariant.of_eq {dir : BackwardDir} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {input size k : Nat} {z : ℝ} {st : State α}
    (h : SweepInvariant dims v input size k z st) : DirInvariant dir dims v input size k z st :=
  ⟨z, DirBound.refl dir z, h⟩

/-- The directed invariant at `z'` gives it at every `z` that `z'` bounds in direction `dir`. -/
theorem DirInvariant.mono {dir : BackwardDir} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {input size k : Nat} {z' z : ℝ} {st : State α} (hb : DirBound dir z' z)
    (h : DirInvariant dir dims v input size k z' st) : DirInvariant dir dims v input size k z st :=
  let ⟨w, hw, h⟩ := h
  ⟨w, hw.trans hb, h⟩

/-- A backward rule is sound in direction `dir` at node `id` of the point `(dims, v)` with
designated input `input` when it keeps the sweep invariant there. The node must be an operator
node whose references all name output `0`, and the state must have the node in its table and
represent `z` with the node discharged. On a coefficient box that encloses `a`, the rule then
fails or gives a state with a table of the same size that represents a value bounding
`z + a · v id` in direction `dir`. -/
def RuleSound (dir : BackwardDir) (G : IR.Graph) (rule : State α → Nat →
      FlatBox α → IR.Operator → List IR.Ref → State α)
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input id : Nat) : Prop :=
  ∀ o refs, G.nodes[id]? = some (.op o refs) → (∀ r ∈ refs.toList, r.out = 0) →
    ∀ (st : State α) (z : ℝ) (aY : FlatBox α) (a : Nat → ℝ),
      id < st.coeffs.size → Represents dims v input id st z → RowEncloses aY (dims id) a →
      DirInvariant dir dims v input st.coeffs.size id (z + Directed.dot (dims id) a (v id))
        (rule st id aY o refs.toList)

/-- Passing the designated input keeps the represented objective. -/
theorem represents_input {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input (input + 1) st z) :
    Represents dims v input input st z := by
  obtain ⟨f, c, hs, hz⟩ := h
  exact ⟨f, c, hs, (frontier_input input f c).symm.trans hz⟩

/-- Discharging a coefficient against the box of node `id` fails or adds the node's term. -/
theorem consumeAt_sound {G : IR.Graph} {P : IR.Payload α}
    {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx} {dims : Nat → Nat}
    {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v) {input k id : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    {aY : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims id) a) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims id) a (v id))
      (consumeAt ibp st id aY) := by
  unfold consumeAt
  split
  · next B hB =>
    rcases represents_consume h ha (hG.ibp_encloses id B hB) with hf | hr
    · exact Or.inl hf
    · exact Or.inr ⟨consume_size .., hr⟩
  · exact Or.inl rfl

/-- One step of a backward sweep with a rule sound in direction `dir` keeps the sweep invariant
in that direction. -/
theorem backStep_preserves (dir : BackwardDir) {G : IR.Graph} {P : IR.Payload α}
    {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx} {dims : Nat → Nat}
    {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    {rule : State α → Nat → FlatBox α → IR.Operator → List IR.Ref →
      State α} {k : Nat} (hrule : RuleSound dir G rule dims v ctx.inputId k)
    (hk : k < G.nodes.size) {z : ℝ} {st : State α}
    (h : DirInvariant dir dims v ctx.inputId G.nodes.size (k + 1) z st) :
    DirInvariant dir dims v ctx.inputId G.nodes.size k z (backStep rule G P ibp ctx st k) := by
  obtain ⟨z, hb, h⟩ := h
  refine DirInvariant.mono hb ?_
  rcases h with hf | ⟨hsize, hrep⟩
  · exact .of_eq (Or.inl (by simp [backStep, hf]))
  have hks : k < st.coeffs.size := hsize ▸ hk
  obtain ⟨j₀, p₀, hin⟩ := hG.input_node
  unfold backStep
  split
  · next hf => exact .of_eq (Or.inl hf)
  split
  · next hnone =>
    refine .of_eq (Or.inr ⟨hsize, ?_⟩)
    by_cases hki : k = ctx.inputId
    · subst hki
      exact represents_input hrep
    · exact represents_drop_none hrep hki hks hnone
  · next aY hsome =>
    have hne : ∀ n, G.nodes[k]? = some n → (∀ j p, n ≠ .input j p) → k ≠ ctx.inputId := by
      rintro n hn hni rfl
      exact hni j₀ p₀ (Option.some.inj (hn.symm.trans hin))
    split
    · next j p hn =>
      split
      · next hki =>
        subst hki
        exact .of_eq (Or.inr ⟨hsize, represents_input hrep⟩)
      · next hki =>
        obtain ⟨a, ha, hrep'⟩ := represents_drop_some hrep hki hks hsome
        exact .of_eq (by simpa only [hsize, sub_add_cancel] using consumeAt_sound hG hrep' ha)
    · next p hn =>
      have hki := hne _ hn (fun _ _ h => by cases h)
      obtain ⟨a, ha, hrep'⟩ := represents_drop_some hrep hki hks hsome
      split
      · next c hc =>
        obtain ⟨hd, hv⟩ := hG.const_eq k c hc
        have hx : RowEncloses (FlatBox.ofTensor c.v) (dims k) (v k) := by
          refine ⟨hd.symm, fun i => ?_⟩
          have := hv ⟨i.val, hd ▸ i.isLt⟩
          simp only [FlatBox.ofTensor] at this ⊢
          rw [this]
          exact ⟨le_rfl, le_rfl⟩
        rcases represents_consume hrep' ha hx with hf | hr
        · exact .of_eq (Or.inl hf)
        · exact .of_eq (Or.inr ⟨by simpa using hsize, by simpa only [sub_add_cancel] using hr⟩)
      · exact .of_eq (Or.inl rfl)
    · next o refs hn =>
      have hki := hne _ hn (fun _ _ h => by cases h)
      obtain ⟨a, ha, hrep'⟩ := represents_drop_some hrep hki hks hsome
      split
      · next hall =>
        have h0 : ∀ r ∈ refs.toList, r.out = 0 := by
          intro r hr
          have hr' : r ∈ refs := by simpa using hr
          obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hr'
          simpa using Array.all_eq_true.mp hall i hi
        simpa only [hsize, sub_add_cancel] using hrule o refs hn h0 st _ aY a hks hrep' ha
      · exact .of_eq (Or.inl rfl)
    · exact .of_eq (Or.inl rfl)

/-- The real value of the objective at the point: `obj · v out`. -/
def objectiveValue (dims : Nat → Nat) (v : Nat → Nat → ℝ) (out : Nat) (obj : FlatTensor α) : ℝ :=
  Directed.dot (dims out) (fun i => value (getAtOrZero obj.v [i])) (v out)

/-- The form of direction `dir` of a pair of lower and upper forms. -/
def dirForm (dir : BackwardDir) {n : Nat} (lower upper : Linear α n 1) :
    Linear α n 1 :=
  match dir with
  | .lower => lower
  | .upper => upper

/-- If its rule is sound in direction `dir` at every node, a backward sweep bounds the objective
`obj · v out` at the point in that direction by the value of its form of direction `dir` at the
input's coordinates. -/
theorem sweep_bound (dir : BackwardDir) {G : IR.Graph} {P : IR.Payload α}
    {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx} {dims : Nat → Nat}
    {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    {rule : State α → Nat → FlatBox α → IR.Operator → List IR.Ref →
      State α}
    (hrule : ∀ k, k < G.nodes.size → RuleSound dir G rule dims v ctx.inputId k)
    {out : Nat} {obj : FlatTensor α} (hobj : obj.n = dims out)
    {lower upper : Linear α ctx.inputDim 1}
    (h : sweep (backStep rule G P ibp ctx) G ctx ibp out obj = some (lower, upper)) :
    DirBound dir (affineValue (dirForm dir lower upper) (fun i => v ctx.inputId i.val))
      (objectiveValue dims v out obj) := by
  unfold sweep at h
  split at h
  · next hout =>
    have hinit := initial_represents dims v ctx.inputId G.nodes.size out hout obj hobj
    have hfin := reverseSweep_preserves (backStep rule G P ibp ctx)
      (fun k st => DirInvariant dir dims v ctx.inputId G.nodes.size k
        (objectiveValue dims v out obj) st) G.nodes.size
      (fun k hk st hst => backStep_preserves dir hG (hrule k hk) hk hst) G.nodes.size le_rfl _
      (.of_eq hinit)
    rw [List.foldl_reverse, ← Fin.foldr_eq_finRange_foldr] at hfin
    dsimp only at h
    generalize Fin.foldr G.nodes.size (fun i acc => backStep rule G P ibp ctx acc i.val) _ = st
      at h hfin
    split at h
    · cases h
    · next hnf =>
      obtain ⟨z, hb, hfin⟩ := hfin
      rcases hfin with hf | ⟨hsize, f, c, hs, hz⟩
      · simp [hf] at hnf
      cases hI : ibp[ctx.inputId]? with
      | none => simp [hI] at h
      | some I? =>
        cases I? with
        | none => simp [hI] at h
        | some I =>
          simp only [hI, Option.bind_eq_bind, Option.bind_some] at h
          obtain ⟨j₀, p₀, hin⟩ := hG.input_node
          have hlt : ctx.inputId < st.coeffs.size := by
            rw [hsize]; exact (Array.getElem?_eq_some_iff.mp hin).1
          have ha := inputRow_encloses hs ctx.inputId ctx.inputDim hlt hG.input_dim
          have hx := hG.ibp_encloses ctx.inputId I (by simp [hI])
          rw [hG.input_dim] at hx
          rw [frontier_zero, hG.input_dim] at hz
          rw [← hz] at hb
          have hlu := inputAffines_row_encloses hx ha hs.2 h
          cases dir
          · exact le_trans (α := ℝ) hlu.1 hb
          · exact le_trans (α := ℝ) hb hlu.2
  · cases h

/-- A backward sweep whose rule is sound in both directions at every node bounds the objective
`obj · v out` at the point between the values of its lower and upper affine forms at the input's
coordinates. -/
theorem sweep_encloses {G : IR.Graph} {P : IR.Payload α}
    {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx} {dims : Nat → Nat}
    {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    {rule : State α → Nat → FlatBox α → IR.Operator → List IR.Ref →
      State α}
    (hrule : ∀ dir k, k < G.nodes.size → RuleSound dir G rule dims v ctx.inputId k)
    {out : Nat} {obj : FlatTensor α} (hobj : obj.n = dims out)
    {lower upper : Linear α ctx.inputDim 1}
    (h : sweep (backStep rule G P ibp ctx) G ctx ibp out obj = some (lower, upper)) :
    affineValue lower (fun i => v ctx.inputId i.val) ≤ objectiveValue dims v out obj ∧
      objectiveValue dims v out obj ≤ affineValue upper (fun i => v ctx.inputId i.val) :=
  ⟨sweep_bound .lower hG (hrule .lower) hobj h, sweep_bound .upper hG (hrule .upper) hobj h⟩

/-! ## The shared rules -/

/-- The real equations the shared backward rules use at operator node `id`, for the lookups that
succeed. Detach and reshape copy their reference, and add and sub add and subtract their
references. A linear layer has the widths of its shapes and on one leading row applies its constant
weight and bias to its input. A sum over every axis in order without `keepdim` is the sum of its
reference's coordinates, and a slice along one axis with step 1 and a slice layout reads its
reference at the layout's flat indices. A concatenation with one reference per size reads each
block of its layout from the block's reference. A permutation of the axes reads a reference of the
width of its input shape, and at each coordinate it equals the permutation of any tensor that
holds its reference's coordinates. A convolution with constant kernel and bias has the widths of
its input and output, and at each coordinate it equals the operator's rule at `ℝ` on the reals of
the kernel and the bias and on any tensor that holds its input's coordinates. Every other operator
needs none. -/
def Equation (G : IR.Graph) (P : IR.Payload α) (dims : Nat → Nat) (v : Nat → Nat → ℝ)
    (id : Nat) : IR.Operator → List IR.Ref → Prop
  | ⟨_, _, .detach⟩, [r] => CopyEquation dims v id r.node
  | ⟨_, _, .reshape _⟩, [r] => CopyEquation dims v id r.node
  | ⟨_, _, .add⟩, [r₁, r₂] => BinaryEquation dims v id r₁.node r₂.node (· + ·)
  | ⟨_, _, .sub⟩, [r₁, r₂] => BinaryEquation dims v id r₁.node r₂.node (· - ·)
  | ⟨_, _, .linear (lead := L) (inDim := n) (outDim := m)⟩, [rW, rb, rx] =>
      ∀ W b, constAt? G P rW (.value [m, n]) = some W → constAt? G P rb (.value [m]) = some b →
        dims rx.node = (L ++ [n]).size ∧ dims id = (L ++ [m]).size ∧
          (L.size = 1 → LinearEquation dims v id rx.node W b)
  | ⟨_, _, .reduce (s := s) .sum axes false _⟩, [r] =>
      axes.map Fin.val = List.range s.rank → v id 0 = ∑ i : Fin (dims r.node), v r.node i.val
  | ⟨_, _, .slice (s := s) axes starts stops steps _⟩, [r] =>
      ∀ axis start stop (layout : SliceLayout), axes.map Fin.val = [axis] → starts = [start] →
        stops = [stop] → steps = [1] →
        sliceLayout? axis start (stop - start) s (s.slice axes starts stops steps) = some layout →
        ∀ i : Fin layout.outputShape.size, v id i.val = v r.node (layout.flatIndex i).val
  | ⟨_, _, .cat (s := s) dim sizes⟩, refs =>
      refs.length = sizes.length →
        ∀ (j : Fin (catLayout s dim.val sizes).lengths.length)
          (i : Fin ((catLayout s dim.val sizes).parentShape j).size),
          v id ((catLayout s dim.val sizes).flatEquiv ⟨j, i⟩).val =
            v (refs.map (·.node)).toArray[j.val]! i.val
  | ⟨_, _, .permute (s := s) σ out hout⟩, [r] =>
      dims r.node = s.size ∧ ∀ x : Tensor ℝ s, (∀ c, x c = v r.node (Coord.linearize c).val) →
        ∀ c, v id (Coord.linearize c).val = Tensor.permute x σ out hout c
  | ⟨_, _, .conv (k := k) cd stride padLow padHigh dilation groups (x := xs) (y := ys) hlead hx
      hy⟩, [rK, rb, rx] =>
      ∀ K b, constAt? G P rK (.value k) = some K →
        constAt? G P rb (.value [k.extent (cd.kernel ⟨0, by omega⟩)]) = some b →
          dims rx.node = xs.size ∧ dims id = ys.size ∧
            ∀ rule, (Op.conv cd stride padLow padHigh dilation groups hlead hx hy).rule?
                (α := ℝ) = some rule →
              ∀ x : Tensor ℝ xs, (∀ c, x c = v rx.node (Coord.linearize c).val) →
                ∀ c, v id (Coord.linearize c).val =
                  (rule.forward (Tensor.map value K, Tensor.map value b, x, ())).1 c
  | _, _ => True

/-- Every node below `k` is pending. -/
theorem mem_pending_of_lt {input k p : Nat} (h : p < k) : p ∈ pending input k := by
  simp [pending, h]

/-- The box of a reference encloses its node's coordinates. -/
theorem box?_encloses {G : IR.Graph} {P : IR.Payload α}
    {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx} {dims : Nat → Nat}
    {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v) {r : IR.Ref} {B : FlatBox α}
    (h : box? ibp r = some B) : RowEncloses B (dims r.node) (v r.node) := by
  unfold box? at h
  split at h
  · exact hG.ibp_encloses _ _ h
  · cases h

/-- A copy passes the coefficient to its reference. -/
theorem copy_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p : Nat} (hp : p < k) (heq : CopyEquation dims v k p)
    {aY : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (addCoeff st p aY) :=
  Or.inr ⟨addCoeff_size .., represents_copy h (by omega) (mem_pending_of_lt hp) ha heq⟩

/-- A sum passes the coefficient to both references. -/
theorem add_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p q : Nat} (hp : p < k) (hq : q < k)
    (heq : BinaryEquation dims v k p q (· + ·))
    {aY : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (addCoeff (addCoeff st p aY) q aY) :=
  Or.inr ⟨by simp, represents_addParents h (by omega) (by omega) (mem_pending_of_lt hp)
    (mem_pending_of_lt hq) ha heq.1 heq.2.1 heq.2.2⟩

/-- A difference passes the coefficient to its first reference and its negation to the second. -/
theorem sub_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p q : Nat} (hp : p < k) (hq : q < k)
    (heq : BinaryEquation dims v k p q (· - ·))
    {aY : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (addCoeff (addCoeff st p aY) q (boxNeg aY)) :=
  Or.inr ⟨by simp, represents_subParents h (by omega) (by omega) (mem_pending_of_lt hp)
    (mem_pending_of_lt hq) ha heq.1 heq.2.1 heq.2.2⟩

omit [LawfulBoundOps α] in
/-- A successful transposed linear map takes a coefficient of width `m` to one of width `n`. -/
theorem _root_.TorchLean.CROWN.Directed.linear_dim {m n : Nat} {aY : FlatBox α}
    {W : Tensor α [m, n]}
    {b : Tensor α [m]} {r : FlatBox α × (α × α)} (h : Directed.linear aY W b = some r) :
    aY.dim = m ∧ r.1.dim = n := by
  unfold Directed.linear at h
  split at h
  · next hm =>
    simp only [Option.map_eq_some_iff] at h
    obtain ⟨c, -, rfl⟩ := h
    exact ⟨hm, rfl⟩
  · cases h

/-- A linear layer with constant weight and bias passes the transposed coefficient to its input
and the bias term to the constant. It does so only when the coefficient has the width of one output
row and the transposed coefficient the width of the input, which forces one leading row or a weight
with no rows and no columns. -/
theorem linear_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p : Nat} (hp : p < k) {L : Shape} {m n : Nat}
    {W : Tensor α [m, n]} {b : Tensor α [m]} (hdp : dims p = (L ++ [n]).size)
    (hdk : dims k = (L ++ [m]).size) (heq : L.size = 1 → LinearEquation dims v k p W b)
    {aY : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (match Directed.linear aY W b with
        | some (aX, c) =>
            if aX.dim = (L ++ [n]).size then
              addConstant (addCoeff st p aX) c.1 c.2
            else st.fail
        | none => st.fail) := by
  split
  · next aX c hres =>
    split
    · next hw =>
      obtain ⟨haY, haX⟩ := linear_dim hres
      have hm : dims k = m := ha.1.symm.trans haY
      have hlin : LinearEquation dims v k p W b := by
        by_cases hL : L.size = 1
        · exact heq hL
        · simp only [Shape.size_append, Shape.size_nil, Shape.size_cons, Nat.mul_one] at hdp hdk hw
          have hn : n = 0 := by
            by_contra hn
            exact hL (Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hn)
              (by rw [Nat.one_mul]; exact (haX.symm.trans hw).symm))
          have hm0 : m = 0 := by
            by_contra hm0
            exact hL (Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hm0)
              (by rw [Nat.one_mul]; exact (hm.symm.trans hdk).symm))
          subst hn hm0
          exact ⟨hm, by simpa using hdp, fun i => i.elim0⟩
      obtain ⟨-, hdp', hy⟩ := hlin
      have ha' : RowEncloses aY m a := hm ▸ ha
      obtain ⟨hr, -, hs⟩ := represents_linear h p (by omega) (mem_pending_of_lt hp) hdp' ha' W b
        hy hres
      exact Or.inr ⟨hs, by simpa only [hm] using hr⟩
    · exact Or.inl rfl
  · exact Or.inl rfl

/-- A sum over every axis passes the coefficient, broadcast, to its reference. -/
theorem sum_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p : Nat} (hp : p < k) {Bx : FlatBox α}
    (hBx : Bx.dim = dims p) (heq : v k 0 = ∑ i : Fin (dims p), v p i.val)
    {aY : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (if h : aY.dim = 1 then
        addCoeff st p
          { dim := Bx.dim
            lo := Tensor.full (.dim Bx.dim .scalar)
              (getAtOrZero (castDimScalar (n := aY.dim) (n' := 1) h aY.lo) [0])
            hi := Tensor.full (.dim Bx.dim .scalar)
              (getAtOrZero (castDimScalar (n := aY.dim) (n' := 1) h aY.hi) [0]) }
      else st.fail) := by
  obtain ⟨adim, alo, ahi⟩ := aY
  by_cases hd : adim = 1
  · subst hd
    simp only [↓reduceDIte, castDimScalar_self]
    have hdk : dims k = 1 := ha.1.symm
    have hb := broadcast_encloses (n := Bx.dim) alo ahi (a 0)
      (by
        have ha1 : RowEncloses { dim := 1, lo := alo, hi := ahi } 1 a := by
          rw [hdk] at ha; exact ha
        have hscalar := ha1.2 (0 : Fin 1)
        change value (getAtOrZero alo [0]) ≤ a 0 ∧
          a 0 ≤ value (getAtOrZero ahi [0]) at hscalar
        rwa [show getAtOrZero alo [0] = alo.getScalar 0 from Tensor.getAtOrZero_eq_getScalar alo 0,
          show getAtOrZero ahi [0] = ahi.getScalar 0 from Tensor.getAtOrZero_eq_getScalar ahi 0]
          at hscalar)
    have hb' : RowEncloses
        { dim := Bx.dim
          lo := Tensor.full (α := α) (.dim Bx.dim .scalar) (getAtOrZero alo [0])
          hi := Tensor.full (α := α) (.dim Bx.dim .scalar) (getAtOrZero ahi [0]) }
        (dims p) (fun _ => a 0) := by simpa only [hBx] using hb
    have hdot : Directed.dot (dims k) a (v k) = Directed.dot (dims p) (fun _ => a 0) (v p) := by
      rw [hdk]
      simp only [Directed.dot, dotProduct, Function.comp_apply, Fin.sum_univ_one,
        Fin.val_zero, heq, Finset.mul_sum]
    refine Or.inr ⟨addCoeff_size .., ?_⟩
    rw [hdot]
    exact represents_add h p (by omega) (mem_pending_of_lt hp) _ _ hb'
  · simp only [hd, ↓reduceDIte]
    exact Or.inl rfl

/-- A slice along one axis with step one passes the coefficient to its reference, scattered to the
entries it reads with zeros elsewhere. -/
theorem slice_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p : Nat} (hp : p < k) (layout : SliceLayout)
    (hdim : dims p = layout.sourceShape.size)
    (hy : ∀ i : Fin layout.outputShape.size, v k i.val = v p (layout.flatIndex i).val)
    {aY c : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a)
    (hc : scatterSliceBoxes? layout aY = some c) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (addCoeff st p c) := by
  have hd : dims k = layout.outputShape.size := by
    unfold scatterSliceBoxes? at hc
    split at hc
    · next hw => exact ha.1.symm.trans hw
    · cases hc
  rw [hd] at ha ⊢
  exact Or.inr ⟨addCoeff_size .., represents_slice h (by omega) (mem_pending_of_lt hp) layout hdim
    ha hy hc⟩

/-- A permutation of the axes passes the coefficient, permuted back by `Shape.unpermute`, to its
reference. -/
theorem permute_sound [Op.Capabilities α] {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {input k : Nat} {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p : Nat} (hp : p < k) {s out : Shape}
    {σ : Equiv.Perm (Fin s.rank)} (hout : out = s.permute σ) (hdim : dims p = s.size)
    (hy : ∀ x : Tensor ℝ s, (∀ c, x c = v p (Coord.linearize c).val) →
      ∀ c, v k (Coord.linearize c).val = Tensor.permute x σ out hout c)
    {aY aX : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a)
    (hX : permuteBack σ hout aY = some aX) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (addCoeff st p aX) := by
  have hw : aY.dim = out.size := by
    unfold permuteBack endpointsBox? at hX
    split at hX
    · assumption
    · cases hX
  have hd : dims k = out.size := ha.1.symm.trans hw
  obtain ⟨A, hA⟩ : ∃ A : Tensor ℝ out, ∀ c, A c = a (Coord.linearize c).val :=
    ⟨Tensor.Rep.ofFn _, Tensor.Rep.get_ofFn _⟩
  obtain ⟨x, hx⟩ : ∃ x : Tensor ℝ s, ∀ c, x c = v p (Coord.linearize c).val :=
    ⟨Tensor.Rep.ofFn _, Tensor.Rep.get_ofFn _⟩
  have hB := Op.Directed.endpointsBox?_encloses rfl
    (F := fun y => Tensor.permute y _ s (s.permute_unpermute σ hout).symm)
    (fun y c => by simp only [Tensor.permute_apply, Tensor.map_apply])
    (fun y y' hyy c => by simp only [Tensor.permute_apply]; exact hyy _)
    (Op.Directed.encloses_of_coord hw fun c => by
      rw [hA]; exact ha.2 ⟨_, hd ▸ (Coord.linearize c).isLt⟩) hX
  have hB' : RowEncloses aX (dims p) fun j => getAtOrZero (flatOfSome
      ⟨s, Tensor.permute A (s.unpermute σ hout) s (s.permute_unpermute σ hout).symm⟩).v [j] := by
    rw [hdim]; exact hB
  refine Or.inr ⟨addCoeff_size .., ?_⟩
  convert represents_add h p (by omega) (mem_pending_of_lt hp) aX _ hB' using 2
  rw [hd, hdim, dot_eq_sum_coord, dot_eq_sum_coord]
  simp only [← hA, hy x hx, ← hx, flatOfSome, Tensor.getAtOrZero_reshape_toFin]
  exact sum_mul_permute σ hout A x

/-! ## The transposed convolution -/

/-- The lower end of entry `i` of a partial table, zero where it has none. -/
def partialLo (cx : PartialCoeffs α) (i : Nat) : α := ((cx[i]?).join.map (·.1)).getD 0

/-- The upper end of entry `i` of a partial table, zero where it has none. -/
def partialHi (cx : PartialCoeffs α) (i : Nat) : α := ((cx[i]?).join.map (·.2)).getD 0

/-- A partial table of `n` entries encloses the real coefficients `f`: each entry's interval holds
its coefficient, and an absent entry stands for zero. -/
def PartialEncloses (cx : PartialCoeffs α) (n : Nat) (f : Nat → ℝ) : Prop :=
  cx.size = n ∧ ∀ i < n, value (partialLo cx i) ≤ f i ∧ f i ≤ value (partialHi cx i)

/-- A table of `n` absent entries encloses the zero coefficients. -/
theorem partialEncloses_replicate (n : Nat) :
    PartialEncloses (Array.replicate n (none : Option (α × α))) n (fun _ => 0) := by
  refine ⟨by simp, fun i hi => ?_⟩
  simp [partialLo, partialHi, hi, LawfulBoundOps.toReal_zero]

/-- Adding `d` to coefficient `i < n` adds `d` times entry `i` to the pairing. -/
theorem dot_bump {n : Nat} {f x : Nat → ℝ} {i : Nat} (hi : i < n) (d : ℝ) :
    Directed.dot n (Function.update f i (f i + d)) x = Directed.dot n f x + d * x i := by
  simp only [Directed.dot, dotProduct, Function.comp_apply]
  have : ∀ j : Fin n, Function.update f i (f i + d) j.val * x j.val =
      f j.val * x j.val + if j = ⟨i, hi⟩ then d * x i else 0 := by
    intro j
    by_cases h : j.val = i
    · have : j = ⟨i, hi⟩ := Fin.ext h
      subst this; simp; ring
    · have : j ≠ ⟨i, hi⟩ := fun e => h (congrArg Fin.val e)
      simp [h, this]
  rw [Finset.sum_congr rfl (fun j _ => this j), Finset.sum_add_distrib]
  simp

/-- If `cx` encloses `f`, accumulating an interval that holds `d` at entry `i < n` encloses `f`
with `d` added to entry `i`. -/
theorem accum_encloses {cx : PartialCoeffs α} {n : Nat} {f : Nat → ℝ}
    (h : PartialEncloses cx n f) {i : Nat} (hi : i < n) {w : α × α} {d : ℝ}
    (hd : value w.1 ≤ d ∧ d ≤ value w.2) :
    PartialEncloses (accum cx i w) n (Function.update f i (f i + d)) := by
  have hi' : i < cx.size := h.1 ▸ hi
  have hfi := h.2 i hi
  unfold accum
  split
  · next w0 hw0 =>
    simp only [partialLo, partialHi, hw0, Option.join_some, Option.map_some,
      Option.getD_some] at hfi
    refine ⟨by simp [h.1], fun j hj => ?_⟩
    by_cases hji : j = i
    · subst hji
      simp only [partialLo, partialHi, Array.set!, Array.getElem?_setIfInBounds_self_of_lt hi',
        Option.join_some, Option.map_some, Option.getD_some, Function.update_self]
      constructor
      · linarith [LawfulBoundOps.addDown_le w0.1 w.1]
      · linarith [LawfulBoundOps.le_addUp w0.2 w.2]
    · have hne : i ≠ j := fun e => hji e.symm
      simp only [partialLo, partialHi, Array.set!, Array.getElem?_setIfInBounds_ne hne,
        Function.update_of_ne hji]
      exact h.2 j hj
  · next hnot =>
    have hnone : (cx[i]?).join = none := by
      rw [Array.getElem?_eq_getElem hi'] at hnot ⊢
      cases hc : cx[i] with
      | none => rfl
      | some w0 => exact (hnot w0 (by rw [hc])).elim
    simp only [partialLo, partialHi, hnone, Option.map_none, Option.getD_none,
      LawfulBoundOps.toReal_zero] at hfi
    have hf0 : f i = 0 := le_antisymm hfi.2 hfi.1
    refine ⟨by simp [h.1], fun j hj => ?_⟩
    by_cases hji : j = i
    · subst hji
      simp only [partialLo, partialHi, Array.set!, Array.getElem?_setIfInBounds_self_of_lt hi',
        Option.join_some, Option.map_some, Option.getD_some, Function.update_self, hf0,
        zero_add]
      exact hd
    · have hne : i ≠ j := fun e => hji e.symm
      simp only [partialLo, partialHi, Array.set!, Array.getElem?_setIfInBounds_ne hne,
        Function.update_of_ne hji]
      exact h.2 j hj

/-- A partial table's box encloses its coefficients. -/
theorem coeffBox_encloses {cx : PartialCoeffs α} {n : Nat} {f : Nat → ℝ}
    (h : PartialEncloses cx n f) {B : FlatBox α} (hB : coeffBox n cx = some B) :
    RowEncloses B n f := by
  unfold coeffBox at hB
  split at hB
  · cases hB
    refine ⟨rfl, fun i => ?_⟩
    simp only [Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_ofFn]
    exact h.2 i.val i.isLt
  · cases hB

/-- A table without entries stands for zero coefficients. -/
theorem coeffBox_none {cx : PartialCoeffs α} {n : Nat} {f : Nat → ℝ}
    (h : PartialEncloses cx n f) (hB : coeffBox n cx = none) (x : Nat → ℝ) :
    Directed.dot n f x = 0 := by
  unfold coeffBox at hB
  split at hB
  · cases hB
  · next hany =>
    simp only [Bool.not_eq_true, Array.any_eq_false] at hany
    simp only [Directed.dot, dotProduct, Function.comp_apply]
    refine Finset.sum_eq_zero fun i _ => ?_
    have hi : i.val < cx.size := h.1 ▸ i.isLt
    have hnone : (cx[i.val]?).join = none := by
      rw [Array.getElem?_eq_getElem hi]
      have := hany i.val hi
      cases hc : cx[i.val] with
      | none => rfl
      | some w => simp [hc] at this
    have hfi := h.2 i.val i.isLt
    simp only [partialLo, partialHi, hnone, Option.map_none, Option.getD_none,
      LawfulBoundOps.toReal_zero] at hfi
    rw [le_antisymm hfi.2 hfi.1, zero_mul]


/-- The row-major Horner fold over the extents and the indices of a coordinate takes the
accumulator `acc` to `acc` times the number of entries plus the coordinate's flat index. -/
theorem foldl_zip_toList : (l : List Nat) → (c : Shape.Coord l) → (acc : Nat) →
    (l.zip (Shape.Coord.toList _ c)).foldl (fun acc x => match x with | (e, i) => acc * e + i)
        acc = acc * Shape.size l + (Coord.linearize c).val
  | [], c, acc => by
      have h : (Coord.linearize c).val = 0 := by
        have := (Coord.linearize c).isLt
        simp only [Shape.size_nil] at this
        omega
      simp [Shape.Coord.toList, h]
  | n :: l, (i, c), acc => by
      simp only [Shape.Coord.toList, List.zip_cons_cons, List.foldl_cons]
      rw [foldl_zip_toList l c]
      have h := Coord.linearize_cons_val (n := n) (s := l) i c
      rw [h]
      simp only [Shape.size_cons]
      ring

/-- The product fold of a list from `acc` is `acc` times the number of entries of its shape. -/
theorem _root_.TorchLean.Shape.foldl_mul_eq_size : (l : List Nat) → (acc : Nat) →
    l.foldl (fun x y => x * y) acc = acc * Shape.size l
  | [], acc => by simp
  | n :: l, acc => by
      rw [List.foldl_cons, foldl_mul_eq_size l]
      simp only [Shape.size_cons]
      ring

/-- A sum below `m * n` is the double sum over the quotient and the remainder. -/
theorem sum_range_mul (f : Nat → ℝ) : (m n : Nat) →
    ∑ o ∈ Finset.range (m * n), f o = ∑ i ∈ Finset.range m, ∑ j ∈ Finset.range n, f (i * n + j)
  | 0, n => by simp
  | m + 1, n => by
      rw [Nat.succ_mul, Finset.sum_range_add, sum_range_mul f m n, Finset.sum_range_succ]



/-- Flat entry `n * t.size + q` of a map over the leading axes is flat entry `q` of the map at
sample `n`, the `n`-th row of the input viewed with one leading axis. -/
theorem getAtOrZero_mapLeading {lead s t : Shape} (f : Tensor ℝ s → Tensor ℝ t)
    (x : Tensor ℝ (lead ++ s)) (h : (lead ++ s).size = Shape.size (lead.size :: s))
    (n : Fin lead.size) (q : Fin t.size) :
    getAtOrZero (Tensor.reshape (Tensor.mapLeading lead f x) [_] (Shape.size_vector _).symm)
        [n.val * t.size + q.val] =
      getAtOrZero (Tensor.reshape (f (Tensor.unstack (Tensor.reshape x (lead.size :: s) h) n))
        [_] (Shape.size_vector _).symm) [q.val] := by
  have h₂ : Shape.size (lead.size :: t) = (lead ++ t).size := by
    simp only [Shape.size_cons, Shape.size_append]
  rw [Tensor.mapLeading_eq_reshape lead f x h h₂, reshape_vector_reshape,
    Op.Directed.getAtOrZero_eqRec]
  have hi : n.val * t.size + q.val < (Shape.dim lead.size t).size := by
    simp only [Shape.size_cons]
    nlinarith [n.isLt, q.isLt]
  rw [Tensor.getAtOrZero_eq_getScalar _ ⟨_, hi⟩, Tensor.getAtOrZero_eq_getScalar _ q]
  exact Tensor.getScalar_reshape_dim (fun i => f (Tensor.unstack (Tensor.reshape x _
      h) i)) n q hi

/-- The entry of row `n` at a coordinate `c` of a tensor viewed with one leading axis is its flat
entry `n * t.size + c`. -/
theorem getAtOrZero_unstack_reshape {s t : Shape} {m : Nat} (x : Tensor ℝ s)
    (h : s.size = Shape.size (m :: t)) (n : Fin m) (c : t.Coord) :
    getAtOrZero (Tensor.unstack (Tensor.reshape x (m :: t) h) n) (Shape.Coord.toList t c) =
      getAtOrZero (Tensor.reshape x [_] (Shape.size_vector _).symm)
        [n.val * t.size + (Coord.linearize c).val] := by
  rw [Tensor.getAtOrZero_coord_toList _ c rfl]
  simp only [Tensor.unstack_apply]
  rw [← Tensor.getAtOrZero_reshape_toFin (Tensor.reshape x (m :: t) h) (n, c),
    reshape_vector_reshape, Op.Directed.getAtOrZero_eqRec]
  congr 2
  have hl := Coord.linearize_cons_val (n := m) (s := t) n c
  rw [hl, Nat.add_comm, Nat.mul_comm]



/-- Reading the entries of a vector of naturals along an equality of lengths lists its data. -/
theorem map_finRange_getScalar {m d : Nat} (e : m = d) (t : Tensor Nat [d]) :
    (List.finRange m).map (fun j => t.getScalar (Fin.cast e j)) = t.data.toList := by
  subst e
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  rw [Tensor.getElem_data_toList t ⟨i, by simpa using h₁⟩]
  simp

/-- The `getD` of a vector's data at an index of the vector is its entry there. -/
theorem getD_data_toList {m d : Nat} (e : m = d) (t : Tensor Nat [d]) (j : Fin m) :
    t.data.toList.getD j.val 0 = t.getScalar (Fin.cast e j) := by
  subst e
  rw [List.getD_eq_getElem _ _ (by simp), Tensor.getElem_data_toList t j]
  rfl

/-- The output extents `convBack?` computes per spatial axis are those of
`outSpatialDilated`. -/
theorem map_finRange_slidingWindow {m d : Nat} (e : m = d)
    (T KT sT dT lT hT : Tensor Nat [d]) :
    (List.finRange m).map (fun j => Shape.slidingWindowOutDimDilated (T.data.toList.getD j.val 0)
        (KT.data.toList.getD j.val 0) (sT.getScalar (Fin.cast e j)) (dT.getScalar (Fin.cast e j))
        (lT.getScalar (Fin.cast e j)) (hT.getScalar (Fin.cast e j))) =
      (outSpatialDilated T KT sT dT lT hT).data.toList := by
  rw [← map_finRange_getScalar e (outSpatialDilated T KT sT dT lT hT)]
  congr 1
  funext j
  simp only [outSpatialDilated, Tensor.getScalar_ofFn, getD_data_toList e]

/-- A left fold adding `f i` over `List.range m` adds the sum of `f` below `m`. -/
theorem foldl_range_add (f : Nat → ℝ) : (m : Nat) → (R : ℝ) →
    (List.range m).foldl (fun R i => R + f i) R = R + ∑ i ∈ Finset.range m, f i
  | 0, R => by simp
  | m + 1, R => by
      rw [List.range_succ, List.foldl_append, foldl_range_add f m R, Finset.sum_range_succ]
      simp only [List.foldl_cons, List.foldl_nil]
      ring

/-- A sum over the flat entries of `(L, O, l...)` is the sum over the leading index, the channel and
the coordinates of `l`, at the row-major flat index `(n * O + oc) * size + c`. -/
theorem sum_range_channels (f : Nat → ℝ) (L O : Nat) (l : List Nat) :
    ∑ i ∈ Finset.range (L * (O * Shape.size l)), f i =
      ∑ n ∈ Finset.range L, ∑ oc ∈ Finset.range O, ∑ c : Shape.Coord l,
        f ((n * O + oc) * Shape.size l + (Coord.linearize c).val) := by
  rw [sum_range_mul]
  refine Finset.sum_congr rfl fun n _ => ?_
  rw [sum_range_mul]
  refine Finset.sum_congr rfl fun oc _ => ?_
  rw [Finset.sum_range]
  refine (Fintype.sum_bijective (fun c => Coord.linearize c)
    (Function.bijective_iff_has_inverse.mpr ⟨Coord.unlinearize,
      Coord.unlinearize_linearize, Coord.linearize_unlinearize⟩) _ _ fun c => ?_).symm
  congr 1
  ring



/-- A relation that each step keeps holds between a fold over `List.range m` and a fold over
`List.finRange m` that read the same indices. -/
theorem foldl_range_rel {β γ : Type} (rel : β → γ → Prop) (m : Nat) {left : β → Nat → β}
    {right : γ → Fin m → γ} {b : β} {c : γ} (h : rel b c)
    (step : ∀ (i : Fin m) b c, rel b c → rel (left b i.val) (right c i)) :
    rel ((List.range m).foldl left b) ((List.finRange m).foldl right c) := by
  rw [← List.map_coe_finRange_eq_range, List.foldl_map]
  exact List.foldl_rel h fun i _ b c hbc => step i b c hbc

/-- A relation that each step keeps at the index lists of the coordinates holds between two index
folds over the same extents. -/
theorem foldlIndices_rel_coord {β γ : Type} (rel : β → γ → Prop) (dims : List Nat)
    {left : β → List Nat → β} {right : γ → List Nat → γ} {b : β} {c : γ} (h : rel b c)
    (step : ∀ (k : Shape.Coord dims) b c, rel b c →
      rel (left b (Shape.Coord.toList _ k)) (right c (Shape.Coord.toList _ k))) :
    rel (Conv.foldlIndices dims b left)
      (Conv.foldlIndices dims c right) := by
  rw [Conv.foldlIndices_eq_foldl, Conv.foldlIndices_eq_foldl]
  refine Tensor.Rep.foldl_rel _ _ _ _ rel _ _ h fun k b c hbc => ?_
  simpa only [Tensor.Rep.coordinates, Tensor.Rep.get_ofFn,
    id] using step k b c hbc

/-- An entry of `grouped` adds, over the channels of its group and the kernel window, the
input read at the dilated index times the weight, then the bias. -/
theorem _root_.TorchLean.Conv.grouped_apply {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter inSpatial : Tensor Nat [d]} (groups : Nat)
    (W : Tensor ℝ (outC :: inC :: kernel.data.toList)) (bias : Tensor ℝ [outC])
    (input : Tensor ℝ (inC :: inSpatial.data.toList)) (oc : Fin outC)
    (c : Shape.Coord (outSpatialDilated inSpatial kernel stride dilation paddingBefore
      paddingAfter).data.toList) :
    grouped groups W bias input (oc, c) =
      (List.finRange (inC / groups)).foldl (fun acc lc =>
        Conv.foldlIndices kernel.data.toList acc fun acc kIdx =>
          acc + (match Conv.mkDilatedInputIdx? (Shape.Coord.toList _ c) kIdx
              stride.data.toList dilation.data.toList paddingBefore.data.toList with
            | none => 0
            | some inIdx =>
                getAtOrZero input ((oc.val / (outC / groups) * (inC / groups) + lc.val) :: inIdx)) *
            getAtOrZero W (oc.val :: (oc.val / (outC / groups) * (inC / groups) + lc.val) :: kIdx))
        0 + getAtOrZero bias [oc.val] := by
  simp only [grouped, add_def, Tensor.zipWith_apply,
    groupedContract, Conv.contractWith,
    Conv.biasBroadcastWith, Tensor.dim_apply,
    Tensor.generate, Tensor.Rep.get_ofFn]
  rfl


/-- An index list of the shape's length whose entries are below the extents is the index list of
a coordinate. -/
theorem exists_toList : (l : List Nat) → (idx : List Nat) → idx.length = l.length →
    (idx.zip l).all (fun x => decide (x.1 < x.2)) = true →
    ∃ c : Shape.Coord l, Shape.Coord.toList _ c = idx
  | [], [], _, _ => ⟨PUnit.unit, rfl⟩
  | n :: l, i :: idx, hlen, hall => by
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hall
      obtain ⟨c, hc⟩ := exists_toList l idx (by simpa using hlen) hall.2
      exact ⟨(⟨i, hall.1⟩, c), by simp [Shape.Coord.toList, hc]⟩

/-- The index list of a coordinate has its entries below the extents. -/
theorem all_toList : (l : List Nat) → (c : Shape.Coord l) →
    ((Shape.Coord.toList _ c).zip l).all (fun x => decide (x.1 < x.2)) = true
  | [], _ => rfl
  | n :: l, (i, c) => by
      simp only [Shape.Coord.toList, List.zip_cons_cons, List.all_cons, Bool.and_eq_true,
        decide_eq_true_eq]
      exact ⟨i.isLt, all_toList l c⟩

/-- A total read at an index list with an entry out of range is zero. -/
theorem _root_.TorchLean.Tensor.getAtOrZero_eq_zero_of_not_all {l : List Nat} (t : Tensor ℝ l)
    {idx : List Nat}
    (h : ¬ (idx.zip l).all (fun x => decide (x.1 < x.2)) = true) :
    getAtOrZero t idx = 0 := by
  rw [Tensor.getAtOrZero_eq_sum_indicator]
  refine Finset.sum_eq_zero fun c _ => ite_eq_right fun hc => h ?_
  rw [hc]
  exact all_toList l c

/-- A dilated input index has one entry per output index. -/
theorem _root_.TorchLean.Conv.length_mkDilatedInputIdx? :
    {outIdx kIdx stride dilation pads inIdx : List Nat} →
    Conv.mkDilatedInputIdx? outIdx kIdx stride dilation pads = some inIdx →
      inIdx.length = outIdx.length
  | [], [], [], [], [], inIdx, h => by
      simp only [Conv.mkDilatedInputIdx?, Option.some.injEq] at h
      simp [← h]
  | o :: os, k :: ks, s :: ss, d :: ds, p :: ps, inIdx, h => by
      simp only [Conv.mkDilatedInputIdx?] at h
      split at h
      · cases h
      · split at h
        · cases h
        · next rest hrest =>
          simp only [Option.some.injEq] at h
          subst h
          simp [length_mkDilatedInputIdx? hrest]
  | [], [], [], [], _ :: _, _, h | [], [], [], _ :: _, _, _, h | [], [], _ :: _, _, _, _, h
  | [], _ :: _, _, _, _, _, h | _ :: _, [], _, _, _, _, h | _ :: _, _ :: _, [], _, _, _, h
  | _ :: _, _ :: _, _ :: _, [], _, _, h | _ :: _, _ :: _, _ :: _, _ :: _, [], _, h => by
      simp [Conv.mkDilatedInputIdx?] at h


/-- A tap that reads the input at entry `j` adds `ao` times the weight `w` there: the pairing gains
`ao` times the input read times the weight. -/
theorem accum_tap {N : Nat} {xf g g₁ : Nat → ℝ} {ao acc read kread : ℝ} {cx : PartialCoeffs α}
    {j : Nat} {alo ahi w : α} (hpe : PartialEncloses cx N g₁)
    (hd : Directed.dot N g₁ xf = Directed.dot N g xf + ao * acc) (hj : j < N)
    (hao : value alo ≤ ao ∧ ao ≤ value ahi) (hread : read = xf j) (hk : kread = value w) :
    ∃ g', PartialEncloses (accum cx j (intervalMul alo ahi w w)) N g' ∧
      Directed.dot N g' xf = Directed.dot N g xf + ao * (acc + read * kread) := by
  refine ⟨_, accum_encloses hpe hj
    (Graph.intervalMul_encloses hao.1 hao.2 (le_refl (value w)) (le_refl (value w))), ?_⟩
  rw [dot_bump hj, hd, hread, hk]
  ring

/-- A tap that reads padding leaves the coefficients and adds nothing. -/
theorem padding_tap {N : Nat} {xf g g₁ : Nat → ℝ} {ao acc read kread : ℝ} {cx : PartialCoeffs α}
    (hpe : PartialEncloses cx N g₁) (hd : Directed.dot N g₁ xf = Directed.dot N g xf + ao * acc)
    (hread : read = 0) :
    ∃ g', PartialEncloses cx N g' ∧ Directed.dot N g' xf =
      Directed.dot N g xf + ao * (acc + read * kread) :=
  ⟨g₁, hpe, by rw [hd, hread]; ring⟩

/-- One output entry of the transposed convolution keeps the pairing: when its taps add `ao` times
the gathered sum `core` to the pairing of the coefficients with the inputs, adding the interval of
`ao` times the bias `bv` to the constant adds `ao` times the output `core + bv` to the objective. -/
theorem bias_step {n : Nat} {xf g : Nat → ℝ} {e R ao core vb : ℝ} {cxI : PartialCoeffs α}
    {c₁ c₂ alo ahi bv : α} (he₁ : value c₁ ≤ e) (he₂ : e ≤ value c₂)
    (hR : Directed.dot n g xf + e = R) (hao : value alo ≤ ao ∧ ao ≤ value ahi) (hbv : value bv = vb)
    (hin : ∃ g', PartialEncloses cxI n g' ∧ Directed.dot n g' xf =
      Directed.dot n g xf + ao * core) :
    ∃ g' e', PartialEncloses cxI n g' ∧
      value (BoundOps.addDown c₁ (intervalMul alo ahi bv bv).1) ≤ e' ∧
        e' ≤ value (BoundOps.addUp c₂ (intervalMul alo ahi bv bv).2) ∧
          Directed.dot n g' xf + e' = R + ao * (core + vb) := by
  subst hbv
  obtain ⟨g', hpe, hd⟩ := hin
  have ht := Graph.intervalMul_encloses hao.1 hao.2 (le_refl (value bv))
    (le_refl (value bv))
  refine ⟨g', e + ao * value bv, hpe, (LawfulBoundOps.addDown_le _ _).trans (add_le_add he₁ ht.1),
    (add_le_add he₂ ht.2).trans (LawfulBoundOps.le_addUp _ _), ?_⟩
  rw [hd, ← hR]
  ring


/-- The transposed convolution `convBack?` of a coefficient box that encloses `a` encloses
the adjoint of the real convolution: there are a real input coefficient `a'`, enclosed by the input
box it returns (and pairing to zero when it returns none), and a real constant `d` in the bias
interval, such that `a` paired with the operator's rule at `ℝ` on the reals of the kernel and the
bias and on any input `x` is `a'` paired with `x` plus `d`. -/
theorem convBack?_sound {s k : Shape} {r : Nat} {dims : Shape.ConvDims s k r}
    {stride padLow padHigh dilation : Fin r → Nat} {groups : Nat} {lead xs ys : Shape}
    {hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size}
    {hx : xs = s.expandAxis (dims.input ⟨0, by omega⟩) lead}
    {hy : ys = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead}
    {rule : Op.Rule ℝ _ _}
    (hs : (Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ) =
      some rule)
    {K : Tensor α k} {b : Tensor α [k.extent (dims.kernel ⟨0, by omega⟩)]} {X aY : FlatBox α}
    {aX : Option (FlatBox α)} {c : α × α}
    (hback : convBack? dims stride padLow padHigh dilation groups lead K b X aY =
      some (aX, c))
    (hX : X.dim = xs.size) {a : Nat → ℝ} (ha : RowEncloses aY ys.size a) (x : Tensor ℝ xs) :
    ∃ (a' : Nat → ℝ) (d : ℝ), (∀ B, aX = some B → RowEncloses B xs.size a') ∧
      (aX = none → ∀ y, Directed.dot xs.size a' y = 0) ∧ value c.1 ≤ d ∧ d ≤ value c.2 ∧
      ∑ o : ys.Coord, a (Coord.linearize o).val *
          (rule.forward (Tensor.map value K, Tensor.map value b, x, ())).1 o =
        ∑ j : xs.Coord, a' (Coord.linearize j).val * x j + d := by
  unfold convBack? at hback
  split at hback
  · rename_i N C sp O I ks dims K b
    extract_lets outSp size at hback
    split at hback
    · next hcond =>
      obtain ⟨hcf, hI, hks, hsp, hg, hXd, hYd⟩ := hcond
      split at hback
      next cx c' hfold =>
      rename_i flat strides dilations pads init _
      simp only [Option.some.injEq, Prod.mk.injEq] at hback
      obtain ⟨rfl, rfl⟩ := hback
      subst hI
      obtain ⟨d, T, hT⟩ : ∃ d, ∃ T : Tensor Nat [d], T.data.toList = sp :=
        ⟨_, vecOf sp rfl, vecOf_toList sp rfl⟩
      subst hT hsp
      obtain ⟨KT, hKT⟩ : ∃ KT : Tensor Nat [d], KT.data.toList = ks :=
        ⟨vecOf ks (by simpa using hks), vecOf_toList _ _⟩
      subst hKT
      have hlen : KT.data.toList.length = T.data.toList.length := by simp
      obtain rfl := channelsFirst_eq hlen hcf
      have hd : T.data.toList.length = d := by simp
      obtain ⟨sT, rfl⟩ : ∃ sT : Tensor Nat [d], stride = fun j => sT.getScalar (Fin.cast hd j) :=
        ⟨Tensor.ofFn fun j => stride (Fin.cast hd.symm j), by funext j; simp⟩
      obtain ⟨lT, rfl⟩ : ∃ lT : Tensor Nat [d], padLow = fun j => lT.getScalar (Fin.cast hd j) :=
        ⟨Tensor.ofFn fun j => padLow (Fin.cast hd.symm j), by funext j; simp⟩
      obtain ⟨hT, rfl⟩ : ∃ hT : Tensor Nat [d], padHigh = fun j => hT.getScalar (Fin.cast hd j) :=
        ⟨Tensor.ofFn fun j => padHigh (Fin.cast hd.symm j), by funext j; simp⟩
      obtain ⟨dT, rfl⟩ : ∃ dT : Tensor Nat [d], dilation = fun j => dT.getScalar (Fin.cast hd j) :=
        ⟨Tensor.ofFn fun j => dilation (Fin.cast hd.symm j), by funext j; simp⟩
      have hN : N = lead.size := hlead
      subst hN hx hy
      have hxs : Shape.expandAxis (lead.size :: C :: T.data.toList)
          ((Shape.ConvDims.channelsFirst (n := lead.size) (c := C) (o := O) (i := C / groups)
            hlen).input ⟨0, by omega⟩) lead =
          lead ++ (C :: Tensor.to T (List Nat)) := by
        rfl
      have h₃ := congrArg Shape.size hxs
      have hF : flatOfSome ⟨_, (rule.forward (Tensor.map value K, Tensor.map value b, x, ())).1⟩ =
          flatOfSome ⟨_, Tensor.mapLeading lead
            (grouped (inSpatial := T) (stride := sT) (dilation := dT)
              (paddingBefore := lT) (paddingAfter := hT) groups
              (Tensor.map value (denseKernel groups K)) (Tensor.map value b))
            (Tensor.reshape x _ h₃)⟩ := by
        rw [Op.rule?_conv] at hs
        cases hs
        refine Eq.trans ?_ (flat_conv hlen hg rfl (Shape.size_eq_of_expandAxis hlead rfl)
          (Shape.size_conv_expandAxis hlead rfl) h₃ (Tensor.map value (denseKernel groups K))
          (Tensor.map value b) x)
        rw [packedKernel_map, packedKernel_denseKernel]
        rfl
      have hstr : strides = sT.data.toList := map_finRange_getScalar hd sT
      have hdil : dilations = dT.data.toList := map_finRange_getScalar hd dT
      have hpad : pads = lT.data.toList := map_finRange_getScalar hd lT
      have hout : outSp = (outSpatialDilated T KT sT dT lT hT).data.toList :=
        map_finRange_slidingWindow hd T KT sT dT lT hT
      have hsz : ∀ l, size l = Shape.size l := fun l =>
        (Shape.foldl_mul_eq_size l 1).trans (Nat.one_mul _)
      have hfl : ∀ l (c : Shape.Coord l), flat l (Shape.Coord.toList _ c) =
          (Coord.linearize c).val := fun l c => (foldl_zip_toList l c 0).trans (by simp)
      rw [hstr, hdil, hpad, hout] at hfold
      rw [hout, hsz] at hYd
      rw [hsz] at hXd
      have hL : (lead ++ (C :: Tensor.to T (List Nat))).size =
          Shape.size (lead.size :: (C :: Tensor.to T (List Nat))) := by
        simp only [Shape.size_append, Shape.size_cons]
      let xf : Nat → ℝ := fun j => getAtOrZero (Tensor.reshape x [_] (Shape.size_vector _).symm) [j]
      let mf : Nat → ℝ := fun o => getAtOrZero (flatOfSome ⟨_, Tensor.mapLeading lead
            (grouped (inSpatial := T) (stride := sT) (dilation := dT)
              (paddingBefore := lT) (paddingAfter := hT) groups
              (Tensor.map value (denseKernel groups K)) (Tensor.map value b))
            (Tensor.reshape x _ h₃)⟩).v [o]
      let Rel : PartialCoeffs α × α × α → ℝ → Prop := fun acc R =>
        ∃ g e, PartialEncloses acc.1 X.dim g ∧ value acc.2.1 ≤ e ∧ e ≤ value acc.2.2 ∧
          Directed.dot X.dim g xf + e = R
      have key : Rel (cx, c') ((List.range lead.size).foldl (fun R n => (List.range O).foldl
          (fun R oc => Conv.foldlIndices (outSpatialDilated T KT sT dT lT
              hT).data.toList R fun R outIdx =>
            R + a ((n * O + oc) * size (outSpatialDilated T KT sT dT lT hT).data.toList + flat
                (outSpatialDilated T KT sT dT lT hT).data.toList outIdx) *
              mf ((n * O + oc) * size (outSpatialDilated T KT sT dT lT hT).data.toList + flat
                  (outSpatialDilated T KT sT dT lT hT).data.toList outIdx)) R) 0) := by
        rw [← hfold]
        refine List.foldl_rel ⟨fun _ => 0, 0, partialEncloses_replicate X.dim,
          by simp [init, LawfulBoundOps.toReal_zero],
          by simp [init, LawfulBoundOps.toReal_zero], by simp [Directed.dot, dotProduct]⟩ ?_
        intro n hn acc R hrel
        refine List.foldl_rel hrel ?_
        intro oc hoc acc R hrel
        refine foldlIndices_rel_coord Rel (outSpatialDilated T KT sT dT lT hT).data.toList hrel
            ?_
        intro cout acc R hrel
        obtain ⟨cxa, ca⟩ := acc
        obtain ⟨g, e, hpe, he₁, he₂, hR⟩ := hrel
        dsimp only
        have hn' : n < lead.size := List.mem_range.mp hn
        have hoc' : oc < O := List.mem_range.mp hoc
        have hP : size (outSpatialDilated T KT sT dT lT hT).data.toList = Shape.size
            (outSpatialDilated T KT sT dT lT hT).data.toList := hsz
                (outSpatialDilated T KT sT dT lT hT).data.toList
        have hflo := hfl (outSpatialDilated T KT sT dT lT hT).data.toList cout
        have hlo : (Coord.linearize (s := O :: (outSpatialDilated T KT sT dT lT
            hT).data.toList) (⟨oc, hoc'⟩, cout)).val =
            (Coord.linearize cout).val + Shape.size (outSpatialDilated T KT sT dT lT
                hT).data.toList * oc := by
          have hl := Coord.linearize_cons_val (n := O) (s :=
              (outSpatialDilated T KT sT dT lT hT).data.toList)
            ⟨oc, hoc'⟩ cout
          rw [hl]
        have ho : (n * O + oc) * size (outSpatialDilated T KT sT dT lT hT).data.toList + flat
            (outSpatialDilated T KT sT dT lT hT).data.toList (Shape.Coord.toList
                (outSpatialDilated T KT sT dT lT hT).data.toList cout) =
            n * Shape.size (O :: (outSpatialDilated T KT sT dT lT hT).data.toList) +
              (Coord.linearize (s := O :: (outSpatialDilated T KT sT dT lT
                  hT).data.toList) (⟨oc, hoc'⟩, cout)).val := by
          rw [hP, hflo, hlo]
          simp only [Shape.size_cons]
          ring
        have hlt : (n * O + oc) * size (outSpatialDilated T KT sT dT lT hT).data.toList + flat
            (outSpatialDilated T KT sT dT lT hT).data.toList (Shape.Coord.toList
                (outSpatialDilated T KT sT dT lT hT).data.toList cout) < aY.dim := by
          rw [hYd, hP, hflo]
          have h := Nat.mul_add_lt_mul_of_lt_of_lt (Nat.mul_add_lt_mul_of_lt_of_lt hn' hoc')
            (Coord.linearize cout).isLt
          rwa [Nat.mul_comm _ n, Nat.mul_comm, Nat.mul_assoc] at h
        have hao := ha.2 ⟨_, ha.1 ▸ hlt⟩
        have hmf : mf ((n * O + oc) * size (outSpatialDilated T KT sT dT lT hT).data.toList +
            flat (outSpatialDilated T KT sT dT lT hT).data.toList (Shape.Coord.toList _ cout)) =
            grouped (inSpatial := T) (stride := sT) (dilation := dT)
              (paddingBefore := lT) (paddingAfter := hT) groups
              (Tensor.map value (denseKernel groups K)) (Tensor.map value b)
              (Tensor.unstack (Tensor.reshape (Tensor.reshape x _ h₃) _ hL) ⟨n, hn'⟩)
              (⟨oc, hoc'⟩, cout) := by
          simp only [mf, flatOfSome]
          rw [ho, getAtOrZero_mapLeading _ _ hL ⟨n, hn'⟩
            (Coord.linearize (s := O :: (outSpatialDilated T KT sT dT lT
                hT).data.toList) (⟨oc, hoc'⟩, cout)),
            Tensor.getAtOrZero_reshape_toFin]
          rfl
        rw [hmf, grouped_apply]
        refine bias_step he₁ he₂ hR hao (getAtOrZero_map_value b [oc]).symm ?_
        let Rin : PartialCoeffs α → ℝ → Prop := fun cxb accb => ∃ g', PartialEncloses cxb X.dim
            g' ∧
            Directed.dot X.dim g' xf =
            Directed.dot X.dim g xf + a ((n * O + oc) * size (outSpatialDilated T KT
                sT dT lT hT).data.toList + flat (outSpatialDilated T KT sT dT lT hT).data.toList
              (Shape.Coord.toList _ cout)) * accb
        refine foldl_range_rel Rin (C / groups) ⟨g, hpe, by ring⟩ ?_
        intro lc cxb accb hrel
        refine foldlIndices_rel_coord Rin KT.data.toList hrel ?_
        intro ck cxk acck hk
        obtain ⟨g₁, hpe₁, hd₁⟩ := hk
        dsimp only [Rin]
        cases hmk : Conv.mkDilatedInputIdx? (Shape.Coord.toList
            (outSpatialDilated T KT sT dT lT hT).data.toList cout)
            (Shape.Coord.toList KT.data.toList ck) sT.data.toList dT.data.toList
            lT.data.toList with
        | none => exact ⟨g₁, hpe₁, by rw [hd₁]; ring⟩
        | some inIdx =>
          dsimp only
          split
          · next hinb =>
            have hlenIn : inIdx.length = T.data.toList.length := by
              rw [length_mkDilatedInputIdx? hmk, Shape.Coord.length_toList]
              simp [hd]
            obtain ⟨cin, rfl⟩ := exists_toList _ inIdx hlenIn hinb
            have hic : oc / (O / groups) * (C / groups) + lc.val < C :=
              packedChannel_lt hg hoc' lc.isLt
            refine accum_tap hpe₁ hd₁ ?_ hao ?_ ?_
            · rw [hsz, hfl, hXd]
              have h := Nat.mul_add_lt_mul_of_lt_of_lt (Nat.mul_add_lt_mul_of_lt_of_lt hn' hic)
                (Coord.linearize cin).isLt
              rwa [Nat.mul_comm _ n, Nat.mul_comm, Nat.mul_assoc] at h
            · have h := getAtOrZero_unstack_reshape (Tensor.reshape x _ h₃) hL ⟨n, hn'⟩
                ((⟨_, hic⟩, cin) : Shape.Coord (C :: T.data.toList))
              simp only [reshape_vector_reshape] at h
              refine h.trans ?_
              show getAtOrZero _ [_] = getAtOrZero _ [_]
              congr 2
              have hl := Coord.linearize_cons_val (n := C)
                (s := T.data.toList) ⟨_, hic⟩ cin
              rw [hl, hsz, hfl]
              simp only [Shape.size_cons, Tensor.to_list_eq_data]
              ring
            · rw [getAtOrZero_map_value]
              congr 1
              have h := getAtOrZero_packedKernel groups hg (denseKernel groups K) ⟨oc, hoc'⟩ lc
                (Shape.Coord.toList _ ck)
              rw [packedKernel_denseKernel] at h
              exact h.symm
          · next hinb =>
            refine padding_tap hpe₁ hd₁ ?_
            refine getAtOrZero_eq_zero_of_not_all (l := C :: T.data.toList) _ fun h => hinb ?_
            simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at h
            exact h.2
      obtain ⟨g, e, hpe, he₁, he₂, hR⟩ := key
      refine ⟨g, e, fun B hB => by rw [← hX]; exact coeffBox_encloses hpe hB,
        fun h y => by rw [← hX]; exact coeffBox_none hpe h y, he₁, he₂, ?_⟩
      have hY : ∀ o, (rule.forward (Tensor.map value K, Tensor.map value b, x, ())).1 o =
          mf (Coord.linearize o).val := fun o => by
        rw [← Tensor.getAtOrZero_reshape_toFin _ o]
        exact congrArg (fun F : FlatTensor ℝ => getAtOrZero F.v [(Coord.linearize o).val]) hF
      have hx' : ∀ j, x j = xf (Coord.linearize j).val := fun j =>
        (Tensor.getAtOrZero_reshape_toFin x j).symm
      simp only [hY, hx']
      rw [← dot_eq_sum_coord, ← dot_eq_sum_coord, ← hX, hR]
      simp only [Directed.dot, dotProduct, Function.comp_apply]
      rw [Fin.sum_univ_eq_sum_range (fun i => a i * mf i), ← ha.1, hYd, sum_range_channels]
      simp only [Conv.foldlIndices_add, foldl_range_add, hsz, hfl, zero_add]
    · cases hback
  · cases hback

/-- A convolution with constant kernel and bias passes its transposed coefficient to its input and
its bias term to the constant. -/
theorem conv_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) {p : Nat} (hp : p < k)
    {s k' : Shape} {r : Nat} {cd : Shape.ConvDims s k' r}
    {stride padLow padHigh dilation : Fin r → Nat} {groups : Nat} {lead xs ys : Shape}
    {hlead : s.extent (cd.input ⟨0, by omega⟩) = lead.size}
    {hx : xs = s.expandAxis (cd.input ⟨0, by omega⟩) lead}
    {hy : ys = (s.conv k' cd stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (cd.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead}
    {K : Tensor α k'} {b : Tensor α [k'.extent (cd.kernel ⟨0, by omega⟩)]} {X aY : FlatBox α}
    {aX : Option (FlatBox α)} {c : α × α}
    (hback : convBack? cd stride padLow padHigh dilation groups lead K b X aY =
      some (aX, c))
    (hX : X.dim = dims p) (hdp : dims p = xs.size) (hdk : dims k = ys.size)
    (heq : ∀ rule, (Op.conv cd stride padLow padHigh dilation groups hlead hx hy).rule?
        (α := ℝ) = some rule → ∀ x : Tensor ℝ xs, (∀ c, x c = v p (Coord.linearize c).val) →
      ∀ c, v k (Coord.linearize c).val =
        (rule.forward (Tensor.map value K, Tensor.map value b, x, ())).1 c)
    {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      (addConstant (match aX with | some B => addCoeff st p B | none => st)
        c.1 c.2) := by
  obtain ⟨rule, hs⟩ : ∃ rule, (Op.conv cd stride padLow padHigh dilation groups hlead hx
      hy).rule? (α := ℝ) = some rule := ⟨_, Op.rule?_conv ..⟩
  obtain ⟨x, hx'⟩ : ∃ x : Tensor ℝ xs, ∀ c, x c = v p (Coord.linearize c).val :=
    ⟨Tensor.Rep.ofFn _, Tensor.Rep.get_ofFn _⟩
  obtain ⟨a', d, hB, hnone, hd₁, hd₂, hdot⟩ := convBack?_sound hs hback (hX.trans hdp)
    (by rw [← hdk]; exact ha) x
  have hdot' : Directed.dot (dims k) a (v k) = Directed.dot (dims p) a' (v p) + d := by
    rw [hdk, hdp, dot_eq_sum_coord, dot_eq_sum_coord]
    simp only [heq rule hs x hx', ← hx']
    exact hdot
  cases aX with
  | none =>
    refine Or.inr ⟨rfl, ?_⟩
    have hr := represents_addConstant h c.1 c.2 d ⟨hd₁, hd₂⟩
    rwa [hdot', hdp, hnone rfl, zero_add]
  | some B =>
    refine Or.inr ⟨addCoeff_size .., ?_⟩
    have hr := represents_add h p (by omega) (mem_pending_of_lt hp) B a'
      (by rw [hdp]; exact hB B rfl)
    have hr' := represents_addConstant hr c.1 c.2 d ⟨hd₁, hd₂⟩
    rwa [hdot', ← add_assoc]

/-- The boxes of a list of references, when all exist, have their nodes' widths. -/
theorem mapM_box?_dim {G : IR.Graph} {P : IR.Payload α}
    {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx} {dims : Nat → Nat}
    {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v) :
    ∀ {l : List IR.Ref} {boxes : List (FlatBox α)}, l.mapM (box? ibp) = some boxes →
      boxes.length = l.length ∧
        ∀ j (h₁ : j < boxes.length) (h₂ : j < l.length), boxes[j].dim = dims l[j].node
  | [], boxes, h => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    simp
  | r :: l, boxes, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨B, hB, bs, hbs, rfl⟩ := h
    obtain ⟨hlen, hdim⟩ := mapM_box?_dim hG hbs
    refine ⟨by simp [hlen], fun j h₁ h₂ => ?_⟩
    cases j with
    | zero => exact (box?_encloses hG hB).1
    | succ j => exact hdim j (by simpa using h₁) (by simpa using h₂)

/-- A concatenation passes each block of the coefficient to the reference it came from. -/
theorem cat_sound {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : State α} {z : ℝ} (h : Represents dims v input k st z)
    (hk : k < st.coeffs.size) (layout : ConcatLayout) (parents : List Nat)
    (hsize : parents.length = layout.lengths.length) (hpar : ∀ p ∈ parents, p < k)
    (hdim : ∀ j : Fin layout.lengths.length,
      dims parents.toArray[j.val]! = (layout.parentShape j).size)
    (hy : ∀ (j : Fin layout.lengths.length) (i : Fin (layout.parentShape j).size),
      v k (layout.flatEquiv ⟨j, i⟩).val = v parents.toArray[j.val]! i.val)
    {aY : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aY (dims k) a)
    {cs : Array (FlatBox α)} (hc : splitCoeff layout aY = some cs) :
    SweepInvariant dims v input st.coeffs.size k (z + Directed.dot (dims k) a (v k))
      ((parents.zip cs.toList).foldl (fun state e => addCoeff state e.1 e.2) st) := by
  have hd : dims k = layout.outputShape.size := by
    unfold splitCoeff at hc
    split at hc
    · next hw => exact ha.1.symm.trans hw
    · cases hc
  rw [hd] at ha ⊢
  have hr := represents_concat h layout parents.toArray (by simpa using hsize)
    (fun p hp => by have := hpar p (by simpa using hp); omega)
    (fun p hp => mem_pending_of_lt (hpar p (by simpa using hp))) hdim ha hy hc
  refine Or.inr ⟨by rw [← List.foldl_toArray]; exact addParents_size .., ?_⟩
  rw [← Array.foldl_toList, Array.toList_zip, List.toList_toArray] at hr
  exact hr

/-- The shared rules are sound in each direction at operator node `k` when its real equation holds.

The proof takes the arms of `directedRule` by operator and reference list. A detach or reshape of
one reference passes the coefficient to it, a sum of two references passes it to both, and a
difference of two references passes it to the first and its negation to the second. A linear layer
of weight, bias and input passes its transposed coefficient to the input and its bias term to the
constant, and so does a convolution of kernel, bias and input with its transposed convolution. A sum
of one reference over every axis in order without `keepdim`, a concatenation, a slice of one
reference along one axis with step 1 and a permutation of the axes of one reference pass their
transposed coefficients to their references. Each rule fails when a lookup or a width check fails
(for the slice: `start ≤ stop`, its slice layout, a box of its reference of the layout's source
width and a coefficient of the layout's output width), and a failed state satisfies the invariant
trivially. Gather, scatter, pad, flip, contract, log-softmax, the cumulative sums, products, maxima
and minima, sort, top-k and custom operators fail with any references. Every other case calls
`consumeAt`, which discharges the coefficient against the node's box and fails when the node has no
box: a sum over other axes, a slice of another form, every other operator, and a copy, sum,
difference, linear layer, axis sum, slice, permutation or convolution with another reference
list. -/
theorem directedRule_sound [Op.Capabilities α] (dir : BackwardDir)
    {G : IR.Graph} {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v) {input k : Nat}
    (heq : ∀ o refs, G.nodes[k]? = some (.op o refs) → Equation G P dims v k o refs.toList) :
    RuleSound dir G (directedRule G P ibp) dims v input k := by
  intro o refs hnode _ st z aY a hk h ha
  refine DirInvariant.of_eq ?_
  obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.mp hnode
  have hpar : ∀ r ∈ refs.toList, r.node < k := fun r hr =>
    hG.topo k hlt o refs hnode r (Array.mem_toList_iff.mp hr)
  have hop := heq o refs hnode
  generalize refs.toList = l at hpar hop
  obtain ⟨ins, outs, op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›)
  case linear =>
    rcases l with _ | ⟨rW, _ | ⟨rb, _ | ⟨rx, _ | ⟨r₄, l⟩⟩⟩⟩
    all_goals try exact consumeAt_sound hG h ha
    unfold directedRule
    dsimp only
    split
    · next W b hW hb =>
      obtain ⟨hdp, hdk, hlin⟩ := hop W b hW hb
      exact linear_sound h hk (hpar _ (by simp)) hdp hdk hlin ha
    · exact Or.inl rfl
  case «reduce» s rop axes kd _ =>
    cases rop <;> cases kd <;> rcases l with _ | ⟨r, _ | ⟨r₂, l⟩⟩
    all_goals try exact consumeAt_sound hG h ha
    unfold directedRule
    dsimp only
    split
    · next hax =>
      split
      · next Bx hBx =>
        exact sum_sound h hk (hpar _ (by simp)) (box?_encloses hG hBx).1 (hop hax) ha
      · exact Or.inl rfl
    · exact consumeAt_sound hG h ha
  case cat s dim sizes =>
    unfold directedRule
    dsimp only
    split
    · next parts hc =>
      split at hc
      · next hlen =>
        simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hc
        obtain ⟨boxes, hboxes, hc⟩ := hc
        split at hc
        · next hall =>
          simp only [Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hc
          obtain ⟨cs, hcs, rfl⟩ := hc
          have hlen' : l.length = sizes.length := beq_iff_eq.mp hlen
          obtain ⟨hbl, hbd⟩ := mapM_box?_dim hG hboxes
          refine cat_sound h hk (catLayout s dim.val sizes) (l.map (·.node))
            (by simpa [catLayout] using hlen')
            (fun p hp => by
              obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hp
              exact hpar r hr)
            (fun j => ?_) (hop hlen') ha hcs
          have hj : j.val < l.length := by
            have := j.isLt; simp only [catLayout] at this; omega
          have hj' : j.val < boxes.length := by omega
          have hmem : (boxes[j.val], j) ∈
              boxes.zip (List.finRange (catLayout s dim.val sizes).lengths.length) := by
            have hz : j.val <
                (boxes.zip (List.finRange (catLayout s dim.val sizes).lengths.length)).length := by
              simp only [List.length_zip, List.length_finRange]; omega
            have := List.getElem_mem hz
            simpa only [List.getElem_zip, List.getElem_finRange, Fin.cast_mk, Fin.eta] using this
          have hB := List.all_eq_true.mp hall _ hmem
          simp only [beq_iff_eq] at hB
          rw [← hB, hbd j.val hj' hj,
            getElem!_pos (c := (l.map (·.node)).toArray) (i := j.val) (by simpa using hj)]
          simp
        · simp at hc
      · simp at hc
    · exact Or.inl rfl
  case slice s axes starts stops steps _ =>
    rcases l with _ | ⟨r, _ | ⟨r₂, l⟩⟩
    all_goals try exact consumeAt_sound hG h ha
    unfold directedRule
    dsimp only
    split
    · rename_i _ _ axis start stop hsv hax
      split
      · next c hc =>
        split at hc
        · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hc
          obtain ⟨layout, hlay, B, hB, hc⟩ := hc
          split at hc
          · next hBd =>
            exact slice_sound h hk (hpar _ (by simp)) layout
              ((box?_encloses hG hB).1.symm.trans (beq_iff_eq.mp hBd))
              (hop axis start stop layout hax rfl rfl rfl hlay) ha hc
          · simp at hc
        · simp at hc
      · exact Or.inl rfl
    · exact consumeAt_sound hG h ha
  case permute s σ out hout =>
    rcases l with _ | ⟨r, _ | ⟨r₂, l⟩⟩
    all_goals try exact consumeAt_sound hG h ha
    unfold directedRule
    dsimp only
    split
    · next aX hX => exact permute_sound h hk (hpar _ (by simp)) hout hop.1 hop.2 ha hX
    · exact Or.inl rfl
  case conv s k' r cd stride padLow padHigh dilation groups lead xs ys hlead hx hy =>
    rcases l with _ | ⟨rK, _ | ⟨rb, _ | ⟨rx, _ | ⟨r₄, l⟩⟩⟩⟩
    all_goals try exact consumeAt_sound hG h ha
    unfold directedRule
    dsimp only
    split
    · next aX c hback =>
      simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hback
      obtain ⟨K, hK, b, hb, X, hXb, hc⟩ := hback
      obtain ⟨hdp, hdk, hy'⟩ := hop K b hK hb
      exact conv_sound h hk (hpar _ (by simp)) hc (box?_encloses hG hXb).1 hdp hdk hy' ha
    · exact Or.inl rfl
  all_goals first
    | exact Or.inl rfl
    | exact consumeAt_sound hG h ha
    | rcases l with _ | ⟨r₁, _ | ⟨r₂, _ | ⟨r₃, l⟩⟩⟩ <;> first
        | exact consumeAt_sound hG h ha
        | exact copy_sound h hk (hpar _ (by simp)) hop ha
        | exact add_sound h hk (hpar _ (by simp)) (hpar _ (by simp)) hop ha
        | exact sub_sound h hk (hpar _ (by simp)) (hpar _ (by simp)) hop ha

/-- At a real point of the graph whose operator nodes satisfy their real equations, the reverse
sweep of the shared rules (`directedStep`) bounds the objective `obj · v out` between the values of
its lower and upper affine forms at the input's coordinates. -/
theorem directedSweep_encloses [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {ibp : Array (Option (FlatBox α))} {ctx : AffineCtx}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} (hG : GraphPoint G P ibp ctx dims v)
    (heq : ∀ k o refs, G.nodes[k]? = some (.op o refs) →
      Equation G P dims v k o refs.toList)
    {out : Nat} {obj : FlatTensor α} (hobj : obj.n = dims out)
    {lower upper : Linear α ctx.inputDim 1}
    (h : sweep (directedStep G P ibp ctx) G ctx ibp out obj = some (lower, upper)) :
    affineValue lower (fun i => v ctx.inputId i.val) ≤ objectiveValue dims v out obj ∧
      objectiveValue dims v out obj ≤ affineValue upper (fun i => v ctx.inputId i.val) :=
  sweep_encloses hG (fun dir k _ => directedRule_sound dir hG (heq k)) hobj h

end Lawful

end TorchLean.CROWN.Op
