/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op
public import TorchLean.Verify.CROWN.FlatBox.Matmul
public import TorchLean.Verify.CROWN.FlatBox.Flatten

/-!
# The IBP certificate on operator graphs

`step?` is the interval step at a node of a graph of registered operators, at any scalar and for
any rule on operator nodes. An input node reads its seed box, a constant node its point box, and an
operator node the rule on the boxes its references name. `shared?` is such a rule at any scalar
with endpoint operations (`BoundOps`, `NonlinearBoundOps`), with a rule for each operator of kind
`shared` in the table `ruleKind`. The certificate rule `cert?` at `ℝ` uses endpoint
min-max for tanh and sigmoid and clipped midpoints for sine and cosine (`CROWN.IBP`, the
operators of kind `backend`) and is `shared?` at every other operator (`cert?_eq_shared?`),
and the certificate step is `step?` at `cert?`.

Tanh and sigmoid increase, so their endpoint min-max is Eq. (7) of Gowal et al. (2018,
arXiv:1810.12715). We wrote the clipped midpoints by hand from `|sin x - sin m| ≤ |x - m|`, and the
same for cosine. They need no case split on the period, at a cost in width: on `[0, π]` the sine
rule gives about `[-0.571, 1]`, where the range is `[0, 1]`.

A linear layer's weight and bias are read from the constant nodes its references name
(`constAt?`), and at an empty input extent its box is the exact bias at each leading row. A matrix
product contracts over the layout of its own ports (`matmulLayout`: batch broadcasting and vector
operands as the operator states them), and a scaled product is the same contraction times its
frozen scale. A concatenation follows the layout of its ports (`catLayout`). A convolution reads
its kernel, in PyTorch's packed layout, and its bias from constant nodes. In PyTorch's layout its
box is the interval convolution on the dense layout of the kernel (`convOpBox?`). Every
other operator, and an operator node with a reference to an output other than `0`, has no box, and
neither has a node whose box would have a NaN endpoint.
`CertLocalOK` says that a table is the fixpoint of the certificate step.

The semantics the certificate is proved against is the graph's own evaluator: `evalNode?` is
`IR.Node.eval` at a node on the outputs of the nodes before it. `SemLocalOK` says that a table of
node outputs is the fixpoint of `evalNode?`, and `semLocalOK_iff_run` proves that the fixpoint is
exactly the outputs of a successful `IR.Graph.run`.
-/

@[expose] public section

namespace TorchLean.CROWN.Op

open Graph

section

variable {α : Type} [Storage α] [Context α]

/-- The interval step at node `id` under the operator rule `rule`: the seed box of an input, the
point box of a constant, and `rule` on the boxes of an operator node's references. An operator node
with a reference to an output other than `0` gets none, and so does a box with a NaN endpoint, which
the comparisons of a later rule would drop. -/
def step? (G : IR.Graph) (P : IR.Payload α)
    (rule : Array (Option (FlatBox α)) → IR.Operator → List IR.Ref → Option (FlatBox α))
    (seeds : Nat → Option (FlatBox α)) (boxes : Array (Option (FlatBox α))) (id : Nat) :
    Option (FlatBox α) :=
  Option.filter (fun B => B.lo.all (!Context.isNaN ·) && B.hi.all (!Context.isNaN ·)) <|
    match G.nodes[id]? with
    | some (.input _ _) => seeds id
    | some (.const _) =>
        (constValue? G P id >>= flatOf).map fun v => { dim := v.n, lo := v.v, hi := v.v }
    | some (.op o refs) =>
        if refs.all (·.out == 0) then rule boxes o refs.toList else none
    | none => none

variable [BoundOps α] [NonlinearBoundOps α]

/-- The box of a concatenation of scalars from the boxes its references name, and none for the
other operators. -/
def catBox? (boxes : Array (Option (FlatBox α))) :
    {ins outs : List Port} → Op ins outs → List IR.Ref → Option (FlatBox α)
  | _, _, .cat (s := s) dim sizes, refs => do
      let bs ← refs.mapM (box? boxes)
      concatFlatBoxes? (catLayout s dim sizes) bs.toArray
  -- The other elements are named, so that the match has equation lemmas.
  | _, _, .cat _ _ .bool, _ | _, _, .cat _ _ (.integer _ _), _ => none
  | _, _, _, _ => none

/-- The box of operator `o` on references `refs` by the rule that the engine (`ibp?`) and the
certificate (`cert?`) share, from the boxes the references name. Tanh, sigmoid, sine, cosine and
every other operator not of kind `shared` in `ruleKind` get none. -/
def shared? (G : IR.Graph) (P : IR.Payload α) (boxes : Array (Option (FlatBox α))) :
    IR.Operator → List IR.Ref → Option (FlatBox α)
  | ⟨_, _, .detach⟩, [r] => box? boxes r
  | ⟨_, _, .add⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some B₁, some B₂ => if _h : B₁.dim = B₂.dim then some (boxAdd B₁ B₂) else none
      | _, _ => none
  | ⟨_, _, .sub⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some B₁, some B₂ => if _h : B₁.dim = B₂.dim then some (boxSub B₁ B₂) else none
      | _, _ => none
  | ⟨_, _, .mul⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some B₁, some B₂ => boxMulElem B₁ B₂
      | _, _ => none
  | ⟨_, _, .relu⟩, [r] => (box? boxes r).map boxReLU
  | ⟨_, _, .silu⟩, [r] => box? boxes r >>= ibpSiLU?
  | ⟨_, _, .softplus⟩, [r] => box? boxes r >>= boxSoftplus?
  | ⟨_, _, .safeLog⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some B, some epsilon => boxSafeLog? B epsilon
      | _, _ => none
  | ⟨_, _, .linear (lead := L) (inDim := n) (outDim := m)⟩, [rW, rb, rx] => do
      let W ← constAt? G P rW (.value [m, n])
      let b ← constAt? G P rb (.value [m])
      let Xin ← box? boxes rx
      ibpLinearAt { inDim := n, outDim := m, spec := { weights := W, bias := b } }
        (L ++ [n]) Xin
  | ⟨_, _, .matmul (batchA := bA) (batchB := bB) (batch := b) (m := m) (n := n) (p := p) lv rv
      _ _ _ _⟩, [r₁, r₂] =>
      match box? boxes r₁, box? boxes r₂ with
      | some L, some R => some (binaryMatmulBox (matmulLayout lv rv bA bB b m n p) L R)
      | _, _ => none
  | ⟨_, _, .scaledMatmul (batchA := bA) (batchB := bB) (batch := b) (m := m) (n := n) (p := p) lv
      rv _ _ _ _⟩, [r₁, r₂, r₃] =>
      match box? boxes r₁, box? boxes r₂, box? boxes r₃ with
      | some L, some R, some S =>
          some (boxScale (binaryMatmulBox (matmulLayout lv rv bA bB b m n p) L R) S)
      | _, _, _ => none
  | ⟨_, _, .conv (k := k) dims stride padLow padHigh dilation groups (lead := L) _ _ _⟩,
      [rK, rb, rx] => do
      let K ← constAt? G P rK (.value k)
      let b ← constAt? G P rb (.value [k.extent (dims.kernel ⟨0, by omega⟩)])
      let X ← box? boxes rx
      convOpBox? dims stride padLow padHigh dilation groups L K b X
  -- The integer elements are named, so that the match has equation lemmas.
  | ⟨_, _, .add (.integer _ _)⟩, _ | ⟨_, _, .sub (.integer _ _)⟩, _
  | ⟨_, _, .mul (.integer _ _)⟩, _ => none
  | ⟨_, _, op⟩, refs => catBox? boxes op refs

end

/-- The kinds of interval rule an operator has (`ruleKind`). -/
inductive RuleKind where
  /-- The engine's rule and the certificate rule are both `shared?`. -/
  | shared
  /-- The engine takes the backend's enclosure, and the certificate rule endpoint min-max or
  clipped midpoints. -/
  | backend
  /-- Only the engine has a rule, from directed endpoint operations. -/
  | directed
  /-- Neither the engine nor the certificate rule gives a box. -/
  | none

/-- The kind of each operator's interval rule. The engine's rule `Op.ibp?` is the certificate
rule `cert?` at an operator of kind `shared` or `none` (`Op.ibp?_eq_cert?`), `cert?` is
`shared?` at every operator not of kind `backend` (`cert?_eq_shared?`), and `Op.ibp?` gives no
box at an operator of kind `none` (`Directed.supported_of_ibp?`). -/
def ruleKind : IR.Operator → RuleKind
  | ⟨_, _, .detach⟩ | ⟨_, _, .add⟩ | ⟨_, _, .sub⟩ | ⟨_, _, .mul⟩ | ⟨_, _, .relu⟩
  | ⟨_, _, .silu⟩ | ⟨_, _, .softplus⟩ | ⟨_, _, .safeLog⟩ | ⟨_, _, .linear⟩
  | ⟨_, _, .matmul ..⟩ | ⟨_, _, .scaledMatmul ..⟩ | ⟨_, _, .conv ..⟩ | ⟨_, _, .cat _ _⟩ => .shared
  | ⟨_, _, .tanh⟩ | ⟨_, _, .sigmoid⟩ | ⟨_, _, .sin⟩ | ⟨_, _, .cos⟩ => .backend
  | ⟨_, _, .reshape _⟩ | ⟨_, _, .exp⟩ | ⟨_, _, .log⟩ | ⟨_, _, .sqrt _⟩ | ⟨_, _, .reciprocal⟩
  | ⟨_, _, .neg⟩ | ⟨_, _, .square⟩ | ⟨_, _, .batchNormRunning _⟩ | ⟨_, _, .scale⟩
  | ⟨_, _, .leakyReLU⟩ | ⟨_, _, .clamp⟩ | ⟨_, _, .elu⟩ | ⟨_, _, .gelu _⟩ | ⟨_, _, .sinh⟩
  | ⟨_, _, .cosh⟩ | ⟨_, _, .logSoftmax _⟩ | ⟨_, _, .cumsum _⟩ | ⟨_, _, .select⟩
  | ⟨_, _, .pad _ _ _⟩ | ⟨_, _, .gather _ _ _⟩ | ⟨_, _, .embedding⟩ | ⟨_, _, .sign⟩
  | ⟨_, _, .floor⟩ | ⟨_, _, .ceil⟩ | ⟨_, _, .round⟩ | ⟨_, _, .trunc⟩ | ⟨_, _, .frac⟩ | ⟨_, _, .atan⟩
  | ⟨_, _, .asin⟩ | ⟨_, _, .acos⟩ | ⟨_, _, .erf⟩ | ⟨_, _, .expm1⟩ | ⟨_, _, .exp2⟩ | ⟨_, _, .asinh⟩
  | ⟨_, _, .tan⟩ | ⟨_, _, .log1p⟩ | ⟨_, _, .log2⟩ | ⟨_, _, .log10⟩ | ⟨_, _, .rsqrt⟩
  | ⟨_, _, .fmax⟩ | ⟨_, _, .fmin⟩ | ⟨_, _, .atan2⟩ | ⟨_, _, .hypot⟩ | ⟨_, _, .contract _ _ _⟩
  | ⟨_, _, .fmod⟩ | ⟨_, _, .remainder⟩ | ⟨_, _, .copysign⟩ | ⟨_, _, .pool _ _ _ _ _ _ _ _⟩
  | ⟨_, _, .abs⟩ | ⟨_, _, .max⟩ | ⟨_, _, .min⟩ | ⟨_, _, .softmax _⟩ | ⟨_, _, .randUniform _⟩
  | ⟨_, _, .bernoulliMask _⟩ | ⟨_, _, .permute _ _ _⟩ | ⟨_, _, .slice _ _ _ _ _⟩ | ⟨_, _, .flip _⟩
  | ⟨_, _, .broadcast _ _⟩ | ⟨_, _, .indexAt _ _⟩ | ⟨_, _, .mseLoss⟩ | ⟨_, _, .maskedSoftmax _⟩
  | ⟨_, _, .layerNorm _ _ _ _⟩ => .directed
  | ⟨_, _, .div rounding⟩ => if rounding = .none ∨ rounding = .trunc then .directed else .none
  | ⟨_, _, .reduce (s := s) op dims keepdim _⟩ =>
      if (op = .sum ∧ keepdim = false ∧ (dims.map Fin.val = List.range s.rank ∨ dims.length = 1)) ∨
          op = .max ∨ op = .min then .directed else .none
  | ⟨_, _, .mean dims keepdim⟩ => if keepdim = false ∧ dims.length = 1 then .directed else .none
  | _ => .none

noncomputable section

/-- The certificate box of operator `o` on references `refs`, from the boxes they name: endpoint
min-max for tanh and sigmoid, clipped midpoints for sine and cosine, and `shared?` at `ℝ` for
every other operator. -/
def cert? (G : IR.Graph) (P : IR.Payload ℝ) (cert : Array (Option (FlatBox ℝ))) :
    IR.Operator → List IR.Ref → Option (FlatBox ℝ)
  | ⟨_, _, .tanh⟩, [r] => (box? cert r).map fun B =>
      toFlatBox (α := ℝ) B.dim (IBP.tanhMinmax (α := ℝ) (ofFlatBox (α := ℝ) B))
  | ⟨_, _, .sigmoid⟩, [r] => (box? cert r).map fun B =>
      toFlatBox (α := ℝ) B.dim (IBP.sigmoidMinmax (α := ℝ) (ofFlatBox (α := ℝ) B))
  | ⟨_, _, .sin⟩, [r] => (box? cert r).map fun B =>
      toFlatBox (α := ℝ) B.dim (IBP.sinMidpoint (α := ℝ) (ofFlatBox (α := ℝ) B))
  | ⟨_, _, .cos⟩, [r] => (box? cert r).map fun B =>
      toFlatBox (α := ℝ) B.dim (IBP.cosMidpoint (α := ℝ) (ofFlatBox (α := ℝ) B))
  | o, refs => shared? (α := ℝ) G P cert o refs

/-- The operators of kind `backend` in `ruleKind`, tanh, sigmoid, sine and cosine: the engine's rule
is the backend's enclosure while the certificate rule keeps endpoint min-max and clipped
midpoints. -/
def BackendEnclosure (o : IR.Operator) : Prop :=
  ruleKind o = .backend

/-- The certificate rule `cert?` for an operator other than tanh, sigmoid, sine and cosine is
`shared?` at ℝ. -/
theorem cert?_eq_shared? {G : IR.Graph} {P : IR.Payload ℝ}
    {c : Array (Option (FlatBox ℝ))} {o : IR.Operator} {refs : List IR.Ref}
    (ho : ¬ BackendEnclosure o) : cert? G P c o refs = shared? (α := ℝ) G P c o refs := by
  unfold cert?
  split
  case h_1 | h_2 | h_3 | h_4 => exact (ho rfl).elim
  case h_5 => rfl

/-- A certificate has one entry per node and is the step's fixpoint at every node. -/
def CertLocalOK (G : IR.Graph) (P : IR.Payload ℝ) (seeds : Nat → Option (FlatBox ℝ))
    (cert : Array (Option (FlatBox ℝ))) : Prop :=
  cert.size = G.nodes.size ∧ ∀ id, id < G.nodes.size →
    cert[id]! = step? G P (cert? G P) seeds cert id

end

section

variable {α : Type} [Storage α] [Context α] [Op.Capabilities α]

/-- The outputs of node `id` by `IR.Node.eval` on the first `id` entries of `values`, or none
where the graph has no node `id` or its evaluation fails. -/
def evalNode? (G : IR.Graph) (P : IR.Payload α) (inputs : List (Op.Value α))
    (values : Array (List (Op.Value α))) (id : Nat) :
    Option (List (Op.Value α)) :=
  G.nodes[id]? >>= fun n => (n.eval P inputs (values.extract 0 id) id).toOption

/-- A table of node outputs has one entry per node and is the fixpoint of `evalNode?` at every
node. -/
def SemLocalOK (G : IR.Graph) (P : IR.Payload α) (inputs : List (Op.Value α))
    (values : Array (List (Op.Value α))) : Prop :=
  values.size = G.nodes.size ∧
    ∀ id, id < G.nodes.size → values[id]? = evalNode? G P inputs values id

/-- A table of node outputs is the fixpoint of `evalNode?` exactly when it is the outputs of a
successful run. -/
theorem semLocalOK_iff_run {G : IR.Graph} {P : IR.Payload α}
    {inputs : List (Op.Value α)} {values : Array (List (Op.Value α))} :
    SemLocalOK G P inputs values ↔ G.run P inputs = .ok values := by
  constructor
  · rintro ⟨hsize, hnode⟩
    have hprefix : ∀ k, k ≤ G.nodes.size →
        IR.runNodes P inputs #[] (G.nodes.toList.take k) = .ok (values.extract 0 k) := by
      intro k
      induction k with
      | zero => intro _; simp [IR.runNodes_nil]
      | succ k ih =>
        intro hk
        have hlt : k < G.nodes.size := hk
        have hvk : k < values.size := hsize ▸ hlt
        have heval : G.nodes[k].eval P inputs (values.extract 0 k) (values.extract 0 k).size =
            .ok values[k] := by
          have hv := hnode k hlt
          simp only [evalNode?, Array.getElem?_eq_getElem hlt, Array.getElem?_eq_getElem hvk,
            Option.bind_eq_bind, Option.bind_some] at hv
          rw [Array.size_extract, Nat.min_eq_left (Nat.le_of_lt hvk), Nat.sub_zero]
          cases he : G.nodes[k].eval P inputs (values.extract 0 k) k with
          | error e => simp [he, Except.toOption] at hv
          | ok x =>
            simp only [he, Except.toOption, Option.some.injEq] at hv
            rw [hv]
        rw [List.take_add_one, IR.runNodes_append, ih (Nat.le_of_lt hlt), Array.getElem?_toList,
          Array.getElem?_eq_getElem hlt, Option.toList_some]
        simp only [bind, Except.bind]
        rw [IR.runNodes_singleton heval, Array.append_singleton,
          Array.extract_succ_right (Nat.succ_pos k) hvk]
    have h := hprefix G.nodes.size (Nat.le_refl _)
    rwa [List.take_of_length_le (Nat.le_of_eq Array.length_toList), ← hsize,
      Array.extract_size] at h
  · intro hrun
    have hsize := IR.Graph.run_size hrun
    refine ⟨hsize, fun id hid => ?_⟩
    obtain ⟨n, hn⟩ : ∃ n, G.nodes[id]? = some n := ⟨_, Array.getElem?_eq_getElem hid⟩
    obtain ⟨pre, outs, hpre, hagree, heval, hout⟩ := IR.Graph.run_step hrun hn
    have hext : pre = values.extract 0 id := Array.ext_getElem? fun k => by
      rw [Array.getElem?_extract, Nat.min_eq_left (by omega : id ≤ values.size), Nat.sub_zero]
      by_cases hk : k < id
      · rw [ite_eq_left hk, Nat.zero_add, hagree k hk]
      · rw [ite_eq_right hk]
        exact Array.getElem?_eq_none (by omega)
    simp only [evalNode?, hn, Option.bind_eq_bind, Option.bind_some, ← hext, heval, hout]
    rfl

end

end TorchLean.CROWN.Op
