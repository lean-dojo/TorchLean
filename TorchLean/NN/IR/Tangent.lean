/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Builder

/-!
# Forward-mode tangent graphs

`Graph.tangent G P seed` appends to the graph `G` of registered operators, with payload `P`, the
nodes that compute the tangent of each node along the given seeds, and returns the extended graph,
its payload and the reference of each node's tangent. The tangent of node `j` is the reference
`seed j` when there is one. Otherwise it is zero at a graph input and at a constant, and at an
operator node it is the rule of the table below on the node's inputs and their tangents, written
with registered operators. These rules are separate from the op table's forward-mode rules
(`Op.Rule.jvp?`), and no theorem relates a tangent graph to those rules or to a derivative. A
tangent that is zero is not built: its entry is `none`. The graph's outputs are `G`'s outputs
followed by their tangents (a zero tangent as a constant of zeros).

The rules, `x` the input, `y` the node's value and `dx` the input's tangent:

| operator | tangent |
| --- | --- |
| `add`, `sub`, `neg` | the same operator on the tangents |
| `mul` | `dx₁ * x₂ + x₁ * dx₂` |
| `div` (true division) | `dx₁ / x₂ - dx₂ * (y / x₂)` |
| `square` | `(2 * x) * dx` |
| `reciprocal` | `(-dx / x) / x` |
| `exp`, `log` | `y * dx`, `dx / x` |
| `sin`, `cos` | `cos x * dx`, `-sin x * dx` |
| `tanh` | `(1 - y ^ 2) * dx` |
| `sigmoid` | `(1/4 - (y - 1/2) ^ 2) * dx` |
| `silu` | `(sigmoid x + x * (1/4 - (sigmoid x - 1/2) ^ 2)) * dx`, `dx` at `+∞`, `-0 * dx` at `-∞` |
| `relu` | `select (x > 0) dx 0` |
| `select` | `select c dx₁ dx₂` |
| `linear` (vectors) | `W dx` with a zero bias, plus `dW x + db` where the parameters vary |
| `matmul` | `dA B + A dB` |
| `reshape`, `permute`, `broadcast`, `slice`, `flip` | the same operator on the tangent |
| `cumsum`, `reduce` (sums), `mean`, `scale`, `cat` | the same operator on the tangents |
| `detach` | zero |

The squares in the `tanh`, `sigmoid` and `silu` rules are `square` nodes, so interval propagation
bounds `1 - y ^ 2` by `[0, 1]` for `y` in `[-1, 1]` and `1/4 - (y - 1/2) ^ 2` by `[0, 1/4]` for `y`
in `[0, 1]`, the ranges of the derivatives, up to the scalar's outward rounding. A product `y * y`
of two copies of one box would give the wider `[0, 2]` and `[0, 1/2]`. A rule reads only its
node's inputs, its value and its inputs' tangents. An input at a port that is not a value port
(frozen, detached, Boolean, index, integer) carries no tangent, and a node whose first output is
not a value port has none. A tangent that reaches an operator without a rule is an error naming
the operator, and reading any output but the first of a node that has a tangent is an error too.
-/

@[expose] public section

namespace TorchLean.IR

namespace Tangent

variable {α : Type} [Storage α] [Context α]

/-- The extended graph so far: its builder, its payload, and the tangent of each node of the
source graph so far (`none`: zero). -/
structure State (α : Type) [Storage α] where
  /-- The nodes so far. -/
  builder : Graph.Builder
  /-- The constants so far. -/
  payload : Payload α
  /-- The tangent of each source node so far. -/
  tangents : Array (Option Ref) := #[]

/-- Append a constant filled with `x` at value port `s`. -/
def State.fill (st : State α) (s : Shape) (x : α) : State α × Ref :=
  let (b, P, r) := st.builder.constant st.payload ⟨.value s, Tensor.full s x⟩
  ({ st with builder := b, payload := P }, r)

/-- Append operator `o` on `refs` and return its first output. -/
def State.op {ins outs : List Port} (st : State α) (o : Op ins outs)
    (refs : List Ref) : State α × Ref :=
  let (b, ys) := st.builder.op o refs
  ({ st with builder := b }, ys.headD ⟨b.nodes.size, 0⟩)

/-- The sum of two tangents, either of which may be zero. -/
def State.add (st : State α) (s : Shape) : Option Ref → Option Ref → State α × Option Ref
  | some a, some b => let (st, r) := st.op (Op.add (s := s)) [a, b]; (st, some r)
  | some a, none | none, some a => (st, some a)
  | none, none => (st, none)

/-- The tangent of the elementwise map with derivative `d` (a reference) along `dx`. -/
def State.scaleBy (st : State α) (s : Shape) (d dx : Ref) : State α × Option Ref :=
  let (st, r) := st.op (Op.mul (s := s)) [d, dx]
  (st, some r)

/-- `1/4 - (v - 1/2) ^ 2`, the derivative of the logistic sigmoid where its value is `v`. -/
def State.sigmoidSlope (st : State α) (s : Shape) (v : Ref) : State α × Ref :=
  let (st, half) := st.fill s (1 / 2)
  let (st, centered) := st.op (Op.sub (s := s)) [v, half]
  let (st, sq) := st.op (Op.square (s := s)) [centered]
  let (st, quarter) := st.fill s (1 / 4)
  st.op (Op.sub (s := s)) [quarter, sq]

/-- Append operator `o`, its ports forgotten, on `refs` and return its first output. -/
def State.apply (st : State α) (o : Operator) (refs : List Ref) : State α × Ref :=
  let (b, ys) := st.builder.add (.op o refs.toArray)
  ({ st with builder := b }, ys.headD ⟨b.nodes.size, 0⟩)

/-- A tangent, or a constant of zeros at value port `s` for a zero tangent. -/
def State.orZero (st : State α) (s : Shape) : Option Ref → State α × Ref
  | some d => (st, d)
  | none => st.fill s 0

/-- Appends to `st` the tangent of operator `o` at a node with input references `xs`, value `y`
and input tangents `dxs` (zero at an input that is not a value port). The tangent is zero when
every input tangent is zero or the node's first output is not a value port, and otherwise follows
the rule in the module doc. An operator without a rule is an error. -/
def rule (st : State α) (y : Ref) (o : Operator) (xs : List Ref) (dxs : List (Option Ref)) :
    Except String (State α × Option Ref) :=
  if dxs.all Option.isNone || (o.2.1.head?.map (·.kind)) != some .value then pure (st, none)
  else
    let some1 (p : State α × Ref) : State α × Option Ref := (p.1, some p.2)
    match o, xs, dxs with
    | ⟨_, _, .add (s := s)⟩, _, [d₁, d₂] => pure (st.add s d₁ d₂)
    | ⟨_, _, .sub (s := s)⟩, _, [d₁, d₂] => pure <| match d₁, d₂ with
        | some a, some b => some1 (st.op (Op.sub (s := s)) [a, b])
        | some a, none => (st, some a)
        | none, some b => some1 (st.op (Op.neg (s := s)) [b])
        | none, none => (st, none)
    | ⟨_, _, .neg (s := s)⟩, _, [some d] => pure (some1 (st.op (Op.neg (s := s)) [d]))
    | ⟨_, _, .mul (s := s)⟩, [x₁, x₂], [d₁, d₂] =>
        let (st, t₁) := match d₁ with
          | some a => some1 (st.op (Op.mul (s := s)) [a, x₂])
          | none => (st, none)
        let (st, t₂) := match d₂ with
          | some b => some1 (st.op (Op.mul (s := s)) [x₁, b])
          | none => (st, none)
        pure (st.add s t₁ t₂)
    | ⟨_, _, .div (s := s) .none⟩, [_, x₂], [d₁, d₂] =>
        let (st, q₁) := match d₁ with
          | some a => some1 (st.op (Op.div (s := s)) [a, x₂])
          | none => (st, none)
        let (st, q₂) := match d₂ with
          | some b =>
            let (st, w) := st.op (Op.div (s := s)) [y, x₂]
            some1 (st.op (Op.mul (s := s)) [b, w])
          | none => (st, none)
        pure <| match q₁, q₂ with
          | some a, some b => some1 (st.op (Op.sub (s := s)) [a, b])
          | some a, none => (st, some a)
          | none, some b => some1 (st.op (Op.neg (s := s)) [b])
          | none, none => (st, none)
    | ⟨_, _, .square (s := s)⟩, [x], [some d] =>
        let (st, two) := st.fill s 2
        let (st, w) := st.op (Op.mul (s := s)) [two, x]
        pure (st.scaleBy s w d)
    | ⟨_, _, .reciprocal (s := s)⟩, [x], [some d] =>
        let (st, n) := st.op (Op.neg (s := s)) [d]
        let (st, q) := st.op (Op.div (s := s)) [n, x]
        pure (some1 (st.op (Op.div (s := s)) [q, x]))
    | ⟨_, _, .exp (s := s)⟩, _, [some d] => pure (st.scaleBy s y d)
    | ⟨_, _, .log (s := s)⟩, [x], [some d] =>
        pure (some1 (st.op (Op.div (s := s)) [d, x]))
    | ⟨_, _, .sin (s := s)⟩, [x], [some d] =>
        let (st, c) := st.op (Op.cos (s := s)) [x]
        pure (st.scaleBy s c d)
    | ⟨_, _, .cos (s := s)⟩, [x], [some d] =>
        let (st, v) := st.op (Op.sin (s := s)) [x]
        let (st, n) := st.op (Op.neg (s := s)) [v]
        pure (st.scaleBy s n d)
    | ⟨_, _, .tanh (s := s)⟩, _, [some d] =>
        let (st, sq) := st.op (Op.square (s := s)) [y]
        let (st, one) := st.fill s 1
        let (st, slope) := st.op (Op.sub (s := s)) [one, sq]
        pure (st.scaleBy s slope d)
    | ⟨_, _, .sigmoid (s := s)⟩, _, [some d] =>
        let (st, slope) := st.sigmoidSlope s y
        pure (st.scaleBy s slope d)
    | ⟨_, _, .silu (s := s)⟩, [x], [some d] =>
        let (st, g) := st.op (Op.sigmoid (s := s)) [x]
        let (st, slope) := st.sigmoidSlope s g
        let (st, t) := st.op (Op.mul (s := s)) [x, slope]
        let (st, slope) := st.op (Op.add (s := s)) [g, t]
        -- `Activation.siluDeriv`'s limits at `±∞`, where `x * slope` is a NaN.
        let (st, inf) := st.op (Op.isInf (s := s)) [x]
        let (st, zeros) := st.fill s 0
        let (st, positive) := st.op (Op.gt (s := s)) [x, zeros]
        let (st, one) := st.fill s 1
        let (st, negZero) := st.fill s (-0)
        let (st, limit) := st.op (Op.select (s := s)) [positive, one, negZero]
        let (st, slope) := st.op (Op.select (s := s)) [inf, limit, slope]
        pure (st.scaleBy s slope d)
    | ⟨_, _, .relu (s := s)⟩, [x], [some d] =>
        let (st, zeros) := st.fill s 0
        let (st, c) := st.op (Op.gt (s := s)) [x, zeros]
        pure (some1 (st.op (Op.select (s := s)) [c, d, zeros]))
    | ⟨_, _, .select (s := s)⟩, [c, _, _], [_, d₁, d₂] =>
        let (st, a) := st.orZero s d₁
        let (st, b) := st.orZero s d₂
        pure (some1 (st.op (Op.select (s := s)) [c, a, b]))
    | ⟨_, _, @Op.linear .scalar n m⟩, [w, _, x], [dw, dc, dx] =>
        let (st, t₁) := match dw, dc with
          | none, none => (st, none)
          | _, _ =>
            let (st, a) := st.orZero [m, n] dw
            let (st, b) := st.orZero [m] dc
            some1 (st.op (Op.linear (lead := []) (inDim := n) (outDim := m)) [a, b, x])
        let (st, t₂) := match dx with
          | some d =>
            let (st, zero) := st.fill [m] 0
            some1 (st.op (Op.linear (lead := []) (inDim := n) (outDim := m)) [w, zero, d])
          | none => (st, none)
        pure (st.add [m] t₁ t₂)
    | ⟨_, _, .matmul ..⟩, [a, b], [da, db] =>
        let out := (o.2.1.head?.map (·.shape)).getD .scalar
        let (st, t₁) := match da with
          | some d => some1 (st.apply o [d, b])
          | none => (st, none)
        let (st, t₂) := match db with
          | some d => some1 (st.apply o [a, d])
          | none => (st, none)
        pure (st.add out t₁ t₂)
    | ⟨_, _, .reduce .sum _ _ _⟩, _, [some d] | ⟨_, _, .mean ..⟩, _, [some d]
    | ⟨_, _, .reshape _⟩, _, [some d] | ⟨_, _, .permute ..⟩, _, [some d]
    | ⟨_, _, .broadcast ..⟩, _, [some d] | ⟨_, _, .slice ..⟩, _, [some d]
    | ⟨_, _, .flip _⟩, _, [some d] | ⟨_, _, .cumsum _⟩, _, [some d] =>
        pure (some1 (st.apply o [d]))
    | ⟨_, _, .scale⟩, [_, k], [some d, _] => pure (some1 (st.apply o [d, k]))
    | ⟨_, _, .cat ..⟩, _, ds =>
        let (st, refs) := (o.1.zip ds).foldl (fun (st, acc) (p, d) =>
          let (st, r) := st.orZero p.shape d
          (st, acc ++ [r])) (st, [])
        pure (some1 (st.apply o refs))
    | _, _, _ => throw s!"tangent: no forward-mode rule for {(Op.encode o).tag}"

/-- Append the tangent of node `j` of `G` to `st`: `seed j`, or zero at an input or a constant,
or the rule of its operator on the tangents of its inputs. -/
def step (G : Graph) (seed : Nat → Option Ref) (st : State α) (j : Nat) :
    Except String (State α) := do
  let (st, t) ← match seed j, G.nodes[j]? with
    | some r, _ => pure (st, some r)
    | none, some (.op o refs) => do
        let dxs ← (o.1.zip refs.toList).mapM fun (p, r) => do
          let d := if p.kind = .value then (st.tangents[r.node]?).join else none
          if d.isSome && r.out ≠ 0 then
            throw s!"tangent: node {j} reads output {r.out} of node {r.node}, which has a tangent"
          pure d
        rule st ⟨j, 0⟩ o refs.toList dxs
    | none, _ => pure (st, none)
  pure { st with tangents := st.tangents.push t }

end Tangent

/-- The tangent graph of `G` with payload `P` along `seed`: `G`'s nodes, then the nodes of each
node's tangent (`Tangent.step`), with `G`'s outputs followed by their tangents (a zero tangent as
a constant of zeros). It also returns the extended payload and the tangent of each node of `G`
(`none`: zero). It fails where `Tangent.step` does and at a graph output that is not a node's
first output at a value port. -/
def Graph.tangent {α : Type} [Storage α] [Context α] (G : Graph) (P : Payload α)
    (seed : Nat → Option Ref) : Except String (Graph × Payload α × Array (Option Ref)) := do
  let st ← (List.range G.nodes.size).foldlM (Tangent.step G seed)
    ({ builder := ⟨G.nodes⟩, payload := P } : Tangent.State α)
  let (st, outs) ← G.outputs.foldlM (init := (st, (#[] : Array Ref))) fun (st, acc) r => do
    let some p := (G.nodes[r.node]?).bind (·.outs[r.out]?)
      | throw s!"tangent: no output {r.out} of node {r.node}"
    unless r.out = 0 ∧ p.kind = .value do
      throw s!"tangent: output {r.out} of node {r.node} is not a value output with a tangent"
    let (st, t) := st.orZero p.shape (st.tangents[r.node]?).join
    pure (st, acc.push t)
  let g ← st.builder.build (G.outputs ++ outs)
  pure (g, st.payload, st.tangents)

end TorchLean.IR
