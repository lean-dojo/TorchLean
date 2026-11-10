/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.Exec
public import TorchLean.Verify.CROWN.FlatBox.Backward

/-!
# CROWN on operator graphs

Two affine bound passes on a graph of registered operators (`TorchLean.IR.Graph`), each reading an
interval table `ibp` (for instance `runIBP`'s) and a designated input node (`AffineCtx`).

The backward pass bounds a linear objective of one node's value by affine forms in the input. A
reverse sweep (`sweep`) carries an interval coefficient per node and a two-sided constant, both
rounded outward through `BoundOps`, from the objective's node to the input. A lower sweep and an
upper sweep run separately (`backwardObjective?`), and the lower form comes from the first and the
upper from the second. Each gives way to the form of a sweep with the shared rules alone, which
discharges every product at its box, when that form is tighter on the input's box or its own sweep
fails.

The rules shared by both directions are `directedRule`. Detach and reshape pass the coefficient
unchanged. Sums, differences, a sum over every axis (in order, without keepdim), a concatenation, a
unit-step slice along one axis and a permutation pass it to the references. A linear map with
constant weight and bias passes the transposed coefficient to its input. It needs the coefficient to
have the width of one output row and the transposed coefficient the input's width, and ends the
sweep otherwise. A convolution with constant kernel and bias passes the transposed convolution of
the coefficient over the same configuration (`convBack?`). Gather, scatter, pad, flip,
contract, log-softmax, cumulative and sorting operators and custom operators end the sweep, and
constants and every other operator discharge the coefficient against their box.

`backwardRule` adds the products (`mul`, `matmul`, `scaledMatmul`). A product term `x·y` with a
point operand passes the scaled coefficient to the other operand. Otherwise a term whose coefficient
interval has one sign is replaced by a McCormick plane of the operand boxes, the lower plane
`ly·x + lx·y - lx·ly` or the upper plane `uy·x + lx·y - lx·uy` by direction and sign. A term whose
coefficient interval contains both signs is discharged against the product of the operand boxes.

`affineBounds?` is the backward pass's one query: a matrix objective at a node gives lower and upper
affine forms in the input, one pair per row, and the identity objective gives the node's own bounds
(`nodeBounds?`). `crownBoxes` uses it at every node. Node by node, the interval rule on the boxes
before the node is met with a supplied enclosing table and with the node's affine bounds evaluated
on the input box. `runCROWN` reads every node's affine bounds over those boxes, taking per row the
tighter on the input box of the query's forms and the constant forms of the node's box. Every box a
wrapper returns is met with the interval box of its node (`meetWith`), or for an objective with the
objective on that box, so no rule gives a wider one. Without that interval box the wrapper is an
error.

There is no check of the whole graph: the import refuses ill-formed graphs, and on its graphs each
node refuses for itself. A node whose operator has no interval rule has no box, nor has any node
that reads it, and a box wrapper at such a node is an error. A sweep fails when it reaches an
operator that ends it, a missing box or a missing constant. When both sweeps of a direction fail at
a row, `affineBounds?` reports none and the box wrappers keep the interval box of the queried node.

The forward pass (`alphaStep?`, `runAlphaCROWN`) gives a node affine lower and upper forms from
those of its references. It has rules for the input (the identity), constants, detach, a linear map
with constant weight and bias, a sum over every axis (in order, without keepdim), reshape and ReLU.
ReLU uses α slopes and phases, with the α-CROWN lower relaxation and the triangle upper relaxation
or the relaxations the phases fix. Another input, a matrix product and a custom operator have no
forms, and every other operator gets the constant forms of its box.

We wrote both passes by hand from Xu et al. (2020), Sec. 3.1. The backward pass is their backward
mode. A sweep starts with one objective row at its node, nothing elsewhere and the constant zero,
which is their Eq. (4) for one row, and each rule is an oracle `F_i` of their Eq. (5): it trades a
node's coefficient for coefficients of its references and a constant. Their Algorithm 2 orders the
nodes by a BFS that counts out-degrees. We sweep in decreasing index order. When references name
earlier nodes, every reader of a node has a larger index and is swept first, so the node's
coefficient is complete when the sweep reaches it.

The backward rules relax no activation. ReLU is among the operators that discharge the coefficient
against their box, so no coefficient passes a ReLU, and the ReLU lines of CROWN (Zhang et al. 2018,
Sec. 3.2) appear in the forward pass alone.

`outputBoxCROWN?` is the IBP+Backward hybrid of Xu et al.: `runIBP` gives every intermediate box
and one query runs at the output. `crownBoxes` runs a query at every operator node, the variant
they call very expensive, and each box it finds feeds the queries after it. A query costs three
sweeps per objective row (the shared rules, then the product rules in each direction), so the
identity objective at a node of width 10 takes 30.

The product planes are the bounds Shi et al. (2020, Sec. 3.3) give for `z = x·y`, two of the four
planes of the McCormick (1976) envelope. On each side the two McCormick planes leave the same
integral of the gap over the box (Shi et al., App. C), and like Shi et al. we take the one that
meets `x·y` along the edge `x = lx`.

The forward pass is the forward mode of Xu et al. (Algorithm 1, Eq. (2)) over the nodes in index
order, with the ReLU lines of α-CROWN (Xu et al. 2021, Eq. (5)). A ReLU reads `l` and `u` from
`ibp` and never evaluates the forms of its input to get them. α-CROWN optimizes the slopes in
`[0, 1]` by projected gradient descent (Eq. (9)), and we don't. A certificate supplies them
(`TorchLean.Verify.Certificate.CROWNNode` replays `alphaStep?` with its slopes), and
`runAlphaCROWN` supplies none, so every ReLU there gets `Cert.defaultAlphaVec`: CROWN's adaptive
slope (Sec. 3.2), 1 when `u > -l` and 0 otherwise. At the tie `u = -l` CROWN takes 1 and we take 0,
and either line leaves the area `u ^ 2 / 2` between itself and the ReLU.

No theorem here relates these forms to a graph's values. At a lawful scalar the boxes and forms of
the backward queries enclose the values of a successful real run (`outputBoxCROWN?_of_run`,
`crownBoxes_encloses` and `runCROWN_encloses` in `Op.BackwardRun`), and at `ℝ` the forward forms
do (`runAlphaCROWN_encloses` in `Op.Alpha`).

References:
- Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond",
  NeurIPS 2020, arXiv:2002.12920: Sec. 3.1, Eq. (2) to (5), Algorithms 1 and 2.
- Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions", NeurIPS 2018, arXiv:1811.00866: Sec. 3.2.
- Xu et al., "Fast and Complete: Enabling Complete Neural Network Verification with Rapid and
  Massively Parallel Incomplete Verifiers", ICLR 2021, arXiv:2011.13824: Eq. (5) and (9).
- Shi et al., "Robustness Verification for Transformers", ICLR 2020, arXiv:2002.06622: Sec. 3.3
  and App. C.
- McCormick, "Computability of global solutions to factorable nonconvex programs: Part I. Convex
  underestimating problems", Mathematical Programming 10(1), 147-175, 1976,
  doi:10.1007/BF01580665.
-/

@[expose] public section

namespace TorchLean.CROWN.Op

open Tensor
open Graph CROWN.Directed

variable {α : Type} [Storage α] [Context α] [BoundOps α]

/-- The id of the node of graph input `k`. -/
def inputId? (G : IR.Graph) (k : Nat) : Option Nat :=
  (List.range G.nodes.size).find? fun id =>
    match G.nodes[id]? with
    | some (.input j _) => j == k
    | _ => false

/-- The number of entries of output `0` of node `id`. -/
def outSize? (G : IR.Graph) (id : Nat) : Option Nat :=
  match G.nodes[id]? with
  | some (.input _ p) => some p.shape.size
  | some (.const p) => some p.shape.size
  | some (.op ⟨_, outs, _⟩ _) => outs.head?.map (·.shape.size)
  | none => none

/-- The meet of boxes `A` and `B` of one size, endpoint by endpoint: the larger lower and the
smaller upper endpoint, with `B`'s where the comparison fails. Boxes of different sizes give `B`. -/
def meetBox (A B : FlatBox α) : FlatBox α :=
  if A.dim = B.dim then
    { dim := B.dim
      lo := Tensor.ofFn fun i =>
        let a := getAtOrZero A.lo [i.val]
        let b := B.lo.getScalar i
        if decide (b < a) then a else b
      hi := Tensor.ofFn fun i =>
        let a := getAtOrZero A.hi [i.val]
        let b := B.hi.getScalar i
        if decide (a < b) then a else b }
  else B

/-- `B` met with the box of node `id` in `ibp`, when it has one. -/
def meetAt (ibp : Array (Option (FlatBox α))) (id : Nat) (B : FlatBox α) : FlatBox α :=
  match (ibp[id]?).join with
  | some I => meetBox B I
  | none => B

/-- `B` met with the box of node `id` in `ibp`. It is an error when there is no box there or that
box has another size. -/
def meetWith (ibp : Array (Option (FlatBox α))) (id : Nat) (B : FlatBox α) :
    Except String (FlatBox α) :=
  match (ibp[id]?).join with
  | some I =>
      if I.dim = B.dim then pure (meetBox B I)
      else throw s!"interval box of node {id} has {I.dim} entries, the CROWN box {B.dim}"
  | none => throw s!"no interval box at node {id}"

/-! ## Backward pass -/

/-- Discharge the coefficient `aY` of node `id` against its box in `ibp`, failing when the node has
no box. -/
def consumeAt (ibp : Array (Option (FlatBox α))) (st : State α) (id : Nat)
    (aY : FlatBox α) : State α :=
  match (ibp[id]?).join with
  | some B => consume st aY B
  | none => st.fail

/-- The coefficient pulled back through a permutation `σ` of the axes of `s` into `out`: the
inverse permutation `Shape.unpermute` (the operator's pullback) applied to both endpoints. -/
def permuteBack [Op.Capabilities α] {s out : Shape}
    (σ : Equiv.Perm (Fin s.rank)) (hout : out = s.permute σ) (aY : FlatBox α) :
    Option (FlatBox α) :=
  Op.endpointsBox?
    (Op.permute (s := out) (s.unpermute σ hout) s (s.permute_unpermute σ hout).symm) aY

/-- Coefficients collected per entry of an operand, `none` where no term reached it. -/
abbrev PartialCoeffs (α : Type) := Array (Option (α × α))

/-- Add the interval `v` to entry `i`, rounding outward. -/
def accum (a : Op.PartialCoeffs α) (i : Nat) (v : α × α) : Op.PartialCoeffs α :=
  match a[i]? with
  | some (some w) => a.set! i (some (BoundOps.addDown w.1 v.1, BoundOps.addUp w.2 v.2))
  | _ => a.set! i (some v)

/-- The negation of the product `e·f`, as an outward interval. -/
def negProduct (e f : α) : α × α :=
  (BoundOps.subDown 0 (BoundOps.mulUp e f), BoundOps.subUp 0 (BoundOps.mulDown e f))

/-- An operand's collected coefficients as a box of `n` entries (zero where no term reached it), or
none when no term reached it. -/
def coeffBox (n : Nat) (a : Op.PartialCoeffs α) : Option (FlatBox α) :=
  if a.any Option.isSome then
    some { dim := n, lo := Tensor.ofFn fun i => ((a[i.val]?).join.map (·.1)).getD 0,
           hi := Tensor.ofFn fun i => ((a[i.val]?).join.map (·.2)).getD 0 }
  else none

/-- The coefficients of a convolution's input and its constant from the coefficient `aY` of its
output, in the layout `convOpBox?` reads (kernel `(outC, inC / groups, kernel...)`, input
`(lead, inC, spatial...)`): the transposed convolution of `aY` with the kernel `K` over the same
stride, padding and dilation, and the bias term `Σ aY · b` in output order, each product and sum
rounded outward (no input coefficient when no term reaches the input). None for another layout or
when the input box `X` or `aY` has another size. -/
def convBack? {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (lead : Shape) (K : Tensor α k)
    (b : Tensor α [k.extent (dims.kernel ⟨0, by omega⟩)]) (X aY : FlatBox α) :
    Option (Option (FlatBox α) × (α × α)) :=
  match s, k, dims, K, b with
  | _ :: C :: sp, O :: I :: ks, dims, K, b =>
      let outSp := (List.finRange r).map fun j =>
        Shape.slidingWindowOutDimDilated (sp.getD j.val 0) (ks.getD j.val 0) (stride j)
          (dilation j) (padLow j) (padHigh j)
      let size (l : List Nat) : Nat := l.foldl (· * ·) 1
      if ChannelsFirst dims ∧ I = C / groups ∧ ks.length = r ∧ sp.length = r ∧ groups ∣ O ∧
          X.dim = lead.size * (C * size sp) ∧ aY.dim = lead.size * (O * size outSp) then
        let flat (extents idx : List Nat) : Nat :=
          (extents.zip idx).foldl (fun acc (e, i) => acc * e + i) 0
        let strides := (List.finRange r).map stride
        let dilations := (List.finRange r).map dilation
        let pads := (List.finRange r).map padLow
        let init : Op.PartialCoeffs α × (α × α) := (Array.replicate X.dim none, (0, 0))
        let (cx, c) := (List.range lead.size).foldl (fun acc n =>
          (List.range O).foldl (fun acc oc =>
            Conv.foldlIndices outSp acc fun (cx, c) outIdx =>
              let o := (n * O + oc) * size outSp + flat outSp outIdx
              let a := (getAtOrZero aY.lo [o], getAtOrZero aY.hi [o])
              let bias := getAtOrZero b [oc]
              let t := intervalMul a.1 a.2 bias bias
              let cx := (List.range I).foldl (fun cx lc =>
                let ic := oc / (O / groups) * I + lc
                Conv.foldlIndices ks cx fun cx kIdx =>
                  match Conv.mkDilatedInputIdx? outIdx kIdx strides dilations
                      pads with
                  | some inIdx =>
                      if (inIdx.zip sp).all (fun (i, e) => decide (i < e)) then
                        let w := getAtOrZero K (oc :: lc :: kIdx)
                        Op.accum cx ((n * C + ic) * size sp + flat sp inIdx)
                          (intervalMul a.1 a.2 w w)
                      else cx
                  | none => cx) cx
              (cx, (BoundOps.addDown c.1 t.1, BoundOps.addUp c.2 t.2))) acc) init
        some (Op.coeffBox X.dim cx, c)
      else none
  | _, _, _, _, _ => none

/-- The backward rules both directions share, at the coefficient `aY` of node `id`. -/
def directedRule [Op.Capabilities α] (G : IR.Graph) (P : IR.Payload α)
    (ibp : Array (Option (FlatBox α))) (st : State α) (id : Nat) (aY : FlatBox α) :
    IR.Operator → List IR.Ref → State α
  | ⟨_, _, .detach⟩, [r] => addCoeff st r.node aY
  | ⟨_, _, .reshape _⟩, [r] => addCoeff st r.node aY
  | ⟨_, _, .add⟩, [r₁, r₂] => addCoeff (addCoeff st r₁.node aY) r₂.node aY
  | ⟨_, _, .sub⟩, [r₁, r₂] =>
      addCoeff (addCoeff st r₁.node aY) r₂.node (boxNeg aY)
  | ⟨_, _, .linear (lead := L) (inDim := n) (outDim := m)⟩, [rW, rb, rx] =>
      match constAt? G P rW (.value [m, n]), constAt? G P rb (.value [m]) with
      | some W, some b =>
          match Directed.linear aY W b with
          | some (aX, c) =>
              -- The input's coefficient must have the input's width.
              if aX.dim = (L ++ [n]).size then
                addConstant (addCoeff st rx.node aX) c.1 c.2
              else st.fail
          | none => st.fail
      | _, _ => st.fail
  | ⟨_, _, .reduce (s := s) .sum dims false _⟩, [r] =>
      if dims.map Fin.val = List.range s.rank then
        match box? ibp r with
        | some Bx =>
            if h : aY.dim = 1 then
              let lo : Tensor α [1] := castDimScalar (n := aY.dim) (n' := 1) h aY.lo
              let hi : Tensor α [1] := castDimScalar (n := aY.dim) (n' := 1) h aY.hi
              addCoeff st r.node
                { dim := Bx.dim
                  lo := Tensor.full (.dim Bx.dim .scalar) (getAtOrZero lo [0])
                  hi := Tensor.full (.dim Bx.dim .scalar) (getAtOrZero hi [0]) }
            else st.fail
        | none => st.fail
      else consumeAt ibp st id aY
  | ⟨_, _, .cat (s := s) dim sizes⟩, refs =>
      let layout := catLayout s dim.val sizes
      let contributions : Option (List (Nat × FlatBox α)) := do
        unless refs.length == layout.lengths.length do failure
        let boxes ← refs.mapM (box? ibp)
        unless (boxes.zip (List.finRange layout.lengths.length)).all
            (fun (B, i) => B.dim == (layout.parentShape i).size) do failure
        let coefficients ← splitCoeff layout aY
        pure ((refs.map (·.node)).zip coefficients.toList)
      match contributions with
      | some parts => parts.foldl (fun state (p, c) => addCoeff state p c) st
      | none => st.fail
  | ⟨_, _, .slice (s := s) dims starts stops steps _⟩, [r] =>
      match dims.map Fin.val, starts, stops, steps with
      | [axis], [start], [stop], [1] =>
          let contribution := do
            unless start ≤ stop do failure
            let layout ← sliceLayout? axis start (stop - start) s (s.slice dims starts stops steps)
            let input ← box? ibp r
            unless input.dim == layout.sourceShape.size do failure
            scatterSliceBoxes? layout aY
          match contribution with
          | some c => addCoeff st r.node c
          | none => st.fail
      | _, _, _, _ => consumeAt ibp st id aY
  | ⟨_, _, .permute σ _ hout⟩, [r] =>
      match permuteBack σ hout aY with
      | some aX => addCoeff st r.node aX
      | none => st.fail
  | ⟨_, _, .conv (k := k) dims stride padLow padHigh dilation groups (lead := L) _ _ _⟩,
      [rK, rb, rx] =>
      let back : Option (Option (FlatBox α) × (α × α)) := do
        let K ← constAt? G P rK (.value k)
        let b ← constAt? G P rb (.value [k.extent (dims.kernel ⟨0, by omega⟩)])
        let X ← box? ibp rx
        convBack? dims stride padLow padHigh dilation groups L K b X aY
      match back with
      | some (aX, c) =>
          let st := match aX with
            | some B => addCoeff st rx.node B
            | none => st
          addConstant st c.1 c.2
      | none => st.fail
  | ⟨_, _, .gather ..⟩, _ | ⟨_, _, .scatter ..⟩, _ | ⟨_, _, .pad ..⟩, _ | ⟨_, _, .flip ..⟩, _
  | ⟨_, _, .contract ..⟩, _ | ⟨_, _, .logSoftmax ..⟩, _ | ⟨_, _, .cumsum ..⟩, _
  | ⟨_, _, .cumprod ..⟩, _ | ⟨_, _, .cummax ..⟩, _ | ⟨_, _, .cummin ..⟩, _ | ⟨_, _, .sort ..⟩, _
  | ⟨_, _, .topk ..⟩, _ | ⟨_, _, .custom ..⟩, _ => st.fail
  | _, _ => consumeAt ibp st id aY

/-- The contribution of the product term `x[i]·y[j]` with coefficient `a`, operands in
`[lx, ux]` and `[ly, uy]`, to the coefficients of `x` and `y` and to the constant. -/
def productTerm (dir : BackwardDir)
    (acc : Op.PartialCoeffs α × Op.PartialCoeffs α × (α × α))
    (a : α × α) (i j : Nat) (lx ux ly uy : α) : Op.PartialCoeffs α × Op.PartialCoeffs α × (α × α) :=
  let (cx, cy, c) := acc
  let addC (v : α × α) : α × α := (BoundOps.addDown c.1 v.1, BoundOps.addUp c.2 v.2)
  if !(decide (ly < uy)) then (Op.accum cx i (intervalMul a.1 a.2 ly ly), cy, c)
  else if !(decide (lx < ux)) then (cx, Op.accum cy j (intervalMul a.1 a.2 lx lx), c)
  else
    let nonneg := !(decide (a.1 < 0))
    let nonpos := !(decide (0 < a.2))
    if nonneg || nonpos then
      -- Lower plane `x·y ≥ ly·x + lx·y - lx·ly` and upper plane `x·y ≤ uy·x + lx·y - lx·uy`.
      let useLower := match dir with
        | .lower => nonneg
        | .upper => !nonneg
      let (p, r) :=
        if useLower then (ly, Op.negProduct lx ly) else (uy, Op.negProduct lx uy)
      (Op.accum cx i (intervalMul a.1 a.2 p p),
        Op.accum cy j (intervalMul a.1 a.2 lx lx),
        addC (intervalMul a.1 a.2 r.1 r.2))
    else
      let z := intervalMul lx ux ly uy
      (cx, cy, addC (intervalMul a.1 a.2 z.1 z.2))

/-- Apply collected coefficients and constant: add the operand coefficients to `rx` and `ry` and
the constant to the state. -/
def applyProduct (st : State α) (rx ry : Nat) (nx ny : Nat)
    (acc : Op.PartialCoeffs α × Op.PartialCoeffs α × (α × α)) : State α :=
  let (cx, cy, c) := acc
  let st := match Op.coeffBox nx cx with
    | some B => addCoeff st rx B
    | none => st
  let st := match Op.coeffBox ny cy with
    | some B => addCoeff st ry B
    | none => st
  addConstant st c.1 c.2

/-- The product terms of a matrix product at layout `dims` with output coefficients `a`
(`none`: the output entry has no coefficient). -/
def matmulTerms (dir : BackwardDir) (dims : IR.MatmulDims)
    (a : Fin dims.outShape.size → Option (α × α)) (L R : FlatBox α)
    (acc : Op.PartialCoeffs α × Op.PartialCoeffs α × (α × α)) :
    Op.PartialCoeffs α × Op.PartialCoeffs α × (α × α) :=
  (List.finRange dims.outShape.size).foldl (fun acc o =>
    match a o with
    | some ao =>
        (List.range dims.inner).foldl (fun acc k =>
          let i := dims.leftIndex o.val k
          let j := dims.rightIndex o.val k
          Op.productTerm dir acc ao i j (getAtOrZero L.lo [i]) (getAtOrZero L.hi [i])
            (getAtOrZero R.lo [j]) (getAtOrZero R.hi [j])) acc
    | none => acc) acc

/-- The backward rules at direction `dir`: the products, then `directedRule`. -/
def backwardRule [Op.Capabilities α] (dir : BackwardDir) (G : IR.Graph)
    (P : IR.Payload α) (ibp : Array (Option (FlatBox α))) (st : State α)
    (id : Nat) (aY : FlatBox α) : IR.Operator → List IR.Ref → State α
  | ⟨_, _, .mul (s := s)⟩, [r₁, r₂] =>
      match box? ibp r₁, box? ibp r₂ with
      | some X, some Y =>
          if aY.dim = s.size ∧ X.dim = s.size ∧ Y.dim = s.size then
            let init : PartialCoeffs α × PartialCoeffs α × (α × α) :=
              (Array.replicate X.dim none, Array.replicate Y.dim none, (0, 0))
            let acc := (List.range aY.dim).foldl (fun acc i =>
              productTerm dir acc (getAtOrZero aY.lo [i], getAtOrZero aY.hi [i]) i i
                (getAtOrZero X.lo [i]) (getAtOrZero X.hi [i]) (getAtOrZero Y.lo [i])
                (getAtOrZero Y.hi [i])) init
            applyProduct st r₁.node r₂.node X.dim Y.dim acc
          else st.fail
      | _, _ => st.fail
  | ⟨_, _, .matmul (batchA := bA) (batchB := bB) (batch := b) (m := m) (n := n) (p := p) lv rv
      _ _ _ _⟩, [r₁, r₂] =>
      let dims := matmulLayout lv rv bA bB b m n p
      match box? ibp r₁, box? ibp r₂ with
      | some L, some R =>
          if aY.dim = dims.outShape.size ∧ L.dim = dims.leftShape.size ∧
              R.dim = dims.rightShape.size then
            let init := (Array.replicate L.dim none, Array.replicate R.dim none, (0, 0))
            let acc := matmulTerms dir dims
              (fun o => some (getAtOrZero aY.lo [o.val], getAtOrZero aY.hi [o.val])) L R init
            applyProduct st r₁.node r₂.node L.dim R.dim acc
          else st.fail
      | _, _ => st.fail
  | ⟨_, _, .scaledMatmul (batchA := bA) (batchB := bB) (batch := b) (m := m) (n := n) (p := p) lv
      rv _ _ _ _⟩, [r₁, r₂, r₃] =>
      let dims := matmulLayout lv rv bA bB b m n p
      match box? ibp r₁, box? ibp r₂, box? ibp r₃ with
      | some L, some R, some S =>
          if aY.dim = dims.outShape.size ∧ L.dim = dims.leftShape.size ∧
              R.dim = dims.rightShape.size ∧ S.dim = 1 then
            let W := binaryMatmulBox dims L R
            -- `s·w_o`: the coefficient of each `w_o` and the contribution to `s`.
            let init : PartialCoeffs α × PartialCoeffs α × (α × α) :=
              (Array.replicate aY.dim none, Array.replicate 1 none, (0, 0))
            let (cw, cs, c) := (List.range aY.dim).foldl (fun acc o =>
              productTerm dir acc (getAtOrZero aY.lo [o], getAtOrZero aY.hi [o]) o 0
                (getAtOrZero W.lo [o]) (getAtOrZero W.hi [o]) (getAtOrZero S.lo [0])
                (getAtOrZero S.hi [0])) init
            let st := match coeffBox 1 cs with
              | some B => addCoeff st r₃.node B
              | none => st
            -- `w_o = Σ_k x·y`: the matrix product's terms at the coefficients of `w`.
            let acc := matmulTerms dir dims (fun o => (cw[o.val]?).join) L R
              (Array.replicate L.dim none, Array.replicate R.dim none, c)
            applyProduct st r₁.node r₂.node L.dim R.dim acc
          else st.fail
      | _, _, _ => st.fail
  | o, refs => directedRule G P ibp st id aY o refs

/-- The step of a backward sweep at node `id` with the operator rule `rule`, for a node with a
coefficient. The designated input keeps it, another input or a constant discharges it against its
box, and an operator node applies `rule` when its references all name output `0`. -/
def backStep (rule : State α → Nat → FlatBox α → IR.Operator →
      List IR.Ref → State α)
    (G : IR.Graph) (P : IR.Payload α) (ibp : Array (Option (FlatBox α))) (ctx : AffineCtx)
    (st : State α) (id : Nat) : State α :=
  if st.failed then st else
  match st.coeffs[id]! with
  | none => st
  | some aY =>
      match G.nodes[id]? with
      | some (.input _ _) => if id = ctx.inputId then st else consumeAt ibp st id aY
      | some (.const _) =>
          match constValue? G P id >>= flatOf with
          | some v => consume st aY (FlatBox.ofTensor v.v)
          | none => st.fail
      | some (.op o refs) =>
          if refs.all (·.out == 0) then rule st id aY o refs.toList else st.fail
      | none => st.fail

/-- The node step of the shared rules. -/
def directedStep [Op.Capabilities α] (G : IR.Graph) (P : IR.Payload α)
    (ibp : Array (Option (FlatBox α))) (ctx : AffineCtx) :
    State α → Nat → State α :=
  backStep (directedRule G P ibp) G P ibp ctx

/-- The node step at direction `dir`, with the product rules. -/
def backwardStep [Op.Capabilities α] (dir : BackwardDir) (G : IR.Graph)
    (P : IR.Payload α) (ibp : Array (Option (FlatBox α))) (ctx : AffineCtx) :
    State α → Nat → State α :=
  backStep (backwardRule dir G P ibp) G P ibp ctx

/-- A reverse sweep of `step` over the nodes of `G` from the objective `obj` at node `out`, giving
the lower and upper affine forms of the coefficient it leaves at the input and its constant. It is
none when the sweep fails or the input has no box. -/
def sweep (step : State α → Nat → State α) (G : IR.Graph)
    (ctx : AffineCtx) (ibp : Array (Option (FlatBox α))) (out : Nat) (obj : FlatTensor α) :
    Option (Linear α ctx.inputDim 1 × Linear α ctx.inputDim 1) := do
  if out < G.nodes.size then
    let initCoeffs :=
      (Array.replicate G.nodes.size none).set! out (some (FlatBox.ofTensor obj.v))
    let init : State α := { coeffs := initCoeffs, cstLo := 0, cstHi := 0 }
    let st := Fin.foldr G.nodes.size (fun i acc => step acc i.val) init
    if st.failed then
      none
    else
      let inputBox ← ibp[ctx.inputId]?
      let inputBox ← inputBox
      let aIn := st.coeffs[ctx.inputId]!.getD
        { dim := ctx.inputDim
          lo := Tensor.full (.dim ctx.inputDim .scalar) 0
          hi := Tensor.full (.dim ctx.inputDim .scalar) 0 }
      inputAffines ctx.inputDim inputBox aIn st.cstLo st.cstHi
  else
    none

variable [Op.Capabilities α]

/-- The value of the affine form `a` on the box `xB`, its lower end when `lower` and its upper end
otherwise. It is none when `xB` has another size. -/
def formValue? (lower : Bool) (xB : FlatBox α) {n : Nat} (a : Linear α n 1) :
    Option α :=
  if h : xB.dim = n then
    let B := FlatAffineBounds.evalOnFlatBox { inDim := n, outDim := 1, loAff := a, hiAff := a } xB h
    some (if lower then getAtOrZero B.lo [0] else getAtOrZero B.hi [0])
  else none

/-- Of two lower forms (`lower`) the second when its value on `xB` is larger, of two upper forms
the second when its value is smaller, and otherwise the first. -/
def tighterForm (lower : Bool) (xB? : Option (FlatBox α)) {n : Nat}
    (a b : Linear α n 1) : Linear α n 1 :=
  match xB? >>= fun xB => (Op.formValue? lower xB a).bind fun va =>
      (Op.formValue? lower xB b).map fun vb => (va, vb) with
  | some (va, vb) => if (if lower then decide (va < vb) else decide (vb < va)) then b else a
  | none => a

/-- The lower and the upper form of `objᵀ · value(out)`, each when one of its sweeps succeeds: the
form of the sweep in its direction (`backwardStep`), or of the sweep with the shared rules alone
(`directedStep`, which discharges every product at its box) when that is tighter on the box `input`
or the directed sweep fails. -/
def objectiveForms (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (out : Nat) (obj : FlatTensor α)
    (input : Option (FlatBox α)) :
    Option (Linear α ctx.inputDim 1) × Option (Linear α ctx.inputDim 1) :=
  let shared := sweep (directedStep G P ibp ctx) G ctx ibp out obj
  let pick (lower : Bool) (d : Option (Linear α ctx.inputDim 1))
      (s : Option (Linear α ctx.inputDim 1)) : Option (Linear α ctx.inputDim 1) :=
    match d, s with
    | some a, some b => some (Op.tighterForm lower input a b)
    | some a, none => some a
    | none, b => b
  (pick true ((sweep (backwardStep .lower G P ibp ctx) G ctx ibp out obj).map (·.1))
      (shared.map (·.1)),
    pick false ((sweep (backwardStep .upper G P ibp ctx) G ctx ibp out obj).map (·.2))
      (shared.map (·.2)))

/-- Lower and upper affine forms of `objᵀ · value(out)` from `objectiveForms`, choosing on
the input's box `ibp[ctx.inputId]`. It is none when either form is missing. -/
def backwardObjective? (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (out : Nat) (obj : FlatTensor α) :
    Option (Linear α ctx.inputDim 1 × Linear α ctx.inputDim 1) :=
  match objectiveForms G P ctx ibp out obj (ibp[ctx.inputId]?).join with
  | (some l, some u) => some (l, u)
  | _ => none

/-- Lower and upper affine forms of `C · value(out)`, one pair per row of the matrix objective `C`,
from `backwardObjective?` of each row. It is none when `C`'s width is not the size of node `out` or
the sweeps of a row fail. -/
def affineBounds? (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (out : Nat) {k n : Nat} (C : Tensor α [k, n]) :
    Option (FlatAffineBounds α) := do
  let outDim ← outSize? G out
  guard (n = outDim)
  let rows ← Tensor.sequenceFinM fun (i : Fin k) =>
    backwardObjective? G P ctx ibp out { n := n, v := Tensor.ofFn fun j => Tensor.get2 C i j }
  pure
    { inDim := ctx.inputDim
      outDim := k
      loAff :=
        { weights := Tensor.matrix fun i j => Tensor.get2 (rows i).1.weights 0 j
          bias := Tensor.ofFn fun i => Tensor.getScalar (rows i).1.bias 0 }
      hiAff :=
        { weights := Tensor.matrix fun i j => Tensor.get2 (rows i).2.weights 0 j
          bias := Tensor.ofFn fun i => Tensor.getScalar (rows i).2.bias 0 } }

/-- Affine bounds of node `out`: `affineBounds?` at the identity objective, or the constant forms of
the node's box when that fails. -/
def nodeBounds? (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (out : Nat) : Option (FlatAffineBounds α) := do
  let outDim ← outSize? G out
  match affineBounds? G P ctx ibp out
      (Tensor.matrix fun (i j : Fin outDim) => if i = j then (1 : α) else 0) with
  | some bounds => pure bounds
  | none =>
    let box ← (ibp[out]?).join
    pure (boundsConst ctx.inputDim box.dim box.lo box.hi)

/-- The node of graph output `i` and the context of graph input `0` with box `xB`. It is an error
when the graph has no output `i` or no input `0`, or output `i` names an output other than `0`. -/
def outputContext (G : IR.Graph) (xB : FlatBox α) (i : Nat) :
    Except String (Nat × AffineCtx) := do
  let some r := G.outputs[i]?
    | throw s!"graph output {i} is out of bounds for {G.outputs.size} outputs"
  unless r.out == 0 do throw s!"graph output {i} reads output {r.out} of node {r.node}"
  let some inputId := inputId? G 0 | throw "the graph has no input 0"
  pure (r.node, { inputId, inputDim := xB.dim })

variable [NonlinearBoundOps α]

/-- The CROWN box of graph output `i` with the box `xB` at graph input `0`: interval propagation,
then `nodeBounds?` evaluated on `xB`, met with the output's interval box (`meetWith`: an error
without one of the right size). -/
@[noinline, nospecialize]
def outputBoxCROWN? (G : IR.Graph) (P : IR.Payload α) (xB : FlatBox α) (i : Nat) :
    Except String (FlatBox α) := do
  let (out, ctx) ← outputContext G xB i
  let ibp := runIBP G P (inputSeeds G [xB])
  let bounds := (Array.replicate G.nodes.size none).set! out (nodeBounds? G P ctx ibp out)
  meetWith ibp out (← evalCROWNOutputBox? bounds xB out ctx.inputDim)

/-- The box of node `id` in full CROWN from the boxes `tbl` of the nodes before it: the interval
rule on `tbl` (`step?` at `ibp?`), met with the supplied box `ibp[id]` when there is one. At an
operator node it is also met with the node's affine bounds at the identity objective
(`affineBounds?` over `tbl` and that box), evaluated on `xB`. -/
def crownStep (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (seeds : Nat → Option (FlatBox α)) (ibp : Array (Option (FlatBox α))) (xB : FlatBox α)
    (tbl : Array (Option (FlatBox α))) (id : Nat) : Option (FlatBox α) := do
  let B ← step? G P (ibp? G P) seeds tbl id
  let B := match (ibp[id]?).join with
    | some S => meetBox S B
    | none => B
  match G.nodes[id]? with
  | some (.op _ _) =>
      let C := affineBounds? G P ctx (tbl.push (some B)) id
        (Tensor.matrix fun (i j : Fin B.dim) => if i = j then (1 : α) else 0)
      match C.bind fun b => (evalCROWNOutputBox? #[some b] xB 0 ctx.inputDim).toOption with
      | some box => pure (meetBox box B)
      | none => pure B
  | _ => pure B

/-- Full CROWN: the box of every node by `crownStep`, in order, from the seeds, the enclosing
table `ibp` and the box `xB` of the designated input. -/
@[noinline, nospecialize]
def crownBoxes (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (seeds : Nat → Option (FlatBox α)) (ibp : Array (Option (FlatBox α))) (xB : FlatBox α) :
    Array (Option (FlatBox α)) :=
  tabulate (crownStep G P ctx seeds ibp xB) G.nodes.size

/-- Per row, of the lower forms of `a` and `b` the one with the larger value on `xB`, and of the
upper forms the one with the smaller value, with `a`'s on a tie. It is `a` when the two have
different sizes or `xB` does not have their input's size. Mixing rows gives a bound, since every row
of either is one. -/
def tighterRows (xB : FlatBox α) (a b : FlatAffineBounds α) : FlatAffineBounds α :=
  if h : xB.dim = a.inDim ∧ xB.dim = b.inDim ∧ a.outDim = b.outDim then
    let ea := a.evalOnFlatBox xB h.1
    let eb := b.evalOnFlatBox xB h.2.1
    let lo (i : Nat) : Bool := decide (getAtOrZero ea.lo [i] < getAtOrZero eb.lo [i])
    let hi (i : Nat) : Bool := decide (getAtOrZero eb.hi [i] < getAtOrZero ea.hi [i])
    { a with
      loAff :=
        { weights := Tensor.matrix fun i j =>
            if lo i.val then getAtOrZero b.loAff.weights [i.val, j.val]
            else Tensor.get2 a.loAff.weights i j
          bias := Tensor.ofFn fun i => if lo i.val then getAtOrZero b.loAff.bias [i.val]
            else a.loAff.bias.getScalar i }
      hiAff :=
        { weights := Tensor.matrix fun i j =>
            if hi i.val then getAtOrZero b.hiAff.weights [i.val, j.val]
            else Tensor.get2 a.hiAff.weights i j
          bias := Tensor.ofFn fun i => if hi i.val then getAtOrZero b.hiAff.bias [i.val]
            else a.hiAff.bias.getScalar i } }
  else a

/-- Affine bounds of every node over the boxes of `crownBoxes`: the identity at the designated
input, the constant forms of a constant's value and of another input's box, and at an operator
node, per row, the tighter on `xB` of `nodeBounds?` and the constant forms of the node's box. -/
@[noinline, nospecialize]
def runCROWN (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (seeds : Nat → Option (FlatBox α)) (ibp : Array (Option (FlatBox α))) (xB : FlatBox α) :
    Array (Option (FlatAffineBounds α)) :=
  let boxes := crownBoxes G P ctx seeds ibp xB
  (Array.range G.nodes.size).map fun (id : Nat) =>
    match (G.nodes[id]? : Option IR.Node) with
    | some (.input _ _) =>
        if id = ctx.inputId then some (boundsIdentity ctx.inputDim)
        else (boxes[id]?).join.map fun B => boundsConst ctx.inputDim B.dim B.lo B.hi
    | some (.const _) =>
        (constValue? G P id >>= flatOf).map fun v => boundsConst ctx.inputDim v.n v.v v.v
    | some (.op _ _) =>
        match nodeBounds? G P ctx boxes id, (boxes[id]?).join with
        | some b, some B =>
            some (tighterRows xB b (boundsConst ctx.inputDim B.dim B.lo B.hi))
        | b, _ => b
    | none => none

/-- The box of `objᵀ · value` at graph output `i` with the box `xB` at graph input `0` and the
interval table `ibp`. The forms of `objectiveForms` chosen on `xB`, or for a direction
without one the objective on the output's box, are evaluated on `xB` and met with the objective on
the output's box. It is an error without an interval box at the output or when the objective does
not fit it. -/
@[noinline, nospecialize]
def backwardObjectiveBox? (G : IR.Graph) (P : IR.Payload α)
    (ibp : Array (Option (FlatBox α))) (xB : FlatBox α) (i : Nat) (obj : FlatTensor α) :
    Except String (FlatBox α) := do
  let (out, ctx) ← outputContext G xB i
  let (lo, hi) := objectiveForms G P ctx ibp out obj (some xB)
  let lo := lo <|> outputBoxBound .lower ibp out ctx.inputDim obj
  let hi := hi <|> outputBoxBound .upper ibp out ctx.inputDim obj
  let B ← match lo, hi with
    | some loAff, some hiAff =>
        evalObjectiveBox? { inDim := ctx.inputDim, outDim := 1, loAff, hiAff } xB
          ctx.inputDim
    | _, _ => throw "CROWN backward objective failed"
  let some I := (ibp[out]?).join | throw s!"no interval box at node {out}"
  match consumeBox .lower obj I, consumeBox .upper obj I with
  | some l, some u =>
      if B.dim = 1 then
        return meetBox B { dim := 1, lo := Tensor.full [1] l, hi := Tensor.full [1] u }
      else throw "the objective's box has more than one entry"
  | _, _ => throw s!"objective of {obj.n} entries on the interval box of node {out} ({I.dim})"

/-! ## Forward pass -/

omit [Op.Capabilities α] [NonlinearBoundOps α] in
/-- The constant forms of node `id`'s box. -/
def boxBounds? (ibp : Array (Option (FlatBox α))) (ctx : AffineCtx) (id : Nat) :
    Option (FlatAffineBounds α) :=
  (ibp[id]?).join.map fun B => boundsConst ctx.inputDim B.dim B.lo B.hi

omit [Op.Capabilities α] [NonlinearBoundOps α] in
/-- The forward affine bounds of ReLU from the reference's bounds `xin`, its box `preB`, the slopes
`alpha` and the phases `phases` of node `id`. -/
def reluBounds? (alpha : Array (Option (FlatTensor α))) (phases : Array (Option (Array Int)))
    (id : Nat) (xin : FlatAffineBounds α) (preB : FlatBox α) : Option (FlatAffineBounds α) :=
  if hout : xin.outDim = preB.dim then
    let xLo := castAffineOut hout xin.loAff
    let xHi := castAffineOut hout xin.hiAff
    let finish (relaxLo relaxHi : Tensor (ReLURelax α) [preB.dim]) :
        FlatAffineBounds α :=
      { inDim := xin.inDim, outDim := preB.dim
        loAff := ReLU.propagateAffine (inDim := xin.inDim) (hidDim := preB.dim)
          relaxLo xLo
        hiAff := ReLU.propagateAffine (inDim := xin.inDim) (hidDim := preB.dim)
          relaxHi xHi }
    match Cert.getPhases? phases id with
    | none =>
        let αt : Tensor α [preB.dim] :=
          match Cert.getAlpha? alpha id with
          | some αv =>
              if hα : αv.n = preB.dim then castDimScalar (n := αv.n) (n' := preB.dim) hα αv.v
              else Cert.defaultAlphaVec (n := preB.dim) preB.lo preB.hi
          | none => Cert.defaultAlphaVec (n := preB.dim) preB.lo preB.hi
        some (finish (Cert.alphaRelaxLowerVec (n := preB.dim) preB.lo preB.hi αt)
          (ReLU.relaxVector (n := preB.dim) preB.lo preB.hi))
    | some nodePhases =>
        let αt? : Option (Tensor α [preB.dim]) :=
          match Cert.getAlpha? alpha id with
          | some αv =>
              if hα : αv.n = preB.dim then some (castDimScalar (n := αv.n) (n' := preB.dim) hα αv.v)
              else none
          | none => some (Cert.defaultAlphaVec (n := preB.dim) preB.lo preB.hi)
        match αt? with
        | some αt =>
            (Cert.phaseRelaxVec? (n := preB.dim) preB.lo preB.hi αt nodePhases).map
              fun (relaxLo, relaxHi) => finish relaxLo relaxHi
        | none => none
  else none

omit [Op.Capabilities α] [NonlinearBoundOps α] in
/-- The forward affine bounds of operator `o` at node `id` on references `refs`, from their bounds
in `cert`, the interval table `ibp`, the slopes `alpha` and the phases `phases`. -/
def alpha? (G : IR.Graph) (P : IR.Payload α) (ibp : Array (Option (FlatBox α)))
    (alpha : Array (Option (FlatTensor α))) (phases : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds α))) (ctx : AffineCtx) (id : Nat) :
    IR.Operator → List IR.Ref → Option (FlatAffineBounds α)
  | ⟨_, _, .detach⟩, [r] => Cert.getAff? cert r.node
  | ⟨_, _, .linear (lead := L) (inDim := n) (outDim := m)⟩, [rW, rb, rx] => do
      let W ← constAt? G P rW (.value [m, n])
      let b ← constAt? G P rb (.value [m])
      let xin ← Cert.getAff? cert rx.node
      if hout : xin.outDim = n then
        if n = 0 then
          let bias := linearBiasRows b L.size
          some (boundsConst ctx.inputDim _ bias bias)
        else
          some (propagateLinearBounds (n := n) (m := m) W b xin hout)
      else none
  | ⟨_, _, .relu⟩, [r] => do
      let xin ← Cert.getAff? cert r.node
      let preB ← box? ibp r
      reluBounds? alpha phases id xin preB
  | ⟨_, _, .reduce (s := s) .sum dims false _⟩, [r] =>
      if dims.map Fin.val = List.range s.rank then do
        let xin ← Cert.getAff? cert r.node
        let onesRow : Tensor α [1, xin.outDim] := Tensor.full (.dim 1 (.dim xin.outDim .scalar)) 1
        let zb : Tensor α [1] := Tensor.full (.dim 1 .scalar) 0
        some (propagateLinearBounds (n := xin.outDim) (m := 1) onesRow zb xin rfl)
      else boxBounds? ibp ctx id
  | ⟨_, _, .reshape (t := t) _⟩, [r] => do
      let xin ← Cert.getAff? cert r.node
      if hout : xin.outDim = t.size then
        some { inDim := xin.inDim, outDim := t.size
               loAff := castAffineOut (n := xin.inDim) (m := xin.outDim) (m' := t.size) hout
                 xin.loAff
               hiAff := castAffineOut (n := xin.inDim) (m := xin.outDim) (m' := t.size) hout
                 xin.hiAff }
      else none
  | ⟨_, _, .matmul ..⟩, _ | ⟨_, _, .custom ..⟩, _ => none
  | _, _ => boxBounds? ibp ctx id

omit [Op.Capabilities α] [NonlinearBoundOps α] in
/-- The forward affine bounds of node `id`: the identity at the designated input, the constant forms
of a constant, and the operator's rule at an operator node whose references all name output `0`. -/
def alphaStep? (G : IR.Graph) (P : IR.Payload α) (ibp : Array (Option (FlatBox α)))
    (alpha : Array (Option (FlatTensor α))) (phases : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds α))) (ctx : AffineCtx) (id : Nat) :
    Option (FlatAffineBounds α) :=
  match G.nodes[id]? with
  | some (.input _ _) => if id = ctx.inputId then some (boundsIdentity ctx.inputDim) else none
  | some (.const _) =>
      (constValue? G P id >>= flatOf).map fun v => boundsConst ctx.inputDim v.n v.v v.v
  | some (.op o refs) =>
      if refs.all (·.out == 0) then alpha? G P ibp alpha phases cert ctx id o refs.toList
      else none
  | none => none

omit [BoundOps α] [Op.Capabilities α] [NonlinearBoundOps α] in
/-- The ReLU phases the interval table justifies, per node. -/
def inferredPhases (G : IR.Graph) (ibp : Array (Option (FlatBox α))) :
    Array (Option (Array Int)) :=
  (List.range G.nodes.size).toArray.map fun (id : Nat) =>
    match G.nodes[id]? with
    | some (.op o refs) =>
        match o, refs.toList with
        | ⟨_, _, .relu⟩, [r] =>
            (box? ibp r).map fun B => (List.finRange B.dim).toArray.map fun i =>
              (Cert.inferredPhase (B.lo.getScalar i) (B.hi.getScalar i)).toInt
        | _, _ => none
    | _ => none

omit [Op.Capabilities α] [NonlinearBoundOps α] in
/-- The forward pass with the phases `ibp` justifies and the default slopes. -/
def runAlphaCROWN (G : IR.Graph) (P : IR.Payload α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) : Array (Option (FlatAffineBounds α)) :=
  tabulate (fun cert id =>
    alphaStep? G P ibp (Array.replicate G.nodes.size none) (inferredPhases G ibp) cert ctx id)
    G.nodes.size

/-- The phased α-CROWN box of graph output `i` with the box `xB` at graph input `0`: at a scalar
without exact reassociation `outputBoxCROWN?`, otherwise the forward pass evaluated on `xB`, met
with the output's interval box (`meetWith`). We fall back there because the forward rules use the
scalar's own arithmetic, and in binary32 `propagateLinearBounds` rounds to nearest, so its forms
need not bound the real value. No theorem covers this box itself. -/
@[noinline, nospecialize]
def outputBoxAlphaCROWN? (G : IR.Graph) (P : IR.Payload α) (xB : FlatBox α)
    (i : Nat) :
    Except String (FlatBox α) := do
  if !BoundOps.exactReassociation (α := α) then
    return ← outputBoxCROWN? G P xB i
  let (out, ctx) ← outputContext G xB i
  let ibp := runIBP G P (inputSeeds G [xB])
  meetWith ibp out (← evalCROWNOutputBox? (runAlphaCROWN G P ctx ibp) xB out ctx.inputDim)

end TorchLean.CROWN.Op
