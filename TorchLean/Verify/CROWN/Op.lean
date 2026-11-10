/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.MatmulLayout
public import TorchLean.NN.IR.Eval
public import TorchLean.Verify.CROWN.FlatBox.Real
public import TorchLean.Verify.CROWN.FlatBox.Conv

/-!
# CROWN on operator graphs: node access

The bound passes over graphs of registered operators (`TorchLean.IR.Graph`, written `IR.Graph` here)
read a node's operator, the boxes of the outputs its references name, and the constants its
parameter references name. Bounds exist for output `0` of a node: `box? boxes r` is the entry of
node `r.node` when `r.out = 0` and none otherwise. A parameter (a linear layer's weight and bias) is
a constant only when its reference names output `0` of a constant node. `constAt? G P r p` reads the
payload constant at the constant node's port and then at the consumer's port `p`, as a run reads it,
and is none for any other reference, so a rule that needs a constant gives no box for a computed
parameter. The file also defines the interval boxes of a linear layer and of a convolution in
PyTorch's layout (`ibpLinearAt`, `convOpBox?`) and the port layouts of a matrix product and a
concatenation (`matmulLayout`, `catLayout`). `runVal` is the flat value of a node in a run, and
`TopoSorted` says that every reference of an operator node names an earlier node.

`tabulate step k` runs a step node by node in order: entry `j` is the step at `j` on the entries
before it. A step that reads only the entries of earlier nodes has its table as a fixpoint
(`tabulate_fix`).
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean.CROWN.Op

open Graph (FlatTensor)

variable {α : Type} [Storage α] [Context α]

/-- The box of output `0` of the node that `r` names, and none for any other output. -/
def box? {β : Type} (boxes : Array (Option β)) (r : IR.Ref) : Option β :=
  if r.out = 0 then (boxes[r.node]?).join else none

/-- The flat tensor of a value at a port of the scalar (a value, frozen or detached port), and none
at an index, Boolean or integer port. -/
def flatOf : Op.Value α → Option (FlatTensor α)
  | ⟨⟨.value, s⟩, x⟩ => some ⟨s.size, Tensor.reshape x [_] (Shape.size_vector _).symm⟩
  | ⟨⟨.frozen, s⟩, x⟩ => some ⟨s.size, Tensor.reshape x [_] (Shape.size_vector _).symm⟩
  | ⟨⟨.detached, s⟩, x⟩ => some ⟨s.size, Tensor.reshape x [_] (Shape.size_vector _).symm⟩
  | _ => none

/-- The value of constant node `id` as the run reads it: the payload constant at the node's
port. -/
def constValue? (G : IR.Graph) (P : IR.Payload α) (id : Nat) :
    Option (Op.Value α) :=
  match G.nodes[id]? with
  | some (.const p) => do
      let v ← P.const? id
      let x ← (p.read v).toOption
      pure ⟨p, x⟩
  | _ => none

/-- The tensor that reference `r` names at input port `p` when `r` is output `0` of a constant
node: the payload constant read at the constant's port and then at `p`, as a run reads it. Any
other reference gives none. -/
def constAt? (G : IR.Graph) (P : IR.Payload α) (r : IR.Ref) (p : Port) :
    Option (p.Carrier α) :=
  if r.out = 0 then do
    let v ← constValue? G P r.node
    (p.read v).toOption
  else none

/-- The interval box of a linear layer with parameters `p` on the box of an input at port shape
`x`: at an empty input extent the exact bias at each leading row of `x` (the box must have width
`0`), otherwise `ibpLinearParams`, which is `none` unless the box has `p.inDim` entries, so an
input with more than one leading row gets no box. -/
def ibpLinearAt [CROWN.BoundOps α] (p : SomeLinear α)
    (x : Shape) (Xin : CROWN.FlatBox α) : Option (CROWN.FlatBox α) :=
  if p.inDim = 0 then
    if Xin.dim = 0 then
      let bias := CROWN.Graph.linearBiasRows p.spec.bias
        (Shape.size x.dropLast)
      some { dim := _, lo := bias, hi := bias }
    else none
  else CROWN.Graph.ibpLinearParams (α := α) p Xin

/-- The flat layout of the product `Op.matmul lv rv` computes at its ports: the batch prefixes
`batchA` and `batchB` broadcast to `batch`, an `m × n` by `n × p` product, a vector operand where
its flag is set. -/
def matmulLayout (lv rv : Bool) (batchA batchB batch : Shape) (m n p : Nat) :
    IR.MatmulDims where
  leading := batch
  rows := m
  inner := n
  cols := p
  leftShape := Shape.matmulLeft lv batchA m n
  rightShape := Shape.matmulRight rv batchB n p
  leftLeading := batchA
  rightLeading := batchB
  leftVector := lv
  rightVector := rv

/-- The concat layout of `Op.cat dim sizes` at its ports: the axes of `s` before and after `dim`,
and the inputs' extents along it. -/
def catLayout (s : Shape) (dim : Nat) (sizes : List Nat) :
    CROWN.Graph.ConcatLayout where
  leading := s.take dim
  trailing := s.drop (dim + 1)
  lengths := sizes

/-- Dimension numbers of a convolution in PyTorch's layout: at every position `p`, the input axis
and the kernel axis are both `p`. At `r = 2` that is the axis order `torch.nn.functional.conv2d`
documents (PyTorch 2.8 docs): input `(minibatch, in_channels, iH, iW)` and weight
`(out_channels, in_channels / groups, kH, kW)`. -/
def ChannelsFirst {s t : Shape} {r : Nat} (dims : Shape.ConvDims s t r) : Prop :=
  ∀ p, (dims.input p).val = p.val ∧ (dims.kernel p).val = p.val

/-- `ChannelsFirst dims` is decided by checking each of its `r + 2` positions. -/
instance {s t : Shape} {r : Nat} (dims : Shape.ConvDims s t r) : Decidable (ChannelsFirst dims) :=
  inferInstanceAs
    (Decidable (∀ p : Fin (r + 2), (dims.input p).val = p.val ∧ (dims.kernel p).val = p.val))

/-- The vector of the entries of a list of length `r`. -/
def vecOf (l : List Nat) {r : Nat} (h : l.length = r) : Tensor Nat [r] :=
  Tensor.ofFn fun j => l[j.val]'(h ▸ j.isLt)

/-- The entries of `vecOf l` are those of `l`. -/
@[simp] theorem vecOf_toList (l : List Nat) {r : Nat} (h : l.length = r) :
    (vecOf l h).data.toList = l := by
  apply List.ext_getElem (by simp [h])
  intro j h₁ h₂
  rw [Tensor.getElem_data_toList (vecOf l h) ⟨j, by simpa using h₁⟩]
  simp [vecOf]

/-- The interval box of `Op.conv` on the box of its input, from its kernel `K` and bias `b`. For a
convolution in PyTorch's layout (`ChannelsFirst`), with an input `(n, inC, spatial...)` whose batch
axis stands for the axes `lead` and a kernel `(outC, inC / groups, kernel...)`, it is
`Graph.convBox?` at the operator's kernel extents, stride, padding before and after, dilation and
spatial extents. Any other layout or shape gives none. -/
def convOpBox? [CROWN.BoundOps α] {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (lead : Shape) (K : Tensor α k)
    (b : Tensor α [k.extent (dims.kernel ⟨0, by omega⟩)]) (X : CROWN.FlatBox α) :
    Option (CROWN.FlatBox α) :=
  match s, k, dims, K, b with
  | _ :: C :: sp, O :: I :: ks, dims, K, b =>
      if h : ChannelsFirst dims ∧ I = C / groups ∧ ks.length = r ∧ sp.length = r then
        CROWN.Graph.convBox? (vecOf ks h.2.2.1) (Tensor.ofFn stride)
          (Tensor.ofFn padLow) (Tensor.ofFn padHigh) (Tensor.ofFn dilation) (vecOf sp h.2.2.2)
          groups lead
          (Tensor.castShape K (by rw [vecOf_toList, h.2.1]) :
            Tensor α (O :: C / groups :: (vecOf ks h.2.2.1).data.toList))
          (Tensor.castShape b (by
            have h0 := (h.1 ⟨0, by omega⟩).2
            simp only [Shape.extent]
            congr
            simp only [h0]
            rfl) : Tensor α [O])
          X
      else none
  | _, _, _, _, _ => none

/-- The flat value of output `0` of node `id` in a run's outputs. -/
def runVal (values : Array (List (Op.Value α))) (id : Nat) : Option (FlatTensor α) :=
  (values[id]? >>= (·[0]?)) >>= flatOf

/-- Every reference of an operator node names an earlier node. -/
def TopoSorted (G : IR.Graph) : Prop :=
  ∀ id, id < G.nodes.size → ∀ (o : IR.Operator) (refs : Array IR.Ref),
    G.nodes[id]? = some (.op o refs) → ∀ r ∈ refs, r.node < id

section Tabulate

variable {β : Type} (step : Array (Option β) → Nat → Option β)

/-- The first `k` entries of the table `step` computes node by node in order. -/
def tabulate : Nat → Array (Option β)
  | 0 => #[]
  | k + 1 => (tabulate k).push (step (tabulate k) k)

/-- `tabulate step k` has `k` entries. -/
theorem size_tabulate : ∀ k, (tabulate step k).size = k
  | 0 => rfl
  | k + 1 => by simp [tabulate, size_tabulate k]

variable {step}

/-- Later steps keep the entries already computed. -/
theorem tabulate_getElem?_of_lt {j : Nat} :
    ∀ {k m : Nat}, j < k → k ≤ m → (tabulate step m)[j]? = (tabulate step k)[j]?
  | k, 0, hj, hkm => by omega
  | k, m + 1, hj, hkm => by
      by_cases hk : k = m + 1
      · subst hk; rfl
      · have hm : k ≤ m := by omega
        have hjm : j < (tabulate step m).size := by rw [size_tabulate]; omega
        simp only [tabulate]
        rw [Array.getElem?_push_lt hjm, ← Array.getElem?_eq_getElem hjm]
        exact tabulate_getElem?_of_lt hj hm

/-- Entry `k` is the step at node `k` on the entries before it. -/
theorem tabulate_getElem?_self {k m : Nat} (hk : k < m) :
    (tabulate step m)[k]? = some (step (tabulate step k) k) := by
  rw [tabulate_getElem?_of_lt (Nat.lt_succ_self k) hk]
  simp only [tabulate]
  have hs : k = (tabulate step k).size := (size_tabulate step k).symm
  conv => lhs; arg 2; rw [hs]
  exact Array.getElem?_push_size

/-- Entry `id` of the table of size `n > id` is the step at node `id` on the first `id` entries, and
those agree with the table below `id`. -/
theorem tabulate_getElem! {n id : Nat} (hid : id < n) :
    (∀ j, j < id → (tabulate step id)[j]? = (tabulate step n)[j]?) ∧
      (tabulate step n)[id]! = step (tabulate step id) id := by
  refine ⟨fun j hj => (tabulate_getElem?_of_lt hj (Nat.le_of_lt hid)).symm, ?_⟩
  have hget := tabulate_getElem?_self (step := step) hid
  rw [getElem!_pos (tabulate step n) id (by rw [size_tabulate]; exact hid)]
  rw [Array.getElem?_eq_getElem (by rw [size_tabulate]; exact hid)] at hget
  exact Option.some.inj hget

/-- A step that reads only the entries of earlier nodes below `n`: its table of size `n` is its
fixpoint at every node. -/
theorem tabulate_fix {n : Nat}
    (hstep : ∀ (c c' : Array (Option β)) (id : Nat), id < n → (∀ j, j < id → c[j]? = c'[j]?) →
      step c id = step c' id) :
    ∀ id, id < n → (tabulate step n)[id]! = step (tabulate step n) id := by
  intro id hid
  obtain ⟨hlt, hself⟩ := tabulate_getElem! (step := step) hid
  rw [hself]
  exact hstep _ _ id hid hlt

end Tabulate

end TorchLean.CROWN.Op
