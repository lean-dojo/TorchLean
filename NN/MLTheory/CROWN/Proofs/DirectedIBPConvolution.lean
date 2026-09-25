/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPBasic
public import NN.MLTheory.CROWN.Proofs.ConvEnclosure

/-!
# Rounded matrix and convolution enclosures

The endpoint computations use directed products and sums. Their real targets retain the
runtime's coordinate maps, including batch broadcasting, vector promotion, and zero padding.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open Spec.Conv.Internal
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open NN.MLTheory.CROWN.IntervalLemmas (intervalMul_encloses value_min2 value_max2)
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The real vector described by a graph point's coordinate function. -/
def pointVector (n : Nat) (f : Nat → ℝ) : Tensor ℝ [n] :=
  Tensor.ofFn fun i => f i.val

/-- An enclosed vector remains enclosed when reads outside its shape return zero. -/
theorem rowEncloses_read {box : FlatBox α} {n : Nat} {f : Nat → ℝ}
    (h : RowEncloses box n f) (i : Nat) :
    value (getAtOrZero box.lo [i]) ≤ getAtOrZero (pointVector n f) [i] ∧
      getAtOrZero (pointVector n f) [i] ≤ value (getAtOrZero box.hi [i]) := by
  rcases box with ⟨d, lo, hi⟩
  obtain ⟨hd, h⟩ := h
  change d = n at hd
  subst d
  by_cases hi' : i < n
  · simpa [get_at_or_zero_dim_cons, hi', pointVector, Tensor.getScalar_ofFn] using h ⟨i, hi'⟩
  · simp [get_at_or_zero_dim_cons, hi', LawfulBoundOps.toReal_zero]

/-- Directed accumulation bounds a real sum in the same list order. -/
theorem orderedSum_encloses {ι : Type} (indices : List ι)
    (bounds : ι → α × α) (f : ι → ℝ)
    (h : ∀ i, value (bounds i).1 ≤ f i ∧ f i ≤ value (bounds i).2)
    (acc : α × α) (total : ℝ)
    (hacc : value acc.1 ≤ total ∧ total ≤ value acc.2) :
    value ((indices.foldl
        (fun a i => (BoundOps.addDown a.1 (bounds i).1,
          BoundOps.addUp a.2 (bounds i).2)) acc).1) ≤
        indices.foldl (fun a i => a + f i) total ∧
      indices.foldl (fun a i => a + f i) total ≤
        value ((indices.foldl
          (fun a i => (BoundOps.addDown a.1 (bounds i).1,
            BoundOps.addUp a.2 (bounds i).2)) acc).2) := by
  induction indices generalizing acc total with
  | nil => exact hacc
  | cons i indices ih =>
      apply ih
      exact ⟨(LawfulBoundOps.addDown_le _ _).trans (add_le_add hacc.1 (h i).1),
        (add_le_add hacc.2 (h i).2).trans (LawfulBoundOps.le_addUp _ _)⟩

/-- The actual rounded contraction encloses the real matrix product with the same index maps. -/
theorem binaryMatmulBox_encloses (layout : OpContracts.MatmulDims)
    {left right : FlatBox α} {n m : Nat} {x y : Nat → ℝ}
    (hx : RowEncloses left n x) (hy : RowEncloses right m y) :
    RowEncloses (binaryMatmulBox layout left right) layout.outShape.size
      (fun i => getAtOrZero
        (NN.IR.Graph.matmulFlat layout (pointVector n x) (pointVector m y)) [i]) := by
  refine ⟨rfl, ?_⟩
  intro output
  simp only [binaryMatmulBox, NN.IR.Graph.matmulFlat, read_fin,
    Tensor.getScalar_ofFn]
  apply orderedSum_encloses
  · intro inner
    exact intervalMul_encloses
      (rowEncloses_read hx (layout.leftIndex output.val inner)).1
      (rowEncloses_read hx (layout.leftIndex output.val inner)).2
      (rowEncloses_read hy (layout.rightIndex output.val inner)).1
      (rowEncloses_read hy (layout.rightIndex output.val inner)).2
  · simp only [LawfulBoundOps.toReal_zero, le_refl, and_self]

/-- The binary product's real equation uses the same validated contraction layout. -/
def BinaryMatmulNodeEquation (nodes : Array Node) (dims : Nat → Nat)
    (v : Nat → Nat → ℝ) (id : Nat) : Prop :=
  ∀ p q, nodes[id]!.parents = #[p, q] →
    ∀ left right, nodes[p]? = some left → nodes[q]? = some right →
      ∀ layout, (OpContracts.matmulDims left.outShape right.outShape).toOption = some layout →
        dims id = layout.outShape.size ∧
          ∀ i : Fin layout.outShape.size, v id i.val =
            getAtOrZero (NN.IR.Graph.matmulFlat layout
              (pointVector (dims p) (v p)) (pointVector (dims q) (v q))) [i.val]

omit [LawfulBoundOps α] in
/-- A successful matrix product has either one stored-weight parent or two tensor parents. -/
theorem ibpStep_matmul_parents [NonlinearBoundOps α]
    {nodes : Array Node} {ps : ParamStore α} {boxes : Array (Option (FlatBox α))}
    {id : Nat} {box : FlatBox α}
    (hkind : nodes[id]!.kind = .matmul)
    (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    nodes[id]!.parents.size = 1 ∨ ∃ p q, nodes[id]!.parents = #[p, q] := by
  rw [ibpStepNodeAt?, hkind] at hstep
  dsimp only at hstep
  cases hparents : nodes[id]!.parents with
  | mk parents =>
    cases parents with
    | nil =>
      rw [hparents] at hstep
      cases hstep
    | cons p parents =>
      cases parents with
      | nil => exact Or.inl rfl
      | cons q parents =>
        cases parents with
        | nil => exact Or.inr ⟨p, q, rfl⟩
        | cons r parents =>
          rw [hparents] at hstep
          cases hstep

/-- A successful two-parent matrix product encloses its real contraction. -/
theorem ibpStep_binaryMatmul_encloses [NonlinearBoundOps α]
    {nodes : Array Node} {ps : ParamStore α} {boxes : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id p q : Nat}
    (hkind : nodes[id]!.kind = .matmul) (hparents : nodes[id]!.parents = #[p, q])
    (heq : BinaryMatmulNodeEquation nodes dims v id)
    (hget : ∀ p ∈ nodes[id]!.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p))
    {box : FlatBox α} (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    RowEncloses box (dims id) (v id) := by
  rw [ibpStepNodeAt?, hkind] at hstep
  dsimp only at hstep
  rw [hparents] at hstep
  change (nodes[p]? >>= fun left => nodes[q]? >>= fun right =>
    (boxes[p]?).join >>= fun X => (boxes[q]?).join >>=
      ibpBinaryMatmul? left.outShape right.outShape X) = some box at hstep
  cases hleft : nodes[p]? with
  | none => simp [hleft] at hstep
  | some left =>
    cases hright : nodes[q]? with
    | none => simp [hright] at hstep
    | some right =>
      cases hx : (boxes[p]?).join with
      | none => simp [hleft, hright, hx] at hstep
      | some X =>
        cases hy : (boxes[q]?).join with
        | none => simp [hleft, hright, hx, hy] at hstep
        | some Y =>
          simp only [hleft, hright, hx, hy, Option.bind_eq_bind, Option.bind_some,
            ibpBinaryMatmul?] at hstep
          cases hlayout : (OpContracts.matmulDims left.outShape right.outShape).toOption with
          | none =>
            rw [hlayout] at hstep
            cases hstep
          | some layout =>
            simp only [hlayout] at hstep
            split at hstep
            · obtain rfl := Option.some.inj hstep
              obtain ⟨hdim, hv⟩ := heq p q hparents left right hleft hright layout hlayout
              rw [hdim]
              exact (binaryMatmulBox_encloses layout
                (hget p (by simp [hparents]) X hx)
                (hget q (by simp [hparents]) Y hy)).congr hv
            · cases hstep

/-- Interpret a shaped endpoint box without changing its coordinates. -/
def realBox {s : Shape} (box : Box α s) : Box ℝ s :=
  ⟨Tensor.map value box.lo, Tensor.map value box.hi⟩

/-- Interpretation commutes with total indexing, including the zero-padding branch. -/
theorem getAtOrZero_map_value {s : Shape} (tensor : Tensor α s) (indices : List Nat) :
    getAtOrZero (Tensor.map value tensor) indices = value (getAtOrZero tensor indices) := by
  induction s generalizing indices with
  | scalar =>
    cases indices <;> simp [LawfulBoundOps.toReal_zero]
  | dim n rest ih =>
    cases indices with
    | nil => simp [LawfulBoundOps.toReal_zero]
    | cons i indices =>
      by_cases hi : i < n
      · simpa only [get_at_or_zero_dim_cons, hi, ↓reduceDIte, Tensor.unstack_map] using
          ih (tensor.unstack ⟨i, hi⟩) indices
      · simp [hi, LawfulBoundOps.toReal_zero]

/-- Interpretation commutes with the checked indices used by spatial convolution. -/
theorem multiIndex_get_map_value {dims : List Nat}
    (tensor : Tensor α (Shape.ofList dims)) (index : MultiIndex dims) :
    index.get (Tensor.map value tensor) = value (index.get tensor) := by
  induction dims with
  | nil => simp [MultiIndex.get]
  | cons n dims ih =>
    simpa only [MultiIndex.get, Tensor.unstack_map] using ih (tensor.unstack index.1) index.2

private theorem foldl_rel {A B ι : Type} (relation : A → B → Prop) (indices : List ι)
    {left : A → ι → A} {right : B → ι → B}
    (step : ∀ a b i, relation a b → relation (left a i) (right b i))
    {a : A} {b : B} (initial : relation a b) :
    relation (indices.foldl left a) (indices.foldl right b) := by
  induction indices generalizing a b with
  | nil => exact initial
  | cons i indices ih => exact ih (step a b i initial)

private theorem foldlIndices_rel {A B : Type} (relation : A → B → Prop) (dims : List Nat)
    {left : A → List Nat → A} {right : B → List Nat → B}
    (step : ∀ a b i, relation a b → relation (left a i) (right b i))
    {a : A} {b : B} (initial : relation a b) :
    relation (Spec.Conv.Internal.foldlIndices dims a left)
      (Spec.Conv.Internal.foldlIndices dims b right) := by
  induction dims generalizing left right a b with
  | nil => exact step a b [] initial
  | cons n dims ih =>
    apply foldl_rel relation (List.finRange n) _ initial
    intro a b head h
    exact ih (fun a b tail hab => step a b (head.val :: tail) hab) h

private theorem directed_weight_product {lo hi weight : α} {x : ℝ}
    (hl : value lo ≤ x) (hu : x ≤ value hi) :
    value (if BoundOps.mulDown weight lo > BoundOps.mulDown weight hi
      then BoundOps.mulDown weight hi else BoundOps.mulDown weight lo) ≤ x * value weight ∧
      x * value weight ≤ value (if BoundOps.mulUp weight lo > BoundOps.mulUp weight hi
        then BoundOps.mulUp weight lo else BoundOps.mulUp weight hi) := by
  have hmin := value_min2 (BoundOps.mulDown weight lo) (BoundOps.mulDown weight hi)
  have hmax := value_max2 (BoundOps.mulUp weight lo) (BoundOps.mulUp weight hi)
  simp only [BoundOps.min2, BoundOps.max2, Bool.decide_iff] at hmin hmax
  rw [hmin, hmax, mul_comm x]
  by_cases hw : 0 ≤ value weight
  · exact ⟨(min_le_left _ _).trans ((LawfulBoundOps.mulDown_le _ _).trans
        (mul_le_mul_of_nonneg_left hl hw)),
      ((mul_le_mul_of_nonneg_left hu hw).trans (LawfulBoundOps.le_mulUp _ _)).trans
        (le_max_right _ _)⟩
  · exact ⟨(min_le_right _ _).trans ((LawfulBoundOps.mulDown_le _ _).trans
        (mul_le_mul_of_nonpos_left hu (le_of_not_ge hw))),
      ((mul_le_mul_of_nonpos_left hl (le_of_not_ge hw)).trans
        (LawfulBoundOps.le_mulUp _ _)).trans (le_max_left _ _)⟩

/-- The directed spatial folds enclose exact grouped convolution of the interpreted parameters,
with arbitrary dilation, padding, channel groups and leading batch dimensions. -/
theorem ibpConv_contains_groupedConv
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)
    (box : Box α (leading.concat (Shape.ofList (inC :: inSpatial.data.toList))))
    (input : Tensor ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList))))
    (h : Box.contains (realBox box) input) :
    Box.contains (realBox (ibpConv layer dilation paddingAfter groups leading box))
      (Tensor.mapLeading leading
        (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding) (paddingAfter := paddingAfter)
          groups (Tensor.map value layer.kernel) (Tensor.map value layer.bias)) input) := by
  induction leading with
  | scalar =>
    apply (ConvProof.contains_iff_multiIndex _ _ _).mpr
    intro index
    rcases index with ⟨outChannel, outIndex⟩
    simp only [realBox, multiIndex_get_map_value, ibpConv, ibpConv.mapRows,
      Tensor.mapLeading, groupedConvSpec, groupedConvCoreSpec, convCoreWith,
      convBiasBroadcastDilatedSpec, convBiasBroadcastWith, MultiIndex.get_addSpec,
      MultiIndex.get_dim, MultiIndex.get_generate, Bool.false_eq_true, ↓reduceIte,
      getAtOrZero_map_value]
    constructor
    · apply (LawfulBoundOps.addDown_le _ _).trans
      apply add_le_add_left
      apply foldl_rel (fun a b => value a ≤ b) _ _ (by simp [LawfulBoundOps.toReal_zero])
      intro lo exactValue localChannel hacc
      apply foldlIndices_rel (fun a b => value a ≤ b) _ _ hacc
      intro lo exactValue kernelIndex hacc
      cases hindex : mkDilatedInputIdx? outIndex.toList kernelIndex
          stride.data.toList dilation.data.toList padding.data.toList with
      | none => simpa [hindex] using hacc
      | some inputIndex =>
        have hx := ConvProof.contains_getAtOrZero h
          (((outChannel.val / (outC / groups)) * (inC / groups) +
            localChannel.val) :: inputIndex)
        simp only [realBox, getAtOrZero_map_value] at hx
        simpa only [hindex, getAtOrZero_map_value] using
          (LawfulBoundOps.addDown_le _ _).trans
            (add_le_add hacc (directed_weight_product
              (weight := getAtOrZero layer.kernel
                (outChannel.val ::
                  ((outChannel.val / (outC / groups)) * (inC / groups) +
                    localChannel.val) :: kernelIndex)) hx.1 hx.2).1)
    · apply le_trans _ (LawfulBoundOps.le_addUp _ _)
      apply add_le_add_left
      apply foldl_rel (fun a b => a ≤ value b) _ _ (by simp [LawfulBoundOps.toReal_zero])
      intro exactValue hi localChannel hacc
      apply foldlIndices_rel (fun a b => a ≤ value b) _ _ hacc
      intro exactValue hi kernelIndex hacc
      cases hindex : mkDilatedInputIdx? outIndex.toList kernelIndex
          stride.data.toList dilation.data.toList padding.data.toList with
      | none => simpa [hindex] using hacc
      | some inputIndex =>
        have hx := ConvProof.contains_getAtOrZero h
          (((outChannel.val / (outC / groups)) * (inC / groups) +
            localChannel.val) :: inputIndex)
        simp only [realBox, getAtOrZero_map_value] at hx
        simpa only [hindex, getAtOrZero_map_value] using
          (add_le_add hacc (directed_weight_product
            (weight := getAtOrZero layer.kernel
              (outChannel.val ::
                ((outChannel.val / (outC / groups)) * (inC / groups) +
                  localChannel.val) :: kernelIndex)) hx.1 hx.2).2).trans
            (LawfulBoundOps.le_addUp _ _)
  | dim n rest ih =>
    intro head
    simpa only [realBox, ibpConv, ibpConv.mapRows, Tensor.mapLeading,
      Tensor.unstack_map, Tensor.unstack_dim] using
      ih ⟨box.lo.unstack head, box.hi.unstack head⟩ (input.unstack head) (by
        simpa only [realBox, Tensor.unstack_map] using h head)

/-- Interpreting endpoints commutes with flattening a shaped box. -/
theorem realBox_flattenBox {s : Shape} (box : Box α s) :
    realBox (flattenBox box) = flattenBox (realBox box) := by
  simp only [realBox, flattenBox, Tensor.map, flattenSpec, Tensor.Internal.Rep.map_reshape]

/-- A shaped enclosure gives the corresponding row enclosure after flattening. -/
theorem rowEncloses_flatten {s : Shape} {box : Box α s} {input : Tensor ℝ s}
    (h : Box.contains (realBox box) input) :
    RowEncloses
      ⟨s.size, flattenSpec box.lo, flattenSpec box.hi⟩ s.size
      (fun i => getAtOrZero (flattenSpec input) [i]) := by
  have hf := (ConvProof.box_contains_flatten_iff (realBox box) input).mpr h
  rw [← realBox_flattenBox] at hf
  rw [rowEncloses_iff]
  intro i
  simpa only [realBox, flattenBox, Box.contains, Tensor.unstack_map, Tensor.getScalar,
    Spec.get, Tensor.item_map, read_fin] using hf i

/-- A flat graph enclosure gives a shaped enclosure after the checked reshape. -/
theorem realBox_contains_ibpUnflatten {s : Shape} {box : FlatBox α} {n : Nat}
    {f : Nat → ℝ} (hdim : box.dim = s.size)
    (hn : n = s.size) (h : RowEncloses box n f) :
    Box.contains
      (realBox
        ⟨ibpUnflatten box.dim box.lo hdim, ibpUnflatten box.dim box.hi hdim⟩)
      (unflattenSpec s (pointVector s.size f)) := by
  rcases box with ⟨d, lo, hi⟩
  change d = s.size at hdim
  subst d n
  apply (ConvProof.box_contains_flatten_iff _ _).mp
  rw [← realBox_flattenBox]
  simp only [flattenBox, ibpUnflatten, eq_mp_eq_cast, cast_eq, flattenSpec_unflattenSpec]
  intro i
  simp only [realBox, Box.contains, Tensor.unstack_map, Tensor.item_map]
  change value (lo.getScalar i) ≤ (pointVector s.size f).getScalar i ∧
    (pointVector s.size f).getScalar i ≤ value (hi.getScalar i)
  simpa only [pointVector, read_fin, Tensor.getScalar_ofFn] using h.2 i

/-- The real grouped convolution determined by a stored payload. -/
def convolutionPoint (parameters : ConvParams α) (leading : Shape) (f : Nat → ℝ) :
    Tensor ℝ [(parameters.output leading).size] :=
  flattenSpec (Tensor.mapLeading leading
    (groupedConvSpec (inSpatial := parameters.inputSpatial)
      (stride := parameters.stride) (dilation := parameters.dilation)
      (paddingBefore := parameters.padding) (paddingAfter := parameters.paddingAfter)
      parameters.groups (Tensor.map value parameters.spec.kernel)
        (Tensor.map value parameters.spec.bias))
    (unflattenSpec (parameters.input leading) (pointVector (parameters.input leading).size f)))

/-- The graph's convolution equation states its spatial real semantics. -/
def ConvolutionNodeEquation (nodes : Array Node) (ps : ParamStore α)
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) (id : Nat) (configuration : ConvConfig) : Prop :=
  ∀ p, unaryParent? nodes[id]!.parents = some p →
    ∀ parameters, ps.convCfg[id]? = some parameters →
      ∀ parent, nodes[p]? = some parent →
        ∀ leading,
          planConvTransfer? configuration parameters parent.outShape nodes[id]!.outShape =
            some leading →
          dims p = (parameters.input leading).size ∧
            dims id = (parameters.output leading).size ∧
            ∀ i : Fin (parameters.output leading).size,
              v id i.val = getAtOrZero (convolutionPoint parameters leading (v p)) [i.val]

/-- A successful graph convolution encloses the real grouped convolution at that node. -/
theorem ibpStep_convolution_encloses [NonlinearBoundOps α]
    {nodes : Array Node} {ps : ParamStore α} {boxes : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat} {configuration : ConvConfig}
    (hkind : nodes[id]!.kind = .conv configuration)
    (heq : ConvolutionNodeEquation nodes ps dims v id configuration)
    (hget : ∀ p ∈ nodes[id]!.parents, ∀ B, (boxes[p]?).join = some B →
      RowEncloses B (dims p) (v p))
    {box : FlatBox α} (hstep : ibpStepNodeAt? nodes ps boxes id nodes[id]! = some box) :
    RowEncloses box (dims id) (v id) := by
  rw [ibpStepNodeAt?, hkind] at hstep
  dsimp only at hstep
  cases hp : unaryParent? nodes[id]!.parents with
  | none => simp [hp] at hstep
  | some p =>
    cases hparent : nodes[p]? with
    | none => simp [hp, hparent] at hstep
    | some parent =>
      cases hinput : (boxes[p]?).join with
      | none => simp [hp, hparent, hinput] at hstep
      | some input =>
        simp only [hp, hparent, hinput, Option.bind_eq_bind, Option.bind_some,
          ibpConvNode] at hstep
        cases hparameters : ps.convCfg[id]? with
        | none => simp [hparameters] at hstep
        | some parameters =>
          simp only [hparameters, Option.bind_some] at hstep
          cases hplan : planConvTransfer? configuration parameters parent.outShape
              nodes[id]!.outShape with
          | none => simp only [hplan, Option.bind_none, reduceCtorEq] at hstep
          | some leading =>
            simp only [hplan, Option.bind_some] at hstep
            split at hstep
            next hdim =>
              obtain rfl := Option.some.inj hstep
              obtain ⟨hinputDim, houtputDim, hv⟩ :=
                heq p hp parameters hparameters parent hparent leading hplan
              have hshaped := realBox_contains_ibpUnflatten hdim hinputDim
                (hget p (mem_of_unaryParent?_eq_some hp) input hinput)
              have houtput := ibpConv_contains_groupedConv parameters.spec
                parameters.dilation parameters.paddingAfter parameters.groups leading
                _ _ hshaped
              rw [houtputDim]
              exact (rowEncloses_flatten houtput).congr hv
            next => cases hstep

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
