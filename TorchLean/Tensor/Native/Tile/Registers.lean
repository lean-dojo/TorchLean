/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Tile
import Mathlib.Tactic.FinCases

/-!
# Register output tiles

The einsum output planner evaluates neighboring output entries as one tile whose lane totals are
separate scalar loop arguments. `outputTiles` lists the tile widths the planner may choose, and
`generate_output_tiles` emits for each of them the tile kernels and the theorems refining them to
the width-polymorphic `updateTile` of `TorchLean.Tensor.Native.Tile`.

Every lane observes contracted coordinates in the same row-major order as the scalar lowering, so
tiling requires no reassociation, commutativity, or distributivity argument.
-/

public meta section

namespace TorchLean.Native

open Lean

/-- The declaration `Tile{width}.leaf` among the generated tile declarations. -/
def outputTileName (width : Nat) (leaf : String) : Name :=
  .str (.str `TorchLean.Native s!"Tile{width}") leaf

/--
A register tile width offered to the einsum output planner, its cost threshold, and the names of
the declarations `generate_output_tiles` emits for it.
-/
structure OutputTile where
  /-- Number of lanes. Each lane total is a separate scalar loop argument. -/
  width : Nat
  /-- Fewest contraction terms for which a flat contraction runs on this tile's scalar loop. -/
  minimumFlatTerms : Nat
  /-- The lane selector `Tile{width}.select`. -/
  select : Name := outputTileName width "select"
  /-- The theorem `Tile{width}.coordinateSum_select`. -/
  coordinateSumSelect : Name := outputTileName width "coordinateSum_select"
  /-- The tile update `Tile{width}.update`. -/
  update : Name := outputTileName width "update"
  /-- The theorem `Tile{width}.update_eq_updateTile`. -/
  updateEqUpdateTile : Name := outputTileName width "update_eq_updateTile"
  /-- The theorem `Tile{width}.ofFn_select_congr`. -/
  ofFnSelectCongr : Name := outputTileName width "ofFn_select_congr"
  /-- The tile append `Tile{width}.push`. -/
  push : Name := outputTileName width "push"
  /-- The theorem `Tile{width}.push_ofFn`. -/
  pushOfFn : Name := outputTileName width "push_ofFn"
  /-- The theorem `Tile{width}.fin_foldl` unrolling a finite fold over the lanes. -/
  finFoldl : Name := outputTileName width "fin_foldl"
  /-- The scalar-state sum `Tile{width}.sum`. -/
  sum : Name := outputTileName width "sum"
  /-- The theorem `Tile{width}.sum_eq_finFoldl`. -/
  sumEqFinFoldl : Name := outputTileName width "sum_eq_finFoldl"
  /-- The appending sum `Tile{width}.sumPush`. -/
  sumPush : Name := outputTileName width "sumPush"
  /-- The theorem `Tile{width}.sumPush_eq_push`. -/
  sumPushEqPush : Name := outputTileName width "sumPush_eq_push"
  /-- The finalizing appending sum `Tile{width}.sumFinalizePush`. -/
  sumFinalizePush : Name := outputTileName width "sumFinalizePush"
  /-- The theorem `Tile{width}.sumFinalizePush_eq_push`. -/
  sumFinalizePushEqPush : Name := outputTileName width "sumFinalizePush_eq_push"

/--
The tile widths of the planner's cost policy, narrowest first. Wider tiles share more
contraction-coordinate work per step and keep more totals live.
-/
def outputTiles : List OutputTile :=
  [{ width := 4, minimumFlatTerms := 32 }, { width := 8, minimumFlatTerms := 128 }]

/-- The tile of the planner's cost policy with the given width. -/
def outputTile? (width : Nat) : Option OutputTile :=
  outputTiles.find? (·.width == width)

end TorchLean.Native

end

meta section

namespace TorchLean.Native

open Lean

/-- The declarations of one register tile, in dependency order. -/
def outputTileCommands (tile : OutputTile) : MacroM (Array (TSyntax `command)) := do
  let width := tile.width
  let u : TSyntax `level := ⟨(mkIdent `u).raw⟩
  let widthLit : Term := quote width
  let lanes := Array.range width
  let lastLane := width - 1
  let laneLit : Array Term := lanes.map fun lane => quote lane
  let indexed (stem : String) : Array Ident :=
    lanes.map fun lane => mkIdent (.mkSimple s!"{stem}{lane}")
  let value := indexed "value"
  let total := indexed "total"
  let finalize := indexed "finalize"
  let reference := indexed "reference"
  let ident (name : Name) : Ident := mkIdent (name.replacePrefix `TorchLean.Native .anonymous)
  let identDot (name : Name) (field : String) : Ident :=
    mkIdent (.str (name.replacePrefix `TorchLean.Native .anonymous) field)
  let doc (text : String) : TSyntax ``Parser.Command.docComment :=
    ⟨mkNode ``Parser.Command.docComment #[mkAtom "/--", mkAtom s!" {text} -/"]⟩
  let pushChain (emitted : Array Term) : MacroM Term := do
    let mut body ← `(storage.push output $(emitted[lastLane]!))
    for lane in (List.range lastLane).reverse do
      body ← `(let output := storage.push output $(emitted[lane]!); $body)
    return body
  let loop := outputTileName width "sumLoop"
  let pushLoop := outputTileName width "sumPushLoop"
  let finalizeLoop := outputTileName width "sumFinalizePushLoop"
  let loopLemma := outputTileName width "sumLoop_eq_finFoldlLoop"
  let updateGet := outputTileName width "update_get"
  let indexFamily ← `((index : USize) → index.toNat < length → α)
  let initials ← lanes.mapM fun _ => `(initial)
  let step ← lanes.mapM fun lane =>
    `(($(total[lane]!):ident + $(value[lane]!):ident index _hIndexNat))
  let updateLanes ← value.mapM fun lane => `(($lane index hIndex))
  let initialLanes ← lanes.mapM fun lane => `(initial[$(laneLit[lane]!)])
  let selectAlternatives ← lanes.mapM fun lane => do
    let pattern ← if lane == lastLane then `(_) else pure laneLit[lane]!
    `(Parser.Term.matchAltExpr| | $pattern:term => $(value[lane]!):ident)
  let selectTile ← `(command|
    $(doc s!"Select one of {width} values by a statically bounded lane index."):docComment
    def $(ident tile.select) {α : Type $u} ($value* : α) : Fin $widthLit → α
      | lane =>
          match lane.val with
          $selectAlternatives:matchAlt*)
  let atCoordinate ← value.mapM fun lane => `(($lane coordinate))
  let laneSums ← value.mapM fun lane =>
    `((Shape.Coord.foldl shape (fun total coordinate => total + $lane coordinate) initial))
  let coordinateSum ← `(command|
    $(doc s!"Selecting a {width}-lane contraction after summing each lane agrees with summing the \
      selected lane, in the coordinate order of `Shape.Coord.foldl`."):docComment
    theorem $(ident tile.coordinateSumSelect)
        {α : Type $u} [Add α] (shape : Shape) ($value* : Shape.Coord shape → α) (initial : α) :
        (fun lane =>
          Shape.Coord.foldl shape
            (fun total coordinate =>
              total + $(ident tile.select) $atCoordinate* lane)
            initial) =
          $(ident tile.select) $laneSums* := by
      funext lane
      fin_cases lane <;> rfl)
  let mut updateBody ←
    `(totals.set $(laneLit[lastLane]!) (totals[$(laneLit[lastLane]!)] + $(value[lastLane]!)))
  for lane in (List.range lastLane).reverse do
    updateBody ←
      `(let totals := totals.set $(laneLit[lane]!) (totals[$(laneLit[lane]!)] + $(value[lane]!));
        $updateBody)
  let updateTile ← `(command|
    $(doc s!"Update the proof-facing state of a {width}-lane tile. For concrete contractions, \
      `compileCoordinateFold` emits `{(ident tile.sum).getId}` for this update, whose \
      totals are separate loop arguments, and the output pass fuses an immediately consumed \
      result into `{(ident tile.sumPush).getId}`."):docComment
    @[inline] def $(ident tile.update)
        {α : Type $u} [Add α] (totals : Vector α $widthLit) ($value* : α) : Vector α $widthLit :=
      $updateBody)
  let pushTileBody ← pushChain (← lanes.mapM fun lane => `(values[$(laneLit[lane]!)]))
  let pushTile ← `(command|
    $(doc s!"Append one completed {width}-lane tile to a row-major output buffer."):docComment
    @[inline] def $(ident tile.push)
        {α : Type $u} [storage : Storage α]
        (output : storage.Buffer) (values : Vector α $widthLit) : storage.Buffer :=
      $pushTileBody)
  let equalities ← lanes.mapM fun lane =>
    `(bracketedBinder|
      ($(mkIdent (.mkSimple s!"h{lane}")) : $(value[lane]!):ident = $(reference[lane]!):ident))
  let congruence ← `(command|
    $(doc s!"Pointwise equal lane values determine equal {width}-lane vectors."):docComment
    theorem $(ident tile.ofFnSelectCongr)
        {α : Type $u} ($(value ++ reference)* : α) $equalities* :
        Vector.ofFn ($(ident tile.select) $value*) =
          Vector.ofFn ($(ident tile.select) $reference*) := by
      subst_vars
      rfl)
  let pushedValues ← pushChain (value.map (·))
  let pushOfFn ← `(command|
    $(doc s!"Appending a {width}-lane function vector emits its values in lane order."):docComment
    theorem $(ident tile.pushOfFn)
        {α : Type $u} [storage : Storage α] (output : storage.Buffer) ($value* : α) :
        $(ident tile.push) output (Vector.ofFn ($(ident tile.select) $value*)) =
          $pushedValues := by
      simp only [$(ident tile.push):ident, Vector.getElem_ofFn, $(ident tile.select):ident])
  let getLemma ← `(command|
    $(doc s!"Projecting a {width}-lane update gives the corresponding scalar update."):docComment
    theorem $(ident updateGet)
        {α : Type $u} [Add α] (totals : Vector α $widthLit) ($value* : α) (lane : Fin $widthLit) :
        ($(ident tile.update) totals $value*)[lane] =
          totals[lane] + $(ident tile.select) $value* lane := by
      fin_cases lane <;>
        simp only [Fin.getElem_fin, $(ident tile.update):ident, $(ident tile.select):ident,
          Vector.getElem_set, Nat.reduceEqDiff, reduceIte])
  let updateRefinement ← `(command|
    $(doc s!"The {width}-register update implements the width-polymorphic tile update."):docComment
    theorem $(ident tile.updateEqUpdateTile)
        {α : Type $u} [Add α] (totals : Vector α $widthLit) ($value* : α) :
        $(ident tile.update) totals $value* =
          updateTile totals ($(ident tile.select) $value*) := by
      apply Vector.ext
      intro lane hLane
      have hTile := $(ident updateGet) totals $value* ⟨lane, hLane⟩
      have hGeneric := updateTile_get totals ($(ident tile.select) $value*) ⟨lane, hLane⟩
      simp only [Fin.getElem_fin] at hTile hGeneric
      rw [hTile, hGeneric])
  let sumLoop ← `(command|
    $(doc s!"Run a native finite sum while carrying {width} lane totals as separate scalar \
      arguments, advancing one contraction coordinate per step."):docComment
    @[specialize] def $(ident loop)
        {α : Type $u} [Add α] (length : Nat) ($value* : $indexFamily)
        (bound : USize) (hBound : bound.toNat = length)
        (index : USize) ($total* : α) : Vector α $widthLit :=
      if hIndex : index < bound then
        have _hIndexNat : index.toNat < length :=
          Nat.lt_of_lt_of_eq (USize.lt_iff_toNat_lt.mp hIndex) hBound
        $(ident loop) length $value* bound hBound (index + 1) $step*
      else
        #v[$total,*]
    termination_by length - index.toNat
    decreasing_by
      have _hIndexNat : index.toNat < bound.toNat :=
        USize.lt_iff_toNat_lt.mp hIndex
      have hNextFits : index.toNat + 1 < USize.size := by
        have hBoundFits := USize.toNat_lt_size bound
        lia
      simp only [USize.toNat_add, USize.toNat_one,
        Nat.mod_eq_of_lt hNextFits]
      lia)
  let sum ← `(command|
    $(doc s!"Sum {width} lanes over a native finite interval without vector operations inside \
      the loop."):docComment
    @[inline] def $(ident tile.sum)
        {α : Type $u} [Add α] (length : Nat)
        (bound : USize) (hBound : bound.toNat = length)
        ($value* : $indexFamily)
        (initial : Vector α $widthLit) : Vector α $widthLit :=
      $(ident loop) length $value* bound hBound 0 $initialLanes*)
  let pushedTotals ← pushChain (total.map (·))
  let sumPushLoop ← `(command|
    $(doc s!"Run {width} scalar accumulators and append their final values directly to the \
      output buffer it receives."):docComment
    @[specialize] def $(ident pushLoop)
        {α : Type $u} [storage : Storage α] [Add α] (length : Nat)
        ($value* : $indexFamily)
        (bound : USize) (hBound : bound.toNat = length)
        (output : storage.Buffer) (index : USize)
        ($total* : α) : storage.Buffer :=
      if hIndex : index < bound then
        have _hIndexNat : index.toNat < length :=
          Nat.lt_of_lt_of_eq (USize.lt_iff_toNat_lt.mp hIndex) hBound
        $(ident pushLoop) length $value* bound hBound output (index + 1) $step*
      else
        $pushedTotals
    termination_by length - index.toNat
    decreasing_by
      have _hIndexNat : index.toNat < bound.toNat :=
        USize.lt_iff_toNat_lt.mp hIndex
      have hNextFits : index.toNat + 1 < USize.size := by
        have hBoundFits := USize.toNat_lt_size bound
        lia
      simp only [USize.toNat_add, USize.toNat_one,
        Nat.mod_eq_of_lt hNextFits]
      lia)
  let sumPush ← `(command|
    $(doc s!"Sum {width} lanes and append them without materializing the result \
      vector."):docComment
    @[inline] def $(ident tile.sumPush)
        {α : Type $u} [storage : Storage α] [Add α] (length : Nat)
        (bound : USize) (hBound : bound.toNat = length)
        ($value* : $indexFamily)
        (output : storage.Buffer) (initial : α) : storage.Buffer :=
      $(ident pushLoop) length $value* bound hBound output 0 $initials*)
  let loopFold ← `(finFoldlLoop length
      (fun totals index hIndex => $(ident tile.update) totals $updateLanes*)
      bound hBound index #v[$total,*])
  let loopRefinement ← `(command|
    $(doc s!"Starting from any native index and {width} scalar totals, the scalar-state loop \
      agrees with the corresponding vector-state native fold."):docComment
    private theorem $(ident loopLemma)
        {α : Type $u} [Add α] (length : Nat) ($value* : $indexFamily)
        (bound : USize) (hBound : bound.toNat = length)
        (index : USize) ($total* : α) :
        $(ident loop) length $value* bound hBound index $total* = $loopFold := by
      refine $(identDot loop "induct") length $value* bound hBound
        (motive := fun index $total* =>
          $(ident loop) length $value* bound hBound index $total* = $loopFold)
        ?_ ?_ index $total*
      · intro index $total* hIndex _hIndexNat inductionHypothesis
        rw [$(identDot loop "eq_1"):ident, dite_eq_left hIndex]
        rw [finFoldlLoop.eq_1, dite_eq_left hIndex]
        simpa [$(ident tile.update):ident] using inductionHypothesis
      · intro index $total* hIndex
        rw [$(identDot loop "eq_1"):ident, dite_eq_right hIndex]
        rw [finFoldlLoop.eq_1, dite_eq_right hIndex])
  let sumRefinement ← `(command|
    $(doc s!"The {width}-scalar native loop is exactly the ordinary native fold using \
      `{(ident tile.update).getId}`, so every lane keeps its reduction order."):docComment
    theorem $(ident tile.sumEqFinFoldl)
        {α : Type $u} [Add α] (length : Nat)
        (bound : USize) (hBound : bound.toNat = length)
        ($value* : $indexFamily)
        (initial : Vector α $widthLit) :
        $(ident tile.sum) length bound hBound $value* initial =
          Native.finFoldl length bound hBound
            (fun totals index hIndex => $(ident tile.update) totals $updateLanes*)
            initial := by
      rw [$(ident tile.sum):ident, Native.finFoldl]
      rw [$(ident loopLemma):ident]
      have hInitial : (#v[$initialLanes,*] : Vector α $widthLit) = initial := by
        apply Vector.ext
        intro lane hLane
        have hLanes :
            ∀ lane : Fin $widthLit,
              (#v[$initialLanes,*] : Vector α $widthLit)[lane] = initial[lane] := by
          intro lane
          fin_cases lane <;> rfl
        exact hLanes ⟨lane, hLane⟩
      rw [hInitial])
  let pushRefinement ← `(command|
    $(doc s!"Directly appending the scalar loop totals is exactly \
      `{(ident tile.push).getId}` applied to the vector-valued loop result."):docComment
    theorem $(ident tile.sumPushEqPush)
        {α : Type $u} [storage : Storage α] [Add α] (length : Nat)
        (bound : USize) (hBound : bound.toNat = length)
        ($value* : $indexFamily)
        (output : storage.Buffer) (initial : α) :
        $(ident tile.sumPush) length bound hBound $value* output initial =
          $(ident tile.push) output (
            $(ident tile.sum) length bound hBound $value*
              (Vector.replicate $widthLit initial)) := by
      rw [$(ident tile.sumPush):ident, $(ident tile.sum):ident]
      refine $(identDot pushLoop "induct") length $value* bound hBound output
        (motive := fun index $total* =>
          $(ident pushLoop) length $value* bound hBound output index $total* =
            $(ident tile.push) output (
              $(ident loop) length $value* bound hBound index $total*))
        ?_ ?_ 0 $initials*
      · intro index $total* hIndex _hIndexNat inductionHypothesis
        rw [$(identDot pushLoop "eq_1"):ident, dite_eq_left hIndex]
        rw [$(identDot loop "eq_1"):ident, dite_eq_left hIndex]
        exact inductionHypothesis
      · intro index $total* hIndex
        rw [$(identDot pushLoop "eq_1"):ident, dite_eq_right hIndex]
        rw [$(identDot loop "eq_1"):ident, dite_eq_right hIndex]
        rfl)
  let finalizedPush ← pushChain (← lanes.mapM fun lane =>
    `(($(finalize[lane]!):ident $(total[lane]!):ident)))
  let finalizePushLoop ← `(command|
    $(doc s!"Run {width} scalar accumulators, apply one terminal function to each total, and \
      append the results to the output buffer it receives. The terminal functions are never called \
      inside the contraction loop, which is the path used when semiring laws move \
      contraction-invariant factors outside a shared tiled sum."):docComment
    @[specialize] def $(ident finalizeLoop)
        {α : Type $u} [storage : Storage α] [Add α] (length : Nat)
        ($value* : $indexFamily)
        ($finalize* : α → α)
        (bound : USize) (hBound : bound.toNat = length)
        (output : storage.Buffer) (index : USize)
        ($total* : α) : storage.Buffer :=
      if hIndex : index < bound then
        have _hIndexNat : index.toNat < length :=
          Nat.lt_of_lt_of_eq (USize.lt_iff_toNat_lt.mp hIndex) hBound
        $(ident finalizeLoop) length $value* $finalize* bound hBound output (index + 1) $step*
      else
        $finalizedPush
    termination_by length - index.toNat
    decreasing_by
      have _hIndexNat : index.toNat < bound.toNat :=
        USize.lt_iff_toNat_lt.mp hIndex
      have hNextFits : index.toNat + 1 < USize.size := by
        have hBoundFits := USize.toNat_lt_size bound
        lia
      simp only [USize.toNat_add, USize.toNat_one,
        Nat.mod_eq_of_lt hNextFits]
      lia)
  let finalizePush ← `(command|
    $(doc s!"Sum {width} lanes and apply their terminal functions directly at the output buffer \
      boundary."):docComment
    @[inline] def $(ident tile.sumFinalizePush)
        {α : Type $u} [storage : Storage α] [Add α] (length : Nat)
        (bound : USize) (hBound : bound.toNat = length)
        ($value* : $indexFamily)
        ($finalize* : α → α)
        (output : storage.Buffer) (initial : α) : storage.Buffer :=
      $(ident finalizeLoop) length $value* $finalize* bound hBound output 0 $initials*)
  let finalizedLanes ← lanes.mapM fun lane =>
    `(($(finalize[lane]!):ident totals[$(laneLit[lane]!)]))
  let finalizeRefinement ← `(command|
    $(doc s!"The finalized {width}-lane append is exactly the vector-valued contraction followed \
      by lane-wise finalization and `{(ident tile.push).getId}`."):docComment
    theorem $(ident tile.sumFinalizePushEqPush)
        {α : Type $u} [storage : Storage α] [Add α] (length : Nat)
        (bound : USize) (hBound : bound.toNat = length)
        ($value* : $indexFamily)
        ($finalize* : α → α)
        (output : storage.Buffer) (initial : α) :
        $(ident tile.sumFinalizePush) length bound hBound $value* $finalize* output
            initial =
          let totals :=
            $(ident tile.sum) length bound hBound $value*
              (Vector.replicate $widthLit initial);
          $(ident tile.push) output <| Vector.ofFn <|
            $(ident tile.select) $finalizedLanes* := by
      rw [$(ident tile.sumFinalizePush):ident, $(ident tile.sum):ident]
      refine $(identDot finalizeLoop "induct") length $value* $(finalize.pop)* bound hBound output
        (motive := fun index $total* =>
          $(ident finalizeLoop) length $value* $finalize* bound hBound output index $total* =
            let totals := $(ident loop) length $value* bound hBound index $total*;
            $(ident tile.push) output <| Vector.ofFn <|
              $(ident tile.select) $finalizedLanes*)
        ?_ ?_ 0 $initials*
      · intro index $total* hIndex _hIndexNat inductionHypothesis
        rw [$(identDot finalizeLoop "eq_1"):ident, dite_eq_left hIndex]
        rw [$(identDot loop "eq_1"):ident, dite_eq_left hIndex]
        exact inductionHypothesis
      · intro index $total* hIndex
        rw [$(identDot finalizeLoop "eq_1"):ident, dite_eq_right hIndex]
        rw [$(identDot loop "eq_1"):ident, dite_eq_right hIndex]
        simp [$(ident tile.push):ident, $(ident tile.select):ident])
  let mut unrolled ← `(initial)
  for lane in lanes do
    unrolled ← `(step $unrolled $(laneLit[lane]!))
  let finFoldl ← `(command|
    $(doc s!"A {width}-element finite fold is the corresponding sequence of {width} \
      updates."):docComment
    theorem $(ident tile.finFoldl)
        {α : Type $u} (step : α → Fin $widthLit → α) (initial : α) :
        Fin.foldl $widthLit step initial = $unrolled := by
      simp only [Fin.foldl_succ, Fin.foldl_zero]
      congr)
  return #[selectTile, coordinateSum, updateTile, pushTile, congruence, pushOfFn, getLemma,
    updateRefinement, sumLoop, sum, sumPushLoop, sumPush, loopRefinement, sumRefinement,
    pushRefinement, finalizePushLoop, finalizePush, finalizeRefinement, finFoldl]

/--
`generate_output_tiles` declares, for every tile of `outputTiles`, its lane selector, tile update
and append, scalar-state loops, and the theorems refining them to `updateTile`.
-/
syntax (name := generateOutputTiles) "generate_output_tiles" : command

macro_rules
  | `(generate_output_tiles) => do
      let mut commands : Array Syntax := #[]
      for tile in outputTiles do
        commands := commands ++ (← outputTileCommands tile).map (·.raw)
      return mkNullNode commands

end TorchLean.Native

end

@[expose] public section

namespace TorchLean.Native

universe u

generate_output_tiles

end TorchLean.Native
