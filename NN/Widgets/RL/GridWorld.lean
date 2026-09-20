/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Spec.RL.Envs.GridWorld
public import Mathlib.Algebra.Order.AbsoluteValue.Basic
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Finset.Attr
public meta import NN.Runtime.RL.Artifacts.GridWorld.Path
public meta import NN.Runtime.RL.Artifacts.GridWorld.Policy
public meta import NN.Widgets.Core.UI
public import NN.Widgets.Core.UI
import NN.Tensor.Constructors
import ProofWidgets.Component.HtmlDisplay

/-!
# GridWorld Widgets

This module provides small infoview widgets for TorchLean's Lean-native GridWorld environment
(`NN.Spec.RL.Envs.GridWorld`):

- `#gridworld_view gw, pos` renders the grid, highlighting `start`, `goal`, and the current
  position.
- `#gridworld_policy_view gw, policy` renders a simple arrow policy overlay.
- `#gridworld_path_view gw, path` renders a grid with a rollout path (first-visit indices).
- `#gridworld_policy_file_view gw, path` renders a saved before/after greedy policy snapshot.
- `#gridworld_path_file_view gw, path` renders a saved before/after episode path snapshot.

These widgets do not run training loops; instead, they help you *see* the state-space objects that
RL algorithms manipulate.

## Main definitions

- `gridworldHtml`: base grid renderer with start/goal/current position highlights.
- `gridworldPolicyHtml`: policy overlay using direction arrows.
- `gridworldPathHtml`: path renderer using first-visit indices.
- `gridworldPolicyDiffHtml` / `gridworldPathDiffHtml`: before/after artifact comparison panels.
- `#gridworld_*_view`: command entry points for interactive use.
-/

namespace NN.Widgets

public meta section

open scoped ProofWidgets.Jsx

open Spec TorchLean
open Spec.RL
open Spec.RL.Envs
open UI

namespace RL
namespace GridWorld

/-- Build a CSS style object from key/value pairs. -/
private def styleObj (xs : List (String × String)) : Lean.Json :=
  Lean.Json.mkObj (xs.map (fun (k, v) => (k, Lean.Json.str v)))

/-- One grid square: a fixed-size rounded box carrying a single label. -/
private def cell (label : String) (bg : String) : ProofWidgets.Html :=
  <div style={styleObj [
    ("width", "34px"),
    ("height", "34px"),
    ("display", "flex"),
    ("align-items", "center"),
    ("justify-content", "center"),
    ("border", "1px solid var(--vscode-panel-border, #e0e0e0)"),
    ("border-radius", "8px"),
    ("background", bg),
    ("font-family",
      "var(--vscode-editor-font-family, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \
      monospace)"),
    ("font-size", "13px"),
    ("font-weight", "600"),
    ("user-select", "none")
  ]}>{.text label}</div>

/-- Format a grid position as `(row,col)` for the header pills. -/
private def posString {width height : Nat} (p : GridWorld.State width height) : String :=
  s!"({p.1.val},{p.2.val})"

/-- Lay cells out in a `width`-column CSS grid, which is what makes the rows line up. -/
private def gridHtml {width : Nat}
    (cells : Array ProofWidgets.Html) : ProofWidgets.Html :=
  let cols : String := s!"repeat({width}, 34px)";
  <div style={styleObj [
    ("display", "grid"),
    ("grid-template-columns", cols),
    ("gap", "6px")
  ]}>
    {...cells}
  </div>

/-- Return the first index of `x` in `xs`, if present. -/
private def firstIndex? {α : Type} [DecidableEq α] (xs : Array α) (x : α) : Option Nat :=
  let rec go (i : Nat) : Option Nat :=
    if h : i < xs.size then
      if xs[i] = x then
        some i
      else
        go (i + 1)
    else
      none
  go 0

/-!
## Renderers
-/

/-- Render a GridWorld state as a grid with start/goal/current highlights. -/
def gridworldHtml {width height : Nat}
    (gw : GridWorld width height) (pos : GridWorld.State width height) : ProofWidgets.Html :=
  let rows : List (Fin height) := List.finRange height
  let cols : List (Fin width) := List.finRange width
  let positions : List (GridWorld.State width height) :=
    rows.foldr (fun r acc => (cols.map (fun c => (r, c))) ++ acc) []
  let cells : Array ProofWidgets.Html :=
    positions.toArray.map (fun p =>
      let label : String :=
        if p = pos then "A"
        else if p = gw.start then "S"
        else if p = gw.goal then "G"
        else ""
      let bg : String :=
        if p = pos then "rgba(66, 133, 244, 0.20)"
        else if p = gw.goal then "rgba(255, 200, 0, 0.25)"
        else if p = gw.start then "rgba(0, 200, 100, 0.20)"
        else "rgba(120, 120, 120, 0.05)"
      cell label bg)
  ;
  <div style={json% {"padding": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap",
        "margin-bottom": "10px"}}>
      {pill s!"GridWorld {width}x{height}"} {pill s!"pos={posString pos}"}
      {pill s!"start={posString gw.start}"} {pill s!"goal={posString gw.goal}"}
    </div>
    {gridHtml (width := width) cells}
  </div>

/-- ASCII arrow for an action index, in the `up, down, left, right` order `GridWorld` uses. -/
private def arrowOfAction {nActions : Nat} (a : Fin nActions) : String :=
  if a.1 = 0 then "^" else if a.1 = 1 then "v" else if a.1 = 2 then "<" else ">"

/-- Render a simple policy overlay `π : State → Action` on GridWorld. -/
def gridworldPolicyHtml {width height : Nat}
    (gw : GridWorld width height) (π : GridWorld.State width height → GridWorld.Action) :
    ProofWidgets.Html :=
  let rows : List (Fin height) := List.finRange height
  let cols : List (Fin width) := List.finRange width
  let positions : List (GridWorld.State width height) :=
    rows.foldr (fun r acc => (cols.map (fun c => (r, c))) ++ acc) []
  let cells : Array ProofWidgets.Html :=
    positions.toArray.map (fun p =>
      let label : String :=
        if p = gw.goal then "G" else arrowOfAction (nActions := 4) (π p)
      let bg : String :=
        if p = gw.goal then "rgba(255, 200, 0, 0.25)"
        else if p = gw.start then "rgba(0, 200, 100, 0.20)"
        else "rgba(120, 120, 120, 0.05)"
      cell label bg)
  ;
  <div style={json% {"padding": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap",
        "margin-bottom": "10px"}}>
      {pill s!"GridWorld policy {width}x{height}"} {pill s!"start={posString gw.start}"}
      {pill s!"goal={posString gw.goal}"}
    </div>
    {gridHtml (width := width) cells}
  </div>

/-
Policy snapshots stored on disk (as JSON) record actions as `Nat` indices `0..3` in row-major order.
The widget layer converts them back to a total Lean function `State → Action` for visualization.
-/
/-- Decode action ids from saved artifacts, defaulting to `up` when malformed. -/
private def actionOfNat (n : Nat) : GridWorld.Action :=
  if h : n < 4 then ⟨n, h⟩ else GridWorld.Action.up

/-- Read a saved row-major action array back as a total policy, defaulting outside its range. -/
private def policyOfActionArray {width height : Nat}
    (actions : Array Nat) : GridWorld.State width height → GridWorld.Action :=
  fun pos =>
    let idx : Nat := (GridWorld.encode (width := width) (height := height) pos).val
    if h : idx < actions.size then
      actionOfNat (actions[idx]'h)
    else
      GridWorld.Action.up

/-- Build a grid position from untrusted row and column numbers, or fail when out of range. -/
private def mkPos? {width height : Nat} (row col : Nat) : Option (GridWorld.State width height) :=
  if hRow : row < height then
    if hCol : col < width then
      some (⟨row, hRow⟩, ⟨col, hCol⟩)
    else
      none
  else
    none

/-- Place two panels side by side, wrapping to one column when the panel gets narrow. -/
private def twoUp (a b : ProofWidgets.Html) : ProofWidgets.Html :=
  <div style={styleObj [
    ("display", "grid"),
    ("grid-template-columns", "repeat(auto-fit, minmax(340px, 1fr))"),
    ("gap", "10px")
  ]}>
    <div>{a}</div>
    <div>{b}</div>
  </div>

/-- Render a before/after policy snapshot (loaded from disk) as two GridWorld policy panels. -/
def gridworldPolicyDiffHtml {width height : Nat}
    (gw : GridWorld width height)
    (diff : Runtime.RL.Artifacts.GridWorld.PolicyDiff) :
    ProofWidgets.Html :=
  let expected : Nat := width * height
  let warns0 : Array String := #[]
  let warns1 :=
    if diff.width != width then
      warns0.push s!"width mismatch (file={diff.width}, expected={width})"
    else
      warns0
  let warns2 :=
    if diff.height != height then
      warns1.push s!"height mismatch (file={diff.height}, expected={height})"
    else
      warns1
  let warns3 :=
    if diff.before.size != expected then
      warns2.push s!"before length mismatch (file={diff.before.size}, expected={expected})"
    else
      warns2
  let warns4 :=
    if diff.after.size != expected then
      warns3.push s!"after length mismatch (file={diff.after.size}, expected={expected})"
    else
      warns3
  let warns5 :=
    if !(diff.before.all (fun a => a < 4)) then
      warns4.push "before contains an out-of-range action (expected 0..3)"
    else
      warns4
  let warns :=
    if !(diff.after.all (fun a => a < 4)) then
      warns5.push "after contains an out-of-range action (expected 0..3)"
    else
      warns5

  let beforePol := policyOfActionArray (width := width) (height := height) diff.before
  let afterPol := policyOfActionArray (width := width) (height := height) diff.after
  ;
  <div style={json% {"padding": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap",
        "margin-bottom": "10px"}}>
      {pill "GridWorld policy snapshot"} {pill s!"expected={expected}"} {... warns.map warnBadge}
    </div>
    {twoUp
      (<div>{pill "before"}{gridworldPolicyHtml gw beforePol}</div>)
      (<div>{pill "after"}{gridworldPolicyHtml gw afterPol}</div>)}
  </div>

/-- Render a rollout path (first-visit indices) on GridWorld. -/
def gridworldPathHtml {width height : Nat}
    (gw : GridWorld width height) (path : Array (GridWorld.State width height)) :
    ProofWidgets.Html :=
  let rows : List (Fin height) := List.finRange height
  let cols : List (Fin width) := List.finRange width
  let positions : List (GridWorld.State width height) :=
    rows.foldr (fun r acc => (cols.map (fun c => (r, c))) ++ acc) []
  let cells : Array ProofWidgets.Html :=
    positions.toArray.map (fun p =>
      let idx? := firstIndex? (α := GridWorld.State width height) path p
      let label : String :=
        match idx? with
        | none =>
            if p = gw.start then "S" else if p = gw.goal then "G" else ""
        | some i => toString i
      let bg : String :=
        if p = gw.goal then "rgba(255, 200, 0, 0.25)"
        else if p = gw.start then "rgba(0, 200, 100, 0.20)"
        else if idx?.isSome then "rgba(66, 133, 244, 0.12)"
        else "rgba(120, 120, 120, 0.05)"
      cell label bg)
  ;
  <div style={json% {"padding": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap",
        "margin-bottom": "10px"}}>
      {pill s!"GridWorld path {width}x{height}"} {pill s!"len={path.size}"}
      {pill s!"start={posString gw.start}"} {pill s!"goal={posString gw.goal}"}
    </div>
    {gridHtml (width := width) cells}
  </div>

/-- Render a before/after episode path snapshot (loaded from disk) as two GridWorld path panels. -/
def gridworldPathDiffHtml {width height : Nat}
    (gw : GridWorld width height)
    (diff : Runtime.RL.Artifacts.GridWorld.PathDiff) :
    ProofWidgets.Html :=
  let warns0 : Array String := #[]
  let warns1 :=
    if diff.width != width then
      warns0.push s!"width mismatch (file={diff.width}, expected={width})"
    else
      warns0
  let warns :=
    if diff.height != height then
      warns1.push s!"height mismatch (file={diff.height}, expected={height})"
    else
      warns1
  let beforePath : Array (GridWorld.State width height) :=
    diff.before.filterMap (fun p => mkPos? (width := width) (height := height) p.1 p.2)
  let afterPath : Array (GridWorld.State width height) :=
    diff.after.filterMap (fun p => mkPos? (width := width) (height := height) p.1 p.2)
  let warns :=
    if beforePath.size != diff.before.size then
      warns.push "before path contained out-of-bounds positions (dropped)"
    else
      warns
  let warns :=
    if afterPath.size != diff.after.size then
      warns.push "after path contained out-of-bounds positions (dropped)"
    else
      warns
  ;
  <div style={json% {"padding": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap",
        "margin-bottom": "10px"}}>
      {pill "GridWorld episode path"} {... warns.map warnBadge}
    </div>
    {twoUp
      (<div>{pill "before"}{gridworldPathHtml gw beforePath}</div>)
      (<div>{pill "after"}{gridworldPathHtml gw afterPath}</div>)}
  </div>

/-!
## Commands
-/

/--
Render the GridWorld layout (and a highlighted position) in the infoview.

Usage: `#gridworld_view gw, gw.start`
-/
syntax (name := gridworldViewCmd) "#gridworld_view " term ", " term : command

/--
Render a greedy-policy map for a GridWorld in the infoview.

Usage: `#gridworld_policy_view gw, policy`, where `policy` is a flattened row-major array of
action indices (`0..3`).
-/
syntax (name := gridworldPolicyViewCmd) "#gridworld_policy_view " term ", " term : command

/--
Render a single episode path as a sequence of positions in the infoview.

Usage: `#gridworld_path_view gw, path`, where `path` is an array of `(row, col)` positions.
-/
syntax (name := gridworldPathViewCmd) "#gridworld_path_view " term ", " term : command

/--
Read a saved GridWorld greedy-policy snapshot (`before` vs `after`) from JSON and render it.

This is intended for executable examples or training jobs that write artifacts to disk, for example:
`lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1
--eval-episodes 1 --eval-max-steps 8`.

The JSON schema matches `Runtime.RL.Artifacts.GridWorld.PolicyDiff`.
-/
syntax (name := gridworldPolicyFileViewCmd) "#gridworld_policy_file_view " term ", " term : command

/--
Read a saved GridWorld episode path snapshot (`before` vs `after`) from JSON and render it.

This is intended for executable examples or training jobs that write artifacts to disk, for example:
`lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1
--eval-episodes 1 --eval-max-steps 8`.

The JSON schema matches `Runtime.RL.Artifacts.GridWorld.PathDiff`.
-/
syntax (name := gridworldPathFileViewCmd) "#gridworld_path_file_view " term ", " term : command

macro "#gridworld_view " gw:term ", " pos:term : command =>
  UI.canonicalCommand <$> `(#html (gridworldHtml $gw $pos))

macro "#gridworld_policy_view " gw:term ", " pol:term : command =>
  UI.canonicalCommand <$> `(#html (gridworldPolicyHtml $gw $pol))

macro "#gridworld_path_view " gw:term ", " path:term : command =>
  UI.canonicalCommand <$> `(#html (gridworldPathHtml $gw $path))

macro "#gridworld_policy_file_view " gw:term ", " path:term : command =>
  UI.canonicalCommand <$> `(#html (do
    let p : System.FilePath := $path
    try
      let diff ← Runtime.RL.Artifacts.GridWorld.PolicyDiff.readJson p
      pure (gridworldPolicyDiffHtml $gw diff)
    catch e =>
      pure <|
        <div style={json% {"padding": "10px"}}>
          {warnBadge "gridworld_policy_file_view"}
          <div style={json% {"margin-top": "8px"}}>
            {.text "Could not read policy snapshot file: "}
            {monospace p.toString}
          </div>
          <div style={json% {"margin-top": "6px", "opacity": "0.9"}}>
            {.text "Tip: this file is usually produced by a TorchLean GridWorld PPO run "}
            {.text "(for example `lake exe torchlean ppo_gridworld`). "}
            {.text "You can also override the output path with `--policy <path>`."}
          </div>
          <div style={json% {"margin-top": "6px"}}>{monospace (toString e)}</div>
        </div>))

macro "#gridworld_path_file_view " gw:term ", " path:term : command =>
  UI.canonicalCommand <$> `(#html (do
    let p : System.FilePath := $path
    try
      let diff ← Runtime.RL.Artifacts.GridWorld.PathDiff.readJson p
      pure (gridworldPathDiffHtml $gw diff)
    catch e =>
      pure <|
        <div style={json% {"padding": "10px"}}>
          {warnBadge "gridworld_path_file_view"}
          <div style={json% {"margin-top": "8px"}}>
            {.text "Could not read path snapshot file: "}
            {monospace p.toString}
          </div>
          <div style={json% {"margin-top": "6px", "opacity": "0.9"}}>
            {.text "Tip: this file is usually produced by a TorchLean GridWorld PPO run "}
            {.text "(for example `lake exe torchlean ppo_gridworld`). "}
            {.text "You can also override the output path with `--path <path>`."}
          </div>
          <div style={json% {"margin-top": "6px"}}>{monospace (toString e)}</div>
        </div>))

end GridWorld
end RL

end
end NN.Widgets
