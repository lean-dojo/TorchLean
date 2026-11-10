/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.NN.RL.Boundary.Json
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public meta import TorchLean.NN.Widget.UI
public import TorchLean.NN.Widget.UI
import ProofWidgets.Component.HtmlDisplay
public import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.SetLike

/-!
# RL Boundary Rollout Viewer

`#rl_boundary_rollout_file_view path, contract, maxErrors` is the debugger for the trust boundary
of TorchLean RL. It reads a Gymnasium-style rollout JSON file, such as one written by
`scripts/rl/gymnasium_server.py --out PATH`, and checks every transition against a Lean
`TorchLean.RL.Boundary.Contract`. The report gives the number of transitions, how many passed and
how many failed, and the first `maxErrors` error messages with their indices.

The viewer is separate from the PPO views because an exported rollout carries no actor
log-probabilities or value predictions, so PPO loss inspection belongs in the training examples.
-/

namespace TorchLean.Widget

public meta section

open scoped ProofWidgets.Jsx

namespace RL
namespace Boundary

open TorchLean.RL.Boundary

/-- Count accepted and rejected transitions, keeping the rejection messages with their index. -/
private def summarize {obsShape : Shape} {nActions : Nat}
    (xs : Array (Except String (Transition obsShape nActions))) :
    Nat × Nat × Array (Nat × String) :=
  Id.run do
    let mut okCount : Nat := 0
    let mut errCount : Nat := 0
    let mut errs : Array (Nat × String) := #[]
    for i in [0:xs.size] do
      match xs[i]! with
      | .ok _ =>
          okCount := okCount + 1
      | .error msg =>
          errCount := errCount + 1
          errs := errs.push (i, msg)
    return (okCount, errCount, errs)

/-- Load the rollout at `path`, check each transition against `c` and render the report with at
most `maxErrors` error messages. A file that fails to load gives an error panel with the
exception text. -/
def rolloutHtml {obsShape : Shape} {nActions : Nat}
    (path : System.FilePath)
    (c : Contract obsShape nActions)
    (maxErrors : Nat := 8) :
    IO ProofWidgets.Html := do
  try
    let xs ← TorchLean.RL.Boundary.loadRolloutAll (obsShape := obsShape) (nActions := nActions)
      (path := path.toString) c
    let (okCount, errCount, errs) := summarize (obsShape := obsShape) (nActions := nActions) xs
    let shown := errs.take maxErrors
    let more := errs.size - shown.size

    let header : ProofWidgets.Html :=
      <div className="torchlean-panel">
        <h3>RL rollout boundary report</h3>
        <div>{UI.monospace path.toString}</div>
        <ul>
          <li>transitions: {UI.monospace s!"{xs.size}"}</li>
          <li>ok: {UI.monospace s!"{okCount}"}</li>
          <li>errors: {UI.monospace s!"{errCount}"}</li>
        </ul>
      </div>

    if xs.isEmpty then
      pure <|
        <div>
          {header}
          <div className="torchlean-warn">
            {.text "No transitions were found; no boundary contract checks ran."}
          </div>
        </div>
    else if errCount = 0 then
      pure <|
        <div>
          {header}
          <div className="torchlean-ok">
            {.text "All transitions passed the boundary contract."}
          </div>
        </div>
    else
      pure <|
        <div>
          {header}
          <div className="torchlean-warn">
            {.text "Some transitions failed contract checks. First errors:"}
          </div>
          <ol>
            {... shown.map (fun (i, msg) =>
              <li>
                <div>{UI.monospace s!"index={i}"}</div>
                <pre>{.text msg}</pre>
              </li>)}
          </ol>
          {
            if more = 0 then
              (<div/>)
            else
              (<div>{.text s!"({more} more errors omitted)"}</div>)
          }
        </div>
  catch e =>
    pure <|
      <div className="torchlean-error">
        <h3>{.text "RL rollout boundary report (error)"}</h3>
        <div>{UI.monospace path.toString}</div>
        <pre>{.text (toString e)}</pre>
      </div>

/-- `#rl_boundary_rollout_file_view path, contract, maxErrors` shows the `rolloutHtml` report of the
rollout file `path` checked against `contract`, with at most `maxErrors` error messages. -/
macro "#rl_boundary_rollout_file_view " path:term ", " contract:term ", " maxErrors:term :
    command =>
  UI.canonicalCommand <$> `(
    #html (rolloutHtml (path := $path) (c := $contract) (maxErrors := $maxErrors))
  )

end Boundary
end RL

end
end TorchLean.Widget
