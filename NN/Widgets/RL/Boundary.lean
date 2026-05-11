/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Runtime.RL.Boundary.Json

public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# RL Boundary Rollout Viewer

This widget is the “trust boundary debugger” for TorchLean RL.

It reads a Gymnasium-style rollout JSON file (typically produced by
`scripts/rl/export_gymnasium_rollout.py`), validates every transition against a Lean-side
`Runtime.RL.Boundary.Contract`, and renders a compact report:

- number of transitions,
- number of valid/invalid transitions,
- the first few error messages (with indices).

This is intentionally separate from PPO-specific viewers: exported rollouts do not contain actor
log-probabilities or value predictions, so PPO loss inspection belongs in the training examples.
-/

namespace NN.Widgets

public meta section

open scoped ProofWidgets.Jsx

namespace RL
namespace Boundary

open _root_.Runtime.RL.Boundary

private def summarize {obsShape : _root_.Spec.Shape} {nActions : Nat}
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

/-- Render a small HTML report for a rollout JSON file under a given contract. -/
def rolloutBoundaryReportHtml {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (path : System.FilePath)
    (c : Contract obsShape nActions)
    (maxErrors : Nat := 8) :
    IO ProofWidgets.Html := do
  try
    let xs ← _root_.Runtime.RL.Boundary.loadRolloutAll (obsShape := obsShape) (nActions := nActions)
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

    if errCount = 0 then
      pure <|
        <div>
          {header}
          <div className="torchlean-ok">{.text "All transitions passed the boundary contract."}</div>
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

syntax (name := rlBoundaryRolloutFileViewCmd)
  "#rl_boundary_rollout_file_view " term ", " term ", " term : command

macro "#rl_boundary_rollout_file_view " path:term ", " contract:term ", " maxErrors:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(
    #html (rolloutBoundaryReportHtml (path := $path) (c := $contract) (maxErrors := $maxErrors))
  )

end Boundary
end RL

end
end NN.Widgets
