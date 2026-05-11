/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Boundary.Json
import Lean.Data.Json

/-!
# Gymnasium Bridge (Client)

This file contains the low-level subprocess client for talking to the Python helper script
`scripts/rl/gymnasium_server.py`.

The client is compact and reusable across examples:

- spawn a Python subprocess hosting an external Gymnasium environment,
- send one JSON object per line to stdin,
- read one JSON object response per line from stdout,
- validate observations/rewards/flags against a Lean-side trust-boundary contract
  (`Runtime.RL.Boundary.Contract`).

We use a small startup handshake (`describe`) to sanity-check that the external environment's
declared spaces match the Lean-side expectations (`obsShape`, `nActions`).

References:
- Gymnasium API docs (`reset`/`step`, `terminated` vs `truncated`): https://gymnasium.farama.org/
- The original Gym API paper (background on the env interface): https://arxiv.org/abs/1606.01540
- Gymnasium source repository (implementation reference): https://github.com/Farama-Foundation/Gymnasium
- Trust-boundary rationale and contract definition: `NN.Runtime.RL.Boundary`.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Gymnasium

open Spec
open Tensor
open Lean
open Json

/-!
## Low-level client

`Client` is a thin wrapper around `IO.Process.Child` that:

- sends a JSON object (one line) to stdin,
- reads one JSON object response from stdout,
- checks `"ok": true` and reports `"error"` strings as `IO.userError`.
-/

/-- Standard stdio configuration for the Gymnasium subprocess protocol. -/
def stdio : IO.Process.StdioConfig :=
  { stdin := .piped, stdout := .piped, stderr := .inherit }

/-- Convenience alias for the subprocess type used by the Gymnasium client. -/
abbrev Child : Type := IO.Process.Child stdio

/--
Typed handle to a running Gymnasium subprocess environment.

`Client` is intentionally small: it only provides request/response JSON helpers and enforces a
Lean-side `Boundary.Contract` on all values received from the external process.
-/
structure Client (obsShape : Shape) (nActions : Nat) where
  /-- The Python subprocess. -/
  child : Child
  /-- Lean-side contract enforced on every transition. -/
  contract : Boundary.Contract obsShape nActions

namespace Client

namespace Internal

/--
Send a request object and return the response object map.

The expected response shape is:
`{"ok": true, ...}` or `{"ok": false, "error": "..."}`

This is deliberately kept in `Client.Internal`. Public callers should use `spawn`, `reset`,
`stepRaw`, `close`, or the higher-level `Session` API, so JSON protocol details stay behind the
Gymnasium trust boundary and out of the curated API facade.
-/
def requestObj {obsShape : Shape} {nActions : Nat}
    (g : Client obsShape nActions) (j : Json) :
    IO (Std.TreeMap.Raw String Json compare) := do
  g.child.stdin.putStrLn (Json.compress j)
  g.child.stdin.flush
  let line ← g.child.stdout.getLine
  if line.isEmpty then
    throw <| IO.userError "Gymnasium: unexpected EOF from server (process exited?)"
  let resp ←
    match Json.parse line with
    | .ok r => pure r
    | .error e => throw <| IO.userError s!"Gymnasium: bad JSON response: {e}\nraw={line}"
  let o ←
    match resp.getObj? with
    | .ok o => pure o
    | .error e => throw <| IO.userError s!"Gymnasium: expected JSON object: {e}\nraw={line}"

  let okField ←
    match o.get? "ok" with
    | some (.bool b) => pure b
    | _ => throw <| IO.userError "Gymnasium: response missing boolean field `ok`"
  unless okField do
    let msg ←
      match o.get? "error" with
      | some (.str s) => pure s
      | _ => throw <| IO.userError "Gymnasium: response had ok=false but missing string field `error`"
    throw <| IO.userError s!"Gymnasium server error: {msg}"

  pure o

/-- Require that a JSON response object contains a given field. -/
def requireField (o : Std.TreeMap.Raw String Json compare) (field : String) : IO Json := do
  match o.get? field with
  | some j => pure j
  | none => throw <| IO.userError s!"Gymnasium: response missing field `{field}`"

end Internal

/--
Spawn the Gymnasium helper subprocess and perform a small handshake (`describe`).

The handshake checks that:
- the external environment reports the expected `n_actions`, and
- the external environment reports an `obs_shape` matching `obsShape.toList`.

If the handshake fails, the subprocess is terminated and an `IO.userError` is thrown.

`makeKwargs` (optional) is a JSON object encoded as a list of fields and passed through to the
Python bridge as `--make-kwargs <json>`, which the server forwards to `gym.make(envId, **kwargs)`.
This is useful for environments that require constructor options, e.g. Atari RAM observations:
`makeKwargs := [("obs_type", .str "ram")]`.
-/
def spawn {obsShape : Shape} {nActions : Nat}
    (serverScript : String)
    (envId : String)
    (contract : Boundary.Contract obsShape nActions)
    (makeKwargs : List (String × Json) := []) :
    IO (Client obsShape nActions) := do
  let extraArgs : Array String :=
    if makeKwargs.isEmpty then
      #[]
    else
      #["--make-kwargs", Json.compress (Json.mkObj makeKwargs)]
  let child : Child ← IO.Process.spawn
    { cmd := "python3"
      args := #["-u", serverScript, "--env-id", envId] ++ extraArgs
      stdin := .piped
      stdout := .piped
      stderr := .inherit }
  let g : Client obsShape nActions := { child := child, contract := contract }
  try
    -- Fail fast if the external env's declared spaces do not match what the Lean code expects.
    -- This is a cheap handshake that catches mismatches before we start collecting rollouts.
    let o ← Internal.requestObj g (Json.mkObj [("cmd", "describe")])
    let nActionsJ ← Internal.requireField o "n_actions"
    let obsShapeJ ← Internal.requireField o "obs_shape"

    let nActions' ←
      match Boundary.parseNatStrict nActionsJ with
      | .ok n => pure n
      | .error e => throw <| IO.userError e
    unless nActions' == nActions do
      throw <| IO.userError s!"Gymnasium: server reports n_actions={nActions'}, expected nActions={nActions}"

    let obsShapeList ←
      match obsShapeJ with
      | .arr xs =>
          match xs.toList.mapM (fun j => Boundary.parseNatStrict j) with
          | .ok ns => pure ns
          | .error e => throw <| IO.userError e
      | _ =>
          throw <| IO.userError "Gymnasium: expected `obs_shape` to be an array of integers"

    let expectedObs : List Nat := Shape.toList obsShape
    unless obsShapeList == expectedObs do
      throw <|
        IO.userError
          s!"Gymnasium: server reports obs_shape={obsShapeList}, expected obsShape={expectedObs} (i.e. {Shape.pretty obsShape})"

    pure g
  catch e =>
    -- Best-effort cleanup if the handshake fails (e.g. wrong env id, wrong shapes).
    try g.child.kill catch _ => pure ()
    let _ ← g.child.wait
    throw e

/-- Reset the environment and return the initial observation. -/
def reset {obsShape : Shape} {nActions : Nat}
    (g : Client obsShape nActions) (seed? : Option Nat := none) :
    IO (Tensor Float obsShape) := do
  let fields : List (String × Json) :=
    match seed? with
    | none => [("cmd", "reset")]
    | some s => [("cmd", "reset"), ("seed", (s : Json))]
  let o ← Internal.requestObj g (Json.mkObj fields)
  let obsJ ← Internal.requireField o "obs"
  let obs ←
    match Boundary.parseTensorE (field := "obs") obsShape obsJ with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  match Boundary.checkObservation (obsShape := obsShape) (nActions := nActions) g.contract (obs := obs) with
  | .ok () => pure obs
  | .error e => throw <| IO.userError e

/--
Step the environment with a raw action index.

The action is “raw” because the Python bridge receives a `Nat`; the checked session layer normally
passes `action.1` from a `Fin nActions`. The response is still parsed through the Lean-side
observation/reward/done contract before it is returned.
-/
def stepRaw {obsShape : Shape} {nActions : Nat}
    (g : Client obsShape nActions) (action : Nat) :
    IO (Tensor Float obsShape × Float × Bool × Bool) := do
  let o ← Internal.requestObj g (Json.mkObj [("cmd", "step"), ("action", (action : Json))])
  let obsJ ← Internal.requireField o "obs"
  let rewardJ ← Internal.requireField o "reward"
  let terminatedJ ← Internal.requireField o "terminated"
  let truncatedJ ← Internal.requireField o "truncated"

  let obs ←
    match Boundary.parseTensorE (field := "obs") obsShape obsJ with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  let reward ←
    match Boundary.parseFloat (field := "reward") rewardJ with
    | .ok r => pure r
    | .error e => throw <| IO.userError e
  let terminated ←
    match Boundary.parseBool (field := "terminated") terminatedJ with
    | .ok b => pure b
    | .error e => throw <| IO.userError e
  let truncated ←
    match Boundary.parseBool (field := "truncated") truncatedJ with
    | .ok b => pure b
    | .error e => throw <| IO.userError e
  pure (obs, reward, terminated, truncated)

/-- Close the subprocess (best-effort). -/
def close {obsShape : Shape} {nActions : Nat} (g : Client obsShape nActions) : IO Unit := do
  try
    let _ ← Internal.requestObj g (Json.mkObj [("cmd", "close")])
    pure ()
  catch _ =>
    pure ()
  let _ ← g.child.wait
  pure ()

/--
Spawn a client, run `k`, and ensure the subprocess is closed even if `k` throws.
-/
def withClient {α : Type} {obsShape : Shape} {nActions : Nat}
    (serverScript envId : String)
    (contract : Boundary.Contract obsShape nActions)
    (k : Client obsShape nActions → IO α) : IO α := do
  let g ← spawn (obsShape := obsShape) (nActions := nActions) serverScript envId contract
  try
    k g
  finally
    g.close

end Client

end Gymnasium
end RL
end Runtime
