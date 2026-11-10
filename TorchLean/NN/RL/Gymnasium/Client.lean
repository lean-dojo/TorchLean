/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Boundary.Json
public import TorchLean.NN.Subprocess

/-!
# Gymnasium Bridge (Client)

`Client` is the low-level subprocess client for the Python helper script
`scripts/rl/gymnasium_server.py`. It spawns a Python subprocess hosting an external Gymnasium
environment, writes one JSON object per line to its stdin and reads one JSON response per line from
its stdout. Received observations, rewards and done flags are checked against a Lean-side
trust-boundary contract (`TorchLean.RL.Boundary.Contract`).

Startup performs a `describe` handshake, checking that the external environment's declared spaces
match the Lean-side expectations (`obsShape`, `nActions`).

We wrote the client by hand against the protocol in the docstring of
`scripts/rl/gymnasium_server.py`. The `describe` reply also carries `obs_dtype`, and the handshake
leaves it unread: every observation entry reaches Lean as a JSON number and becomes a `Float`, so a
`uint8` frame and a `float32` one land in the same tensor type.

References:
- Gymnasium API reference (`reset`/`step`, `terminated` vs `truncated`):
  https://gymnasium.farama.org/
- Brockman et al., "OpenAI Gym" (2016), arXiv:1606.01540, background on the env interface:
  https://arxiv.org/abs/1606.01540
- Gymnasium source repository (implementation reference):
  https://github.com/Farama-Foundation/Gymnasium
- Trust-boundary rationale and contract definition: `TorchLean.NN.RL.Boundary`.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Gymnasium

open TorchLean.Tensor
open Lean
open Lean.Json

/-!
## Low-level client

`Client` owns an `IO.Process.Child` and:

- sends a JSON object (one line) to stdin,
- reads one JSON object response from stdout,
- checks `"ok": true` and reports `"error"` strings as `IO.userError`.
-/

/-- Stdio configuration of the Gymnasium subprocess: piped stdin and stdout carry the JSON
protocol, and stderr is inherited. -/
def stdio : IO.Process.StdioConfig :=
  { stdin := .piped, stdout := .piped, stderr := .inherit }

/--
Typed handle to a running Gymnasium subprocess environment, with the Lean-side `Boundary.Contract`
that its observations, rewards and done flags are checked against.
-/
structure Client (obsShape : Shape) (nActions : Nat) where
  /-- The Python subprocess. -/
  child : IO.Process.Child stdio
  /-- Lean-side contract, checked by `reset` on each initial observation and by
`Session.stepChecked` on every transition. -/
  contract : Boundary.Contract obsShape nActions

namespace Client

/--
Send a request object and return the response object map. The server answers
`{"ok": true, ...}` or `{"ok": false, "error": "..."}`.

The helper lives in `Gymnasium.Client`, which keeps the JSON protocol behind the Gymnasium trust
boundary and out of the curated API. Public callers use `spawn`, `reset`, `close` or the
higher-level `Session` API.
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
      | _ =>
        throw <| IO.userError
          "Gymnasium: response had ok=false but missing string field `error`"
    throw <| IO.userError s!"Gymnasium server error: {msg}"

  pure o

/--
Spawn the Gymnasium helper subprocess under `$TORCHLEAN_PYTHON`, or `python3` if that variable is
unset, and perform a small handshake (`describe`).

The handshake checks that:
- the external environment reports the expected `n_actions`, and
- the external environment reports an `obs_shape` matching `obsShape`.

If the handshake fails, the subprocess is terminated and the error is rethrown.

`makeKwargs` (optional) is a JSON object encoded as an array of fields and passed through to the
Python bridge as `--make-kwargs <json>`, which the server forwards to `gym.make(envId, **kwargs)`.
Environments with constructor options need it, for example Atari RAM observations with
`makeKwargs := #[("obs_type", .str "ram")]`.
-/
def spawn {obsShape : Shape} {nActions : Nat}
    (serverScript : String)
    (envId : String)
    (contract : Boundary.Contract obsShape nActions)
    (makeKwargs : Array (String × Json) := #[]) :
    IO (Client obsShape nActions) := do
  let extraArgs : Array String :=
    if makeKwargs.isEmpty then
      #[]
    else
      #["--make-kwargs", Json.compress (Json.mkObj makeKwargs.toList)]
  let python ← Subprocess.command "TORCHLEAN_PYTHON" "python3"
  let child : IO.Process.Child stdio ← IO.Process.spawn
    { cmd := python
      args := #["-u", serverScript, "--env-id", envId] ++ extraArgs
      stdin := .piped
      stdout := .piped
      stderr := .inherit }
  let g : Client obsShape nActions := { child := child, contract := contract }
  try
    -- Fail fast when the declared spaces differ from `obsShape` and `nActions`, so a protocol
    -- mismatch surfaces before any rollout is collected.
    let o ← requestObj g (Json.mkObj [("cmd", "describe")])
    let nActionsJ ← IO.ofExcept <| Json.field "Gymnasium: response" "n_actions" o
    let obsShapeJ ← IO.ofExcept <| Json.field "Gymnasium: response" "obs_shape" o

    let nActions' ← IO.ofExcept <|
      Json.parseNatStrict "Gymnasium: response.n_actions" nActionsJ
    unless nActions' == nActions do
      throw <| IO.userError
        s!"Gymnasium: server reports n_actions={nActions'}, expected nActions={nActions}"

    let obsShapeList ← IO.ofExcept do
      (← Json.array "Gymnasium: response.obs_shape" obsShapeJ).toList.mapM
        (Json.parseNatStrict "Gymnasium: response.obs_shape")

    let expectedObs : List Nat := obsShape
    unless obsShapeList == expectedObs do
      throw <|
        IO.userError
          (s!"Gymnasium: server reports obs_shape={obsShapeList}, "
            ++ s!"expected obsShape={expectedObs} (i.e. {obsShape})")

    pure g
  catch e =>
    -- A failed handshake (wrong env id, wrong shapes) must not leave the Python process running.
    -- Kill errors are ignored because the child may already have exited.
    try g.child.kill catch _ => pure ()
    let _ ← g.child.wait
    throw e

/-- Reset the environment, sending `seed?` as the `seed` field when given, and return the initial
observation after `Boundary.checkObservation` accepts it. -/
def reset {obsShape : Shape} {nActions : Nat}
    (g : Client obsShape nActions) (seed? : Option Nat := none) :
    IO (Tensor Float obsShape) := do
  let fields : List (String × Json) :=
    match seed? with
    | none => [("cmd", "reset")]
    | some s => [("cmd", "reset"), ("seed", (s : Json))]
  let o ← requestObj g (Json.mkObj fields)
  let obsJ ← IO.ofExcept <| Json.field "Gymnasium: response" "obs" o
  let obs ← IO.ofExcept (Boundary.parseTensorE (field := "obs") obsShape obsJ)
  IO.ofExcept (Boundary.checkObservation (obsShape := obsShape) (nActions := nActions) g.contract
    (obs := obs))
  pure obs

/-- Send one action through the JSON protocol and parse the resulting transition fields. The
contract check runs in `Session.stepChecked`, which sees the whole transition. -/
def step {obsShape : Shape} {nActions : Nat}
    (g : Client obsShape nActions) (action : Nat) :
    IO (Tensor Float obsShape × Float × Bool × Bool) := do
  let o ← Client.requestObj g (Json.mkObj [("cmd", "step"), ("action", (action : Json))])
  let ctx := "Gymnasium: response"
  let obsJ ← IO.ofExcept <| Json.field ctx "obs" o
  let reward ← IO.ofExcept <| Json.floatField ctx "reward" o
  let terminated ← IO.ofExcept <| Json.boolField ctx "terminated" o
  let truncated ← IO.ofExcept <| Json.boolField ctx "truncated" o

  let obs ← IO.ofExcept (Boundary.parseTensorE (field := "obs") obsShape obsJ)
  pure (obs, reward, terminated, truncated)

/-- Send `close` to the subprocess, kill it if the request fails, and wait for it to exit. -/
def close {obsShape : Shape} {nActions : Nat} (g : Client obsShape nActions) : IO Unit := do
  try
    let _ ← requestObj g (Json.mkObj [("cmd", "close")])
    pure ()
  catch _ =>
    -- A failed close request can leave the helper serving requests, so kill it before
    -- waiting for it to exit.
    try g.child.kill catch _ => pure ()
  let _ ← g.child.wait
  pure ()

end Client

end Gymnasium
end RL
end TorchLean
