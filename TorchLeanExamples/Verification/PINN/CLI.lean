/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.PINN
public import TorchLeanExamples.Verification.PINN.PDEParse
public import TorchLeanExamples.Verification.PINN.ResidualAffine
public import TorchLean.NN.CLI.Command -- shake: keep
public import TorchLeanExamples.Verification.PINN.PyTorch.Model -- shake: keep

/-!
# PINN CLI

An interactive tool that bounds the PDE residual of a trained physics-informed neural network
(PINN) over a box of inputs. It
- parses a compact PDE mini-language (`PINN.Expr`, read by `PINN.parseExpr`),
- evaluates that expression using interval bounds for `u`, `du` and `d2u`,
- computes those primitive bounds by IBP/CROWN-style propagation on the network's graph of
  registered operators and its tangent graphs (`ofGraph`),
- optionally tightens bounds by recursively splitting the input box (1D/2D).

Use it to inspect residual bounds while iterating on a PDE expression or an input box. A stable
artifact for docs, papers or CI comes from the certificate checker:
`scripts/lake.sh exe verify -- pinn-cert [TorchLeanExamples/Verification/PINN/pinn_cert.json]`

Run this CLI via the unified verification dispatcher:
`scripts/lake.sh exe verify -- pinn-cli -- [flags] "<PDE>" x eps`
or for 2D:
`scripts/lake.sh exe verify -- pinn-cli -- [flags] "<PDE>" x y eps`

Examples:
- `scripts/lake.sh exe verify -- pinn-cli -- "u_xx + u" 0.0 0.1`
- `scripts/lake.sh exe verify -- pinn-cli -- "u_t - u_xx" 0.0 0.0 0.05`

References:
- PINNs: `https://arxiv.org/abs/1711.10561`
- IBP: `https://arxiv.org/abs/1810.12715`
- CROWN: `https://arxiv.org/abs/1811.00866`
-/

@[expose] public section


namespace TorchLean.Examples.PINN

open CROWN
open Derivatives
open Expr
open TorchLean.Tensor

/-- Bound method for the PINN output value, selected by `--method`: interval-only (`ibp`), the
output's box in full CROWN (`crown-fwd`) or the backward objective (`crown-bwd`). -/
inductive Method
  /-- The IBP interval alone (`ibp`). -/
  | ibp
  /-- The output box of full CROWN (`crown-fwd`). -/
  | crownFwd
  /-- The CROWN backward objective (`crown-bwd` or `crown`). -/
  | crownBwd

instance : ToString Method :=
  ⟨fun
    | .ibp => "ibp"
    | .crownFwd => "crown-fwd"
    | .crownBwd => "crown-bwd"⟩

/-- Bound the primitives `u`, `duX`, `duY`, `d2uX` and `d2uY` of the PINN graph `G` with payload
`P` over the input box `xB`, for 1D and 2D models. With `crown-fwd` or `crown-bwd`, a CROWN bound
replaces the IBP interval for `u` whenever one is available. -/
def computePrimsAt (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float)
    (uMethod : Method := .ibp) : IO Prims := do
  let ibp := Op.runIBP G P (Op.inputSeeds G [xB])
  let uBox ← IO.ofExcept (outputInterval "IBP failed at output" G ibp 0)
  let inDim : Nat ←
    match G.nodes.find? (· matches .input 0 _) with
    | some (.input _ p) =>
      match p.shape with
      | .dim n .scalar => pure n
      | _ => throw <| IO.userError "PINN input node must have a one-dimensional vector shape"
    | _ => throw <| IO.userError "PINN graph has no input node"
  -- `Derivatives.ofGraph` builds one graph for the value and both tangents, or for the first
  -- tangents alone. When it fails, every derivative bound is `none`.
  let derivatives? := (ofGraph G P).toOption
  let boundsAlong (index : Nat) :
      Option (Option (FloatLib.Numerics.Interval Float) ×
        Option (FloatLib.Numerics.Interval Float)) :=
    if h : index < inDim then
      some <| match derivatives? with
        | some D => axisDerivativeBounds D xB inDim ⟨index, h⟩
        | none => (none, none)
    else none
  let endpoints (iv? : Option (FloatLib.Numerics.Interval Float)) : Option (Float × Float) :=
    iv?.map fun iv => (iv.lo, iv.hi)
  let (duX, d2uX) ←
    match boundsAlong 0 with
    | some (first, second) => pure (endpoints first, endpoints second)
    | none => throw <| IO.userError "PINN input dimension must be positive"
  let (duY, d2uY) :=
    match boundsAlong 1 with
    | some (first, second) => (endpoints first, endpoints second)
    | none => (none, none)
  let base : Prims :=
    { u := some (uBox.lo, uBox.hi), duX := duX, duY := duY, d2uX := d2uX, d2uY := d2uY }
  match uMethod with
  | .ibp => pure base
  | .crownFwd =>
    match crownUBoundsNodewise G P xB ibp with
    | some (al, ah) => pure { base with u := some (al, ah) }
    | none => pure base
  | .crownBwd =>
    match crownUBoundsBackward G P xB ibp with
    | some (al, ah) => pure { base with u := some (al, ah) }
    | none => pure base

/--
Load optional PyTorch-exported PINN weights for the exploratory CLI.

If the file cannot be read or parsed, does not match the expected shapes or has the wrong input
dimension, the CLI prints a warning and falls back to the given built-in graph and payload. That
permissive fallback suits `pinn-cli` as an interactive inspection tool, while stable certificate
checkers should reject bad artifacts.
-/
def loadWeightsOrDefault
    (weights? : Option String)
    (expectedInputDim : Nat)
    (default : IR.Graph × IR.Payload Float) :
    IO (IR.Graph × IR.Payload Float) := do
  match weights? with
  | none => pure default
  | some path => do
      try
        let j ← Json.readFile path
        match Checkpoint.ofJson? j with
        | some sd =>
            if sd.arch.inputDim ≠ expectedInputDim then
              IO.eprintln <|
                (s!"[PINN] Loaded weights expect input dim={sd.arch.inputDim}; using " ++
                  s!"built-in {expectedInputDim}D weights.")
              pure default
            else
              IO.ofExcept (Checkpoint.model (fun x => x) sd)
        | none =>
            IO.eprintln <|
              ("[PINN] Weights JSON did not match expected shapes; falling back to " ++
                "built-in weights")
            pure default
      catch e =>
        IO.eprintln s!"[PINN] Failed to parse weights JSON: {e}; falling back to built-in weights"
        pure default

/-- Parse a `--method` value, ignoring case, with `crown` as an alias of `crown-bwd`. -/
def Method.parse? (s : String) : Option Method :=
  match s.toLower with
  | "ibp" => some .ibp
  | "crown" => some .crownBwd
  | "crown-fwd" => some .crownFwd
  | "crown-bwd" => some .crownBwd
  | _ => none

/-- Parsed options for the PINN residual CLI. -/
structure Options where
  /-- Bound propagation method used for the output interval. -/
  method : Method := .ibp
  /-- Optional PyTorch-exported PINN weights JSON. -/
  weights? : Option String := none
  /-- Depth of recursive input-box splitting in 1D and 2D (`0` for no splitting). -/
  splitDepth : Nat := 0

/-- Parse recognized flags and return the remaining positional arguments. -/
def parseFlags (args : List String) : Except String (Options × List String) := do
  let args := CLI.dropDashDash args
  let (weights?, args) ← CLI.takeFlagValue? args "weights"
  let (splitDepth, args) ← CLI.takeNatFlag args "split-depth" (default := 0)
  let (method, args) ←
    CLI.takeParsedFlag args "method" (default := "ibp") fun s =>
      match Method.parse? s with
      | some method => pure method
      | none => throw s!"--method: expected ibp, crown, crown-fwd, or crown-bwd; got `{s}`"
  pure ({ method := method, weights? := weights?, splitDepth := splitDepth }, args)

/--
Entry point of the interactive PINN residual-bounding tool `pinn-cli`. Certificate checking goes
through
`scripts/lake.sh exe verify -- pinn-cert [TorchLeanExamples/Verification/PINN/pinn_cert.json]`.

Run it in 1D or 2D as
```text
scripts/lake.sh exe verify -- pinn-cli -- [--method=ibp|crown-fwd|crown-bwd] [--split-depth=N]
  [--weights=PATH.json] "<PDE>" x eps
scripts/lake.sh exe verify -- pinn-cli -- [flags] "<PDE>" x y eps
```
-/
def main (args : List String) : IO Unit := do
  let (options, rest) ←
    match parseFlags args with
    | .ok parsed => pure parsed
    | .error e => throw <| IO.userError e
  let method := options.method
  let weights? := options.weights?
  let splitDepth := options.splitDepth

  let rec split1D (x : Float) (eps : Float) (d : Nat)
    (evalAt : Float → Float → IO (Float × Float)) : IO (Float × Float) := do
    match d with
    | 0 => evalAt x eps
    | Nat.succ d' =>
      let eps' := eps * 0.5
      let (l1, h1) ← split1D (x - eps') eps' d' evalAt
      let (l2, h2) ← split1D (x + eps') eps' d' evalAt
      pure (if l1 < l2 then l1 else l2, if h1 > h2 then h1 else h2)

  let rec split2D (x y eps : Float) (d : Nat)
    (evalAt : Float → Float → Float → IO (Float × Float)) : IO (Float × Float) := do
    match d with
    | 0 => evalAt x y eps
    | Nat.succ d' =>
      let eps' := eps * 0.5
      let (l1, h1) ← split2D (x - eps') (y - eps') eps' d' evalAt
      let (l2, h2) ← split2D (x - eps') (y + eps') eps' d' evalAt
      let (l3, h3) ← split2D (x + eps') (y - eps') eps' d' evalAt
      let (l4, h4) ← split2D (x + eps') (y + eps') eps' d' evalAt
      let lo12 := if l1 < l2 then l1 else l2
      let lo34 := if l3 < l4 then l3 else l4
      let hi12 := if h1 > h2 then h1 else h2
      let hi34 := if h3 > h4 then h3 else h4
      pure (if lo12 < lo34 then lo12 else lo34, if hi12 > hi34 then hi12 else hi34)
  match rest with
  | .cons pdeStr (.cons xStr (.cons epsStr .nil)) =>
    let x? := CLI.parseFloatLit? xStr
    let eps? := CLI.parseFloatLit? epsStr
    match x?, eps? with
    | some x, some eps => do
      unless x.isFinite do
        throw <| IO.userError s!"x must be finite, got {x}"
      unless eps.isFinite && eps ≥ 0.0 do
        throw <| IO.userError s!"eps must be finite and nonnegative, got {eps}"
      let expr ←
        match parseExpr (fun _ => none) pdeStr with
        | .ok e => pure e
        | .error msg => throw <| IO.userError s!"Parse error: {msg}"
      let (g, payload) ←
        loadWeightsOrDefault weights? 1 (← IO.ofExcept (referenceModel 1))
      let evalAt : Float → Float → IO (Float × Float) :=
        fun xc epsc => do
          let center : Tensor Float [1] :=
            Tensor.dim fun _ => Tensor.scalar xc
          let prims ← computePrimsAt g payload (FlatBox.lInfBall center epsc) method
          match eval prims expr with
          | some (lo, hi) =>
              if lo.isNaN || hi.isNaN then
                throw <| IO.userError "PDE evaluation produced a NaN bound"
              else
                pure (lo, hi)
          | none => throw <| IO.userError "PDE evaluation failed (insufficient primitives)"
      let (lo0, hi0) ← evalAt x eps
      let (loS, hiS) ←
        if splitDepth = 0 then
          pure (lo0, hi0)
        else
          split1D x eps splitDepth evalAt
      -- Intersect the unsplit and split results, or take their hull when they are disjoint.
      let loI := if lo0 > loS then lo0 else loS
      let hiI := if hi0 < hiS then hi0 else hiS
      let (lo, hi) :=
        if loI ≤ hiI then
          (loI, hiI)
        else
          (if lo0 < loS then lo0 else loS, if hi0 > hiS then hi0 else hiS)
      IO.println <|
        (s!"PDE='{pdeStr}' at x={x}, eps={eps}, method={method}, " ++
          s!"splitDepth={splitDepth}: residual ∈ [{lo},{hi}]")
    | _, _ =>
      throw <| IO.userError s!"invalid float input(s): x={xStr}, eps={epsStr}"
  | .cons pdeStr (.cons xStr (.cons yStr (.cons epsStr .nil))) =>
    let x? := CLI.parseFloatLit? xStr
    let y? := CLI.parseFloatLit? yStr
    let eps? := CLI.parseFloatLit? epsStr
    match x?, y?, eps? with
    | some x, some y, some eps => do
      unless x.isFinite && y.isFinite do
        throw <| IO.userError s!"x and y must be finite, got ({x}, {y})"
      unless eps.isFinite && eps ≥ 0.0 do
        throw <| IO.userError s!"eps must be finite and nonnegative, got {eps}"
      let expr ←
        match parseExpr (fun _ => none) pdeStr with
        | .ok e => pure e
        | .error msg => throw <| IO.userError s!"Parse error: {msg}"
      let (g, payload) ←
        loadWeightsOrDefault weights? 2 (← IO.ofExcept (referenceModel 2))
      let evalAt : Float → Float → Float → IO (Float × Float) :=
        fun xc yc epsc => do
          let center : Tensor Float [2] :=
            Tensor.dim fun i => Tensor.scalar <| if i.val = 0 then xc else yc
          let prims ← computePrimsAt g payload (FlatBox.lInfBall center epsc) method
          match eval prims expr with
          | some (lo, hi) =>
              if lo.isNaN || hi.isNaN then
                throw <| IO.userError "PDE evaluation produced a NaN bound"
              else
                pure (lo, hi)
          | none => throw <| IO.userError "PDE evaluation failed (insufficient primitives)"
      let (lo0, hi0) ← evalAt x y eps
      let (loS, hiS) ←
        if splitDepth = 0 then
          pure (lo0, hi0)
        else
          split2D x y eps splitDepth evalAt
      let loI := if lo0 > loS then lo0 else loS
      let hiI := if hi0 < hiS then hi0 else hiS
      let (lo, hi) :=
        if loI ≤ hiI then
          (loI, hiI)
        else
          (if lo0 < loS then lo0 else loS, if hi0 > hiS then hi0 else hiS)
      IO.println <|
        (s!"PDE='{pdeStr}' at (x,y)=({x},{y}), eps={eps}, method={method}, " ++
          s!"splitDepth={splitDepth}: residual ∈ [{lo},{hi}]")
    | _, _, _ =>
      throw <| IO.userError s!"invalid float input(s): x={xStr}, y={yStr}, eps={epsStr}"
  | _ =>
    throw <| IO.userError <|
      ("Usage:\n  lake exe verify -- pinn-cli -- " ++
        "[--method=ibp|crown-fwd|crown-bwd] [--split-depth=N] " ++
        "[--weights=path.json] \"<PDE>\" x eps   " ++
        " # 1D\n  lake exe verify -- pinn-cli -- " ++
        "[--method=ibp|crown-fwd|crown-bwd] [--split-depth=N] " ++
        "[--weights=path.json] \"<PDE>\" x y eps " ++
        " # 2D")

end TorchLean.Examples.PINN
