/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core
public import NN.Verification.PINN.PdeParse
public import NN.Verification.PINN.PyTorch

/-!
# PINN CLI

PINN residual-bounding CLI.

This file implements a small interactive tool for bounding PDE residuals of a trained
physics-informed neural network (PINN) over a box input region. Concretely, it:
- parses a compact PDE mini-language (see `PdeAst` / `PdeParse`),
- evaluates that expression using interval bounds for `u`, `du`, and `d2u`,
- computes those primitive bounds via IBP/CROWN-style propagation on a CROWN `Graph`,
- optionally tightens bounds by recursively splitting the input box (1D/2D).

This CLI is the interactive PINN residual-bound tool:
- use it when you want to inspect residual bounds interactively;
- use it when you are iterating on a PDE expression or input box;
- use the certificate checker when you want a stable artifact for docs, papers, or CI.

The stable artifact checker is:
`lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`

Run this CLI via the unified verification dispatcher:
`lake exe verify -- pinn-cli -- [flags] "<PDE>" x eps`
or for 2D:
`lake exe verify -- pinn-cli -- [flags] "<PDE>" x y eps`

Examples:
- `lake exe verify -- pinn-cli -- "u_xx + u" 0.0 0.1`
- `lake exe verify -- pinn-cli -- --backend=float "u_t - u_xx" 0.0 0.05`

References:
- PINNs: `https://arxiv.org/abs/1711.10561`
- IBP: `https://arxiv.org/abs/1810.12715`
- CROWN: `https://arxiv.org/abs/1811.00866`
-/

@[expose] public section


namespace NN.Verification.PINN.CLI

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.PINN
open NN.Verification.PINN.PdeAst
open NN.Verification.PINN.PdeParse
open NN.Verification.PINN.ResidualAffine
open _root_.Spec
open _root_.Spec.Tensor
open TorchLean.Floats

/- Approximate conversion ℝ → Float by rounding to `digits` decimal places. -/
noncomputable def realToFloat (x : ℝ) (digits : Nat := 6) : Float :=
  let pow10 : ℝ := (10 : ℝ) ^ digits
  let y : ℝ := x * pow10
  -- round to nearest, ties toward +∞ for simplicity
  let y' : ℝ := if y ≥ 0 then y + 0.5 else y - 0.5
  let n : Int := Int.floor y'
  let nAbs : Nat := n.natAbs
  let sgn : Float := if n ≥ 0 then 1.0 else -1.0
  let num : Float := sgn * Float.ofNat nAbs
  num / Float.ofNat (Nat.pow 10 digits)

-- Helpers and NF backend computation live here so the CLI can switch scalar backends
-- without dragging that backend-selection logic into the higher-level PDE residual code.

/-- Compute primitives using a rounded backend α = NF with single-precision nearest-even. -/
noncomputable def computePrimsAtNF (g : Graph) {β : NeuralRadix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
  [NeuralValidExp fexp] [NeuralValidRnd rnd]
    (ps : ParamStore (NF β fexp rnd)) (_useAffineU : Bool := false) : IO Prims := do
  let outId := NN.Verification.PINN.SequentialPINNArch.graphOutputId g
  let ibp := runIBP (α:=NF β fexp rnd) g ps
  let some outB := ibp[outId]! | throw <| IO.userError "IBP failed at output (NF)"
  let uLoA := Spec.Tensor.sumSpec outB.lo
  let uHiA := Spec.Tensor.sumSpec outB.hi
  let uLo := realToFloat (NF.toReal uLoA)
  let uHi := realToFloat (NF.toReal uHiA)
  -- input dimension from node 0
  let inDim : Nat :=
    match g.nodes[0]? with
    | some n0 => (match n0.outShape with | .dim n .scalar => n | _ => 1)
    | none => 1
  let hasY : Bool := match inDim with | 0 => false | 1 => false | _ => true
  -- First/second derivative along X (dir 0)
  let seedX := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := NF β fexp rnd) inDim 0)
  let d1x := runDerivDirectional (α:=NF β fexp rnd) g ps ibp seedX
  let d2x := runDeriv2D (α:=NF β fexp rnd) g ps ibp d1x
  let d1xOpt := d1x[outId]!
  let d2xOpt := d2x[outId]!
  let (duX, d2uX) :=
    match d1xOpt, d2xOpt with
    | some dxB, some d2xB =>
      let l1 := realToFloat (NF.toReal (Spec.Tensor.sumSpec dxB.lo))
      let h1 := realToFloat (NF.toReal (Spec.Tensor.sumSpec dxB.hi))
      let l2 := realToFloat (NF.toReal (Spec.Tensor.sumSpec d2xB.lo))
      let h2 := realToFloat (NF.toReal (Spec.Tensor.sumSpec d2xB.hi))
      (some (l1, h1), some (l2, h2))
    | _, _ => (none, none)
  -- Y direction if available
  let duY : Option (Float × Float) :=
    if hasY then
      let seedY := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := NF β fexp rnd) inDim 1)
      let d1y := runDerivDirectional (α:=NF β fexp rnd) g ps ibp seedY
      match d1y[outId]! with
      | some dyB =>
        let l := realToFloat (NF.toReal (Spec.Tensor.sumSpec dyB.lo))
        let h := realToFloat (NF.toReal (Spec.Tensor.sumSpec dyB.hi))
        some (l, h)
      | none => none
    else none
  let d2uY : Option (Float × Float) :=
    if hasY then
      let seedY := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := NF β fexp rnd) inDim 1)
      let d1y := runDerivDirectional (α:=NF β fexp rnd) g ps ibp seedY
      let d2y := runDeriv2D (α:=NF β fexp rnd) g ps ibp d1y
      match d2y[outId]! with
      | some d2yB =>
        let l := realToFloat (NF.toReal (Spec.Tensor.sumSpec d2yB.lo))
        let h := realToFloat (NF.toReal (Spec.Tensor.sumSpec d2yB.hi))
        some (l, h)
      | none => none
    else none
  let base : Prims := { u := some (uLo, uHi), duX := duX, duY := duY, d2uX := d2uX, d2uY := d2uY }
  pure base

/- Backend selection for PINN CLI: choose how bounds are computed.
   - `float`: the default executable path.
   - `neuralfloat`: a rounding-aware path where available, with Float fallback where the
     graph-wide propagation uses the Float propagation engine in this CLI path.
-/
inductive Backend where
  | float
  | neuralfloat

def parseBackendArg (s : String) : Option Backend :=
  match s with
  | "--backend=float" => some .float
  | "--backend=neuralfloat" => some .neuralfloat
  | _ => none

private def parseWeightsArg (s : String) : Option String :=
  if s.startsWith "--weights=" then some ((s.drop 10).toString) else none

def parseFloat (s : String) : Option Float :=
  match parseNumber { s := s } with
  | .ok (v, st) => if st.i = s.rawEndPos then some v else none
  | .error _ => none

/-- Parse a non-scientific decimal string into a generic numeric α using Numbers and Context.
  Supports optional leading '-' and a single '.'. -/
def parseAlphaDecimal {α : Type} [Context α] (s : String) : Option α :=
  let rec pow10 (k : Nat) : α :=
    match k with
    | 0 => Numbers.one
    | Nat.succ k' => Numbers.ten * pow10 k'
  let mk (neg : Bool) (intPart fracPart : String) : Option α :=
    let intVal : α := (
      (intPart.toList.foldl (fun (acc : α × Bool) ch =>
        let (accv, seen) := acc
        if seen then (accv, true) else
          if ch = '0' then (accv * Numbers.ten, false)
          else if ch ≥ '0' ∧ ch ≤ '9' then
            (accv * Numbers.ten + ((ch.toNat - '0'.toNat) : Nat), false)
          else (accv, true)
      ) (Numbers.zero, false)).fst)
    let fracVal? : Option α :=
      if fracPart.isEmpty then some Numbers.zero else
      let digits := fracPart.toList
      let numDen := digits.foldl (fun (acc : α × Nat × Bool) ch =>
        let (n, d, bad) := acc
        if bad then (n, d, bad) else
          if ch ≥ '0' ∧ ch ≤ '9' then
            (n * Numbers.ten + ((ch.toNat - '0'.toNat) : Nat), d + 1, false)
          else (n, d, true)
      ) (Numbers.zero, 0, false)
      let (num, denK, bad) := numDen
      if bad then none else
        some (num / (pow10 denK))
    match fracVal? with
    | some fracVal =>
      let v := intVal + fracVal
      some (if neg then (-v) else v)
    | none => none
  let s := s.trimAscii.toString
  if s.isEmpty then none else
  let neg := s.front = '-'
  let body : String := if neg then (s.drop 1).toString else s
  match body.splitOn "." with
  | [intPart] => mk neg intPart ""
  | [intPart, fracPart] => mk neg intPart fracPart
  | _ => none



/-- Compute primitives u, duX, duY, d2uX, d2uY at the unique output node (id=5) for 1D/2D models.
    Optionally, replace the `u`-interval with a tighter CROWN/DeepPoly bound. -/
inductive UBoundsMethod
  | ibp
  | crownFwd
  | crownBwd

def computePrimsAt (g : Graph) (ps : ParamStore Float) (uMethod : UBoundsMethod := .ibp) (backend :
  Backend := .float) : IO Prims := do
  let outId := NN.Verification.PINN.SequentialPINNArch.graphOutputId g
  let ibp ←
    match backend with
    | .float => pure (runIBP (α:=Float) g ps)
    | .neuralfloat => do
      -- This CLI path still selects Float graph-wide propagation; rounding-aware routines are
      -- announced and used only where the backend exposes them.
      IO.eprintln <|
        ("[PINN] backend=neuralfloat: using Float propagation; rounding-aware " ++
          "routines will be used where available.")
      pure (runIBP (α:=Float) g ps)
  let some outB := ibp[outId]! | throw <| IO.userError "IBP failed at output"
  let uLo := Spec.Tensor.sumSpec outB.lo
  let uHi := Spec.Tensor.sumSpec outB.hi
  -- Determine input dimension from graph's input node shape
  let inDim : Nat :=
    match g.nodes[0]? with
    | some n0 => (match n0.outShape with | .dim n .scalar => n | _ => 1)
    | none => 1
  let hasY : Bool :=
    match inDim with
    | 0 => false
    | 1 => false
    | _ => true
  -- First/second derivative along X (dir 0)
  let seedX := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 0)
  let d1x := runDerivDirectional (α:=Float) g ps ibp seedX
  let d2x := runDeriv2D (α:=Float) g ps ibp d1x
  let d1xOpt := d1x[outId]!
  let d2xOpt := d2x[outId]!
  let (duX, d2uX) :=
    match d1xOpt, d2xOpt with
    | some dxB, some d2xB =>
      (some (Spec.Tensor.sumSpec dxB.lo, Spec.Tensor.sumSpec dxB.hi),
       some (Spec.Tensor.sumSpec d2xB.lo, Spec.Tensor.sumSpec d2xB.hi))
    | _, _ => (none, none)
  -- First/second derivative along Y (dir 1) if available
  let duY : Option (Float × Float) :=
    if hasY then
      let seedY := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 1)
      let d1y := runDerivDirectional (α:=Float) g ps ibp seedY
      match d1y[outId]! with
      | some dyB => some (Spec.Tensor.sumSpec dyB.lo, Spec.Tensor.sumSpec dyB.hi)
      | none => none
    else none
  let d2uY : Option (Float × Float) :=
    if hasY then
      let seedY := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 1)
      let d1y := runDerivDirectional (α:=Float) g ps ibp seedY
      let d2y := runDeriv2D (α:=Float) g ps ibp d1y
      match d2y[outId]! with
      | some d2yB => some (Spec.Tensor.sumSpec d2yB.lo, Spec.Tensor.sumSpec d2yB.hi)
      | none => none
    else none
  let base : Prims := { u := some (uLo, uHi), duX := duX, duY := duY, d2uX := d2uX, d2uY := d2uY }
  match uMethod with
  | .ibp => pure base
  | .crownFwd =>
    match crownUBoundsForward g ps ibp with
    | some (al, ah) => pure { base with u := some (al, ah) }
    | none => pure base
  | .crownBwd =>
    match crownUBoundsBackward g ps ibp with
    | some (al, ah) => pure { base with u := some (al, ah) }
    | none => pure base

inductive Method
  | ibp
  | crownFwd
  | crownBwd

instance : ToString Method :=
  ⟨fun
    | .ibp => "ibp"
    | .crownFwd => "crown-fwd"
    | .crownBwd => "crown-bwd"⟩

def parseMethodVal (s : String) : Option Method :=
  match s.toLower with
  | "ibp" => some .ibp
  | "partial" => some .crownBwd -- backwards-compatible alias
  | "crown" => some .crownBwd
  | "crown-fwd" => some .crownFwd
  | "crown-bwd" => some .crownBwd
  | _ => none

structure Opts where
  /-- method. -/
  method : Method := .ibp
  /-- Execution backend selection. -/
  backend : Backend := .float
  weights? : Option String := none
  /-- split Depth. -/
  splitDepth : Nat := 0

partial def parseFlags (opts : Opts) (args : List String) : Opts × List String :=
  match args with
  | [] => (opts, [])
  | s :: xs =>
    if s.startsWith "--weights=" then
      parseFlags { opts with weights? := some ((s.drop 10).toString) } xs
    else if s.startsWith "--split-depth=" then
      match ((s.drop 14).toString).toNat? with
      | some n => parseFlags { opts with splitDepth := n } xs
      | none => parseFlags opts xs
    else if s.startsWith "--method=" then
      match parseMethodVal ((s.drop 9).toString) with
      | some m => parseFlags { opts with method := m } xs
      | none => parseFlags opts xs
    else if s = "--method" then
      match xs with
      | v :: rest =>
        match parseMethodVal v with
        | some m => parseFlags { opts with method := m } rest
        | none => parseFlags opts xs
      | [] => (opts, [])
    else
      match parseBackendArg s with
      | some b => parseFlags { opts with backend := b } xs
      | none =>
        if s = "--backend" then
          match xs with
          | v :: rest =>
            match parseBackendArg ("--backend=" ++ v) with
            | some b => parseFlags { opts with backend := b } rest
            | none => (opts, args)
          | [] => (opts, [])
        else
          (opts, args)

/--
Entry point for the PINN residual-bounding CLI.

This is an interactive tool registered as:
`lake exe verify -- pinn-cli -- ...`

For certificate checking, use:
`lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`

Run:
`lake exe verify -- pinn-cli -- [--method=ibp|crown-fwd|crown-bwd|partial] [--split-depth=N]
  [--backend=float|neuralfloat] [--weights=PATH.json] "<PDE>" x eps`
or (2D):
`lake exe verify -- pinn-cli -- [flags] "<PDE>" x y eps`
-/
def main (args : List String) : IO Unit := do
  let (opts, rest) := parseFlags {} args
  let method := opts.method
  let backend := opts.backend
  let weights? := opts.weights?
  let splitDepth := opts.splitDepth
  let uMethod : UBoundsMethod :=
    match method with
    | .ibp => .ibp
    | .crownFwd => .crownFwd
    | .crownBwd => .crownBwd

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
  match rest.toArray with
  | #[pdeStr, xStr, epsStr] =>
    match backend with
    | .float | .neuralfloat =>
      let x? := parseFloat xStr
      let eps? := parseFloat epsStr
      match x?, eps? with
      | some x, some eps => do
        let expr ←
          match parseExpr (fun _ => none) pdeStr with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"Parse error: {msg}"
        let (g, ps0) ← match weights? with
          | some path => do
            let jsonStr ← IO.FS.readFile path
            match Lean.Json.parse jsonStr with
            | Except.error msg =>
              IO.eprintln
                s!"[PINN] Failed to parse weights JSON: {msg}; falling back to built-in weights"
              pure (buildGraph, seedParamsFloat)
            | Except.ok j =>
              match Import.PINNPyTorch.loadPinnState j with
              | some sd =>
                if sd.arch.inputDim ≠ 1 then
                  IO.eprintln <|
                    (s!"[PINN] Loaded weights expect input dim={sd.arch.inputDim}; using " ++
                      "built-in 1D weights.")
                  pure (buildGraph, seedParamsFloat)
                else
                  pure (Import.PINNPyTorch.buildGraph sd, Import.PINNPyTorch.toParamStore sd)
              | none =>
                IO.eprintln <|
                  ("[PINN] Weights JSON did not match expected shapes; falling back to " ++
                    "built-in weights")
                pure (buildGraph, seedParamsFloat)
          | none => pure (buildGraph, seedParamsFloat)
        let evalAt : Float → Float → IO (Float × Float) :=
          fun xc epsc => do
            let ps := seedInputFloat ps0 xc epsc
            let prims ← computePrimsAt g ps uMethod backend
            match eval prims expr with
            | some (lo, hi) => pure (lo, hi)
            | none => throw <| IO.userError "PDE evaluation failed (insufficient primitives)"
        let (lo0, hi0) ← evalAt x eps
        let (loS, hiS) ←
          if splitDepth = 0 then
            pure (lo0, hi0)
          else
            split1D x eps splitDepth evalAt
        -- Never return a worse interval when splitting: intersect when consistent, otherwise fall
        -- back to hull.
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
        IO.eprintln s!"invalid float input(s): x={xStr}, eps={epsStr}"
  | #[pdeStr, xStr, yStr, epsStr] =>
    match backend with
    | .float | .neuralfloat =>
      let x? := parseFloat xStr
      let y? := parseFloat yStr
      let eps? := parseFloat epsStr
      match x?, y?, eps? with
      | some x, some y, some eps => do
        let expr ←
          match parseExpr (fun _ => none) pdeStr with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"Parse error: {msg}"
        let (g, ps0) ← match weights? with
          | some path => do
            let jsonStr ← IO.FS.readFile path
            match Lean.Json.parse jsonStr with
            | Except.error msg =>
              IO.eprintln
                s!"[PINN] Failed to parse weights JSON: {msg}; falling back to built-in weights"
              pure (buildGraph2D, seedParamsFloat2D)
            | Except.ok j =>
              match Import.PINNPyTorch.loadPinnState j with
              | some sd =>
                if sd.arch.inputDim ≠ 2 then
                  IO.eprintln <|
                    (s!"[PINN] Loaded weights expect input dim={sd.arch.inputDim}; using " ++
                      "built-in 2D weights.")
                  pure (buildGraph2D, seedParamsFloat2D)
                else
                  pure (Import.PINNPyTorch.buildGraph sd, Import.PINNPyTorch.toParamStore sd)
              | none =>
                IO.eprintln <|
                  ("[PINN] Weights JSON did not match expected shapes; falling back to " ++
                    "built-in weights")
                pure (buildGraph2D, seedParamsFloat2D)
          | none => pure (buildGraph2D, seedParamsFloat2D)
        let evalAt : Float → Float → Float → IO (Float × Float) :=
          fun xc yc epsc => do
            let ps := seedInputFloat2D ps0 xc yc epsc
            let prims ← computePrimsAt g ps uMethod backend
            match eval prims expr with
            | some (lo, hi) => pure (lo, hi)
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
        IO.eprintln s!"invalid float input(s): x={xStr}, y={yStr}, eps={epsStr}"
  | _ =>
    throw <| IO.userError <|
      ("Usage:\n  lake exe verify -- pinn-cli -- " ++
        "[--method=ibp|crown-fwd|crown-bwd|partial] [--split-depth=N] " ++
        "[--backend=float|neuralfloat] [--weights=path.json] \"<PDE>\" x eps   " ++
        " # 1D\n  lake exe verify -- pinn-cli -- " ++
        "[--method=ibp|crown-fwd|crown-bwd|partial] [--split-depth=N] " ++
        "[--backend=float|neuralfloat] [--weights=path.json] \"<PDE>\" x y eps " ++
        " # 2D")

end NN.Verification.PINN.CLI
