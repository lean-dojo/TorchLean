/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Runtime.PyTorch.Import.Core
public import NN.Verification.Util.Json

/-!
# MnistFcVerify

VNN-COMP-style mini-suite checker (MNIST-FC, VNN-COMP 2022 benchmark).

This tool is intentionally compact:
- it consumes ONNX+VNNLIB instances via JSON artifacts, and
- it runs TorchLean's IBP / (basic) CROWN bounds on the imported MLP, then checks the VNNLIB
  disjunction-of-conjunctions constraints using a sufficient condition on the output box.

The checker expects exported JSON artifacts. Keep large VNN-COMP snapshots outside git, for example
under `_external/vnncomp/mnist_fc/`, and pass explicit paths when needed.

Run (Lean):
  `lake exe verify -- vnncomp-mnistfc`
-/

@[expose] public section


namespace NN.Examples.Verification.VNNComp.MnistFcVerify

open Lean
open Data
open Json
open _root_.Spec
open Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.Json
open Import.PyTorch

/--
Command-line options for the MNIST-FC VNN-COMP mini-suite verifier.

The defaults point at ignored local artifact paths under `_external/vnncomp/mnist_fc/`.
-/
structure MnistFCOpts where
  /-- Path to exported weights (`model_weights.json`). -/
  weights : String := "_external/vnncomp/mnist_fc/model_weights.json"
  /-- Path to exported instance suite (`suite.json`). -/
  suite   : String := "_external/vnncomp/mnist_fc/suite.json"
  /-- Maximum number of instances to check (useful for quick local checks). -/
  max     : Nat := 30
  /--
  Bound propagation mode:
  - `ibp`: IBP only (fast, loose)
  - `crown`: forward CROWN (still produces just an output box)
  - `crownobj`: backward objective pass for each spec row
  - `crownobj-alpha`: objective pass with precomputed ReLU slopes (`--alphas`)
  -/
  mode    : String := "ibp"
  /-- Path to exported alpha slopes JSON (required for `mode = crownobj-alpha`). -/
  alphas  : String := ""
  deriving Repr

/-- Usage text printed by this tool (returned as an error message on `--help`). -/
def usage : String :=
  String.intercalate "\n" [
    "Usage:",
    "  lake exe verify -- vnncomp-mnistfc",
    ("    [--weights=PATH.json] [--suite=PATH.json] [--max=30] " ++
      "[--mode=ibp|crown|crownobj|crownobj-alpha]"),
    "    [--alphas=PATH.json]   (required for crownobj-alpha)",
    "",
    "Default inputs (ignored local artifacts):",
    "  _external/vnncomp/mnist_fc/model_weights.json",
    "  _external/vnncomp/mnist_fc/suite.json",
    "",
    "Large VNN-COMP exports are not committed. Generate or download them first, then pass",
    "--weights=... and --suite=... explicitly if you store them elsewhere."
  ]

/--
Parse CLI flags into `MnistFCOpts`.

On `--help` / `-h` this returns `usage` as the error message (so callers can print it and exit).
-/
def parseArgs : List String → Except String MnistFCOpts
  | [] => .ok {}
  | a :: rest =>
    match parseArgs rest with
    | .error e => .error e
    | .ok o =>
      if a == "--help" || a == "-h" then
        .error usage
      else if a.startsWith "--weights=" then
        .ok { o with weights := (a.drop 10).toString }
      else if a.startsWith "--suite=" then
        .ok { o with suite := (a.drop 8).toString }
      else if a.startsWith "--max=" then
        match (a.drop 6).toString.toNat? with
        | some n => .ok { o with max := n }
        | none => .error s!"bad --max: {a}"
      else if a.startsWith "--mode=" then
        .ok { o with mode := (a.drop 7).toString }
      else if a.startsWith "--alphas=" then
        .ok { o with alphas := (a.drop 9).toString }
      else
        .error s!"unknown arg: {a}\n\n{usage}"

/--
Weights for one exported fully-connected layer (`y = Wx + b`).

This is the lightest data structure we need to rebuild a CROWN `Graph` for MNIST-FC.
-/
structure LayerWB where
  /-- Input dimension for this layer. -/
  inDim : Nat
  /-- Output dimension for this layer. -/
  outDim : Nat
  /-- Weight matrix, shape `(outDim × inDim)`. -/
  w : Tensor Float (.dim outDim (.dim inDim .scalar))
  /-- Bias vector, shape `(outDim)`. -/
  b : Tensor Float (.dim outDim .scalar)

/-- State-dict keys for the `i`-th linear layer exported by the Python script. -/
def keysForLayer (i : Nat) : (String × String) :=
  (s!"layers.{i}.weight", s!"layers.{i}.bias")

/-- Convert a JSON float array into a length-`n` TorchLean vector tensor (returns `none` if sizes
  mismatch). -/
def tensorVecOfArray (n : Nat) (arr : Array Float) : Option (Tensor Float (.dim n .scalar)) := by
  classical
  if h : arr.size = n then
    -- Build by indexing; safe because of `h`.
    let t : Tensor Float (.dim n .scalar) :=
      Tensor.dim (fun i => Tensor.scalar (arr[i.val]!))
    exact some t
  else
    exact none

/-- Parse a JSON matrix payload (array-of-array-of-floats) into `Array (Array Float)`. -/
def parseFloatMatrixArray (j : Json) : Option (Array (Array Float)) := do
  match j with
  | .arr rows =>
      let mut out : Array (Array Float) := #[]
      for r in rows do
        let xs ← NN.Verification.Json.parseFloatArray r
        out := out.push xs
      pure out
  | _ => none

/-- Convert a `rows × cols` float matrix payload into a TorchLean matrix tensor (checks both
  dimensions). -/
def tensorMatOfArray (rows cols : Nat) (m : Array (Array Float)) :
    Option (Tensor Float (.dim rows (.dim cols .scalar))) :=
  if m.size != rows then
    none
  else
    let okCols := (List.finRange rows).all (fun i => (m[i.val]!).size = cols)
    if !okCols then
      none
    else
      some <|
        Tensor.dim (fun i =>
          Tensor.dim (fun j =>
            Tensor.scalar ((m[i.val]!)[j.val]!)))

/-- Deduplicate a sorted list of natural numbers (used to normalize layer indices discovered from
  JSON keys). -/
def dedupSortedNat (xs : Array Nat) : Array Nat :=
  xs.foldl (init := #[]) (fun acc x =>
    match acc[acc.size - 1]? with
    | some y => if y = x then acc else acc.push x
    | none => acc.push x)

/--
Load an exported MNIST-FC model from JSON weights.

This expects keys like `layers.0.weight`, `layers.0.bias`, etc, and checks that the linear layer
dimensions form a consistent chain.
-/
def loadWeights (path : String) : IO (Array LayerWB) := do
  let j ← readJsonObjectFile path
  let some sd := Import.PyTorch.loadWeights? j
    | throw <| IO.userError "Weights JSON must be an object (or {\"params\": {...}})"

  -- Discover layers by scanning keys of the form layers.<i>.weight
  let layerIdxs : Array Nat :=
    sd.foldl (init := #[]) (fun acc k _v =>
      match Import.PyTorch.parseIndexedKey "layers." ".weight" k with
      | some i => acc.push i
      | none => acc)
  let layerIdxsSorted := dedupSortedNat (layerIdxs.qsort (· < ·))
  if layerIdxsSorted.isEmpty then
    throw <| IO.userError "No keys of the form layers.<i>.weight found"

  let mut layers : Array LayerWB := #[]
  for i in layerIdxsSorted do
    let (kW, kB) := keysForLayer i
    let wJ ←
      match sd.get? kW with
      | some v => pure v
      | none => throw <| IO.userError s!"Missing key: {kW}"
    let bJ ←
      match sd.get? kB with
      | some v => pure v
      | none => throw <| IO.userError s!"Missing key: {kB}"
    let some (rows, cols) := Import.PyTorch.inferMatrixDims wJ
      | throw <| IO.userError s!"Bad matrix for {kW}"
    let some wArr := parseFloatMatrixArray wJ
      | throw <| IO.userError s!"Bad matrix payload for {kW}"
    let some w := tensorMatOfArray rows cols wArr
      | throw <| IO.userError s!"Bad matrix shape for {kW} (expected {rows}x{cols})"
    let some bArr := NN.Verification.Json.parseFloatArray bJ
      | throw <| IO.userError s!"Bad bias payload for {kB}"
    let some bT := tensorVecOfArray rows bArr
      | throw <| IO.userError s!"Bad bias shape for {kB} (expected length {rows})"
    layers := layers.push { inDim := cols, outDim := rows, w := w, b := bT }

  -- Basic chain check
  let rec checkChain : List LayerWB → IO Unit
    | [] | [_] => pure ()
    | a :: b :: rest =>
        if a.outDim != b.inDim then
          throw <| IO.userError s!"Layer dim mismatch: out={a.outDim} ≠ in={b.inDim}"
        else
          checkChain (b :: rest)
  checkChain layers.toList

  pure layers

/--
One exported VNN-COMP instance.

`spec` is a disjunction-of-conjunctions: each term is a conjunction `mat * y <= rhs` over the
network output vector `y`.
-/
structure Instance where
  /-- Instance id (copied from the exported suite JSON). -/
  id : Nat
  /-- Lower bound for the input box (`x_lo`). -/
  inputLo : Array Float
  /-- Upper bound for the input box (`x_hi`). -/
  inputHi : Array Float
  /-- Disjunction terms, each a conjunction `mat * y <= rhs`. -/
  spec : Array (Array (Array Float) × Array Float)

/-- Parse a matrix payload (`Array (Array Float)`) from JSON, throwing if the shape is not right. -/
def parseMat (j : Json) : IO (Array (Array Float)) := do
  match j with
  | .arr rows =>
      let mut out : Array (Array Float) := #[]
      for r in rows do
        let some xs := parseFloatArray r
          | throw <| IO.userError "spec.mat row must be float array"
        out := out.push xs
      pure out
  | _ => throw <| IO.userError "spec.mat must be array of float arrays"

/--
Load the exported VNN-COMP suite JSON.

This expects the `vnnlib_suite_v0_1` format produced by the Python export script.
-/
def loadSuite (path : String) : IO (Array Instance) := do
  let top ← readJsonObjectFile path
  expectFormat top "vnnlib_suite_v0_1"
  let instArr ← expectFieldArray top "instances" "top-level"
  let mut out : Array Instance := #[]
  for ex in instArr do
    let exo ← expectObj ex "instance"
    let id ← expectFieldNat exo "id" "instance"
    let lo ← expectFieldFloatArray exo "input_lo" "instance"
    let hi ← expectFieldFloatArray exo "input_hi" "instance"
    let specArr ← expectFieldArray exo "spec" "instance"
    let mut specOut : Array (Array (Array Float) × Array Float) := #[]
    for t in specArr do
      let termObj ← expectObj t "spec term"
      let matJ ← expectField termObj "mat" "spec term"
      let mat ← parseMat matJ
      let rhs ← expectFieldFloatArray termObj "rhs" "spec term"
      specOut := specOut.push (mat, rhs)
    out := out.push { id := id, inputLo := lo, inputHi := hi, spec := specOut }
  pure out

/--
One set of precomputed ReLU alpha slopes for `crownobj-alpha` mode.

This checker assumes the MNIST-FC graph has ReLU nodes at fixed ids (`2` and `4`), so we store the
alpha vectors in a format that can be mapped back to those nodes.
-/
structure AlphaEntry where
  /-- Instance id this alpha payload applies to. -/
  id : Nat
  /-- Alpha vectors for ReLU node 2, indexed by disjunct term. -/
  alpha2 : Array (Array Float)
  /-- Alpha vectors for ReLU node 4, indexed by disjunct term. -/
  alpha4 : Array (Array Float)

/--
Load the exported alpha slopes database (for `mode = crownobj-alpha`).

This expects the `mnist_fc_crownobj_alpha_v0_1` format produced by the Python export script.
-/
def loadAlphas (path : String) : IO (Array AlphaEntry) := do
  let top ← readJsonObjectFile path
  expectFormat top "mnist_fc_crownobj_alpha_v0_1"
  let instArr ← expectFieldArray top "instances" "top-level"
  let mut out : Array AlphaEntry := #[]
  for ex in instArr do
    let exo ← expectObj ex "alpha instance"
    let id ← expectFieldNat exo "id" "alpha instance"
    let alphaObj ← expectFieldObj exo "alpha" "alpha instance"
    let a2J ← expectField alphaObj "2" "alpha"
    let a4J ← expectField alphaObj "4" "alpha"
    let a2 ← parseMat a2J
    let a4 ← parseMat a4J
    out := out.push { id := id, alpha2 := a2, alpha4 := a4 }
  pure out

/--
Lower the loaded weights into a CROWN `Graph` + `ParamStore`.

For MNIST-FC we build:
`input -> linear -> relu -> linear -> relu -> linear`
and return the graph plus the inferred `inDim`, `outDim`, and output node id.
-/
def buildGraphAndParams (layers : Array LayerWB) : IO (Graph × ParamStore Float × Nat × Nat × Nat)
  := do
  let layersL := layers.toList
  let first ←
    match layersL with
    | [] => throw <| IO.userError "Empty layer list"
    | a :: _ => pure a
  let inDim := first.inDim
  let outDim := (layersL.getLastD first).outDim

  let rec go (parentId : Nat) (nodes : Array Node) (ps : ParamStore Float) (rem : List LayerWB) :
      (Array Node × ParamStore Float × Nat) :=
    match rem with
    | [] => (nodes, ps, parentId)
    | layer :: more =>
        let linId := parentId + 1
        let nodes := nodes.push
          { id := linId
            parents := [parentId]
            kind := .linear
            outShape := .dim layer.outDim .scalar }
        let ps := { ps with
          linearWB := ps.linearWB.insert linId
            { m := layer.outDim
              n := layer.inDim
              w := layer.w
              b := layer.b } }
        if more.isEmpty then
          go linId nodes ps more
      else
        let reluId := linId + 1
        let nodes := nodes.push
          { id := reluId
            parents := [linId]
            kind := .relu
            outShape := .dim layer.outDim .scalar }
        go reluId nodes ps more

  let nodes0 : Array Node :=
    #[{ id := 0
        parents := []
        kind := .input
        outShape := .dim inDim .scalar }]
  let ps0 : ParamStore Float := {}
  let (nodes, ps, outId) := go 0 nodes0 ps0 layersL
  let g : Graph := { nodes := nodes }
  pure (g, ps, inDim, outDim, outId)

/--
Check whether an *unsafe* VNNLIB spec is refuted by an output interval box.

Given an output box `y ∈ [yLo, yHi]`, we conservatively lower-bound each linear constraint row
`aᵀ y` and check if **some** constraint in each disjunct term is violated (`lb > rhs`).

If every disjunct term is refuted this way, the unsafe spec is unsatisfiable for this output box,
so the instance is certified **safe** (a sufficient condition).
-/
def vnnlibRefutedByOutputBox (yLo yHi : Array Float)
    (spec : Array (Array (Array Float) × Array Float)) : Bool :=
  let outDim := yLo.size
  if yHi.size != outDim then
    false
  else
    -- For each disjunct term, check it is refuted by some row lower bound > rhs.
    spec.all (fun (term : Array (Array Float) × Array Float) =>
      let mat := term.fst
      let rhs := term.snd
      if rhs.size != mat.size then
        false
      else
        (List.range mat.size).any (fun i =>
          let row := mat[i]!
          if row.size != outDim then
            false
          else
            let lb :=
              (List.range outDim).foldl (fun acc j =>
                let a := row[j]!
                let lo := yLo[j]!
                let hi := yHi[j]!
                acc + min (a * lo) (a * hi)) 0.0
            lb > rhs[i]!))

/-- Compute an output interval box using IBP (fast, loose). -/
def outputBoxIBP (g : Graph) (ps : ParamStore Float) (outId : Nat) : IO (Array Float × Array Float)
  := do
  let ibp := runIBP (α := Float) g ps
  let some outB := ibp[outId]! | throw <| IO.userError "IBP produced no output box"
  match outB.lo, outB.hi with
  | .dim vlo, .dim vhi =>
      let n := outB.dim
      let lo : Array Float := (List.finRange n).map (fun i => match vlo i with | .scalar x => x)
        |>.toArray
      let hi : Array Float := (List.finRange n).map (fun i => match vhi i with | .scalar x => x)
        |>.toArray
      pure (lo, hi)
  | _, _ => throw <| IO.userError "Unexpected tensor shape in output box"

/-- Compute an output interval box by running forward CROWN and evaluating the affine bounds on the
  input box. -/
def outputBoxCROWN (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId outId inDim : Nat) : IO (Array Float × Array Float) := do
  let ibp := runIBP (α := Float) g ps
  let ctx : AffineCtx := { inputId := inId, inputDim := inDim }
  let crown := runCROWN (α := Float) g ps ctx ibp
  match crown[outId]! with
  | none => throw <| IO.userError "CROWN produced no affine bound at output"
  | some outAff =>
      if hIn : outAff.inDim = inDim then
        if hXB : xB.dim = inDim then
          let xBox : Box Float (.dim outAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := Float) (n := xB.dim) (m := outAff.inDim)
                (by simpa [hXB] using hIn.symm) xB.lo
              hi := Tensor.castVecDim (α := Float) (n := xB.dim) (m := outAff.inDim)
                (by simpa [hXB] using hIn.symm) xB.hi }
          let loBox := AffineVec.evalOnBox (α := Float) outAff.loAff xBox
          let hiBox := AffineVec.evalOnBox (α := Float) outAff.hiAff xBox
          match loBox.lo, hiBox.hi with
          | .dim vlo, .dim vhi =>
              let n := outAff.outDim
              let lo : Array Float :=
                (List.finRange n).map (fun i => match vlo i with | .scalar x => x) |>.toArray
              let hi : Array Float :=
                (List.finRange n).map (fun i => match vhi i with | .scalar x => x) |>.toArray
              pure (lo, hi)
          | _, _ => throw <| IO.userError "Unexpected eval_on_box shape"
        else
          throw <| IO.userError "Input box dim mismatch"
      else
        throw <| IO.userError "Unexpected CROWN inDim mismatch"

/--
Compute per-node interval boxes to be used by the backward objective pass.

We start from IBP boxes and (when the input dimension is small) refine a small set of parent-node
intervals by evaluating forward CROWN affines on the input box. This makes objective bounds tighter
without paying the full cost of "evaluate every affine on the input box".
-/
def boxesForObjective (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId inDim : Nat) : IO (Array (Option (FlatBox Float))) := do
  -- Use IBP as a baseline and then refine boxes using forward CROWN affine bounds
  -- evaluated on the input box (tighter for deeper linear nodes).
  let ibp0 := runIBP (α := Float) g ps
  -- For large input dimensions, evaluating forward CROWN affines on the input box is too slow
  -- in this compact checker. Fall back to pure IBP boxes (still sound, just looser).
  if inDim > 64 then
    return ibp0
  let ctx : AffineCtx := { inputId := inId, inputDim := inDim }
  let crown := runCROWN (α := Float) g ps ctx ibp0
  let mut out : Array (Option (FlatBox Float)) := ibp0
  -- Only refine boxes for nodes that will be queried as parents of a unary relaxation
  -- in the backward objective pass (these are the only nodes where tighter pre-activation
  -- intervals matter for CROWN objective bounds).
  let mut need : Array Bool := Array.replicate g.nodes.size false
  for id in List.range g.nodes.size do
    let node := g.nodes[id]!
    match node.parents with
    | p1 :: _ =>
      let isUnaryRelax : Bool :=
        match node.kind with
        | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softmax _ | .layernorm _ => true
        | _ => false
      if isUnaryRelax then
        if hp : p1 < g.nodes.size then
          need := need.set! p1 true
        else
          pure ()
      else
        pure ()
    | _ => pure ()

  for p in List.range g.nodes.size do
    if need[p]! then
      match crown[p]! with
      | none => pure ()
      | some b =>
        if hXB : xB.dim = inDim then
          if hIn : b.inDim = inDim then
            let xBox : Box Float (.dim b.inDim .scalar) :=
              { lo := Tensor.castVecDim (α := Float) (n := xB.dim) (m := b.inDim) (by simpa [hXB]
                using hIn.symm) xB.lo
                hi := Tensor.castVecDim (α := Float) (n := xB.dim) (m := b.inDim) (by simpa [hXB]
                  using hIn.symm) xB.hi }
            let loB := AffineVec.evalOnBox (α := Float) b.loAff xBox
            let hiB := AffineVec.evalOnBox (α := Float) b.hiAff xBox
            let newB : FlatBox Float := { dim := b.outDim, lo := loB.lo, hi := hiB.hi }
            out := out.set! p (some newB)
          else
            pure ()
        else
          pure ()
    else
      pure ()
  pure out

/--
Try to refute a single spec row using a backward CROWN objective bound.

We build an objective `obj = row` and lower-bound `rowᵀ y` over the input box. If the lower bound is
strictly greater than `rhs`, then the constraint `rowᵀ y <= rhs` cannot hold.
-/
def refutesRowByCROWNObjective
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (ibp : Array (Option (FlatBox Float))) (ctx : AffineCtx)
    (outId inDim outDim : Nat) (row : Array Float) (rhs : Float) : IO Bool := do
  let some rowT := tensorVecOfArray outDim row
    | throw <| IO.userError "spec row dim mismatch"
  let obj : FlatVec Float := { n := outDim, v := rowT }
  let some bout := runCROWNBackwardObjective (α := Float) g ps ctx ibp outId obj
    | throw <| IO.userError "CROWN backward objective failed"
  if hIn : bout.inDim = inDim then
    if hXB : xB.dim = inDim then
      let xBox : Box Float (.dim bout.inDim .scalar) :=
        { lo := Tensor.castVecDim (α := Float) (n := xB.dim) (m := bout.inDim) (by simpa [hXB] using
          hIn.symm) xB.lo
          hi := Tensor.castVecDim (α := Float) (n := xB.dim) (m := bout.inDim) (by simpa [hXB] using
            hIn.symm) xB.hi }
      let loBox := AffineVec.evalOnBox (α := Float) bout.loAff xBox
      let lo := getAtOrZero loBox.lo [0]
      pure (lo > rhs)
    else
      throw <| IO.userError "Input box dim mismatch"
  else
      throw <| IO.userError "Unexpected objective inDim mismatch"

/--
Like `refutesRowByCROWNObjective`, but use precomputed ReLU alpha slopes.

This is a workflow hook for comparing against "fixed alpha" CROWN objective bounds exported from a
reference implementation.
-/
def refutesRowByCROWNObjectiveWithReluAlpha
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (ibp : Array (Option (FlatBox Float))) (inId outId inDim outDim : Nat)
    (row : Array Float) (rhs : Float)
    (reluAlpha : Array (Option (FlatVec Float))) : IO Bool := do
  let some rowT := tensorVecOfArray outDim row
    | throw <| IO.userError "spec row dim mismatch"
  let obj : FlatVec Float := { n := outDim, v := rowT }
  let ctx : AffineCtx := { inputId := inId, inputDim := inDim }
  let some loAff := runCROWNBackwardObjectiveLowerWithReluAlpha (α := Float) g ps ctx ibp outId obj
    reluAlpha
    | throw <| IO.userError "CROWN backward objective (alpha) failed"
  if hXB : xB.dim = inDim then
    -- `xB.lo`/`xB.hi` already have shape `.dim inDim .scalar` by `hXB`.
    let xBox : Box Float (.dim inDim .scalar) :=
      { lo := Tensor.castVecDim (α := Float) (n := xB.dim) (m := inDim) (by simp [hXB]) xB.lo
        hi := Tensor.castVecDim (α := Float) (n := xB.dim) (m := inDim) (by simp [hXB]) xB.hi }
    let loBox := AffineVec.evalOnBox (α := Float) loAff xBox
    let lo := getAtOrZero loBox.lo [0]
    pure (lo > rhs)
  else
    throw <| IO.userError "Input box dim mismatch"

/--
Refute a VNNLIB disjunction-of-conjunctions spec using per-row CROWN objective bounds.

This is strictly stronger than checking an output interval box, but it is also slower.
-/
def vnnlibRefutedByCROWNObjectives
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId outId inDim outDim : Nat)
    (spec : Array (Array (Array Float) × Array Float)) : IO Bool := do
  let ibp ← boxesForObjective g ps xB inId inDim
  let ctx : AffineCtx := { inputId := inId, inputDim := inDim }
  for term in spec do
    let mat := term.fst
    let rhs := term.snd
    if rhs.size != mat.size then
      return false
    let mut termRefuted := false
    for i in List.range mat.size do
      let row := mat[i]!
      if row.size != outDim then
        return false
      let ok ← refutesRowByCROWNObjective g ps xB ibp ctx outId inDim outDim row rhs[i]!
      if ok then
        termRefuted := true
        break
    if !termRefuted then
      return false
  return true

/--
Refute a VNNLIB spec using CROWN objectives with externally-provided ReLU alphas.

This mode is meant for apples-to-apples comparisons against alpha-CROWN style tools where the
alphas are optimized elsewhere and then imported into TorchLean for checking.
-/
def vnnlibRefutedByCROWNObjectivesAlpha
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId outId inDim outDim : Nat)
    (spec : Array (Array (Array Float) × Array Float))
    (alphas : AlphaEntry) (hid1 hid2 : Nat) : IO Bool := do
  let ibp ← boxesForObjective g ps xB inId inDim
  for termIdx in List.range spec.size do
    let term := spec[termIdx]!
    let mat := term.fst
    let rhs := term.snd
    if rhs.size != mat.size then
      return false
    -- Build per-node α vector for this disjunct term (objective index = termIdx).
    let a2Row? := alphas.alpha2[termIdx]?
    let a4Row? := alphas.alpha4[termIdx]?
    let mut reluAlpha : Array (Option (FlatVec Float)) := Array.replicate g.nodes.size none
    match a2Row?, a4Row? with
    | some a2Row, some a4Row =>
      let some a2T := tensorVecOfArray hid1 a2Row
        | throw <| IO.userError s!"Alpha dim mismatch for node 2 (expected {hid1})"
      let some a4T := tensorVecOfArray hid2 a4Row
        | throw <| IO.userError s!"Alpha dim mismatch for node 4 (expected {hid2})"
      if 2 < g.nodes.size then
        reluAlpha := reluAlpha.set! 2 (some { n := hid1, v := a2T })
      else
        throw <| IO.userError "Graph too small for ReLU node 2"
      if 4 < g.nodes.size then
        reluAlpha := reluAlpha.set! 4 (some { n := hid2, v := a4T })
      else
        throw <| IO.userError "Graph too small for ReLU node 4"
    | _, _ =>
      throw <| IO.userError s!"Missing alphas for instance {alphas.id} termIdx={termIdx}"

    let mut termRefuted := false
    for i in List.range mat.size do
      let row := mat[i]!
      if row.size != outDim then
        return false
      let ok ← refutesRowByCROWNObjectiveWithReluAlpha g ps xB ibp inId outId inDim outDim row
        rhs[i]! reluAlpha
      if ok then
        termRefuted := true
        break
    if !termRefuted then
      return false
  return true

/--
CLI entry point.

This is wired into `lake exe verify -- vnncomp-mnistfc` (see `NN/Examples/README.md`).
-/
def main (args : List String) : IO Unit := do
  IO.println "== TorchLean VNN-COMP mini-suite: MNIST-FC (vnncomp2022) =="
  let args :=
    match args with
    | "--" :: rest => rest
    | _ => args
  let opts ←
    match parseArgs args with
    | .ok o => pure o
    | .error msg => throw <| IO.userError msg
  let layers ← loadWeights opts.weights
  let instances0 ← loadSuite opts.suite
  let instances := instances0.take opts.max
  let alphaDB? ←
    if opts.mode = "crownobj-alpha" then
      if opts.alphas.isEmpty then
        throw <| IO.userError "--alphas is required for --mode=crownobj-alpha"
      else
        pure (some (← loadAlphas opts.alphas))
    else
      pure none
  let (g, ps0, inDim, outDim, outId) ← buildGraphAndParams layers
  let inId := 0
  IO.println s!"[mnist_fc] layers={layers.size} nodes={g.nodes.size} inDim={inDim} outDim={outDim}"
  IO.println s!"[mnist_fc] instances={instances.size} mode={opts.mode}"

  let mut safe : Nat := 0
  let mut unknown : Nat := 0
  for inst in instances do
    if inst.inputLo.size != inDim || inst.inputHi.size != inDim then
      throw <| IO.userError s!"Instance {inst.id}: input dim mismatch"
    let some loT := tensorVecOfArray inDim inst.inputLo
      | throw <| IO.userError s!"Instance {inst.id}: bad input_lo"
    let some hiT := tensorVecOfArray inDim inst.inputHi
      | throw <| IO.userError s!"Instance {inst.id}: bad input_hi"
    let xB : FlatBox Float := { dim := inDim, lo := loT, hi := hiT }
    let ps : ParamStore Float := { ps0 with inputBoxes := ps0.inputBoxes.insert inId xB }
    let isSafe ←
      if opts.mode = "crownobj" then
        vnnlibRefutedByCROWNObjectives g ps xB inId outId inDim outDim inst.spec
      else if opts.mode = "crownobj-alpha" then
        let some alphaDB := alphaDB?
          | throw <| IO.userError "internal: alphaDB missing"
        let entry? : Option AlphaEntry :=
          alphaDB.foldl (init := none) (fun acc e =>
            match acc with
            | some _ => acc
            | none => if e.id = inst.id then some e else none)
        let some entry := entry?
          | throw <| IO.userError s!"No alpha entry for instance id {inst.id}"
        let (hid1, hid2) ←
          match layers[0]?, layers[1]? with
          | some l0, some l1 => pure (l0.outDim, l1.outDim)
          | _, _ => throw <| IO.userError "Need at least 2 hidden layers for MNIST-FC"
        vnnlibRefutedByCROWNObjectivesAlpha g ps xB inId outId inDim outDim inst.spec entry hid1
          hid2
      else
        let (yLo, yHi) ←
          if opts.mode = "crown" then
            outputBoxCROWN g ps xB inId outId inDim
          else
            outputBoxIBP g ps outId
        pure (vnnlibRefutedByOutputBox yLo yHi inst.spec)
    if isSafe then safe := safe + 1 else unknown := unknown + 1

  IO.println s!"[mnist_fc] safe={safe} unknown={unknown} (mode={opts.mode}; sufficient UNSAT check)"

end NN.Examples.Verification.VNNComp.MnistFcVerify
