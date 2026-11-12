/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Affine
public import TorchLean.NN.IR.Graph
public import TorchLean.Verify.Certificate.DecimalJSON
public import TorchLean.Verify.DecimalRounding
public import TorchLean.Verify.Certificate.DecimalJSONRoundTrip -- shake: keep
public import TorchLean.Verify.Tensor

/-!
# Node-Certificate Replay

Parsing, exact comparison, and local replay for node-wise verification certificates.

The IBP and CROWN checkers consume the same basic artifact shapes: flat interval boxes, affine
lower/upper bounds, and optional per-node vectors. Those format-level operations live here so each
checker can focus on its propagation rule. `IBPNode.check` checks interval propagation.
`CROWNNode.replayDiagnostic` compares affine entries with Lean's binary32 α-CROWN replay, with
optional ReLU phases, and `CROWNNode.check` decides a certificate by the exact rational checks of
`FiniteArtifact.accepts`.

The JSON artifact is always untrusted. The helpers here only parse and compare data.
`IBPNode.check` and `CROWNNode.replayDiagnostic` recompute each bound inside Lean and compare it
with the artifact. Numbers are read as exact signed decimals (`TorchLean.Certificate.DecimalJson`),
so `-0.0` is the negative zero.

Each box endpoint and affine coefficient stands for its rounded binary32 value. The decimal is
converted once by FloatLib's `Model.convertDecimalText`, as in IEEE 754-2019 section 5.12, outward
for an interval endpoint and to nearest for a coefficient, and it overflows as section 7.4 says for
that rounding direction. To nearest, a magnitude of at least `2 ^ 128 - 2 ^ 103`, halfway between
the largest finite value and `2 ^ 128`, is the infinity of its sign, and a smaller one is finite. So
`1e39` is `+∞` as an upper endpoint or a coefficient and the largest finite value as a lower
endpoint, and `-1e39` symmetrically. An α entry is rounded to nearest after its decimal is checked
to lie in `[0, 1]`, and the integers of `ctx` and of the phases are read exactly.

Interval claims are compared by containment, and an affine transcript agrees with the replay
only when it equals the executable binary32 result bit for bit. A matrix entry must be finite. An
offset may be the infinity that bounds nothing on its side, `-∞` in `loC` and `+∞` in `hiC`, but
not the other infinity. The writers `flatBoxValue` and `affineBoundsValue?` emit each finite
binary32 value as its exact decimal, which these readers round back to the same value, and an
infinity as `±1e39`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Certificate.NodeReplay

open CROWN
open Graph
open DecimalJson (Value)
open FloatLib.Numerics.DecimalText (Decimal)
open _root_.TorchLean.Tensor

/--
Whether `outer` contains `inner` componentwise, and `false` for boxes of different dimensions.

The comparison has no tolerance after decimal decoding. The decoded lower endpoint may lie farther
down and the upper endpoint farther up, but neither may lie inward of the recomputed box.
-/
def flatBoxContains {α : Type} [Storage α] [Context α] (outer inner : FlatBox α) : Bool :=
  if h : outer.dim = inner.dim then
    Tensor.boundsOrdered outer.lo (castDimScalar h.symm inner.lo) &&
      Tensor.boundsOrdered (castDimScalar h.symm inner.hi) outer.hi
  else
    false

/-!
## Proved local replay

The node diagnostics of `CROWNNode.mismatches` replay bounds incrementally so that they can report
the first bad node. `CROWNNode.replayAgrees` also runs the pure test `isReplay` on the complete
imported certificate, over a graph of `size` nodes, and `isReplay_spec` states that every entry is
the replay of the entries. `TorchLean.CROWN.Op.Alpha.encloses_of_run` takes that consistency over
`ℝ` as a hypothesis. In binary32 the α-CROWN replay rounds to nearest, and an entry equal to its
replay need not enclose its node's value. So agreement with the replay is a diagnostic, not a
safety verdict.
-/

/--
Whether the certificate has `size` affine entries, each equal to `step` applied to the imported
certificate at its index.

The test covers only the entries. Parsing, IBP containment and shapes are tested separately.
-/
def isReplay {α : Type} [Storage α] [Context α] [DecidableEq α]
    (size : Nat)
    (step : Array (Option (FlatAffineBounds α)) → Nat → Option (FlatAffineBounds α))
    (cert : Array (Option (FlatAffineBounds α))) : Bool :=
  if hsize : cert.size = size then
    (List.finRange size).all fun i =>
      decide ((cert[i.val]'(by rw [hsize]; exact i.isLt)) = step cert i.val)
  else
    false

/-- A certificate that passes `isReplay` is locally consistent: `size` entries, each the replay of
the entries. -/
theorem isReplay_spec {α : Type} [Storage α] [Context α] [DecidableEq α]
    (size : Nat)
    (step : Array (Option (FlatAffineBounds α)) → Nat → Option (FlatAffineBounds α))
    (cert : Array (Option (FlatAffineBounds α)))
    (haccept : isReplay size step cert = true) :
    cert.size = size ∧ ∀ id : Nat, id < size → cert[id]! = step cert id := by
  unfold isReplay at haccept
  split at haccept
  next hsize =>
    refine ⟨hsize, ?_⟩
    intro id hid
    have hentry := List.all_eq_true.mp haccept ⟨id, hid⟩ (List.mem_finRange ⟨id, hid⟩)
    have heq := of_decide_eq_true hentry
    have hidCert : id < cert.size := by rw [hsize]; exact hid
    rw [getElem!_pos cert id hidCert]
    exact heq
  next => simp at haccept

/--
Whether the node diagnostics passed (`diagnosticsOk`) and the certificate passes `isReplay`. This is
agreement with the replay, not a safety verdict.
-/
def agrees {α : Type} [Storage α] [Context α] [DecidableEq α]
    (size : Nat)
    (step : Array (Option (FlatAffineBounds α)) → Nat → Option (FlatAffineBounds α))
    (cert : Array (Option (FlatAffineBounds α)))
    (diagnosticsOk : Bool) : Bool :=
  diagnosticsOk && isReplay size step cert

/-- A certificate that passes `agrees` is locally consistent: `size` entries, each the replay of
the entries. -/
theorem agrees_spec {α : Type} [Storage α] [Context α] [DecidableEq α]
    (size : Nat)
    (step : Array (Option (FlatAffineBounds α)) → Nat → Option (FlatAffineBounds α))
    (cert : Array (Option (FlatAffineBounds α)))
    (diagnosticsOk : Bool)
    (haccept : agrees size step cert diagnosticsOk = true) :
    cert.size = size ∧ ∀ id : Nat, id < size → cert[id]! = step cert id := by
  simp only [agrees, Bool.and_eq_true] at haccept
  exact isReplay_spec size step cert haccept.2

/-- The decimals of a length-`dim` JSON vector, with `ctx` naming the vector in the error. -/
def decimalVec (ctx : String) (dim : Nat) (v : Value) : IO (Fin dim → Decimal) := do
  let some ds := v.decimals? dim
    | throw <| IO.userError s!"Invalid {ctx}: expected numeric array length {dim}"
  pure ds

/-- The decimals of a `rows × cols` JSON matrix, with `ctx` naming the matrix in the error. -/
def decimalMatrix (ctx : String) (rows cols : Nat) (v : Value) :
    IO (Fin rows → Fin cols → Decimal) := do
  let err := IO.userError s!"Invalid {ctx}: expected numeric matrix {rows}x{cols}"
  let .arr xs := v | throw err
  let some parsed := xs.mapM (Value.decimals? cols) | throw err
  if h : parsed.size = rows then
    pure fun i => parsed[i.val]'(by rw [h]; exact i.isLt)
  else
    throw err

/-- A binary32 vector of decimals, each rounded once with `mode`. -/
def binary32Vec (mode : Model.IEEERoundingMode) (dim : Nat) (ds : Fin dim → Decimal) :
    Tensor (ExecFloat.Binary 8 23) [dim] :=
  Tensor.ofFn fun i =>
    ExecFloat.Binary.ofModel (Model.convertDecimalText .binary32 mode (ds i)).value

/-- A binary32 matrix of decimals, each rounded once to nearest. -/
def binary32Matrix (rows cols : Nat) (ds : Fin rows → Fin cols → Decimal) :
    Tensor (ExecFloat.Binary 8 23) [rows, cols] :=
  Tensor.matrix fun i k =>
    ExecFloat.Binary.ofModel (Model.convertDecimalText .binary32 .nearest (ds i k)).value

/--
Parse a flat interval box `{"lo": [...], "hi": [...]}` from JSON, with `null` for no box.

Each endpoint is an exact decimal, each lower one at most its upper one, rounded once into binary32:
lower endpoints toward `-∞` and upper endpoints toward `+∞`. The parsed box is therefore the
smallest binary32 box containing the decimal box the artifact states. A decimal beyond the binary32
range rounds outward, so `1e39` is `+∞` as an upper endpoint and the largest finite value as a
lower one, and `-1e39` symmetrically. Containment is then checked exactly on binary32 values, and a
decimal endpoint that lies inward of Lean's endpoint by less than one binary32 ulp is read as
Lean's value. A serialization that prints each endpoint as its shortest round-trip decimal passes
the check.
-/
def parseFlatBox? (dim : Nat) (v : Value) : IO (Option (FlatBox (ExecFloat.Binary 8 23))) := do
  match v with
  | .null => pure none
  | _ =>
      let lo ← decimalVec "ibp[i].lo" dim (← IO.ofExcept (v.field "lo" "ibp[i]"))
      let hi ← decimalVec "ibp[i].hi" dim (← IO.ofExcept (v.field "hi" "ibp[i]"))
      unless (List.finRange dim).all fun i => DecimalJson.Decimal.le (lo i) (hi i) do
        throw <| IO.userError "Invalid ibp[i]: every lower bound must be <= its upper bound"
      pure (some { dim := dim, lo := binary32Vec .down dim lo, hi := binary32Vec .up dim hi })

/--
Parse an optional α vector for α-CROWN ReLU relaxations, with `null` for no vector.

The soundness theorem for the lower ReLU relaxation assumes every α component is in `[0, 1]`. The
parser checks that range on the exact decimals at the JSON boundary, so a certificate whose α breaks
the theorem's hypothesis is refused before any replay. Each entry is then rounded once to the
nearest binary32 value. Rounding is monotone and `0` and `1` are exact, so the range survives the
conversion.
-/
def parseAlphaVec? (dim : Nat) (v : Value) (ctx : String := "alpha[i]") :
    IO (Option (FlatTensor (ExecFloat.Binary 8 23))) := do
  match v with
  | .null => pure none
  | _ =>
      let a ← decimalVec ctx dim v
      let zero : Decimal := { negative := false, significand := 0, exponent := 0 }
      let one : Decimal := { negative := false, significand := 1, exponent := 0 }
      for k in List.finRange dim do
        unless DecimalJson.Decimal.le zero (a k) && DecimalJson.Decimal.le (a k) one do
          let got := DecimalJson.Decimal.render (a k)
          throw <| IO.userError
            s!"Invalid {ctx}[{k.val}]: α-CROWN requires 0 ≤ alpha ≤ 1, got {got}"
      pure (some { n := dim, v := binary32Vec .nearest dim a })

/-- The bits of the binary32 infinity that bounds nothing on the side of an affine offset: `+∞` for
an upper bound (`upper`) and `-∞` for a lower one. -/
def vacuousOffsetBits (upper : Bool) : Nat :=
  if upper then 0x7F800000 else 0xFF800000

/-- Whether a binary32 affine form `x ↦ W x + b` of an upper bound (`upper`) or a lower one has
finite matrix entries and offsets that are finite or the infinity `vacuousOffsetBits upper`. -/
def affineFormOk {n m : Nat} (upper : Bool) (a : Linear (ExecFloat.Binary 8 23) n m) :
    Bool :=
  (List.finRange m).all fun i =>
    ((a.bias.getScalar i).isFinite || (a.bias.getScalar i).toNatBits == vacuousOffsetBits upper) &&
      (List.finRange n).all fun j => (Tensor.get2 a.weights i j).isFinite

/--
Parse flattened affine bounds `{"loA": [[...]], "loC": [...], "hiA": [[...]], "hiC": [...]}` from
JSON, with `null` for no bound.

The replay diagnostic compares the affine transcript with Lean's binary32 replay bit for bit, so
each exact decimal coefficient is read as the binary32 value nearest to it, with a single rounding
that keeps the sign of a zero. A matrix entry whose binary32 value is not finite is refused. An
offset may be the infinity that bounds nothing on its side, `-∞` in `loC` and `+∞` in `hiC` (written
`-1e39` and `1e39`, as by `endpointValue`), and an offset rounding to the other infinity is refused.
-/
def parseAffineBounds? (inDim outDim : Nat) (v : Value) :
    IO (Option (FlatAffineBounds (ExecFloat.Binary 8 23))) := do
  match v with
  | .null => pure none
  | _ =>
      let field (k : String) : IO Value := IO.ofExcept (v.field k "crown[i]")
      let loA ← decimalMatrix "crown[i].loA" outDim inDim (← field "loA")
      let loC ← decimalVec "crown[i].loC" outDim (← field "loC")
      let hiA ← decimalMatrix "crown[i].hiA" outDim inDim (← field "hiA")
      let hiC ← decimalVec "crown[i].hiC" outDim (← field "hiC")
      let loAff : Linear (ExecFloat.Binary 8 23) inDim outDim :=
        { weights := binary32Matrix outDim inDim loA, bias := binary32Vec .nearest outDim loC }
      let hiAff : Linear (ExecFloat.Binary 8 23) inDim outDim :=
        { weights := binary32Matrix outDim inDim hiA, bias := binary32Vec .nearest outDim hiC }
      unless affineFormOk false loAff && affineFormOk true hiAff do
        throw <| IO.userError
          "Invalid crown[i]: an affine coefficient overflows binary32 (only loC to -∞, hiC to +∞)"
      pure (some { inDim := inDim, outDim := outDim, loAff := loAff, hiAff := hiAff })

/--
Parse one JSON entry per graph node, where `dims` holds the flattened output size of each node. The
array must have exactly `dims.size` entries, and each entry is parsed with its node's output size.
-/
def parsePerNode {β : Type} (dims : Array Nat) (label : String) (arr : Array Value)
    (parse : Nat → Value → IO β) : IO (Array β) := do
  unless arr.size = dims.size do
    throw <| IO.userError s!"{label} length {arr.size} ≠ node count {dims.size}"
  (dims.zip arr).mapM fun (dim, entry) => parse dim entry

/-- The flattened size of output `0` of a node of a graph of registered operators, which is the
length of its certificate entries, and `0` for a node without outputs. -/
def outputSize (n : IR.Node) : Nat :=
  (n.outs.head?.map (·.shape.size)).getD 0

/--
A node-wise CROWN certificate: the affine context, the interval and affine bounds per node, the
optional α vectors of the ReLU lower relaxations, and optional ReLU phases.
-/
structure _root_.TorchLean.Certificate.CROWNNode where
  /-- Affine-propagation context, including the chosen input node and flattened input dimension. -/
  ctx : AffineCtx
  /-- Per-node interval boxes, `null` at a node without a scalar output.
  `CROWNNode.replayDiagnostic` reports each box that does not contain Lean's recomputed box, and
  its replay reads Lean's boxes. `CROWNNode.check` does not read them. -/
  ibp : Array (Option (FlatBox (ExecFloat.Binary 8 23)))
  /-- Optional per-node affine lower/upper bounds. -/
  crown : Array (Option (FlatAffineBounds (ExecFloat.Binary 8 23)))
  /-- Optional per-node α values for ReLU lower relaxations. -/
  alpha : Array (Option (FlatTensor (ExecFloat.Binary 8 23)))
  /-- Optional per-node ReLU phases: `-1` inactive, `0` unconstrained, `1` active. A phase must
  be one that Lean's IBP interval already proves, or the replay gives the node no bound and
  `CROWNNode.replayDiagnostic` reports it. `CROWNNode.check` does not read the phases. -/
  phases : Array (Option (Array Int)) := #[]

/-- One node's ReLU phases from JSON: `null`, or `dim` integers each `-1`, `0` or `1`. -/
def parsePhases? (dim : Nat) (v : Value) : IO (Option (Array Int)) := do
  match v with
  | .null => pure none
  | .arr xs =>
      unless xs.size = dim do
        throw <| IO.userError s!"Invalid phases[i]: expected int array length {dim}"
      let mut out : Array Int := Array.mkEmpty dim
      for h : k in [0:xs.size] do
        let some i := xs[k].int?
          | throw <| IO.userError s!"Invalid phases[i][{k}]: expected int"
        unless i = (-1) || i = 0 || i = 1 do
          throw <| IO.userError s!"Invalid phases[i][{k}]: expected -1/0/1"
        out := out.push i
      pure (some out)
  | _ => throw <| IO.userError "Invalid phases[i]: expected null or int array"

/--
Parse a CROWN node certificate for a graph whose nodes have the flattened output sizes `dims`.

The producer may omit `"alpha"` and `"phases"`, and then no node has a custom α vector or a phase.
Whenever α is present, `parseAlphaVec?` enforces the `[0,1]` side condition of the ReLU relaxation
proof.
-/
def _root_.TorchLean.Certificate.CROWNNode.parse (dims : Array Nat) (top : Value) :
    IO CROWNNode := do
  let ctxV ← IO.ofExcept <| top.field "ctx" "top-level"
  let inputId ← IO.ofExcept <| ctxV.field "inputId" "ctx" >>= (·.nat "ctx.inputId")
  let inputDim ← IO.ofExcept <| ctxV.field "inputDim" "ctx" >>= (·.nat "ctx.inputDim")
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }

  let ibpArr ← IO.ofExcept <| top.field "ibp" "top-level" >>= (·.array "top-level.ibp")
  let crownArr ← IO.ofExcept <| top.field "crown" "top-level" >>= (·.array "top-level.crown")
  let alphaArr ←
    match ← IO.ofExcept (top.optionalField? "alpha" "top-level") with
    | none => pure (Array.replicate dims.size Value.null)
    | some alphaV => IO.ofExcept (alphaV.array "top-level.alpha")

  let ibp ← parsePerNode dims "ibp" ibpArr parseFlatBox?
  let crown ← parsePerNode dims "crown" crownArr (parseAffineBounds? ctx.inputDim)
  let alpha ← parsePerNode dims "alpha" alphaArr fun dim entry => parseAlphaVec? dim entry
  let phasesArr ←
    match ← IO.ofExcept (top.optionalField? "phases" "top-level") with
    | none => pure (Array.replicate dims.size Value.null)
    | some phasesV => IO.ofExcept (phasesV.array "top-level.phases")
  let phases ← parsePerNode dims "phases" phasesArr parsePhases?
  pure { ctx, ibp, crown, alpha, phases }

/-! ## Writing certificate entries -/

/-- A box endpoint as a JSON number: its exact decimal with the sign of zero, `1e39` for `+∞` and
`-1e39` for `-∞`, and `null` for NaN, which `parseFlatBox?` refuses. -/
def endpointValue (x : ExecFloat.Binary 8 23) : Value :=
  match DecimalJson.Decimal.ofBinary32? x with
  | some d => .num d
  | none =>
      if decide (0 < x) then .num { negative := false, significand := 1, exponent := 39 }
      else if decide (x < 0) then .num { negative := true, significand := 1, exponent := 39 }
      else .null

/-- A flat interval box as the JSON object `{"lo": [...], "hi": [...]}` of its endpoints. Reading
it with `parseFlatBox?` rounds `±1e39` outward, so the box read contains the box written. -/
def flatBoxValue (B : FlatBox (ExecFloat.Binary 8 23)) : Value :=
  .obj #[("lo", .arr (Array.ofFn fun i => endpointValue (B.lo.getScalar i))),
    ("hi", .arr (Array.ofFn fun i => endpointValue (B.hi.getScalar i)))]

/-- A binary32 coefficient as its exact decimal JSON number, the negative zero as `-0.0`. Only
finite values are written, so an infinity or NaN gives `none`. -/
def coefficientValue? (x : ExecFloat.Binary 8 23) : Option Value :=
  (DecimalJson.Decimal.ofBinary32? x).map .num

/-- An offset of an upper bound (`upper`) or a lower one as a JSON number: its exact decimal, or
`endpointValue` of the infinity `vacuousOffsetBits upper`, and `none` for the other infinity and
NaN. -/
def offsetValue? (upper : Bool) (x : ExecFloat.Binary 8 23) : Option Value :=
  match DecimalJson.Decimal.ofBinary32? x with
  | some d => some (.num d)
  | none => if x.toNatBits == vacuousOffsetBits upper then some (endpointValue x) else none

/-- An affine form `x ↦ W x + b` of an upper bound (`upper`) or a lower one as the JSON values of
its matrix and its offset vector, or `none` unless `affineFormOk upper` holds. -/
def affineValues? {n m : Nat} (upper : Bool) (a : Linear (ExecFloat.Binary 8 23) n m) :
    Option (Value × Value) := do
  let rows ← (Array.ofFn fun (i : Fin m) =>
    (Array.ofFn fun (j : Fin n) => coefficientValue? (Tensor.get2 a.weights i j)).mapM id).mapM id
  let c ← (Array.ofFn fun i => offsetValue? upper (a.bias.getScalar i)).mapM id
  pure (.arr (rows.map .arr), .arr c)

/-- Affine bounds as the JSON object `{"loA", "loC", "hiA", "hiC"}` that `parseAffineBounds?`
reads, or `none` when a matrix entry is not finite or an offset is NaN or the infinity on its
bounded side (`+∞` in `loC`, `-∞` in `hiC`). -/
def affineBoundsValue? (B : FlatAffineBounds (ExecFloat.Binary 8 23)) : Option Value := do
  let (loA, loC) ← affineValues? false B.loAff
  let (hiA, hiC) ← affineValues? true B.hiAff
  pure (.obj #[("loA", loA), ("loC", loC), ("hiA", hiA), ("hiC", hiC)])

/-- Pretty-printer for a flat box, used in certificate mismatch messages. -/
def prettyFlatBox {α : Type} [Storage α] [Context α] [ToString α] (B : FlatBox α) : String :=
  s!"dim={B.dim}, lo={Tensor.pretty B.lo}, hi={Tensor.pretty B.hi}"

/-- Pretty-printer for affine bounds, used in certificate mismatch messages. -/
def prettyAffineBounds {α : Type} [Storage α] [Context α] [ToString α]
    (B : FlatAffineBounds α) : String :=
  s!"inDim={B.inDim}, outDim={B.outDim}, " ++
    s!"loA={Tensor.pretty B.loAff.weights}, loC={Tensor.pretty B.loAff.bias}, " ++
    s!"hiA={Tensor.pretty B.hiAff.weights}, hiC={Tensor.pretty B.hiAff.bias}"

end TorchLean.Certificate.NodeReplay
