/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.IR
public import NN.Verification.Numerical
public import NN.Spec.Core.Context.Real

/-!
# Pointwise operator execution

Each case is a typed IR artifact with one pointwise node, exact argument words, and the words
eager PyTorch 2.13 returns on CPU for the same arguments. `NN.Verification.IR.check` imports,
type checks and executes it with the CPU reference interpreter. A NaN result needs only to be a
NaN: the sign and payload of a NaN produced by an invalid operation are implementation choices,
and `compare` lets two NaNs agree for the same reason.

Integer cases follow PyTorch's integer kernels, except that the most negative 32-bit value divided
by `-1` with truncation wraps, where PyTorch's CPU kernel traps with `SIGFPE`. Conversions of
integers and binary64 to bfloat16 and binary16 run as one IR cast, which rounds the exact value
once, and as two casts through binary32, which return the words of PyTorch's `Tensor.to` and
XLA's `convert_element_type`.
Operators that FloatLib cannot evaluate in binary arithmetic must report themselves as
unsupported. The exact real meanings reject poles and inputs outside the real domain, and keep
the real value of a quotient that overflows binary32. The comparison cases check `compare`'s NaN
policy, `compare_bitwise`, that every mismatch fails, and that a Jaxpr failure lists XLA's
documented behaviours in `consistent_with` exactly when re-running the program as XLA's CPU code
computes it reproduces the reference words, which come from eager JAX or `jax.jit` 0.11.2 on CPU:
not for a reference from another platform, a value outside the window of XLA's `rsqrt` or power,
or a `jax.jit` reference whose primitive-by-primitive reference the IR does not reproduce.
-/

@[expose] public section

namespace NN.Tests.IR.Pointwise

open Lean

/-- The binary32 element descriptor in the artifact encoding. -/
def float32 : Json := "float32"

/-- A signed integer element descriptor in the artifact encoding. -/
def signed (width : Nat) : Json :=
  Json.mkObj [("kind", "integer"), ("signed", true), ("width", toJson width)]

/-- A rank-one tensor in the artifact encoding. -/
def tensorJson (dtype : Json) (words : List Nat) : Json :=
  Json.mkObj [("dtype", dtype), ("shape", toJson [words.length]), ("bits", toJson words)]

/-- A typed artifact with the given nodes, whose first `inputs` nodes are its arguments. -/
def program (nodes : List Json) (inputs : Nat) (arguments : List Json) : Json :=
  Json.mkObj [
    ("format", "torchlean.ir.v2"), ("input_ids", toJson (List.range inputs)),
    ("output_ids", toJson [nodes.length - 1]), ("nodes", Json.arr nodes.toArray),
    ("arguments", Json.arr arguments.toArray)]

/-- One node of a rank-one artifact. -/
def nodeJson (id : Nat) (kind : List (String × Json)) (parents : List Nat) (length : Nat)
    (dtype : Json) : Json :=
  Json.mkObj ([("id", toJson id), ("parents", toJson parents), ("shape", toJson [length]),
    ("dtype", dtype)] ++ kind)

/-- One pointwise node over input nodes holding `arguments`, every node at `dtype`. -/
def pointwiseProgram (kind : List (String × Json)) (dtype : Json)
    (arguments : List (List Nat)) : Json :=
  let length := (arguments.headD []).length
  let count := arguments.length
  let inputs := (List.range count).map fun id =>
    nodeJson id [("kind", "input")] [] length dtype
  program (inputs ++ [nodeJson count kind (List.range count) length dtype]) count
    (arguments.map (tensorJson dtype))

/-- Whether a binary32 word encodes a NaN. -/
def isNaN32 (word : Nat) : Bool :=
  (word / 2 ^ 23) % 256 == 255 && word % 2 ^ 23 != 0

/-- Run the structure, type and execution checks on an artifact. -/
def report (artifact : Json) : Json :=
  NN.Verification.IR.check artifact #["structure", "types", "execute"]

/-- The status and message of each requested check, for failure reports. -/
def summary (report : Json) : String :=
  (report.getObjValD "checks").compress

/-- Require execution and compare the first result with `expected`, NaN-aware for binary32. -/
def expectWords (name : String) (artifact : Json) (expected : List Nat)
    (floating : Bool := true) : IO Unit := do
  let report := report artifact
  unless report.getObjValD "ok" == .bool true do
    throw <| IO.userError s!"pointwise {name}: checks failed: {summary report}"
  let outputs ← IO.ofExcept ((report.getObjValD "outputs").getArr?)
  let some output := outputs[0]?
    | throw <| IO.userError s!"pointwise {name}: no output"
  let actual ← IO.ofExcept (fromJson? (α := List Nat) (output.getObjValD "bits"))
  unless actual.length == expected.length do
    throw <| IO.userError s!"pointwise {name}: {actual.length} results, expected {expected.length}"
  for (word, wanted) in actual.zip expected do
    let agrees := if floating && isNaN32 wanted then isNaN32 word else word == wanted
    unless agrees do
      throw <| IO.userError s!"pointwise {name}: got words {actual}, expected {expected}"
  IO.println s!"  pointwise {name}: ok"

/-- Require the checks to accept the artifact and execution to fail with `message`. -/
def expectExecutionError (name : String) (artifact : Json) (message : String) : IO Unit := do
  let report := report artifact
  let checks ← IO.ofExcept ((report.getObjValD "checks").getArr?)
  let status (index : Nat) : String :=
    ((checks[index]?.getD Json.null).getObjValD "status").getStr?.toOption.getD ""
  let detail := ((checks[2]?.getD Json.null).getObjValD "message").getStr?.toOption.getD ""
  unless status 0 == "checked" && status 1 == "checked" && status 2 == "failed" &&
      (detail.splitOn message).length > 1 do
    throw <| IO.userError s!"pointwise {name}: expected execution error `{message}`, got \
      {summary report}"
  IO.println s!"  pointwise {name}: rejected as expected"

/-! ## Binary32 cases recorded from eager PyTorch -/

/-- Dividends: `1, -1, 0, -0, 5, -5, 1, 7, -7, 0.5, -0.5, -∞, 1e30, 1`. -/
def dividends : List Nat :=
  [1065353216, 3212836864, 0, 2147483648, 1084227584, 3231711232, 1065353216, 1088421888,
    3235905536, 1056964608, 3204448256, 4286578688, 1900671690, 1065353216]

/-- Divisors: `0, 0, 0, 5, ∞, ∞, -∞, 2, 2, 1, 1, 2, 3e-10, 0.1`. -/
def divisors : List Nat :=
  [0, 0, 0, 1084227584, 2139095040, 2139095040, 4286578688, 1073741824, 1073741824,
    1065353216, 1065353216, 1073741824, 799337791, 1036831949]

/-- `torch.div` with each rounding mode on `dividends` and `divisors`. -/
def divisionCases : List (String × List (String × Json) × List Nat) :=
  [ ("div", [("kind", "div")],
      [2139095040, 4286578688, 4290772992, 2147483648, 0, 2147483648, 2147483648, 1080033280,
        3227516928, 1056964608, 3204448256, 4286578688, 2139095040, 1092616192])
  , ("div trunc", [("kind", "div"), ("rounding_mode", "trunc")],
      [2139095040, 4286578688, 4290772992, 2147483648, 0, 2147483648, 2147483648, 1077936128,
        3225419776, 0, 2147483648, 4286578688, 2139095040, 1092616192])
  , ("div floor", [("kind", "div"), ("rounding_mode", "floor")],
      [2139095040, 4286578688, 4290772992, 2147483648, 0, 3212836864, 3212836864, 1077936128,
        3229614080, 0, 3212836864, 4290772992, 2139095040, 1091567616]) ]

/-- Run the binary32 cases. -/
def runBinary : IO Unit := do
  for (name, kind, expected) in divisionCases do
    expectWords name (pointwiseProgram kind float32 [dividends, divisors]) expected
  expectWords "neg" (pointwiseProgram [("kind", "neg")] float32
      [[2143289344, 2147483648, 0, 1065353216, 4286578688, 1080033280]])
    [4290772992, 0, 2147483648, 3212836864, 2139095040, 3227516928]
  expectWords "rsqrt" (pointwiseProgram [("kind", "rsqrt")] float32
      [[0, 2147483648, 3212836864, 2139095040, 1082130432, 1048576000, 2143289344]])
    [2139095040, 4286578688, 4290772992, 0, 1056964608, 1073741824, 2143289344]
  -- Bases `nan, 1, -8, 0, -0, -1, 2, -2, 0, -0, ∞, -∞, 3` and exponents
  -- `0, nan, 1/3, -1, -1, ∞, 10, 3, 0, 3, -2, 3, 2`.
  expectWords "pow" (pointwiseProgram [("kind", "pow")] float32
      [[2143289344, 1065353216, 3238002688, 0, 2147483648, 3212836864, 1073741824, 3221225472,
          0, 2147483648, 2139095040, 4286578688, 1077936128],
        [0, 2143289344, 1051372203, 3212836864, 3212836864, 2139095040, 1092616192, 1077936128,
          0, 1077936128, 3221225472, 1077936128, 1073741824]])
    [1065353216, 1065353216, 4290772992, 2139095040, 4286578688, 1065353216, 1149239296,
      3238002688, 1065353216, 2147483648, 0, 4286578688, 1091567616]
  expectWords "gelu tanh special values"
    (pointwiseProgram [("kind", "gelu"), ("approximate", "tanh")] float32
      [[0, 2147483648, 2139095040, 4286578688, 2143289344]])
    [0, 2147483648, 2139095040, 4290772992, 2143289344]
  -- Inputs where `Activation.Math.geluSpec` rounds differently from PyTorch's kernel. Both of
  -- PyTorch's CPU loops return these words.
  expectWords "gelu tanh in PyTorch's order"
    (pointwiseProgram [("kind", "gelu"), ("approximate", "tanh")] float32
      [[3205634036, 1075102217, 3221921991, 1071658933, 1067053738, 3207754005, 3219949296,
          1061655685, 3224078953]])
    [3190164965, 1075005691, 3171251796, 1071071605, 1065896305, 3190647839, 3178540305,
      1058805493, 3155804041]
  -- Inputs where the result depends on rounding `x + κ x³` once: both of PyTorch's CPU loops fuse
  -- the sum at the AVX2 and AVX-512 capability levels and return these words; the unfused sum
  -- gives others.
  expectWords "gelu tanh with the fused inner sum"
    (pointwiseProgram [("kind", "gelu"), ("approximate", "tanh")] float32
      [[3213029748, 3216135843, 3224853051, 1075050716, 1075169974, 3223801888]])
    [3189807638, 3186216060, 3149198996, 1074951440, 1075076975, 3157776459]
  -- `clamp_min` with the bounds `0`, `-0` and NaN at `-0, 0, NaN, -∞, -1, 1, ∞`: eager PyTorch
  -- keeps a negative zero and returns a NaN for a NaN bound.
  let clampInputs := [2147483648, 0, 2143289344, 4286578688, 3212836864, 1065353216, 2139095040]
  expectWords "clamp_min at 0" (pointwiseProgram [("kind", "clamp_min")] float32
      [clampInputs, List.replicate 7 0])
    [2147483648, 0, 2143289344, 0, 0, 1065353216, 2139095040]
  expectWords "clamp_min at -0" (pointwiseProgram [("kind", "clamp_min")] float32
      [clampInputs, List.replicate 7 2147483648])
    [2147483648, 0, 2143289344, 2147483648, 2147483648, 1065353216, 2139095040]
  expectWords "clamp_min at NaN" (pointwiseProgram [("kind", "clamp_min")] float32
      [clampInputs, List.replicate 7 2143289344]) (List.replicate 7 2143289344)
  expectExecutionError "erf" (pointwiseProgram [("kind", "erf")] float32 [[0]])
    "erf has no implementation for float32"
  expectExecutionError "gelu none"
    (pointwiseProgram [("kind", "gelu"), ("approximate", "none")] float32 [[0]])
    "gelu(approximate=none) has no implementation for float32"

/-! ## Integer cases -/

/-- Two's-complement word of a signed 32-bit integer. -/
def word32 (value : Int) : Nat :=
  (value % 2 ^ 32).toNat

/-- Run the integer cases, including a zero divisor and the most negative value. -/
def runInteger : IO Unit := do
  let int32 := signed 32
  let a := [7, -7, 7, -7, -2147483648].map word32
  let b := [2, 2, -2, -2, -1].map word32
  expectWords "int32 div trunc" (pointwiseProgram [("kind", "div"), ("rounding_mode", "trunc")]
      int32 [a, b]) ([3, -3, -3, 3, -2147483648].map word32) (floating := false)
  expectWords "int32 div floor" (pointwiseProgram [("kind", "div"), ("rounding_mode", "floor")]
      int32 [a, b]) ([3, -4, -4, 3, -2147483648].map word32) (floating := false)
  expectExecutionError "int32 division by zero"
    (pointwiseProgram [("kind", "div"), ("rounding_mode", "floor")] int32 [[5], [0]])
    "integer division by zero"
  expectExecutionError "int32 true division"
    (pointwiseProgram [("kind", "div")] int32 [[5], [2]]) "div has no implementation for int32"
  expectWords "int32 neg" (pointwiseProgram [("kind", "neg")] int32
      [[-2147483648, 5, 0].map word32]) ([-2147483648, -5, 0].map word32) (floating := false)
  expectWords "int32 pow" (pointwiseProgram [("kind", "pow")] int32
      [[3, -2, 2, 0, 1, -1, -1, 5].map word32, [5, 3, 31, -1, -1, -1, -2, 0].map word32])
    ([243, -8, -2147483648, 0, 1, -1, 1, 1].map word32) (floating := false)
  expectWords "int8 pow wraps" (pointwiseProgram [("kind", "pow")] (signed 8) [[3], [5]])
    [243] (floating := false)
  expectWords "int32 clamp_min" (pointwiseProgram [("kind", "clamp_min")] int32
      [[-5, 0, 7, -2147483648].map word32, [0, 0, 0, -1].map word32])
    ([0, 0, 7, -1].map word32) (floating := false)
  let castProgram := program
    [ nodeJson 0 [("kind", "input")] [] 3 int32
    , nodeJson 1 [("kind", "custom"), ("primitive", "cast"), ("rounding", "nearestEven")]
        [0] 3 float32 ] 1 [tensorJson int32 ([16777217, -3, 0].map word32)]
  expectWords "int32 to float32 cast" castProgram [1266679808, 3225419776, 0]
  let flags := (report castProgram).getObjValD "conversion_flags"
  unless flags.getObjValD "inexact" == .bool true do
    throw <| IO.userError s!"pointwise cast: expected an inexact conversion, got {flags.compress}"

/-! ## Conversions to bfloat16 and binary16 -/

/-- An artifact converting `words` of `source` through each dtype of `targets` in turn. -/
def castChain (source : Json) (targets : List Json) (words : List Nat) : Json :=
  let length := words.length
  let casts := ((List.range targets.length).zip targets).map fun (index, dtype) =>
    nodeJson (index + 1) [("kind", "custom"), ("primitive", "cast"), ("rounding", "nearestEven")]
      [index] length dtype
  program (nodeJson 0 [("kind", "input")] [] length source :: casts) 1 [tensorJson source words]

/-- Source words with their bfloat16 and binary16 words through binary32 and in one rounding. -/
structure ConversionCase where
  /-- Source dtype name, for reports. -/
  name : String
  /-- Source element descriptor. -/
  source : Json
  /-- Source words. -/
  words : List Nat
  /-- Words of two casts through binary32, eager PyTorch's `Tensor.to` results. -/
  bfloat16 : List Nat
  /-- Words of one cast, the exact value rounded once. -/
  bfloat16Once : List Nat
  /-- Binary16 words of two casts through binary32. -/
  float16 : List Nat
  /-- Binary16 words of one cast. -/
  float16Once : List Nat

/--
Int32 `16842753, -16842753, 3, 65519, 65520`; int64 `16842753, 2^40 + 2^32 + 1,
-(2^40 + 2^32 + 1), 2^62 + 2^54 + 1`; binary64 `1 + 2^-8 + 2^-30, -(1 + 2^-8 + 2^-30),
1 + 2^-11 + 2^-40, 0.1`. Binary32 rounds `±16842753`, `±(2^40 + 2^32 + 1)`, `2^62 + 2^54 + 1` and
`±(1 + 2^-8 + 2^-30)` to bfloat16 midpoints and `1 + 2^-11 + 2^-40` to a binary16 midpoint; each
midpoint then rounds to even, while the exact value lies above it. The other values convert alike
both ways. `3` is exact and binary32 holds the integers `65519` and `65520` exactly. `65519`
rounds to binary16's largest finite value `65504`; `65520` is the tie between `65504` and `2^16`,
binary16's overflow threshold, and rounds to infinity.
-/
def conversionCases : List ConversionCase :=
  [ { name := "int32", source := signed 32,
      words := [16842753, 4278124543, 3, 65519, 65520],
      bfloat16 := [19328, 52096, 16448, 18304, 18304],
      bfloat16Once := [19329, 52097, 16448, 18304, 18304],
      float16 := [31744, 64512, 16896, 31743, 31744],
      float16Once := [31744, 64512, 16896, 31743, 31744] }
  , { name := "int64", source := signed 64,
      words := [16842753, 1103806595073, 18446742969902956543, 4629700416936869889],
      bfloat16 := [19328, 21376, 54144, 24192], bfloat16Once := [19329, 21377, 54145, 24193],
      float16 := [31744, 31744, 64512, 31744], float16Once := [31744, 31744, 64512, 31744] }
  , { name := "float64", source := "float64",
      words := [4607200010990256128, 13830572047845031936, 4607184617823277056,
        4591870180066957722],
      bfloat16 := [16256, 49024, 16256, 15821], bfloat16Once := [16257, 49025, 16256, 15821],
      float16 := [15364, 48132, 15360, 11878], float16Once := [15364, 48132, 15361, 11878] } ]

/-- Check one- and two-rounding conversions, and directed overflow with finite results. -/
def runConversions : IO Unit := do
  for case in conversionCases do
    for (name, twice, once) in [("bfloat16", case.bfloat16, case.bfloat16Once),
        ("float16", case.float16, case.float16Once)] do
      expectWords s!"{case.name} to {name} through float32"
        (castChain case.source [float32, Json.str name] case.words) twice (floating := false)
      expectWords s!"{case.name} to {name} in one cast"
        (castChain case.source [Json.str name] case.words) once (floating := false)
  -- Directed overflow can return the largest finite value rather than infinity.
  for (rounding, words) in [("nearestEven", [31744, 64512]), ("towardZero", [31743, 64511]),
      ("towardPositiveInfinity", [31744, 64511]),
      ("towardNegativeInfinity", [31743, 64512])] do
    let artifact := program
      [ nodeJson 0 [("kind", "input")] [] 2 (signed 32)
      , nodeJson 1 [("kind", "custom"), ("primitive", "cast"), ("rounding", Json.str rounding)]
          [0] 2 "float16" ] 1 [tensorJson (signed 32) [65536, 4294901760]]
    expectWords s!"int32 to float16 overflow ({rounding})" artifact words (floating := false)
    let flags := (report artifact).getObjValD "conversion_flags"
    unless flags.getObjValD "overflow" == .bool true &&
        flags.getObjValD "inexact" == .bool true &&
        flags.getObjValD "underflow" == .bool false &&
        flags.getObjValD "invalid" == .bool false do
      throw <| IO.userError s!"pointwise cast ({rounding}): wrong overflow flags: {flags.compress}"

/-! ## Exact real meanings -/

example : (Context.realPointwise (.div .none)).map (fun f => f ![1, 0]) =
    some (.error "div: a zero divisor has no real quotient") := by
  simp only [Context.realPointwise, Option.map]
  simp [Spec.PointwiseFn.binary]

example : (Context.realPointwise (.div .floor)).map (fun f => f ![-7, 2]) = some (.ok (-4)) := by
  simp only [Context.realPointwise, Option.map]
  have hFloor : ⌊(-7 : ℝ) / 2⌋ = -4 := by
    rw [Int.floor_eq_iff]
    norm_num
  simp [Spec.PointwiseFn.binary, hFloor]

example : (Context.realPointwise .pow).map (fun f => f ![-8, 1 / 3]) =
    some (.error "pow: a negative base with a nonintegral exponent has no real power") := by
  simp only [Context.realPointwise, Option.map]
  have hFract : Int.fract ((3 : ℝ)⁻¹) ≠ 0 := by
    rw [Int.fract_ne_zero_iff]
    rintro ⟨n, hn⟩
    have h1 : (n : ℝ) < 1 := by rw [hn]; norm_num
    have h0 : (0 : ℝ) < n := by rw [hn]; norm_num
    have : n < 1 := by exact_mod_cast h1
    have : 0 < n := by exact_mod_cast h0
    omega
  simp [Spec.PointwiseFn.binary, hFract]

example : (Context.realPointwise .rsqrt).map (fun f => f ![0]) =
    some (.error "rsqrt: a nonpositive input has no real reciprocal square root") := by
  simp only [Context.realPointwise, Option.map]
  simp [Spec.PointwiseFn.unary]

-- `1e30 / 3e-10` is an infinity in binary32, as `divisionCases` records, and a real number here.
example : (Context.realPointwise (.div .none)).map (fun f => f ![1e30, 3e-10]) =
    some (.ok (1e30 / 3e-10)) := by
  simp only [Context.realPointwise, Option.map]
  norm_num [Spec.PointwiseFn.binary]

/-! ## Sample comparisons -/

/-- An artifact with reference outputs, recorded by `frontend`; a Jaxpr reference also records the
platform that computed it, XLA's CPU backend unless `platform` says otherwise. -/
def withReference (artifact : Json) (frontend : String) (reference : List Json)
    (platform : Option String := some "cpu") : Json :=
  let artifact := (artifact.setObjVal! "frontend" frontend).setObjVal! "reference_outputs"
    (Json.arr reference.toArray)
  match frontend, platform with
  | "jaxpr", some name => artifact.setObjVal! "reference_platform" name
  | _, _ => artifact

/-- Record a reference computed one primitive at a time, inside nested `jax.jit` calls too. -/
def withEagerReference (artifact : Json) (reference : List Json) : Json :=
  artifact.setObjVal! "eager_reference_outputs" (Json.arr reference.toArray)

/-- Record the computation XLA compiles a node in, as the JAX importer does for eager JAX. -/
def computed (computation : Nat) (node : Json) : Json :=
  node.setObjVal! "computation" (toJson computation)

/-- Record that the reference came from `jax.jit`. -/
def jitted (artifact : Json) : Json :=
  artifact.setObjVal! "reference_execution" "jit"

/-- A binary32 `3` of shape `[]`, broadcast to `[3]`, as the JAX importer writes the divisor of
`x / 3.0`: nodes `id` and `id + 1`, in `computation`. -/
def broadcastThree (id computation : Nat) : List Json :=
  [ computed computation <| Json.mkObj [("id", toJson id), ("parents", toJson ([] : List Nat)),
      ("shape", toJson ([] : List Nat)), ("dtype", float32), ("kind", "const"),
      ("bits", toJson [1077936128]), ("values", toJson [(3.0 : Float)])]
  , computed computation <| nodeJson (id + 1)
      [("kind", "broadcast_to"), ("from_shape", toJson ([] : List Nat)), ("to_shape", toJson [3])]
      [id] 3 float32 ]

/-- Binary32 `4.11, 8.22, 15.07`. -/
def dividends3 : List Nat := [1082361119, 1090749727, 1097932472]

/-- Binary32 `1.3, 2.7, 0.9`. -/
def subtrahends3 : List Nat := [1067869798, 1076677837, 1063675494]

/-- Binary32 `0.7, 3.1, 1.9`. -/
def divisors3 : List Nat := [1060320051, 1078355558, 1072902963]

/-- `x / 3.0` as the JAX importer writes it, the divisor a broadcast scalar. -/
def constantDivision : Json :=
  program ([nodeJson 0 [("kind", "input")] [] 3 float32] ++ broadcastThree 1 0 ++
    [computed 0 (nodeJson 3 [("kind", "div")] [0, 2] 3 float32)]) 1 [tensorJson float32 dividends3]

/-- IEEE quotients of `constantDivision`, the IR's results. -/
def quotients : List Nat := [1068457001, 1076845609, 1084276517]

/-- XLA's products for `constantDivision`, one ulp above each quotient: eager JAX multiplies by
`0.333333343`, the binary32 reciprocal of `3`. -/
def products : List Nat := [1068457002, 1076845610, 1084276518]

/-- `(x - y) / 3.0` with the operands of the subtraction swapped, as an importer bug would write
it. Each element has the wrong sign, which no documented XLA behaviour explains. -/
def swappedDivision : Json :=
  program ([nodeJson 0 [("kind", "input")] [] 3 float32,
      nodeJson 1 [("kind", "input")] [] 3 float32,
      computed 0 (nodeJson 2 [("kind", "sub")] [1, 0] 3 float32)] ++ broadcastThree 3 1 ++
    [computed 1 (nodeJson 5 [("kind", "div")] [2, 4] 3 float32)]) 2
    [tensorJson float32 dividends3, tensorJson float32 subtrahends3]

/-- XLA's `(x - y) / 3.0` for `swappedDivision`'s arguments. -/
def swappedReference : List Nat := [1064290660, 1072399648, 1083647372]

/-- `x / c` for an array `c` of threes that the function closes over. Eager JAX passes `c` to the
division as a runtime array and returns the IEEE quotients; `jax.jit` folds it and multiplies by
the reciprocal. -/
def closedDivision : Json :=
  program
    [ nodeJson 0 [("kind", "input")] [] 3 float32
    , nodeJson 1 [("kind", "const"), ("bits", toJson [1077936128, 1077936128, 1077936128]),
        ("values", toJson [(3.0 : Float), 3.0, 3.0])] [] 3 float32
    , computed 0 (nodeJson 2 [("kind", "div")] [0, 1] 3 float32) ] 1
    [tensorJson float32 dividends3]

/-- `(x / y) / z`, two divisions; `jax.jit` computes it as `x / (y * z)`. -/
def quotientOfQuotient : Json :=
  program
    [ nodeJson 0 [("kind", "input")] [] 3 float32, nodeJson 1 [("kind", "input")] [] 3 float32
    , nodeJson 2 [("kind", "input")] [] 3 float32
    , computed 0 (nodeJson 3 [("kind", "div")] [0, 1] 3 float32)
    , computed 1 (nodeJson 4 [("kind", "div")] [3, 2] 3 float32) ] 3
    [tensorJson float32 dividends3, tensorJson float32 subtrahends3, tensorJson float32 divisors3]

/-- `jax.jit`'s words for `quotientOfQuotient`, `x / (y * z)`: the last differs from the IEEE
`(x / y) / z` by one ulp. -/
def jitQuotients : List Nat := [1083213577, 1065052550, 1091371392]

/-- `quotientOfQuotient` with the divisors swapped, as an importer bug would write `(x / y) / z`:
`jax.jit` computes both as `x / (y * z)`. -/
def swappedQuotients : Json :=
  program
    [ nodeJson 0 [("kind", "input")] [] 3 float32, nodeJson 1 [("kind", "input")] [] 3 float32
    , nodeJson 2 [("kind", "input")] [] 3 float32
    , computed 0 (nodeJson 3 [("kind", "div")] [0, 2] 3 float32)
    , computed 1 (nodeJson 4 [("kind", "div")] [3, 1] 3 float32) ] 3
    [tensorJson float32 dividends3, tensorJson float32 subtrahends3, tensorJson float32 divisors3]

/-- `quotientOfQuotient`'s IEEE `(x / y) / z`, which eager JAX computes primitive by primitive. -/
def ieeeQuotients : List Nat := [1083213577, 1065052550, 1091371391]

/-- Binary16 `x ** 3.0` under `jax.jit`, as the JAX importer writes it: the power in binary32 of
the widened operands, rounded once to binary16. -/
def halfCube : Json :=
  let float16 : Json := "float16"
  program
    [ nodeJson 0 [("kind", "input")] [] 4 float16
    , Json.mkObj [("id", toJson 1), ("parents", toJson ([] : List Nat)),
        ("shape", toJson ([] : List Nat)), ("dtype", float16), ("kind", "const"),
        ("bits", toJson [16896])]
    , nodeJson 2 [("kind", "broadcast_to"), ("from_shape", toJson ([] : List Nat)),
        ("to_shape", toJson [4])] [1] 4 float16
    , nodeJson 3 [("kind", "custom"), ("primitive", "cast"), ("rounding", "nearestEven")]
        [0] 4 float32
    , nodeJson 4 [("kind", "custom"), ("primitive", "cast"), ("rounding", "nearestEven")]
        [2] 4 float32
    , nodeJson 5 [("kind", "pow")] [3, 4] 4 float32
    , nodeJson 6 [("kind", "custom"), ("primitive", "cast"), ("rounding", "nearestEven")]
        [5] 4 float16 ] 1
    [tensorJson float16 [50601, 16932, 16553, 48916]]

/-- Bfloat16 `rsqrt` as the JAX importer writes it, computed in binary32 and rounded once, of
`2, 3, 1.5`. -/
def bfloat16Rsqrt : Json :=
  let bfloat16 : Json := "bfloat16"
  program
    [ nodeJson 0 [("kind", "input")] [] 3 bfloat16
    , nodeJson 1 [("kind", "custom"), ("primitive", "cast"), ("rounding", "nearestEven")]
        [0] 3 float32
    , nodeJson 2 [("kind", "rsqrt")] [1] 3 float32
    , nodeJson 3 [("kind", "custom"), ("primitive", "cast"), ("rounding", "nearestEven")]
        [2] 3 bfloat16 ] 1
    [tensorJson bfloat16 [16384, 16448, 16320]]

/-- Binary32 `1e-20` and `3e-20` squared: the IR's results are the subnormals `1e-40` and
`9e-40`; XLA's CPU code flushes them to zero. -/
def subnormalSquare : Json :=
  program
    [ nodeJson 0 [("kind", "input")] [] 2 float32
    , computed 0 (nodeJson 1 [("kind", "mul_elem")] [0, 0] 2 float32) ] 1
    [tensorJson float32 [507307272, 520989638]]

/-- `x - y` for `x = [1e-40, 1.5]` and `y = [2, 0.25]`, with the operands swapped. XLA flushes the
subnormal `1e-40` and returns `[-2, 1.25]`; flushing does not account for the swap. -/
def swappedSubnormal : Json :=
  program
    [ nodeJson 0 [("kind", "input")] [] 2 float32, nodeJson 1 [("kind", "input")] [] 2 float32
    , computed 0 (nodeJson 2 [("kind", "sub")] [1, 0] 2 float32) ] 2
    [tensorJson float32 [71362, 1069547520], tensorJson float32 [1073741824, 1048576000]]

/-- Run one comparison request and return its finding. -/
def finding (artifact : Json) (request : String) : Json × Json :=
  let report := NN.Verification.IR.check artifact #["structure", "types", request]
  (report, ((report.getObjValD "checks").getArrVal? 2).toOption.getD Json.null)

/-- The `kind` fields of a JSON array. -/
def kinds (items : Json) : List String :=
  ((items.getArr?).toOption.getD #[]).toList.map fun item =>
    match item with
    | .str code => code
    | _ => (item.getObjValD "kind").getStr?.toOption.getD ""

/-- Require a comparison request to end with `status`, to list `advice` in `consistent_with`, and
the emulation to apply `emulated` to the first result, the window of XLA's own function included
when it holds a differing element. -/
def expectComparison (name : String) (artifact : Json) (request status : String)
    (advice : List String := []) (emulated : List String := advice) : IO Unit := do
  let (report, entry) := finding artifact request
  let actual := (entry.getObjValD "status").getStr?.toOption.getD ""
  let passed := report.getObjValD "ok" == .bool true
  unless actual == status && passed == (status == "compared") do
    throw <| IO.userError s!"comparison {name}: expected {status}, got {summary report}"
  let listed := kinds (entry.getObjValD "consistent_with")
  unless listed == advice do
    throw <| IO.userError s!"comparison {name}: expected consistent_with {advice}, got {listed}"
  let first := ((entry.getObjValD "results").getArrVal? 0).toOption.getD Json.null
  let window := first.getObjValD "window"
  let used := kinds (first.getObjValD "emulated") ++
    (if (window.getObjValD "elements").getNat?.toOption.getD 0 > 0 then kinds (Json.arr #[window])
      else [])
  unless used == emulated do
    throw <| IO.userError s!"comparison {name}: expected emulated {emulated}, got {used}"
  IO.println s!"  comparison {name}: {status} {listed}"

/-- A comparison case: an artifact, the request, the status the finding must have, the behaviours
it must list in `consistent_with`, and those the emulation must apply to the first result. -/
structure ComparisonCase where
  /-- Name for reports. -/
  name : String
  /-- The artifact, with its reference outputs. -/
  artifact : Json
  /-- `compare` or `compare_bitwise`. -/
  request : String := "compare"
  /-- Expected status. -/
  status : String := "failed"
  /-- Expected `consistent_with` codes. -/
  advice : List String := []
  /-- Expected codes of the first result's `emulated` entries and of its window, if it holds a
  differing element. -/
  emulated : List String := advice

/-- Comparison outcomes: the NaN policy, and the emulation of XLA's documented CPU behaviour. Every
mismatch fails; `consistent_with` lists the behaviours whose emulation reproduces it. -/
def comparisonCases : List ComparisonCase :=
  -- FloatLib's quiet NaN for `0 / 0` against a NaN with the sign bit and a payload set.
  let nan := withReference (pointwiseProgram [("kind", "div")] float32 [[0], [0]]) "pytorch"
    [tensorJson float32 [4290772993]]
  let ieee := tensorJson float32 quotients
  let xla := tensorJson float32 products
  let negation := pointwiseProgram [("kind", "neg")] float32 [[71362]]
  let rsqrt := pointwiseProgram [("kind", "rsqrt")] float32 [[1082130432, 1048576000]]
  let largest := pointwiseProgram [("kind", "pow")] float32 [[2139095039], [1065353216]]
  let constantExponent := program
    [ nodeJson 0 [("kind", "input")] [] 1 float32
    , nodeJson 1 [("kind", "const"), ("bits", toJson [1065353216]),
        ("values", toJson [(1.0 : Float)])] [] 1 float32
    , nodeJson 2 [("kind", "pow")] [0, 1] 1 float32 ] 1 [tensorJson float32 [2139095039]]
  let bfloat16 : Json := "bfloat16"
  let float16 : Json := "float16"
  [ { name := "NaN sign and payload", artifact := nan, status := "compared" }
  , { name := "NaN words, bitwise", artifact := nan, request := "compare_bitwise" }
  , { name := "x / 3.0, IEEE reference", artifact := withReference constantDivision "jaxpr" [ieee]
      status := "compared" }
  , { name := "x / 3.0, XLA reference", artifact := withReference constantDivision "jaxpr" [xla]
      advice := ["divisor_reciprocal"] }
  , { name := "x / 3.0, PyTorch artifact"
      artifact := withReference constantDivision "pytorch" [xla] }
  , { name := "x / 3.0, reference from a GPU"
      artifact := withReference constantDivision "jaxpr" [xla] (platform := some "gpu") }
  , { name := "x / 3.0, no reference platform"
      artifact := withReference constantDivision "jaxpr" [xla] (platform := none) }
  , { name := "swapped operands next to a constant divisor"
      artifact := withReference swappedDivision "jaxpr" [tensorJson float32 swappedReference]
      emulated := ["divisor_reciprocal"] }
  , { name := "closed-over divisor array, eager reference"
      artifact := withReference closedDivision "jaxpr" [xla] }
  , { name := "closed-over divisor array, jax.jit reference"
      artifact := jitted (withReference closedDivision "jaxpr" [xla])
      advice := ["divisor_reciprocal"] }
  , { name := "quotient of a quotient, jax.jit reference"
      artifact := jitted (withReference quotientOfQuotient "jaxpr"
        [tensorJson float32 jitQuotients])
      advice := ["rewritten_division"] }
  -- The primitive-by-primitive reference: the IR's own quotients, or `x / (y * z)` written as
  -- `(x / z) / y`, which `jax.jit` computes as the reference does.
  , { name := "quotient of a quotient, jax.jit and primitive-by-primitive references"
      artifact := withEagerReference (jitted (withReference quotientOfQuotient "jaxpr"
        [tensorJson float32 jitQuotients])) [tensorJson float32 ieeeQuotients]
      advice := ["rewritten_division"] }
  , { name := "swapped divisors, jax.jit and primitive-by-primitive references"
      artifact := withEagerReference (jitted (withReference swappedQuotients "jaxpr"
        [tensorJson float32 jitQuotients])) [tensorJson float32 ieeeQuotients]
      emulated := ["rewritten_division"] }
  , { name := "quotient of a quotient, eager reference"
      artifact := withReference quotientOfQuotient "jaxpr" [tensorJson float32 jitQuotients] }
  , { name := "flushed subnormals"
      artifact := withReference subnormalSquare "jaxpr" [tensorJson float32 [0, 0]]
      advice := ["subnormal"] }
  , { name := "one subnormal operand and swapped operands"
      artifact := withReference swappedSubnormal "jaxpr"
        [tensorJson float32 [3221225472, 1067450368]]
      emulated := ["subnormal"] }
  -- Negation keeps a subnormal operand in XLA's CPU code, so a flushed reference is not XLA's.
  , { name := "negation of a subnormal against a zero reference"
      artifact := withReference negation "jaxpr" [tensorJson float32 [2147483648]] }
  , { name := "negation without a documented difference"
      artifact := withReference (pointwiseProgram [("kind", "neg")] float32 [[1065353216]])
        "jaxpr" [tensorJson float32 [3212836865]] }
  -- XLA's `rsqrt` and power: a window of 2 and 1 binary32 ulps, which never crosses zero or
  -- reaches an infinity, and in a reduced format applies before the binary32 value is rounded.
  , { name := "rsqrt within 2 ulps"
      artifact := withReference rsqrt "jaxpr" [tensorJson float32 [1056964610, 1073741822]]
      advice := ["rsqrt"] }
  , { name := "rsqrt 3 ulps away"
      artifact := withReference rsqrt "jaxpr" [tensorJson float32 [1056964611, 1073741824]] }
  , { name := "rsqrt of infinity against a negative zero"
      artifact := withReference (pointwiseProgram [("kind", "rsqrt")] float32 [[2139095040]])
        "jaxpr" [tensorJson float32 [2147483648]] }
  , { name := "pow of arguments within 1 ulp"
      artifact := withReference largest "jaxpr" [tensorJson float32 [2139095038]]
      advice := ["pow"] }
  , { name := "pow of the largest finite value against an infinity"
      artifact := withReference largest "jaxpr" [tensorJson float32 [2139095040]] }
  , { name := "pow with a constant exponent, 1 ulp away"
      artifact := withReference constantExponent "jaxpr" [tensorJson float32 [2139095038]] }
  , { name := "bfloat16 rsqrt one bfloat16 ulp away"
      artifact := withReference bfloat16Rsqrt "jaxpr" [tensorJson bfloat16 [16182, 16148, 16209]] }
  -- `jax.jit`'s binary16 cube rounds each product to binary16; primitive by primitive, XLA's
  -- power returns the IR's words.
  , { name := "binary16 cube, jax.jit reference"
      artifact := withEagerReference (jitted (withReference halfCube "jaxpr"
        [tensorJson float16 [55722, 20285, 19027, 50570]]))
        [tensorJson float16 [55723, 20284, 19027, 50570]]
      advice := ["pow_products"] } ]

/-- Run the comparison cases, after checking the IR's words that some of them rely on. -/
def runComparisons : IO Unit := do
  expectWords "pow of the largest finite value"
    (pointwiseProgram [("kind", "pow")] float32 [[2139095039], [1065353216]]) [2139095039]
  expectWords "bfloat16 rsqrt" bfloat16Rsqrt [16181, 16148, 16209] (floating := false)
  expectWords "binary16 cube" halfCube [55723, 20284, 19027, 50570] (floating := false)
  for case in comparisonCases do
    expectComparison case.name case.artifact case.request case.status case.advice case.emulated

/--
Capture the same operators live from PyTorch and JAX with `scripts/checks/test_ir_capture.py` and
check them with the built `verify` executable. The script skips any frontend that is not installed,
and every case when the checker has not been built.
-/
def runCaptures : IO Unit := do
  let out ← IO.Process.output
    { cmd := "python3", args := #["scripts/checks/test_ir_capture.py"], cwd := some "." }
  let summary := (out.stderr.splitOn "\n").filter (fun line => line.startsWith "Ran " ||
    line.startsWith "OK" || line.startsWith "FAILED")
  unless out.exitCode == 0 do
    throw <| IO.userError s!"pointwise captures failed:\n{out.stderr}"
  IO.println s!"  PyTorch and JAX captures: {" ".intercalate summary}"

/-- Concrete error certificates must execute their graphs and reject invalid budgets and values.
The supplied reference-output recording is deliberately forged: the checker must ignore it. -/
def runNumericalCertificates : IO Unit := do
  let artifact (words : List Nat) :=
    program [nodeJson 0 [("kind", "input")] [] words.length float32] 1
      [tensorJson float32 words]
  let reference := artifact [0x3f800000, 0x40000000] -- 1, 2
  let nearby := artifact [0x3fa00000, 0x40000000] -- 1.25, 2
  let changed := artifact [0x40400000, 0x40000000] -- 3, 2
  let forged := reference.setObjVal! "reference_outputs"
    (toJson [tensorJson float32 [0x7fc00000, 0x7f800000]])
  let check := NN.Verification.Numerical.checkError
  let stable := NN.Verification.Numerical.checkPrediction
  for (name, actual, expected) in #[
      ("finite budget", check reference nearby (1 / 4), true),
      ("insufficient budget", check reference nearby (1 / 8), false),
      ("negative budget", check reference reference (-1), false),
      ("shape mismatch", check reference (artifact [0x3f800000]) 1, false),
      ("NaN rejected", check reference (artifact [0x7fc00000, 0x40000000]) 100, false),
      ("infinity rejected", check reference (artifact [0x7f800000, 0x40000000]) 100, false),
      ("recordings ignored", check reference forged 0, true),
      ("winner preserved", stable reference nearby (1 / 4) 0 1, true),
      ("half-margin boundary", stable reference reference (1 / 2) 0 1, false),
      ("invalid winner", stable reference reference 0 0 2, false),
      ("changed winner", stable reference changed 2 0 1, false)] do
    unless actual == expected do
      throw <| IO.userError s!"numerical execution certificate: {name}: got {actual}"
  IO.println "  numerical execution certificates: bounds, margins and rejection checks passed"

/-- Run every pointwise case and the shared numerical certificate checks. -/
def run : IO Unit := do
  IO.println "== pointwise operators through the typed IR checker =="
  runBinary
  runInteger
  runConversions
  runComparisons
  runNumericalCertificates
  runCaptures

end NN.Tests.IR.Pointwise
