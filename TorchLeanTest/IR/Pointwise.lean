/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Pointwise
public import TorchLean.NN.IR.Artifact
public import TorchLean.Verify.Capture

/-!
# Pointwise operator execution

Each case is a capture document whose typed artifact (`TorchLean.IR.Artifact`) holds one pointwise
operator with exact argument words and the expected words. The operator is built from its form at
the operands' element type (`IR.Pointwise.form?`) or from a registered operator. The expected
words are those eager PyTorch 2.13 returns on CPU for the same arguments, except where a case gives
another source (the zero `remainder`, Python's `%`, and the tanh GELU, at TorchLean's constants).
`TorchLean.Capture.check` loads, type checks and executes it with the CPU reference interpreter.
A NaN result needs only to be a NaN: the sign and payload of a NaN produced by an invalid
operation are implementation choices, and `compare` lets two NaNs agree for the same reason.

A result word also passes where `deviations` lists TorchLean's word, which follows IEEE 754 where
PyTorch does not. Flooring division gives `-∞` at `(-∞, 2)`, the floor of the quotient `-∞`,
where PyTorch gives a NaN. Truncating division gives `9` at `(1, 0.1)`, the exact quotient
`9.99999985` truncated, where PyTorch truncates the rounded quotient `10`.

Integer cases follow PyTorch's integer kernels, including `add`, `sub`, `mul`, `neg` and `pow`
modulo `2 ^ width` and `maximum`, `minimum`, `clamp_min`, `div` and `remainder` in the signed or
unsigned order. Conversions of integers and binary64 to bfloat16 and binary16 run as two casts
through binary32, which return the words of PyTorch's `Tensor.to` and XLA's
`convert_element_type`. The comparison cases check that a mismatch with a reference from eager JAX
or `jax.jit` 0.11.2 on CPU fails. The failure lists XLA's documented behaviours in
`consistent_with` if and only if re-running the program as XLA's CPU code computes it reproduces
the reference words. It lists none for a program with swapped operands, as an importer bug would
write it, or for a `jax.jit` reference whose primitive-by-primitive reference the IR does not
reproduce. Their documents record the computation that XLA compiles each step in, as the JAX
importer does.
-/

@[expose] public section

namespace TorchLean.Test.Pointwise

open Lean

/-! ## Typed artifacts -/

/-- A step of a test artifact: the program step, the scalar of its value, frozen and detached
ports, and the words of a constant. -/
structure Step where
  /-- The program step. -/
  node : IR.Program.Node
  /-- The scalar of its value, frozen and detached ports, where it has one. -/
  dtype : IR.DType := .float32
  /-- The words of a constant. -/
  words : Option (List Nat) := none

/-- The artifact that runs `steps` in order and returns the first output of the last one, each
step's scalar recorded where the step has a value, frozen or detached port. -/
def artifact (steps : List Step) : IR.Artifact :=
  { program :=
      { nodes := (steps.map (·.node)).toArray
        outputs := #[⟨steps.length - 1, 0⟩]
        scalars := (steps.map fun step =>
          if step.node.ports.any (·.kind.isScalar) then some step.dtype else none).toArray }
    constants := (steps.map fun step => step.words.map List.toArray).toArray }

/-- A rank-one tensor of element type `d` in the capture document's encoding. -/
def tensorJson (d : IR.DType) (words : List Nat) : Json :=
  Json.mkObj [("dtype", toJson d), ("shape", toJson [words.length]), ("bits", toJson words)]

/-- The capture document of the artifact of `steps` with `arguments`, each of element type `d`. -/
def document (steps : List Step) (d : IR.DType) (arguments : List (List Nat)) : Json :=
  Json.mkObj [("artifact", (artifact steps).toJson),
    ("arguments", Json.arr (arguments.map (tensorJson d)).toArray)]

/-- The type of a value of element type `d` at the shape `s` (`Program.Typed.ofDType?`). -/
def typed (d : IR.DType) (s : Shape) : Except String IR.Program.Typed :=
  match IR.Program.Typed.ofDType? d s with
  | some t => pure t
  | none => throw s!"pointwise: {d} has no port"

/-- The steps reading arguments `0` to `count - 1`, each of element type `d` at the shape `s`. -/
def inputs (d : IR.DType) (s : Shape) (count : Nat) : Except String (List Step) := do
  let t ← typed d s
  pure ((List.range count).map fun k => ({ node := .node (.input k t.port), dtype := d } : Step))

/-- The document of the pointwise operator `op` on rank-one operands of element type `d` holding
`arguments`, its result at `d`: the steps of its form (`IR.Pointwise.form?`). -/
def pointwise (op : PointwiseOp) (d : IR.DType) (arguments : List (List Nat)) :
    Except String Json := do
  let s := [(arguments.headD []).length]
  let count := arguments.length
  let some (f, e) := IR.Pointwise.form? op d d s
    | throw s!"pointwise: {op.describe} has no form at {d}"
  if IR.Pointwise.Fragment.usesZero f then
    throw s!"pointwise: the form of {op.describe} reads the zero tensor"
  let operands := (List.range count).map fun k => (⟨k, 0⟩ : IR.Ref)
  let some nodes := IR.Pointwise.Fragment.nodes f operands ⟨count, 0⟩ count
    | throw s!"pointwise: {op.describe} reads more than {count} operands"
  let steps := nodes.map fun n => ({ node := .node n, dtype := e } : Step)
  pure (document ((← inputs d s count) ++ steps) d arguments)

/-- The document of the registered operator `o s` on rank-one operands of element type `d`
holding `arguments`, at the shape `s` of the operands. -/
def operator (o : Shape → IR.Operator) (d : IR.DType)
    (arguments : List (List Nat)) : Except String Json := do
  let s := [(arguments.headD []).length]
  let refs := (List.range arguments.length).map fun k => (⟨k, 0⟩ : IR.Ref)
  let step : Step := { node := .node (.op (o s) refs.toArray), dtype := d }
  pure (document ((← inputs d s arguments.length) ++ [step]) d arguments)

/-- Run the structure, type and execution checks on a document. -/
def report (document : Json) : Json :=
  Capture.check document #["structure", "types", "execute"]

/-- The status and message of each requested check, for failure reports. -/
def summary (report : Json) : String :=
  (report.getObjValD "checks").compress

/-- A result word where TorchLean departs from the oracle: TorchLean's word at a position of the
result of a case, with the reason. -/
structure Deviation where
  /-- The case. -/
  name : String
  /-- The position in the result. -/
  position : Nat
  /-- TorchLean's word there. -/
  word : Nat
  /-- Why TorchLean's word differs from the oracle's. -/
  reason : String

/-- The result words where TorchLean departs from eager PyTorch 2.13. -/
def deviations : List Deviation :=
  [ { name := "div floor", position := 11, word := 4286578688
      reason := "the quotient -∞ / 2 is -∞ and IEEE 754's floor of -∞ is -∞, where PyTorch \
        gives a NaN" }
  , { name := "div trunc", position := 13, word := 1091567616
      reason := "the exact quotient 9.99999985 of 1 / 0.1 truncates to 9, where PyTorch \
        truncates the rounded quotient 10" } ]

/-- Whether TorchLean's `word` at `position` of case `name` passes against the oracle's `wanted`:
the words are equal, both are binary32 NaNs when `floating`, or `deviations` lists `word` there. -/
def passes (name : String) (floating : Bool) (position word wanted : Nat) : Bool :=
  Capture.agree (!floating) .float32 word wanted ||
    deviations.any fun d => d.name == name && d.position == position && d.word == word

/-- Require the document to load, type check and execute, and each word of its first result to
pass against `expected` (`passes`), reporting each deviation the result shows. -/
def expectWords (name : String) (document : Except String Json) (expected : List Nat)
    (floating : Bool := true) : IO Unit := do
  let document ← IO.ofExcept (document.mapError fun e => s!"pointwise {name}: {e}")
  let report := report document
  unless report.getObjValD "ok" == .bool true do
    throw <| IO.userError s!"pointwise {name}: checks failed: {summary report}"
  let outputs ← IO.ofExcept ((report.getObjValD "outputs").getArr?)
  let some output := outputs[0]?
    | throw <| IO.userError s!"pointwise {name}: no output"
  let actual ← IO.ofExcept (fromJson? (α := List Nat) (output.getObjValD "bits"))
  unless actual.length == expected.length do
    throw <| IO.userError s!"pointwise {name}: {actual.length} results, expected {expected.length}"
  for (position, word, wanted) in (List.range actual.length).zip (actual.zip expected) do
    unless passes name floating position word wanted do
      throw <| IO.userError s!"pointwise {name}: got words {actual}, expected {expected}"
  let shown := deviations.filter fun d => d.name == name && actual[d.position]? == some d.word
  let notes := shown.map fun d => s!" (TorchLean's word at {d.position}: {d.reason})"
  IO.println s!"  pointwise {name}: ok{String.join notes}"

/-! ## Binary32 cases, from eager PyTorch except where a case says otherwise -/

/-- Dividends: `1, -1, 0, -0, 5, -5, 1, 7, -7, 0.5, -0.5, -∞, 1e30, 1`. -/
def dividends : List Nat :=
  [1065353216, 3212836864, 0, 2147483648, 1084227584, 3231711232, 1065353216, 1088421888,
    3235905536, 1056964608, 3204448256, 4286578688, 1900671690, 1065353216]

/-- Divisors: `0, 0, 0, 5, ∞, ∞, -∞, 2, 2, 1, 1, 2, 3e-10, 0.1`. -/
def divisors : List Nat :=
  [0, 0, 0, 1084227584, 2139095040, 2139095040, 4286578688, 1073741824, 1073741824,
    1065353216, 1065353216, 1073741824, 799337791, 1036831949]

/-- `torch.div` with each rounding mode on `dividends` and `divisors`. -/
def divisionCases : List (String × PointwiseOp × List Nat) :=
  [ ("div", .div .none,
      [2139095040, 4286578688, 4290772992, 2147483648, 0, 2147483648, 2147483648, 1080033280,
        3227516928, 1056964608, 3204448256, 4286578688, 2139095040, 1092616192])
  , ("div trunc", .div .trunc,
      [2139095040, 4286578688, 4290772992, 2147483648, 0, 2147483648, 2147483648, 1077936128,
        3225419776, 0, 2147483648, 4286578688, 2139095040, 1092616192])
  , ("div floor", .div .floor,
      [2139095040, 4286578688, 4290772992, 2147483648, 0, 3212836864, 3212836864, 1077936128,
        3229614080, 0, 3212836864, 4290772992, 2139095040, 1091567616]) ]

/-- Run the binary32 cases. -/
def runBinary : IO Unit := do
  for (name, op, expected) in divisionCases do
    expectWords name (pointwise op .float32 [dividends, divisors]) expected
  expectWords "neg" (pointwise .neg .float32
      [[2143289344, 2147483648, 0, 1065353216, 4286578688, 1080033280]])
    [4290772992, 0, 2147483648, 3212836864, 2139095040, 3227516928]
  expectWords "rsqrt" (pointwise .rsqrt .float32
      [[0, 2147483648, 3212836864, 2139095040, 1082130432, 1048576000, 2143289344]])
    [2139095040, 4286578688, 4290772992, 0, 1056964608, 1073741824, 2143289344]
  -- Bases `nan, 1, -8, 0, -0, -1, 2, -2, 0, -0, ∞, -∞, 3` and exponents
  -- `0, nan, 1/3, -1, -1, ∞, 10, 3, 0, 3, -2, 3, 2`.
  expectWords "pow" (pointwise .pow .float32
      [[2143289344, 1065353216, 3238002688, 0, 2147483648, 3212836864, 1073741824, 3221225472,
          0, 2147483648, 2139095040, 4286578688, 1077936128],
        [0, 2143289344, 1051372203, 3212836864, 3212836864, 2139095040, 1092616192, 1077936128,
          0, 1077936128, 3221225472, 1077936128, 1073741824]])
    [1065353216, 1065353216, 4290772992, 2139095040, 4286578688, 1065353216, 1149239296,
      3238002688, 1065353216, 2147483648, 0, 4286578688, 1091567616]
  -- The expected GELU words are PyTorch's DEFAULT order with the constants `√(2 / π)` computed in
  -- binary32 and `0.044715` rounded once, each operation rounded in binary32 and `tanh` rounded to
  -- nearest from a 60-digit value (a numerical check, not a proved enclosure). With PyTorch's `β`,
  -- `√(2 / π)` rounded once, and the C library's `tanhf` the same order returns PyTorch's words. At
  -- these inputs `x * sigmoid (2 u)` rounds differently from `x / 2 * (1 + tanh u)` in all but two
  -- places.
  expectWords "gelu tanh in PyTorch's order" (pointwise (.gelu .tanh) .float32
      [[3205634036, 1075102217, 3221921991, 1071658933, 1067053738, 3207754005, 3219949296,
          1061655685, 3224078953]])
    [3190164966, 1075005690, 3171251813, 1071071604, 1065896304, 3190647840, 3178540319,
      1058805492, 3155804127]
  -- Inputs where the result depends on rounding `κ x³` and `x + κ x³` separately, as PyTorch's
  -- loops do at the DEFAULT capability level. At the AVX2 and AVX-512 levels they fuse the sum,
  -- which gives other words at every one of these inputs.
  expectWords "gelu tanh with the unfused inner sum" (pointwise (.gelu .tanh) .float32
      [[3213029748, 1073400980, 3218028668, 3217515369, 1070121126, 3222688349]])
    [3189807640, 1072990025, 3182375635, 3183391933, 1069351183, 3165793976]
  -- `clamp_min` with the bounds `0`, `-0` and NaN at `-0, 0, NaN, -∞, -1, 1, ∞`: eager PyTorch
  -- keeps a negative zero and returns a NaN for a NaN bound.
  let clampInputs := [2147483648, 0, 2143289344, 4286578688, 3212836864, 1065353216, 2139095040]
  expectWords "clamp_min at 0" (pointwise .clampMin .float32 [clampInputs, List.replicate 7 0])
    [2147483648, 0, 2143289344, 0, 0, 1065353216, 2139095040]
  expectWords "clamp_min at -0"
    (pointwise .clampMin .float32 [clampInputs, List.replicate 7 2147483648])
    [2147483648, 0, 2143289344, 2147483648, 2147483648, 1065353216, 2139095040]
  expectWords "clamp_min at NaN"
    (pointwise .clampMin .float32 [clampInputs, List.replicate 7 2143289344])
    (List.replicate 7 2143289344)
  -- Python's `%` on `dividends` and `divisors`, as `numpy.remainder` computes it in binary32:
  -- `fmod`, then the divisor added where the remainder is nonzero with the other sign, so
  -- `-5 % ∞` is `∞`, and a zero result with the divisor's sign, so `-0 % 5` is `+0`.
  expectWords "remainder" (pointwise .remainder .float32 [dividends, divisors])
    [4290772992, 4290772992, 4290772992, 0, 1084227584, 2139095040, 4286578688,
      1065353216, 1065353216, 1056964608, 1056964608, 4290772992, 797290482, 1036831947]
  -- The inverse trigonometric functions are rounded correctly, and PyTorch returns the same words
  -- at these inputs. `atan` at `0, -0, 1, -1, ∞, -∞, NaN, 0.5, -0.5, 1e-30, 3e38, 7`:
  expectWords "atan" (pointwise .atan .float32
      [[0, 2147483648, 1065353216, 3212836864, 2139095040, 4286578688, 2143289344, 1056964608,
          3204448256, 228737632, 2137108966, 1088421888]])
    [0, 2147483648, 1061752795, 3209236443, 1070141403, 3217625051, 2143289344, 1055744824,
      3203228472, 228737632, 1070141403, 1068951084]
  -- `asin` and `acos` at `0, -0, 1, -1, 0.5, -0.5, 2, -∞, NaN, 1e-30, 0.9999999, 0.3`:
  let unitInputs := [0, 2147483648, 1065353216, 3212836864, 1056964608, 3204448256, 1073741824,
    4286578688, 2143289344, 228737632, 1065353214, 1050253722]
  expectWords "asin" (pointwise .asin .float32 [unitInputs])
    [0, 2147483648, 1070141403, 3217625051, 1057360530, 3204844178, 2143289344, 2143289344,
      2143289344, 228737632, 1070137307, 1050411181]
  expectWords "acos" (pointwise .acos .float32 [unitInputs])
    [1070141403, 1070141403, 0, 1078530011, 1065749138, 1074137746, 2143289344, 2143289344,
      2143289344, 1070141403, 973078528, 1067585455]
  -- `atan2 y x` at `(0, 0), (-0, -0), (0, -0), (-0, -1), (1, -1), (-1, 1), (∞, ∞), (-∞, -∞),
  -- (∞, 2), (1, ∞), (1, -∞), (NaN, 1), (0, -5), (3, 4)`: the Annex F values at zeros and
  -- infinities.
  expectWords "atan2" (pointwise .atan2 .float32
      [[0, 2147483648, 0, 2147483648, 1065353216, 3212836864, 2139095040, 4286578688, 2139095040,
          1065353216, 1065353216, 2143289344, 0, 1077936128],
        [0, 2147483648, 2147483648, 3212836864, 3212836864, 1065353216, 2139095040, 4286578688,
          1073741824, 2139095040, 4286578688, 1065353216, 3231711232, 1082130432]])
    [0, 3226013659, 1078530011, 3226013659, 1075235812, 3209236443, 1061752795, 3222719460,
      1070141403, 0, 1078530011, 2143289344, 1078530011, 1059372157]
  -- `atan2 y x` at `(-0, 0), (∞, -∞), (-∞, ∞), (1, NaN), (-1, -∞), (-0, 1), (-∞, -2), (5, -0)`.
  expectWords "atan2 at signed zeros, infinities and NaN" (pointwise .atan2 .float32
      [[2147483648, 2139095040, 4286578688, 1065353216, 3212836864, 2147483648, 4286578688,
          1084227584],
        [0, 4286578688, 2139095040, 2143289344, 4286578688, 1065353216, 3221225472, 2147483648]])
    [2147483648, 1075235812, 3209236443, 2143289344, 3226013659, 2147483648, 3217625051,
      1070141403]

/-! ## Integer cases -/

/-- Two's-complement word of a signed 32-bit integer. -/
def word32 (value : Int) : Nat :=
  (value % 2 ^ 32).toNat

/-- Run the integer cases, including the most negative value. -/
def runInteger : IO Unit := do
  let int8 : IR.DType := .integer true (some 8)
  let int32 : IR.DType := .integer true (some 32)
  let uint8 : IR.DType := .integer false (some 8)
  let a := [7, -7, 7, -7].map word32
  let b := [2, 2, -2, -2].map word32
  expectWords "int32 div trunc" (pointwise (.div .trunc) int32 [a, b])
    ([3, -3, -3, 3].map word32) (floating := false)
  expectWords "int32 div floor" (pointwise (.div .floor) int32 [a, b])
    ([3, -4, -4, 3].map word32) (floating := false)
  expectWords "int32 neg" (pointwise .neg int32 [[-2147483648, 5, 0].map word32])
    ([-2147483648, -5, 0].map word32) (floating := false)
  expectWords "int32 pow" (pointwise .pow int32
      [[3, -2, 2, 0, 1, -1, -1, 5].map word32, [5, 3, 31, -1, -1, -1, -2, 0].map word32])
    ([243, -8, -2147483648, 0, 1, -1, 1, 1].map word32) (floating := false)
  expectWords "int8 pow wraps" (pointwise .pow int8 [[3], [5]]) [243] (floating := false)
  expectWords "int32 clamp_min" (pointwise .clampMin int32
      [[-5, 0, 7, -2147483648].map word32, [0, 0, 0, -1].map word32])
    ([0, 0, 7, -1].map word32) (floating := false)
  let c := [2147483647, -2147483648, 5, -3].map word32
  let d := [1, 1, -7, 4].map word32
  expectWords "int32 add wraps"
    (operator (fun s => ⟨_, _, Op.add (s := s) (.integer true 32)⟩) int32 [c, d])
    ([-2147483648, -2147483647, -2, 1].map word32) (floating := false)
  expectWords "int32 sub wraps"
    (operator (fun s => ⟨_, _, Op.sub (s := s) (.integer true 32)⟩) int32 [c, d])
    ([2147483646, 2147483647, 12, -7].map word32) (floating := false)
  expectWords "int8 mul wraps"
    (operator (fun s => ⟨_, _, Op.mul (s := s) (.integer true 8)⟩) int8
      [[100, 255, 16], [3, 255, 16]])
    [44, 1, 0] (floating := false)
  expectWords "int32 maximum"
    (operator (fun s => ⟨_, _, Op.max (s := s) (.integer true 32)⟩) int32 [c, d])
    ([2147483647, 1, 5, 4].map word32) (floating := false)
  expectWords "int32 minimum"
    (operator (fun s => ⟨_, _, Op.min (s := s) (.integer true 32)⟩) int32 [c, d])
    ([1, -2147483648, -7, -3].map word32) (floating := false)
  expectWords "uint8 maximum in the unsigned order"
    (operator (fun s => ⟨_, _, Op.max (s := s) (.integer false 8)⟩) uint8
      [[255, 0, 128], [1, 1, 127]])
    [255, 1, 128] (floating := false)
  -- Operands at or above `2 ^ (width - 1)` read as unsigned values.
  let p := [200, 255, 128]
  let q := [3, 7, 3]
  let unsignedCases : List (String × PointwiseOp × List (List Nat) × List Nat) := [
    ("uint8 div trunc", .div .trunc, [p, q], [66, 36, 42]),
    ("uint8 div floor", .div .floor, [p, q], [66, 36, 42]),
    ("uint8 remainder", .remainder, [p, q], [2, 3, 2]),
    ("uint8 pow wraps", .pow, [[3, 2, 255], [200, 7, 2]], [161, 128, 1]),
    ("uint8 neg", .neg, [[200, 0, 128]], [56, 0, 128]),
    ("uint8 clamp_min", .clampMin, [[200, 3], [100, 255]], [200, 255])]
  for (name, op, arguments, expected) in unsignedCases do
    expectWords name (pointwise op uint8 arguments) expected (floating := false)

/-! ## Conversions to bfloat16 and binary16 -/

/-- The document converting rank-one `words` of element type `source` to each element type of
`targets` in turn, each conversion rounding to nearest with ties to even. -/
def castChain (source : IR.DType) (targets : List IR.DType) (words : List Nat) :
    Except String Json := do
  let s := [words.length]
  let casts ← ((List.range targets.length).zip targets).mapM fun (k, target) => do
    let t ← typed target s
    pure ({ node := .cast .nearest ⟨k, 0⟩ t.port, dtype := target } : Step)
  pure (document ((← inputs source s 1) ++ casts) source [words])

/-- Source words with their bfloat16 and binary16 words through binary32. -/
structure ConversionCase where
  /-- Source dtype name, for reports. -/
  name : String
  /-- Source element type. -/
  source : IR.DType
  /-- Source words. -/
  words : List Nat
  /-- Words of two casts through binary32, eager PyTorch's `Tensor.to` results. -/
  bfloat16 : List Nat
  /-- Binary16 words of two casts through binary32. -/
  float16 : List Nat

/--
The sources are int32 `16842753, -16842753, 3, 65519, 65520`, int64
`16842753, 2^40 + 2^32 + 1, -(2^40 + 2^32 + 1), 2^62 + 2^54 + 1` and binary64
`1 + 2^-8 + 2^-30, -(1 + 2^-8 + 2^-30), 1 + 2^-11 + 2^-40, 0.1`. Binary32 rounds `±16842753`,
`±(2^40 + 2^32 + 1)`, `2^62 + 2^54 + 1` and `±(1 + 2^-8 + 2^-30)` to bfloat16 midpoints and
`1 + 2^-11 + 2^-40` to a binary16 midpoint, and each midpoint then rounds to even. `3` is exact and
binary32 holds the integers `65519` and `65520` exactly. `65519` rounds to binary16's largest
finite value `65504`. `65520` is the tie between `65504` and `2^16`, binary16's overflow threshold,
and rounds to infinity.
-/
def conversionCases : List ConversionCase :=
  [ { name := "int32", source := .integer true (some 32),
      words := [16842753, 4278124543, 3, 65519, 65520],
      bfloat16 := [19328, 52096, 16448, 18304, 18304],
      float16 := [31744, 64512, 16896, 31743, 31744] }
  , { name := "int64", source := .integer true (some 64),
      words := [16842753, 1103806595073, 18446742969902956543, 4629700416936869889],
      bfloat16 := [19328, 21376, 54144, 24192],
      float16 := [31744, 31744, 64512, 31744] }
  , { name := "float64", source := .float64,
      words := [4607200010990256128, 13830572047845031936, 4607184617823277056,
        4591870180066957722],
      bfloat16 := [16256, 49024, 16256, 15821],
      float16 := [15364, 48132, 15360, 11878] } ]

/-- Check conversions to bfloat16 and binary16 through binary32. -/
def runConversions : IO Unit := do
  for case in conversionCases do
    for (target, twice) in [(IR.DType.bfloat16, case.bfloat16), (.float16, case.float16)] do
      expectWords s!"{case.name} to {target} through float32"
        (castChain case.source [.float32, target] case.words) twice (floating := false)

/-! ## Sample comparisons -/

/-- Argument `k` of a JAX program, a rank-one tensor of `n` entries of element type `d`. -/
def input (k n : Nat) (d : IR.DType := .float32) : Step :=
  { node := .node (.input k (.value [n])), dtype := d }

/-- A constant of element type `d` at the shape `s` with the words `words`. -/
def const (s : Shape) (words : List Nat) (d : IR.DType := .float32) : Step :=
  { node := .node (.const (.value s)), dtype := d, words := some words }

/-- The registered operator `o` on the first outputs of the steps `refs`, at the scalar `d`. -/
def operation (o : IR.Operator) (refs : List Nat) (d : IR.DType := .float32) : Step :=
  { node := .node (.op o (refs.map fun j => (⟨j, 0⟩ : IR.Ref)).toArray), dtype := d }

/-- The scalar of step `j` broadcast to the shape `[n]`, as the JAX importer writes a scalar
operand of an elementwise primitive. -/
def broadcast (j n : Nat) (d : IR.DType := .float32) : Step :=
  operation ⟨_, _, Op.broadcast (s := []) (t := [n]) _
    (Shape.BroadcastDims.scalar [n])⟩ [j] d

/-- The conversion of step `j`, of shape `[n]`, to the scalar `d`, rounding to nearest with ties to
even. -/
def convert (j n : Nat) (d : IR.DType) : Step :=
  { node := .cast .nearest ⟨j, 0⟩ (.value [n]), dtype := d }

/-- The capture document of a JAX program of `steps` on `arguments`. Its `computation` gives, for
each step, the computation that XLA compiles it in, `none` where no equation produced the step,
and its `primitive_index` is null at every step. -/
def jaxProgram (steps : List (Step × Option Nat)) (arguments : List Json) : Json :=
  Json.mkObj [("artifact", (artifact (steps.map (·.1))).toJson),
    ("arguments", Json.arr arguments.toArray), ("frontend", "jaxpr"),
    ("reference_execution", "eager"), ("jit_computations", Json.arr #[]),
    ("computation", toJson (steps.map (·.2))),
    ("primitive_index", toJson (steps.map fun _ => (none : Option Nat)))]

/-- A JAX document with the reference outputs `reference`, which XLA's CPU backend computed. -/
def withReference (document : Json) (reference : List Json) : Json :=
  (document.setObjVal! "reference_outputs" (Json.arr reference.toArray)).setObjVal!
    "reference_platform" "cpu"

/-- Record a reference computed one primitive at a time, inside nested `jax.jit` calls too. -/
def withEagerReference (document : Json) (reference : List Json) : Json :=
  document.setObjVal! "eager_reference_outputs" (Json.arr reference.toArray)

/-- Record that the reference came from `jax.jit`. -/
def jitted (document : Json) : Json :=
  document.setObjVal! "reference_execution" "jit"

/-- Binary32 `4.11, 8.22, 15.07`. -/
def dividends3 : List Nat := [1082361119, 1090749727, 1097932472]

/-- Binary32 `1.3, 2.7, 0.9`. -/
def subtrahends3 : List Nat := [1067869798, 1076677837, 1063675494]

/-- Binary32 `0.7, 3.1, 1.9`. -/
def divisors3 : List Nat := [1060320051, 1078355558, 1072902963]

/-- Binary32 division at the shape `[3]`. -/
def division3 : IR.Operator := ⟨_, _, Op.div (s := [3])⟩

/-- `x / 3.0` as the JAX importer writes it, the divisor a broadcast scalar. -/
def constantDivision : Json :=
  jaxProgram [(input 0 3, none), (const [] [1077936128], some 0), (broadcast 1 3, some 0),
      (operation division3 [0, 2], some 0)]
    [tensorJson .float32 dividends3]

/-- XLA's products for `constantDivision`, one ulp above each quotient: eager JAX multiplies by
`0.333333343`, the binary32 reciprocal of `3`. -/
def products : List Nat := [1068457002, 1076845610, 1084276518]

/-- `(x - y) / 3.0` with the operands of the subtraction swapped, as an importer bug would write
it. Each element has the wrong sign, which no documented XLA behaviour explains. -/
def swappedDivision : Json :=
  jaxProgram [(input 0 3, none), (input 1 3, none),
      (operation ⟨_, _, Op.sub (s := [3])⟩ [1, 0], some 0),
      (const [] [1077936128], some 1), (broadcast 3 3, some 1),
      (operation division3 [2, 4], some 1)]
    [tensorJson .float32 dividends3, tensorJson .float32 subtrahends3]

/-- XLA's `(x - y) / 3.0` for `swappedDivision`'s arguments. -/
def swappedReference : List Nat := [1064290660, 1072399648, 1083647372]

/-- `x / c` for an array `c` of threes that the function closes over. Eager JAX passes `c` to the
division as a runtime array and returns the IEEE quotients, while `jax.jit` folds it and multiplies
by the reciprocal. -/
def closedDivision : Json :=
  jaxProgram [(input 0 3, none), (const [3] [1077936128, 1077936128, 1077936128], none),
      (operation division3 [0, 1], some 0)]
    [tensorJson .float32 dividends3]

/-- `(x / y) / z` as two divisions, which `jax.jit` computes as `x / (y * z)`. -/
def quotientOfQuotient : Json :=
  jaxProgram [(input 0 3, none), (input 1 3, none), (input 2 3, none),
      (operation division3 [0, 1], some 0), (operation division3 [3, 2], some 1)]
    [tensorJson .float32 dividends3, tensorJson .float32 subtrahends3,
      tensorJson .float32 divisors3]

/-- `jax.jit`'s words for `quotientOfQuotient`, `x / (y * z)`: the last differs from the IEEE
`(x / y) / z` by one ulp. -/
def jitQuotients : List Nat := [1083213577, 1065052550, 1091371392]

/-- `quotientOfQuotient` with the divisors swapped, as an importer bug would write `(x / y) / z`:
`jax.jit` computes both as `x / (y * z)`. -/
def swappedQuotients : Json :=
  jaxProgram [(input 0 3, none), (input 1 3, none), (input 2 3, none),
      (operation division3 [0, 2], some 0), (operation division3 [3, 1], some 1)]
    [tensorJson .float32 dividends3, tensorJson .float32 subtrahends3,
      tensorJson .float32 divisors3]

/-- `quotientOfQuotient`'s IEEE `(x / y) / z`, which eager JAX computes primitive by primitive. -/
def ieeeQuotients : List Nat := [1083213577, 1065052550, 1091371391]

/-- Binary16 `x ** 3.0` under `jax.jit`, as the JAX importer writes it: the power in binary32 of
the widened operands, rounded once to binary16. -/
def halfCube : Json :=
  jaxProgram [(input 0 4 .float16, none), (const [] [16896] .float16, none),
      (broadcast 1 4 .float16, none), (convert 0 4 .float32, none), (convert 2 4 .float32, none),
      (operation ⟨_, _, Op.pow (s := [4])⟩ [3, 4], none),
      (convert 5 4 .float16, none)]
    [tensorJson .float16 [50601, 16932, 16553, 48916]]

/-- Binary32 `1e-20` and `3e-20` squared: the IR's results are the subnormals `1e-40` and
`9e-40`, which XLA's CPU code flushes to zero. -/
def subnormalSquare : Json :=
  jaxProgram [(input 0 2, none), (operation ⟨_, _, Op.mul (s := [2])⟩ [0, 0], some 0)]
    [tensorJson .float32 [507307272, 520989638]]

/-- `x - y` for `x = [1e-40, 1.5]` and `y = [2, 0.25]`, with the operands swapped. XLA flushes the
subnormal `1e-40` and returns `[-2, 1.25]`. Flushing does not account for the swap. -/
def swappedSubnormal : Json :=
  jaxProgram [(input 0 2, none), (input 1 2, none),
      (operation ⟨_, _, Op.sub (s := [2])⟩ [1, 0], some 0)]
    [tensorJson .float32 [71362, 1069547520], tensorJson .float32 [1073741824, 1048576000]]

/-- Run one comparison request and return the report and its finding. -/
def finding (document : Json) (request : String) : Json × Json :=
  let report := Capture.check document #["structure", "types", request]
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
def expectComparison (name : String) (document : Json) (request status : String)
    (advice : List String := []) (emulated : List String := advice) : IO Unit := do
  let (report, entry) := finding document request
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

/-- A comparison case: a document, the request, the status the finding must have, the behaviours
it must list in `consistent_with`, and those the emulation must apply to the first result. -/
structure ComparisonCase where
  /-- Name for reports. -/
  name : String
  /-- The capture document, with its reference outputs. -/
  document : Json
  /-- `compare` or `compare_bitwise`. -/
  request : String := "compare"
  /-- Expected status. -/
  status : String := "failed"
  /-- Expected `consistent_with` codes. -/
  advice : List String := []
  /-- Expected codes of the first result's `emulated` entries and of its window, if it holds a
  differing element. -/
  emulated : List String := advice

/-- Comparisons with recorded XLA output and the emulation of XLA's documented CPU behaviour. Every
mismatch fails, and `consistent_with` lists the behaviours whose emulation reproduces it. -/
def comparisonCases : List ComparisonCase :=
  let xla := tensorJson .float32 products
  [ { name := "x / 3.0, XLA reference", document := withReference constantDivision [xla]
      advice := ["divisor_reciprocal"] }
  , { name := "swapped operands next to a constant divisor"
      document := withReference swappedDivision [tensorJson .float32 swappedReference]
      emulated := ["divisor_reciprocal"] }
  , { name := "closed-over divisor array, jax.jit reference"
      document := jitted (withReference closedDivision [xla])
      advice := ["divisor_reciprocal"] }
  , { name := "quotient of a quotient, jax.jit reference"
      document := jitted (withReference quotientOfQuotient [tensorJson .float32 jitQuotients])
      advice := ["rewritten_division"] }
  -- The primitive-by-primitive reference: the IR's own quotients, or `x / (y * z)` written as
  -- `(x / z) / y`, which `jax.jit` computes as the reference does.
  , { name := "quotient of a quotient, jax.jit and primitive-by-primitive references"
      document := withEagerReference (jitted (withReference quotientOfQuotient
        [tensorJson .float32 jitQuotients])) [tensorJson .float32 ieeeQuotients]
      advice := ["rewritten_division"] }
  , { name := "swapped divisors, jax.jit and primitive-by-primitive references"
      document := withEagerReference (jitted (withReference swappedQuotients
        [tensorJson .float32 jitQuotients])) [tensorJson .float32 ieeeQuotients]
      emulated := ["rewritten_division"] }
  , { name := "flushed subnormals"
      document := withReference subnormalSquare [tensorJson .float32 [0, 0]]
      advice := ["subnormal"] }
  , { name := "one subnormal operand and swapped operands"
      document := withReference swappedSubnormal [tensorJson .float32 [3221225472, 1067450368]]
      emulated := ["subnormal"] }
  -- `jax.jit`'s binary16 cube rounds each product to binary16. Primitive by primitive, XLA's
  -- power returns the IR's words.
  , { name := "binary16 cube, jax.jit reference"
      document := withEagerReference (jitted (withReference halfCube
        [tensorJson .float16 [55722, 20285, 19027, 50570]]))
        [tensorJson .float16 [55723, 20284, 19027, 50570]]
      advice := ["pow_products"] } ]

/-- Run the comparison cases, after checking the IR's words that some of them rely on. -/
def runComparisons : IO Unit := do
  expectWords "binary16 cube" (pure halfCube) [55723, 20284, 19027, 50570] (floating := false)
  for case in comparisonCases do
    expectComparison case.name case.document case.request case.status case.advice case.emulated

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

/-- Run the pointwise cases, the comparisons and the live captures. -/
def run : IO Unit := do
  IO.println "== pointwise operators through the typed IR checker =="
  runBinary
  runInteger
  runConversions
  runComparisons
  runCaptures

end TorchLean.Test.Pointwise
