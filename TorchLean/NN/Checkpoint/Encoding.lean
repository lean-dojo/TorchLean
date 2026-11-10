/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Complex
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import TorchLean.CUDA.Kernel.Scalar

/-!
# Scalar encodings for model checkpoints

`Checkpoint.Encoding` maps each scalar to one word of a fixed bit width and checks every word it
decodes, while the state container checks shapes, lengths and word widths. Native `Float` and
`Float32` values store the bits they export. Lean canonicalizes native NaNs, so their decoders
reject any NaN word that would not read back bit for bit. Configured binary formats preserve every
encoding, NaN payloads included. A complex value packs both coordinate words into one word of twice
the component width, and no value passes through a narrower native format.
-/

@[expose] public section

namespace TorchLean.Checkpoint

/--
Checkpoint encoding of a scalar type: a versioned format tag, a word width, an encoder, and a
decoder that rejects malformed words.
-/
class Encoding (α : Type) where
  /-- Identifies the scalar format and schema. Different interpretations need different tags. -/
  format : String
  /-- Number of bits in one stored word. -/
  width : Nat
  /-- Encode one scalar as a word below `2 ^ width`, without changing its precision. -/
  encode : α → Nat
  /-- Decode one word, rejecting out-of-range or noncanonical payloads. -/
  decode : Nat → Except String α

namespace Encoding

/--
Accept `bits` as a `width`-bit storage word when `bits < 2 ^ width` and fail otherwise, so that no
out-of-range word wraps modulo the width.
-/
def checkedWord (width bits : Nat) : Except String Nat :=
  if bits < 2 ^ width then pure bits
  else throw s!"Checkpoint: scalar bits out of range for {width}-bit storage"

/-- The tag of a scalar representation. A configured binary format's tag names its exponent width,
fraction width, bias and exceptional-value encoding but not its storage plan, so two equal-width
formats with different interpretations cannot be confused on loading. -/
def tag : Kernel.Precision → String
  | .native .binary64 => "torchlean_state_bits_v1"
  | .native .binary32 => "torchlean_state_float32_bits_v1"
  | .binary format =>
    let encoding := match format.encoding with
      | .ieee => "ieee"
      | .finiteMaxNaN => "finite-max-nan"
      | .finiteUnsignedZero => "finite-unsigned-zero"
      | .finite => "finite"
    s!"torchlean_state_binary_{format.expWidth}_{format.fracWidth}_" ++
      s!"{format.exponentBias}_{encoding}_v1"

/-- The name of a scalar representation in decoding errors. -/
def valueName : Kernel.Precision → String
  | .native .binary64 => "native Float"
  | .native .binary32 => "native Float32"
  | .binary _ => "configured binary"

/-- A scalar of custom computations is stored as its word. A stored word must be the word of its
value, so a native NaN word that Lean would change is rejected. -/
instance {α : Type} [scalar : Kernel.Scalar α] : Encoding α where
  format := tag scalar.precision
  width := scalar.width
  encode value := (scalar.toBits value).toNat
  decode word := do
    let value := scalar.ofNatBits (← checkedWord scalar.width word)
    if (scalar.toBits value).toNat == word then pure value
    else throw s!"Checkpoint: noncanonical {valueName scalar.precision} NaN encoding"

/-- The checkpoint word of a scalar of custom computations decodes to that scalar, NaN payloads
included. -/
theorem decode_encode_scalar {β : Type} [Kernel.Scalar β] (x : β) :
    (Checkpoint.Encoding.decode (Checkpoint.Encoding.encode x) :
      Except String β) = .ok x := by
  have hlt : (Numeric.Word.toBits x).toNat < 2 ^ Numeric.Word.width β :=
    BitVec.isLt _
  simp [Checkpoint.Encoding.decode, Checkpoint.Encoding.encode,
    Checkpoint.Encoding.checkedWord, Numeric.Word.ofNatBits_eq,
    Numeric.Word.ofBits_toBits, hlt, bind, Except.bind, pure, Except.pure]

/--
A complex scalar stores its real word in the low half and its imaginary word in the high half,
each in the component encoding.
-/
instance {α : Type} [Encoding α] : Encoding (Complex α) where
  format := s!"torchlean_state_complex_v1({Encoding.format (α := α)})"
  width := 2 * Encoding.width (α := α)
  encode value :=
    Encoding.encode value.re + 2 ^ Encoding.width (α := α) * Encoding.encode value.im
  decode word := do
    let base := 2 ^ Encoding.width (α := α)
    pure ⟨← Encoding.decode (word % base), ← Encoding.decode (word / base)⟩

end Encoding
end TorchLean.Checkpoint
