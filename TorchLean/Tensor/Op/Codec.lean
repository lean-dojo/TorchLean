/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Wire

/-!
# The round trip of the wire codec

`Op.decode_encode` shows that `Op.decode` inverts `Op.encode` (`TorchLean.Tensor.Op.Wire`) at every
erased operator, from one lemma `Op.Wire.decode_encode_<op>` for each constructor of the table.
Equality of erased operators is decided through it, by comparing wire records
(`Op.instDecidableEqErased`).
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

namespace Op.Wire

/-- The round trip of `relu`. -/
theorem decode_encode_relu (s : Shape) :
    decode (encode ⟨_, _, Op.relu (s := s)⟩) = some ⟨_, _, Op.relu (s := s)⟩ :=
  rfl

/-- The round trip of `sigmoid`. -/
theorem decode_encode_sigmoid (s : Shape) :
    decode (encode ⟨_, _, Op.sigmoid (s := s)⟩) = some ⟨_, _, Op.sigmoid (s := s)⟩ :=
  rfl

/-- The round trip of `tanh`. -/
theorem decode_encode_tanh (s : Shape) :
    decode (encode ⟨_, _, Op.tanh (s := s)⟩) = some ⟨_, _, Op.tanh (s := s)⟩ :=
  rfl

/-- The round trip of `exp`. -/
theorem decode_encode_exp (s : Shape) :
    decode (encode ⟨_, _, Op.exp (s := s)⟩) = some ⟨_, _, Op.exp (s := s)⟩ :=
  rfl

/-- The round trip of `abs`. -/
theorem decode_encode_abs (s : Shape) :
    decode (encode ⟨_, _, Op.abs (s := s)⟩) = some ⟨_, _, Op.abs (s := s)⟩ :=
  rfl

/-- The round trip of `sqrt`. -/
theorem decode_encode_sqrt (s : Shape) (policy : Tensor.SqrtPolicy) :
    decode (encode ⟨_, _, Op.sqrt (s := s) policy⟩) =
      some ⟨_, _, Op.sqrt (s := s) policy⟩ := by
  cases policy <;> rfl

/-- The round trip of `sin`. -/
theorem decode_encode_sin (s : Shape) :
    decode (encode ⟨_, _, Op.sin (s := s)⟩) = some ⟨_, _, Op.sin (s := s)⟩ :=
  rfl

/-- The round trip of `cos`. -/
theorem decode_encode_cos (s : Shape) :
    decode (encode ⟨_, _, Op.cos (s := s)⟩) = some ⟨_, _, Op.cos (s := s)⟩ :=
  rfl

/-- The round trip of `neg`. -/
theorem decode_encode_neg (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.neg (s := s) e⟩) = some ⟨_, _, Op.neg (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `add`. -/
theorem decode_encode_add (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.add (s := s) e⟩) = some ⟨_, _, Op.add (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `mul`. -/
theorem decode_encode_mul (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.mul (s := s) e⟩) = some ⟨_, _, Op.mul (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `max`. -/
theorem decode_encode_max (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.max (s := s) e⟩) = some ⟨_, _, Op.max (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `log`. -/
theorem decode_encode_log (s : Shape) :
    decode (encode ⟨_, _, Op.log (s := s)⟩) = some ⟨_, _, Op.log (s := s)⟩ :=
  rfl

/-- The round trip of `detach`. -/
theorem decode_encode_detach (s : Shape) :
    decode (encode ⟨_, _, Op.detach (s := s)⟩) = some ⟨_, _, Op.detach (s := s)⟩ :=
  rfl

/-- The round trip of `softplus`. -/
theorem decode_encode_softplus (s : Shape) :
    decode (encode ⟨_, _, Op.softplus (s := s)⟩) = some ⟨_, _, Op.softplus (s := s)⟩ :=
  rfl

/-- The round trip of `silu`. -/
theorem decode_encode_silu (s : Shape) :
    decode (encode ⟨_, _, Op.silu (s := s)⟩) = some ⟨_, _, Op.silu (s := s)⟩ :=
  rfl

/-- The round trip of `square`. -/
theorem decode_encode_square (s : Shape) :
    decode (encode ⟨_, _, Op.square (s := s)⟩) = some ⟨_, _, Op.square (s := s)⟩ :=
  rfl

/-- The round trip of `reciprocal`. -/
theorem decode_encode_reciprocal (s : Shape) :
    decode (encode ⟨_, _, Op.reciprocal (s := s)⟩) = some ⟨_, _, Op.reciprocal (s := s)⟩ :=
  rfl

/-- The round trip of `sub`. -/
theorem decode_encode_sub (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.sub (s := s) e⟩) = some ⟨_, _, Op.sub (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `div`. -/
theorem decode_encode_div (s : Shape) (rounding : DivRounding) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.div (s := s) rounding e⟩) =
      some ⟨_, _, Op.div (s := s) rounding e⟩ := by
  cases rounding <;> cases e <;> rfl

/-- The round trip of `min`. -/
theorem decode_encode_min (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.min (s := s) e⟩) = some ⟨_, _, Op.min (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `cumsum`. -/
theorem decode_encode_cumsum (s : Shape) (dim : Fin (s.rank)) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.cumsum (s := s) (dim := dim) (order := order)⟩) =
      some ⟨_, _, Op.cumsum (s := s) (dim := dim) (order := order)⟩ := by
  have heval : decode (encode ⟨_, _, Op.cumsum (s := s) (dim := dim) (order := order)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cumsum (s := s) (dim := dim) (order := order)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `safeLog`. -/
theorem decode_encode_safeLog (s : Shape) :
    decode (encode ⟨_, _, Op.safeLog (s := s)⟩) = some ⟨_, _, Op.safeLog (s := s)⟩ :=
  rfl

/-- The round trip of `clamp`. -/
theorem decode_encode_clamp (s : Shape) :
    decode (encode ⟨_, _, Op.clamp (s := s)⟩) = some ⟨_, _, Op.clamp (s := s)⟩ :=
  rfl

/-- The round trip of `scale`. -/
theorem decode_encode_scale (s : Shape) :
    decode (encode ⟨_, _, Op.scale (s := s)⟩) = some ⟨_, _, Op.scale (s := s)⟩ :=
  rfl

/-- The round trip of `elu`. -/
theorem decode_encode_elu (s : Shape) :
    decode (encode ⟨_, _, Op.elu (s := s)⟩) = some ⟨_, _, Op.elu (s := s)⟩ :=
  rfl

/-- The round trip of `leakyReLU`. -/
theorem decode_encode_leakyReLU (s : Shape) :
    decode (encode ⟨_, _, Op.leakyReLU (s := s)⟩) = some ⟨_, _, Op.leakyReLU (s := s)⟩ :=
  rfl

/-- The round trip of `softmax`. -/
theorem decode_encode_softmax (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.softmax (s := s) (dim := dim)⟩) = some ⟨_, _, Op.softmax (s := s)
        (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.softmax (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.softmax (s := s) (dim := dim)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `logSoftmax`. -/
theorem decode_encode_logSoftmax (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.logSoftmax (s := s) (dim := dim)⟩) = some ⟨_, _, Op.logSoftmax
        (s := s) (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.logSoftmax (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.logSoftmax (s := s) (dim := dim)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `linear`. -/
theorem decode_encode_linear (lead : Shape) (inDim : Nat) (outDim : Nat) :
    decode (encode ⟨_, _, Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)⟩) =
      some ⟨_, _, Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)⟩ :=
  rfl

/-- The round trip of `attentionFromScores`. -/
theorem decode_encode_attentionFromScores (batch : Shape) (rows : Nat) (keys : Nat) (width : Nat)
    (mask : Option (Tensor Bool [rows, keys])) :
    decode (encode ⟨_, _, Op.attentionFromScores (batch := batch) (rows := rows) (keys := keys)
        (width := width) (mask := mask)⟩) = some ⟨_, _, Op.attentionFromScores (batch := batch)
        (rows := rows) (keys := keys) (width := width) (mask := mask)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.attentionFromScores (batch := batch) (rows := rows)
      (keys := keys) (width := width) (mask := mask)⟩) =
    (do
      let mask ← mask? rows keys ((mask.map maskEntries))
      pure ⟨_, _, Op.attentionFromScores (batch := batch) (rows := rows) (keys := keys)
          (width := width) (mask := mask)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `sort`. -/
theorem decode_encode_sort (s : Shape) (dim : Fin (s.rank)) (descending : Bool)
    (e : Op.Element) :
    decode (encode ⟨_, _, Op.sort (s := s) (dim := dim)
        (descending := descending) e⟩) = some ⟨_, _, Op.sort (s := s) (dim := dim)
        (descending := descending) e⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.sort (s := s) (dim := dim) (descending := descending)
      e⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.sort (s := s) (dim := dim) (descending := descending) e⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `topk`. -/
theorem decode_encode_topk (s : Shape) (dim : Fin (s.rank)) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) :
    decode (encode ⟨_, _, Op.topk (s := s) (dim := dim) (k := k) (largest := largest)
        (h := h)⟩) = some ⟨_, _, Op.topk (s := s) (dim := dim) (k := k) (largest := largest)
        (h := h)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.topk (s := s) (dim := dim) (k := k) (largest := largest)
      (h := h)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      let ⟨h⟩ ← prop? (k ≤ s.extent dim)
      pure ⟨_, _, Op.topk (s := s) (dim := dim) (k := k) (largest := largest) (h := h)⟩) := rfl
  rw [heval]
  simp [h]

/-- The round trip of `gather`. -/
theorem decode_encode_gather (s : Shape) (t : Shape) (dims : List (Fin (s.rank)))
    (batchDims : List (Fin (s.rank) × Fin (t.rank))) (h : Shape.GatherValid s t dims batchDims)
    (e : Op.Element) :
    decode (encode ⟨_, _, Op.gather (s := s) (t := t) (dims := dims) (batchDims := batchDims)
        (h := h) e⟩) = some ⟨_, _, Op.gather (s := s) (t := t) (dims := dims)
        (batchDims := batchDims) (h := h) e⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.gather (s := s) (t := t) (dims := dims)
      (batchDims := batchDims) (h := h) e⟩) =
    (do
      let dims ← finList? (s.rank) ((dims.map (·.val)))
      let batchDims ← finPairs? (s.rank) (t.rank) ((batchDims.map fun p => (p.1.val, p.2.val)))
      let ⟨h⟩ ← prop? (Shape.GatherValid s t dims batchDims)
      pure ⟨_, _, Op.gather (s := s) (t := t) (dims := dims) (batchDims := batchDims)
          (h := h) e⟩) := rfl
  rw [heval]
  simp [h]

/-- The round trip of `randUniform`. -/
theorem decode_encode_randUniform (s : Shape) (key : UInt64) :
    decode (encode ⟨_, _, Op.randUniform (s := s) (key := key)⟩) = some ⟨_, _, Op.randUniform
        (s := s) (key := key)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.randUniform (s := s) (key := key)⟩) =
    (do
      let key ← uint64? (key.toNat)
      pure ⟨_, _, Op.randUniform (s := s) (key := key)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `randn`. -/
theorem decode_encode_randn (s : Shape) (key : UInt64) :
    decode (encode ⟨_, _, Op.randn (s := s) (key := key)⟩) = some ⟨_, _, Op.randn
        (s := s) (key := key)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.randn (s := s) (key := key)⟩) =
    (do
      let key ← uint64? (key.toNat)
      pure ⟨_, _, Op.randn (s := s) (key := key)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `sign`. -/
theorem decode_encode_sign (s : Shape) :
    decode (encode ⟨_, _, Op.sign (s := s)⟩) = some ⟨_, _, Op.sign (s := s)⟩ :=
  rfl

/-- The round trip of `floor`. -/
theorem decode_encode_floor (s : Shape) :
    decode (encode ⟨_, _, Op.floor (s := s)⟩) = some ⟨_, _, Op.floor (s := s)⟩ :=
  rfl

/-- The round trip of `ceil`. -/
theorem decode_encode_ceil (s : Shape) :
    decode (encode ⟨_, _, Op.ceil (s := s)⟩) = some ⟨_, _, Op.ceil (s := s)⟩ :=
  rfl

/-- The round trip of `round`. -/
theorem decode_encode_round (s : Shape) :
    decode (encode ⟨_, _, Op.round (s := s)⟩) = some ⟨_, _, Op.round (s := s)⟩ :=
  rfl

/-- The round trip of `trunc`. -/
theorem decode_encode_trunc (s : Shape) :
    decode (encode ⟨_, _, Op.trunc (s := s)⟩) = some ⟨_, _, Op.trunc (s := s)⟩ :=
  rfl

/-- The round trip of `frac`. -/
theorem decode_encode_frac (s : Shape) :
    decode (encode ⟨_, _, Op.frac (s := s)⟩) = some ⟨_, _, Op.frac (s := s)⟩ :=
  rfl

/-- The round trip of `rsqrt`. -/
theorem decode_encode_rsqrt (s : Shape) :
    decode (encode ⟨_, _, Op.rsqrt (s := s)⟩) = some ⟨_, _, Op.rsqrt (s := s)⟩ :=
  rfl

/-- The round trip of `exp2`. -/
theorem decode_encode_exp2 (s : Shape) :
    decode (encode ⟨_, _, Op.exp2 (s := s)⟩) = some ⟨_, _, Op.exp2 (s := s)⟩ :=
  rfl

/-- The round trip of `expm1`. -/
theorem decode_encode_expm1 (s : Shape) :
    decode (encode ⟨_, _, Op.expm1 (s := s)⟩) = some ⟨_, _, Op.expm1 (s := s)⟩ :=
  rfl

/-- The round trip of `log2`. -/
theorem decode_encode_log2 (s : Shape) :
    decode (encode ⟨_, _, Op.log2 (s := s)⟩) = some ⟨_, _, Op.log2 (s := s)⟩ :=
  rfl

/-- The round trip of `log10`. -/
theorem decode_encode_log10 (s : Shape) :
    decode (encode ⟨_, _, Op.log10 (s := s)⟩) = some ⟨_, _, Op.log10 (s := s)⟩ :=
  rfl

/-- The round trip of `log1p`. -/
theorem decode_encode_log1p (s : Shape) :
    decode (encode ⟨_, _, Op.log1p (s := s)⟩) = some ⟨_, _, Op.log1p (s := s)⟩ :=
  rfl

/-- The round trip of `tan`. -/
theorem decode_encode_tan (s : Shape) :
    decode (encode ⟨_, _, Op.tan (s := s)⟩) = some ⟨_, _, Op.tan (s := s)⟩ :=
  rfl

/-- The round trip of `asinh`. -/
theorem decode_encode_asinh (s : Shape) :
    decode (encode ⟨_, _, Op.asinh (s := s)⟩) = some ⟨_, _, Op.asinh (s := s)⟩ :=
  rfl

/-- The round trip of `acosh`. -/
theorem decode_encode_acosh (s : Shape) :
    decode (encode ⟨_, _, Op.acosh (s := s)⟩) = some ⟨_, _, Op.acosh (s := s)⟩ :=
  rfl

/-- The round trip of `atanh`. -/
theorem decode_encode_atanh (s : Shape) :
    decode (encode ⟨_, _, Op.atanh (s := s)⟩) = some ⟨_, _, Op.atanh (s := s)⟩ :=
  rfl

/-- The round trip of `atan`. -/
theorem decode_encode_atan (s : Shape) :
    decode (encode ⟨_, _, Op.atan (s := s)⟩) = some ⟨_, _, Op.atan (s := s)⟩ :=
  rfl

/-- The round trip of `asin`. -/
theorem decode_encode_asin (s : Shape) :
    decode (encode ⟨_, _, Op.asin (s := s)⟩) = some ⟨_, _, Op.asin (s := s)⟩ :=
  rfl

/-- The round trip of `acos`. -/
theorem decode_encode_acos (s : Shape) :
    decode (encode ⟨_, _, Op.acos (s := s)⟩) = some ⟨_, _, Op.acos (s := s)⟩ :=
  rfl

/-- The round trip of `erf`. -/
theorem decode_encode_erf (s : Shape) :
    decode (encode ⟨_, _, Op.erf (s := s)⟩) = some ⟨_, _, Op.erf (s := s)⟩ :=
  rfl

/-- The round trip of `atan2`. -/
theorem decode_encode_atan2 (s : Shape) :
    decode (encode ⟨_, _, Op.atan2 (s := s)⟩) = some ⟨_, _, Op.atan2 (s := s)⟩ :=
  rfl

/-- The round trip of `fmod`. -/
theorem decode_encode_fmod (s : Shape) :
    decode (encode ⟨_, _, Op.fmod (s := s)⟩) = some ⟨_, _, Op.fmod (s := s)⟩ :=
  rfl

/-- The round trip of `remainder`. -/
theorem decode_encode_remainder (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.remainder (s := s) e⟩) = some ⟨_, _, Op.remainder (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `hypot`. -/
theorem decode_encode_hypot (s : Shape) :
    decode (encode ⟨_, _, Op.hypot (s := s)⟩) = some ⟨_, _, Op.hypot (s := s)⟩ :=
  rfl

/-- The round trip of `copysign`. -/
theorem decode_encode_copysign (s : Shape) :
    decode (encode ⟨_, _, Op.copysign (s := s)⟩) = some ⟨_, _, Op.copysign (s := s)⟩ :=
  rfl

/-- The round trip of `fmax`. -/
theorem decode_encode_fmax (s : Shape) :
    decode (encode ⟨_, _, Op.fmax (s := s)⟩) = some ⟨_, _, Op.fmax (s := s)⟩ :=
  rfl

/-- The round trip of `fmin`. -/
theorem decode_encode_fmin (s : Shape) :
    decode (encode ⟨_, _, Op.fmin (s := s)⟩) = some ⟨_, _, Op.fmin (s := s)⟩ :=
  rfl

/-- The round trip of `nanToNum`. -/
theorem decode_encode_nanToNum (s : Shape) :
    decode (encode ⟨_, _, Op.nanToNum (s := s)⟩) = some ⟨_, _, Op.nanToNum (s := s)⟩ :=
  rfl

/-- The round trip of `select`. -/
theorem decode_encode_select (s : Shape) (e : Op.Element) :
    decode (encode ⟨_, _, Op.select (s := s) e⟩) = some ⟨_, _, Op.select (s := s) e⟩ :=
  rfl

/-- The round trip of `lerp`. -/
theorem decode_encode_lerp (s : Shape) :
    decode (encode ⟨_, _, Op.lerp (s := s)⟩) = some ⟨_, _, Op.lerp (s := s)⟩ :=
  rfl

/-- The round trip of `addcmul`. -/
theorem decode_encode_addcmul (s : Shape) :
    decode (encode ⟨_, _, Op.addcmul (s := s)⟩) = some ⟨_, _, Op.addcmul (s := s)⟩ :=
  rfl

/-- The round trip of `addcdiv`. -/
theorem decode_encode_addcdiv (s : Shape) :
    decode (encode ⟨_, _, Op.addcdiv (s := s)⟩) = some ⟨_, _, Op.addcdiv (s := s)⟩ :=
  rfl

/-- The round trip of `gelu`. -/
theorem decode_encode_gelu (s : Shape) (approximate : GeluApproximation) :
    decode (encode ⟨_, _, Op.gelu (s := s) approximate⟩) =
      some ⟨_, _, Op.gelu (s := s) approximate⟩ := by
  cases approximate <;> rfl

/-- The round trip of `sinh`. -/
theorem decode_encode_sinh (s : Shape) :
    decode (encode ⟨_, _, Op.sinh (s := s)⟩) = some ⟨_, _, Op.sinh (s := s)⟩ :=
  rfl

/-- The round trip of `cosh`. -/
theorem decode_encode_cosh (s : Shape) :
    decode (encode ⟨_, _, Op.cosh (s := s)⟩) = some ⟨_, _, Op.cosh (s := s)⟩ :=
  rfl

/-- The round trip of `safeInv`. -/
theorem decode_encode_safeInv (s : Shape) :
    decode (encode ⟨_, _, Op.safeInv (s := s)⟩) = some ⟨_, _, Op.safeInv (s := s)⟩ :=
  rfl

/-- The round trip of `smoothAbs`. -/
theorem decode_encode_smoothAbs (s : Shape) :
    decode (encode ⟨_, _, Op.smoothAbs (s := s)⟩) = some ⟨_, _, Op.smoothAbs (s := s)⟩ :=
  rfl

/-- The round trip of `safeDiv`. -/
theorem decode_encode_safeDiv (s : Shape) :
    decode (encode ⟨_, _, Op.safeDiv (s := s)⟩) = some ⟨_, _, Op.safeDiv (s := s)⟩ :=
  rfl

/-- The round trip of `pow`. -/
theorem decode_encode_pow (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.pow (s := s) e⟩) = some ⟨_, _, Op.pow (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `powValue`. -/
theorem decode_encode_powValue (s : Shape) :
    decode (encode ⟨_, _, Op.powValue (s := s)⟩) = some ⟨_, _, Op.powValue (s := s)⟩ :=
  rfl

/-- The round trip of `cumprod`. -/
theorem decode_encode_cumprod (s : Shape) (dim : Fin (s.rank)) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.cumprod (s := s) (dim := dim) (order := order)⟩) =
      some ⟨_, _, Op.cumprod (s := s) (dim := dim) (order := order)⟩ := by
  have heval : decode (encode ⟨_, _, Op.cumprod (s := s) (dim := dim) (order := order)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cumprod (s := s) (dim := dim) (order := order)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `cummax`. -/
theorem decode_encode_cummax (s : Shape) (dim : Fin (s.rank)) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.cummax (s := s) (dim := dim) (order := order)⟩) =
      some ⟨_, _, Op.cummax (s := s) (dim := dim) (order := order)⟩ := by
  have heval : decode (encode ⟨_, _, Op.cummax (s := s) (dim := dim) (order := order)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cummax (s := s) (dim := dim) (order := order)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `cummin`. -/
theorem decode_encode_cummin (s : Shape) (dim : Fin (s.rank)) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.cummin (s := s) (dim := dim) (order := order)⟩) =
      some ⟨_, _, Op.cummin (s := s) (dim := dim) (order := order)⟩ := by
  have heval : decode (encode ⟨_, _, Op.cummin (s := s) (dim := dim) (order := order)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cummin (s := s) (dim := dim) (order := order)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `mseLoss`. -/
theorem decode_encode_mseLoss (s : Shape) :
    decode (encode ⟨_, _, Op.mseLoss (s := s)⟩) = some ⟨_, _, Op.mseLoss (s := s)⟩ :=
  rfl

/-- The round trip of `maeLoss`. -/
theorem decode_encode_maeLoss (s : Shape) :
    decode (encode ⟨_, _, Op.maeLoss (s := s)⟩) = some ⟨_, _, Op.maeLoss (s := s)⟩ :=
  rfl

/-- The round trip of `huberLoss`, at the mean and at the sum. -/
theorem decode_encode_huberLoss (s : Shape) (mean : Bool) :
    decode (encode ⟨_, _, Op.huberLoss (s := s) mean⟩) =
      some ⟨_, _, Op.huberLoss (s := s) mean⟩ :=
  rfl

/-- The round trip of `hingeLoss`. -/
theorem decode_encode_hingeLoss (s : Shape) :
    decode (encode ⟨_, _, Op.hingeLoss (s := s)⟩) = some ⟨_, _, Op.hingeLoss (s := s)⟩ :=
  rfl

/-- The round trip of `poissonLoss`. -/
theorem decode_encode_poissonLoss (s : Shape) :
    decode (encode ⟨_, _, Op.poissonLoss (s := s)⟩) = some ⟨_, _, Op.poissonLoss (s := s)⟩ :=
  rfl

/-- The round trip of `logCoshLoss`. -/
theorem decode_encode_logCoshLoss (s : Shape) :
    decode (encode ⟨_, _, Op.logCoshLoss (s := s)⟩) = some ⟨_, _, Op.logCoshLoss (s := s)⟩ :=
  rfl

/-- The round trip of `binaryCrossEntropy`. -/
theorem decode_encode_binaryCrossEntropy (s : Shape) :
    decode (encode ⟨_, _, Op.binaryCrossEntropy (s := s)⟩) = some ⟨_, _, Op.binaryCrossEntropy
        (s := s)⟩ :=
  rfl

/-- The round trip of `cosineLoss`. -/
theorem decode_encode_cosineLoss (s : Shape) :
    decode (encode ⟨_, _, Op.cosineLoss (s := s)⟩) = some ⟨_, _, Op.cosineLoss (s := s)⟩ :=
  rfl

/-- The round trip of `crossEntropy`. -/
theorem decode_encode_crossEntropy (s : Shape) (dim : Fin (s.rank)) (mean : Bool) :
    decode (encode ⟨_, _, Op.crossEntropy (s := s) (dim := dim)
        (mean := mean)⟩) = some ⟨_, _, Op.crossEntropy (s := s) (dim := dim) (mean := mean)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.crossEntropy (s := s) (dim := dim) (mean := mean)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.crossEntropy (s := s) (dim := dim) (mean := mean)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `crossEntropyProbabilities`. -/
theorem decode_encode_crossEntropyProbabilities (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.crossEntropyProbabilities (s := s)
        (dim := dim)⟩) = some ⟨_, _, Op.crossEntropyProbabilities (s := s) (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.crossEntropyProbabilities (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.crossEntropyProbabilities (s := s) (dim := dim)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `distill`. -/
theorem decode_encode_distill (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.distill (s := s) (dim := dim)⟩) = some ⟨_, _, Op.distill (s := s)
        (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.distill (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.distill (s := s) (dim := dim)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `reshape`. -/
theorem decode_encode_reshape (s : Shape) (t : Shape) (h : s.size = t.size) (e : Op.Element) :
    decode (encode ⟨_, _, Op.reshape (s := s) (t := t) (h := h) e⟩) = some ⟨_, _, Op.reshape
        (s := s) (t := t) (h := h) e⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.reshape (s := s) (t := t) (h := h) e⟩) =
    (do
      let ⟨h⟩ ← prop? (s.size = t.size)
      pure ⟨_, _, Op.reshape (s := s) (t := t) (h := h) e⟩) := rfl
  rw [heval]
  simp [h]

/-- The round trip of `permute`. -/
theorem decode_encode_permute (s : Shape) (perm : Equiv.Perm (Fin (s.rank))) (out : Shape)
    (hout : out = s.permute perm) (e : Op.Element) :
    decode (encode ⟨_, _, Op.permute (s := s) (perm := perm) (out := out)
        (hout := hout) e⟩) = some ⟨_, _, Op.permute (s := s) (perm := perm) (out := out)
        (hout := hout) e⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.permute (s := s) (perm := perm) (out := out)
      (hout := hout) e⟩) =
    (do
      let perm ← equiv? (s.rank) (s.rank) ((List.ofFn fun i => (perm i).val))
      let ⟨hout⟩ ← prop? (out = s.permute perm)
      pure ⟨_, _, Op.permute (s := s) (perm := perm) (out := out) (hout := hout) e⟩) := rfl
  rw [heval]
  simp [hout]

/-- The round trip of `broadcast`. -/
theorem decode_encode_broadcast (s : Shape) (t : Shape) (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (e : Op.Element) :
    decode (encode ⟨_, _, Op.broadcast (s := s) (t := t) dims h e⟩) =
      some ⟨_, _, Op.broadcast (s := s) (t := t) dims h e⟩ := by
  have heval : decode (encode ⟨_, _, Op.broadcast (s := s) (t := t) dims h e⟩) =
    (do
      let dims ← finFun? s.rank t.rank (List.ofFn fun i => (dims i).val)
      let ⟨h⟩ ← prop? (Shape.BroadcastDims s t dims)
      pure ⟨_, _, Op.broadcast (s := s) (t := t) dims h e⟩) := rfl
  rw [heval]
  simp [h]

/-- The round trip of `fullLike`. -/
theorem decode_encode_fullLike (s : Shape) :
    decode (encode ⟨_, _, Op.fullLike (s := s)⟩) = some ⟨_, _, Op.fullLike (s := s)⟩ :=
  rfl

/-- The round trip of `dropout`. -/
theorem decode_encode_dropout (s : Shape) :
    decode (encode ⟨_, _, Op.dropout (s := s)⟩) = some ⟨_, _, Op.dropout (s := s)⟩ :=
  rfl

/-- The round trip of `embedding`. -/
theorem decode_encode_embedding (num : Nat) (dim : Nat) (t : Shape) :
    decode (encode ⟨_, _, Op.embedding (num := num) (dim := dim)
        (t := t)⟩) = some ⟨_, _, Op.embedding (num := num) (dim := dim) (t := t)⟩ :=
  rfl

/-- The round trip of `matmul`, at every value of the vector flags. -/
theorem decode_encode_matmul (batchA : Shape) (batchB : Shape) (batch : Shape) (m : Nat)
    (n : Nat) (p : Nat) (lv rv : Bool) (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch)
        (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩) =
      some ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
        (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩ := by
  have heval : decode (encode ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB)
      (batch := batch) (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩) =
    (do
      let ⟨hl⟩ ← prop? (lv = true → m = 1)
      let ⟨hr⟩ ← prop? (rv = true → p = 1)
      let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
      let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
      pure ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
          (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩) := rfl
  rw [heval, prop?_of hl, prop?_of hr, prop?_of broadcastA, prop?_of broadcastB]
  rfl

/-- The round trip of `scaledMatmul`, at every value of the vector flags. -/
theorem decode_encode_scaledMatmul (batchA : Shape) (batchB : Shape) (batch : Shape) (m : Nat)
    (n : Nat) (p : Nat) (lv rv : Bool) (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch)
        (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩) =
      some ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
        (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩ := by
  have heval : decode (encode ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB)
      (batch := batch) (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩) =
    (do
      let ⟨hl⟩ ← prop? (lv = true → m = 1)
      let ⟨hr⟩ ← prop? (rv = true → p = 1)
      let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
      let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
      pure ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
          (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩) := rfl
  rw [heval, prop?_of hl, prop?_of hr, prop?_of broadcastA, prop?_of broadcastB]
  rfl

/-- The round trip of `conv`, at every leading shape and over any input and result shapes given by
their equations. -/
theorem decode_encode_conv (s : Shape) (k : Shape) (r : Nat) (dims : Shape.ConvDims s k r)
    (stride : Fin r → Nat) (padLow : Fin r → Nat) (padHigh : Fin r → Nat)
    (dilation : Fin r → Nat) (groups : Nat) (lead x y : Shape)
    (hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size)
    (hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead)
    (hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead) :
    decode (encode ⟨_, _, Op.conv (s := s) (k := k) (r := r) (dims := dims) (stride := stride)
        (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
        (groups := groups) hlead hx hy⟩) = some ⟨_, _, Op.conv (s := s) (k := k) (r := r)
        (dims := dims) (stride := stride) (padLow := padLow) (padHigh := padHigh)
        (dilation := dilation) (groups := groups) hlead hx hy⟩ := by
  subst hx hy
  have heval : decode (encode ⟨_, _, Op.conv (s := s) (k := k) (r := r) (dims := dims)
      (stride := stride) (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
      (groups := groups) hlead rfl rfl⟩) =
    (do
      let dimsIn ← equiv? (r + 2) s.rank ((List.ofFn fun i => (dims.input i).val))
      let dimsKer ← equiv? (r + 2) k.rank ((List.ofFn fun i => (dims.kernel i).val))
      let dims : Shape.ConvDims s k r := ⟨dimsIn, dimsKer⟩
      let stride ← fun? r ((List.ofFn stride))
      let padLow ← fun? r ((List.ofFn padLow))
      let padHigh ← fun? r ((List.ofFn padHigh))
      let dilation ← fun? r ((List.ofFn dilation))
      let ⟨hlead⟩ ← prop? (s.extent (dims.input ⟨0, Nat.zero_lt_succ _⟩) = lead.size)
      pure ⟨_, _, Op.conv (s := s) (k := k) (r := r) (dims := dims) (stride := stride)
          (padLow := padLow) (padHigh := padHigh) (dilation := dilation) (groups := groups)
          (lead := lead) hlead rfl rfl⟩) := rfl
  rw [heval]
  simp only [equiv?_ofFn, fun?_ofFn, bind, Option.bind, prop?_of hlead]
  rfl

/-- The round trip of `pool`. -/
theorem decode_encode_pool (s : Shape) (op : PoolOp) (dims : List (Fin (s.rank)))
    (kernel : List Nat) (stride : List Nat) (padLow : List Nat) (padHigh : List Nat)
    (dilation : List Nat) (ceilMode : Bool) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.pool (s := s) (op := op) (dims := dims) (kernel := kernel)
        (stride := stride) (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
        (ceilMode := ceilMode) (order := order)⟩) = some ⟨_, _, Op.pool (s := s) (op := op)
        (dims := dims) (kernel := kernel) (stride := stride) (padLow := padLow)
        (padHigh := padHigh) (dilation := dilation) (ceilMode := ceilMode) (order := order)⟩ := by
  have heval : decode (encode ⟨_, _, Op.pool (s := s) (op := op) (dims := dims)
      (kernel := kernel) (stride := stride) (padLow := padLow) (padHigh := padHigh)
      (dilation := dilation) (ceilMode := ceilMode) (order := order)⟩) =
    (do
      let op ← poolOp? ((poolOpCode op))
      let dims ← finList? (s.rank) ((dims.map (·.val)))
      pure ⟨_, _, Op.pool (s := s) (op := op) (dims := dims) (kernel := kernel)
          (stride := stride) (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
          (ceilMode := ceilMode) (order := order)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `smoothMaxPool`. -/
theorem decode_encode_smoothMaxPool (s : Shape) (dims : List (Fin (s.rank))) (kernel : List Nat)
    (stride : List Nat) (padLow : List Nat) (padHigh : List Nat) (dilation : List Nat)
    (ceilMode : Bool) :
    decode (encode ⟨_, _, Op.smoothMaxPool (s := s) (dims := dims) (kernel := kernel)
        (stride := stride) (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
        (ceilMode := ceilMode)⟩) = some ⟨_, _, Op.smoothMaxPool (s := s) (dims := dims)
        (kernel := kernel) (stride := stride) (padLow := padLow) (padHigh := padHigh)
        (dilation := dilation) (ceilMode := ceilMode)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.smoothMaxPool (s := s) (dims := dims) (kernel := kernel)
      (stride := stride) (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
      (ceilMode := ceilMode)⟩) =
    (do
      let dims ← finList? (s.rank) ((dims.map (·.val)))
      pure ⟨_, _, Op.smoothMaxPool (s := s) (dims := dims) (kernel := kernel) (stride := stride)
          (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
          (ceilMode := ceilMode)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `layerNorm`, at every leading shape and over any input given by its
equation. -/
theorem decode_encode_layerNorm (seqLen embedDim : Nat) (hSeq : seqLen > 0)
    (hEmbed : embedDim > 0) (lead x : Shape) (hlead : seqLen = lead.size)
    (hx : x = lead ++ [embedDim]) :
    decode (encode ⟨_, _, Op.layerNorm (seqLen := seqLen) (embedDim := embedDim) hSeq hEmbed hlead
        hx⟩) =
      some ⟨_, _, Op.layerNorm (seqLen := seqLen) (embedDim := embedDim) hSeq hEmbed hlead hx⟩ := by
  subst hx
  have heval : decode (encode ⟨_, _, Op.layerNorm (seqLen := seqLen) (embedDim := embedDim)
      (lead := lead) hSeq hEmbed hlead rfl⟩) =
    (do
      let ⟨hSeq⟩ ← prop? (seqLen > 0)
      let ⟨hEmbed⟩ ← prop? (embedDim > 0)
      let ⟨hlead⟩ ← prop? (seqLen = lead.size)
      pure ⟨_, _, Op.layerNorm (seqLen := seqLen) (embedDim := embedDim) (lead := lead) hSeq
          hEmbed hlead rfl⟩) := rfl
  rw [heval]
  simp only [bind, Option.bind, prop?_of hSeq, prop?_of hEmbed, prop?_of hlead]
  rfl

/-- The round trip of `batchNorm`. -/
theorem decode_encode_batchNorm (channels : Nat) (spatial : Shape) (hWellFormed :
    (Shape.dim channels spatial).Positive) :
    decode (encode ⟨_, _, Op.batchNorm (channels := channels) (spatial := spatial)
        (hWellFormed := hWellFormed)⟩) = some ⟨_, _, Op.batchNorm (channels := channels)
        (spatial := spatial) (hWellFormed := hWellFormed)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.batchNorm (channels := channels) (spatial := spatial)
      (hWellFormed := hWellFormed)⟩) =
    (do
      let ⟨hWellFormed⟩ ← prop? ((Shape.dim channels spatial).Positive)
      pure ⟨_, _, Op.batchNorm (channels := channels) (spatial := spatial)
          (hWellFormed := hWellFormed)⟩) := rfl
  rw [heval]
  simp [hWellFormed]

/-- The round trip of `batchNormRunning`. -/
theorem decode_encode_batchNormRunning (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.batchNormRunning (s := s) dim⟩) =
      some ⟨_, _, Op.batchNormRunning (s := s) dim⟩ := by
  have heval : decode (encode ⟨_, _, Op.batchNormRunning (s := s) dim⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.batchNormRunning (s := s) dim⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `multiheadAttention`. -/
theorem decode_encode_multiheadAttention (batch : Shape) (n : Nat) (numHeads : Nat)
    (dModel : Nat) (headDim : Nat) (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n])) :
    decode (encode ⟨_, _, Op.multiheadAttention (batch := batch) (n := n) (numHeads := numHeads)
        (dModel := dModel) (headDim := headDim) (h1 := h1)
        (mask := mask)⟩) = some ⟨_, _, Op.multiheadAttention (batch := batch) (n := n)
        (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        (mask := mask)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.multiheadAttention (batch := batch) (n := n)
      (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      (mask := mask)⟩) =
    (do
      let ⟨h1⟩ ← prop? (n ≠ 0)
      let mask ← mask? n n (mask.map maskEntries)
      pure ⟨_, _, Op.multiheadAttention (batch := batch) (n := n) (numHeads := numHeads)
          (dModel := dModel) (headDim := headDim) (h1 := h1) (mask := mask)⟩) := rfl
  rw [heval]
  simp [h1]

/-- The round trip of `convTranspose`. -/
theorem decode_encode_convTranspose (s : Shape) (k : Shape) (r : Nat)
    (dims : Shape.ConvDims s k r) (stride : Fin r → Nat) (padLow : Fin r → Nat)
    (padHigh : Fin r → Nat) (outputPadding : Fin r → Nat) (dilation : Fin r → Nat) (groups : Nat) :
    decode (encode ⟨_, _, Op.convTranspose (s := s) (k := k) (r := r) (dims := dims)
        (stride := stride) (padLow := padLow) (padHigh := padHigh)
        (outputPadding := outputPadding) (dilation := dilation)
        (groups := groups)⟩) = some ⟨_, _, Op.convTranspose (s := s) (k := k) (r := r)
        (dims := dims) (stride := stride) (padLow := padLow) (padHigh := padHigh)
        (outputPadding := outputPadding) (dilation := dilation) (groups := groups)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.convTranspose (s := s) (k := k) (r := r) (dims := dims)
      (stride := stride) (padLow := padLow) (padHigh := padHigh) (outputPadding := outputPadding)
      (dilation := dilation) (groups := groups)⟩) =
    (do
      let dimsIn ← equiv? (r + 2) s.rank ((List.ofFn fun i => (dims.input i).val))
      let dimsKer ← equiv? (r + 2) k.rank ((List.ofFn fun i => (dims.kernel i).val))
      let dims : Shape.ConvDims s k r := ⟨dimsIn, dimsKer⟩
      let stride ← fun? r ((List.ofFn stride))
      let padLow ← fun? r ((List.ofFn padLow))
      let padHigh ← fun? r ((List.ofFn padHigh))
      let outputPadding ← fun? r ((List.ofFn outputPadding))
      let dilation ← fun? r ((List.ofFn dilation))
      pure ⟨_, _, Op.convTranspose (s := s) (k := k) (r := r) (dims := dims) (stride := stride)
          (padLow := padLow) (padHigh := padHigh) (outputPadding := outputPadding)
          (dilation := dilation) (groups := groups)⟩) := rfl
  rw [heval]
  simp only [equiv?_ofFn, fun?_ofFn]
  rfl

/-- The round trip of `eq`. -/
theorem decode_encode_eq (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.eq (s := s) e⟩) = some ⟨_, _, Op.eq (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `ne`. -/
theorem decode_encode_ne (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.ne (s := s) e⟩) = some ⟨_, _, Op.ne (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `lt`. -/
theorem decode_encode_lt (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.lt (s := s) e⟩) = some ⟨_, _, Op.lt (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `le`. -/
theorem decode_encode_le (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.le (s := s) e⟩) = some ⟨_, _, Op.le (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `gt`. -/
theorem decode_encode_gt (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.gt (s := s) e⟩) = some ⟨_, _, Op.gt (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `ge`. -/
theorem decode_encode_ge (s : Shape) (e : Op.NumericElement) :
    decode (encode ⟨_, _, Op.ge (s := s) e⟩) = some ⟨_, _, Op.ge (s := s) e⟩ := by
  cases e <;> rfl

/-- The round trip of `isNaN`. -/
theorem decode_encode_isNaN (s : Shape) :
    decode (encode ⟨_, _, Op.isNaN (s := s)⟩) = some ⟨_, _, Op.isNaN (s := s)⟩ :=
  rfl

/-- The round trip of `isInf`. -/
theorem decode_encode_isInf (s : Shape) :
    decode (encode ⟨_, _, Op.isInf (s := s)⟩) = some ⟨_, _, Op.isInf (s := s)⟩ :=
  rfl

/-- The round trip of `isFinite`. -/
theorem decode_encode_isFinite (s : Shape) :
    decode (encode ⟨_, _, Op.isFinite (s := s)⟩) = some ⟨_, _, Op.isFinite (s := s)⟩ :=
  rfl

/-- The round trip of `logicalNot`. -/
theorem decode_encode_logicalNot (s : Shape) :
    decode (encode ⟨_, _, Op.logicalNot (s := s)⟩) = some ⟨_, _, Op.logicalNot (s := s)⟩ :=
  rfl

/-- The round trip of `logicalAnd`. -/
theorem decode_encode_logicalAnd (s : Shape) :
    decode (encode ⟨_, _, Op.logicalAnd (s := s)⟩) = some ⟨_, _, Op.logicalAnd (s := s)⟩ :=
  rfl

/-- The round trip of `logicalOr`. -/
theorem decode_encode_logicalOr (s : Shape) :
    decode (encode ⟨_, _, Op.logicalOr (s := s)⟩) = some ⟨_, _, Op.logicalOr (s := s)⟩ :=
  rfl

/-- The round trip of `logicalXor`. -/
theorem decode_encode_logicalXor (s : Shape) :
    decode (encode ⟨_, _, Op.logicalXor (s := s)⟩) = some ⟨_, _, Op.logicalXor (s := s)⟩ :=
  rfl

/-- The round trip of `bitwiseAnd`. -/
theorem decode_encode_bitwiseAnd (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.bitwiseAnd (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.bitwiseAnd (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `bitwiseOr`. -/
theorem decode_encode_bitwiseOr (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.bitwiseOr (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.bitwiseOr (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `bitwiseXor`. -/
theorem decode_encode_bitwiseXor (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.bitwiseXor (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.bitwiseXor (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `bitwiseNot`. -/
theorem decode_encode_bitwiseNot (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.bitwiseNot (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.bitwiseNot (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `bitwiseLeftShift`. -/
theorem decode_encode_bitwiseLeftShift (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.bitwiseLeftShift (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.bitwiseLeftShift (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `shiftRightArithmetic`. -/
theorem decode_encode_shiftRightArithmetic (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.shiftRightArithmetic (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.shiftRightArithmetic (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `shiftRightLogical`. -/
theorem decode_encode_shiftRightLogical (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.shiftRightLogical (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.shiftRightLogical (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `populationCount`. -/
theorem decode_encode_populationCount (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.populationCount (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.populationCount (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `countLeadingZeros`. -/
theorem decode_encode_countLeadingZeros (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.countLeadingZeros (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.countLeadingZeros (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `mulhi`. -/
theorem decode_encode_mulhi (signed : Bool) (width : Nat) (s : Shape) :
    decode (encode ⟨_, _, Op.mulhi (signed := signed) (width := width) (s := s)⟩) =
      some ⟨_, _, Op.mulhi (signed := signed) (width := width) (s := s)⟩ :=
  rfl

/-- The round trip of `bitcast`. -/
theorem decode_encode_bitcast (s : Shape) (width : Nat) (source target : Op.Word) :
    decode (encode ⟨_, _, Op.bitcast (s := s) width source target⟩) =
      some ⟨_, _, Op.bitcast (s := s) width source target⟩ := by
  obtain _ | ⟨_ | _⟩ := source <;> obtain _ | ⟨_ | _⟩ := target <;> rfl

/-- The round trip of `cbrt`. -/
theorem decode_encode_cbrt (s : Shape) :
    decode (encode ⟨_, _, Op.cbrt (s := s)⟩) = some ⟨_, _, Op.cbrt (s := s)⟩ :=
  rfl

/-- The round trip of `special`, which reads the function back from its name
(`SpecialFn.ofTorchName?_torchName`). -/
theorem decode_encode_special (orders : List ℕ) (f : SpecialFn orders.length) (s : Shape) :
    decode (encode ⟨_, _, Op.special orders f (s := s)⟩) =
      some ⟨_, _, Op.special orders f (s := s)⟩ := by
  have h : decode (encode ⟨_, _, Op.special orders f (s := s)⟩) =
      (do let f ← SpecialFn.ofTorchName? orders.length f.torchName
          pure ⟨_, _, Op.special orders f (s := s)⟩) := rfl
  rw [h, SpecialFn.ofTorchName?_torchName]
  rfl

/-- The round trip of `nextafter`. -/
theorem decode_encode_nextafter (s : Shape) :
    decode (encode ⟨_, _, Op.nextafter (s := s)⟩) = some ⟨_, _, Op.nextafter (s := s)⟩ :=
  rfl

/-- The round trip of `frexp`. -/
theorem decode_encode_frexp (s : Shape) :
    decode (encode ⟨_, _, Op.frexp (s := s)⟩) = some ⟨_, _, Op.frexp (s := s)⟩ :=
  rfl

/-- The round trip of `fma`. -/
theorem decode_encode_fma (s : Shape) :
    decode (encode ⟨_, _, Op.fma (s := s)⟩) = some ⟨_, _, Op.fma (s := s)⟩ :=
  rfl

/-- The round trip of `complex`, which reads the function back from its name
(`ComplexFn.ofJaxName?_jaxName`). -/
theorem decode_encode_complex (f : ComplexFn) (s : Shape) :
    decode (encode ⟨_, _, Op.complex f (s := s)⟩) = some ⟨_, _, Op.complex f (s := s)⟩ := by
  have h : decode (encode ⟨_, _, Op.complex f (s := s)⟩) =
      (do let f ← ComplexFn.ofJaxName? f.jaxName
          pure ⟨_, _, Op.complex f (s := s)⟩) := rfl
  rw [h, ComplexFn.ofJaxName?_jaxName]
  rfl

/-- The round trip of `complexPow`. -/
theorem decode_encode_complexPow (s : Shape) :
    decode (encode ⟨_, _, Op.complexPow (s := s)⟩) = some ⟨_, _, Op.complexPow (s := s)⟩ :=
  rfl

/-- The round trip of `argmax`. -/
theorem decode_encode_argmax (s : Shape) (dim : Fin (s.rank)) (keepdim : Bool)
    (h : 0 < s.extent dim) :
    decode (encode ⟨_, _, Op.argmax (s := s) (dim := dim) (keepdim := keepdim)
        (h := h)⟩) = some ⟨_, _, Op.argmax (s := s) (dim := dim) (keepdim := keepdim) (h := h)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.argmax (s := s) (dim := dim) (keepdim := keepdim)
      (h := h)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      let ⟨h⟩ ← prop? (0 < s.extent dim)
      pure ⟨_, _, Op.argmax (s := s) (dim := dim) (keepdim := keepdim) (h := h)⟩) := rfl
  rw [heval]
  simp [h]

/-- The round trip of `argmin`. -/
theorem decode_encode_argmin (s : Shape) (dim : Fin (s.rank)) (keepdim : Bool)
    (h : 0 < s.extent dim) :
    decode (encode ⟨_, _, Op.argmin (s := s) (dim := dim) (keepdim := keepdim)
        (h := h)⟩) = some ⟨_, _, Op.argmin (s := s) (dim := dim) (keepdim := keepdim) (h := h)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.argmin (s := s) (dim := dim) (keepdim := keepdim)
      (h := h)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      let ⟨h⟩ ← prop? (0 < s.extent dim)
      pure ⟨_, _, Op.argmin (s := s) (dim := dim) (keepdim := keepdim) (h := h)⟩) := rfl
  rw [heval]
  simp [h]

/-- The round trip of `flip`. -/
theorem decode_encode_flip (s : Shape) (dims : List (Fin (s.rank))) (e : Op.Element) :
    decode (encode ⟨_, _, Op.flip (s := s) (dims := dims) e⟩) = some ⟨_, _, Op.flip (s := s)
        (dims := dims) e⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.flip (s := s) (dims := dims) e⟩) =
    (do
      let dims ← finList? (s.rank) ((dims.map (·.val)))
      pure ⟨_, _, Op.flip (s := s) (dims := dims) e⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `indices`. -/
theorem decode_encode_indices (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.indices (s := s) (dim := dim)⟩) = some ⟨_, _, Op.indices (s := s)
        (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.indices (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.indices (s := s) (dim := dim)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `cast`. -/
theorem decode_encode_cast (s : Shape) (source : Op.CastSource) :
    decode (encode ⟨_, _, Op.cast (s := s) source⟩) = some ⟨_, _, Op.cast (s := s) source⟩ := by
  have heval : decode (encode ⟨_, _, Op.cast (s := s) source⟩) =
      some ⟨_, _, Op.cast (s := s) (castSource? (castSourceCode source))⟩ := rfl
  rw [heval, castSource?_code]

/-- The round trip of `encodeIndex`. -/
theorem decode_encode_encodeIndex (s : Shape) (bounds : List Nat) (signed : Bool) (width : Nat) :
    decode (encode ⟨_, _, Op.encodeIndex (s := s) bounds signed width⟩) =
      some ⟨_, _, Op.encodeIndex (s := s) bounds signed width⟩ :=
  rfl

/-- The round trip of `decodeIndex`. -/
theorem decode_encode_decodeIndex (s : Shape) (bounds : List Nat) (signed : Bool) (width : Nat) :
    decode (encode ⟨_, _, Op.decodeIndex (s := s) bounds signed width⟩) =
      some ⟨_, _, Op.decodeIndex (s := s) bounds signed width⟩ :=
  rfl

/-- The round trip of `attention`. -/
theorem decode_encode_attention (batch : Shape) (rows : Nat) (keys : Nat) (depth : Nat)
    (width : Nat) (mask : Option (Tensor Bool [rows, keys])) (schedule : Softmax.Schedule) :
    decode (encode ⟨_, _, Op.attention (batch := batch) (rows := rows) (keys := keys)
        (depth := depth) (width := width) (mask := mask)
        (schedule := schedule)⟩) = some ⟨_, _, Op.attention (batch := batch) (rows := rows)
        (keys := keys) (depth := depth) (width := width) (mask := mask) (schedule := schedule)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.attention (batch := batch) (rows := rows) (keys := keys)
      (depth := depth) (width := width) (mask := mask) (schedule := schedule)⟩) =
    (do
      let mask ← mask? rows keys ((mask.map maskEntries))
      let schedule := schedule? ((scheduleCode schedule))
      pure ⟨_, _, Op.attention (batch := batch) (rows := rows) (keys := keys) (depth := depth)
          (width := width) (mask := mask) (schedule := schedule)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `reduce`. -/
theorem decode_encode_reduce (s : Shape) (op : Tensor.ReduceOp) (dims : List (Fin (s.rank)))
    (keepdim : Bool) (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d)
    (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.reduce (s := s) op dims keepdim h order⟩) =
      some ⟨_, _, Op.reduce (s := s) op dims keepdim h order⟩ := by
  have heval : decode (encode ⟨_, _, Op.reduce (s := s) op dims keepdim h order⟩) =
    (do
      let op ← reduceOp? (reduceOpCode op)
      let dims ← finList? (s.rank) (dims.map (·.val))
      let ⟨h⟩ ← prop? (op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d)
      pure ⟨_, _, Op.reduce (s := s) op dims keepdim h order⟩) := rfl
  rw [heval]
  simp [prop?_of h]

/-- The round trip of `mean`. -/
theorem decode_encode_mean (s : Shape) (dims : List (Fin (s.rank))) (keepdim : Bool)
    (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.mean (s := s) dims keepdim order⟩) =
      some ⟨_, _, Op.mean (s := s) dims keepdim order⟩ := by
  have heval : decode (encode ⟨_, _, Op.mean (s := s) dims keepdim order⟩) =
    (do
      let dims ← finList? (s.rank) (dims.map (·.val))
      pure ⟨_, _, Op.mean (s := s) dims keepdim order⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `contract`. -/
theorem decode_encode_contract (s t : Shape) (dims batchDims : List (Fin s.rank × Fin t.rank))
    (h : Shape.ContractValid s t dims batchDims) (order : ReduceOrder) :
    decode (encode ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h order⟩) =
      some ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h order⟩ := by
  have heval : decode (encode ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h order⟩) =
    (do
      let dims ← finPairs? s.rank t.rank (dims.map fun p => (p.1.val, p.2.val))
      let batchDims ← finPairs? s.rank t.rank (batchDims.map fun p => (p.1.val, p.2.val))
      let ⟨h⟩ ← prop? (Shape.ContractValid s t dims batchDims)
      pure ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h order⟩) := rfl
  rw [heval]
  simp [prop?_of h]

/-- The round trip of `maskedSoftmax`. -/
theorem decode_encode_maskedSoftmax (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.maskedSoftmax (s := s) dim⟩) =
      some ⟨_, _, Op.maskedSoftmax (s := s) dim⟩ := by
  have heval : decode (encode ⟨_, _, Op.maskedSoftmax (s := s) dim⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.maskedSoftmax (s := s) dim⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `slice`. -/
theorem decode_encode_slice (s : Shape) (dims : List (Fin (s.rank))) (starts stops steps : List Nat)
    (h : Shape.SliceValid s dims starts stops steps) (e : Op.Element) :
    decode (encode ⟨_, _, Op.slice (s := s) dims starts stops steps h e⟩) =
      some ⟨_, _, Op.slice (s := s) dims starts stops steps h e⟩ := by
  have heval : decode (encode ⟨_, _, Op.slice (s := s) dims starts stops steps h e⟩) =
    (do
      let dims ← finList? (s.rank) (dims.map (·.val))
      let ⟨h⟩ ← prop? (Shape.SliceValid s dims starts stops steps)
      pure ⟨_, _, Op.slice (s := s) dims starts stops steps h e⟩) := rfl
  rw [heval]
  simp [prop?_of h]

/-- The round trip of `pad`. -/
theorem decode_encode_pad (s : Shape) (low high interior : Fin s.rank → Nat) (e : Op.Element) :
    decode (encode ⟨_, _, Op.pad (s := s) low high interior e⟩) =
      some ⟨_, _, Op.pad (s := s) low high interior e⟩ := by
  have heval : decode (encode ⟨_, _, Op.pad (s := s) low high interior e⟩) =
    (do
      let low ← fun? s.rank (List.ofFn low)
      let high ← fun? s.rank (List.ofFn high)
      let interior ← fun? s.rank (List.ofFn interior)
      pure ⟨_, _, Op.pad (s := s) low high interior e⟩) := rfl
  rw [heval]
  simp only [fun?_ofFn]
  rfl

/-- The round trip of `indexAt`. -/
theorem decode_encode_indexAt (s : Shape) (dim : Fin (s.rank)) (index : Fin (s.extent dim)) :
    decode (encode ⟨_, _, Op.indexAt (s := s) dim index⟩) =
      some ⟨_, _, Op.indexAt (s := s) dim index⟩ := by
  have heval : decode (encode ⟨_, _, Op.indexAt (s := s) dim index⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      let index ← fin? (s.extent dim) (index.val)
      pure ⟨_, _, Op.indexAt (s := s) dim index⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `cat`. -/
theorem decode_encode_cat (s : Shape) (dim : Fin (s.rank)) (sizes : List Nat) (e : Op.Element) :
    decode (encode ⟨_, _, Op.cat (s := s) dim sizes e⟩) =
      some ⟨_, _, Op.cat (s := s) dim sizes e⟩ := by
  have heval : decode (encode ⟨_, _, Op.cat (s := s) dim sizes e⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cat (s := s) dim sizes e⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `scatter`. -/
theorem decode_encode_scatter (s t : Shape) (rule : Tensor.ScatterReduce)
    (dims : List (Fin (s.rank))) (h : dims.Nodup) (e : Op.Element) :
    decode (encode ⟨_, _, Op.scatter (s := s) (t := t) rule dims h e⟩) =
      some ⟨_, _, Op.scatter (s := s) (t := t) rule dims h e⟩ := by
  have heval : decode (encode ⟨_, _, Op.scatter (s := s) (t := t) rule dims h e⟩) =
    (do
      let rule ← scatter? (scatterCode rule)
      let dims ← finList? (s.rank) (dims.map (·.val))
      let ⟨h⟩ ← prop? dims.Nodup
      pure ⟨_, _, Op.scatter (s := s) (t := t) rule dims h e⟩) := rfl
  rw [heval]
  simp [prop?_of h]

/-- The round trip of `topKMask`. -/
theorem decode_encode_topKMask (s : Shape) (dim : Fin (s.rank)) (k : Nat) :
    decode (encode ⟨_, _, Op.topKMask (s := s) dim k⟩) =
      some ⟨_, _, Op.topKMask (s := s) dim k⟩ := by
  have heval : decode (encode ⟨_, _, Op.topKMask (s := s) dim k⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.topKMask (s := s) dim k⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `bernoulliMask`. -/
theorem decode_encode_bernoulliMask (s : Shape) (key : UInt64) :
    decode (encode ⟨_, _, Op.bernoulliMask (s := s) key⟩) =
      some ⟨_, _, Op.bernoulliMask (s := s) key⟩ := by
  have heval : decode (encode ⟨_, _, Op.bernoulliMask (s := s) key⟩) =
    (do
      let key ← uint64? key.toNat
      pure ⟨_, _, Op.bernoulliMask (s := s) key⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `rfft`. -/
theorem decode_encode_rfft (s : Shape) (dim : Fin (s.rank)) (n : Nat) (hn : 0 < n) :
    decode (encode ⟨_, _, Op.rfft (s := s) dim n hn⟩) =
      some ⟨_, _, Op.rfft (s := s) dim n hn⟩ := by
  have heval : decode (encode ⟨_, _, Op.rfft (s := s) dim n hn⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      let ⟨hn⟩ ← prop? (0 < n)
      pure ⟨_, _, Op.rfft (s := s) dim n hn⟩) := rfl
  rw [heval]
  simp [hn]

/-- The round trip of `irfft`. -/
theorem decode_encode_irfft (s : Shape) (dim : Fin (s.rank)) (n : Nat) (hn : 0 < n) :
    decode (encode ⟨_, _, Op.irfft (s := s) dim n hn⟩) =
      some ⟨_, _, Op.irfft (s := s) dim n hn⟩ := by
  have heval : decode (encode ⟨_, _, Op.irfft (s := s) dim n hn⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      let ⟨hn⟩ ← prop? (0 < n)
      pure ⟨_, _, Op.irfft (s := s) dim n hn⟩) := rfl
  rw [heval]
  simp [hn]

/-- The round trip of `selectiveScan`. -/
theorem decode_encode_selectiveScan (s : Shape) (dim : Fin (s.rank)) (sharedA sharedB : Bool) :
    decode (encode ⟨_, _, Op.selectiveScan (s := s) dim sharedA sharedB⟩) =
      some ⟨_, _, Op.selectiveScan (s := s) dim sharedA sharedB⟩ := by
  have heval : decode (encode ⟨_, _, Op.selectiveScan (s := s) dim sharedA sharedB⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.selectiveScan (s := s) dim sharedA sharedB⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `custom`. -/
theorem decode_encode_custom (name : String) (inputs : List Shape) (output : Shape) :
    decode (encode ⟨_, _, Op.custom (name := name) (inputs := inputs)
        (output := output)⟩) = some ⟨_, _, Op.custom (name := name) (inputs := inputs)
        (output := output)⟩ :=
  rfl

end Op.Wire

/-- Decoding inverts encoding, for every operator. -/
theorem Op.decode_encode (op : Σ ins outs, Op ins outs) : Op.decode (Op.encode op) = some op := by
  obtain ⟨ins, outs, op⟩ := op
  cases op
  case relu => exact Op.Wire.decode_encode_relu ..
  case sigmoid => exact Op.Wire.decode_encode_sigmoid ..
  case tanh => exact Op.Wire.decode_encode_tanh ..
  case exp => exact Op.Wire.decode_encode_exp ..
  case abs => exact Op.Wire.decode_encode_abs ..
  case sqrt => exact Op.Wire.decode_encode_sqrt ..
  case sin => exact Op.Wire.decode_encode_sin ..
  case cos => exact Op.Wire.decode_encode_cos ..
  case neg => exact Op.Wire.decode_encode_neg ..
  case add => exact Op.Wire.decode_encode_add ..
  case mul => exact Op.Wire.decode_encode_mul ..
  case max => exact Op.Wire.decode_encode_max ..
  case log => exact Op.Wire.decode_encode_log ..
  case detach => exact Op.Wire.decode_encode_detach ..
  case softplus => exact Op.Wire.decode_encode_softplus ..
  case silu => exact Op.Wire.decode_encode_silu ..
  case square => exact Op.Wire.decode_encode_square ..
  case reciprocal => exact Op.Wire.decode_encode_reciprocal ..
  case sub => exact Op.Wire.decode_encode_sub ..
  case div => exact Op.Wire.decode_encode_div ..
  case min => exact Op.Wire.decode_encode_min ..
  case cumsum => exact Op.Wire.decode_encode_cumsum ..
  case safeLog => exact Op.Wire.decode_encode_safeLog ..
  case clamp => exact Op.Wire.decode_encode_clamp ..
  case scale => exact Op.Wire.decode_encode_scale ..
  case elu => exact Op.Wire.decode_encode_elu ..
  case leakyReLU => exact Op.Wire.decode_encode_leakyReLU ..
  case softmax => exact Op.Wire.decode_encode_softmax ..
  case logSoftmax => exact Op.Wire.decode_encode_logSoftmax ..
  case linear => exact Op.Wire.decode_encode_linear ..
  case attentionFromScores => exact Op.Wire.decode_encode_attentionFromScores ..
  case sort => exact Op.Wire.decode_encode_sort ..
  case topk => exact Op.Wire.decode_encode_topk ..
  case gather => exact Op.Wire.decode_encode_gather ..
  case randUniform => exact Op.Wire.decode_encode_randUniform ..
  case randn => exact Op.Wire.decode_encode_randn ..
  case sign => exact Op.Wire.decode_encode_sign ..
  case floor => exact Op.Wire.decode_encode_floor ..
  case ceil => exact Op.Wire.decode_encode_ceil ..
  case round => exact Op.Wire.decode_encode_round ..
  case trunc => exact Op.Wire.decode_encode_trunc ..
  case frac => exact Op.Wire.decode_encode_frac ..
  case rsqrt => exact Op.Wire.decode_encode_rsqrt ..
  case exp2 => exact Op.Wire.decode_encode_exp2 ..
  case expm1 => exact Op.Wire.decode_encode_expm1 ..
  case log2 => exact Op.Wire.decode_encode_log2 ..
  case log10 => exact Op.Wire.decode_encode_log10 ..
  case log1p => exact Op.Wire.decode_encode_log1p ..
  case tan => exact Op.Wire.decode_encode_tan ..
  case asinh => exact Op.Wire.decode_encode_asinh ..
  case acosh => exact Op.Wire.decode_encode_acosh ..
  case atanh => exact Op.Wire.decode_encode_atanh ..
  case atan => exact Op.Wire.decode_encode_atan ..
  case asin => exact Op.Wire.decode_encode_asin ..
  case acos => exact Op.Wire.decode_encode_acos ..
  case erf => exact Op.Wire.decode_encode_erf ..
  case atan2 => exact Op.Wire.decode_encode_atan2 ..
  case fmod => exact Op.Wire.decode_encode_fmod ..
  case remainder => exact Op.Wire.decode_encode_remainder ..
  case hypot => exact Op.Wire.decode_encode_hypot ..
  case copysign => exact Op.Wire.decode_encode_copysign ..
  case fmax => exact Op.Wire.decode_encode_fmax ..
  case fmin => exact Op.Wire.decode_encode_fmin ..
  case nanToNum => exact Op.Wire.decode_encode_nanToNum ..
  case select => exact Op.Wire.decode_encode_select ..
  case lerp => exact Op.Wire.decode_encode_lerp ..
  case addcmul => exact Op.Wire.decode_encode_addcmul ..
  case addcdiv => exact Op.Wire.decode_encode_addcdiv ..
  case gelu => exact Op.Wire.decode_encode_gelu ..
  case sinh => exact Op.Wire.decode_encode_sinh ..
  case cosh => exact Op.Wire.decode_encode_cosh ..
  case safeInv => exact Op.Wire.decode_encode_safeInv ..
  case smoothAbs => exact Op.Wire.decode_encode_smoothAbs ..
  case safeDiv => exact Op.Wire.decode_encode_safeDiv ..
  case pow => exact Op.Wire.decode_encode_pow ..
  case powValue => exact Op.Wire.decode_encode_powValue ..
  case cumprod => exact Op.Wire.decode_encode_cumprod ..
  case cummax => exact Op.Wire.decode_encode_cummax ..
  case cummin => exact Op.Wire.decode_encode_cummin ..
  case mseLoss => exact Op.Wire.decode_encode_mseLoss ..
  case maeLoss => exact Op.Wire.decode_encode_maeLoss ..
  case huberLoss => exact Op.Wire.decode_encode_huberLoss ..
  case hingeLoss => exact Op.Wire.decode_encode_hingeLoss ..
  case poissonLoss => exact Op.Wire.decode_encode_poissonLoss ..
  case logCoshLoss => exact Op.Wire.decode_encode_logCoshLoss ..
  case binaryCrossEntropy => exact Op.Wire.decode_encode_binaryCrossEntropy ..
  case cosineLoss => exact Op.Wire.decode_encode_cosineLoss ..
  case crossEntropy => exact Op.Wire.decode_encode_crossEntropy ..
  case crossEntropyProbabilities => exact Op.Wire.decode_encode_crossEntropyProbabilities ..
  case distill => exact Op.Wire.decode_encode_distill ..
  case reshape => exact Op.Wire.decode_encode_reshape ..
  case permute => exact Op.Wire.decode_encode_permute ..
  case broadcast => exact Op.Wire.decode_encode_broadcast ..
  case fullLike => exact Op.Wire.decode_encode_fullLike ..
  case dropout => exact Op.Wire.decode_encode_dropout ..
  case embedding => exact Op.Wire.decode_encode_embedding ..
  case matmul => exact Op.Wire.decode_encode_matmul ..
  case scaledMatmul => exact Op.Wire.decode_encode_scaledMatmul ..
  case conv => exact Op.Wire.decode_encode_conv ..
  case pool => exact Op.Wire.decode_encode_pool ..
  case smoothMaxPool => exact Op.Wire.decode_encode_smoothMaxPool ..
  case layerNorm => exact Op.Wire.decode_encode_layerNorm ..
  case batchNorm => exact Op.Wire.decode_encode_batchNorm ..
  case batchNormRunning => exact Op.Wire.decode_encode_batchNormRunning ..
  case multiheadAttention => exact Op.Wire.decode_encode_multiheadAttention ..
  case convTranspose => exact Op.Wire.decode_encode_convTranspose ..
  case eq => exact Op.Wire.decode_encode_eq ..
  case ne => exact Op.Wire.decode_encode_ne ..
  case lt => exact Op.Wire.decode_encode_lt ..
  case le => exact Op.Wire.decode_encode_le ..
  case gt => exact Op.Wire.decode_encode_gt ..
  case ge => exact Op.Wire.decode_encode_ge ..
  case isNaN => exact Op.Wire.decode_encode_isNaN ..
  case isInf => exact Op.Wire.decode_encode_isInf ..
  case isFinite => exact Op.Wire.decode_encode_isFinite ..
  case logicalNot => exact Op.Wire.decode_encode_logicalNot ..
  case logicalAnd => exact Op.Wire.decode_encode_logicalAnd ..
  case logicalOr => exact Op.Wire.decode_encode_logicalOr ..
  case logicalXor => exact Op.Wire.decode_encode_logicalXor ..
  case bitwiseAnd => exact Op.Wire.decode_encode_bitwiseAnd ..
  case bitwiseOr => exact Op.Wire.decode_encode_bitwiseOr ..
  case bitwiseXor => exact Op.Wire.decode_encode_bitwiseXor ..
  case bitwiseNot => exact Op.Wire.decode_encode_bitwiseNot ..
  case bitwiseLeftShift => exact Op.Wire.decode_encode_bitwiseLeftShift ..
  case shiftRightArithmetic => exact Op.Wire.decode_encode_shiftRightArithmetic ..
  case shiftRightLogical => exact Op.Wire.decode_encode_shiftRightLogical ..
  case populationCount => exact Op.Wire.decode_encode_populationCount ..
  case countLeadingZeros => exact Op.Wire.decode_encode_countLeadingZeros ..
  case mulhi => exact Op.Wire.decode_encode_mulhi ..
  case bitcast => exact Op.Wire.decode_encode_bitcast ..
  case cbrt => exact Op.Wire.decode_encode_cbrt ..
  case special => exact Op.Wire.decode_encode_special ..
  case nextafter => exact Op.Wire.decode_encode_nextafter ..
  case frexp => exact Op.Wire.decode_encode_frexp ..
  case fma => exact Op.Wire.decode_encode_fma ..
  case complex => exact Op.Wire.decode_encode_complex ..
  case complexPow => exact Op.Wire.decode_encode_complexPow ..
  case argmax => exact Op.Wire.decode_encode_argmax ..
  case argmin => exact Op.Wire.decode_encode_argmin ..
  case flip => exact Op.Wire.decode_encode_flip ..
  case indices => exact Op.Wire.decode_encode_indices ..
  case cast => exact Op.Wire.decode_encode_cast ..
  case encodeIndex => exact Op.Wire.decode_encode_encodeIndex ..
  case decodeIndex => exact Op.Wire.decode_encode_decodeIndex ..
  case attention => exact Op.Wire.decode_encode_attention ..
  case reduce => exact Op.Wire.decode_encode_reduce ..
  case mean => exact Op.Wire.decode_encode_mean ..
  case contract => exact Op.Wire.decode_encode_contract ..
  case maskedSoftmax => exact Op.Wire.decode_encode_maskedSoftmax ..
  case slice => exact Op.Wire.decode_encode_slice ..
  case pad => exact Op.Wire.decode_encode_pad ..
  case indexAt => exact Op.Wire.decode_encode_indexAt ..
  case cat => exact Op.Wire.decode_encode_cat ..
  case scatter => exact Op.Wire.decode_encode_scatter ..
  case topKMask => exact Op.Wire.decode_encode_topKMask ..
  case bernoulliMask => exact Op.Wire.decode_encode_bernoulliMask ..
  case rfft => exact Op.Wire.decode_encode_rfft ..
  case irfft => exact Op.Wire.decode_encode_irfft ..
  case selectiveScan => exact Op.Wire.decode_encode_selectiveScan ..
  case custom => exact Op.Wire.decode_encode_custom ..

/-- Equality of erased operators, decided by their wire records. -/
instance Op.instDecidableEqErased : DecidableEq (Σ ins outs, Op ins outs) := fun a b =>
  if h : Op.encode a = Op.encode b then
    isTrue (Option.some.inj ((Op.decode_encode a).symm.trans (h ▸ Op.decode_encode b)))
  else isFalse fun e => h (e ▸ rfl)

end TorchLean
