/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table
public import Mathlib.Data.Fintype.Card
public import Mathlib.Logic.Equiv.Fintype

/-!
# The wire codec of the registered operators

An erased operator is `Σ ins outs, Op ins outs`. `Op.encode` writes it as a wire record
(`Op.Wire`): the format version `Op.Wire.formatVersion`, the constructor's name as its tag
(`leakyRelu` for `leakyReLU`), and one field (`Op.Field`) for each implicit shape and attribute,
in binder order. The fields are
shapes, naturals, Booleans, lists of naturals for axis lists and functions on `Fin r`, and pairs for
batch dimensions. A mask is an optional array of Booleans, a custom node's name is a string, and a
generalized operator's element type is an `Op.Element`. Every attribute has its field in every
record, at every value: a square root writes its policy code, a generalized operator its element
(the scalar too), `huberLoss` its mean flag and `multiheadAttention` its leading shape, `[]` for
one sequence.

Records are version 3 (`Op.Wire.formatVersion`), and `Op.decode` accepts version 3 only. Proof
attributes are not sent, and `Op.decode` decides each one. It checks what the typed operator
needs: an axis below the rank, a mask of the right length, an equivalence that is a bijection and a
key below `2 ^ 64`. On an arithmetic operator or a comparison the element must be numeric. It
rejects any unknown tag and any field list other than its tag's. It does not run `Op.check`: a
decoded pooling with an ill-formed window is an operator value, which a recorder rejects before
reading its input. The ports are recomputed from the attributes, so two operators at different
shapes have different records. `Op.decode_encode` shows that decoding inverts encoding, for every
constructor, and equality of erased operators is decided through it (`Op.instDecidableEqErased`).

The record holds attributes only. A frozen scalar (an epsilon, a slope) is a graph input and a
custom node's program is graph payload, so neither is part of the operator value. A record names
an operator, not its arithmetic. What the operator computes at a scalar type (the `InverseTrig`
instance that computes `atan`, `Context.sign` at a NaN, the sign of a zero `remainder`) is that of
the TorchLean revision that evaluates the record. An artifact whose results must be reproduced
names that revision beside its records.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/-- Every dimension of a shape is positive, decided. -/
instance Shape.decidablePositive : (s : Shape) → Decidable (Shape.Positive s)
  | [] => isTrue trivial
  | n :: s =>
    haveI := Shape.decidablePositive s
    inferInstanceAs (Decidable (n > 0 ∧ Shape.Positive s))

/-- The validity of a gather's dimension numbers, decided from its three conditions. -/
instance Shape.decidableGatherValid (s t : Shape) (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) :
    Decidable (Shape.GatherValid s t dims batchDims) :=
  decidable_of_iff ((dims ++ batchDims.map Prod.fst).Nodup ∧ (batchDims.map Prod.snd).Nodup ∧
      ∀ p ∈ batchDims, s.extent p.1 = t.extent p.2)
    ⟨fun ⟨a, b, c⟩ => ⟨a, b, c⟩, fun h => ⟨h.nodup, h.nodup_snd, h.extent_eq⟩⟩

/-- The validity of a contraction's axis pairs, decided from its three conditions. -/
instance Shape.decidableContractValid (s t : Shape)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) :
    Decidable (Shape.ContractValid s t dims batchDims) :=
  decidable_of_iff (((batchDims ++ dims).map Prod.fst).Nodup ∧
      ((batchDims ++ dims).map Prod.snd).Nodup ∧
      ∀ p ∈ batchDims ++ dims, s.extent p.1 = t.extent p.2)
    ⟨fun ⟨a, b, c⟩ => ⟨a, b, c⟩, fun h => ⟨h.nodup_left, h.nodup_right, h.extent_eq⟩⟩

/-- One field of an operator's wire record. -/
inductive Op.Field where
  /-- A shape. -/
  | shape (value : Shape)
  /-- A list of shapes. -/
  | shapes (value : List Shape)
  /-- A natural number. -/
  | nat (value : Nat)
  /-- A list of naturals: an axis list, the images of an equivalence, a function on `Fin r`. -/
  | nats (value : List Nat)
  /-- A list of pairs of naturals: batch dimensions. -/
  | pairs (value : List (Nat × Nat))
  /-- A Boolean. -/
  | bool (value : Bool)
  /-- An optional mask, row-major. -/
  | mask (value : Option (Array Bool))
  /-- A name. -/
  | name (value : String)
  /-- The element type of a generalized operator. -/
  | element (value : Op.Element)
  deriving DecidableEq, Repr

/-- The wire record of an erased operator. -/
structure Op.Wire where
  /-- The format version. -/
  version : Nat
  /-- The constructor's name (`leakyRelu` for `leakyReLU`). -/
  tag : String
  /-- The implicit shapes and attributes, in binder order. -/
  fields : List Op.Field
  deriving DecidableEq, Repr

namespace Op.Wire

/-- The format version of the records `Op.encode` writes and the one version `Op.decode` reads,
raised by any change that makes a stored record name another operator or other attributes, or fail
to decode. -/
def formatVersion : Nat := 3

/-- `i` as an index below `n`, if it is one. -/
def fin? (n i : Nat) : Option (Fin n) := if h : i < n then some ⟨i, h⟩ else none

/-- `fin? n` reads back every index below `n` from its value. -/
@[simp] theorem fin?_val {n : Nat} (i : Fin n) : fin? n i.val = some i := by simp [fin?]

/-- Every entry of `l` as an index below `n`, if each is one. -/
def finList? (n : Nat) (l : List Nat) : Option (List (Fin n)) := l.mapM (fin? n)

/-- `finList? n` reads back every list of indices below `n` from their values. -/
@[simp] theorem finList?_map {n : Nat} (l : List (Fin n)) :
    finList? n (l.map (·.val)) = some l := by
  induction l with
  | nil => rfl
  | cons i l ih => simp_all [finList?, List.mapM_cons]

/-- Every pair of `l` as a pair of indices below `m` and `n`, if each is one. -/
def finPairs? (m n : Nat) (l : List (Nat × Nat)) : Option (List (Fin m × Fin n)) :=
  l.mapM fun p => do pure (← fin? m p.1, ← fin? n p.2)

/-- `finPairs? m n` reads back every list of index pairs from their values. -/
@[simp] theorem finPairs?_map {m n : Nat} (l : List (Fin m × Fin n)) :
    finPairs? m n (l.map fun p => (p.1.val, p.2.val)) = some l := by
  induction l with
  | nil => rfl
  | cons p l ih => simp_all [finPairs?, List.mapM_cons]

/-- A proof of `p`, if it holds. -/
def prop? (p : Prop) [Decidable p] : Option (PLift p) := if h : p then some ⟨h⟩ else none

/-- `prop? p` is the proof of `p` when `p` holds. -/
@[simp] theorem prop?_of {p : Prop} [Decidable p] (h : p) : prop? p = some ⟨h⟩ := by
  simp [prop?, h]

/-- The function on `Fin n` reading `l`, if `l` has length `n`. -/
def fun? (n : Nat) (l : List Nat) : Option (Fin n → Nat) :=
  if h : l.length = n then some fun i => l[i.val]'(h ▸ i.isLt) else none

/-- `fun? n` reads back every function on `Fin n` from the list of its values. -/
@[simp] theorem fun?_ofFn {n : Nat} (f : Fin n → Nat) : fun? n (List.ofFn f) = some f := by
  simp [fun?]

/-- The function `Fin m → Fin n` reading `l`, if `l` has length `m` and its entries are below `n`.
It need not be injective or surjective (`equiv?` reads a bijection). -/
def finFun? (m n : Nat) (l : List Nat) : Option (Fin m → Fin n) := do
  let f ← fun? m l
  if h : ∀ i, f i < n then some fun i => ⟨f i, h i⟩ else none

/-- `finFun? m n` reads back every function `Fin m → Fin n` from the list of its values. -/
@[simp] theorem finFun?_ofFn {m n : Nat} (f : Fin m → Fin n) :
    finFun? m n (List.ofFn fun i => (f i).val) = some f := by
  simp [finFun?]

/-- `k` as a 64-bit word, if it is below `2 ^ 64`. -/
def uint64? (k : Nat) : Option UInt64 := if k < UInt64.size then some (UInt64.ofNat k) else none

/-- `uint64?` reads back every 64-bit word from its value. -/
@[simp] theorem uint64?_toNat (k : UInt64) : uint64? k.toNat = some k := by
  simp [uint64?, UInt64.toNat_lt_size]

/-- The equivalence `Fin m ≃ Fin n` sending `i` to the `i`-th entry of `l`, if `l` lists a
bijection. -/
def equiv? (m n : Nat) (l : List Nat) : Option (Fin m ≃ Fin n) :=
  if hlen : l.length = m then
    if hlt : ∀ i : Fin m, l[i.val]'(hlen ▸ i.isLt) < n then
      let f : Fin m → Fin n := fun i => ⟨l[i.val]'(hlen ▸ i.isLt), hlt i⟩
      if hb : Function.Bijective f then
        some ⟨f, Fintype.bijInv hb, Fintype.leftInverse_bijInv hb, Fintype.rightInverse_bijInv hb⟩
      else none
    else none
  else none

/-- `equiv? m n` reads back every equivalence `Fin m ≃ Fin n` from the list of its values. -/
@[simp] theorem equiv?_ofFn {m n : Nat} (e : Fin m ≃ Fin n) :
    equiv? m n (List.ofFn fun i => (e i).val) = some e := by
  unfold equiv?
  simp only [List.length_ofFn, dite_true, List.getElem_ofFn, Fin.is_lt, implies_true]
  split
  · congr 1
    exact Equiv.ext fun i => by simp
  · rename_i h
    exact absurd (by simp) h

/-- A pooling reduction as a natural. -/
def poolOpCode : PoolOp → Nat
  | .sum => 0
  | .max => 1
  | .min => 2
  | .avg false => 3
  | .avg true => 4

/-- The pooling reduction of a code. -/
def poolOp? : Nat → Option PoolOp
  | 0 => some .sum
  | 1 => some .max
  | 2 => some .min
  | 3 => some (.avg false)
  | 4 => some (.avg true)
  | _ => none

/-- `poolOp?` reads back every `PoolOp` from its code. -/
@[simp] theorem poolOp?_code (op : PoolOp) : poolOp? (poolOpCode op) = some op := by
  cases op with
  | avg countIncludePad => cases countIncludePad <;> rfl
  | _ => rfl

/-- A reduction as a natural. -/
def reduceOpCode : Tensor.ReduceOp → Nat
  | .sum => 0
  | .prod => 1
  | .max => 2
  | .min => 3

/-- The reduction of a code. -/
def reduceOp? : Nat → Option Tensor.ReduceOp
  | 0 => some .sum
  | 1 => some .prod
  | 2 => some .max
  | 3 => some .min
  | _ => none

/-- `reduceOp?` reads back every `Tensor.ReduceOp` from its code. -/
@[simp] theorem reduceOp?_code (op : Tensor.ReduceOp) : reduceOp? (reduceOpCode op) = some op := by
  cases op <;> rfl

/-- A scatter rule as a natural. -/
def scatterCode : Tensor.ScatterReduce → Nat
  | .replace => 0
  | .sum => 1
  | .prod => 2
  | .max => 3
  | .min => 4

/-- The scatter rule of a code. -/
def scatter? : Nat → Option Tensor.ScatterReduce
  | 0 => some .replace
  | 1 => some .sum
  | 2 => some .prod
  | 3 => some .max
  | 4 => some .min
  | _ => none

/-- `scatter?` reads back every `Tensor.ScatterReduce` from its code. -/
@[simp] theorem scatter?_code (rule : Tensor.ScatterReduce) :
    scatter? (scatterCode rule) = some rule := by
  cases rule <;> rfl

/-- A division's rounding as a natural: `0` for true division, `1` toward zero, `2` flooring. -/
def divRoundingCode : DivRounding → Nat
  | .none => 0
  | .trunc => 1
  | .floor => 2

/-- The division rounding of a code. -/
def divRounding? : Nat → Option DivRounding
  | 0 => some .none
  | 1 => some .trunc
  | 2 => some .floor
  | _ => none

/-- `divRounding?` reads back every `DivRounding` from its code. -/
@[simp] theorem divRounding?_code (rounding : DivRounding) :
    divRounding? (divRoundingCode rounding) = some rounding := by
  cases rounding <;> rfl

/-- A GELU approximation as a natural: `0` for the exact formula, `1` for the tanh formula. -/
def geluApproximationCode : GeluApproximation → Nat
  | .none => 0
  | .tanh => 1

/-- The GELU approximation of a code. -/
def geluApproximation? : Nat → Option GeluApproximation
  | 0 => some .none
  | 1 => some .tanh
  | _ => none

/-- `geluApproximation?` reads back every `GeluApproximation` from its code. -/
@[simp] theorem geluApproximation?_code (approximate : GeluApproximation) :
    geluApproximation? (geluApproximationCode approximate) = some approximate := by
  cases approximate <;> rfl

/-- A square root's policy as a natural: `0` for IEEE's root (`.raw`), `1` for the root of the
argument clamped below at zero (`.clampNegative`). -/
def sqrtPolicyCode : Tensor.SqrtPolicy → Nat
  | .raw => 0
  | .clampNegative => 1

/-- The square-root policy of a code. -/
def sqrtPolicy? : Nat → Option Tensor.SqrtPolicy
  | 0 => some .raw
  | 1 => some .clampNegative
  | _ => none

/-- `sqrtPolicy?` reads back every `Tensor.SqrtPolicy` from its code. -/
@[simp] theorem sqrtPolicy?_code (policy : Tensor.SqrtPolicy) :
    sqrtPolicy? (sqrtPolicyCode policy) = some policy := by
  cases policy <;> rfl

/-- A cast source as a natural: `0` for Booleans, `bound + 1` for indices below `bound`. -/
def castSourceCode : Op.CastSource → Nat
  | .bool => 0
  | .index bound => bound + 1

/-- The cast source of a code. Every natural is the code of one. -/
def castSource? : Nat → Op.CastSource
  | 0 => .bool
  | bound + 1 => .index bound

/-- `castSource?` reads back every `Op.CastSource` from its code. -/
@[simp] theorem castSource?_code (source : Op.CastSource) :
    castSource? (castSourceCode source) = source := by
  cases source <;> rfl

/-- The code of an index below `bound` is the code of an index below `0` plus `bound`, so that the
op table lists one index code. -/
theorem castSourceCode_index (bound : Nat) :
    castSourceCode (.index bound) = castSourceCode (.index 0) + bound := by
  show bound + 1 = 0 + 1 + bound
  omega

/-- A side of `Op.bitcast` as a natural: `0` for the scalar, `1` for an unsigned and `2` for a
signed integer. -/
def wordCode : Op.Word → Nat
  | .scalar => 0
  | .integer false => 1
  | .integer true => 2

/-- The side of `Op.bitcast` of a code. -/
def word? : Nat → Option Op.Word
  | 0 => some .scalar
  | 1 => some (.integer false)
  | 2 => some (.integer true)
  | _ => none

/-- `word?` reads back every `Op.Word` from its code. -/
@[simp] theorem word?_code (word : Op.Word) : word? (wordCode word) = some word := by
  obtain _ | ⟨_ | _⟩ := word <;> rfl

/-- A softmax schedule as a natural: `0` for the two-pass schedule, `size + 1` for the online
schedule over blocks of `size` keys. -/
def scheduleCode : Softmax.Schedule → Nat
  | .twoPass => 0
  | .online blocks => blocks.size + 1

/-- The softmax schedule of a code. Every natural is the code of one. -/
def schedule? : Nat → Softmax.Schedule
  | 0 => .twoPass
  | size + 1 => .online ⟨size⟩

/-- `schedule?` reads back every `Softmax.Schedule` from its code. -/
@[simp] theorem schedule?_code (schedule : Softmax.Schedule) :
    schedule? (scheduleCode schedule) = schedule := by
  cases schedule <;> rfl

/-- The entries of a mask, row-major. -/
def maskEntries {a b : Nat} (mask : Tensor Bool [a, b]) : Array Bool :=
  Array.ofFn fun i => mask (Shape.Coord.unlinearize i)

/-- The mask with the given entries, if they are `a * b` of them. `none` stays `none`. -/
def mask? (a b : Nat) : Option (Array Bool) → Option (Option (Tensor Bool [a, b]))
  | none => some none
  | some l =>
    if h : l.size = Shape.size [a, b] then
      some (some (Rep.ofFn fun c => l[(Shape.Coord.linearize c).val]'(by
        rw [h]; exact (Shape.Coord.linearize c).isLt)))
    else none

/-- `mask? a b` reads back every optional mask from its entries. -/
@[simp] theorem mask?_map {a b : Nat} (m : Option (Tensor Bool [a, b])) :
    mask? a b (m.map maskEntries) = some m := by
  cases m with
  | none => rfl
  | some mask =>
    simp only [Option.map_some, mask?, maskEntries, Array.size_ofFn, dite_true]
    congr 2
    apply Tensor.Rep.ext
    intro c
    simp only [Rep.get_ofFn, Array.getElem_ofFn]
    exact congrArg (Rep.get mask) (Shape.Coord.unlinearize_linearize c)

end Op.Wire

open Op.Wire in
/-- The wire record of an erased operator. -/
def Op.encode : (Σ ins outs, Op ins outs) → Op.Wire
  | ⟨_, _, @Op.relu s⟩ => ⟨formatVersion, "relu", [.shape s]⟩
  | ⟨_, _, @Op.sigmoid s⟩ => ⟨formatVersion, "sigmoid", [.shape s]⟩
  | ⟨_, _, @Op.tanh s⟩ => ⟨formatVersion, "tanh", [.shape s]⟩
  | ⟨_, _, @Op.exp s⟩ => ⟨formatVersion, "exp", [.shape s]⟩
  | ⟨_, _, @Op.abs s⟩ => ⟨formatVersion, "abs", [.shape s]⟩
  | ⟨_, _, @Op.sqrt s policy⟩ => ⟨formatVersion, "sqrt", [.shape s, .nat (sqrtPolicyCode policy)]⟩
  | ⟨_, _, @Op.sin s⟩ => ⟨formatVersion, "sin", [.shape s]⟩
  | ⟨_, _, @Op.cos s⟩ => ⟨formatVersion, "cos", [.shape s]⟩
  | ⟨_, _, @Op.neg s e⟩ => ⟨formatVersion, "neg", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.add s e⟩ => ⟨formatVersion, "add", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.mul s e⟩ => ⟨formatVersion, "mul", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.max s e⟩ => ⟨formatVersion, "max", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.log s⟩ => ⟨formatVersion, "log", [.shape s]⟩
  | ⟨_, _, @Op.detach s⟩ => ⟨formatVersion, "detach", [.shape s]⟩
  | ⟨_, _, @Op.softplus s⟩ => ⟨formatVersion, "softplus", [.shape s]⟩
  | ⟨_, _, @Op.silu s⟩ => ⟨formatVersion, "silu", [.shape s]⟩
  | ⟨_, _, @Op.square s⟩ => ⟨formatVersion, "square", [.shape s]⟩
  | ⟨_, _, @Op.reciprocal s⟩ => ⟨formatVersion, "reciprocal", [.shape s]⟩
  | ⟨_, _, @Op.sub s e⟩ => ⟨formatVersion, "sub", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.div s rounding e⟩ =>
      ⟨formatVersion, "div", [.shape s, .nat (divRoundingCode rounding), .element e.toElement]⟩
  | ⟨_, _, @Op.min s e⟩ => ⟨formatVersion, "min", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.cumsum s dim⟩ => ⟨formatVersion, "cumsum", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.safeLog s⟩ => ⟨formatVersion, "safeLog", [.shape s]⟩
  | ⟨_, _, @Op.clamp s⟩ => ⟨formatVersion, "clamp", [.shape s]⟩
  | ⟨_, _, @Op.scale s⟩ => ⟨formatVersion, "scale", [.shape s]⟩
  | ⟨_, _, @Op.elu s⟩ => ⟨formatVersion, "elu", [.shape s]⟩
  | ⟨_, _, @Op.leakyReLU s⟩ => ⟨formatVersion, "leakyRelu", [.shape s]⟩
  | ⟨_, _, @Op.softmax s dim⟩ => ⟨formatVersion, "softmax", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.logSoftmax s dim⟩ => ⟨formatVersion, "logSoftmax", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.linear lead inDim outDim⟩ =>
      ⟨formatVersion, "linear", [.shape lead, .nat inDim, .nat outDim]⟩
  | ⟨_, _, @Op.attentionFromScores batch rows keys width mask⟩ =>
      ⟨formatVersion, "attentionFromScores", [.shape batch, .nat rows, .nat keys, .nat width,
        .mask (mask.map maskEntries)]⟩
  | ⟨_, _, @Op.sort s dim descending⟩ =>
      ⟨formatVersion, "sort", [.shape s, .nat dim.val, .bool descending]⟩
  | ⟨_, _, @Op.topk s dim k largest _⟩ =>
      ⟨formatVersion, "topk", [.shape s, .nat dim.val, .nat k, .bool largest]⟩
  | ⟨_, _, @Op.gather s t dims batchDims _ e⟩ =>
      ⟨formatVersion, "gather", [.shape s, .shape t, .nats (dims.map (·.val)),
        .pairs (batchDims.map fun p => (p.1.val, p.2.val)), .element e]⟩
  | ⟨_, _, @Op.randUniform s key⟩ => ⟨formatVersion, "randUniform", [.shape s, .nat key.toNat]⟩
  | ⟨_, _, @Op.randn s key⟩ => ⟨formatVersion, "randn", [.shape s, .nat key.toNat]⟩
  | ⟨_, _, @Op.sign s⟩ => ⟨formatVersion, "sign", [.shape s]⟩
  | ⟨_, _, @Op.floor s⟩ => ⟨formatVersion, "floor", [.shape s]⟩
  | ⟨_, _, @Op.ceil s⟩ => ⟨formatVersion, "ceil", [.shape s]⟩
  | ⟨_, _, @Op.round s⟩ => ⟨formatVersion, "round", [.shape s]⟩
  | ⟨_, _, @Op.trunc s⟩ => ⟨formatVersion, "trunc", [.shape s]⟩
  | ⟨_, _, @Op.frac s⟩ => ⟨formatVersion, "frac", [.shape s]⟩
  | ⟨_, _, @Op.rsqrt s⟩ => ⟨formatVersion, "rsqrt", [.shape s]⟩
  | ⟨_, _, @Op.exp2 s⟩ => ⟨formatVersion, "exp2", [.shape s]⟩
  | ⟨_, _, @Op.expm1 s⟩ => ⟨formatVersion, "expm1", [.shape s]⟩
  | ⟨_, _, @Op.log2 s⟩ => ⟨formatVersion, "log2", [.shape s]⟩
  | ⟨_, _, @Op.log10 s⟩ => ⟨formatVersion, "log10", [.shape s]⟩
  | ⟨_, _, @Op.log1p s⟩ => ⟨formatVersion, "log1p", [.shape s]⟩
  | ⟨_, _, @Op.tan s⟩ => ⟨formatVersion, "tan", [.shape s]⟩
  | ⟨_, _, @Op.asinh s⟩ => ⟨formatVersion, "asinh", [.shape s]⟩
  | ⟨_, _, @Op.acosh s⟩ => ⟨formatVersion, "acosh", [.shape s]⟩
  | ⟨_, _, @Op.atanh s⟩ => ⟨formatVersion, "atanh", [.shape s]⟩
  | ⟨_, _, @Op.atan s⟩ => ⟨formatVersion, "atan", [.shape s]⟩
  | ⟨_, _, @Op.asin s⟩ => ⟨formatVersion, "asin", [.shape s]⟩
  | ⟨_, _, @Op.acos s⟩ => ⟨formatVersion, "acos", [.shape s]⟩
  | ⟨_, _, @Op.erf s⟩ => ⟨formatVersion, "erf", [.shape s]⟩
  | ⟨_, _, @Op.atan2 s⟩ => ⟨formatVersion, "atan2", [.shape s]⟩
  | ⟨_, _, @Op.fmod s⟩ => ⟨formatVersion, "fmod", [.shape s]⟩
  | ⟨_, _, @Op.remainder s e⟩ => ⟨formatVersion, "remainder", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.hypot s⟩ => ⟨formatVersion, "hypot", [.shape s]⟩
  | ⟨_, _, @Op.copysign s⟩ => ⟨formatVersion, "copysign", [.shape s]⟩
  | ⟨_, _, @Op.fmax s⟩ => ⟨formatVersion, "fmax", [.shape s]⟩
  | ⟨_, _, @Op.fmin s⟩ => ⟨formatVersion, "fmin", [.shape s]⟩
  | ⟨_, _, @Op.nanToNum s⟩ => ⟨formatVersion, "nanToNum", [.shape s]⟩
  | ⟨_, _, @Op.select s e⟩ => ⟨formatVersion, "select", [.shape s, .element e]⟩
  | ⟨_, _, @Op.lerp s⟩ => ⟨formatVersion, "lerp", [.shape s]⟩
  | ⟨_, _, @Op.addcmul s⟩ => ⟨formatVersion, "addcmul", [.shape s]⟩
  | ⟨_, _, @Op.addcdiv s⟩ => ⟨formatVersion, "addcdiv", [.shape s]⟩
  | ⟨_, _, @Op.gelu s approximate⟩ =>
      ⟨formatVersion, "gelu", [.shape s, .nat (geluApproximationCode approximate)]⟩
  | ⟨_, _, @Op.sinh s⟩ => ⟨formatVersion, "sinh", [.shape s]⟩
  | ⟨_, _, @Op.cosh s⟩ => ⟨formatVersion, "cosh", [.shape s]⟩
  | ⟨_, _, @Op.safeInv s⟩ => ⟨formatVersion, "safeInv", [.shape s]⟩
  | ⟨_, _, @Op.smoothAbs s⟩ => ⟨formatVersion, "smoothAbs", [.shape s]⟩
  | ⟨_, _, @Op.safeDiv s⟩ => ⟨formatVersion, "safeDiv", [.shape s]⟩
  | ⟨_, _, @Op.pow s e⟩ => ⟨formatVersion, "pow", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.cumprod s dim⟩ => ⟨formatVersion, "cumprod", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.cummax s dim⟩ => ⟨formatVersion, "cummax", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.cummin s dim⟩ => ⟨formatVersion, "cummin", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.mseLoss s⟩ => ⟨formatVersion, "mseLoss", [.shape s]⟩
  | ⟨_, _, @Op.maeLoss s⟩ => ⟨formatVersion, "maeLoss", [.shape s]⟩
  | ⟨_, _, @Op.huberLoss s isMean⟩ =>
      ⟨formatVersion, "huberLoss", [.shape s, .bool isMean]⟩
  | ⟨_, _, @Op.hingeLoss s⟩ => ⟨formatVersion, "hingeLoss", [.shape s]⟩
  | ⟨_, _, @Op.poissonLoss s⟩ => ⟨formatVersion, "poissonLoss", [.shape s]⟩
  | ⟨_, _, @Op.logCoshLoss s⟩ => ⟨formatVersion, "logCoshLoss", [.shape s]⟩
  | ⟨_, _, @Op.binaryCrossEntropy s⟩ => ⟨formatVersion, "binaryCrossEntropy", [.shape s]⟩
  | ⟨_, _, @Op.cosineLoss s⟩ => ⟨formatVersion, "cosineLoss", [.shape s]⟩
  | ⟨_, _, @Op.crossEntropy s dim isMean⟩ =>
      ⟨formatVersion, "crossEntropy", [.shape s, .nat dim.val, .bool isMean]⟩
  | ⟨_, _, @Op.crossEntropyProbabilities s dim⟩ =>
      ⟨formatVersion, "crossEntropyProbabilities", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.distill s dim⟩ => ⟨formatVersion, "distill", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.reshape s t _ e⟩ =>
      ⟨formatVersion, "reshape", [.shape s, .shape t, .element e]⟩
  | ⟨_, _, @Op.permute s perm out _ e⟩ =>
      ⟨formatVersion, "permute", [.shape s, .nats (List.ofFn fun i => (perm i).val), .shape out,
        .element e]⟩
  | ⟨_, _, @Op.broadcast s t dims _ e⟩ =>
      ⟨formatVersion, "broadcast", [.shape s, .shape t, .nats (List.ofFn fun i => (dims i).val),
        .element e]⟩
  | ⟨_, _, @Op.fullLike s⟩ => ⟨formatVersion, "fullLike", [.shape s]⟩
  | ⟨_, _, @Op.dropout s⟩ => ⟨formatVersion, "dropout", [.shape s]⟩
  | ⟨_, _, @Op.embedding num dim t⟩ => ⟨formatVersion, "embedding", [.nat num, .nat dim, .shape t]⟩
  | ⟨_, _, @Op.matmul batchA batchB batch m n p lv rv _ _ _ _⟩ =>
      ⟨formatVersion, "matmul", [.shape batchA, .shape batchB, .shape batch, .nat m, .nat n,
        .nat p, .bool lv, .bool rv]⟩
  | ⟨_, _, @Op.scaledMatmul batchA batchB batch m n p lv rv _ _ _ _⟩ =>
      ⟨formatVersion, "scaledMatmul", [.shape batchA, .shape batchB, .shape batch, .nat m, .nat n,
        .nat p, .bool lv, .bool rv]⟩
  | ⟨_, _, @Op.conv s k r dims stride padLow padHigh dilation groups lead _ _ _ _ _⟩ =>
      ⟨formatVersion, "conv", [.shape s, .shape k, .nat r,
        .nats (List.ofFn fun i => (dims.input i).val),
        .nats (List.ofFn fun i => (dims.kernel i).val), .nats (List.ofFn stride),
        .nats (List.ofFn padLow), .nats (List.ofFn padHigh), .nats (List.ofFn dilation),
        .nat groups, .shape lead]⟩
  | ⟨_, _, @Op.pool s op dims kernel stride padLow padHigh dilation ceilMode⟩ =>
      ⟨formatVersion, "pool", [.shape s, .nat (poolOpCode op), .nats (dims.map (·.val)),
        .nats kernel, .nats stride, .nats padLow, .nats padHigh, .nats dilation, .bool ceilMode]⟩
  | ⟨_, _, @Op.smoothMaxPool s dims kernel stride padLow padHigh dilation ceilMode⟩ =>
      ⟨formatVersion, "smoothMaxPool", [.shape s, .nats (dims.map (·.val)), .nats kernel,
        .nats stride, .nats padLow, .nats padHigh, .nats dilation, .bool ceilMode]⟩
  | ⟨_, _, @Op.layerNorm seqLen embedDim _ _ lead _ _ _⟩ =>
      ⟨formatVersion, "layerNorm", [.nat seqLen, .nat embedDim, .shape lead]⟩
  | ⟨_, _, @Op.batchNorm channels spatial _⟩ =>
      ⟨formatVersion, "batchNorm", [.nat channels, .shape spatial]⟩
  | ⟨_, _, @Op.batchNormRunning s dim⟩ =>
      ⟨formatVersion, "batchNormRunning", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.multiheadAttention batch n numHeads dModel headDim _ mask⟩ =>
      ⟨formatVersion, "multiheadAttention", [.shape batch, .nat n, .nat numHeads, .nat dModel,
        .nat headDim, .mask (mask.map maskEntries)]⟩
  | ⟨_, _, @Op.convTranspose s k r dims stride padLow padHigh outputPadding dilation groups⟩ =>
      ⟨formatVersion, "convTranspose", [.shape s, .shape k, .nat r,
        .nats (List.ofFn fun i => (dims.input i).val),
        .nats (List.ofFn fun i => (dims.kernel i).val), .nats (List.ofFn stride),
        .nats (List.ofFn padLow), .nats (List.ofFn padHigh), .nats (List.ofFn outputPadding),
        .nats (List.ofFn dilation), .nat groups]⟩
  | ⟨_, _, @Op.eq s e⟩ => ⟨formatVersion, "eq", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.ne s e⟩ => ⟨formatVersion, "ne", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.lt s e⟩ => ⟨formatVersion, "lt", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.le s e⟩ => ⟨formatVersion, "le", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.gt s e⟩ => ⟨formatVersion, "gt", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.ge s e⟩ => ⟨formatVersion, "ge", [.shape s, .element e.toElement]⟩
  | ⟨_, _, @Op.isNaN s⟩ => ⟨formatVersion, "isNaN", [.shape s]⟩
  | ⟨_, _, @Op.isInf s⟩ => ⟨formatVersion, "isInf", [.shape s]⟩
  | ⟨_, _, @Op.isFinite s⟩ => ⟨formatVersion, "isFinite", [.shape s]⟩
  | ⟨_, _, @Op.logicalNot s⟩ => ⟨formatVersion, "logicalNot", [.shape s]⟩
  | ⟨_, _, @Op.logicalAnd s⟩ => ⟨formatVersion, "logicalAnd", [.shape s]⟩
  | ⟨_, _, @Op.logicalOr s⟩ => ⟨formatVersion, "logicalOr", [.shape s]⟩
  | ⟨_, _, @Op.logicalXor s⟩ => ⟨formatVersion, "logicalXor", [.shape s]⟩
  | ⟨_, _, @Op.bitwiseAnd signed width s⟩ =>
      ⟨formatVersion, "bitwiseAnd", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.bitwiseOr signed width s⟩ =>
      ⟨formatVersion, "bitwiseOr", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.bitwiseXor signed width s⟩ =>
      ⟨formatVersion, "bitwiseXor", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.bitwiseNot signed width s⟩ =>
      ⟨formatVersion, "bitwiseNot", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.bitwiseLeftShift signed width s⟩ =>
      ⟨formatVersion, "bitwiseLeftShift", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.shiftRightArithmetic signed width s⟩ =>
      ⟨formatVersion, "shiftRightArithmetic", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.shiftRightLogical signed width s⟩ =>
      ⟨formatVersion, "shiftRightLogical", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.populationCount signed width s⟩ =>
      ⟨formatVersion, "populationCount", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.countLeadingZeros signed width s⟩ =>
      ⟨formatVersion, "countLeadingZeros", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.mulhi signed width s⟩ =>
      ⟨formatVersion, "mulhi", [.bool signed, .nat width, .shape s]⟩
  | ⟨_, _, @Op.bitcast s width source target⟩ =>
      ⟨formatVersion, "bitcast", [.shape s, .nat width, .nat (wordCode source),
        .nat (wordCode target)]⟩
  | ⟨_, _, @Op.cbrt s⟩ => ⟨formatVersion, "cbrt", [.shape s]⟩
  | ⟨_, _, @Op.nextafter s⟩ => ⟨formatVersion, "nextafter", [.shape s]⟩
  | ⟨_, _, @Op.argmax s dim keepdim _⟩ =>
      ⟨formatVersion, "argmax", [.shape s, .nat dim.val, .bool keepdim]⟩
  | ⟨_, _, @Op.argmin s dim keepdim _⟩ =>
      ⟨formatVersion, "argmin", [.shape s, .nat dim.val, .bool keepdim]⟩
  | ⟨_, _, @Op.flip s dims e⟩ =>
      ⟨formatVersion, "flip", [.shape s, .nats (dims.map (·.val)), .element e]⟩
  | ⟨_, _, @Op.indices s dim⟩ => ⟨formatVersion, "indices", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.cast s source⟩ =>
      ⟨formatVersion, "cast", [.shape s, .nat (castSourceCode source)]⟩
  | ⟨_, _, @Op.encodeIndex s bounds signed width⟩ =>
      ⟨formatVersion, "encodeIndex", [.shape s, .nats bounds, .bool signed, .nat width]⟩
  | ⟨_, _, @Op.decodeIndex s bounds signed width⟩ =>
      ⟨formatVersion, "decodeIndex", [.shape s, .nats bounds, .bool signed, .nat width]⟩
  | ⟨_, _, @Op.attention batch rows keys depth width mask schedule⟩ =>
      ⟨formatVersion, "attention", [.shape batch, .nat rows, .nat keys, .nat depth, .nat width,
        .mask (mask.map maskEntries), .nat (scheduleCode schedule)]⟩
  | ⟨_, _, @Op.reduce s op dims keepdim _⟩ =>
      ⟨formatVersion, "reduce", [.shape s, .nat (reduceOpCode op), .nats (dims.map (·.val)),
        .bool keepdim]⟩
  | ⟨_, _, @Op.mean s dims keepdim⟩ =>
      ⟨formatVersion, "mean", [.shape s, .nats (dims.map (·.val)), .bool keepdim]⟩
  | ⟨_, _, @Op.contract s t dims batchDims _⟩ =>
      ⟨formatVersion, "contract", [.shape s, .shape t,
        .pairs (dims.map fun p => (p.1.val, p.2.val)),
        .pairs (batchDims.map fun p => (p.1.val, p.2.val))]⟩
  | ⟨_, _, @Op.maskedSoftmax s dim⟩ => ⟨formatVersion, "maskedSoftmax", [.shape s, .nat dim.val]⟩
  | ⟨_, _, @Op.slice s dims starts stops steps _ e⟩ =>
      ⟨formatVersion, "slice", [.shape s, .nats (dims.map (·.val)), .nats starts, .nats stops,
        .nats steps, .element e]⟩
  | ⟨_, _, @Op.pad s low high interior e⟩ =>
      ⟨formatVersion, "pad", [.shape s, .nats (List.ofFn low), .nats (List.ofFn high),
        .nats (List.ofFn interior), .element e]⟩
  | ⟨_, _, @Op.indexAt s dim index⟩ =>
      ⟨formatVersion, "indexAt", [.shape s, .nat dim.val, .nat index.val]⟩
  | ⟨_, _, @Op.cat s dim sizes e⟩ =>
      ⟨formatVersion, "cat", [.shape s, .nat dim.val, .nats sizes, .element e]⟩
  | ⟨_, _, @Op.scatter s t rule dims _ e⟩ =>
      ⟨formatVersion, "scatter", [.shape s, .shape t, .nat (scatterCode rule),
        .nats (dims.map (·.val)), .element e]⟩
  | ⟨_, _, @Op.topKMask s dim k⟩ => ⟨formatVersion, "topKMask", [.shape s, .nat dim.val, .nat k]⟩
  | ⟨_, _, @Op.bernoulliMask s key⟩ =>
      ⟨formatVersion, "bernoulliMask", [.shape s, .nat key.toNat]⟩
  | ⟨_, _, @Op.rfft s dim n _⟩ => ⟨formatVersion, "rfft", [.shape s, .nat dim.val, .nat n]⟩
  | ⟨_, _, @Op.irfft s dim n _⟩ => ⟨formatVersion, "irfft", [.shape s, .nat dim.val, .nat n]⟩
  | ⟨_, _, @Op.selectiveScan s dim sharedA sharedB⟩ =>
      ⟨formatVersion, "selectiveScan", [.shape s, .nat dim.val, .bool sharedA, .bool sharedB]⟩
  | ⟨_, _, @Op.custom name inputs output⟩ =>
      ⟨formatVersion, "custom", [.name name, .shapes inputs, .shape output]⟩

open Op.Wire in
/-- The erased operator of a wire record of version `Op.Wire.formatVersion` whose tag is known and
whose fields are the tag's, in order. The fields must meet the restrictions of the typed operator
(an axis below the rank, a positive extent, a numeric element on an arithmetic operator), and
`Op.check` checks further attributes, such as the positive width of an index adapter. -/
def Op.decode (wire : Op.Wire) : Option (Σ ins outs, Op ins outs) :=
  if wire.version = Op.Wire.formatVersion then
    match wire.tag, wire.fields with
    | "relu", [.shape s] => some ⟨_, _, Op.relu (s := s)⟩
    | "sigmoid", [.shape s] => some ⟨_, _, Op.sigmoid (s := s)⟩
    | "tanh", [.shape s] => some ⟨_, _, Op.tanh (s := s)⟩
    | "exp", [.shape s] => some ⟨_, _, Op.exp (s := s)⟩
    | "abs", [.shape s] => some ⟨_, _, Op.abs (s := s)⟩
    | "sqrt", [.shape s, .nat code] => do
        let policy ← sqrtPolicy? code
        pure ⟨_, _, Op.sqrt (s := s) policy⟩
    | "sin", [.shape s] => some ⟨_, _, Op.sin (s := s)⟩
    | "cos", [.shape s] => some ⟨_, _, Op.cos (s := s)⟩
    | "neg", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.neg (s := s) e⟩
    | "add", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.add (s := s) e⟩
    | "mul", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.mul (s := s) e⟩
    | "max", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.max (s := s) e⟩
    | "log", [.shape s] => some ⟨_, _, Op.log (s := s)⟩
    | "detach", [.shape s] => some ⟨_, _, Op.detach (s := s)⟩
    | "softplus", [.shape s] => some ⟨_, _, Op.softplus (s := s)⟩
    | "silu", [.shape s] => some ⟨_, _, Op.silu (s := s)⟩
    | "square", [.shape s] => some ⟨_, _, Op.square (s := s)⟩
    | "reciprocal", [.shape s] => some ⟨_, _, Op.reciprocal (s := s)⟩
    | "sub", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.sub (s := s) e⟩
    | "div", [.shape s, .nat code, .element e'] => do
        let rounding ← divRounding? code
        let e ← e'.numeric?
        pure ⟨_, _, Op.div (s := s) rounding e⟩
    | "min", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.min (s := s) e⟩
    | "cumsum", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cumsum (s := s) (dim := dim)⟩
    | "safeLog", [.shape s] => some ⟨_, _, Op.safeLog (s := s)⟩
    | "clamp", [.shape s] => some ⟨_, _, Op.clamp (s := s)⟩
    | "scale", [.shape s] => some ⟨_, _, Op.scale (s := s)⟩
    | "elu", [.shape s] => some ⟨_, _, Op.elu (s := s)⟩
    | "leakyRelu", [.shape s] => some ⟨_, _, Op.leakyReLU (s := s)⟩
    | "softmax", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.softmax (s := s) (dim := dim)⟩
    | "logSoftmax", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.logSoftmax (s := s) (dim := dim)⟩
    | "linear", [.shape lead, .nat inDim, .nat outDim] => some ⟨_, _, Op.linear (lead := lead)
        (inDim := inDim) (outDim := outDim)⟩
    | "attentionFromScores", [.shape batch, .nat rows, .nat keys, .nat width, .mask mask'] => do
        let mask ← mask? rows keys mask'
        pure ⟨_, _, Op.attentionFromScores (batch := batch) (rows := rows) (keys := keys)
            (width := width)
              (mask := mask)⟩
    | "sort", [.shape s, .nat dim', .bool descending] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.sort (s := s) (dim := dim) (descending := descending)⟩
    | "topk", [.shape s, .nat dim', .nat k, .bool largest] => do
        let dim ← fin? (s.rank) dim'
        let ⟨h⟩ ← prop? (k ≤ s.extent dim)
        pure ⟨_, _, Op.topk (s := s) (dim := dim) (k := k) (largest := largest) (h := h)⟩
    | "gather", [.shape s, .shape t, .nats dims', .pairs batchDims', .element e] => do
        let dims ← finList? (s.rank) dims'
        let batchDims ← finPairs? (s.rank) (t.rank) batchDims'
        let ⟨h⟩ ← prop? (Shape.GatherValid s t dims batchDims)
        pure ⟨_, _, Op.gather (s := s) (t := t) (dims := dims) (batchDims := batchDims) (h := h) e⟩
    | "randUniform", [.shape s, .nat key'] => do
        let key ← uint64? key'
        pure ⟨_, _, Op.randUniform (s := s) (key := key)⟩
    | "randn", [.shape s, .nat key'] => do
        let key ← uint64? key'
        pure ⟨_, _, Op.randn (s := s) (key := key)⟩
    | "sign", [.shape s] => some ⟨_, _, Op.sign (s := s)⟩
    | "floor", [.shape s] => some ⟨_, _, Op.floor (s := s)⟩
    | "ceil", [.shape s] => some ⟨_, _, Op.ceil (s := s)⟩
    | "round", [.shape s] => some ⟨_, _, Op.round (s := s)⟩
    | "trunc", [.shape s] => some ⟨_, _, Op.trunc (s := s)⟩
    | "frac", [.shape s] => some ⟨_, _, Op.frac (s := s)⟩
    | "rsqrt", [.shape s] => some ⟨_, _, Op.rsqrt (s := s)⟩
    | "exp2", [.shape s] => some ⟨_, _, Op.exp2 (s := s)⟩
    | "expm1", [.shape s] => some ⟨_, _, Op.expm1 (s := s)⟩
    | "log2", [.shape s] => some ⟨_, _, Op.log2 (s := s)⟩
    | "log10", [.shape s] => some ⟨_, _, Op.log10 (s := s)⟩
    | "log1p", [.shape s] => some ⟨_, _, Op.log1p (s := s)⟩
    | "tan", [.shape s] => some ⟨_, _, Op.tan (s := s)⟩
    | "asinh", [.shape s] => some ⟨_, _, Op.asinh (s := s)⟩
    | "acosh", [.shape s] => some ⟨_, _, Op.acosh (s := s)⟩
    | "atanh", [.shape s] => some ⟨_, _, Op.atanh (s := s)⟩
    | "atan", [.shape s] => some ⟨_, _, Op.atan (s := s)⟩
    | "asin", [.shape s] => some ⟨_, _, Op.asin (s := s)⟩
    | "acos", [.shape s] => some ⟨_, _, Op.acos (s := s)⟩
    | "erf", [.shape s] => some ⟨_, _, Op.erf (s := s)⟩
    | "atan2", [.shape s] => some ⟨_, _, Op.atan2 (s := s)⟩
    | "fmod", [.shape s] => some ⟨_, _, Op.fmod (s := s)⟩
    | "remainder", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.remainder (s := s) e⟩
    | "hypot", [.shape s] => some ⟨_, _, Op.hypot (s := s)⟩
    | "copysign", [.shape s] => some ⟨_, _, Op.copysign (s := s)⟩
    | "fmax", [.shape s] => some ⟨_, _, Op.fmax (s := s)⟩
    | "fmin", [.shape s] => some ⟨_, _, Op.fmin (s := s)⟩
    | "nanToNum", [.shape s] => some ⟨_, _, Op.nanToNum (s := s)⟩
    | "select", [.shape s, .element e] => some ⟨_, _, Op.select (s := s) e⟩
    | "lerp", [.shape s] => some ⟨_, _, Op.lerp (s := s)⟩
    | "addcmul", [.shape s] => some ⟨_, _, Op.addcmul (s := s)⟩
    | "addcdiv", [.shape s] => some ⟨_, _, Op.addcdiv (s := s)⟩
    | "gelu", [.shape s, .nat code] => do
        let approximate ← geluApproximation? code
        pure ⟨_, _, Op.gelu (s := s) approximate⟩
    | "sinh", [.shape s] => some ⟨_, _, Op.sinh (s := s)⟩
    | "cosh", [.shape s] => some ⟨_, _, Op.cosh (s := s)⟩
    | "safeInv", [.shape s] => some ⟨_, _, Op.safeInv (s := s)⟩
    | "smoothAbs", [.shape s] => some ⟨_, _, Op.smoothAbs (s := s)⟩
    | "safeDiv", [.shape s] => some ⟨_, _, Op.safeDiv (s := s)⟩
    | "pow", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.pow (s := s) e⟩
    | "cumprod", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cumprod (s := s) (dim := dim)⟩
    | "cummax", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cummax (s := s) (dim := dim)⟩
    | "cummin", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cummin (s := s) (dim := dim)⟩
    | "mseLoss", [.shape s] => some ⟨_, _, Op.mseLoss (s := s)⟩
    | "maeLoss", [.shape s] => some ⟨_, _, Op.maeLoss (s := s)⟩
    | "huberLoss", [.shape s, .bool isMean] => some ⟨_, _, Op.huberLoss (s := s) isMean⟩
    | "hingeLoss", [.shape s] => some ⟨_, _, Op.hingeLoss (s := s)⟩
    | "poissonLoss", [.shape s] => some ⟨_, _, Op.poissonLoss (s := s)⟩
    | "logCoshLoss", [.shape s] => some ⟨_, _, Op.logCoshLoss (s := s)⟩
    | "binaryCrossEntropy", [.shape s] => some ⟨_, _, Op.binaryCrossEntropy (s := s)⟩
    | "cosineLoss", [.shape s] => some ⟨_, _, Op.cosineLoss (s := s)⟩
    | "crossEntropy", [.shape s, .nat dim', .bool isMean] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.crossEntropy (s := s) (dim := dim) (mean := isMean)⟩
    | "crossEntropyProbabilities", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.crossEntropyProbabilities (s := s) (dim := dim)⟩
    | "distill", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.distill (s := s) (dim := dim)⟩
    | "reshape", [.shape s, .shape t, .element e] => do
        let ⟨h⟩ ← prop? (s.size = t.size)
        pure ⟨_, _, Op.reshape (s := s) (t := t) (h := h) e⟩
    | "permute", [.shape s, .nats perm', .shape out, .element e] => do
        let perm ← equiv? (s.rank) (s.rank) perm'
        let ⟨hout⟩ ← prop? (out = s.permute perm)
        pure ⟨_, _, Op.permute (s := s) (perm := perm) (out := out) (hout := hout) e⟩
    | "broadcast", [.shape s, .shape t, .nats dims', .element e] => do
        let dims ← finFun? s.rank t.rank dims'
        let ⟨h⟩ ← prop? (Shape.BroadcastDims s t dims)
        pure ⟨_, _, Op.broadcast (s := s) (t := t) dims h e⟩
    | "fullLike", [.shape s] => some ⟨_, _, Op.fullLike (s := s)⟩
    | "dropout", [.shape s] => some ⟨_, _, Op.dropout (s := s)⟩
    | "embedding", [.nat num, .nat dim, .shape t] => some ⟨_, _, Op.embedding (num := num)
        (dim := dim) (t := t)⟩
    | "matmul", [.shape batchA, .shape batchB, .shape batch, .nat m, .nat n, .nat p, .bool lv,
        .bool rv] => do
        let ⟨hl⟩ ← prop? (lv = true → m = 1)
        let ⟨hr⟩ ← prop? (rv = true → p = 1)
        let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
        let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
        pure ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
            (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩
    | "scaledMatmul", [.shape batchA, .shape batchB, .shape batch, .nat m, .nat n, .nat p,
        .bool lv, .bool rv] => do
        let ⟨hl⟩ ← prop? (lv = true → m = 1)
        let ⟨hr⟩ ← prop? (rv = true → p = 1)
        let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
        let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
        pure ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch)
            (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩
    | "conv", [.shape s, .shape k, .nat r, .nats dimsInput, .nats dimsKernel, .nats stride',
        .nats padLow', .nats padHigh', .nats dilation', .nat groups, .shape lead] => do
        let dimsIn ← equiv? (r + 2) s.rank dimsInput
        let dimsKer ← equiv? (r + 2) k.rank dimsKernel
        let dims : Shape.ConvDims s k r := ⟨dimsIn, dimsKer⟩
        let stride ← fun? r stride'
        let padLow ← fun? r padLow'
        let padHigh ← fun? r padHigh'
        let dilation ← fun? r dilation'
        let ⟨hlead⟩ ← prop? (s.extent (dims.input ⟨0, Nat.zero_lt_succ _⟩) = lead.size)
        pure ⟨_, _, Op.conv (s := s) (k := k) (r := r) (dims := dims) (stride := stride)
            (padLow := padLow) (padHigh := padHigh) (dilation := dilation) (groups := groups)
            (lead := lead) hlead rfl rfl⟩
    | "pool", [.shape s, .nat op', .nats dims', .nats kernel, .nats stride, .nats padLow,
        .nats padHigh, .nats dilation, .bool ceilMode] => do
        let op ← poolOp? op'
        let dims ← finList? (s.rank) dims'
        pure ⟨_, _, Op.pool (s := s) (op := op) (dims := dims) (kernel := kernel)
            (stride := stride) (padLow := padLow)
              (padHigh := padHigh) (dilation := dilation) (ceilMode := ceilMode)⟩
    | "smoothMaxPool", [.shape s, .nats dims', .nats kernel, .nats stride, .nats padLow,
        .nats padHigh, .nats dilation, .bool ceilMode] => do
        let dims ← finList? (s.rank) dims'
        pure ⟨_, _, Op.smoothMaxPool (s := s) (dims := dims) (kernel := kernel)
            (stride := stride) (padLow := padLow)
              (padHigh := padHigh) (dilation := dilation) (ceilMode := ceilMode)⟩
    | "layerNorm", [.nat seqLen, .nat embedDim, .shape lead] => do
        let ⟨hSeq⟩ ← prop? (seqLen > 0)
        let ⟨hEmbed⟩ ← prop? (embedDim > 0)
        let ⟨hlead⟩ ← prop? (seqLen = lead.size)
        pure ⟨_, _, Op.layerNorm (seqLen := seqLen) (embedDim := embedDim) (lead := lead) hSeq
            hEmbed hlead rfl⟩
    | "batchNorm", [.nat channels, .shape spatial] => do
        let ⟨hWellFormed⟩ ← prop? ((Shape.dim channels spatial).Positive)
        pure ⟨_, _, Op.batchNorm (channels := channels) (spatial := spatial)
            (hWellFormed := hWellFormed)⟩
    | "batchNormRunning", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.batchNormRunning (s := s) dim⟩
    | "multiheadAttention", [.shape batch, .nat n, .nat numHeads, .nat dModel, .nat headDim,
        .mask mask'] => do
        let ⟨h1⟩ ← prop? (n ≠ 0)
        let mask ← mask? n n mask'
        pure ⟨_, _, Op.multiheadAttention (batch := batch) (n := n) (numHeads := numHeads)
            (dModel := dModel) (headDim := headDim) (h1 := h1) (mask := mask)⟩
    | "convTranspose", [.shape s, .shape k, .nat r, .nats dimsInput, .nats dimsKernel,
        .nats stride',
        .nats padLow', .nats padHigh', .nats outputPadding', .nats dilation', .nat groups] => do
        let dimsIn ← equiv? (r + 2) s.rank dimsInput
        let dimsKer ← equiv? (r + 2) k.rank dimsKernel
        let dims : Shape.ConvDims s k r := ⟨dimsIn, dimsKer⟩
        let stride ← fun? r stride'
        let padLow ← fun? r padLow'
        let padHigh ← fun? r padHigh'
        let outputPadding ← fun? r outputPadding'
        let dilation ← fun? r dilation'
        pure ⟨_, _, Op.convTranspose (s := s) (k := k) (r := r) (dims := dims) (stride := stride)
            (padLow := padLow)
              (padHigh := padHigh) (outputPadding := outputPadding) (dilation := dilation)
              (groups := groups)⟩
    | "eq", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.eq (s := s) e⟩
    | "ne", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.ne (s := s) e⟩
    | "lt", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.lt (s := s) e⟩
    | "le", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.le (s := s) e⟩
    | "gt", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.gt (s := s) e⟩
    | "ge", [.shape s, .element e'] => do
        let e ← e'.numeric?
        pure ⟨_, _, Op.ge (s := s) e⟩
    | "isNaN", [.shape s] => some ⟨_, _, Op.isNaN (s := s)⟩
    | "isInf", [.shape s] => some ⟨_, _, Op.isInf (s := s)⟩
    | "isFinite", [.shape s] => some ⟨_, _, Op.isFinite (s := s)⟩
    | "logicalNot", [.shape s] => some ⟨_, _, Op.logicalNot (s := s)⟩
    | "logicalAnd", [.shape s] => some ⟨_, _, Op.logicalAnd (s := s)⟩
    | "logicalOr", [.shape s] => some ⟨_, _, Op.logicalOr (s := s)⟩
    | "logicalXor", [.shape s] => some ⟨_, _, Op.logicalXor (s := s)⟩
    | "bitwiseAnd", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.bitwiseAnd (signed := signed) (width := width) (s := s)⟩
    | "bitwiseOr", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.bitwiseOr (signed := signed) (width := width) (s := s)⟩
    | "bitwiseXor", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.bitwiseXor (signed := signed) (width := width) (s := s)⟩
    | "bitwiseNot", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.bitwiseNot (signed := signed) (width := width) (s := s)⟩
    | "bitwiseLeftShift", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.bitwiseLeftShift (signed := signed) (width := width) (s := s)⟩
    | "shiftRightArithmetic", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.shiftRightArithmetic (signed := signed) (width := width) (s := s)⟩
    | "shiftRightLogical", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.shiftRightLogical (signed := signed) (width := width) (s := s)⟩
    | "populationCount", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.populationCount (signed := signed) (width := width) (s := s)⟩
    | "countLeadingZeros", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.countLeadingZeros (signed := signed) (width := width) (s := s)⟩
    | "mulhi", [.bool signed, .nat width, .shape s] =>
        some ⟨_, _, Op.mulhi (signed := signed) (width := width) (s := s)⟩
    | "bitcast", [.shape s, .nat width, .nat source', .nat target'] => do
        let source ← word? source'
        let target ← word? target'
        pure ⟨_, _, Op.bitcast (s := s) width source target⟩
    | "cbrt", [.shape s] => some ⟨_, _, Op.cbrt (s := s)⟩
    | "nextafter", [.shape s] => some ⟨_, _, Op.nextafter (s := s)⟩
    | "argmax", [.shape s, .nat dim', .bool keepdim] => do
        let dim ← fin? (s.rank) dim'
        let ⟨h⟩ ← prop? (0 < s.extent dim)
        pure ⟨_, _, Op.argmax (s := s) (dim := dim) (keepdim := keepdim) (h := h)⟩
    | "argmin", [.shape s, .nat dim', .bool keepdim] => do
        let dim ← fin? (s.rank) dim'
        let ⟨h⟩ ← prop? (0 < s.extent dim)
        pure ⟨_, _, Op.argmin (s := s) (dim := dim) (keepdim := keepdim) (h := h)⟩
    | "flip", [.shape s, .nats dims', .element e] => do
        let dims ← finList? (s.rank) dims'
        pure ⟨_, _, Op.flip (s := s) (dims := dims) e⟩
    | "indices", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.indices (s := s) (dim := dim)⟩
    | "cast", [.shape s, .nat source'] => some ⟨_, _, Op.cast (s := s) (castSource? source')⟩
    | "encodeIndex", [.shape s, .nats bounds, .bool signed, .nat width] =>
        some ⟨_, _, Op.encodeIndex (s := s) bounds signed width⟩
    | "decodeIndex", [.shape s, .nats bounds, .bool signed, .nat width] =>
        some ⟨_, _, Op.decodeIndex (s := s) bounds signed width⟩
    | "attention", [.shape batch, .nat rows, .nat keys, .nat depth, .nat width, .mask mask',
        .nat schedule'] => do
        let mask ← mask? rows keys mask'
        let schedule := schedule? schedule'
        pure ⟨_, _, Op.attention (batch := batch) (rows := rows) (keys := keys) (depth := depth)
            (width := width)
              (mask := mask) (schedule := schedule)⟩
    | "reduce", [.shape s, .nat op', .nats dims', .bool keepdim] => do
        let op ← reduceOp? op'
        let dims ← finList? (s.rank) dims'
        let ⟨h⟩ ← prop? (op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d)
        pure ⟨_, _, Op.reduce (s := s) op dims keepdim h⟩
    | "mean", [.shape s, .nats dims', .bool keepdim] => do
        let dims ← finList? (s.rank) dims'
        pure ⟨_, _, Op.mean (s := s) dims keepdim⟩
    | "contract", [.shape s, .shape t, .pairs dims', .pairs batchDims'] => do
        let dims ← finPairs? s.rank t.rank dims'
        let batchDims ← finPairs? s.rank t.rank batchDims'
        let ⟨h⟩ ← prop? (Shape.ContractValid s t dims batchDims)
        pure ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h⟩
    | "maskedSoftmax", [.shape s, .nat dim'] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.maskedSoftmax (s := s) dim⟩
    | "slice", [.shape s, .nats dims', .nats starts, .nats stops, .nats steps, .element e] => do
        let dims ← finList? (s.rank) dims'
        let ⟨h⟩ ← prop? (Shape.SliceValid s dims starts stops steps)
        pure ⟨_, _, Op.slice (s := s) dims starts stops steps h e⟩
    | "pad", [.shape s, .nats low', .nats high', .nats interior', .element e] => do
        let low ← fun? s.rank low'
        let high ← fun? s.rank high'
        let interior ← fun? s.rank interior'
        pure ⟨_, _, Op.pad (s := s) low high interior e⟩
    | "indexAt", [.shape s, .nat dim', .nat index'] => do
        let dim ← fin? (s.rank) dim'
        let index ← fin? (s.extent dim) index'
        pure ⟨_, _, Op.indexAt (s := s) dim index⟩
    | "cat", [.shape s, .nat dim', .nats sizes, .element e] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cat (s := s) dim sizes e⟩
    | "scatter", [.shape s, .shape t, .nat rule', .nats dims', .element e] => do
        let rule ← scatter? rule'
        let dims ← finList? (s.rank) dims'
        let ⟨h⟩ ← prop? dims.Nodup
        pure ⟨_, _, Op.scatter (s := s) (t := t) rule dims h e⟩
    | "topKMask", [.shape s, .nat dim', .nat k] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.topKMask (s := s) dim k⟩
    | "bernoulliMask", [.shape s, .nat key'] => do
        let key ← uint64? key'
        pure ⟨_, _, Op.bernoulliMask (s := s) key⟩
    | "rfft", [.shape s, .nat dim', .nat n] => do
        let dim ← fin? (s.rank) dim'
        let ⟨hn⟩ ← prop? (0 < n)
        pure ⟨_, _, Op.rfft (s := s) dim n hn⟩
    | "irfft", [.shape s, .nat dim', .nat n] => do
        let dim ← fin? (s.rank) dim'
        let ⟨hn⟩ ← prop? (0 < n)
        pure ⟨_, _, Op.irfft (s := s) dim n hn⟩
    | "selectiveScan", [.shape s, .nat dim', .bool sharedA, .bool sharedB] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.selectiveScan (s := s) dim sharedA sharedB⟩
    | "custom", [.name name, .shapes inputs, .shape output] => some ⟨_, _, Op.custom
        (name := name) (inputs := inputs) (output := output)⟩
    | _, _ => none
  else none

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
theorem decode_encode_cumsum (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.cumsum (s := s) (dim := dim)⟩) = some ⟨_, _, Op.cumsum (s := s)
        (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.cumsum (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cumsum (s := s) (dim := dim)⟩) := rfl
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
theorem decode_encode_sort (s : Shape) (dim : Fin (s.rank)) (descending : Bool) :
    decode (encode ⟨_, _, Op.sort (s := s) (dim := dim)
        (descending := descending)⟩) = some ⟨_, _, Op.sort (s := s) (dim := dim)
        (descending := descending)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.sort (s := s) (dim := dim) (descending := descending)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.sort (s := s) (dim := dim) (descending := descending)⟩) := rfl
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

/-- The round trip of `cumprod`. -/
theorem decode_encode_cumprod (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.cumprod (s := s) (dim := dim)⟩) = some ⟨_, _, Op.cumprod (s := s)
        (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.cumprod (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cumprod (s := s) (dim := dim)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `cummax`. -/
theorem decode_encode_cummax (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.cummax (s := s) (dim := dim)⟩) = some ⟨_, _, Op.cummax (s := s)
        (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.cummax (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cummax (s := s) (dim := dim)⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `cummin`. -/
theorem decode_encode_cummin (s : Shape) (dim : Fin (s.rank)) :
    decode (encode ⟨_, _, Op.cummin (s := s) (dim := dim)⟩) = some ⟨_, _, Op.cummin (s := s)
        (dim := dim)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.cummin (s := s) (dim := dim)⟩) =
    (do
      let dim ← fin? (s.rank) (dim.val)
      pure ⟨_, _, Op.cummin (s := s) (dim := dim)⟩) := rfl
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
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    decode (encode ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch)
        (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩) =
      some ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
        (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩ := by
  have heval : decode (encode ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB)
      (batch := batch) (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩) =
    (do
      let ⟨hl⟩ ← prop? (lv = true → m = 1)
      let ⟨hr⟩ ← prop? (rv = true → p = 1)
      let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
      let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
      pure ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
          (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩) := rfl
  rw [heval, prop?_of hl, prop?_of hr, prop?_of broadcastA, prop?_of broadcastB]
  rfl

/-- The round trip of `scaledMatmul`, at every value of the vector flags. -/
theorem decode_encode_scaledMatmul (batchA : Shape) (batchB : Shape) (batch : Shape) (m : Nat)
    (n : Nat) (p : Nat) (lv rv : Bool) (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    decode (encode ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch)
        (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩) =
      some ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
        (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩ := by
  have heval : decode (encode ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB)
      (batch := batch) (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩) =
    (do
      let ⟨hl⟩ ← prop? (lv = true → m = 1)
      let ⟨hr⟩ ← prop? (rv = true → p = 1)
      let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
      let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
      pure ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
          (n := n) (p := p) lv rv hl hr broadcastA broadcastB⟩) := rfl
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
    (dilation : List Nat) (ceilMode : Bool) :
    decode (encode ⟨_, _, Op.pool (s := s) (op := op) (dims := dims) (kernel := kernel)
        (stride := stride) (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
        (ceilMode := ceilMode)⟩) = some ⟨_, _, Op.pool (s := s) (op := op) (dims := dims)
        (kernel := kernel) (stride := stride) (padLow := padLow) (padHigh := padHigh)
        (dilation := dilation) (ceilMode := ceilMode)⟩ :=
  by
  have heval : decode (encode ⟨_, _, Op.pool (s := s) (op := op) (dims := dims)
      (kernel := kernel) (stride := stride) (padLow := padLow) (padHigh := padHigh)
      (dilation := dilation) (ceilMode := ceilMode)⟩) =
    (do
      let op ← poolOp? ((poolOpCode op))
      let dims ← finList? (s.rank) ((dims.map (·.val)))
      pure ⟨_, _, Op.pool (s := s) (op := op) (dims := dims) (kernel := kernel)
          (stride := stride) (padLow := padLow) (padHigh := padHigh) (dilation := dilation)
          (ceilMode := ceilMode)⟩) := rfl
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

/-- The round trip of `nextafter`. -/
theorem decode_encode_nextafter (s : Shape) :
    decode (encode ⟨_, _, Op.nextafter (s := s)⟩) = some ⟨_, _, Op.nextafter (s := s)⟩ :=
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
    (keepdim : Bool) (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d) :
    decode (encode ⟨_, _, Op.reduce (s := s) op dims keepdim h⟩) =
      some ⟨_, _, Op.reduce (s := s) op dims keepdim h⟩ := by
  have heval : decode (encode ⟨_, _, Op.reduce (s := s) op dims keepdim h⟩) =
    (do
      let op ← reduceOp? (reduceOpCode op)
      let dims ← finList? (s.rank) (dims.map (·.val))
      let ⟨h⟩ ← prop? (op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d)
      pure ⟨_, _, Op.reduce (s := s) op dims keepdim h⟩) := rfl
  rw [heval]
  simp [prop?_of h]

/-- The round trip of `mean`. -/
theorem decode_encode_mean (s : Shape) (dims : List (Fin (s.rank))) (keepdim : Bool) :
    decode (encode ⟨_, _, Op.mean (s := s) dims keepdim⟩) =
      some ⟨_, _, Op.mean (s := s) dims keepdim⟩ := by
  have heval : decode (encode ⟨_, _, Op.mean (s := s) dims keepdim⟩) =
    (do
      let dims ← finList? (s.rank) (dims.map (·.val))
      pure ⟨_, _, Op.mean (s := s) dims keepdim⟩) := rfl
  rw [heval]
  simp

/-- The round trip of `contract`. -/
theorem decode_encode_contract (s t : Shape) (dims batchDims : List (Fin s.rank × Fin t.rank))
    (h : Shape.ContractValid s t dims batchDims) :
    decode (encode ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h⟩) =
      some ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h⟩ := by
  have heval : decode (encode ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h⟩) =
    (do
      let dims ← finPairs? s.rank t.rank (dims.map fun p => (p.1.val, p.2.val))
      let batchDims ← finPairs? s.rank t.rank (batchDims.map fun p => (p.1.val, p.2.val))
      let ⟨h⟩ ← prop? (Shape.ContractValid s t dims batchDims)
      pure ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h⟩) := rfl
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
  case nextafter => exact Op.Wire.decode_encode_nextafter ..
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
