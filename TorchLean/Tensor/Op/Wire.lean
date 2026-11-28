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
# The wire records of the registered operators

An erased operator is `Σ ins outs, Op ins outs`. `Op.encode` writes it as a wire record
(`Op.Wire`): the format version `Op.Wire.formatVersion`, the constructor's name as its tag
(`leakyRelu` for `leakyReLU`), and one field (`Op.Field`) for each implicit shape and attribute,
in binder order. The fields are
shapes, naturals, Booleans, lists of naturals for axis lists and functions on `Fin r`, and pairs for
batch dimensions. A mask is an optional array of Booleans, a custom node's name is a string, a
generalized operator's element type is an `Op.Element` and a reduction's order is a
`ReduceOrder`. Every attribute has its field in every record, at every value: a square root writes
its policy code, a generalized operator its element (the scalar too), a reduction, contraction,
matrix product, pool or cumulative sum, product, maximum or minimum its order (the left fold too),
`huberLoss` its mean flag and `multiheadAttention` its leading shape, `[]` for one sequence.

Records are version 5 (`Op.Wire.formatVersion`), and `Op.decode` accepts version 5 only. Proof
attributes are not sent, and `Op.decode` decides each one. It checks what the typed operator
needs: an axis below the rank, a mask of the right length, an equivalence that is a bijection and a
key below `2 ^ 64`. On an arithmetic operator or a comparison the element must be numeric. It
rejects any unknown tag and any field list other than its tag's. It does not run `Op.check`: a
decoded pooling with an ill-formed window is an operator value, which a recorder rejects before
reading its input. The ports are recomputed from the attributes, so two operators at different
shapes have different records. `TorchLean.Tensor.Op.Codec` proves that decoding inverts
encoding.

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
  /-- The order of a reduction. -/
  | order (value : ReduceOrder)
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
def formatVersion : Nat := 5

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
  | ⟨_, _, @Op.cumsum s dim order⟩ =>
      ⟨formatVersion, "cumsum", [.shape s, .nat dim.val, .order order]⟩
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
  | ⟨_, _, @Op.sort s dim descending e⟩ =>
      ⟨formatVersion, "sort", [.shape s, .nat dim.val, .bool descending, .element e]⟩
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
  | ⟨_, _, @Op.powValue s⟩ => ⟨formatVersion, "powValue", [.shape s]⟩
  | ⟨_, _, @Op.cumprod s dim order⟩ =>
      ⟨formatVersion, "cumprod", [.shape s, .nat dim.val, .order order]⟩
  | ⟨_, _, @Op.cummax s dim order⟩ =>
      ⟨formatVersion, "cummax", [.shape s, .nat dim.val, .order order]⟩
  | ⟨_, _, @Op.cummin s dim order⟩ =>
      ⟨formatVersion, "cummin", [.shape s, .nat dim.val, .order order]⟩
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
  | ⟨_, _, @Op.matmul batchA batchB batch m n p lv rv _ _ _ _ order⟩ =>
      ⟨formatVersion, "matmul", [.shape batchA, .shape batchB, .shape batch, .nat m, .nat n,
        .nat p, .bool lv, .bool rv, .order order]⟩
  | ⟨_, _, @Op.scaledMatmul batchA batchB batch m n p lv rv _ _ _ _ order⟩ =>
      ⟨formatVersion, "scaledMatmul", [.shape batchA, .shape batchB, .shape batch, .nat m, .nat n,
        .nat p, .bool lv, .bool rv, .order order]⟩
  | ⟨_, _, @Op.conv s k r dims stride padLow padHigh dilation groups lead _ _ _ _ _⟩ =>
      ⟨formatVersion, "conv", [.shape s, .shape k, .nat r,
        .nats (List.ofFn fun i => (dims.input i).val),
        .nats (List.ofFn fun i => (dims.kernel i).val), .nats (List.ofFn stride),
        .nats (List.ofFn padLow), .nats (List.ofFn padHigh), .nats (List.ofFn dilation),
        .nat groups, .shape lead]⟩
  | ⟨_, _, @Op.pool s op dims kernel stride padLow padHigh dilation ceilMode order⟩ =>
      ⟨formatVersion, "pool", [.shape s, .nat (poolOpCode op), .nats (dims.map (·.val)),
        .nats kernel, .nats stride, .nats padLow, .nats padHigh, .nats dilation, .bool ceilMode,
        .order order]⟩
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
  | ⟨_, _, @Op.special orders f s⟩ =>
      ⟨formatVersion, "special", [.name f.torchName, .nats orders, .shape s]⟩
  | ⟨_, _, @Op.nextafter s⟩ => ⟨formatVersion, "nextafter", [.shape s]⟩
  | ⟨_, _, @Op.frexp s⟩ => ⟨formatVersion, "frexp", [.shape s]⟩
  | ⟨_, _, @Op.fma s⟩ => ⟨formatVersion, "fma", [.shape s]⟩
  | ⟨_, _, @Op.complex f s⟩ => ⟨formatVersion, "complex", [.name f.jaxName, .shape s]⟩
  | ⟨_, _, @Op.complexPow s⟩ => ⟨formatVersion, "complex_pow", [.shape s]⟩
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
  | ⟨_, _, @Op.reduce s op dims keepdim _ order⟩ =>
      ⟨formatVersion, "reduce", [.shape s, .nat (reduceOpCode op), .nats (dims.map (·.val)),
        .bool keepdim, .order order]⟩
  | ⟨_, _, @Op.mean s dims keepdim order⟩ =>
      ⟨formatVersion, "mean", [.shape s, .nats (dims.map (·.val)), .bool keepdim, .order order]⟩
  | ⟨_, _, @Op.contract s t dims batchDims _ order⟩ =>
      ⟨formatVersion, "contract", [.shape s, .shape t,
        .pairs (dims.map fun p => (p.1.val, p.2.val)),
        .pairs (batchDims.map fun p => (p.1.val, p.2.val)), .order order]⟩
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
    | "cumsum", [.shape s, .nat dim', .order order] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cumsum (s := s) (dim := dim) (order := order)⟩
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
    | "sort", [.shape s, .nat dim', .bool descending, .element e] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.sort (s := s) (dim := dim) (descending := descending) e⟩
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
    | "powValue", [.shape s] => some ⟨_, _, Op.powValue (s := s)⟩
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
    | "cumprod", [.shape s, .nat dim', .order order] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cumprod (s := s) (dim := dim) (order := order)⟩
    | "cummax", [.shape s, .nat dim', .order order] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cummax (s := s) (dim := dim) (order := order)⟩
    | "cummin", [.shape s, .nat dim', .order order] => do
        let dim ← fin? (s.rank) dim'
        pure ⟨_, _, Op.cummin (s := s) (dim := dim) (order := order)⟩
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
        .bool rv, .order order] => do
        let ⟨hl⟩ ← prop? (lv = true → m = 1)
        let ⟨hr⟩ ← prop? (rv = true → p = 1)
        let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
        let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
        pure ⟨_, _, Op.matmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m)
            (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩
    | "scaledMatmul", [.shape batchA, .shape batchB, .shape batch, .nat m, .nat n, .nat p,
        .bool lv, .bool rv, .order order] => do
        let ⟨hl⟩ ← prop? (lv = true → m = 1)
        let ⟨hr⟩ ← prop? (rv = true → p = 1)
        let ⟨broadcastA⟩ ← prop? (Shape.CanBroadcastTo batchA batch)
        let ⟨broadcastB⟩ ← prop? (Shape.CanBroadcastTo batchB batch)
        pure ⟨_, _, Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch)
            (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB order⟩
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
        .nats padHigh, .nats dilation, .bool ceilMode, .order order] => do
        let op ← poolOp? op'
        let dims ← finList? (s.rank) dims'
        pure ⟨_, _, Op.pool (s := s) (op := op) (dims := dims) (kernel := kernel)
            (stride := stride) (padLow := padLow)
              (padHigh := padHigh) (dilation := dilation) (ceilMode := ceilMode) (order := order)⟩
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
    | "special", [.name f', .nats orders, .shape s] => do
        let f ← SpecialFn.ofTorchName? orders.length f'
        pure ⟨_, _, Op.special orders f (s := s)⟩
    | "nextafter", [.shape s] => some ⟨_, _, Op.nextafter (s := s)⟩
    | "frexp", [.shape s] => some ⟨_, _, Op.frexp (s := s)⟩
    | "fma", [.shape s] => some ⟨_, _, Op.fma (s := s)⟩
    | "complex", [.name f', .shape s] => do
        let f ← ComplexFn.ofJaxName? f'
        pure ⟨_, _, Op.complex f (s := s)⟩
    | "complex_pow", [.shape s] => some ⟨_, _, Op.complexPow (s := s)⟩
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
    | "reduce", [.shape s, .nat op', .nats dims', .bool keepdim, .order order] => do
        let op ← reduceOp? op'
        let dims ← finList? (s.rank) dims'
        let ⟨h⟩ ← prop? (op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d)
        pure ⟨_, _, Op.reduce (s := s) op dims keepdim h order⟩
    | "mean", [.shape s, .nats dims', .bool keepdim, .order order] => do
        let dims ← finList? (s.rank) dims'
        pure ⟨_, _, Op.mean (s := s) dims keepdim order⟩
    | "contract", [.shape s, .shape t, .pairs dims', .pairs batchDims', .order order] => do
        let dims ← finPairs? s.rank t.rank dims'
        let batchDims ← finPairs? s.rank t.rank batchDims'
        let ⟨h⟩ ← prop? (Shape.ContractValid s t dims batchDims)
        pure ⟨_, _, Op.contract (s := s) (t := t) dims batchDims h order⟩
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

end TorchLean
