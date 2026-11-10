/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table

/-!
# The rules of the registered operators

`Op.rule?` gives each operator its executable rule (`Op.Rule`): the tensor operator, its pullback
and, where they differ from the defaults, its prepared pullback, contribution order, alias rule,
value-domain check and forward-mode rule. An operator that needs a capability the scalar lacks
(`Rounding`, `Erf`, `InverseTrig`, `Cbrt`, `NextAfter`, or `Bits` of the operator's width) has no
rule there. Every operator but `custom` has an equation lemma for one or more of its arms
(`Op.rule?_relu`, ...), so a proof about one operator rewrites with that lemma or reduces that one
arm, never the whole table.

The definitions before `Op.rule?` are the pieces its arms share: the parts and gradient blocks of
`cat` (`Op.catElementParts`, `Op.catGrads`), the index tuples of `gather` and `scatter`
(`Op.axisTuples`, `Op.gatherSources`, `Op.scatterForward`), the index words of `encodeIndex` and
`decodeIndex` (`Op.indexWords`, `Op.wordIndices`) and the zero-divisor check of the integer division
and remainder (`Op.checkDivisors`).

The equation lemmas of the arms are in one module per operator family:
`TorchLean.Tensor.Op.Table.Elementwise`, `TorchLean.Tensor.Op.Table.Compare`,
`TorchLean.Tensor.Op.Table.Word`, `TorchLean.Tensor.Op.Table.Reduce`,
`TorchLean.Tensor.Op.Table.Movement`, `TorchLean.Tensor.Op.Table.Linear`,
`TorchLean.Tensor.Op.Table.Conv`, `TorchLean.Tensor.Op.Table.Normalization`,
`TorchLean.Tensor.Op.Table.Attention`, `TorchLean.Tensor.Op.Table.Loss` and
`TorchLean.Tensor.Op.Table.Random`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

namespace Op

/-- The tangents of the inputs of `cat` along `dim`, as the family of their tensors by position. -/
def catTangentParts {α : Type} [Storage α] {s : Shape} (dim : Fin s.rank) : (sizes : List Nat) →
    Tangents α (sizes.map fun n => Port.value (s.set dim n)) →
    (i : Fin sizes.length) → Tensor α (s.set dim (sizes.get i))
  | [], _, i => i.elim0
  | _ :: _, (x, _), ⟨0, _⟩ => x
  | _ :: rest, (_, xs), ⟨k + 1, h⟩ => catTangentParts dim rest xs ⟨k, Nat.lt_of_succ_lt_succ h⟩

/-- The inputs of `cat` along `dim` at the element `e`, as the family of their tensors by
position. -/
def catElementParts {α : Type} [Storage α] {e : Op.Element} {s : Shape} (dim : Fin s.rank) :
    (sizes : List Nat) → Values α (sizes.map fun n => e.port .value (s.set dim n)) →
    (i : Fin sizes.length) → (e.port .value (s.set dim (sizes.get i))).Carrier α
  | [], _, i => i.elim0
  | _ :: _, (x, _), ⟨0, _⟩ => x
  | _ :: rest, (_, xs), ⟨k + 1, h⟩ => catElementParts dim rest xs ⟨k, Nat.lt_of_succ_lt_succ h⟩

/-- The block of `g` along `dim` of `n` entries from `offset`. -/
def catBlock {α : Type} [Storage α] {s : Shape} {total : Nat} (dim : Fin s.rank)
    (g : Tensor α (s.set dim total)) (offset n : Nat) (h : offset + n ≤ total) :
    Tensor α (s.set dim n) :=
  Rep.ofFn fun c =>
    let split := Shape.Coord.setEquiv dim.val (shape := s) n c
    g ((Shape.Coord.setEquiv dim.val (shape := s) total).symm
      (⟨offset + split.1.val, by omega⟩, split.2))

/-- The gradients of the inputs of `cat`: the consecutive blocks of `g` along `dim`, from
`offset`. -/
def catGrads {α : Type} [Storage α] {s : Shape} {total : Nat} (dim : Fin s.rank)
    (g : Tensor α (s.set dim total)) :
    (offset : Nat) → (sizes : List Nat) → offset + sizes.sum ≤ total →
    Tangents α (sizes.map fun n => Port.value (s.set dim n))
  | _, [], _ => ()
  | offset, n :: rest, h =>
      (catBlock dim g offset n (by simp at h; omega),
        catGrads dim g (offset + n) rest (by simp at h; omega))

/-- The tangents of the inputs of `cat` along `dim` at the element `e`, `unit t` at the input of
shape `t`: the units of Boolean or integer inputs. -/
def catUnits {α : Type} [Storage α] {e : Op.Element} {s : Shape} (dim : Fin s.rank)
    (unit : (t : Shape) → (e.port .value t).Tangent α) :
    (sizes : List Nat) → Tangents α (sizes.map fun n => e.port .value (s.set dim n))
  | [] => ()
  | _ :: rest => (unit _, catUnits dim unit rest)

/-- The index tuples `Tensor.gather` and `Tensor.scatter` read, from the entries of an index port:
one index below the extent of each axis of `dims`, in order. -/
def axisTuples {r : Nat} (extent : Fin r → Nat) :
    (dims : List (Fin r)) → Port.IndexTuple (dims.map extent) → Einops.AxisTuple extent dims
  | [], _ => fun k => k.elim0
  | [_], index => fun k => Fin.cast (by rw [Fin.fin_one_eq_zero k]; rfl) index
  | _ :: _ :: _, tuple => fun k =>
    Fin.cast (List.getElem_map ..) (tuple ⟨k.val, by simpa using k.isLt⟩)

/-- The entry of an index port holding the index tuple `tuple` along `dims`: one index for one
axis and the tuple otherwise. `axisTuples` reads it back. -/
def indexOfAxisTuples {r : Nat} (extent : Fin r → Nat) :
    (dims : List (Fin r)) → Einops.AxisTuple extent dims → Port.IndexTuple (dims.map extent)
  | [], _ => fun k => k.elim0
  | [_], tuple => tuple ⟨0, Nat.one_pos⟩
  | _ :: _ :: _, tuple => fun k =>
    Fin.cast (List.getElem_map ..).symm (tuple ⟨k.val, by simpa using k.isLt⟩)

/-- `axisTuples` reads back the tuple `indexOfAxisTuples` stores. -/
theorem axisTuples_indexOfAxisTuples {r : Nat} (extent : Fin r → Nat) :
    (dims : List (Fin r)) → (tuple : Einops.AxisTuple extent dims) →
      axisTuples extent dims (indexOfAxisTuples extent dims tuple) = tuple
  | [], _ => funext fun k => k.elim0
  | [_], tuple => by
      funext k
      rw [Fin.fin_one_eq_zero k]
      rfl
  | _ :: _ :: _, tuple => by
      funext k
      apply Fin.ext
      rfl

/-- The scatter indices of `gather`'s pullback: at a coordinate `c` of the index tensor, the
gathered index along each axis of `dims`, then the index of `c` along the paired axis of each
batch pair. -/
def gatherSources {s t : Shape} {dims : List (Fin s.rank)}
    {batchDims : List (Fin s.rank × Fin t.rank)} (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t) :
    Tensor (Option (Einops.AxisTuple s.extent (dims ++ batchDims.map Prod.fst))) t :=
  Rep.ofFn fun c => some fun k =>
    if hk : k.val < dims.length then
      Fin.cast (by simp [List.getElem_append_left hk]) (tuples c ⟨k.val, hk⟩)
    else
      have hj : k.val - dims.length < batchDims.length := by
        have := k.isLt; simp at this; omega
      Fin.cast (by
          rw [← h.extent_eq _ (List.getElem_mem hj)]
          simp [List.getElem_append_right (Nat.le_of_not_lt hk)])
        (Shape.Coord.get c (batchDims[k.val - dims.length]'hj).2)

/-- The forward map of `scatter` at the element `β`: the updates written into `base` at the index
tuples of `index` along `dims` by `rule` (`Tensor.scatter`), a sum, product, maximum or minimum
combining by the given steps. -/
def scatterForward {β : Type} [Storage β] [Tensor.ScatterReduce.Scalar .sum β]
    [Tensor.ScatterReduce.Scalar .prod β] [maxStep : Tensor.ScatterReduce.Scalar .max β]
    [minStep : Tensor.ScatterReduce.Scalar .min β] {s t : Shape} (rule : Tensor.ScatterReduce)
    (dims : List (Fin s.rank)) (base : Tensor β s)
    (index : Tensor (Port.IndexTuple (dims.map s.extent)) t)
    (updates : Tensor β (t ++ s.eraseAxes dims)) : Tensor β s :=
  let positions := Tensor.map (fun i => some (Op.axisTuples _ dims i)) index
  match rule with
  | .sum => Tensor.scatter .sum base dims positions updates
  | .replace => Tensor.scatter .replace base dims positions updates
  | .prod => Tensor.scatter .prod base dims positions updates
  | .max => Tensor.scatter .max base dims positions updates
  | .min => Tensor.scatter .min base dims positions updates

/-- The components of the index tuples of `x`, tuple after tuple in row-major order. -/
def indexComponents {bounds : List Nat} {s : Shape} (x : Tensor (Port.IndexTuple bounds) s) :
    List Nat :=
  (Rep.data x).toList.flatMap (Port.IndexTuple.components bounds)

/-- The words of `width` bits of the components of the index tuples of `x`, in the layout of
`Port.irShape`, each component `c` as `BitVec.ofNat width c`. -/
def indexWords {bounds : List Nat} {s : Shape} (width : Nat)
    (x : Tensor (Port.IndexTuple bounds) s) :
    Tensor (BitVec width) (Port.irShape ⟨.index bounds, s⟩) :=
  Rep.ofFlatFn fun j => BitVec.ofNat width ((indexComponents x).getD j.val 0)

/-- Whether every component of the index tuples of `x` is a value of the integer type of `width`
bits, signed when `signed`: an error at the first that is not. -/
def checkIndexWords {bounds : List Nat} {s : Shape} (signed : Bool) (width : Nat)
    (x : Tensor (Port.IndexTuple bounds) s) : Except String Unit :=
  match (indexComponents x).find? fun c => !Word.fits signed width c with
  | some c => .error s!"index {c} does not fit {if signed then "a signed" else "an unsigned"} \
      {width}-bit integer"
  | none => .ok ()

/-- Whether every word of the divisor `y` is nonzero: an error at the first zero word, with the
reference interpreter's message under `label`. -/
def checkDivisors {w : Nat} {s : Shape} (label : String) (y : Tensor (BitVec w) s) :
    Except String Unit :=
  if (Rep.data y).toList.any (· == 0) then
    .error s!"{label}: integer division by zero (PyTorch raises ZeroDivisionError)"
  else .ok ()

/-- The index tuple at flat position `j` of the words `x`: component `k` the value of the word at
`j * bounds.length + k`, read as a signed integer when `signed`, an error where a component lies
outside its bound. -/
def wordIndex {bounds : List Nat} {s : Shape} {width : Nat} (signed : Bool)
    (x : Tensor (BitVec width) (Port.irShape ⟨.index bounds, s⟩)) (j : Nat) :
    Except String (Port.IndexTuple bounds) :=
  Port.IndexTuple.ofComponents bounds fun k =>
    Word.toInteger signed ((Rep.data x).getD (j * bounds.length + k) 0)

/-- The index tuples held by the words `x`, an error at the first component outside its bound. -/
def wordIndices {bounds : List Nat} {s : Shape} {width : Nat} (signed : Bool)
    (x : Tensor (BitVec width) (Port.irShape ⟨.index bounds, s⟩)) :
    Except String (Tensor (Port.IndexTuple bounds) s) :=
  Tensor.generateFlatM s fun j => wordIndex signed x j.val

/-- The index tuples held by the words `x`, or where a component lies outside its bound the zero
tuples (every bound positive), or the empty tensor (`s` has no entry). -/
def wordIndicesOr {bounds : List Nat} {s : Shape} {width : Nat}
    (h : (∀ b ∈ bounds, 0 < b) ∨ s.size = 0) (signed : Bool)
    (x : Tensor (BitVec width) (Port.irShape ⟨.index bounds, s⟩)) :
    Tensor (Port.IndexTuple bounds) s :=
  match wordIndices signed x with
  | .ok y => y
  | .error _ => Rep.ofFlatFn fun j =>
      if hb : ∀ b ∈ bounds, 0 < b then Port.IndexTuple.zero bounds hb
      else (Nat.not_lt_zero j.val (by
        have hj : j.val < Shape.size s := j.isLt
        have hz : Shape.size s = 0 := by
          exact h.resolve_left hb
        omega)).elim

end Op

variable {α : Type} [Storage α] [Context α]

/-- The executable rule of each operator, or `none`. The result is `none` for `custom`, whose rule
is the checked program of its node, and where the scalar lacks a capability the operator reads. It
is also `none` for a `bitcast` with a scalar side whose words do not have `width` bits, a
`decodeIndex` with a zero bound and entries, an integer true division and a Boolean scatter other
than a replacement. -/
def Op.rule? [Op.Capabilities α] :
    {ins outs : List Port} → Op ins outs → Option (Op.Rule α ins outs)
  | _, _, .relu => some
      { forward := fun xs => (Tensor.map Activation.relu xs.1, ())
        backward := fun xs g =>
          (Tensor.map Activation.reluDeriv xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Activation.reluDeriv xs.1 * dxs.1, ()) }
  | _, _, .sigmoid => some
      { forward := fun xs => (Tensor.map Context.sigmoid xs.1, ())
        backward := fun xs g => (Tensor.sigmoidBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.sigmoidBackward xs.1 dxs.1, ()) }
  | _, _, .tanh => some
      { forward := fun xs => (Tensor.map MathFunctions.tanh xs.1, ())
        backward := fun xs g => (Tensor.tanhBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.tanhBackward xs.1 dxs.1, ()) }
  | _, _, .exp => some
      { forward := fun xs => (Tensor.exp xs.1, ())
        backward := fun xs g => (Tensor.zipWith Context.expMul xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith Context.expMul xs.1 dxs.1, ()) }
  | _, _, .abs => some
      { forward := fun xs => (Tensor.abs xs.1, ())
        backward := fun xs g => (Tensor.sign xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.sign xs.1 * dxs.1, ()) }
  | _, _, .sqrt .raw => some
      { forward := fun xs => (Tensor.sqrt xs.1, ())
        backward := fun xs g => (Tensor.zipWith (fun value seed =>
          Numeric.sqrtBackward (MathFunctions.abs (MathFunctions.sqrt value))
            seed) xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith (fun value seed =>
          Numeric.sqrtBackward (MathFunctions.abs (MathFunctions.sqrt value))
            seed) xs.1 dxs.1, ()) }
  | _, _, .sqrt .clampNegative => some
      { forward := fun xs => (Tensor.sqrt xs.1 .clampNegative, ())
        backward := fun xs g => (Tensor.zipWith (fun value seed =>
          if value > 0 then
            Numeric.sqrtBackward (MathFunctions.sqrt value) seed
          else (0 : α) * seed) xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith (fun value seed =>
          if value > 0 then
            Numeric.sqrtBackward (MathFunctions.sqrt value) seed
          else (0 : α) * seed) xs.1 dxs.1, ()) }
  | _, _, .sin => some
      { forward := fun xs => (Tensor.sin xs.1, ())
        backward := fun xs g => (Tensor.cos xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.cos xs.1 * dxs.1, ()) }
  | _, _, .cos => some
      { forward := fun xs => (Tensor.cos xs.1, ())
        backward := fun xs g => (Tensor.neg (Tensor.sin xs.1) * g.1, ())
        jvp? := some fun xs dxs => (Tensor.neg (Tensor.sin xs.1) * dxs.1, ()) }
  | _, _, .neg => some
      { forward := fun xs => (Tensor.neg xs.1, ())
        backward := fun _ g => (Tensor.neg g.1, ())
        jvp? := some fun _ dxs => (Tensor.neg dxs.1, ()) }
  | _, _, .neg (.integer _ _) => some
      { forward := fun xs => (Tensor.map (- ·) xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .add => some
      { forward := fun xs => (xs.1 + xs.2.1, ())
        backward := fun _ g => (g.1, g.1, ())
        jvp? := some fun _ dxs => (dxs.1 + dxs.2.1, ()) }
  | _, _, .add (.integer _ _) => some
      { forward := fun xs => (Tensor.zipWith (· + ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .mul => some
      { forward := fun xs => (xs.1 * xs.2.1, ())
        backward := fun xs g => (g.1 * xs.2.1, g.1 * xs.1, ())
        jvp? := some fun xs dxs => ((dxs.1 * xs.2.1) + (xs.1 * dxs.2.1), ()) }
  | _, _, .mul (.integer _ _) => some
      { forward := fun xs => (Tensor.zipWith (· * ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .max => some
      { forward := fun xs => (Tensor.maximum xs.1 xs.2.1, ())
        backward := fun xs g =>
          let half : α := Rat.cast (1 / 2 : Rat)
          (Tensor.zipWith3 (fun a b seed =>
              if a > b then seed else if b > a then (0 : α) else half * seed) xs.1 xs.2.1 g.1,
            Tensor.zipWith3 (fun a b seed =>
              if a > b then seed else if b > a then (0 : α) else half * seed) xs.2.1 xs.1 g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            if xs.1 c > xs.2.1 c then dxs.1 c else if xs.2.1 c > xs.1 c then dxs.2.1 c
            else Numeric.average (dxs.1 c) (dxs.2.1 c), ())
        alias? := some
          { first := ⟨0, by simp⟩
            second := ⟨1, by simp⟩
            distinct := by simp
            same := rfl
            enabled := (inferInstance : Numeric.QuotientArithmetic α).supported
            prepare := fun xs flags => ⟨(Tensor.maximum xs.1 xs.2.1, ()), fun g =>
              (if flags.1 then some (Tensor.map (fun gradient =>
                Numeric.average gradient gradient) g.1)
              else none, none, ())⟩
            prepare_eq := by intro _ ⟨_, b, _⟩ _; cases b <;> rfl } }
  | _, _, .max (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.maximum signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .log => some
      { forward := fun xs => (Tensor.log xs.1, ())
        backward := fun xs g => (g.1 / xs.1, ())
        jvp? := some fun xs dxs => (dxs.1 / xs.1, ()) }
  | _, _, .detach => some
      { forward := fun xs => (Tensor.detach xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, .softplus => some
      { forward := fun xs => (Tensor.map Activation.softplus xs.1, ())
        backward := fun xs g =>
          (Tensor.map Context.sigmoid xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Context.sigmoid xs.1 * dxs.1, ()) }
  | _, _, .silu => some
      { forward := fun xs => (Tensor.map Activation.silu xs.1, ())
        backward := fun xs g =>
          (Tensor.map Activation.siluDeriv xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Activation.siluDeriv xs.1 * dxs.1, ()) }
  | _, _, .square => some
      { forward := fun xs => (Tensor.square xs.1, ())
        backward := fun xs g =>
          ((Tensor.full _ 2 * xs.1) * g.1, ())
        jvp? := some fun xs dxs => ((Tensor.full _ 2 * xs.1) * dxs.1, ()) }
  | _, _, .reciprocal => some
      { forward := fun xs => (Tensor.reciprocal xs.1, ())
        backward := fun xs g =>
          ((Tensor.neg g.1 / xs.1) / xs.1, ())
        jvp? := some fun xs dxs => ((Tensor.neg dxs.1 / xs.1) / xs.1, ()) }
  | _, _, .sub => some
      { forward := fun xs => (xs.1 - xs.2.1, ())
        backward := fun _ g => (g.1, Tensor.neg g.1, ())
        jvp? := some fun _ dxs => (dxs.1 - dxs.2.1, ()) }
  | _, _, .sub (.integer _ _) => some
      { forward := fun xs => (Tensor.zipWith (· - ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .div .none => some
      { forward := fun xs => (xs.1 / xs.2.1, ())
        backward := fun xs g =>
          (g.1 / xs.2.1, Tensor.neg (g.1 * ((xs.1 / xs.2.1) / xs.2.1)), ())
        jvp? := some fun xs dxs =>
          ((dxs.1 / xs.2.1) - (dxs.2.1 * ((xs.1 / xs.2.1) / xs.2.1)), ()) }
  | _, _, .div .trunc => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      -- The flooring division of the magnitudes corrects the rounded quotient with the exact
      -- remainder: `9` at `(1, 0.1)`, where the rounded quotient truncates to `10`.
      { forward := fun xs =>
          (Tensor.copysign (Tensor.floorDivide (Tensor.abs xs.1) (Tensor.abs xs.2.1))
            (xs.1 / xs.2.1), ())
        backward := fun _ g =>
          (Tensor.map (fun _ => (0 : α)) g.1, Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) }
    | none => none
  | _, _, .div .floor => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.floorDivide xs.1 xs.2.1, ())
        backward := fun _ g =>
          (Tensor.map (fun _ => (0 : α)) g.1, Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) }
    | none => none
  | _, _, .div .none (.integer _ _) => none
  | _, _, .div .trunc (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.divTrunc signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        validate := fun xs => Op.checkDivisors "div" xs.2.1 }
  | _, _, .div .floor (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.divFloor signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        validate := fun xs => Op.checkDivisors "div" xs.2.1 }
  | _, _, .min => some
      { forward := fun xs => (Tensor.minimum xs.1 xs.2.1, ())
        backward := fun xs g =>
          let half : α := Rat.cast (1 / 2 : Rat)
          (Tensor.zipWith3 (fun a b seed =>
              if b > a then seed else if a > b then (0 : α) else half * seed) xs.1 xs.2.1 g.1,
            Tensor.zipWith3 (fun a b seed =>
              if b > a then seed else if a > b then (0 : α) else half * seed) xs.2.1 xs.1 g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            if xs.1 c < xs.2.1 c then dxs.1 c else if xs.2.1 c < xs.1 c then dxs.2.1 c
            else Numeric.average (dxs.1 c) (dxs.2.1 c), ())
        alias? := some
          { first := ⟨0, by simp⟩
            second := ⟨1, by simp⟩
            distinct := by simp
            same := rfl
            enabled := (inferInstance : Numeric.QuotientArithmetic α).supported
            prepare := fun xs flags => ⟨(Tensor.minimum xs.1 xs.2.1, ()), fun g =>
              (if flags.1 then some (Tensor.map (fun gradient =>
                Numeric.average gradient gradient) g.1)
              else none, none, ())⟩
            prepare_eq := by intro _ ⟨_, b, _⟩ _; cases b <;> rfl } }
  | _, _, .min (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.minimum signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .cumsum dim => some
      { forward := fun xs => (Tensor.cumsum xs.1 dim, ())
        backward := fun _ g => (Tensor.cumsumBackward g.1 dim, ())
        jvp? := some fun _ dxs => (Tensor.cumsum dxs.1 dim, ()) }
  | _, _, .safeLog => some
      { forward := fun xs =>
          (Tensor.map (fun x => Activation.safeLog x xs.2.1.item) xs.1, ())
        backward := fun xs g =>
          ((Tensor.map (fun x => Activation.safeLogDeriv x xs.2.1.item)
            xs.1) * g.1, (), ())
        jvp? := some fun xs dxs =>
          ((Tensor.map (fun x => Activation.safeLogDeriv x xs.2.1.item)
            xs.1) * dxs.1, ()) }
  | _, _, .clamp => some
      { forward := fun xs =>
          (Tensor.clamp xs.1 (Tensor.full _ xs.2.1.item)
          (Tensor.full _ xs.2.2.1.item) .interior, ())
        backward := fun xs g =>
          (Tensor.clampDerivative xs.1 (Tensor.full _ xs.2.1.item)
              (Tensor.full _ xs.2.2.1.item) * g.1, (), (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.clampDerivative xs.1 (Tensor.full _ xs.2.1.item)
              (Tensor.full _ xs.2.2.1.item) * dxs.1, ()) }
  | _, _, .scale => some
      { forward := fun xs => (Tensor.scale xs.1 xs.2.1.item, ())
        backward := fun xs g => (Tensor.scale g.1 xs.2.1.item, (), ())
        jvp? := some fun xs dxs => (Tensor.scale dxs.1 xs.2.1.item, ()) }
  | _, _, .elu => some
      { forward := fun xs =>
          let alpha := xs.2.1.item
          (Tensor.map (fun x => Activation.elu x alpha) xs.1, ())
        backward := fun xs g =>
          let alpha := xs.2.1.item
          (Tensor.map (fun x => Activation.eluDeriv x alpha) xs.1 * g.1, (), ())
        jvp? := some fun xs dxs =>
          let alpha := xs.2.1.item
          (Tensor.map (fun x => Activation.eluDeriv x alpha) xs.1 * dxs.1, ()) }
  | _, _, .leakyReLU => some
      { forward := fun xs =>
          let slope := xs.2.1.item
          (Tensor.map (fun x => Activation.leakyReLU x slope) xs.1, ())
        backward := fun xs g =>
          let slope := xs.2.1.item
          (Tensor.map (fun x => Activation.leakyReLUDeriv x slope) xs.1 * g.1, (), ())
        jvp? := some fun xs dxs =>
          let slope := xs.2.1.item
          (Tensor.map (fun x => Activation.leakyReLUDeriv x slope) xs.1 * dxs.1, ()) }
  | _, _, .softmax dim => some
      { forward := fun xs => (Tensor.softmax xs.1 dim, ())
        backward := fun xs g => (Tensor.softmaxBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs => (Tensor.softmaxBackward xs.1 dxs.1 dim, ()) }
  | _, _, .logSoftmax dim => some
      { forward := fun xs => (Tensor.logSoftmax xs.1 dim, ())
        backward := fun xs g =>
          (Tensor.logSoftmaxBackward (Tensor.logSoftmax xs.1 dim) g.1 dim, ())
        prepare := fun xs flags =>
          let y := Tensor.logSoftmax xs.1 dim
          ⟨(y, ()), fun g =>
            (if flags.1 then some (Tensor.logSoftmaxBackward y g.1 dim) else none, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Tensor.logSoftmaxJVP (Tensor.logSoftmax xs.1 dim) dxs.1 dim, ()) }
  | _, _, @Op.linear .scalar _ _ => some
      { forward := fun xs => (Linear.forward { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ())
        backward := fun xs g =>
          (Linear.weightsDeriv xs.2.2.1 g.1, g.1, Linear.inputDeriv xs.1 g.1, ())
        prepare := fun xs flags =>
          ⟨(Linear.forward { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ()), fun g =>
          (if flags.1 then some (Linear.weightsDeriv xs.2.2.1 g.1) else none,
            if flags.2.1 then some g.1 else none,
            if flags.2.2.1 then some (Linear.inputDeriv xs.1 g.1) else none, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Linear.forward { weights := dxs.1, bias := dxs.2.1 } xs.2.2.1 +
            Linear.forward { weights := xs.1, bias := Tensor.full _ (0 : α) } dxs.2.2.1, ()) }
  | _, _, @Op.linear (.dim n rest) _ _ => some
      { forward := fun xs =>
          (Linear.forwardLeading (.dim n rest) { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ())
        backward := fun xs g =>
          (Linear.weightsDerivLeading (.dim n rest) xs.2.2.1 g.1,
            Linear.biasDerivLeading (.dim n rest) g.1,
            Linear.inputDerivLeading (.dim n rest) xs.1 g.1, ())
        prepare := fun xs flags =>
          ⟨(Linear.forwardLeading (.dim n rest) { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ()),
            fun g =>
            (if flags.1 then some (Linear.weightsDerivLeading (.dim n rest) xs.2.2.1 g.1)
              else none,
              if flags.2.1 then some (Linear.biasDerivLeading (.dim n rest) g.1) else none,
              if flags.2.2.1 then
                some (Linear.inputDerivLeading (.dim n rest) xs.1 g.1)
              else none, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Linear.forwardLeading (.dim n rest) { weights := dxs.1, bias := dxs.2.1 } xs.2.2.1 +
            Linear.forwardLeading (.dim n rest) { weights := xs.1, bias := Tensor.full _ (0 : α) }
              dxs.2.2.1, ()) }
  | _, _, .attentionFromScores mask => some
      { forward := fun xs =>
          (Attention.values (Attention.weights xs.1 mask) xs.2.1
            (some xs.2.2.1), ())
        backward := fun xs g =>
          let weights := Attention.weights xs.1 mask
          let (ds, dv, dm) := Attention.valuesBackward weights xs.2.1 xs.2.2.1 g.1
          (ds, dv, dm, ())
        prepare := fun xs flags =>
          let weights := Attention.weights xs.1 mask
          ⟨(Attention.values weights xs.2.1 (some xs.2.2.1), ()), fun g =>
            let (ds, dv, dm) := Attention.valuesBackward weights xs.2.1 xs.2.2.1 g.1
            Tangents.select _ flags (ds, dv, dm, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Attention.valuesJVP (Attention.weights xs.1 mask) xs.2.1
            xs.2.2.1 dxs.1 dxs.2.1 dxs.2.2.1, ()) }
  | _, _, .sort dim descending => some
      { forward := fun xs =>
          let sorted := Tensor.sort xs.1 dim descending
          (sorted.1, sorted.2, ())
        backward := fun xs g => (Tensor.sortBackward xs.1 g.1 dim descending, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            dxs.1 (Shape.Coord.update c dim ((Tensor.sort xs.1 dim descending).2 c)), (), ()) }
  | _, _, .topk (s := s) dim k largest h => some
      { forward := fun xs =>
          let top := Tensor.topk xs.1 dim k largest h
          (top.1, top.2, ())
        backward := fun xs g => (Tensor.topkBackward xs.1 dim k largest h g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.slice (Tensor.Rep.ofFn fun c =>
              dxs.1 (Shape.Coord.update c dim ((Tensor.sort xs.1 dim largest).2 c)))
            [dim] [0] [k] [1]
            (by simpa using Shape.sliceValid_singleton s dim 0 k (by simpa using h)), (), ()) }
  | _, _, .gather dims batchDims h => some
      { forward := fun xs =>
          (Tensor.gather xs.1 dims batchDims (Tensor.map (Op.axisTuples _ dims) xs.2.1) _ h, ())
        backward := fun xs g =>
          (Tensor.scatter .sum (Tensor.full _ 0) (dims ++ batchDims.map Prod.fst)
            (Op.gatherSources h (Tensor.map (Op.axisTuples _ dims) xs.2.1)) g.1, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.gather dxs.1 dims batchDims (Tensor.map (Op.axisTuples _ dims) xs.2.1) _ h,
            ()) }
  | _, _, .gather dims batchDims h .bool
  | _, _, .gather dims batchDims h (.integer _ _) => some
      { forward := fun xs =>
          (Tensor.gather xs.1 dims batchDims (Tensor.map (Op.axisTuples _ dims) xs.2.1) _ h, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .randUniform key => some
      { forward := fun _ => (Random.uniform key, ())
        backward := fun _ _ => ()
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, .randn key => some
      { forward := fun _ => (Random.normal key, ())
        backward := fun _ _ => ()
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, .sign => some
      { forward := fun xs => (Tensor.sign xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) }
  | _, _, .floor => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.floor xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) }
    | none => none
  | _, _, .ceil => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.ceil xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) }
    | none => none
  | _, _, .round => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.round xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) }
    | none => none
  | _, _, .trunc => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.trunc xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) }
    | none => none
  | _, _, .frac => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.frac xs.1, ())
        backward := fun _ g => (g.1, ())
        jvp? := some fun _ dxs => (dxs.1, ()) }
    | none => none
  | _, _, .rsqrt => some
      { forward := fun xs => (Tensor.rsqrt xs.1, ())
        backward := fun xs g => (Tensor.rsqrtBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.rsqrtBackward xs.1 dxs.1, ()) }
  | _, _, .exp2 => some
      { forward := fun xs => (Tensor.exp2 xs.1, ())
        backward := fun xs g => (Tensor.exp2Backward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.exp2Backward xs.1 dxs.1, ()) }
  | _, _, .expm1 => some
      { forward := fun xs => (Tensor.expm1 xs.1, ())
        backward := fun xs g => (Tensor.zipWith Context.expMul xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith Context.expMul xs.1 dxs.1, ()) }
  | _, _, .log2 => some
      { forward := fun xs => (Tensor.log2 xs.1, ())
        backward := fun xs g => (Tensor.log2Backward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.log2Backward xs.1 dxs.1, ()) }
  | _, _, .log10 => some
      { forward := fun xs => (Tensor.log10 xs.1, ())
        backward := fun xs g => (Tensor.log10Backward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.log10Backward xs.1 dxs.1, ()) }
  | _, _, .log1p => some
      { forward := fun xs => (Tensor.log1p xs.1, ())
        backward := fun xs g => (Tensor.log1pBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.log1pBackward xs.1 dxs.1, ()) }
  | _, _, .tan => some
      { forward := fun xs => (Tensor.tan xs.1, ())
        backward := fun xs g => (Tensor.tanBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.tanBackward xs.1 dxs.1, ()) }
  | _, _, .asinh => some
      { forward := fun xs => (Tensor.asinh xs.1, ())
        backward := fun xs g => (Tensor.asinhBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.asinhBackward xs.1 dxs.1, ()) }
  | _, _, .acosh => some
      { forward := fun xs => (Tensor.acosh xs.1, ())
        backward := fun xs g => (Tensor.acoshBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.acoshBackward xs.1 dxs.1, ()) }
  | _, _, .atanh => some
      { forward := fun xs => (Tensor.atanh xs.1, ())
        backward := fun xs g => (Tensor.atanhBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.atanhBackward xs.1 dxs.1, ()) }
  | _, _, .atan => match Op.Capabilities.inverseTrig? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.atan xs.1, ())
        backward := fun xs g => (Tensor.atanBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.atanBackward xs.1 dxs.1, ()) }
    | none => none
  | _, _, .asin => match Op.Capabilities.inverseTrig? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.asin xs.1, ())
        backward := fun xs g => (Tensor.asinBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.asinBackward xs.1 dxs.1, ()) }
    | none => none
  | _, _, .acos => match Op.Capabilities.inverseTrig? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.acos xs.1, ())
        backward := fun xs g => (Tensor.acosBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.acosBackward xs.1 dxs.1, ()) }
    | none => none
  | _, _, .erf => match Op.Capabilities.erf? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.erf xs.1, ())
        backward := fun xs g => (Tensor.erfBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.erfBackward xs.1 dxs.1, ()) }
    | none => none
  | _, _, .atan2 => match Op.Capabilities.inverseTrig? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.atan2 xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.atan2Backward xs.1 xs.2.1 g.1
          (da, db, ())
        -- One joint `angleTangent`: each partial term can overflow where the sum is finite.
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            Numeric.angleTangent (xs.2.1 c) (xs.1 c) (dxs.2.1 c) (dxs.1 c), ()) }
    | none => none
  | _, _, .fmod => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.fmod xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.fmodBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.fmodBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.fmodBackward xs.1 xs.2.1 dxs.2.1).2, ()) }
    | none => none
  | _, _, .remainder => match Op.Capabilities.rounding? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.remainder xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.remainderBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.remainderBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.remainderBackward xs.1 xs.2.1 dxs.2.1).2, ()) }
    | none => none
  | _, _, .remainder (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.remainder signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        validate := fun xs => Op.checkDivisors "remainder" xs.2.1 }
  | _, _, .hypot => some
      { forward := fun xs => (Tensor.hypot xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.hypotBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.hypotBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.hypotBackward xs.1 xs.2.1 dxs.2.1).2, ()) }
  | _, _, .copysign => some
      { forward := fun xs => (Tensor.copysign xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.copysignBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.copysignBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.copysignBackward xs.1 xs.2.1 dxs.2.1).2, ()) }
  | _, _, .fmax => some
      { forward := fun xs => (Tensor.fmax xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.fmaxBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.fmaxBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.fmaxBackward xs.1 xs.2.1 dxs.2.1).2, ()) }
  | _, _, .fmin => some
      { forward := fun xs => (Tensor.fmin xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.fminBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.fminBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.fminBackward xs.1 xs.2.1 dxs.2.1).2, ()) }
  | _, _, .nanToNum => some
      { forward := fun xs =>
          (Tensor.nanToNum xs.1 xs.2.1.item xs.2.2.1.item xs.2.2.2.1.item, ())
        backward := fun xs g => (Tensor.nanToNumBackward xs.1 g.1, (), (), (), ())
        jvp? := some fun xs dxs => (Tensor.nanToNumBackward xs.1 dxs.1, ()) }
  | _, _, .select => some
      { forward := fun xs => (Tensor.select xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun xs g =>
          let (dx, dy) := Tensor.selectBackward xs.1 g.1
          ((), dx, dy, ())
        jvp? := some fun xs dxs => (Tensor.select xs.1 dxs.2.1 dxs.2.2.1, ()) }
  | _, _, .select .bool | _, _, .select (.integer _ _) => some
      { forward := fun xs => (Tensor.select xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun _ _ => ((), (), (), ()) }
  | _, _, .lerp => some
      { forward := fun xs => (Tensor.lerp xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun xs g =>
          let (da, db, dw) := Tensor.lerpBackward xs.1 xs.2.1 xs.2.2.1 g.1
          (da, db, dw, ())
        jvp? := some fun xs dxs =>
          let pullback := Tensor.lerpBackward xs.1 xs.2.1 xs.2.2.1
          (((pullback dxs.1).1 + (pullback dxs.2.1).2.1) + (pullback dxs.2.2.1).2.2, ()) }
  | _, _, .addcmul => some
      { forward := fun xs => (Tensor.addcmul xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item, ())
        backward := fun xs g =>
          let (dx, da, db) := Tensor.addcmulBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item g.1
          (dx, da, db, (), ())
        jvp? := some fun xs dxs =>
          let pullback := Tensor.addcmulBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item
          (((pullback dxs.1).1 + (pullback dxs.2.1).2.1) + (pullback dxs.2.2.1).2.2, ()) }
  | _, _, .addcdiv => some
      { forward := fun xs => (Tensor.addcdiv xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item, ())
        backward := fun xs g =>
          let (dx, da, db) := Tensor.addcdivBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item g.1
          (dx, da, db, (), ())
        jvp? := some fun xs dxs =>
          let pullback := Tensor.addcdivBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item
          (((pullback dxs.1).1 + (pullback dxs.2.1).2.1) + (pullback dxs.2.2.1).2.2, ()) }
  | _, _, .gelu .tanh => some
      { forward := fun xs => (Tensor.map Activation.geluTanh xs.1, ())
        backward := fun xs g =>
          (Tensor.map Activation.geluTanhDeriv xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Activation.geluTanhDeriv xs.1 * dxs.1, ()) }
  | _, _, .gelu .none => match Op.Capabilities.erf? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.gelu xs.1, ())
        backward := fun xs g => (Tensor.geluBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.geluBackward xs.1 dxs.1, ()) }
    | none => none
  | _, _, .sinh => some
      { forward := fun xs => (Tensor.sinh xs.1, ())
        backward := fun xs g => (Tensor.map MathFunctions.cosh xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.map MathFunctions.cosh xs.1 * dxs.1, ()) }
  | _, _, .cosh => some
      { forward := fun xs => (Tensor.cosh xs.1, ())
        backward := fun xs g => (Tensor.map MathFunctions.sinh xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.map MathFunctions.sinh xs.1 * dxs.1, ()) }
  | _, _, .safeInv => some
      { forward := fun xs => (Tensor.safeDiv (Tensor.full _ (1 : α)) xs.1, ())
        backward := fun xs g =>
          ((Tensor.full _ (-(1 : α)) *
            (Tensor.square (Tensor.map (fun y =>
                (1 : α) / (y + Context.defaultEpsilon)) xs.1))) * g.1, ())
        jvp? := some fun xs dxs =>
          ((Tensor.full _ (-(1 : α)) *
            (Tensor.square (Tensor.map (fun y =>
                (1 : α) / (y + Context.defaultEpsilon)) xs.1))) * dxs.1, ()) }
  | _, _, .smoothAbs => some
      { forward := fun xs =>
          (Tensor.map (fun x => Activation.smoothAbs x xs.2.1.item) xs.1, ())
        backward := fun xs g =>
          (Tensor.map (fun x => Activation.smoothAbsDeriv x xs.2.1.item) xs.1 *
            g.1, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.map (fun x => Activation.smoothAbsDeriv x xs.2.1.item) xs.1 *
            dxs.1, ()) }
  | _, _, .safeDiv => some
      { forward := fun xs => (Tensor.safeDiv xs.1 xs.2.1, ())
        backward := fun xs g => (Tensor.safeDiv g.1 xs.2.1, (), ())
        jvp? := some fun xs dxs => (Tensor.safeDiv dxs.1 xs.2.1, ()) }
  | _, _, .pow => some
      { forward := fun xs => (Tensor.pow xs.1 xs.2.1, ())
        backward := fun xs g =>
          (Tensor.Rep.ofFlatFn fun index =>
            Numeric.powerBackward Context.isZero
              (Tensor.Rep.getFlat xs.1 index) (Tensor.Rep.getFlat xs.2.1 index)
              (Tensor.Rep.getFlat g.1 index), (), ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFlatFn fun index =>
            Numeric.powerBackward Context.isZero
              (Tensor.Rep.getFlat xs.1 index) (Tensor.Rep.getFlat xs.2.1 index)
              (Tensor.Rep.getFlat dxs.1 index), ()) }
  | _, _, .pow (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.pow signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .cumprod dim => some
      { forward := fun xs => (Tensor.cumprod xs.1 dim, ())
        backward := fun xs g => (Tensor.cumprodBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs => (Tensor.cumprodJVP xs.1 dxs.1 dim, ()) }
  | _, _, .cummax dim => some
      { forward := fun xs =>
          let running := Tensor.cummax xs.1 dim
          (running.1, running.2, ())
        backward := fun xs g => (Tensor.cummaxBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            dxs.1 (Shape.Coord.update c dim ((Tensor.cummax xs.1 dim).2 c)), (), ()) }
  | _, _, .cummin dim => some
      { forward := fun xs =>
          let running := Tensor.cummin xs.1 dim
          (running.1, running.2, ())
        backward := fun xs g => (Tensor.cumminBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            dxs.1 (Shape.Coord.update c dim ((Tensor.cummin xs.1 dim).2 c)), (), ()) }
  | _, _, .mseLoss (s := s) => some
      { forward := fun xs => (Tensor.scalar (Tensor.meanSquaredError xs.1 xs.2.1), ())
        backward := fun xs g =>
          let gradient := Tensor.scale (Tensor.meanSquaredErrorBackward xs.1 xs.2.1)
            (Tensor.item g.1)
          (gradient, Tensor.neg gradient, ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.sum
            (Tensor.meanSquaredErrorBackward xs.1 xs.2.1 * (dxs.1 - dxs.2.1))), ()) }
  | _, _, .maeLoss => some
      { forward := fun xs => (Tensor.scalar (Tensor.l1Loss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.l1LossBackward xs.1 xs.2.1) g.1.item, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.l1LossBackward xs.1 xs.2.1) dxs.1), ()) }
  | _, _, .huberLoss isMean => some
      { forward := fun xs => (Tensor.scalar (Tensor.huberLoss xs.1 xs.2.1 xs.2.2.1.item isMean), ())
        backward := fun xs g =>
            (Tensor.scale (Tensor.huberLossBackward xs.1 xs.2.1 xs.2.2.1.item isMean) g.1.item, (),
              (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.huberLossBackward xs.1 xs.2.1 xs.2.2.1.item isMean)
            dxs.1), ()) }
  | _, _, .hingeLoss => some
      { forward := fun xs => (Tensor.scalar (Tensor.hingeLoss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.hingeLossBackward xs.1 xs.2.1) g.1.item, (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.hingeLossBackward xs.1 xs.2.1) dxs.1), ()) }
  | _, _, .poissonLoss => some
      { forward := fun xs => (Tensor.scalar (Tensor.poissonNLLLoss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.poissonNLLLossBackward xs.1 xs.2.1) g.1.item,
            (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.poissonNLLLossBackward xs.1 xs.2.1) dxs.1), ()) }
  | _, _, .logCoshLoss => some
      { forward := fun xs => (Tensor.scalar (Tensor.logCoshLoss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.logCoshLossBackward xs.1 xs.2.1) g.1.item, (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.logCoshLossBackward xs.1 xs.2.1) dxs.1), ()) }
  | _, _, .binaryCrossEntropy => some
      { forward := fun xs =>
          (Tensor.scalar (Tensor.binaryCrossEntropy xs.1 xs.2.1 xs.2.2.1.item), ())
        backward := fun xs g =>
            (Tensor.scale
            (Tensor.binaryCrossEntropyBackward xs.1 xs.2.1 xs.2.2.1.item) g.1.item, (), (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
            (Tensor.binaryCrossEntropyBackward xs.1 xs.2.1 xs.2.2.1.item) dxs.1), ()) }
  | _, _, .cosineLoss (s := s) => some
      { forward := fun xs =>
          (Tensor.reshape (Tensor.cosineLoss xs.1 xs.2.1 (List.finRange s.rank) false
            xs.2.2.1.item) Shape.scalar
            ((Shape.size_reduce_finRange s false).trans Shape.size_nil.symm), ())
        backward := fun xs g =>
          (Tensor.scale (CosineLoss.fiberLossDeriv xs.1 xs.2.1 xs.2.2.1.item) g.1.item, (),
            (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
            (CosineLoss.fiberLossDeriv xs.1 xs.2.1 xs.2.2.1.item) dxs.1), ()) }
  | _, _, .crossEntropy dim isMean => some
      { forward := fun xs =>
          (Tensor.scalar (Tensor.crossEntropy .logits xs.1 xs.2.1 dim isMean), ())
        backward := fun xs g =>
          (Tensor.scale (Tensor.crossEntropyBackward .logits xs.1 xs.2.1 dim isMean) g.1.item,
            Tensor.crossEntropyTargetBackward xs.1 dim g.1.item isMean, ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
              (Tensor.crossEntropyBackward .logits xs.1 xs.2.1 dim isMean) dxs.1 +
            Tensor.crossEntropy .logits xs.1 dxs.2.1 dim isMean), ()) }
  | _, _, .crossEntropyProbabilities dim => some
      { forward := fun xs =>
          (Tensor.scalar (Tensor.crossEntropy (.probabilities xs.2.2.1.item) xs.1 xs.2.1 dim), ())
        backward := fun xs g =>
          (Tensor.scale
              (Tensor.crossEntropyBackward (.probabilities xs.2.2.1.item) xs.1 xs.2.1 dim) g.1.item,
            (), (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
            (Tensor.crossEntropyBackward (.probabilities xs.2.2.1.item) xs.1 xs.2.1 dim) dxs.1),
            ()) }
  | _, _, .distill dim => some
      { forward := fun xs => (Tensor.scalar (Tensor.distillLoss dim xs.1 xs.2.1), ())
        backward := fun xs g =>
          (Tensor.scale (Tensor.distillLossBackward dim xs.1 xs.2.1) g.1.item, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.distillLossBackward dim xs.1 xs.2.1) dxs.1), ()) }
  | _, _, .reshape (s := s) (t := t) h => some
      { forward := fun xs => (Tensor.reshape (source := s) xs.1 t h, ())
        backward := fun _ g => (Tensor.reshape (source := t) g.1 s h.symm, ())
        jvp? := some fun _ dxs => (Tensor.reshape (source := s) dxs.1 t h, ()) }
  | _, _, .reshape (s := s) (t := t) h .bool
  | _, _, .reshape (s := s) (t := t) h (.integer _ _) => some
      { forward := fun xs => (Tensor.reshape (source := s) xs.1 t h, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .permute (s := s) perm out hout => some
      { forward := fun xs => (Tensor.permute xs.1 perm out hout, ())
        backward := fun _ g =>
          (Tensor.permute g.1 (s.unpermute perm hout) s (s.permute_unpermute perm hout).symm, ())
        jvp? := some fun _ dxs => (Tensor.permute dxs.1 perm out hout, ()) }
  | _, _, .permute perm out hout .bool | _, _, .permute perm out hout (.integer _ _) => some
      { forward := fun xs => (Tensor.permute xs.1 perm out hout, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .broadcast dims h => some
      { forward := fun xs => (Tensor.broadcast xs.1 _ dims h, ())
        backward := fun _ g => (Tensor.reduceBroadcast dims h g.1, ())
        jvp? := some fun _ dxs => (Tensor.broadcast dxs.1 _ dims h, ()) }
  | _, _, .broadcast dims h .bool | _, _, .broadcast dims h (.integer _ _) => some
      { forward := fun xs => (Tensor.broadcast xs.1 _ dims h, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .fullLike (s := s) => some
      { forward := fun xs => (Tensor.full s xs.2.1.item, ())
        backward := fun xs _ => (Tensor.map (fun _ => 0) xs.1, (), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, .dropout => some
      { forward := fun xs =>
          (Dropout.masked xs.2.2.1.item (Tensor.ge xs.2.1 (Tensor.full _ xs.2.2.1.item))
            xs.1, ())
        backward := fun xs g =>
          (Dropout.masked xs.2.2.1.item
            (Tensor.ge xs.2.1 (Tensor.full _ xs.2.2.1.item)) g.1, (), (), ())
        jvp? := some fun xs dxs =>
          (Dropout.masked xs.2.2.1.item (Tensor.ge xs.2.1 (Tensor.full _ xs.2.2.1.item))
            dxs.1, ()) }
  | _, _, .embedding => some
      { forward := fun xs => (Tensor.embedding xs.1 xs.2.1, ())
        backward := fun xs g => (Tensor.embeddingBackward xs.2.1 g.1, (), ())
        jvp? := some fun xs dxs => (Tensor.embedding dxs.1 xs.2.1, ()) }
  | _, _, .matmul _ _ hl hr broadcastA broadcastB => some
      { forward := fun xs =>
          (Tensor.reshape (Tensor.batchMatmul broadcastA broadcastB
              (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _)))
            _ (Shape.size_matmulOut hl hr _).symm, ())
        backward := fun xs g =>
          let gradients := Tensor.matmulBackward broadcastA broadcastB
            (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
            (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _))
            (Tensor.reshape g.1 _ (Shape.size_matmulOut hl hr _))
          (Tensor.reshape gradients.1 _ (Shape.size_matmulLeft hl _ _).symm,
            Tensor.reshape gradients.2 _ (Shape.size_matmulRight hr _ _).symm, ())
        jvp? := some fun xs dxs =>
          (Tensor.reshape (Tensor.batchMatmul broadcastA broadcastB
                (Tensor.reshape dxs.1 _ (Shape.size_matmulLeft hl _ _))
                (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _)) +
              Tensor.batchMatmul broadcastA broadcastB
                (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
                (Tensor.reshape dxs.2.1 _ (Shape.size_matmulRight hr _ _)))
            _ (Shape.size_matmulOut hl hr _).symm, ()) }
  | _, _, .scaledMatmul _ _ hl hr broadcastA broadcastB => some
      { forward := fun xs =>
          (Tensor.reshape (Attention.scaledMatmul broadcastA broadcastB
              (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _)) xs.2.2.1.item)
            _ (Shape.size_matmulOut hl hr _).symm, ())
        backward := fun xs g =>
          let gradients := Attention.scaledMatmulBackward broadcastA broadcastB
            (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
            (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _))
            (Tensor.reshape g.1 _ (Shape.size_matmulOut hl hr _)) xs.2.2.1.item
          (Tensor.reshape gradients.1 _ (Shape.size_matmulLeft hl _ _).symm,
            Tensor.reshape gradients.2 _ (Shape.size_matmulRight hr _ _).symm, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.reshape (Attention.scaledMatmulJVP broadcastA broadcastB
              (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape dxs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _))
              (Tensor.reshape dxs.2.1 _ (Shape.size_matmulRight hr _ _)) xs.2.2.1.item)
            _ (Shape.size_matmulOut hl hr _).symm, ()) }
  | _, _, .conv (s := s) (k := k) dims stride padLow padHigh dilation groups hlead hx hy =>
    some
      { forward := fun xs =>
          (Tensor.reshape
            (Tensor.conv (Tensor.reshape xs.2.2.1 _ (Shape.size_eq_of_expandAxis hlead hx)) xs.1
                dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
              Tensor.broadcast xs.2.1
                (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1) (fun _ =>
                  dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨1, by omega⟩)
                (dims.broadcastDims_conv stride padLow padHigh (fun _ => 1) dilation 1))
            _ (Shape.size_conv_expandAxis hlead hy), ())
        backward := fun xs g =>
          (Tensor.convBackwardWeight (Tensor.reshape xs.2.2.1 _
              (Shape.size_eq_of_expandAxis hlead hx)) dims stride padLow padHigh dilation groups
              (Tensor.reshape g.1 _ (Shape.size_conv_expandAxis hlead hy).symm),
            Tensor.castShape
                (Tensor.reduce .sum
                  (Tensor.reshape g.1 _ (Shape.size_conv_expandAxis hlead hy).symm)
                (Shape.otherAxes (s.conv k dims stride padLow padHigh (fun _ =>
                1) dilation 1) [dims.output stride padLow padHigh (fun _ =>
                1) dilation 1 ⟨1, by omega⟩]) false)
              (Shape.reduce_conv_otherAxes dims stride padLow padHigh (fun _ => 1) dilation 1),
            Tensor.reshape (Tensor.convBackwardInput xs.1 dims stride padLow padHigh dilation groups
              (Tensor.reshape g.1 _ (Shape.size_conv_expandAxis hlead hy).symm)) _
              (Shape.size_eq_of_expandAxis hlead hx).symm, ())
        jvp? := some fun xs dxs =>
          let bias (b : Tensor α [k.extent (dims.kernel ⟨0, by omega⟩)]) :=
            Tensor.broadcast b (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1)
              (fun _ => dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨1, by omega⟩)
              (dims.broadcastDims_conv stride padLow padHigh (fun _ => 1) dilation 1)
          (Tensor.reshape
            ((Tensor.conv (Tensor.reshape dxs.2.2.1 _ (Shape.size_eq_of_expandAxis hlead hx)) xs.1
                  dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
                bias (Tensor.full _ (0 : α))) +
              (Tensor.conv (Tensor.reshape xs.2.2.1 _ (Shape.size_eq_of_expandAxis hlead hx))
                  dxs.1 dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
                bias dxs.2.1))
            _ (Shape.size_conv_expandAxis hlead hy), ()) }
  | _, _, .pool op dims kernel stride padLow padHigh dilation ceilMode => some
      { forward := fun xs =>
          (Tensor.pool op xs.1 dims kernel stride padLow padHigh dilation ceilMode, ())
        backward := fun xs g =>
          (Tensor.poolBackward op xs.1 dims kernel stride padLow padHigh dilation ceilMode g.1,
              ())
        jvp? := some fun xs dxs =>
          (Tensor.poolJVP op xs.1 dxs.1 dims kernel stride padLow padHigh dilation ceilMode, ()) }
  | _, _, .smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode =>
    some
      { forward := fun xs =>
          (Tensor.smoothMaxPool xs.2.1.item xs.1 dims kernel stride padLow padHigh dilation
            ceilMode, ())
        backward := fun xs g =>
          (Tensor.smoothMaxPoolBackward xs.2.1.item xs.1 dims kernel stride padLow padHigh dilation
            ceilMode g.1, (), ())
        validate := fun xs =>
          if xs.2.1.item == 0 || !(xs.2.1.item - xs.2.1.item == 0) then
            .error "autograd: smooth_max_pool requires finite nonzero beta"
          else .ok ()
        jvp? := some fun xs dxs =>
          (Tensor.smoothMaxPoolJVP xs.2.1.item xs.1 dxs.1 dims kernel stride padLow padHigh
            dilation ceilMode, ()) }
  | _, _, .layerNorm (seqLen := seqLen) (embedDim := embedDim) (x := x) hSeq hEmbed hlead hx =>
    some
      { forward := fun xs =>
          (Tensor.reshape (Tensor.layerNorm (Tensor.reshape xs.1 [seqLen, embedDim]
              (Shape.size_eq_of_append hlead hx)) [⟨1, Nat.one_lt_two⟩] xs.2.1 xs.2.2.1
              xs.2.2.2.1.item) x (Shape.size_eq_of_append hlead hx).symm, ())
        backward := fun xs g =>
          let gradients := LayerNorm.backward (sequenceLengthPositive := hSeq)
            (embeddingWidthPositive := hEmbed)
            (input := Tensor.reshape xs.1 [seqLen, embedDim] (Shape.size_eq_of_append hlead hx))
            (scale := xs.2.1)
            (outputGradient := Tensor.reshape g.1 [seqLen, embedDim]
              (Shape.size_eq_of_append hlead hx)) (epsilon := xs.2.2.2.1.item)
          (Tensor.reshape gradients.inputGradient x (Shape.size_eq_of_append hlead hx).symm,
            gradients.scaleGradient, gradients.biasGradient, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.reshape (LayerNorm.jvp (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
            (x := Tensor.reshape xs.1 [seqLen, embedDim] (Shape.size_eq_of_append hlead hx))
            (tangent := Tensor.reshape dxs.1 [seqLen, embedDim] (Shape.size_eq_of_append hlead hx))
            (gamma := xs.2.1) (dgamma := dxs.2.1) (_beta := xs.2.2.1) (dbeta := dxs.2.2.1)
            (epsilon := xs.2.2.2.1.item)) x (Shape.size_eq_of_append hlead hx).symm, ()) }
  | _, _, .batchNorm (channels := channels) (spatial := spatial) _ => some
      { forward := fun xs =>
          (Tensor.batchNorm xs.1 ⟨0, Nat.succ_pos _⟩ xs.2.1 xs.2.2.1 .batch xs.2.2.2.1.item, ())
        backward := fun xs g =>
          let gradients := BatchNorm.backward (x := xs.1) (gamma := xs.2.1) (gradOutput := g.1)
            (epsilon := xs.2.2.2.1.item)
          (gradients.inputGradient, gradients.scaleGradient, gradients.biasGradient, (), ())
        jvp? := some fun xs dxs =>
          (BatchNorm.jvp (channels := channels) (sSpatial := spatial) (x := xs.1)
            (tangent := dxs.1) (gamma := xs.2.1) (dgamma := dxs.2.1) (_beta := xs.2.2.1)
            (dbeta := dxs.2.2.1) (epsilon := xs.2.2.2.1.item), ()) }
  | _, _, .batchNormRunning (s := s) dim => some
      { forward := fun xs =>
          (Tensor.batchNorm xs.1 dim xs.2.1 xs.2.2.1 (.running xs.2.2.2.1 xs.2.2.2.2.1)
            xs.2.2.2.2.2.1.item, ())
        backward := fun xs g =>
          let gradients := BatchNorm.runningBackward xs.1 dim xs.2.1 xs.2.2.2.1 xs.2.2.2.2.1
            xs.2.2.2.2.2.1.item g.1
          (gradients.inputGradient, gradients.scaleGradient, gradients.biasGradient, (), (), (),
            ())
        jvp? := some fun xs dxs =>
          let spread := fun t : Tensor α [s.extent dim] =>
            Tensor.broadcast t s _ (s.broadcastDims_axis dim)
          let std := Tensor.sqrt (spread (Tensor.maximum xs.2.2.2.2.1 (Tensor.full _ 0)) +
            Tensor.full s xs.2.2.2.2.2.1.item) .clampNegative
          (((dxs.1 * spread xs.2.1) / std + ((xs.1 - spread xs.2.2.2.1) / std) * spread dxs.2.1) +
            spread dxs.2.2.1, ()) }
  | _, _, .multiheadAttention (batch := batch) (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) h1 mask => some
      { forward := fun xs =>
          (Tensor.mapLeading batch (fun x => MultiHeadAttention.forward (hQ := h1) (hK := h1)
            (mha := ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩) (query := x) (keyValue := x)
            (mask := mask))
            xs.2.2.2.2.1, ())
        backward := fun xs g =>
          let mha : MultiHeadAttention α numHeads dModel headDim :=
            ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩
          let gradients := MultiHeadAttention.backwardLeading h1 mha mask batch xs.2.2.2.2.1 g.1
          let parameters := MultiHeadAttention.sum mha gradients.2
          (parameters.queryWeight, parameters.keyWeight, parameters.valueWeight,
            parameters.outputWeight, gradients.1, ())
        order := [4, 0, 1, 2, 3]
        order_perm := (by decide : ([4, 0, 1, 2, 3] : List (Fin 5)).Perm (List.finRange 5))
        jvp? := some fun xs dxs =>
          (Tensor.zipEach batch [n, dModel] (fun (x dx : Tensor α [n, dModel]) =>
            MultiHeadAttention.jvp (h1 := h1) (mha := ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩)
              (dmha := ⟨dxs.1, dxs.2.1, dxs.2.2.1, dxs.2.2.2.1⟩) (x := x) (dx := dx)
              (mask := mask)) xs.2.2.2.2.1 dxs.2.2.2.2.1, ()) }
  | _, _, .convTranspose (s := s) (k :=
      k) dims stride padLow padHigh outputPadding dilation groups => some
      { forward := fun xs =>
          ((Tensor.convTranspose xs.2.2.1 xs.1 dims stride padLow padHigh outputPadding
              dilation groups) +
            (Tensor.broadcast xs.2.1
                (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)
                (fun _ =>
                dims.transposeOutput stride padLow padHigh outputPadding dilation groups ⟨1,
                by omega⟩)
              (dims.broadcastDims_convTranspose stride padLow padHigh outputPadding dilation
                  groups)), ())
        backward := fun xs g =>
          (Tensor.convTransposeBackwardWeight xs.2.2.1 dims stride padLow padHigh outputPadding
              dilation groups g.1,
            Tensor.castShape
                (Tensor.reduce .sum g.1
                (Shape.otherAxes
                (s.convTranspose k dims stride padLow padHigh outputPadding dilation
                groups) [dims.transposeOutput stride padLow padHigh outputPadding dilation groups
                ⟨1, by omega⟩]) false)
              (Shape.reduce_convTranspose_otherAxes dims stride padLow padHigh outputPadding
                  dilation
                groups),
            Tensor.convTransposeBackwardInput xs.1 dims stride padLow padHigh outputPadding dilation
              groups g.1, ())
        jvp? := some fun xs dxs =>
          let bias (b : Tensor α [groups * k.extent (dims.kernel ⟨0, by omega⟩)]) :=
            Tensor.broadcast b
              (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)
              (fun _ =>
                dims.transposeOutput stride padLow padHigh outputPadding dilation groups ⟨1,
                by omega⟩)
              (dims.broadcastDims_convTranspose stride padLow padHigh outputPadding dilation groups)
          (((Tensor.convTranspose dxs.2.2.1 xs.1 dims stride padLow padHigh outputPadding
                dilation groups) +
              bias (Tensor.full _ (0 : α))) +
            ((Tensor.convTranspose xs.2.2.1 dxs.1 dims stride padLow padHigh outputPadding
                dilation groups) +
              bias dxs.2.1), ()) }
  | _, _, .eq => some
      { forward := fun xs => (Tensor.eq xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .eq (.integer _ _) => some
      { forward := fun xs => (Tensor.zipWith (fun x y => decide (x = y)) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .ne => some
      { forward := fun xs => (Tensor.ne xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .ne (.integer _ _) => some
      { forward := fun xs => (Tensor.zipWith (fun x y => decide (x ≠ y)) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .lt => some
      { forward := fun xs => (Tensor.lt xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .lt (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.lt signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .le => some
      { forward := fun xs => (Tensor.le xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .le (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (Word.le signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .gt => some
      { forward := fun xs => (Tensor.gt xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .gt (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (fun x y => Word.lt signed y x) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .ge => some
      { forward := fun xs => (Tensor.ge xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .ge (.integer signed _) => some
      { forward := fun xs => (Tensor.zipWith (fun x y => Word.le signed y x) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .isNaN => some
      { forward := fun xs => (Tensor.isNaN xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .isInf => some
      { forward := fun xs => (Tensor.isInf xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .isFinite => some
      { forward := fun xs => (Tensor.isFinite xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .logicalNot => some
      { forward := fun xs => (Tensor.logicalNot xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .logicalAnd => some
      { forward := fun xs => (Tensor.logicalAnd xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .logicalOr => some
      { forward := fun xs => (Tensor.logicalOr xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .logicalXor => some
      { forward := fun xs => (Tensor.logicalXor xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .bitwiseAnd => some
      { forward := fun xs => (Tensor.zipWith (· &&& ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .bitwiseOr => some
      { forward := fun xs => (Tensor.zipWith (· ||| ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .bitwiseXor => some
      { forward := fun xs => (Tensor.zipWith (· ^^^ ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .bitwiseLeftShift => some
      { forward := fun xs => (Tensor.zipWith Word.shiftLeft xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .shiftRightArithmetic => some
      { forward := fun xs => (Tensor.zipWith Word.shiftRightArithmetic xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .shiftRightLogical => some
      { forward := fun xs => (Tensor.zipWith Word.shiftRightLogical xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .bitwiseNot => some
      { forward := fun xs => (Tensor.map (~~~ ·) xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .populationCount => some
      { forward := fun xs => (Tensor.map BitVec.cpop xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .countLeadingZeros => some
      { forward := fun xs => (Tensor.map BitVec.clz xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, @Op.mulhi signed _ _ => some
      { forward := fun xs => (Tensor.zipWith (Word.mulhi signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .bitcast _ (.integer _) (.integer _) => some
      { forward := fun xs => (xs.1, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .bitcast width .scalar (.integer _) => match Op.Capabilities.bits? (α := α) with
    | some instance_ => haveI := instance_
      if h : Numeric.Word.width α = width then some
        { forward := fun xs => (Tensor.map (fun (x : α) => (Numeric.Word.toBits x).cast h) xs.1, ())
          backward := fun _ _ => ((), ()) }
      else none
    | none => none
  | _, _, .bitcast width (.integer _) .scalar => match Op.Capabilities.bits? (α := α) with
    | some instance_ => haveI := instance_
      if h : Numeric.Word.width α = width then some
        { forward := fun xs =>
            (Tensor.map (fun (b : BitVec width) => Numeric.Word.ofBits (α := α) (b.cast h.symm))
              xs.1, ())
          backward := fun _ _ => ((), ())
          jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
      else none
    | none => none
  | _, _, .bitcast width .scalar .scalar => match Op.Capabilities.bits? (α := α) with
    | some instance_ => haveI := instance_
      if Numeric.Word.width α = width then some
        { forward := fun xs => (xs.1, ())
          backward := fun _ _ => ((), ())
          jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
      else none
    | none => none
  | _, _, .cbrt => match Op.Capabilities.cbrt? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.map Cbrt.cbrt xs.1, ())
        backward := fun xs g => (Tensor.zipWith (fun x seed =>
          let y := Cbrt.cbrt x
          seed * (((1 / 3 : Rat) : α) * (1 / (y * y)))) xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith (fun x dx =>
          let y := Cbrt.cbrt x
          dx * (((1 / 3 : Rat) : α) * (1 / (y * y)))) xs.1 dxs.1, ()) }
    | none => none
  | _, _, .nextafter => match Op.Capabilities.nextafter? (α := α) with
    | some instance_ => haveI := instance_; some
      { forward := fun xs => (Tensor.zipWith NextAfter.nextafter xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
    | none => none
  | _, _, .argmax dim keepdim h => some
      { forward := fun xs => (Tensor.argmax xs.1 dim keepdim h, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .argmin dim keepdim h => some
      { forward := fun xs => (Tensor.argmin xs.1 dim keepdim h, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .flip dims => some
      { forward := fun xs => (Tensor.flip xs.1 dims, ())
        backward := fun _ g => (Tensor.flip g.1 dims, ())
        jvp? := some fun _ dxs => (Tensor.flip dxs.1 dims, ()) }
  | _, _, .flip dims .bool | _, _, .flip dims (.integer _ _) => some
      { forward := fun xs => (Tensor.flip xs.1 dims, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .indices s dim => some
      { forward := fun _ => (Tensor.indices s dim, ())
        backward := fun _ _ => () }
  | _, _, .cast .bool => some
      { forward := fun xs => (Tensor.map (fun b => if b then (1 : α) else 0) xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, .cast (.index _) => some
      { forward := fun xs => (Tensor.map (fun j => ((j.val : Nat) : α)) xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, @Op.encodeIndex _ _ signed width => some
      { forward := fun xs => (Op.indexWords width xs.1, ())
        backward := fun _ _ => ((), ())
        validate := fun xs => Op.checkIndexWords signed width xs.1 }
  | _, _, @Op.decodeIndex s bounds signed _ =>
      if h : (∀ b ∈ bounds, 0 < b) ∨ s.size = 0 then some
        { forward := fun xs => (Op.wordIndicesOr h signed xs.1, ())
          backward := fun _ _ => ((), ())
          validate := fun xs => (Op.wordIndices signed xs.1).map fun _ => () }
      else none
  | _, _, .attention mask schedule => some
      { forward := fun xs =>
          (Tensor.attention xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item mask none schedule, ())
        backward := fun xs g =>
          let gradients :=
              Tensor.attentionBackward xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item mask none g.1
          (gradients.1, gradients.2.1, gradients.2.2.1, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.attentionJVP xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item mask none dxs.1 dxs.2.1
            dxs.2.2.1 none, ()) }
  | _, _, .reduce op dims keepdim h => some
      { forward := fun xs => (match op, h with
          | .sum, _ => Tensor.reduce .sum xs.1 dims keepdim
          | .prod, _ => Tensor.reduce .prod xs.1 dims keepdim
          | .max, h => Tensor.reduce .max xs.1 dims keepdim fun _ => h (Or.inl rfl)
          | .min, h => Tensor.reduce .min xs.1 dims keepdim fun _ => h (Or.inr rfl), ())
        backward := fun xs g => (Tensor.reduceBackward op xs.1 dims keepdim h g.1, ())
        jvp? := match op with
          | .sum => some fun _ dxs => (Tensor.reduce .sum dxs.1 dims keepdim, ())
          | _ => some fun xs dxs => (Tensor.reduceJVP _ xs.1 dims keepdim h dxs.1, ()) }
  | _, _, .mean dims keepdim => some
      { forward := fun xs => (Tensor.mean xs.1 dims keepdim, ())
        backward := fun _ g => (Tensor.meanBackward dims keepdim g.1, ())
        jvp? := some fun _ dxs => (Tensor.mean dxs.1 dims keepdim, ()) }
  | _, _, .contract dims batchDims h => some
      { forward := fun xs => (Tensor.contract xs.1 xs.2.1 dims batchDims _ h rfl, ())
        backward := fun xs g =>
          (Tensor.contractBackwardLeft dims batchDims h g.1 xs.2.1,
            Tensor.contractBackwardRight dims batchDims h xs.1 g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.contract dxs.1 xs.2.1 dims batchDims _ h rfl +
            Tensor.contract xs.1 dxs.2.1 dims batchDims _ h rfl, ()) }
  | _, _, .maskedSoftmax dim => some
      { forward := fun xs => (Tensor.softmax xs.1 dim (some xs.2.1), ())
        backward := fun xs g => (Tensor.maskedSoftmaxBackward xs.1 xs.2.1 g.1 dim, (), ())
        jvp? := some fun xs dxs => (Tensor.maskedSoftmaxBackward xs.1 xs.2.1 dxs.1 dim, ()) }
  | _, _, .slice dims starts stops steps h => some
      { forward := fun xs => (Tensor.slice xs.1 dims starts stops steps h, ())
        backward := fun _ g => (Tensor.sliceBackward dims starts stops steps g.1, ())
        jvp? := some fun _ dxs => (Tensor.slice dxs.1 dims starts stops steps h, ()) }
  | _, _, .slice dims starts stops steps h .bool
  | _, _, .slice dims starts stops steps h (.integer _ _) => some
      { forward := fun xs => (Tensor.slice xs.1 dims starts stops steps h, ())
        backward := fun _ _ => ((), ()) }
  | _, _, .pad (s := s) low high interior => some
      { forward := fun xs => (Tensor.pad xs.1 low high interior xs.2.1.item, ())
        backward := fun _ g =>
          (Tensor.padBackward low high interior g.1,
            Tensor.scalar (Tensor.sum (Tensor.select
              (Tensor.pad (Tensor.full s false) low high interior true) g.1 (Tensor.full _ 0))),
            ())
        jvp? := some fun _ dxs => (Tensor.pad dxs.1 low high interior dxs.2.1.item, ()) }
  | _, _, .pad low high interior .bool | _, _, .pad low high interior (.integer _ _) => some
      { forward := fun xs => (Tensor.pad xs.1 low high interior xs.2.1.item, ())
        backward := fun _ _ => ((), (), ()) }
  | _, _, .indexAt dim index => some
      { forward := fun xs => (Tensor.reshape
            (Tensor.slice xs.1 [dim] [index.val] [index.val + 1] [1]
              (Shape.sliceValid_singleton _ dim index.val 1 index.isLt)) _
            (by rw [Shape.slice_singleton]; exact Tensor.size_shapeAfterSumKeepDim _ dim), ())
        backward := fun _ g => (Tensor.sliceBackward [dim] [index.val] [index.val + 1] [1]
            (Tensor.reshape g.1 _ (by
              rw [Shape.slice_singleton]
              exact (Tensor.size_shapeAfterSumKeepDim _ dim).symm)), ())
        jvp? := some fun _ dxs => (Tensor.reshape
            (Tensor.slice dxs.1 [dim] [index.val] [index.val + 1] [1]
              (Shape.sliceValid_singleton _ dim index.val 1 index.isLt)) _
            (by rw [Shape.slice_singleton]; exact Tensor.size_shapeAfterSumKeepDim _ dim), ()) }
  | _, _, .cat dim sizes => some
      { forward := fun xs =>
          (Tensor.cat dim sizes.get (Op.catElementParts dim sizes xs) _ (by simp), ())
        backward := fun _ g => Op.catGrads dim g.1 0 sizes (by simp)
        jvp? := some fun _ dxs =>
          (Tensor.cat dim sizes.get (Op.catTangentParts dim sizes dxs) _ (by simp), ()) }
  | _, _, .cat dim sizes .bool | _, _, .cat dim sizes (.integer _ _) => some
      { forward := fun xs =>
          (Tensor.cat dim sizes.get (Op.catElementParts dim sizes xs) _ (by simp), ())
        backward := fun _ _ => Op.catUnits dim (fun _ => ()) sizes }
  | _, _, .scatter rule dims h => some
      { forward := fun xs => (Op.scatterForward rule dims xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun xs g =>
          let positions := Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1
          match rule with
          | .sum => (g.1, (),
              Tensor.scatterUpdatesBackward .sum dims h (Tensor.map (Op.axisTuples _ dims) xs.2.1)
                g.1, ())
          | .replace => (Tensor.scatter .replace g.1 dims positions (Tensor.full _ 0), (),
              Tensor.scatterUpdatesBackward .replace dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) g.1, ())
          | .prod =>
              let gradients := Tensor.scatterProdBackward xs.1 dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) xs.2.2.1 g.1
              (gradients.1, (), gradients.2, ())
          | .max =>
              let gradients := Tensor.scatterExtremumBackward
                (Tensor.scatter .max xs.1 dims positions xs.2.2.1) xs.1 dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) xs.2.2.1 g.1
              (gradients.1, (), gradients.2, ())
          | .min =>
              let gradients := Tensor.scatterExtremumBackward
                (Tensor.scatter .min xs.1 dims positions xs.2.2.1) xs.1 dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) xs.2.2.1 g.1
              (gradients.1, (), gradients.2, ())
        jvp? := match rule with
          | .sum => some fun xs dxs =>
              let positions := Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1
              (Tensor.scatter .sum dxs.1 dims positions dxs.2.2.1, ())
          | .replace => some fun xs dxs =>
              let positions := Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1
              (Tensor.scatter .replace dxs.1 dims positions dxs.2.2.1, ())
          | _ => none }
  | _, _, .scatter .replace dims _ .bool => some
      { forward := fun xs => (Tensor.scatter .replace xs.1 dims
          (Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1) xs.2.2.1, ())
        backward := fun _ _ => ((), (), (), ()) }
  | _, _, .scatter .sum _ _ .bool | _, _, .scatter .prod _ _ .bool
  | _, _, .scatter .max _ _ .bool | _, _, .scatter .min _ _ .bool => none
  | _, _, .scatter rule dims _ (.integer signed _) => some
      { forward := fun xs => (Op.scatterForward (maxStep := ⟨Word.maximum signed⟩)
          (minStep := ⟨Word.minimum signed⟩) rule dims xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun _ _ => ((), (), (), ()) }
  | _, _, .topKMask dim k => some
      { forward := fun xs => (MoE.topKMask k xs.1 dim, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, .bernoulliMask key => some
      { forward := fun xs => (Random.mask key xs.1.item, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) }
  | _, _, .rfft dim n _ => some
      { forward := fun xs => (Tensor.rfft dim n xs.1, ())
        backward := fun _ g => (Tensor.rfftBackward dim n g.1, ())
        jvp? := some fun _ dxs => (Tensor.rfft dim n dxs.1, ()) }
  | _, _, .irfft dim n _ => some
      { forward := fun xs => (Tensor.irfft dim n xs.1, ())
        backward := fun _ g => (Tensor.irfftBackward dim n g.1, ())
        jvp? := some fun _ dxs => (Tensor.irfft dim n dxs.1, ()) }
  | _, _, .selectiveScan dim sharedA sharedB => some
      { forward := fun xs =>
          (Tensor.selectiveScan dim sharedA sharedB xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1, ())
        backward := fun xs g =>
          let gradients :=
            Tensor.selectiveScanBackward dim sharedA sharedB xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1 g.1
          (gradients.1, gradients.2.1, gradients.2.2.1, gradients.2.2.2, ())
        jvp? := some fun xs dxs =>
          (Tensor.selectiveScanJVP dim sharedA sharedB xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1 dxs.1
            dxs.2.1 dxs.2.2.1 dxs.2.2.2.1, ()) }
  | _, _, .custom .. => none

end TorchLean
