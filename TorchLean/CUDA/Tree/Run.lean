/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Device
public import TorchLean.CUDA.Tree.Fold
public import TorchLean.CUDA.Tree.Row
public import TorchLean.Tensor.List

/-!
# Folds on the device

The device runs a fold one level at a time, from the deepest level up. Each level is one launch
of a row kernel (`CUDA.emitRow`): thread `k` reads row `k` of the level from the columns of the
arena and the results of the next level as the input after the step's own, and writes the result
of the row. `Step.schedule` is this computation in Lean. It succeeds when every row of every
level succeeds, and `Step.schedule_eq` proves that it then gives the results of `Step.results`,
all of them successes, so each root it returns is the reference result of its value
(`Fold.schedule_decodeAt`). `Step.denotes_level` proves that the native source of a level computes
the rows of `Step.level` from the results below, failed reads included, and `Step.denotes_source`
that `Step.source` renders that source for native scalars whose words are the bits of their
literals, as they are for `Float` and `Float32` (`Scalar.bits_float`, `Scalar.bits_float32`).

Three parts are trusted and tested rather than proved. The step runs with the arithmetic of the
`Scalar` instance (`Step.native`), and that this is the arithmetic of the GPU is the assumption of
every kernel of `Kernel.Launch`. The launch contract of `tree.cpp` (one launch per level, `count`
the size of the level, its full columns, and the results of the level below as the last input) is
checked by the tests. Configured formats render through `emitRowBinary`, which, like upstream
`emitBinary`, is tested only.

`Step.runDevice` launches the levels through the private `@[extern]` entry point
`Arena.Handle.fold` of `csrc/libtorch/tree.cpp`, trusted native code like the kernel launchers of
`Kernel.Launch`, and private like them, so that only kernels generated here reach it.
`Step.runHandle` does the same for any native handle with its level sizes, which is how the levels
that a hylomorphism builds on the device are folded. A read that fails on the device sets the error
record, every later thread returns at once, and `Fold.run` then computes the results on the host
from the downloaded arena, so a failure comes back as the error of `Step.results`.

The lemmas on optional `mapM` give each entry of a successful result at its position, the converse
of `List.mapM_some_of_getElem`.
-/

@[expose] public section

namespace List

/-- A successful optional `mapM` gives each entry of its result at its position. -/
theorem getElem?_of_mapM_some {β γ : Type} {f : β → Option γ} :
    ∀ {l : List β} {l' : List γ}, l.mapM f = some l' → ∀ m (hm : m < l.length), l'[m]? = f l[m]
  | [], _, _, _, hm => absurd hm (Nat.not_lt_zero _)
  | x :: xs, l', h, m, hm => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, pure,
        Option.some.injEq] at h
      obtain ⟨y, hy, ys, hys, rfl⟩ := h
      cases m with
      | zero => simp [hy]
      | succ m => simpa using getElem?_of_mapM_some hys m (by simpa using hm)

end List

namespace Array

/-- A successful optional `mapM` over an array keeps the size. -/
theorem size_of_mapM_some {β γ : Type} {f : β → Option γ} {xs : Array β} {ys : Array γ}
    (h : xs.mapM f = some ys) : ys.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => simp [hl] at h
  | some l =>
    simp [hl] at h
    subst h
    simpa using List.mapM_some_length hl

/-- A successful optional `mapM` over an array gives each entry of its result at its position. -/
theorem getElem?_of_mapM_some {β γ : Type} {f : β → Option γ} {xs : Array β} {ys : Array γ}
    (h : xs.mapM f = some ys) (k : Nat) (hk : k < xs.size) : ys[k]? = f xs[k] := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => simp [hl] at h
  | some l =>
    simp [hl] at h
    subst h
    rw [List.getElem?_toArray]
    exact List.getElem?_of_mapM_some hl k (by simpa using hk)

end Array

namespace Except

/-- The option of an `Except` value is `some y` exactly when the value is the success `y`. -/
theorem toOption_eq_some_iff {ε β : Type} {x : Except ε β} {y : β} :
    x.toOption = some y ↔ x = .ok y := by
  cases x <;> simp [Except.toOption]

end Except

namespace TorchLean.Kernel

open CUDA

variable {α : Type}

/-- The reader of the arrays `inputs`: input `op` at index `i` is entry `i` of `inputs[op]`. A
read past the end of an array, or of an input that is not there, fails. -/
def Reader.ofArrays [Inhabited α] (inputs : Array (Array α)) : Reader α := fun op i =>
  let size := ((inputs[op]?.map Array.size).getD 0).toUInt64
  if i ≥ size then .error (.bounds op i size.toNat)
  else .ok ((inputs[op]?.bind (·[i.toNat]?)).getD default)

/-- The words of `Float` scalars are the bits of their binary64 literals. -/
theorem Scalar.bits_float (x : Float) :
    (instScalarFloat.toBits x).toNat.toUInt64 = (Literal.ofValue .binary64 x).bits :=
  UInt64.ofNat_toNat

/-- The words of `Float32` scalars are the bits of their binary32 literals. -/
theorem Scalar.bits_float32 (x : Float32) :
    (instScalarFloat32.toBits x).toNat.toUInt64 = (Literal.ofValue .binary32 x).bits := by
  apply UInt64.toNat.inj
  simp [Literal.ofValue, Literal.bits, Numeric.Word.toBits]

namespace Step

section

variable {m s : Nat} [Inhabited α]

/-- A level has one result per row. -/
@[simp] theorem size_level (S : Step α m s) (read : Reader α) (level : Arena.Level α)
    (below : Array (Except Error α)) : (S.level read level below).size = level.tag.size :=
  Array.size_ofFn

/-- Row `k` of a level computes its result from the variables of the row. -/
theorem getElem_level (S : Step α m s) (read : Reader α) (level : Arena.Level α)
    (below : Array (Except Error α)) {k : Nat} (hk : k < (S.level read level below).size) :
    (S.level read level below)[k] =
      S.expr.eval S.arithmetic (reader read S.inputs below) (level.env m s k) := by
  have hk' : k < level.tag.size := by simpa using hk
  simp [Step.level, Arena.Level.env, hk']

/-- The device schedule of a fold: the results of the levels from the deepest up, each level from
the results of the next, and no results at all once a row fails. -/
def schedule (S : Step α m s) (read : Reader α) (levels : List (Arena.Level α)) :
    Option (Array α) :=
  levels.foldr (fun level below => below.bind fun below =>
    (S.level read level (below.map .ok)).mapM Except.toOption) (some #[])

private theorem map_ok_of_mapM {ε β : Type} {xs : Array (Except ε β)} {vs : Array β}
    (h : xs.mapM Except.toOption = some vs) : vs.map .ok = xs := by
  have hs := Array.size_of_mapM_some h
  apply Array.ext (by simp [hs])
  intro k _ hk
  have hv := Array.getElem?_of_mapM_some h k hk
  rw [Array.getElem?_eq_getElem (by omega)] at hv
  simp [Except.toOption_eq_some_iff.1 hv.symm]

/-- A schedule that succeeds gives the results of the levels, all of them successes. -/
theorem schedule_eq (S : Step α m s) (read : Reader α) :
    ∀ (levels : List (Arena.Level α)) {vs : Array α}, S.schedule read levels = some vs →
      vs.map .ok = levels.foldr (S.level read) #[]
  | [], vs, h => by simp [schedule] at h; simp [← h]
  | level :: levels, vs, h => by
    simp only [schedule, List.foldr_cons] at h
    obtain ⟨below, hb, hvs⟩ := Option.bind_eq_some_iff.1 h
    rw [List.foldr_cons, ← schedule_eq S read levels hb]
    exact map_ok_of_mapM hvs

end

/-- The native source of a level computes the rows of `Step.level`. Thread `index` of a launch
over the columns of `level`, whose last input holds the results `below` of the next level, gives
the result of row `index`, failed reads included. -/
theorem denotes_level {format : Format} {m s : Nat} [Inhabited format.Value]
    (S : Step format.Value m s)
    (read : Reader format.Value) (size : Nat → UInt64) (load : Nat → UInt64 → format.Value)
    (hRead : ∀ op, op < S.inputs → ∀ i, read op i = if i ≥ size op then
      .error (.bounds op i (size op).toNat) else .ok (load op i))
    (level : Arena.Level format.Value) (below : Array format.Value)
    (hBelow : below.size < UInt64.size) {generated : Source}
    (hSource : emitRow format (fun x => (Literal.ofValue format x).bits) (S.inputs + 1) S.expr =
      .ok generated) :
    RowSource.Denotes format S.arithmetic (S.inputs + 1)
      (fun op => if op = S.inputs then below.size.toUInt64 else size op)
      (fun op i => if op = S.inputs then below[i.toNat]?.getD default else load op i)
      (fun index => level.env m s index.toNat) generated.text
      (fun recorded block blockSize thread count _ =>
        let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
        if index ≥ count then some (.ok none)
        else if recorded then some (.ok none)
        else some ((S.expr.eval S.arithmetic (reader read S.inputs (below.map .ok))
          (level.env m s index.toNat)).map some)) := by
  refine denotes_emitRow format S.arithmetic (reader read S.inputs (below.map .ok))
    (S.inputs + 1) _ _ (fun op hop i => ?_) (fun index => level.env m s index.toNat) S.expr hSource
  have hs : below.size.toUInt64.toNat = below.size := UInt64.toNat_ofNat_of_lt' hBelow
  by_cases h : op = S.inputs
  · subst h
    by_cases hi : i ≥ below.size.toUInt64
    · have hle : below.size ≤ i.toNat := by
        rw [← hs]
        exact UInt64.le_iff_toNat_le.1 hi
      simp [reader, hi, hs, hle]
    · have hlt : i.toNat < below.size := by
        rw [← hs]
        exact UInt64.lt_iff_toNat_lt.1 (UInt64.not_le.1 hi)
      simp [reader, hi, hlt]
  · simp [reader, h, hRead op (by omega) i]

end Step

/-- The levels of a native handle folded from the deepest up by the row kernel `source`, one
launch per level over its rows, with the blocks `inputs` as the kernel's first inputs and the
results of the next level as its last. Returns the results of level zero as words of `width`
bytes, or `none` when a read failed on the device. The runtime reads the handle with the checks of
`Arena.Handle.words`, and checks its slots and that the level sizes `sizes` add up to its rows. -/
@[never_extract, extern "torchlean_tree_fold"]
private opaque Arena.Handle.fold (handle : @& Arena.Handle) (format : UInt32)
    (descriptor : @& String) (width indexSlots scalarSlots : UInt64) (source : @& String)
    (sizes : @& Array UInt64) (inputs : @& Array ByteArray) : IO (Option ByteArray)

namespace Step

variable {m s : Nat} [Inhabited α]

/-- The source of the row kernel of `e` with `inputs` inputs, for the precision of the scalars
`scalar`: thread `index` reads variable `p` from column `p` and writes `e` to `output[index]`. -/
@[no_expose] def rowSource (scalar : Scalar α) (inputs : Nat) {Γ : List Ty} {t : Ty}
    (e : Expr α Γ t) : Except String String :=
  match scalar.precision with
  | .native format => do
    let generated ← emitRow format (fun x => (scalar.toBits x).toNat.toUInt64) inputs e
    return generated.text
  | .binary format => emitRowBinary format (fun x => (scalar.toBits x).toNat) inputs e

/-- The source of the row kernel of a step for the precision of its scalars, with the results of
the next level as the input after the step's own. -/
@[no_expose] def source (S : Step α m s) : Except String String :=
  rowSource S.scalar (S.inputs + 1) S.expr

/-- For native scalars whose words are the bits of their literals, the text of `Step.source` is
the row kernel of `Step.denotes_level`: it computes the rows of a level from the results below,
failed reads included. -/
theorem denotes_source {format : Format} {m s : Nat} [Inhabited format.Value]
    (S : Step format.Value m s)
    (hNative : S.scalar.precision = .native format)
    (hBits : ∀ x, (S.scalar.toBits x).toNat.toUInt64 = (Literal.ofValue format x).bits)
    {text : String} (hSource : S.source = .ok text)
    (read : Reader format.Value) (size : Nat → UInt64) (load : Nat → UInt64 → format.Value)
    (hRead : ∀ op, op < S.inputs → ∀ i, read op i = if i ≥ size op then
      .error (.bounds op i (size op).toNat) else .ok (load op i))
    (level : Arena.Level format.Value) (below : Array format.Value)
    (hBelow : below.size < UInt64.size) :
    RowSource.Denotes format S.arithmetic (S.inputs + 1)
      (fun op => if op = S.inputs then below.size.toUInt64 else size op)
      (fun op i => if op = S.inputs then below[i.toNat]?.getD default else load op i)
      (fun index => level.env m s index.toNat) text
      (fun recorded block blockSize thread count _ =>
        let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
        if index ≥ count then some (.ok none)
        else if recorded then some (.ok none)
        else some ((S.expr.eval S.arithmetic (reader read S.inputs (below.map .ok))
          (level.env m s index.toNat)).map some)) := by
  unfold source rowSource at hSource
  rw [hNative] at hSource
  simp only [hBits] at hSource
  cases hEmit : emitRow format (fun x => (Literal.ofValue format x).bits) (S.inputs + 1)
      S.expr with
  | error message => simp [hEmit, bind, Except.bind] at hSource
  | ok generated =>
      simp [hEmit, bind, Except.bind, pure, Except.pure] at hSource
      subst hSource
      exact denotes_level S read size load hRead level below hBelow hEmit

omit [Inhabited α] in
/-- The blocks of the first `count` arrays of `inputs` for the native entry points, each the words
of its scalars in little-endian words of `Precision.bytes` bytes, and an empty block past the end
of `inputs`. -/
def inputBlocks (scalar : Scalar α) (count : Nat) (inputs : Array (Array α)) : Array ByteArray :=
  (Array.range count).map fun op =>
    Bytes.pushWords .empty scalar.precision.bytes (fun x => (scalar.toBits x).toNat)
      (inputs[op]?.getD #[])

/-- The results of level zero of the native handle `handle`, whose levels have the sizes `sizes`,
under a step: one launch of `Step.source` per level from the deepest up, with the blocks `blocks`
of `Step.inputBlocks` as the step's inputs. Returns `none` when a read failed on the device. Fails
when the source does not render or compile, or when the handle has other slots than the step or
other rows than the sizes add up to. -/
@[no_expose] def runBlocks (S : Step α m s) (handle : Arena.Handle) (sizes : Array Nat)
    (blocks : Array ByteArray) : IO (Option (Array α)) :=
  letI := S.scalar
  do
    let source ← IO.ofExcept S.source
    let precision := S.scalar.precision
    let width := precision.bytes
    let some bytes ← handle.fold precision.tag precision.descriptor width.toUInt64
        m.toUInt64 s.toUInt64 source (sizes.map (·.toUInt64)) blocks
      | return none
    let roots := sizes[0]?.getD 0
    unless bytes.size = width * roots do
      throw <| IO.userError "tree fold: the device returned results of the wrong size"
    return some <| Array.ofFn fun k : Fin roots =>
      S.scalar.ofNatBits (Bytes.wordFromLE bytes (width * k) width)

/-- The results of level zero of the native handle `handle`, whose levels have the sizes `sizes`,
under a step: one launch of `Step.source` per level from the deepest up, with the arrays `inputs`
as the step's inputs (`Step.runBlocks`). Returns `none` when a read failed on the device. Fails
when the source does not render or compile, or when the handle has other slots than the step or
other rows than the sizes add up to. -/
@[no_expose] def runHandle (S : Step α m s) (handle : Arena.Handle) (sizes : Array Nat)
    (inputs : Array (Array α)) : IO (Option (Array α)) :=
  S.runBlocks handle sizes (inputBlocks S.scalar S.inputs inputs)

/-- The results of level zero of a device arena under a step, computed on the device by one launch
of `Step.source` per level from the deepest up, with the arrays `inputs` as the step's inputs
(`Step.runHandle`). Returns `none` when a read failed on the device. Fails when the source does
not render or compile, or when the arena has other numbers of slots than the step. -/
@[no_expose] def runDevice (S : Step α m s) (d : @Arena.Device α S.scalar)
    (inputs : Array (Array α)) : IO (Option (Array α)) :=
  letI := S.scalar
  do
    unless d.indexSlots = m ∧ d.scalarSlots = s do
      throw <| IO.userError "tree fold: the arena has other slots than the step"
    S.runHandle d.handle d.sizes inputs

end Step

namespace Fold

variable {σ : Sig} {T : Fin σ.sorts → Type} [σ.Inductive α T] [Inhabited α]

/-- When the schedule of an arena succeeds, each root that decodes has the reference result of
its value, which is then a success. -/
theorem schedule_decodeAt (F : Fold σ α T) (read : Reader α) {a : Arena α} {vs : Array α}
    (h : F.schedule read a.levels.toList = some vs) {fuel k : Nat} {s : Fin σ.sorts} {y : T s}
    (hy : a.decodeAt T fuel 0 k s = some y) :
    (vs.map .ok)[k]? = some (F.reference read s y) := by
  rw [F.schedule_eq read _ h]
  simpa [Step.results] using F.results_of_decodeAt read a hy

/-- The results of the roots of a device arena under a fold, with the arrays `inputs` as the
fold's inputs. They come from the device (`Step.runDevice`), or, when a read failed there, from
`Step.results` on the downloaded arena, which gives each failure as the error of its row. -/
@[no_expose] def run (F : Fold σ α T) (d : @Arena.Device α F.scalar) (inputs : Array (Array α)) :
    IO (Array (Except Error α)) := do
  if let some vs ← F.runDevice d inputs then
    return vs.map .ok
  let a ← @Arena.Device.download α F.scalar d
  return F.results (Reader.ofArrays inputs) a 0

end Fold

end TorchLean.Kernel
