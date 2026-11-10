# Recursion over trees on the GPU

This directory runs recursive functions over inductive types as data-parallel passes over the
levels of a tree, on the GPU through LibTorch or on the host. Its declarations are in the
namespace `TorchLean.Kernel`.

- `Sig.lean`: signatures of inductive families, whose fields are scalars, `UInt64` indices,
  children and lists of children, and the class `Sig.Inductive`.
- `Derive.lean`: the deriving handler of `Sig.Inductive`, for one type or a mutual block.
- `Arena.lean`: leveled arenas, in which level `r + 1` holds the children of the rows of level
  `r`, with `Arena.encode`, `Arena.decode` and `Arena.decode_encode`.
- `Device.lean`: arenas in device memory, with `Arena.upload` and `Arena.Device.download`.
- `Word.lean`: `FloatBits` and `Float32Bits`, binary64 and binary32 scalars held as their
  encodings, so that arenas keep NaN payloads.
- `Fold.lean`: folds, which compute the levels from the deepest up, with the certificate
  `Fold.correct` and `Fold.eval_encode`.
- `Elab.lean`: `Fold.of`, which builds a fold and its certificate from a Lean function.
- `Row.lean`: row kernels, the CUDA source that computes one expression for every row.
- `Run.lean`: the runs of folds on the device, `Step.runDevice` and `Fold.run`.
- `Expand.lean`: expansions of calls, one level per depth, placed by prefix sums.
- `Hylo.lean`: hylomorphisms, recursive functions whose calls need not follow a value.
- `Unfold.lean`: unfolds, which build the arena of a tree from seeds, in device memory as well.

## Folding a tree

With `TorchLean.Kernel` open, `deriving Sig.Inductive` gives `BTree` its signature `BTree.sig`, and
`Fold.of` turns a structurally recursive function into a fold whose certificate Lean's kernel
checks:

```lean
import TorchLean.CUDA.Tree.Elab
import TorchLean.CUDA.Tree.Run
open TorchLean.Kernel

inductive BTree where
  | leaf
  | node (l : BTree) (x : Float) (r : BTree)
  deriving Sig.Inductive

def BTree.sum : BTree → Float
  | .leaf => 0
  | .node l x r => l.sum + x + r.sum

def sumFold := Fold.of BTree.sum

def BTree.arena (t : BTree) : Arena Float :=
  Arena.encode (σ := BTree.sig) (T := fun _ => BTree) (s := 0) t

def small : BTree := .node (.node .leaf 1 .leaf) 2 (.node .leaf 3 .leaf)

#eval sumFold.eval (Reader.ofArrays #[]) small.arena  -- Except.ok 6.000000

theorem sumFold_eval (t : BTree)
    (h : Sig.Inductive.count (σ := BTree.sig) (T := fun _ => BTree) (s := 0) t < UInt64.size) :
    sumFold.eval (Reader.ofArrays #[]) t.arena = .ok t.sum :=
  sumFold.eval_encode _ t h

def sumOnGpu : IO (Array (Except Error Float)) := do
  sumFold.run (← Arena.upload BTree.sig small.arena) #[]
```

`Fold.of` accepts the forms `T → α`, `T → Except Error α` and `Reader α → T → Except Error α`,
for a type of one sort whose scalars are `Float`, `Float32` or a configured binary format. It
rejects a function it cannot compile with a message that names the term. Types with list fields
and mutual blocks get their folds by hand, with `Fold.select` and `Fold.correct_select`, as in
`TorchLeanExamples/Tree/Folds.lean`.

`Fold.run` folds an arena in device memory and computes on the host from the downloaded arena
when a read fails there. `Hylo.run` and `Unfold.run` use the device when LibTorch and a device
are present, and the host otherwise. `Unfold.buildDevice` leaves the arena it builds in device
memory, where `Fold.run` folds it. In builds without LibTorch, uploads fail with an `IO` error.
Tables of correctly rounded functions inside kernels are in
[`Kernel/Table.lean`](../Kernel/README.md#tables).

## What is proved

- `Arena.decode_encode`: every value with fewer than `2 ^ 64` nodes decodes from its arena, and
  `Arena.decode_ofBlocks_encode` states the same through the byte blocks of the device.
- `Fold.eval_encode` and `Step.schedule_eq`: the level schedule of a fold gives its Lean function,
  read failures included.
- `denotes_emitRow` and `Step.denotes_source`: the CUDA text of a row kernel computes its
  expression for the native formats.
- `Hylo.results_eq` and `Unfold.results_eq` for every capacity, `Unfold.build_eq` for the arena
  that an unfold builds, and `Unfold.results_fold` for a fold over it. These certificates give
  partial correctness: they do not show that an expansion finishes, and a run past its capacity
  computes on the host.

## What is trusted

The device run is trusted: NVRTC, the launches of `csrc/libtorch/tree.cpp` and their contract, the
bytes, and the agreement of the device arithmetic with `Scalar.arithmetic`, which the class
`Scalar` does not state. The CUDA text of the configured formats (`emitRowBinary`) is tested and
has no proof. The tests compare every device result with the Lean function bit for bit.

## Tests and benchmarks

`scripts/lake.sh exe tree_tests` runs the host checks of `TorchLeanTest/Tree`, then the device
checks in a `-Kcuda=true` build with a usable device. With `TORCHLEAN_REQUIRE_CUDA=1` a missing
device is an error. `scripts/lake.sh test` runs the host checks too.
`scripts/lake.sh exe torchlean tree_bench [--log2 K,...] [--device auto|cpu|cuda]` times the
stages of an arena and the folds, hylomorphisms, unfolds and tables of `TorchLeanExamples/Tree`,
on the host by default.
