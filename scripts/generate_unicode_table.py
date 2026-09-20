#!/usr/bin/env python3
"""Generate the Unicode property table used by tensor-pattern syntax parsing."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path


EXPECTED_PYTHON = (3, 12, 13)
EXPECTED_UNICODE = "15.0.0"
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "NN"
    / "Tensor"
    / "Internal"
    / "Syntax"
    / "UnicodeData.lean"
)

ALPHANUMERIC = 1
IDENTIFIER_START = 2
IDENTIFIER_CONTINUE = 4
DECIMAL = 8
LINE_WIDTH = 100
PART_SIZE = 400


def check_runtime() -> None:
    """Reject runtimes whose character predicates would generate another table."""
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON:
        raise SystemExit(
            "Unicode generation requires CPython "
            f"{'.'.join(map(str, EXPECTED_PYTHON))}, found "
            f"{'.'.join(map(str, actual_python))}"
        )
    if unicodedata.unidata_version != EXPECTED_UNICODE:
        raise SystemExit(
            f"Unicode generation requires {EXPECTED_UNICODE}, found "
            f"{unicodedata.unidata_version}"
        )


def character_flags(codepoint: int) -> int:
    """Encode the parser-relevant Boolean properties of one code point."""
    character = chr(codepoint)
    flags = 0
    if character.isalnum():
        flags |= ALPHANUMERIC
    if character.isidentifier():
        flags |= IDENTIFIER_START
    if ("a" + character).isidentifier():
        flags |= IDENTIFIER_CONTINUE
    if unicodedata.decimal(character, None) is not None:
        flags |= DECIMAL
    return flags


def character_payload(codepoint: int) -> int:
    """Encode the flags together with the decimal zero for interval splitting.

    Decimal digits with different zero characters must land in different
    intervals so that the digit value is the offset from the interval start.
    """
    flags = character_flags(codepoint)
    decimal_value = unicodedata.decimal(chr(codepoint), None)
    if decimal_value is None:
        return flags
    return flags | 16 * (codepoint - decimal_value + 1)


def property_intervals() -> list[tuple[int, int, int]]:
    """Coalesce adjacent code points with equal nonzero flags into intervals."""
    intervals: list[tuple[int, int, int]] = []
    interval_start = 0
    current_payload = character_payload(0)

    for codepoint in range(1, sys.maxunicode + 2):
        next_payload = (
            character_payload(codepoint)
            if codepoint <= sys.maxunicode
            else -1
        )
        if next_payload == current_payload:
            continue
        if current_payload != 0:
            intervals.append(
                (interval_start, codepoint - 1, current_payload % 16)
            )
        interval_start = codepoint
        current_payload = next_payload

    for start, stop, flags in intervals:
        if flags & DECIMAL:
            assert stop - start == 9, (start, stop)
            assert unicodedata.decimal(chr(start)) == 0, start
    return intervals


def render_entries(entries: list[str]) -> str:
    """Render list entries, packing as many per line as fit."""
    lines: list[str] = []
    current = "  "
    for index, entry in enumerate(entries):
        piece = entry + ("," if index + 1 < len(entries) else "")
        candidate = current + (" " if current.strip() else "") + piece
        if len(candidate) > LINE_WIDTH:
            lines.append(current)
            current = "  " + piece
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def render_table(intervals: list[tuple[int, int, int]]) -> str:
    """Render the interval table as a few list literals joined into one array.

    A single literal with every interval exceeds the elaborator's recursion
    limit, so the table is split into parts of at most `PART_SIZE` entries.
    """
    entries = [
        f"⟨0x{start:X}, 0x{stop:X}, {flags}⟩" for start, stop, flags in intervals
    ]
    parts = [
        entries[index : index + PART_SIZE]
        for index in range(0, len(entries), PART_SIZE)
    ]
    blocks: list[str] = []
    for index, part in enumerate(parts):
        blocks.append(
            f"/-- Part {index} of `intervals`, split to stay within elaborator limits. -/\n"
            f"def intervalsPart{index} : List Interval := [\n{render_entries(part)}]\n"
        )
    joined = " ++ ".join(f"intervalsPart{index}" for index in range(len(parts)))
    blocks.append(
        f"/-- The {len(intervals)} property runs of Unicode {EXPECTED_UNICODE}, "
        "ascending and disjoint. -/\n"
        f"def intervals : Array Interval :=\n  Array.mk ({joined})\n"
    )
    return "\n".join(blocks)


def render(intervals: list[tuple[int, int, int]]) -> str:
    """Render the deterministic Lean module."""
    table = render_table(intervals)
    return f"""\
/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

/-!
# Generated Python Unicode properties

This file is generated by `scripts/generate_unicode_table.py`; do not edit it by hand. It
freezes the CPython 3.12.13 character predicates for Unicode 15.0.0 that the einops lexer
reproduces.

The table lists the maximal runs of code points sharing the same nonzero property flags, in
ascending order and pairwise disjoint. Bit 0 records `str.isalnum`, bit 1 records
`str.isidentifier` on the single character, bit 2 records whether the character may continue
an identifier, and bit 3 records `str.isdecimal`. Every decimal run covers exactly the ten
digits of one decimal alphabet in order, so the digit value of a decimal character is its
offset from the start of its run.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax.PythonUnicode

/-- A maximal run of code points sharing the same Python property flags. -/
structure Interval where
  /-- The first code point of the run. -/
  lo : UInt32
  /-- The last code point of the run. -/
  hi : UInt32
  /-- The property flags shared by every code point of the run. -/
  flags : UInt8
deriving Repr, DecidableEq

/-- Flag bit recording `str.isalnum`. -/
def alphanumericFlag : UInt8 := {ALPHANUMERIC}

/-- Flag bit recording that the character may start an identifier. -/
def identifierStartFlag : UInt8 := {IDENTIFIER_START}

/-- Flag bit recording that the character may continue an identifier. -/
def identifierContinueFlag : UInt8 := {IDENTIFIER_CONTINUE}

/-- Flag bit recording `str.isdecimal`. -/
def decimalFlag : UInt8 := {DECIMAL}

{table}
/--
Binary search for the run containing `codepoint` within the index range `[lo, hi)` of
`intervals`. The `fuel` argument bounds the recursion so the definition is structural and
reduces inside the kernel.
-/
def search (codepoint : UInt32) : (fuel lo hi : Nat) → Option Interval
  | 0, _, _ => none
  | fuel + 1, lo, hi =>
      if lo < hi then
        let mid := (lo + hi) / 2
        match intervals[mid]? with
        | none => none
        | some interval =>
            if codepoint < interval.lo then
              search codepoint fuel lo mid
            else if interval.hi < codepoint then
              search codepoint fuel (mid + 1) hi
            else
              some interval
      else
        none

/-- The property run containing a character, if it has any recorded property. -/
def lookup? (char : Char) : Option Interval :=
  search char.val intervals.size 0 intervals.size

/-- Whether a character carries the given property flag. -/
def hasFlag (char : Char) (flag : UInt8) : Bool :=
  match lookup? char with
  | none => false
  | some interval => interval.flags &&& flag != 0

/-- The digit value of a `str.isdecimal` character. -/
def decimalValue? (char : Char) : Option Nat :=
  match lookup? char with
  | none => none
  | some interval =>
      if interval.flags &&& decimalFlag != 0 then
        some (char.toNat - interval.lo.toNat)
      else
        none

end TorchLean.Tensor.Internal.Syntax.PythonUnicode
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the checked-in table differs",
    )
    arguments = parser.parse_args()

    check_runtime()
    intervals = property_intervals()
    generated = render(intervals)

    if arguments.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"{OUTPUT} is not the generated Unicode table")
        print(f"checked {len(intervals)} Unicode {EXPECTED_UNICODE} intervals")
        return

    OUTPUT.write_text(generated, encoding="utf-8")
    print(f"wrote {len(intervals)} intervals to {OUTPUT}")


if __name__ == "__main__":
    main()
