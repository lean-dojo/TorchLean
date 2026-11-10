"""Lean's operator table, the codec facts and pointwise forms the Python captures read from Lean.

`op_table(checker)` returns the JSON of `IR.table` for the 13 dtypes in `DTYPES`: the
artifact format name, the codec record version, each dtype's JSON and the typed forms of every
pointwise operator. It also holds the codes the codec writes for division rounding, GELU
approximation, reductions, pools, scatters, square root policies, bitcast words and cast sources.
A capture reads it from the caller's checker, so a `checker=` argument supplies the table too.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import tempfile

from ._checker import Checker

DTYPES = ("float16", "bfloat16", "float32", "float64", "bool", "int8", "int16", "int32",
          "int64", "uint8", "uint16", "uint32", "uint64")

_KEYS = ("format", "version", "dtypes", "codes", "pointwise")

# The parsed tables, one per checker: None, an executable path as a string, or the tuple of a
# command's arguments.
_TABLES: dict[str | tuple[str, ...] | None, dict] = {}


def _key(checker: str | Path | Sequence[str] | None) -> str | tuple[str, ...] | None:
    # `Checker` raises TypeError for a malformed command, so only a valid command forms a key.
    if checker is None or isinstance(checker, (str, Path)):
        return None if checker is None else str(checker)
    Checker(checker)
    return tuple(checker)


def op_table(checker: str | Path | Sequence[str] | None = None) -> dict:
    """Return the parsed op table for `DTYPES`, read once per process for each `checker`.

    The table is the file named by `TORCHLEAN_OP_TABLE` when that variable is set, else the one
    `Checker(checker).op_table` returns, `checker` as in `torchlean.verify`: None finds
    `TORCHLEAN_VERIFY`, else the native `_torchlean.op_table`, else the local `verify` executable.
    """
    key = _key(checker)
    table = _TABLES.get(key)
    if table is None:
        path = os.environ.get("TORCHLEAN_OP_TABLE")
        if path:
            text = Path(path).read_text(encoding="utf-8")
        else:
            with tempfile.TemporaryDirectory(prefix="torchlean-table-") as temporary:
                text = Checker(checker).op_table(DTYPES, Path(temporary))
        table = json.loads(text)
        if not isinstance(table, dict) or any(name not in table for name in _KEYS):
            raise RuntimeError("the checker did not produce a TorchLean op table")
        _TABLES[key] = table
    return table
