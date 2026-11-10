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
import shutil
import tempfile

from ._checker import Checker

DTYPES = ("float16", "bfloat16", "float32", "float64", "bool", "int8", "int16", "int32",
          "int64", "uint8", "uint16", "uint32", "uint64")

_KEYS = ("format", "version", "dtypes", "codes", "pointwise")

# Cache by the resolved checker choice and optional table file. Environment overrides can change
# during a process, so a default argument alone does not identify the source of a table.
_TABLES: dict[tuple, dict] = {}


def _file_key(path: str | Path) -> tuple:
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path),)
    return (str(path), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def _key(checker: str | Path | Sequence[str] | None) -> tuple:
    # `Checker` raises TypeError for a malformed command, so only a valid command forms a key.
    if checker is None:
        return (None, _file_key(Path(__file__).resolve().parents[2] / ".lake/build/bin/verify"),
                ("ir",))
    if isinstance(checker, (str, Path)):
        command = (str(checker), "ir")
    else:
        Checker(checker)
        command = tuple(checker)
    return (_file_key(shutil.which(command[0]) or command[0]), command[1:])


def op_table(checker: str | Path | Sequence[str] | None = None) -> dict:
    """Return the parsed op table for `DTYPES`, cached until its source changes.

    The table is the file named by `TORCHLEAN_OP_TABLE` when that variable is set, else the one
    `Checker(checker).op_table` returns, `checker` as in `torchlean.verify`: None finds
    `TORCHLEAN_VERIFY`, else the native `_torchlean.op_table`, else the local `verify` executable.
    """
    path = os.environ.get("TORCHLEAN_OP_TABLE")
    selected = checker if checker is not None else os.environ.get("TORCHLEAN_VERIFY")
    key = (_key(selected), _file_key(path) if path else None)
    table = _TABLES.get(key)
    if table is None:
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
