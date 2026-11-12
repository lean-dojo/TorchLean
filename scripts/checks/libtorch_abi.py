#!/usr/bin/env python3
"""Check the ATen operators and torch C functions the LibTorch bridge uses against TRUST.

The bridge in csrc/libtorch calls ATen operators only as `aten<"name.overload">(...)`, through the
stable dispatcher of `stable.h`, which reads each operator's schema from the generated table
`schemas.h`. Without options the command checks the sources: their `aten` keys are the table's
keys, the table is the list under "Dispatcher Schemas" in docs/TRUST_BOUNDARIES.md, the one call
of the dispatcher is `stable.h`'s, with the names of an `aten` key, every torch C shim they call is
in that file's list of "C Shims", and they include from torch only the stable headers
(`unstable.cpp`, the one source compiled against torch's C++ API, is checked through the library
instead).

`--torch DIR` also compares each table entry, and each schema under "Operators Behind C Shims", with
the schema that the torch package at DIR declares in its generated operator headers
(`include/ATen/ops/*_ops.h`), and, when the package has torchgen's `native_functions.yaml`, compares
the table's list `no_device_check` with the operators of both that it declares
`device_check: NoCheck`. `stable.h` passes a CPU 0-dim operand (`scalar_operand`) only to an
operator of that list, which the compiler checks, since ATen's CUDA kernel of any other operator
rejects a CPU tensor beside CUDA tensors. `--write` regenerates the table from the source keys and
those headers, and the list from that declaration file. `--library SO` also checks a built bridge:
the operators it names are the table's, and it names no other `aten::` operator, the torch C shims
it imports are the TRUST list, every other torch symbol it imports, apart from functions of the C++
standard library, is under "Unstable Remainder", and it does not link the CUDA runtime. With both
options, the libraries of the torch package define every torch symbol the bridge imports, so the
bridge loads with them. `scripts/lake.sh lint` runs the source checks, and the build runs `--torch`
and `--library` on the library it builds.

Each bullet of a TRUST list names its entries in code spans, then gives one reason for all of them
after a colon: `- `a`, `b`: reason`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BRIDGE = ROOT / "csrc/libtorch"
TABLE = BRIDGE / "schemas.h"
TRUST = ROOT / "docs/TRUST_BOUNDARIES.md"
SECTION = "## LibTorch Stable ABI"
SCHEMAS, SHIMS, SHIM_OPERATORS, UNSTABLE = (
    "Dispatcher Schemas", "C Shims", "Operators Behind C Shims", "Unstable Remainder")
# The one source compiled against torch's C++ API. TRUST lists what it links, read from a library.
UNSTABLE_SOURCES = {"unstable.cpp", "unstable.h"}
LINE_LIMIT = 100

KEY_RE = re.compile(r'\baten<"([^"]+)">')
SHIM_RE = re.compile(r"\b((?:aoti_torch|torch)_\w+)\s*\(")
SHIM_NAME_RE = re.compile(r"(?:aoti_torch|torch)_\w+")
INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]((?:torch|ATen|c10|caffe2)/[^>"]*)[>"]',
                        re.M)
# The headers of the Stable ABI: torch's stable C++ types, header-only utilities and C shims.
STABLE_HEADERS = ("torch/csrc/stable/", "torch/headeronly/", "torch/csrc/inductor/aoti_torch/c/",
                  "torch/csrc/inductor/aoti_torch/generated/")
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|(,)')
REGISTRY_RE = re.compile(
    r'name = "aten::(\w+)";\s*'
    r'static constexpr const char\* overload_name = "(\w*)";\s*'
    r'static constexpr const char\* schema_str = "((?:[^"\\]|\\.)*)";'
)
LIBRARY_KEY_RE = re.compile(rb"aten::(\w+)\x00(\w*)\x00")
LIBRARY_OPERATOR_RE = re.compile(rb"aten::(\w+)")
# The one call of torch's dispatcher, in `call` of `stable.h`: its operator and overload are the
# names of an `aten` key, which the table must have. Another use of any dispatcher entry could
# name an operator at run time, which neither the keys nor the library's strings would show.
DISPATCHER_RE = re.compile(r"\b\w*call_dispatcher\b")
DISPATCHER_CALL = "torch_call_dispatcher(names<Key>::op, names<Key>::overload,"
DISPATCHER_SOURCE = "stable.h"
# torchgen's operator declarations, beside the torch directory of a pip package.
NATIVE_FUNCTIONS = "torchgen/packaged/ATen/native/native_functions.yaml"
NATIVE_ENTRY_RE = re.compile(r"^- func: ([\w.]+)\((.*?)(?=^- func: |\Z)", re.M | re.S)
# Imports that are not torch's: the C and C++ runtimes (versioned), the weak hooks that GCC's
# startup files reference, Lean's runtime and its bundled mimalloc, the CUDA driver API and NVRTC.
RUNTIME_VERSION_RE = re.compile(r"(?:GLIBC|GLIBCXX|CXXABI|GCC)_[0-9.]+")
FOREIGN_RE = re.compile(r"__gmon_start__|_ITM_\w+|lean_\w+|mi_\w+|cu[A-Z]\w*|nvrtc[A-Z]\w*")
# GCC's thread support can reference these libc hooks weakly, without a symbol version.
WEAK_RUNTIME_RE = re.compile(r"__pthread_key_create|pthread_once")
# Demangled imports of the C++ standard library: names in namespace std over no torch type.
# libstdc++ defines each, with a version, but libc10 and libtorch export some of the same
# instantiations without one, and the linker binds an import to the first library defining it.
STANDARD_RE = re.compile(r"(?:(?:typeinfo(?: name)?|vtable|VTT) for )?std::")
TORCH_NAMESPACE_RE = re.compile(r"\b(?:at|c10|caffe2|torch)::")

HEADER = """\
// Generated by `scripts/checks/libtorch_abi.py --write --torch <torch package>`. Do not edit.
//
// The schema of each ATen operator the LibTorch bridge calls, one for each `aten` key in
// csrc/libtorch, as torch's generated operator headers declare it. `stable.h` reads the arguments
// of each call from this text at compile time. `scripts/checks/libtorch_abi.py` compares the table
// with the keys in the sources, the list in docs/TRUST_BOUNDARIES.md, a torch package's operator
// headers and a built library, and `no_device_check` with the package's native_functions.yaml.
#pragma once

#include <string_view>

namespace torchlean::schema {

inline constexpr std::string_view table[] = {
"""
DEVICES = """\
};

// The operators of the table and of the operators behind the C shims (docs/TRUST_BOUNDARIES.md)
// that torch declares `device_check: NoCheck`, so that ATen's generated CUDA wrappers of these do
// not reject a CPU 0-dim tensor beside CUDA tensors. `stable.h` passes its CPU 0-dim operands to no
// other operator.
inline constexpr std::string_view no_device_check[] = {
"""
FOOTER = """\
};

}  // namespace torchlean::schema
"""


def key(schema: str) -> str:
    """The `name.overload` key of a schema: its text before the argument list."""
    return schema.partition("(")[0]


def sources() -> list[Path]:
    """The bridge's own sources, without the generated table and the tests."""
    return sorted(path for path in BRIDGE.iterdir()
                  if path.suffix in {".c", ".cpp", ".h"} and path != TABLE)


def source_keys() -> list[str]:
    """The sorted distinct `aten<"...">` keys outside comments in the bridge sources."""
    found = set()
    for path in sources():
        found.update(KEY_RE.findall(COMMENT_RE.sub(" ", path.read_text(encoding="utf-8"))))
    return sorted(found)


def source_shims() -> dict[str, list[str]]:
    """Each torch C shim the stable sources call, with the files that call it."""
    found: dict[str, list[str]] = {}
    for path in sources():
        if path.name not in UNSTABLE_SOURCES:
            text = COMMENT_RE.sub(" ", path.read_text(encoding="utf-8"))
            for name in sorted(set(SHIM_RE.findall(text))):
                found.setdefault(name, []).append(path.name)
    return found


def dispatcher_calls() -> list[str]:
    """Errors for each use of a dispatcher entry in the sources but the one call of `stable.h`."""
    errors, found = [], False
    for path in sources():
        text = COMMENT_RE.sub(" ", path.read_text(encoding="utf-8"))
        for match in DISPATCHER_RE.finditer(text):
            call = " ".join(text[match.start():match.start() + 2 * len(DISPATCHER_CALL)].split())
            if path.name == DISPATCHER_SOURCE and call.startswith(DISPATCHER_CALL) and not found:
                found = True
            else:
                errors.append(f"{path.name} uses `{match.group(0)}` other than as "
                              f"`{DISPATCHER_CALL} ...)` in {DISPATCHER_SOURCE}: an operator is "
                              f"called only as `aten<\"name.overload\">(...)`")
    if not found:
        errors.append(f"{DISPATCHER_SOURCE} has no call `{DISPATCHER_CALL} ...)`")
    return errors


def unstable_includes() -> list[str]:
    """Errors for each torch header outside the stable ABI that a stable source includes."""
    return [f"{path.name} includes <{header}>, which is not a header of the Stable ABI"
            for path in sources() if path.name not in UNSTABLE_SOURCES
            for header in INCLUDE_RE.findall(COMMENT_RE.sub(" ", path.read_text(encoding="utf-8")))
            if not header.startswith(STABLE_HEADERS)]


def read_table(name: str = "table") -> list[str]:
    """The entries of the array `name` of `schemas.h`, adjacent string literals joined."""
    text = TABLE.read_text(encoding="utf-8")
    body = text.partition(f" {name}[] = {{")[2].partition("};")[0]
    entries, current = [], ""
    for literal, comma in LITERAL_RE.findall(body):
        if comma:
            entries.append(current)
            current = ""
        else:
            current += literal.replace('\\"', '"').replace("\\\\", "\\")
    if current:
        entries.append(current)
    return entries


def literal_lines(schema: str) -> list[str]:
    """`schema` as an entry of the table: string literals split after commas to fit the limit."""
    escaped = schema.replace("\\", "\\\\").replace('"', '\\"')
    pieces = re.split(r"(?<=, )", escaped)
    lines, current, indent = [], "", 4
    for piece in pieces:
        # Each line ends in `",` or `"`, so two columns after the text always suffice.
        if current and indent + 1 + len(current) + len(piece) + 2 > LINE_LIMIT:
            lines.append(" " * indent + f'"{current}"')
            current, indent = "", 8
        current += piece
    lines.append(" " * indent + f'"{current}",')
    if any(len(line) > LINE_LIMIT for line in lines):
        raise ValueError(f"schema has a parameter too long for one line: {schema}")
    return lines


def render_table(schemas: list[str], unchecked: list[str]) -> str:
    """The text of `schemas.h` for `schemas` and the `no_device_check` keys `unchecked`."""
    body = "".join(line + "\n" for schema in schemas for line in literal_lines(schema))
    lines, current = [], ""
    for name in unchecked:
        item = f'"{name}",'
        if current and 4 + len(current) + 1 + len(item) > LINE_LIMIT:
            lines.append("    " + current)
            current = ""
        current += (" " if current else "") + item
    if current:
        lines.append("    " + current)
    return HEADER + body + DEVICES + "".join(line + "\n" for line in lines) + FOOTER


def registry(torch: Path) -> dict[str, str]:
    """The schema of each ATen operator key, as the generated operator headers of the torch package
    at `torch` declare it."""
    ops = torch / "include/ATen/ops"
    headers = sorted(ops.glob("*_ops.h"))
    if not headers:
        raise ValueError(f"no ATen operator headers in {ops}")
    found = {}
    for header in headers:
        for name, overload, schema in REGISTRY_RE.findall(header.read_text(encoding="utf-8")):
            found[name + ("." + overload if overload else "")] = schema.replace('\\"', '"')
    return found


def unchecked(torch: Path, names: list[str]) -> list[str] | None:
    """The sorted `names` that `native_functions.yaml` beside the torch package at `torch` declares
    `device_check: NoCheck`, or None if the package has no torchgen beside it, as in a LibTorch
    SDK."""
    declarations = torch.parent / NATIVE_FUNCTIONS
    if not declarations.is_file():
        return None
    found = {name for name, entry in
             NATIVE_ENTRY_RE.findall(declarations.read_text(encoding="utf-8"))
             if re.search(r"^\s+device_check: NoCheck\b", entry, re.M)}
    return sorted(set(names) & found)


def trust_lists() -> dict[str, list[str]]:
    """The entries of each list, a `###` heading, of TRUST's stable ABI section.

    A bullet's entries are its code spans before its first colon outside code spans, and its reason
    is the text after that colon, which must not be empty. A code span may wrap across lines, and
    its whitespace is normalized to single spaces.
    """
    text = TRUST.read_text(encoding="utf-8")
    section = re.search(rf"^{re.escape(SECTION)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    lists: dict[str, list[str]] = {}
    if section:
        for part in re.split(r"^### ", section.group(1), flags=re.M)[1:]:
            title, _, body = part.partition("\n")
            items = []
            for bullet in re.findall(r"^- (.*?)(?=^\S|\Z)", body, re.M | re.S):
                names, reason = [], ""
                for match in re.finditer(r"`([^`]+)`|:", bullet):
                    if match.group(0) == ":":
                        reason = bullet[match.end():].strip()
                        break
                    names.append(" ".join(match.group(1).split()))
                if not names or not reason:
                    raise ValueError(f"a bullet under `### {title.strip()}` needs entries, a colon "
                                     f"and a reason: {' '.join(bullet.split())[:80]}")
                items += names
            lists[title.strip()] = items
    return lists


def compare(found: list[str], listed: list[str], found_in: str, listed_in: str) -> list[str]:
    """Errors for each item in only one of `found` and `listed`, and each item listed twice."""
    errors = [f"`{item}` is listed twice in {listed_in}" for item in sorted(set(listed))
              if listed.count(item) > 1]
    errors += [f"`{item}` is in {found_in} but not in {listed_in}"
               for item in sorted(set(found) - set(listed))]
    errors += [f"`{item}` is in {listed_in} but not in {found_in}"
               for item in sorted(set(listed) - set(found))]
    return errors


def check_sources(trust: dict[str, list[str]]) -> list[str]:
    """The source-level checks: keys against the table, table and shims against TRUST, includes."""
    errors = [f"{TRUST.name} has no `### {title}` list under `{SECTION}`"
              for title in (SCHEMAS, SHIMS, SHIM_OPERATORS, UNSTABLE) if title not in trust]
    table = read_table()
    keys = [key(schema) for schema in table]
    if keys != sorted(set(keys)):
        errors.append(f"the keys of {TABLE.name} are not sorted and distinct")
    named = keys + [key(schema) for schema in trust.get(SHIM_OPERATORS, [])]
    devices = read_table("no_device_check")
    if devices != sorted(set(devices)):
        errors.append(f"`no_device_check` of {TABLE.name} is not sorted and distinct")
    errors += [f"`no_device_check` of {TABLE.name} has `{name}`, which is not an operator of the "
               f"table or of TRUST's {SHIM_OPERATORS}" for name in devices if name not in named]
    errors += compare(source_keys(), keys, "the aten keys of the sources", TABLE.name)
    errors += compare(table, trust.get(SCHEMAS, []), TABLE.name, f"TRUST's {SCHEMAS}")
    listed = set(trust.get(SHIMS, []))
    for name, files in sorted(source_shims().items()):
        if name not in listed:
            errors.append(f"`{name}` is called in {', '.join(files)} but not in TRUST's {SHIMS}")
    return errors + dispatcher_calls() + unstable_includes()


def check_registry(torch: Path, trust: dict[str, list[str]]) -> list[str]:
    """The table's entries and TRUST's operators behind the C shims against the schemas in the
    generated operator headers of the torch package at `torch`, and `no_device_check` against its
    device checks."""
    known = registry(torch)
    errors = []
    schemas = read_table() + trust.get(SHIM_OPERATORS, [])
    for schema in schemas:
        registered = known.get(key(schema))
        if registered is None:
            errors.append(f"{torch} declares no operator `aten::{key(schema)}` in its headers")
        elif registered != schema:
            errors.append(f"{torch} declares `{registered}`, not `{schema}`")
    declared = unchecked(torch, [key(schema) for schema in schemas])
    if declared is not None:
        errors += compare(declared, read_table("no_device_check"),
                          f"the operators {torch} declares without a device check",
                          f"`no_device_check` of {TABLE.name}")
    return errors


def tool(*command: str, stdin: str | None = None) -> str:
    """The standard output of `command`, which must succeed."""
    return subprocess.run(command, input=stdin, check=True, capture_output=True, text=True).stdout


def without_parameters(symbol: str) -> str:
    """A demangled symbol without its parameter list and qualifiers, so overloads share an entry."""
    end = symbol.rfind(")")
    depth = 0
    for index in range(end, -1, -1):
        if symbol[index] == ")":
            depth += 1
        elif symbol[index] == "(":
            depth -= 1
            if depth == 0:
                return symbol[:index]
    return symbol


def torch_imports(library: Path) -> tuple[list[str], list[str]]:
    """The torch C shims `library` imports, and its other imports that are not foreign."""
    shims, others = [], []
    for line in tool("nm", "-D", "--with-symbol-versions", "--undefined-only",
                     str(library)).splitlines():
        binding, symbol = line.split()[-2:]
        name, _, version = symbol.partition("@")
        if SHIM_NAME_RE.fullmatch(name):
            shims.append(name)
        elif not (RUNTIME_VERSION_RE.fullmatch(version) or FOREIGN_RE.fullmatch(name) or
                  (binding == "w" and WEAK_RUNTIME_RE.fullmatch(name))):
            others.append(name)
    return shims, others


def check_exports(library: Path, torch: Path) -> list[str]:
    """Each torch symbol `library` imports against the symbols the libraries of `torch` define."""
    defined = set()
    for path in sorted((torch / "lib").glob("lib*.so*")):
        defined.update(line.split()[-1].partition("@")[0] for line in
                       tool("nm", "-D", "--defined-only", str(path)).splitlines())
    shims, others = torch_imports(library)
    return [f"{torch} defines no `{name}`, which {library.name} imports"
            for name in shims + others if name not in defined]


def check_library(library: Path, trust: dict[str, list[str]]) -> list[str]:
    """A built bridge's operators, shims and torch imports against the table and TRUST."""
    data = library.read_bytes()
    keys = [key(schema) for schema in read_table()]
    named = sorted({(op + b"." + overload if overload else op).decode()
                    for op, overload in LIBRARY_KEY_RE.findall(data)})
    errors = compare(named, keys, f"the operators {library.name} names", TABLE.name)
    operators = {name.partition(".")[0] for name in keys}
    errors += [f"{library.name} names `aten::{name}`, which is not an operator of {TABLE.name}"
               for name in sorted({op.decode() for op in LIBRARY_OPERATOR_RE.findall(data)})
               if name not in operators]
    shims, others = torch_imports(library)
    errors += compare(shims, trust.get(SHIMS, []), f"the imports of {library.name}",
                      f"TRUST's {SHIMS}")
    demangled = tool("c++filt", stdin="".join(name + "\n" for name in others)).splitlines()
    torch_names = {without_parameters(symbol) for symbol in demangled
                   if not (STANDARD_RE.match(symbol) and not TORCH_NAMESPACE_RE.search(symbol))}
    errors += compare(sorted(torch_names), trust.get(UNSTABLE, []),
                      f"the imports of {library.name}", f"TRUST's {UNSTABLE}")
    for line in tool("readelf", "-d", str(library)).splitlines():
        needed = re.search(r"\(NEEDED\).*\[(libcudart[^\]]*)\]", line)
        if needed:
            errors.append(f"{library.name} links the CUDA runtime {needed.group(1)}")
    return errors


def main() -> int:
    """Run the checks the options select, print each error and return 1 if there is any."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torch", type=Path, help="a torch package or LibTorch SDK directory")
    parser.add_argument("--library", type=Path, help="a built libtorchlean_libtorch.so")
    parser.add_argument("--write", action="store_true",
                        help="regenerate schemas.h from the source keys and the --torch headers")
    args = parser.parse_args()
    try:
        trust = trust_lists()
        if args.write:
            if not args.torch:
                parser.error("--write needs --torch")
            known = registry(args.torch)
            missing = [name for name in source_keys() if name not in known]
            if missing:
                raise ValueError(f"{args.torch} declares no operator for {', '.join(missing)}")
            declared = unchecked(args.torch, source_keys() +
                                 [key(schema) for schema in trust.get(SHIM_OPERATORS, [])])
            if declared is None:
                raise ValueError(f"--write needs {NATIVE_FUNCTIONS} beside {args.torch}")
            TABLE.write_text(render_table([known[name] for name in source_keys()], declared),
                             encoding="utf-8")
        errors = check_sources(trust)
        if args.torch:
            errors += check_registry(args.torch, trust)
        if args.library:
            errors += check_library(args.library, trust)
        if args.torch and args.library:
            errors += check_exports(args.library, args.torch)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        errors = [str(error)]
    for error in errors:
        print(f"error: LibTorch ABI: {error}", file=sys.stderr)
    if not errors:
        print(f"OK: {len(read_table())} dispatcher schemas, the C shims and the operators behind "
              f"them match TRUST.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
