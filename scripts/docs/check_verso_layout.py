#!/usr/bin/env python3
"""Check the guide's reading structure in rendered HTML and section references."""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


DEEP_NUMBER = re.compile(r"^\s*\d+(?:\.\d+){2,}\.?(?:\s|$)")


class Page(HTMLParser):
    """Collect anchors, headings, and the table of contents' number cells."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.labels: list[str] = []
        self.pending: list[tuple[str, list[str]]] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        # Restrict the scan to section labels so version links such as "1.2.3"
        # in prose or code do not look like unwanted section numbering.
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} or (
            tag == "td" and "num" in (attributes.get("class") or "").split()
        ):
            self.pending.append((tag, []))

    def handle_data(self, data: str) -> None:
        for _, pieces in self.pending:
            pieces.append(data)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.pending) - 1, -1, -1):
            name, pieces = self.pending[index]
            if name == tag:
                self.labels.append(" ".join("".join(pieces).split()))
                self.pending.pop(index)
                break


def check_layout(root: Path) -> tuple[int, int]:
    """Keep chapter/page numbers, with all deeper sections on their parent page."""
    root = root.resolve()
    pages: dict[Path, Page] = {}
    errors: list[str] = []
    for path in root.rglob("*.html"):
        text = path.read_text()
        if 'href="book.css"' not in text:
            continue
        page = Page(text)
        pages[path] = page
        for label in page.labels:
            if DEEP_NUMBER.match(label):
                errors.append(f"{path.relative_to(root)}: deep section number in {label!r}")
    if not pages:
        raise ValueError(f"no rendered guide pages under {root}")

    xref = json.loads((root / "xref.json").read_text())
    sections = [
        entry
        for entries in xref["Verso.Genre.Manual.section"]["contents"].values()
        for entry in entries
    ]
    if not sections:
        raise ValueError("the guide has no section references")

    # Context contains the book, chapter, page, then any headings within that page.
    # Compare addresses rather than slugs: changing a title must not split its children.
    page_addresses = {
        tuple(part["title"] for part in entry["data"]["context"]): entry["address"]
        for entry in sections
        if len(entry["data"]["context"]) == 3
    }
    for entry in sections:
        data = entry["data"]
        title = data["title"]
        context = data["context"]
        number = data.get("sectionNum") or ""
        if DEEP_NUMBER.match(number):
            errors.append(f"{title!r}: deep section number in cross-reference {number!r}")
        if len(context) > 3:
            parent = tuple(part["title"] for part in context[:3])
            address = page_addresses.get(parent)
            if address is None or entry["address"] != address:
                errors.append(f"{title!r}: section is split away from its parent page")

        address = unquote(urlsplit(entry["address"]).path).lstrip("/")
        target = (root / address / "index.html").resolve()
        if not target.is_relative_to(root):
            errors.append(f"{title!r}: section address leaves the guide")
            continue
        page = pages.get(target)
        if page is None or entry["id"] not in page.ids:
            errors.append(f"{title!r}: section anchor is missing from {address or '/'}")

    if errors:
        sample = "\n".join(dict.fromkeys(errors[:20]))
        raise ValueError(f"guide reading-layout check failed ({len(errors)} findings):\n{sample}")
    return len(pages), len(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", required=True, type=Path)
    args = parser.parse_args()
    try:
        pages, sections = check_layout(args.guide)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Guide layout checked: {pages} pages, {sections} sections; numbering stops at N.M.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
