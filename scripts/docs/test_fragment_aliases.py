"""Regression checks for guide polishing followed by the real site link checker."""

import json
from pathlib import Path
import tempfile
import unittest

from check_site_links import check_site
from polish_verso_guide import add_fragment_aliases


class FragmentAliasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def page(self, address: str, content: str) -> Path:
        path = self.root / address / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<html><body><main>{content}</main></body></html>")
        return path

    def xref(self, *entries: tuple[str, str, int]) -> None:
        contents = {
            str(i): [{
                "address": address,
                "id": fragment,
                "data": {"context": [{"title": str(j)} for j in range(depth)]},
            }]
            for i, (address, fragment, depth) in enumerate(entries)
        }
        (self.root / "xref.json").write_text(json.dumps({
            "Verso.Genre.Manual.section": {"contents": contents},
        }))

    def test_unknown_links_remain_broken_without_metadata(self) -> None:
        index = self.page("", '<a href="#typo">Local</a><a href="chapter/#gone">Other</a>')
        chapter = self.page("chapter", '<h1 id="real">Chapter</h1>')
        before = {path: path.read_text() for path in (index, chapter)}
        errors = check_site(self.root)
        self.assertEqual(len(errors), 2)

        add_fragment_aliases(self.root)

        self.assertEqual(check_site(self.root), errors)
        self.assertEqual({path: path.read_text() for path in before}, before)

    def test_metadata_titles_get_aliases_but_typos_do_not(self) -> None:
        index = self.page("", '<h1>Book</h1><a href="chapter/page/#page">Page</a>')
        self.page("chapter", "<h1>Chapter</h1>")
        page = self.page(
            "chapter/page",
            '<h1>Page</h1><h2 id="real">Heading</h2><a href="#typo">Broken</a>',
        )
        self.xref(("", "book", 1), ("chapter/", "chapter", 2),
                  ("chapter/page/", "page", 3), ("chapter/page/", "real", 4))

        add_fragment_aliases(self.root)

        self.assertIn('id="book"', index.read_text())  # No incoming link to the book title.
        self.assertIn('id="chapter"', (self.root / "chapter/index.html").read_text())
        self.assertIn('<main><span id="page"', page.read_text())
        self.assertEqual(page.read_text().count('id="real"'), 1)
        self.assertEqual(check_site(self.root), [
            "chapter/page/index.html: missing anchor `#typo` in chapter/page/index.html",
        ])
        polished = page.read_text()
        add_fragment_aliases(self.root)
        self.assertEqual(page.read_text(), polished)

    def test_missing_deep_heading_is_not_moved_to_page_start(self) -> None:
        self.page("", '<h1>Book</h1><a href="#deleted-heading">Deleted</a>')
        self.xref(("", "deleted-heading", 4))

        add_fragment_aliases(self.root)

        self.assertEqual(check_site(self.root), [
            "index.html: missing anchor `#deleted-heading` in index.html",
        ])

    def test_encoded_addresses_and_escaped_ids(self) -> None:
        self.page("", '<a href="some%20page/#title%22%26%5C1">Page</a>')
        page = self.page("some page", "<h1>Page</h1>")
        self.xref(("some%20page/", 'title"&\\1', 3))

        add_fragment_aliases(self.root)

        self.assertIn('id="title&quot;&amp;\\1"', page.read_text())
        self.assertEqual(check_site(self.root), [])

    def test_external_and_escaping_metadata_cannot_create_aliases(self) -> None:
        page = self.page("", "<h1>Book</h1>")
        # An existing sibling makes the traversal check meaningful.
        with tempfile.TemporaryDirectory(dir=self.root.parent) as sibling:
            outside = Path(sibling) / "index.html"
            outside.write_text("<html><body>Outside</body></html>")
            self.xref(
                ("https://example.com/", "external", 1),
                ("//example.com/", "network", 1),
                (f"../{outside.parent.name}/", "escaped", 1),
            )
            add_fragment_aliases(self.root)
            self.assertEqual(outside.read_text(), "<html><body>Outside</body></html>")
        self.assertNotIn("tl-anchor-alias", page.read_text())


if __name__ == "__main__":
    unittest.main()
