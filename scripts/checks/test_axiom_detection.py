"""Keep source axioms visible to the trust-boundary lint, including qualified declarations."""

import unittest

import repo_lint


class AxiomDetectionTests(unittest.TestCase):
    def test_declarations_and_masking(self):
        source = '''\
axiom plain : True
public axiom visible : True
private axiom hidden : True
@[simp]
protected axiom Nested.αβ : True
-- axiom comment : True
/- axiom outer : True /- axiom nested : True -/ -/
def text := "axiom inString : True"
def axiomLike := 1
public /- modifier comment -/ axiom actual : True
'''
        masked = repo_lint._mask_lean_comments_and_strings(source)
        self.assertEqual(repo_lint.AXIOM_DECL_RE.findall(masked),
                         ["plain", "visible", "hidden", "Nested.αβ", "actual"])


if __name__ == "__main__":
    unittest.main()
