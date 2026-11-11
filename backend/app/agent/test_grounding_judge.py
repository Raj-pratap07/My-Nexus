"""
Offline tests for the Known-limitations grounding grader. No network, no
model, no MCP: pure text in, report out.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.agent.test_grounding_judge -v
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import app.agent.github_prompt_eval as ev
import app.agent.quote_fidelity as qf
from app.agent.grounding_judge import (
    GradeReport,
    grade_limitations,
    parse_section,
    segment_answer,
)

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[2]
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
FIXTURE = AGENT_DIR / "eval_fixtures" / "limitations_answers.json"
CHECK_NAMES = {"section_missing", "completeness", "unsupported", "polarity_lost",
               "status_invented", "quote_fidelity"}

# The README's section exactly as inspected (continuation lines joined).
PINNED_ITEMS = [
    "Authentication is a single static local token: anyone holding it *is* the user. "
    "Do not expose the API to a network.",
    "Not yet tested against PostgreSQL.",
    "No rate limiting or per-user quotas; the per-conversation lock is in-memory, so "
    "run a single worker process.",
    "Research records contain titles and metadata but no abstracts, so model summaries "
    "of a paper's content are inference from its title.",
    "Long-term memory is schema only.",
]

# Small, fixed source for unit tests, independent of the real README.
SOURCE = """# Demo
Intro line. The web UI is planned for later.
Prompt injection is **not** prevented by the prompt.

## Known limitations
- Single static token: anyone holding it is the user.
- Not yet tested against PostgreSQL.
- Long-term memory is schema only.

## Layout
- backend/ code
"""
GOOD = ("- Authentication uses a single static token; anyone holding it is the user.\n"
        "- Not yet tested against PostgreSQL.\n"
        "- Long-term memory is schema only.")


def grade(answer: str, source: str = SOURCE, **kw) -> GradeReport:
    return grade_limitations(answer, source, **kw)


class ParseSectionTests(unittest.TestCase):
    def test_real_readme_has_exactly_the_five_pinned_items(self):
        # Fails on purpose when README.md's section changes: re-label the fixtures then.
        self.assertEqual(parse_section(README), PINNED_ITEMS)

    def test_continuation_lines_are_joined(self):
        md = "## Known limitations\n- first line\n  continues here\n    and here.\n- second\n"
        self.assertEqual(parse_section(md), ["first line continues here and here.", "second"])

    def test_section_ends_at_next_level_one_or_two_heading(self):
        for nxt in ("## Next", "# Top"):
            with self.subTest(nxt=nxt):
                md = f"## Known limitations\n- only item\n{nxt}\n- not an item\n"
                self.assertEqual(parse_section(md), ["only item"])

    def test_deeper_heading_does_not_end_the_section(self):
        md = "## Known limitations\n- a\n### Detail\n- b\n## End\n- c\n"
        self.assertEqual(parse_section(md), ["a", "b"])

    def test_missing_heading_returns_none(self):
        self.assertIsNone(parse_section("# Title\n## Limitations\n- x\n"))
        self.assertIsNone(parse_section(""))

    def test_heading_match_is_exact_but_case_insensitive(self):
        self.assertEqual(parse_section("## known LIMITATIONS\n- x\n"), ["x"])
        self.assertIsNone(parse_section("## Known limitations and plans\n- x\n"))

    def test_empty_section_is_an_empty_list(self):
        self.assertEqual(parse_section("## Known limitations\n\n## Next\n"), [])

    def test_blank_lines_between_items_are_ignored(self):
        self.assertEqual(parse_section("## Known limitations\n- a\n\n- b\n"), ["a", "b"])


class SegmentAnswerTests(unittest.TestCase):
    def test_list_items_numbered_and_bulleted(self):
        self.assertEqual(segment_answer("1. One\n2) Two\n- Three\n* Four"),
                         ["One", "Two", "Three", "Four"])

    def test_continuation_lines_join_their_item(self):
        self.assertEqual(segment_answer("- starts here\n  ends here"), ["starts here ends here"])

    def test_headings_and_table_rules_are_dropped(self):
        self.assertEqual(segment_answer("### Heading\n|---|---|\nText"), ["Text"])

    def test_compound_bullet_is_split_into_sentences(self):
        claims = segment_answer("- Authentication uses a static local token. The system "
                                "also lacks OAuth and multi-factor authentication.")
        self.assertEqual(claims, ["Authentication uses a static local token.",
                                  "The system also lacks OAuth and multi-factor authentication."])

    def test_semicolons_and_explicit_additions_split(self):
        self.assertEqual(segment_answer("No quotas; one worker"), ["No quotas", "one worker"])
        self.assertEqual(segment_answer("Static token and also no OAuth"),
                         ["Static token", "no OAuth"])
        self.assertEqual(segment_answer("Static token, additionally no OAuth"),
                         ["Static token", "no OAuth"])

    def test_comma_conjunction_splits_but_bare_and_does_not(self):
        self.assertEqual(segment_answer("No abstracts, and Crossref caps results"),
                         ["No abstracts", "Crossref caps results"])
        self.assertEqual(segment_answer("It has titles and metadata"),
                         ["It has titles and metadata"])

    def test_prose_paragraph_is_split_into_sentences(self):
        self.assertEqual(segment_answer("First claim here. Second claim here."),
                         ["First claim here.", "Second claim here."])


class GradeCheckTests(unittest.TestCase):
    """Each of the five checks, isolated on a small fixed source."""

    def test_correct_answer_passes(self):
        report = grade(GOOD)
        self.assertTrue(report.passed, report.explain())
        self.assertEqual(report.missing, [])

    # 1. Completeness
    def test_missing_item_fails_completeness_and_names_it(self):
        report = grade(GOOD.rsplit("\n", 1)[0])
        self.assertEqual(report.failed_checks, ["completeness"])
        self.assertEqual(report.missing, [2])
        self.assertIn("MISSING", report.explain())

    # 2. Unsupported claims
    def test_invented_claim_is_not_in_source(self):
        report = grade(GOOD + "\n- There is no HTTPS support, so traffic is unencrypted.")
        self.assertEqual(report.failed_checks, ["unsupported"])
        [bad] = report.unsupported
        self.assertEqual(bad.classification, "not-in-source")

    def test_claim_from_elsewhere_in_source_is_outside_section_with_evidence(self):
        report = grade(GOOD + "\n- Prompt injection is not prevented.")
        [bad] = report.unsupported
        self.assertEqual(bad.classification, "outside-section")
        self.assertIn("Prompt injection", bad.evidence)

    def test_unsupported_addition_inside_a_supported_bullet_is_reported(self):
        answer = GOOD.replace("is the user.", "is the user. It also lacks OAuth and MFA.")
        report = grade(answer)
        self.assertEqual(report.failed_checks, ["unsupported"])
        self.assertEqual([c.claim for c in report.unsupported], ["It also lacks OAuth and MFA."])
        # The supported sentence of the same bullet is still credited.
        self.assertEqual(report.missing, [])

    def test_aligned_claim_with_new_technical_terms_is_unsupported(self):
        report = grade(GOOD.replace("Long-term memory is schema only.",
                                    "Long-term memory is schema only via Redis."))
        [bad] = report.unsupported
        self.assertIn("Redis", bad.novel_technical)

    def test_weakly_aligned_claim_is_unsupported(self):
        report = grade(GOOD + '\n- The token is a shared password reused for every account '
                       "everywhere.")
        self.assertIn("unsupported", report.failed_checks)
        self.assertTrue(any("weakly aligned" in n for c in report.unsupported for n in c.notes))

    def test_meta_sentences_are_not_claims(self):
        report = grade("Here are the known limitations listed in the README:\n" + GOOD
                       + "\nThese are the only limitations it documents.")
        self.assertTrue(report.passed, report.explain())

    def test_labelled_inference_goes_to_review_not_failure(self):
        report = grade(GOOD + "\nInterpretation: sharing the token shares the account.")
        self.assertTrue(report.passed, report.explain())
        self.assertTrue(any("labelled inference" in r for r in report.needs_review))

    # 3. Status polarity lost
    def test_reversed_negation_fails_polarity(self):
        report = grade(GOOD.replace("Not yet tested against PostgreSQL.",
                                    "Tested against PostgreSQL."))
        self.assertEqual(report.failed_checks, ["polarity_lost"])

    def test_positive_status_for_a_restricted_item_fails_polarity(self):
        report = grade(GOOD.replace("Long-term memory is schema only.",
                                    "Long-term memory is fully implemented."))
        self.assertEqual(report.failed_checks, ["polarity_lost"])

    def test_restriction_in_another_clause_does_not_mask_a_reversal(self):
        report = grade(GOOD.replace("Not yet tested against PostgreSQL.",
                                    "It is tested against PostgreSQL, and only one worker runs."))
        self.assertIn("polarity_lost", report.failed_checks)

    def test_nobody_counts_as_negation(self):
        report = grade(GOOD.replace("Not yet tested against PostgreSQL.",
                                    "Nobody has tested it against PostgreSQL yet."))
        self.assertNotIn("polarity_lost", report.failed_checks, report.explain())

    # 4. Status invented
    def test_planned_status_not_in_item_is_invented(self):
        report = grade(GOOD.replace("Long-term memory is schema only.",
                                    "Long-term memory is schema only; it is planned for later."))
        self.assertIn("status_invented", report.failed_checks)
        self.assertEqual(report.status_invented[0]["item"], 2)

    def test_status_not_specified_is_accepted(self):
        answer = GOOD.replace("schema only.", "schema only. Status: the README does not "
                                              "say whether it is planned.")
        report = grade(answer)
        self.assertTrue(report.passed, report.explain())

    def test_status_note_inherits_the_previous_claim(self):
        answer = GOOD.replace("schema only.", "schema only. Status: planned.")
        report = grade(answer)
        self.assertEqual(report.failed_checks, ["status_invented"])

    def test_grader_never_assigns_a_status_itself(self):
        # No status words in the answer -> no status findings, whatever the item says.
        report = grade(GOOD)
        self.assertEqual((report.status_invented, report.polarity_lost), ([], []))

    # 5. Quote fidelity
    def test_misquote_fails_quote_fidelity(self):
        report = grade(GOOD + '\n- It warns that "the token is a shared password".')
        self.assertIn("quote_fidelity", report.failed_checks)
        self.assertEqual(report.quotes_unmatched, ["the token is a shared password"])

    def test_verbatim_quote_and_allowed_extra_source_pass(self):
        q = "Do not invent details, please"
        report = grade(GOOD + '\n- The README says "anyone holding it is the user".\n'
                       f'You asked: "{q}".', extra_quote_sources=[q])
        self.assertEqual(report.quotes_unmatched, [])

    # Section missing
    def test_source_without_the_section_fails_but_still_checks_quotes(self):
        report = grade('Says "this is not in the source text".', "# No section here\n")
        self.assertFalse(report.section_found)
        self.assertEqual(report.failed_checks, ["section_missing", "quote_fidelity"])
        self.assertTrue(report.needs_review)

    # Explainability
    def test_explain_lists_every_claim_and_disclaims_semantic_certainty(self):
        report = grade(GOOD + "\n- There is no HTTPS support.")
        text = report.explain()
        self.assertIn("not semantic proof", text)
        for c in report.claims:
            self.assertIn(repr(c.claim), text)
        self.assertIn("not-in-source", text)


class LabelledFixtureTests(unittest.TestCase):
    """The grader must agree with every hand label before it gates anything."""

    REQUIRED = {
        "complete-paraphrased", "concise-bullets", "prose", "missing-limitation",
        "invented-limitation", "outside-section-claim", "compound-unsupported-addition",
        "memory-described-as-implemented", "postgresql-described-as-planned",
        "misquoted-source", "turn1-summary-only", "failed-turn2-reconstructed",
    }

    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]

    def test_fixture_covers_every_required_category(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(self.REQUIRED, set(ids))

    def test_labels_use_only_known_check_names(self):
        for c in self.cases:
            with self.subTest(case=c["id"]):
                self.assertLessEqual(set(c["expected"]["failed_checks"]), CHECK_NAMES)

    def test_reconstructed_case_is_not_presented_as_the_original(self):
        [case] = [c for c in self.cases if c["id"] == "failed-turn2-reconstructed"]
        self.assertIn("NOT the full original answer", case["description"])

    def test_grader_agrees_with_every_label(self):
        for c in self.cases:
            if "grader_known_disagreement" in c:
                continue
            with self.subTest(case=c["id"]):
                report = grade_limitations(c["answer"], README)
                self.assertEqual(sorted(report.failed_checks),
                                 sorted(c["expected"]["failed_checks"]), report.explain())

    def test_known_disagreements_are_pinned_exactly(self):
        # A recorded disagreement must still occur as recorded; if the grader
        # changes, the case must be re-examined, not silently reclassified.
        known = [c for c in self.cases if "grader_known_disagreement" in c]
        self.assertEqual([c["id"] for c in known], ["heldout-heavy-paraphrase"])
        for c in known:
            with self.subTest(case=c["id"]):
                report = grade_limitations(c["answer"], README)
                self.assertEqual(sorted(report.failed_checks),
                                 sorted(c["grader_known_disagreement"]["grader_failed_checks"]))
                self.assertNotEqual(sorted(report.failed_checks),
                                    sorted(c["expected"]["failed_checks"]))

    def test_outside_section_claim_is_classified_with_readme_evidence(self):
        [case] = [c for c in self.cases if c["id"] == "outside-section-claim"]
        report = grade_limitations(case["answer"], README)
        self.assertEqual([(u.claim, u.classification) for u in report.unsupported],
                         [(case["expected"]["outside_section"][0], "outside-section")])
        self.assertIn("prompt injection", report.unsupported[0].evidence.lower())


class ModuleBoundaryTests(unittest.TestCase):
    def test_quote_helpers_moved_without_changing_the_eval_names(self):
        for name in ("extract_quotes", "unmatched_quotes", "_normalize",
                     "QUOTE_RE", "MIN_QUOTE_CHARS", "MIN_SEGMENT_CHARS"):
            with self.subTest(name=name):
                self.assertIs(getattr(ev, name), getattr(qf, name))

    def test_grader_and_quote_helpers_import_no_model_or_mcp_code(self):
        # Pure modules: no providers, orchestrator, MCP or eval imports (no cycles,
        # and no way to call another LLM).
        for module in ("grounding_judge.py", "quote_fidelity.py"):
            tree = ast.parse((AGENT_DIR / module).read_text(encoding="utf-8"))
            imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                         for a in n.names}
            with self.subTest(module=module):
                app_imports = {m for m in imported if m and m.startswith("app")}
                self.assertLessEqual(app_imports, {"app.agent.quote_fidelity"})


if __name__ == "__main__":
    unittest.main()
