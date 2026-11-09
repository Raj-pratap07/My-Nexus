"""
Offline tests for the research-result assessment used by the live test.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.agent.test_research_assessment -v
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.agent.ollama_integration_test import assess_research_result, cited_records

REC = {"source": "openalex", "source_id": "W1", "title": "A paper", "doi": "10.1/x"}


def result(data=None, is_error=False, text=""):
    return SimpleNamespace(
        is_error=is_error, structured_content=data,
        content=[SimpleNamespace(text=text)],
    )


class AssessResearchResultTests(unittest.TestCase):
    def test_valid_records_succeed(self):
        ok, _ = assess_research_result(result({"records": [REC], "result_count": 1}))
        self.assertTrue(ok)

    def test_tool_error_is_provider_failure(self):
        ok, reason = assess_research_result(
            result(is_error=True, text="Error executing tool search_openalex_works"))
        self.assertFalse(ok)
        self.assertIn("provider failure", reason)

    def test_visible_429_is_named(self):
        ok, reason = assess_research_result(
            result(is_error=True, text="Upstream rate limit reached (HTTP 429)."))
        self.assertFalse(ok)
        self.assertIn("429", reason)

    def test_empty_result_is_not_success(self):
        ok, reason = assess_research_result(result({"records": [], "result_count": 0}))
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_malformed_results_rejected(self):
        for res in (
            result(None),
            result({"records": "x", "result_count": 1}),
            result({"records": [REC], "result_count": 2}),
            result({"records": [{"source": "openalex", "title": "  ", "doi": "x"}],
                    "result_count": 1}),
            result({"records": [{"source": "openalex", "title": "T"}], "result_count": 1}),
            result({"records": ["str"], "result_count": 1}),
        ):
            with self.subTest(res=res.structured_content):
                self.assertFalse(assess_research_result(res)[0])


class CitedRecordsTests(unittest.TestCase):
    RECS = [{"title": "Graph Retrieval-Augmented Generation: A Survey", "doi": "10.1/abc"},
            {"title": "Ragas: Automated Evaluation", "doi": None}]

    def test_title_match_ignores_case_and_punctuation(self):
        out = cited_records('The paper "graph retrieval augmented generation - a survey".', self.RECS)
        self.assertEqual(out, [self.RECS[0]["title"]])

    def test_doi_match(self):
        self.assertEqual(len(cited_records("See doi 10.1/ABC for details.", self.RECS)), 1)

    def test_unrelated_answer_cites_nothing(self):
        self.assertEqual(cited_records("RAG is useful.", self.RECS), [])

    def test_short_title_inside_longer_title_is_not_a_separate_citation(self):
        recs = [{"title": "Retrieval-Augmented Generation", "doi": None},
                {"title": "Graph-based Retrieval-augmented Generation", "doi": None}]
        out = cited_records("1. Graph-based Retrieval-augmented Generation", recs)
        self.assertEqual(out, ["Graph-based Retrieval-augmented Generation"])

    def test_short_title_counts_when_it_appears_on_its_own(self):
        recs = [{"title": "Retrieval-Augmented Generation", "doi": None},
                {"title": "Graph-based Retrieval-augmented Generation", "doi": None}]
        out = cited_records("Graph-based Retrieval-augmented Generation and also "
                            "Retrieval-Augmented Generation", recs)
        self.assertEqual(len(out), 2)

    def test_partial_word_does_not_match(self):
        self.assertEqual(cited_records("Ragas rocks", [{"title": "Rag", "doi": None}]), [])

    def test_empty_titles_never_match(self):
        self.assertEqual(cited_records("anything", [{"title": "", "doi": None}]), [])


if __name__ == "__main__":
    unittest.main()
