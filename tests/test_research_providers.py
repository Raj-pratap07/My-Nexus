"""
Offline unit tests for the research providers (no network).

Run from the project root:
    backend/.venv/Scripts/python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from custom_mcp_servers.research_server import providers
from custom_mcp_servers.research_server.providers import ProviderError

OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/ABC",
            "title": "  Deep   Learning ",
            "authorships": [
                {"author": {"display_name": "Ada Lovelace"}},
                {"author": {"display_name": None}},
                {"author": {"display_name": "Alan Turing"}},
            ],
            "publication_year": 2020,
            "primary_location": {
                "landing_page_url": "https://example.org/paper",
                "source": {"display_name": "Journal of Tests"},
            },
            "cited_by_count": 42,
            "type": "article",
        }
    ]
}

CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "DOI": "10.1000/xyz",
                "title": ["Graph Networks"],
                "author": [
                    {"given": "Grace", "family": "Hopper"},
                    {"name": "Some Consortium"},
                    {},
                ],
                "issued": {"date-parts": [[2019, 5]]},
                "container-title": ["Journal of Graphs"],
                "is-referenced-by-count": 7,
                "URL": "https://doi.org/10.1000/xyz",
                "type": "journal-article",
            }
        ]
    }
}


class NormalizationTests(unittest.TestCase):
    def test_openalex_normalization(self) -> None:
        (rec,) = providers.parse_openalex_response(OPENALEX_PAYLOAD)
        self.assertEqual(rec.source, "openalex")
        self.assertEqual(rec.source_id, "W123")
        self.assertEqual(rec.title, "Deep Learning")
        self.assertEqual(rec.authors, ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(rec.publication_year, 2020)
        self.assertEqual(rec.doi, "10.1000/ABC")
        self.assertEqual(rec.venue, "Journal of Tests")
        self.assertEqual(rec.citation_count, 42)
        self.assertEqual(rec.url, "https://example.org/paper")
        self.assertEqual(rec.work_type, "article")

    def test_crossref_normalization(self) -> None:
        (rec,) = providers.parse_crossref_response(CROSSREF_PAYLOAD)
        self.assertEqual(rec.source, "crossref")
        self.assertEqual(rec.doi, "10.1000/xyz")
        self.assertEqual(rec.title, "Graph Networks")
        self.assertEqual(rec.authors, ["Grace Hopper", "Some Consortium"])
        self.assertEqual(rec.publication_year, 2019)
        self.assertEqual(rec.venue, "Journal of Graphs")
        self.assertEqual(rec.citation_count, 7)
        self.assertEqual(rec.work_type, "journal-article")

    def test_openalex_missing_metadata_is_null_not_invented(self) -> None:
        (rec,) = providers.parse_openalex_response({"results": [{}]})
        self.assertEqual(rec.source, "openalex")
        self.assertEqual(rec.authors, [])
        for field in (
            "source_id", "title", "publication_year", "doi",
            "venue", "citation_count", "url", "work_type",
        ):
            self.assertIsNone(getattr(rec, field), field)

    def test_crossref_missing_metadata_is_null_not_invented(self) -> None:
        (rec,) = providers.parse_crossref_response(
            {"message": {"items": [{"title": [], "issued": {"date-parts": [[None]]}}]}}
        )
        self.assertEqual(rec.authors, [])
        for field in (
            "source_id", "title", "publication_year", "doi",
            "venue", "citation_count", "url", "work_type",
        ):
            self.assertIsNone(getattr(rec, field), field)

    def test_wrong_types_are_dropped(self) -> None:
        (rec,) = providers.parse_openalex_response(
            {"results": [{"title": 5, "cited_by_count": "9", "publication_year": True}]}
        )
        self.assertIsNone(rec.title)
        self.assertIsNone(rec.citation_count)
        self.assertIsNone(rec.publication_year)

    def test_malformed_payloads_rejected(self) -> None:
        for bad in (None, [], "x", {"results": "nope"}, {}):
            with self.assertRaises(ProviderError):
                providers.parse_openalex_response(bad)
        for bad in (None, [], {"message": []}, {"message": {"items": 3}}):
            with self.assertRaises(ProviderError):
                providers.parse_crossref_response(bad)

    def test_non_dict_items_are_skipped(self) -> None:
        self.assertEqual(
            providers.parse_openalex_response({"results": [1, None, "x"]}), []
        )


class ValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_query_rejected_without_network(self) -> None:
        with patch.object(providers, "fetch_json") as fetch:
            for q in ("", "   \n\t", None, 5):
                with self.assertRaises(ProviderError):
                    await providers.search_openalex(q)
                with self.assertRaises(ProviderError):
                    await providers.search_crossref(q)
            fetch.assert_not_called()

    async def test_overlong_query_rejected(self) -> None:
        with patch.object(providers, "fetch_json") as fetch:
            with self.assertRaises(ProviderError):
                await providers.search_crossref("a" * 301)
            fetch.assert_not_called()

    async def test_invalid_limits_rejected(self) -> None:
        with patch.object(providers, "fetch_json") as fetch:
            for limit in (0, -1, 11, 1.5, "5", True, None):
                with self.assertRaises(ProviderError, msg=repr(limit)):
                    await providers.search_openalex("x", limit)
                with self.assertRaises(ProviderError, msg=repr(limit)):
                    await providers.search_crossref("x", limit)
            fetch.assert_not_called()

    async def test_search_result_shape_and_provenance(self) -> None:
        with patch.object(providers, "fetch_json", return_value=OPENALEX_PAYLOAD) as fetch:
            result = await providers.search_openalex("  deep   learning ", 3)
        url = fetch.call_args.args[0]
        self.assertIn("per-page=3", url)
        self.assertEqual(result.source, "openalex")
        self.assertEqual(result.query, "deep learning")
        self.assertEqual(result.result_count, 1)
        self.assertTrue(all(r.source == "openalex" for r in result.records))

        with patch.object(providers, "fetch_json", return_value=CROSSREF_PAYLOAD) as fetch:
            result = await providers.search_crossref("graphs", 2)
        self.assertIn("rows=2", fetch.call_args.args[0])
        self.assertEqual(result.source, "crossref")
        self.assertTrue(all(r.source == "crossref" for r in result.records))

    async def test_upstream_failure_propagates_as_provider_error(self) -> None:
        with patch.object(providers, "fetch_json", side_effect=ProviderError("boom")):
            with self.assertRaises(ProviderError):
                await providers.search_openalex("x")

    async def test_user_agent_includes_mailto_only_when_configured(self) -> None:
        with patch.dict("os.environ", {"CROSSREF_MAILTO": "me@example.org"}):
            self.assertIn("mailto:me@example.org", providers._user_agent())
        with patch.dict("os.environ", {"CROSSREF_MAILTO": ""}):
            self.assertNotIn("mailto", providers._user_agent())


class FetchJsonTests(unittest.TestCase):
    def _http_error(self, code: int, headers: dict | None = None):
        msg = MagicMock()
        msg.get.side_effect = (headers or {}).get
        return urllib.error.HTTPError("http://x", code, "err", msg, io.BytesIO())

    def test_http_500(self) -> None:
        with patch("urllib.request.urlopen", side_effect=self._http_error(500)):
            with self.assertRaisesRegex(ProviderError, "500"):
                providers.fetch_json("http://x", {})

    def test_rate_limit_429(self) -> None:
        err = self._http_error(429, {"Retry-After": "12"})
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaisesRegex(ProviderError, "rate limit.*12"):
                providers.fetch_json("http://x", {})

    def test_network_error_and_timeout(self) -> None:
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns")):
            with self.assertRaisesRegex(ProviderError, "reach"):
                providers.fetch_json("http://x", {})
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertRaisesRegex(ProviderError, "timed out"):
                providers.fetch_json("http://x", {})

    def _response(self, body: bytes):
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    def test_invalid_json_and_oversize(self) -> None:
        with patch("urllib.request.urlopen", return_value=self._response(b"<html>")):
            with self.assertRaisesRegex(ProviderError, "malformed"):
                providers.fetch_json("http://x", {})
        big = b"x" * (providers.MAX_RESPONSE_BYTES + 1)
        with patch("urllib.request.urlopen", return_value=self._response(big)):
            with self.assertRaisesRegex(ProviderError, "size limit"):
                providers.fetch_json("http://x", {})

    def test_valid_json(self) -> None:
        with patch("urllib.request.urlopen", return_value=self._response(b'{"a": 1}')):
            self.assertEqual(providers.fetch_json("http://x", {}), {"a": 1})


if __name__ == "__main__":
    unittest.main()
