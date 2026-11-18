import unittest
from unittest.mock import patch

import httpx

from backend.services.web_search_service import (
    WebSearchService,
    build_research_context,
    should_search_web,
)


class WebSearchServiceTests(unittest.TestCase):
    def test_search_collects_content_and_removes_duplicate_urls(self):
        response = httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Official docs",
                        "url": "https://example.com/docs?utm_source=search",
                        "raw_content": "Current documentation text",
                    },
                    {
                        "title": "Duplicate docs",
                        "url": "https://EXAMPLE.com/docs/#section",
                        "content": "Duplicate copy",
                    },
                    {
                        "title": "Release notes",
                        "url": "https://example.org/releases",
                        "content": "Recent releases",
                        "published_date": "2026-10-15",
                    },
                ]
            },
        )
        service = WebSearchService(api_key="test-key")

        with patch(
            "backend.services.web_search_service.httpx.post",
            return_value=response,
        ) as post:
            result = service.search("  latest   library docs ")

        self.assertEqual(result["status"], "success")
        self.assertEqual([source["id"] for source in result["sources"]], ["S1", "S2"])
        self.assertEqual(result["sources"][0]["content"], "Current documentation text")
        self.assertEqual(result["sources"][1]["published_date"], "2026-10-15")
        self.assertEqual(post.call_args.kwargs["json"]["query"], "latest library docs")
        self.assertTrue(post.call_args.kwargs["json"]["include_raw_content"])

    def test_missing_api_key_is_graceful_and_does_not_make_a_request(self):
        service = WebSearchService(api_key="")
        with patch("backend.services.web_search_service.httpx.post") as post:
            result = service.search("latest Python release")

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["sources"], [])
        post.assert_not_called()

    def test_invalid_api_key_is_graceful(self):
        service = WebSearchService(api_key="invalid-key")
        with patch(
            "backend.services.web_search_service.httpx.post",
            return_value=httpx.Response(403),
        ):
            result = service.search("latest Python release")

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("credentials are invalid", result["message"])

    def test_rate_limit_is_reported_without_exposing_provider_details(self):
        service = WebSearchService(api_key="test-key")
        with patch(
            "backend.services.web_search_service.httpx.post",
            return_value=httpx.Response(429),
        ):
            result = service.search("current framework version")

        self.assertEqual(result["status"], "rate_limited")
        self.assertEqual(result["sources"], [])
        self.assertIn("rate limited", result["message"])

    def test_successful_queries_are_cached(self):
        service = WebSearchService(api_key="test-key")
        response = httpx.Response(200, json={"results": []})
        with patch(
            "backend.services.web_search_service.httpx.post",
            return_value=response,
        ) as post:
            first = service.search("current Python release")
            second = service.search("current Python release")

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        post.assert_called_once()

    def test_cache_size_is_bounded(self):
        service = WebSearchService(api_key="test-key")
        service.CACHE_MAX_ENTRIES = 1
        with patch(
            "backend.services.web_search_service.httpx.post",
            return_value=httpx.Response(200, json={"results": []}),
        ) as post:
            service.search("first query")
            service.search("second query")
            service.search("first query")

        self.assertEqual(post.call_count, 3)

    def test_research_mode_requests_deeper_multiple_sources(self):
        service = WebSearchService(api_key="test-key")
        with patch(
            "backend.services.web_search_service.httpx.post",
            return_value=httpx.Response(200, json={"results": []}),
        ) as post:
            service.search("compare current approaches", deep=True)

        self.assertEqual(post.call_args.kwargs["json"]["search_depth"], "advanced")
        self.assertEqual(post.call_args.kwargs["json"]["max_results"], 8)

    def test_search_gate_and_source_context(self):
        self.assertTrue(should_search_web("Search the web for current Python docs"))
        self.assertTrue(should_search_web("Verify the latest release date"))
        self.assertFalse(should_search_web("Explain recursion with an example"))
        context = build_research_context(
            [{
                "id": "S1",
                "title": "Release notes",
                "url": "https://example.com/releases",
                "published_date": "2026-10-15",
                "content": "A supported fact.",
            }]
        )
        self.assertIn("[S1] Release notes", context)
        self.assertIn("conflicting reports", context)
        self.assertIn("assumptions", context)


if __name__ == "__main__":
    unittest.main()