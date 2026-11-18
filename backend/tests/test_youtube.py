import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from backend.services.youtube_service import YouTubeService


def youtube_response(items):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps({"items": items}).encode()
    return response


def video_item(video_id="AbCdEfGhI_1", title="Database normalization tutorial"):
    return {
        "id": {"kind": "youtube#video", "videoId": video_id},
        "snippet": {
            "title": title,
            "channelTitle": "Computer Science Lessons",
            "thumbnails": {
                "medium": {"url": "https://i.ytimg.com/vi/sample/mqdefault.jpg"}
            },
        },
    }


class YouTubeServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = YouTubeService()
        self.service.api_key = "test-youtube-key"

    def test_search_builds_topic_specific_youtube_request_and_real_video_links(self):
        with patch(
            "backend.services.youtube_service.urllib.request.urlopen",
            return_value=youtube_response(
                [
                    video_item(
                        video_id="UnRelAtEd11",
                        title="Unrelated cooking recipe",
                    ),
                    video_item(),
                ]
            ),
        ) as urlopen:
            result = self.service.search("database normalization examples")

        request = urlopen.call_args.args[0]
        self.assertIn("database+normalization+examples", request.full_url)
        self.assertIn("type=video", request.full_url)
        self.assertEqual(len(result["videos"]), 1)
        self.assertEqual(result["videos"][0]["id"], "AbCdEfGhI_1")
        self.assertEqual(
            result["videos"][0]["url"],
            "https://www.youtube.com/watch?v=AbCdEfGhI_1",
        )
        self.assertEqual(result["videos"][0]["channel"], "Computer Science Lessons")
        self.assertEqual(result["videos"][0]["thumbnail"], "https://i.ytimg.com/vi/sample/mqdefault.jpg")

    def test_missing_api_key_is_reported_without_request(self):
        self.service.api_key = ""
        with patch(
            "backend.services.youtube_service.urllib.request.urlopen"
        ) as urlopen:
            result = self.service.search("DBMS")

        self.assertEqual(result["videos"], [])
        self.assertIn("YOUTUBE_API_KEY", result["message"])
        urlopen.assert_not_called()

    def test_invalid_key_error_is_reported(self):
        error = urllib.error.HTTPError(
            self.service.API_URL,
            400,
            "Bad Request",
            {},
            io.BytesIO(
                b'{"error":{"errors":[{"reason":"keyInvalid"}]}}'
            ),
        )
        with patch(
            "backend.services.youtube_service.urllib.request.urlopen",
            side_effect=error,
        ):
            result = self.service.search("DBMS")

        self.assertIn("configuration is invalid", result["message"])

    def test_quota_error_is_reported(self):
        error = urllib.error.HTTPError(
            self.service.API_URL,
            403,
            "Forbidden",
            {},
            io.BytesIO(
                b'{"error":{"errors":[{"reason":"quotaExceeded"}]}}'
            ),
        )
        with patch(
            "backend.services.youtube_service.urllib.request.urlopen",
            side_effect=error,
        ):
            result = self.service.search("DBMS")

        self.assertIn("quota has been exceeded", result["message"])

    def test_network_failure_is_reported(self):
        with patch(
            "backend.services.youtube_service.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            result = self.service.search("DBMS")

        self.assertIn("could not connect", result["message"])

    def test_empty_and_invalid_results_are_not_returned_as_videos(self):
        with patch(
            "backend.services.youtube_service.urllib.request.urlopen",
            return_value=youtube_response(
                [video_item(video_id="not-a-valid-video-id"), {"invalid": True}]
            ),
        ):
            result = self.service.search("DBMS")

        self.assertEqual(result["videos"], [])
        self.assertIn("No relevant YouTube videos", result["message"])


if __name__ == "__main__":
    unittest.main()
