import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from backend import config as _config


logger = logging.getLogger(__name__)


class YouTubeService:
    API_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
    SEARCH_STOP_WORDS = {
        "a", "about", "an", "and", "basic", "basics", "beginner", "beginners",
        "course", "explain", "explained", "for", "from", "guide", "how", "in",
        "intro", "introduction", "learn", "lesson", "of", "on", "overview",
        "the", "tutorial", "tutorials", "video", "videos", "with",
    }

    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY", "").strip()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> dict:
        if not self.api_key:
            return {
                "videos": [],
                "message": (
                    "YouTube recommendations need configuration: set "
                    "YOUTUBE_API_KEY and enable YouTube Data API v3."
                ),
            }
        if not query.strip():
            return {"videos": [], "message": "No study topic was available to search."}

        params = urllib.parse.urlencode(
            {
                "part": "snippet",
                "type": "video",
                "q": query.strip(),
                "maxResults": min(max(max_results, 1), 5),
                "order": "relevance",
                "safeSearch": "moderate",
                "key": self.api_key,
            }
        )
        request = urllib.request.Request(
            f"{self.API_URL}?{params}",
            headers={"Accept": "application/json"},
            method="GET",
        )

        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                details = error.read().decode("utf-8", errors="replace")
            finally:
                error.close()
            return {"videos": [], "message": self._http_error_message(error.code, details)}
        except (urllib.error.URLError, TimeoutError, OSError):
            logger.warning("YouTube search failed due to a network error.")
            return {
                "videos": [],
                "message": "YouTube recommendations are temporarily unavailable because the search could not connect.",
            }
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("YouTube returned an unreadable search response.")
            return {
                "videos": [],
                "message": "YouTube returned an unreadable search response. Please try again later.",
            }

        if not isinstance(payload, dict):
            return {
                "videos": [],
                "message": "YouTube returned an invalid search response.",
            }

        videos = []
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query.casefold())
            if term not in self.SEARCH_STOP_WORDS
        }
        items = payload.get("items")
        if isinstance(items, list):
            for item in items:
                video = self._parse_video(item)
                if video and (
                    not query_terms
                    or query_terms.intersection(
                        re.findall(
                            r"[a-z0-9]+",
                            f"{video['title']} {video['channel']}".casefold(),
                        )
                    )
                ):
                    videos.append(video)
                if len(videos) == 5:
                    break

        if not videos:
            return {
                "videos": [],
                "message": f"No relevant YouTube videos were found for “{query}”.",
            }
        return {"videos": videos, "message": None}

    @classmethod
    def _parse_video(cls, item: object) -> dict | None:
        if not isinstance(item, dict):
            return None
        video_id = item.get("id")
        snippet = item.get("snippet")
        if not isinstance(video_id, dict) or not isinstance(snippet, dict):
            return None
        video_id = video_id.get("videoId")
        title = snippet.get("title")
        channel = snippet.get("channelTitle")
        thumbnails = snippet.get("thumbnails")
        if (
            not isinstance(video_id, str)
            or not cls.VIDEO_ID_PATTERN.fullmatch(video_id)
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(channel, str)
            or not channel.strip()
            or not isinstance(thumbnails, dict)
        ):
            return None

        thumbnail = next(
            (
                image.get("url")
                for size in ("high", "medium", "default")
                if isinstance((image := thumbnails.get(size)), dict)
                and isinstance(image.get("url"), str)
                and image["url"].startswith("https://i.ytimg.com/")
            ),
            None,
        )
        if not thumbnail:
            return None

        return {
            "id": video_id,
            "title": title.strip(),
            "channel": channel.strip(),
            "thumbnail": thumbnail,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

    @staticmethod
    def _http_error_message(status_code: int, details: str) -> str:
        try:
            payload = json.loads(details)
        except json.JSONDecodeError:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        errors = error.get("errors", []) if isinstance(error, dict) else []
        reasons = {
            item.get("reason")
            for item in errors
            if isinstance(item, dict) and isinstance(item.get("reason"), str)
        }

        if status_code == 429 or reasons.intersection(
            {
                "quotaExceeded",
                "dailyLimitExceeded",
                "userRateLimitExceeded",
                "rateLimitExceeded",
            }
        ):
            return "YouTube search is temporarily unavailable because the API quota has been exceeded."
        if reasons.intersection(
            {"keyInvalid", "accessNotConfigured", "youtubeSignupRequired"}
        ) or status_code == 400:
            return (
                "YouTube API configuration is invalid. Check YOUTUBE_API_KEY and "
                "confirm YouTube Data API v3 is enabled for that key."
            )
        if status_code == 403:
            return (
                "YouTube denied the search request. Check the API key permissions "
                "and YouTube Data API v3 quota."
            )
        logger.warning("YouTube search returned HTTP %s.", status_code)
        return "YouTube recommendations could not be retrieved right now."


youtube_service = YouTubeService()
