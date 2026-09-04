"""Small auditable HTTP transport for an OpenAI-compatible endpoint.

Works against either a ``/chat/completions`` URL or a ``/responses`` URL. Usage is
retained; private reasoning is not. ``reasoning_content``, ``reasoning_details`` and
reasoning output items are never returned to the caller.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError

USER_AGENT = "tastebench/1.0"


class ChatTransport:
    def __init__(
        self,
        *,
        api: str,
        api_key: str,
        model: str,
        request_fields: Mapping[str, Any] | None = None,
        reasoning_content_fallback: bool = False,
        timeout_seconds: float = 900,
        retries: int = 4,
    ) -> None:
        if not api:
            raise ValueError("an API URL is required; set `api` in the config or pass --api")
        self.api = api
        self.api_key = api_key
        self.model = model
        self.request_fields = dict(request_fields or {})
        self.reasoning_content_fallback = reasoning_content_fallback
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    @property
    def responses_mode(self) -> bool:
        return self.api.rstrip("/").endswith("/responses")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urlrequest.Request(
                    endpoint, data=body, headers=self._headers(), method="POST"
                )
                with urlrequest.urlopen(request, timeout=self.timeout_seconds) as response:
                    value = json.loads(response.read().decode())
                if not isinstance(value, dict):
                    raise ValueError("endpoint response is not a JSON object")
                return value
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code == 403 and (
                    "SAFETY_CHECK_TYPE_" in detail or "Content violates usage guidelines" in detail
                ):
                    # A provider policy refusal is a model outcome, not an
                    # infrastructure failure. Keep only a normalized category;
                    # do not persist the provider's response body.
                    return {"_tastebench_provider_refusal": "safety_policy"}
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
            except Exception as exc:  # noqa: BLE001
                last = exc
            if attempt + 1 < self.retries:
                time.sleep(min(30, 2 ** (attempt + 1)))
        raise RuntimeError(f"request failed after {self.retries} attempts: {last}")

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        started = time.monotonic()
        request_key = "input" if self.responses_mode else "messages"
        value = self._post(
            self.api, {"model": self.model, request_key: messages, **self.request_fields}
        )
        refusal = value.get("_tastebench_provider_refusal")
        if refusal:
            return {
                "text": "",
                "response_model": self.model,
                "finish_reason": "content_filter",
                "usage": {},
                "provider_metadata": {"refusal": str(refusal)},
                "latency_seconds": round(time.monotonic() - started, 3),
            }
        if self.responses_mode:
            text_parts = []
            for item in value.get("output") or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for content in item.get("content") or []:
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        text_parts.append(str(content.get("text") or ""))
            raw_usage: dict[str, Any] = (
                value["usage"] if isinstance(value.get("usage"), dict) else {}
            )
            usage: dict[str, Any] = {
                "prompt_tokens": int(raw_usage.get("input_tokens") or 0),
                "completion_tokens": int(raw_usage.get("output_tokens") or 0),
                "total_tokens": int(raw_usage.get("total_tokens") or 0),
                "completion_tokens_details": (
                    raw_usage.get("output_tokens_details")
                    if isinstance(raw_usage.get("output_tokens_details"), dict)
                    else {}
                ),
            }
            status = str(value.get("status") or "")
            return {
                "text": "\n".join(part for part in text_parts if part),
                "response_model": value.get("model"),
                "finish_reason": "stop" if status == "completed" else status or None,
                "usage": usage,
                "provider_metadata": {
                    "response_status": status or None,
                    "incomplete_details": value.get("incomplete_details"),
                },
                "latency_seconds": round(time.monotonic() - started, 3),
            }

        choice = (value.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        if not content and self.reasoning_content_fallback:
            # Some providers place the entire response in reasoning_content. Extract
            # only the explicit final label; never retain the private reasoning text.
            reasoning: str = (
                message["reasoning_content"]
                if isinstance(message.get("reasoning_content"), str)
                else ""
            )
            matches = re.findall(r"(?:^|\n)\s*ANSWER:\s*([A-Z])\s*$", reasoning, re.I)
            if matches:
                content = f"ANSWER: {matches[-1].upper()}"
        # Never return reasoning_content/reasoning_details/reasoning_items.
        return {
            "text": content,
            "response_model": value.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": value.get("usage") if isinstance(value.get("usage"), dict) else {},
            "provider_metadata": (
                value.get("provider_metadata")
                if isinstance(value.get("provider_metadata"), dict)
                else {}
            ),
            "latency_seconds": round(time.monotonic() - started, 3),
        }
