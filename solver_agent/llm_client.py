"""OpenAI-compatible LLM client used by the knowledge / evolutionary modules.

The default endpoint reuses the same custom provider that the solver agent
hermes_home/config.yaml points at (gemma4 @ 171.214.10.150:11600). All values
can be overridden via env vars so the same module can serve multiple roles
(retrieval LLM, reflection LLM, etc.).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


DEFAULT_BASE_URL = "http://171.214.10.150:11600/v1/"
DEFAULT_API_KEY = "mysecurekey123"
DEFAULT_MODEL = "gemma4"


class LLMClient:
    """Minimal OpenAI-compatible chat client (no streaming, no tools)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (base_url or os.getenv("KB_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("KB_LLM_API_KEY", DEFAULT_API_KEY)
        self.model = model or os.getenv("KB_LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format_json: bool = False,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected LLM response shape: {data!r}") from exc

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Call ``chat`` and parse the response as JSON, tolerating fenced output."""

        raw = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise
