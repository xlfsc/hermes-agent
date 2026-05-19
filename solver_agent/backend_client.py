"""HTTP client for the math-solving backend service.

Centralised wrapper around the backend endpoints used by both the MCP server
and the evolutionary solver agent:

  * POST /api/solve_problem   - run a single LLM solving attempt.
  * POST /api/analysis_verify - step-by-step verification of an analysis.
  * GET  /api/health          - liveness probe.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


DEFAULT_BASE_URL = "http://172.168.80.46:8000"
DEFAULT_SOLVE_TIMEOUT = 600
DEFAULT_VERIFY_TIMEOUT = 300


class BackendClient:
    """Thin synchronous client around the math solver backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        solve_timeout: Optional[int] = None,
        verify_timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("SOLVER_API_BASE", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("SOLVER_API_KEY", "")
        self.solve_timeout = solve_timeout or int(os.getenv("SOLVER_SOLVE_TIMEOUT", str(DEFAULT_SOLVE_TIMEOUT)))
        self.verify_timeout = verify_timeout or int(os.getenv("SOLVER_VERIFY_TIMEOUT", str(DEFAULT_VERIFY_TIMEOUT)))

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=timeout)
        except requests.exceptions.Timeout:
            return {"ok": False, "error": f"request timed out after {timeout}s", "endpoint": path}
        except requests.exceptions.RequestException as exc:
            return {"ok": False, "error": f"request failed: {exc}", "endpoint": path}

        if resp.status_code >= 400:
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}",
                "endpoint": path,
                "body": resp.text[:1000],
            }

        try:
            data = resp.json()
        except ValueError:
            return {"ok": False, "error": "invalid JSON response", "endpoint": path, "body": resp.text[:1000]}

        if isinstance(data, dict) and "ok" not in data:
            data["ok"] = True
        return data

    def solve_problem(
        self,
        text_input: str = "",
        *,
        image_base64: Optional[str] = None,
        url_image: Optional[str] = None,
        solve_platform: str = "DeepSeek",
        solve_model: str = "deepseek-v3.2",
        thinking: bool = True,
        verify: bool = False,
        verify_platform: str = "Qwen",
        verify_model: str = "qwen3-235b-a22b",
        verify_round: int = 5,
        question_type: str = "",
        prompt: str = "",
        example: str = "",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "text_input": text_input,
            "image_base64": image_base64 or "",
            "url_image": url_image or "",
            "solve_platform": solve_platform,
            "solve_model": solve_model,
            "thinking": thinking,
            "verify": verify,
            "verify_platform": verify_platform,
            "verify_model": verify_model,
            "verify_round": verify_round,
            "question_type": question_type,
            "prompt": prompt or "",
            "example": example or "",
        }
        return self._post("/api/solve_problem", payload, self.solve_timeout)

    def verify_analysis(
        self,
        stem_text: str,
        analysis: str,
        *,
        verify_platform: str = "Qwen",
        verify_model: str = "qwen3-235b-a22b",
    ) -> Dict[str, Any]:
        payload = {
            "stem_text": stem_text,
            "analysis": analysis,
            "verify_platform": verify_platform,
            "verify_model": verify_model,
        }
        return self._post("/api/analysis_verify", payload, self.verify_timeout)

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/health"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=10)
            ok = resp.status_code < 400
            return {"ok": ok, "status_code": resp.status_code, "body": resp.text[:500]}
        except requests.exceptions.RequestException as exc:
            return {"ok": False, "error": str(exc)}
