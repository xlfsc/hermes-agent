"""HTTP client for the math-solving backend service.

Centralised wrapper around the backend endpoints used by both the MCP server
and the evolutionary solver agent:

  * POST /api/solve_problem   - run a single LLM solving attempt.
  * POST /api/analysis_verify - step-by-step verification of an analysis.
  * GET  /api/health          - liveness probe.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://172.168.80.46:8000"
DEFAULT_SOLVE_TIMEOUT = 600
DEFAULT_VERIFY_TIMEOUT = 300


class BackendClient:
    """Async client around the math solver backend."""

    def __init__(
            self,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            solve_timeout: Optional[int] = None,
            verify_timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (
                base_url or os.getenv("SOLVER_API_BASE", DEFAULT_BASE_URL)
        ).rstrip("/")

        self.api_key = api_key if api_key is not None else os.getenv(
            "SOLVER_API_KEY", "")
        self.solve_timeout = solve_timeout or int(
            os.getenv("SOLVER_SOLVE_TIMEOUT", str(DEFAULT_SOLVE_TIMEOUT)))
        self.verify_timeout = verify_timeout or int(
            os.getenv("SOLVER_VERIFY_TIMEOUT", str(DEFAULT_VERIFY_TIMEOUT)))

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post(
            self, path: str, payload: Dict[str, Any], timeout: int
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers=self._headers()
                )
        except httpx.TimeoutException:
            logger.warning(
                "后端请求超时 | 路径=%s | 超时=%ds",
                path,
                timeout
            )
            return {
                "ok": False,
                "error": f"request timed out after {timeout}s",
                "endpoint": path
            }
        except httpx.RequestError as exc:
            logger.warning(
                "后端请求异常 | 路径=%s | 错误=%s",
                path,
                exc
            )
            return {
                "ok": False,
                "error": f"request failed: {exc}",
                "endpoint": path
            }

        if resp.status_code >= 400:
            logger.warning(
                "后端返回 HTTP 错误 | 路径=%s | 状态码=%d",
                path,
                resp.status_code
            )
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}",
                "endpoint": path,
                "body": resp.text[:1000],
            }

        try:
            data = resp.json()
        except ValueError:
            logger.warning("后端返回非 JSON | 路径=%s", path)
            return {
                "ok": False,
                "error": "invalid JSON response",
                "endpoint": path,
                "body": resp.text[:1000]
            }

        if isinstance(data, dict) and "ok" not in data:
            data["ok"] = True
        return data

    async def solve_problem(
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
        return await self._post(
            "/api/solve_problem", payload, self.solve_timeout
        )

    async def verify_analysis(
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
        return await self._post(
            "/api/analysis_verify", payload, self.verify_timeout
        )

    async def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/health"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=self._headers())
            ok = resp.status_code < 400
            return {
                "ok": ok,
                "status_code": resp.status_code,
                "body": resp.text[:500]
            }
        except httpx.RequestError as exc:
            return {"ok": False, "error": str(exc)}
