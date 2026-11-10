"""Local Ollama API client; no model-pull or hosted-inference endpoints are used."""

from dataclasses import dataclass

import httpx

from pda.summary_config import SummaryError, SummarySettings


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    digest: str
    server_version: str


class OllamaClient:
    def __init__(self, settings: SummarySettings, transport=None):
        self.settings = settings
        self.http = httpx.Client(
            base_url=settings.endpoint.rstrip("/"),
            timeout=httpx.Timeout(settings.timeout_seconds, connect=5),
            transport=transport,
            trust_env=False,
        )
        self._send_think = True

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.http.request(method, path, **kwargs)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("Expected an object")
            if data.get("error"):
                raise SummaryError(f"Ollama inference failed: {str(data['error'])[:300]}")
            return data
        except httpx.TimeoutException as exc:
            raise SummaryError(
                "Ollama timed out. Retry, or increase PDA_SUMMARY_TIMEOUT_SECONDS."
            ) from exc
        except httpx.ConnectError as exc:
            raise SummaryError(
                "Cannot connect to Ollama. Start Ollama locally, then retry."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise SummaryError(
                    f"The local model is unavailable. Run `ollama pull {self.settings.model}` during setup, then retry."
                ) from exc
            raise SummaryError(
                f"Ollama returned HTTP {exc.response.status_code}. Check its log and retry."
            ) from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise SummaryError(
                "Ollama returned an invalid response. Check its log and retry."
            ) from exc

    def identity(self) -> ModelIdentity:
        data = self._request("GET", "/api/tags")
        models = data.get("models", [])
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise SummaryError("Ollama returned an invalid model list. Update Ollama and retry.")
        for item in models:
            if item.get("name", item.get("model")) == self.settings.model:
                if item.get("remote_host") or item.get("remote_model"):
                    raise SummaryError(
                        "This Ollama model uses a remote host. Install local Qwen weights instead."
                    )
                if not item.get("digest"):
                    raise SummaryError(
                        "Ollama did not report a model digest; its version cannot be verified."
                    )
                version = self._request("GET", "/api/version").get("version", "unknown")
                shown = self._request("POST", "/api/show", json={"model": self.settings.model})
                if shown.get("remote_host") or shown.get("remote_model"):
                    raise SummaryError(
                        "This model delegates to a remote host; local weights are required."
                    )
                info = shown.get("model_info", {})
                if not isinstance(info, dict):
                    raise SummaryError(
                        "Ollama returned invalid model information. Update Ollama and retry."
                    )
                limits = [
                    v
                    for k, v in info.items()
                    if k.endswith(".context_length") and isinstance(v, int)
                ]
                if limits and self.settings.num_ctx > min(limits):
                    raise SummaryError(
                        "Configured context exceeds the installed model's context limit."
                    )
                return ModelIdentity(self.settings.model, item["digest"], str(version))
        raise SummaryError(
            f"{self.settings.model} is not installed locally. Run `ollama pull {self.settings.model}` during setup."
        )

    def fits(self, messages: list[dict]) -> bool:
        # A conservative UTF-8 byte bound for Qwen's byte-level tokenizer, plus template
        # and output reserves. This intentionally favors more chunks over truncation.
        upper_bound = sum(len(m["content"].encode("utf-8")) for m in messages)
        return (
            upper_bound + self.settings.template_reserve + self.settings.num_predict
            <= self.settings.num_ctx
        )

    def chat(self, messages: list[dict]) -> str:
        if not self.fits(messages):
            raise SummaryError(
                "A summarization input exceeds the configured context budget; no paper text was truncated."
            )
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": {
                "num_ctx": self.settings.num_ctx,
                "num_predict": self.settings.num_predict,
                "temperature": self.settings.temperature,
                "seed": self.settings.seed,
            },
        }
        if self._send_think:
            payload["think"] = self.settings.think
        try:
            data = self._request("POST", "/api/chat", json=payload)
        except SummaryError as exc:
            cause = exc.__cause__
            if (
                isinstance(cause, httpx.HTTPStatusError)
                and cause.response.status_code == 400
                and "think" in cause.response.text.casefold()
            ):
                self._send_think = False
                payload.pop("think", None)
                data = self._request("POST", "/api/chat", json=payload)
            else:
                raise
        if data.get("done") is not True or data.get("done_reason") in {
            "length",
            "context_length",
            "max_tokens",
        }:
            raise SummaryError(
                "Ollama stopped before completing its response. Increase the generation/context budget and retry."
            )
        count = data.get("prompt_eval_count")
        if not isinstance(count, int) or count + self.settings.num_predict > self.settings.num_ctx:
            raise SummaryError(
                "Ollama could not verify a safe prompt/context size. Update Ollama or adjust the context budget."
            )
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise SummaryError(
                "Ollama returned no summary text. Retry or check model/thinking settings."
            )
        return content.strip()
