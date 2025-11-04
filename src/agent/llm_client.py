#!/usr/bin/env python3
"""Thin abstraction around optional LLM backends."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent, shorten
from typing import Sequence


class BaseLLMClient:
    """Interface for LLM backends used by the support agent."""

    def generate(self, prompt: str, context_chunks: Sequence[str]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LlamaCppClient(BaseLLMClient):
    """Wrapper around llama.cpp for local inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        n_threads: int | None = None,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "llama_cpp Python bindings are required for LlamaCppClient. "
                "Install via `pip install llama-cpp-python` or choose the fallback client."
            ) from exc

        self.max_tokens = max_tokens
        self.temperature = temperature
        n_threads = n_threads or max(os.cpu_count() or 2, 2)
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=n_threads,
        )

    def generate(self, prompt: str, context_chunks: Sequence[str]) -> str:
        response = self._llama(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["User question:"],
        )
        text = response["choices"][0]["text"].strip()
        return text


class FallbackLLMClient(BaseLLMClient):
    """Deterministic response builder used when a real LLM is unavailable."""

    def __init__(self, *, include_disclaimer: bool = True) -> None:
        self.include_disclaimer = include_disclaimer

    def _summarise_context(self, context_chunks: Sequence[str], limit: int = 3) -> str:
        summaries = []
        for idx, chunk in enumerate(context_chunks[:limit]):
            snippet = shorten(" ".join(chunk.split()), width=220, placeholder="…")
            summaries.append(f"- Source {idx+1}: {snippet}")
        if not summaries:
            summaries.append("- No project documentation snippets were available for this query.")
        return "\n".join(summaries)

    def generate(self, prompt: str, context_chunks: Sequence[str]) -> str:
        summary = self._summarise_context(context_chunks)
        disclaimer = ""
        if self.include_disclaimer:
            disclaimer = (
                "\n\n_Caution: This response was produced by a rule-based fallback. "
                "Run the FastAPI service with a local LLM (e.g., llama.cpp) for richer language generation._"
            )

        answer = dedent(
            f"""
            ### Summary
            Leveraging the highlighted documentation, here is a recommended response crafted without an LLM:

            {summary}

            ### Suggested talking points
            - Reference the top sources above when addressing the customer.
            - Emphasise that metrics should be validated in the analytics notebook or model card.
            - Offer to escalate to a human analyst if deeper investigation is required.
            {disclaimer}
            """
        ).strip()
        return answer
