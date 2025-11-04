#!/usr/bin/env python3
"""FastAPI service exposing the FusionGuard generative assistant."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm_client import BaseLLMClient, FallbackLLMClient, LlamaCppClient
from .prompt_manager import PromptManager
from .vector_store import DocumentStore, RetrievedChunk

app = FastAPI(title="FusionGuard Support Agent", version="0.1.0")

PERSONA_GUIDELINES: Dict[str, List[str]] = {
    "support": [
        "Keep language empathetic and customer-friendly.",
        "Offer to connect the user with a human representative for complex cases.",
    ],
    "executive": [
        "Highlight business impact in monetary or KPI terms.",
        "Keep explanations concise (<=3 bullets) and avoid technical jargon.",
    ],
    "analyst": [
        "Reference relevant notebooks or model cards for follow-up validation.",
        "Mention key metrics (precision, recall, AUCPR) when explaining model behaviour.",
    ],
}


class AssistRequest(BaseModel):
    query: str = Field(..., description="User question or scenario requiring assistance.")
    top_k: int = Field(4, ge=1, le=8, description="How many documentation snippets to retrieve.")
    persona: Optional[str] = Field(
        None,
        description=f"Optional response persona. Supported: {', '.join(PERSONA_GUIDELINES)}",
    )


class AssistResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]]


@lru_cache()
def get_document_store() -> DocumentStore:
    return DocumentStore([Path("docs")])


@lru_cache()
def get_prompt_manager() -> PromptManager:
    return PromptManager()


@lru_cache()
def get_llm_client() -> BaseLLMClient:
    model_path = os.getenv("LLAMA_CPP_MODEL")
    if model_path and Path(model_path).exists():
        try:
            return LlamaCppClient(model_path)
        except RuntimeError:
            pass  # fall back to deterministic client
    return FallbackLLMClient()


def _apply_persona_guidelines(prompt_manager: PromptManager, persona: Optional[str]) -> PromptManager:
    if persona and persona in PERSONA_GUIDELINES:
        # Create a shallow copy with extra guidelines to avoid mutating cached instance
        return PromptManager(
            system_prompt=prompt_manager.system_prompt,
            safety_blurb=prompt_manager.safety_blurb,
            max_context_snippets=prompt_manager.max_context_snippets,
            additional_guidelines=PERSONA_GUIDELINES[persona],
        )
    return prompt_manager


def _serialize_sources(chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
    return [
        {
            "source": str(chunk.source),
            "score": f"{chunk.score:.3f}",
            "preview": chunk.text[:180].replace("\n", " ") + ("…" if len(chunk.text) > 180 else ""),
        }
        for chunk in chunks
    ]


@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    model = "llama.cpp" if isinstance(get_llm_client(), LlamaCppClient) else "fallback"
    return {"status": "ok", "llm_backend": model}


@app.post("/v1/assist", response_model=AssistResponse)
def generate_response(request: AssistRequest) -> AssistResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    store = get_document_store()
    prompt_manager = _apply_persona_guidelines(get_prompt_manager(), request.persona)
    llm = get_llm_client()

    retrieved = store.query(request.query, top_k=request.top_k)
    prompt = prompt_manager.build_prompt(request.query, [chunk.text for chunk in retrieved])
    answer = llm.generate(prompt, [chunk.text for chunk in retrieved])

    return AssistResponse(response=answer, sources=_serialize_sources(retrieved))
