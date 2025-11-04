#!/usr/bin/env python3
"""Prompt utilities for the churn/fraud support agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Iterable, List


DEFAULT_SYSTEM_PROMPT = dedent(
    """
    You are FusionGuard's virtual analytics assistant. Provide concise, business-friendly
    explanations for data science outputs (fraud detection, churn modelling, data quality checks).
    Always include actionable next steps when possible. Keep tone professional, empathetic, and
    avoid over-promising what the models can do. If a question is out of scope, politely state the
    limitation and suggest an alternative resource.
    """
).strip()

SAFETY_GUARDRAILS = dedent(
    """
    Safety & Compliance checklist:
    - Do not provide legal or regulatory advice.
    - Flag if data privacy or fairness concerns are raised; encourage contacting compliance.
    - Avoid leaking sensitive customer identifiers. Use aggregate terms (e.g., "this segment").
    - If confidence is low, emphasise that results should be validated by a human analyst.
    """
).strip()


@dataclass
class PromptManager:
    """Compose prompts with system instructions, retrieved context, and guardrails."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    safety_blurb: str = SAFETY_GUARDRAILS
    max_context_snippets: int = 5
    additional_guidelines: List[str] = field(default_factory=list)

    def build_prompt(self, query: str, context_snippets: Iterable[str]) -> str:
        """Return a formatted prompt for the LLM."""
        snippets = list(context_snippets)[: self.max_context_snippets]
        context_block = "\n\n".join(f"[Source {idx+1}]\n{chunk.strip()}" for idx, chunk in enumerate(snippets))
        guidelines_block = ""
        if self.additional_guidelines:
            guidelines_block = "\nAdditional business rules:\n- " + "\n- ".join(self.additional_guidelines)

        prompt = dedent(
            f"""
            {self.system_prompt}

            {self.safety_blurb}
            {guidelines_block}

            Context to reference:
            {context_block or 'No supporting documents were retrieved. Use general project knowledge.'}

            User question:
            {query.strip()}

            Provide a structured answer with:
            1. Brief summary (2 sentences)
            2. Recommended actions / talking points
            3. Optional cautionary notes if relevant
            """
        ).strip()
        return prompt
