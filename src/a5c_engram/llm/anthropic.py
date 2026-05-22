from __future__ import annotations

import json
import os

from a5c_engram.llm.base import ExtractionCandidate

EXTRACT_SYSTEM = """\
You extract durable knowledge from conversation snippets for an agent's memory.

For each piece of knowledge, classify as one of:
  - fact: stable, atomic knowledge ("project uses GraphQL")
  - event: time-stamped occurrence ("deployed v3.6 on 2026-05-18")
  - instruction: procedural/behavioural rule ("never deploy on Friday")
  - task: ephemeral work item ("review PR #42 tomorrow")

For facts and instructions, generate a snake_case fact_key naming the topic
(e.g. "api_style", "deploy_policy") so future versions can supersede old ones.

Return JSON: {"memories": [{"type": ..., "content": ..., "fact_key": ...,
"confidence": 0-1, "evidence": "quoted text"}]}. If no durable knowledge,
return {"memories": []}.
"""

HYDE_SYSTEM = """\
Given a user query, write a brief hypothetical answer (1-2 sentences) as if
you knew the answer. This is used for HyDE retrieval — do not say you don't
know. Plausibility beats accuracy.
"""

SYNTH_SYSTEM = """\
Given a query and supporting memories, write a 1-2 sentence answer grounded
in those memories. If the memories contradict, prefer the most recent. If
they don't answer the query, say "no relevant memory".
"""


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicLLM needs the anthropic package. "
                "Install with: uv add 'a5c-engram[anthropic]'"
            ) from e
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._model = model

    def _call(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def extract(self, text: str) -> list[ExtractionCandidate]:
        raw = self._call(EXTRACT_SYSTEM, text)
        try:
            payload = json.loads(_strip_codefence(raw))
        except json.JSONDecodeError:
            return []
        out = []
        for m in payload.get("memories", []):
            try:
                out.append(
                    ExtractionCandidate(
                        type=m["type"],
                        content=m["content"],
                        fact_key=m.get("fact_key"),
                        confidence=float(m.get("confidence", 0.5)),
                        evidence=m.get("evidence"),
                    )
                )
            except (KeyError, ValueError):
                continue
        return out

    def hyde(self, query: str) -> str:
        return self._call(HYDE_SYSTEM, query, max_tokens=200).strip()

    def synthesise(self, query: str, memories: list[str]) -> str:
        joined = "\n".join(f"- {m}" for m in memories[:12])
        prompt = f"Query: {query}\n\nMemories:\n{joined}"
        return self._call(SYNTH_SYSTEM, prompt, max_tokens=200).strip()


def _strip_codefence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()
