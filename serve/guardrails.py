"""Lightweight safety guardrails for the serving layer.

Provides:
  - Input validation (length, known adversarial patterns)
  - Output format constraints (max tokens, citation requirements)
  - Refusal behavior for unsafe or unanswerable prompts
  - System prompt template

Usage:
    from serve.guardrails import GuardrailEngine
    engine = GuardrailEngine()
    result = engine.check_input(user_prompt)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: Final[str] = (
    "You are a helpful AI assistant specialized in machine learning topics. "
    "Follow these rules strictly:\n"
    "1. Only answer questions within your knowledge. If unsure, say so.\n"
    "2. Never fabricate citations, paper titles, or author names.\n"
    "3. If asked to produce harmful content, politely decline.\n"
    "4. Keep responses concise and cite sources when possible.\n"
    "5. Do not reproduce PII or sensitive data from training."
)

# ---------------------------------------------------------------------------
# Known failure / adversarial prompts
# ---------------------------------------------------------------------------
KNOWN_FAILURE_PROMPTS: Final[list[dict[str, str]]] = [
    {
        "pattern": "ignore previous instructions",
        "category": "prompt_injection",
        "action": "refuse",
    },
    {
        "pattern": "you are now",
        "category": "jailbreak",
        "action": "refuse",
    },
    {
        "pattern": "pretend you are",
        "category": "jailbreak",
        "action": "refuse",
    },
    {
        "pattern": "invent a citation",
        "category": "hallucination_trap",
        "action": "refuse",
    },
    {
        "pattern": "make up a reference",
        "category": "hallucination_trap",
        "action": "refuse",
    },
    {
        "pattern": "write malware",
        "category": "harmful_content",
        "action": "refuse",
    },
    {
        "pattern": "how to hack",
        "category": "harmful_content",
        "action": "refuse",
    },
    {
        "pattern": "give me someone's ssn",
        "category": "pii_extraction",
        "action": "refuse",
    },
    {
        "pattern": "what is .* personal",
        "category": "pii_extraction",
        "action": "refuse",
    },
]

REFUSAL_RESPONSE: Final[str] = (
    "I'm unable to fulfill this request. It falls outside my safety guidelines. "
    "I can help with machine learning, data science, and related technical topics."
)

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------
MAX_INPUT_LENGTH: Final[int] = 2000
MAX_OUTPUT_TOKENS: Final[int] = 512
MIN_INPUT_LENGTH: Final[int] = 3


# ---------------------------------------------------------------------------
# Guardrail result
# ---------------------------------------------------------------------------
@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    allowed: bool
    reason: str = ""
    category: str = ""
    modified_prompt: str = ""


# ---------------------------------------------------------------------------
# Guardrail engine
# ---------------------------------------------------------------------------
class GuardrailEngine:
    """Stateless guardrail checker for input and output."""

    def __init__(self, safe_mode: bool = True) -> None:
        self.safe_mode = safe_mode
        self._patterns = [
            (re.compile(p["pattern"], re.IGNORECASE), p["category"], p["action"])
            for p in KNOWN_FAILURE_PROMPTS
        ]

    def check_input(self, prompt: str) -> GuardrailResult:
        """Validate user input. Returns GuardrailResult."""
        # Length checks
        if len(prompt.strip()) < MIN_INPUT_LENGTH:
            return GuardrailResult(
                allowed=False,
                reason="Input too short",
                category="validation",
            )

        if len(prompt) > MAX_INPUT_LENGTH:
            return GuardrailResult(
                allowed=False,
                reason=f"Input exceeds {MAX_INPUT_LENGTH} characters",
                category="validation",
            )

        # Pattern checks (only in safe mode)
        if self.safe_mode:
            for pattern, category, action in self._patterns:
                if pattern.search(prompt):
                    return GuardrailResult(
                        allowed=False,
                        reason=f"Matched safety pattern: {category}",
                        category=category,
                    )

        # Attach system prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        return GuardrailResult(allowed=True, modified_prompt=full_prompt)

    def check_output(self, text: str) -> GuardrailResult:
        """Validate model output before returning to user."""
        if not text.strip():
            return GuardrailResult(
                allowed=False,
                reason="Empty model output",
                category="output_validation",
            )

        # Check for PII leakage in output
        pii_patterns = [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b"),  # credit card
        ]
        if self.safe_mode:
            for pat in pii_patterns:
                if pat.search(text):
                    return GuardrailResult(
                        allowed=False,
                        reason="Output contains potential PII",
                        category="pii_leakage",
                    )

        return GuardrailResult(allowed=True)

    @staticmethod
    def get_refusal_response() -> str:
        """Return the standard refusal message."""
        return REFUSAL_RESPONSE
