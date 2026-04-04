"""
Query refusal and safety policies (PII prompts, sensitive domains).
Heuristic, regex-based — no third-party safety SDK.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PolicyResult:
    blocked: bool
    message: str | None
    flags: list[str]


_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

_LEGAL = re.compile(
    r"\b(lawsuit|sue|subpoena|legal advice|lawyer|contract review|"
    r"terms of service violation|court order)\b",
    re.I,
)
_MEDICAL = re.compile(
    r"\b(diagnos(e|is)|prescription|dosage|medical advice|treatment plan|"
    r"should i take|is it cancer)\b",
    re.I,
)


def evaluate_query_policies(query: str) -> PolicyResult:
    flags: list[str] = []
    if _SSN.search(query):
        flags.append("pii_ssn_pattern")
    if _CC.search(query) and any(c.isdigit() for c in query):
        flags.append("pii_payment_pattern")
    if re.search(r"\b(my|our) (ssn|social security)\b", query, re.I):
        flags.append("pii_ssn_request")

    if flags:
        return PolicyResult(
            blocked=True,
            message="I cannot process requests that appear to solicit or contain sensitive personal "
            "identifiers. Remove PII and ask a general question instead.",
            flags=flags,
        )

    if _LEGAL.search(query):
        return PolicyResult(
            blocked=True,
            message="I am not a lawyer and cannot provide legal advice. This assistant only "
            "summarizes your uploaded documents for informational purposes.",
            flags=["legal_disclaimer"],
        )

    if _MEDICAL.search(query):
        return PolicyResult(
            blocked=True,
            message="I cannot provide medical advice. Consult a qualified clinician. "
            "I can only reflect what appears in your uploaded documents, not assess your health.",
            flags=["medical_disclaimer"],
        )

    return PolicyResult(blocked=False, message=None, flags=[])
