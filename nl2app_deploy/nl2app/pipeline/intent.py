"""
pipeline/intent.py — STAGE 1: Intent Extraction

Parses raw, open-ended natural language into a structured IntentSpec.
This is deliberately the ONLY stage that touches raw user text. Every
later stage consumes structured JSON, never free text again — this is
what makes the pipeline "compiler-like" rather than a single mega-prompt.

Handles vague/conflicting input per the FAILURE HANDLING requirement:
the model is instructed to either make a documented assumption, or set
requires_clarification=True with a specific question if it truly cannot
proceed.
"""

from models import IntentSpec
from llm_client import call_structured, MODEL_CHEAP

SYSTEM_PROMPT = """You are the Intent Extraction stage of a natural-language-to-application compiler.

Your ONLY job: read the user's app description and convert it into a structured IntentSpec.

Rules:
- Extract every entity (noun the app manages, e.g. Contact, Order, Invoice) the user implies, even if not explicit.
- Extract every role mentioned or implied (if none mentioned, assume a single 'user' role).
- Map mentioned capabilities to the closest matching feature enum values. Do not invent features not implied by the text.
- If something is ambiguous (e.g. "premium plan" without specifying price/billing), do NOT block the whole pipeline.
  Instead, record it in `ambiguities` with a reasonable, clearly-stated assumption.
- Only set requires_clarification=true if the request is SO vague (e.g. "build me an app") that no
  reasonable assumption set would produce a coherent system. In that case also fill `clarification_question`.
- Never hallucinate entities or features that have no basis in the text.
- app_type should be a short category label (e.g. "CRM", "e-commerce", "booking system", "internal tool").
"""


def extract_intent(user_prompt: str) -> tuple[IntentSpec, dict]:
    """
    Args:
        user_prompt: raw natural language app description from the user.
    Returns:
        (IntentSpec, telemetry)
    """
    intent, telemetry = call_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"User request:\n\"\"\"\n{user_prompt}\n\"\"\"",
        response_model=IntentSpec,
        model=MODEL_CHEAP,       # low-ambiguity extraction task -> cheap model is fine
        temperature=0.1,         # low temp: we want consistency, not creativity
    )
    return intent, telemetry
