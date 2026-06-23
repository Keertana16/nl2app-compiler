"""
pipeline/architecture.py — STAGE 2: System Design Layer

Converts IntentSpec (what the user wants) into ArchitectureSpec (how the
system is structured): concrete entity fields, role permissions, and
named business flows (e.g. premium gating, role-based access).

This stage NEVER sees the original raw user text — only the structured
IntentSpec. That boundary is intentional: it forces the architecture
decisions to be traceable back to extracted intent, not re-interpreted
from scratch each time (this is part of what gives us determinism).
"""

from models import IntentSpec, ArchitectureSpec
from llm_client import call_structured, MODEL_STRONG

SYSTEM_PROMPT = """You are the System Design stage of a natural-language-to-application compiler.

You receive a structured IntentSpec (entities, roles, features, ambiguities) and must produce
an ArchitectureSpec: concrete entity field definitions, role-to-entity permissions, and named
business flows.

Rules:
- Every entity in the input IntentSpec must appear in `entities`, with realistic fields
  (at minimum: an id-like primary identifier is implied, plus fields matching the entity's purpose).
- Use `relation` field type + relation_target when one entity references another
  (e.g. Order.customer_id -> relation to Customer).
- Every role in the input must appear in `roles` with explicit permissions per entity
  (create/read/update/delete). Be conservative: don't grant delete/update unless implied.
- For every feature flagged in IntentSpec.features (e.g. payments, analytics), create a
  corresponding named Flow describing trigger + steps (e.g. 'premium_gating' flow triggered
  by a non-premium user accessing a gated page, with steps like "check subscription status",
  "redirect to upgrade page if false").
- If IntentSpec.roles included an 'admin'-like role and features include 'analytics', ensure
  an explicit flow describing which role(s) can view analytics and which cannot.
- Do not invent entities/roles/features that are not present in the IntentSpec input.
"""


def design_architecture(intent: IntentSpec) -> tuple[ArchitectureSpec, dict]:
    """
    Args:
        intent: validated IntentSpec from Stage 1.
    Returns:
        (ArchitectureSpec, telemetry)
    """
    architecture, telemetry = call_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"IntentSpec (JSON):\n{intent.model_dump_json(indent=2)}",
        response_model=ArchitectureSpec,
        model=MODEL_STRONG,      # cross-entity reasoning + permission logic -> stronger model
        temperature=0.2,
    )
    return architecture, telemetry
