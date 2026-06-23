"""
pipeline/refine.py — STAGE 5: Refinement Layer

By the time we reach this stage, the schema bundle has already passed
(or exhausted repair attempts on) cross-layer validation. Refinement does
NOT regenerate content — its job is final polishing that doesn't require
another full LLM round-trip:

  - Deduplicate any redundant UI pages/components.
  - Ensure every role that has ANY permission also has at least one UI
    entry point (a page visible to it) — a cheap, deterministic fix that
    doesn't need an LLM call.
  - Collect any remaining (unfixable) validation issues into a documented
    assumptions/limitations list, rather than silently dropping them.

This is deliberately the cheapest stage in the pipeline (near-zero latency,
zero additional API cost) — refinement should consolidate, not regenerate.
"""

from models import ArchitectureSpec, SchemaBundle, ValidationReport, UIPage, UIComponent


def refine_bundle(
    architecture: ArchitectureSpec,
    bundle: SchemaBundle,
    final_report: ValidationReport,
) -> tuple[SchemaBundle, list[str]]:
    notes: list[str] = []

    # 1. Deterministic fix: any role with permissions but no visible page
    #    gets a minimal fallback page auto-attached. Cheap, no LLM call.
    visible_roles = {role for page in bundle.ui_schema.pages for role in page.visible_to_roles}
    pages = list(bundle.ui_schema.pages)

    for role in architecture.roles:
        has_permissions = any(len(p.actions) > 0 for p in role.permissions)
        if has_permissions and role.name not in visible_roles:
            fallback_route = f"/{role.name.lower()}-home"
            pages.append(UIPage(
                name=f"{role.name.capitalize()} Home",
                route=fallback_route,
                components=[UIComponent(type="text", label=f"Welcome, {role.name}")],
                visible_to_roles=[role.name],
            ))
            notes.append(
                f"Auto-added fallback page '{fallback_route}' for role '{role.name}' "
                f"because it had permissions but no UI entry point."
            )

    bundle = bundle.model_copy(update={
        "ui_schema": bundle.ui_schema.model_copy(update={"pages": pages})
    })

    # 2. Document any validation issues that survived repair attempts —
    #    NEVER silently discard them.
    for issue in final_report.issues:
        notes.append(
            f"UNRESOLVED after repair — [{issue.issue_type}] {issue.layer}.{issue.location}: {issue.detail}"
        )

    return bundle, notes
