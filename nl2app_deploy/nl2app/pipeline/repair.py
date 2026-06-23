"""
pipeline/repair.py — STAGE 4b: Targeted Repair Engine

The task explicitly warns against "blind full retry" and asks for either
automatic repair OR re-generation of SPECIFIC broken parts. This module
implements the latter, layer by layer:

  - If issues exist only in `ui_schema` -> only regenerate ui_schema
    (re-using the now-trusted db/api schemas as ground truth context, plus
    the exact list of issues found, so the model fixes precisely those problems).
  - Same for api_schema, auth_schema.
  - db_schema issues are rarer (DB is generated first, nothing to contradict)
    but handled the same way if they occur (e.g. dangling foreign keys).

We cap repair attempts (default 3) to avoid infinite loops, and every attempt
is logged so the eval framework can report "retries per request".
"""

from models import ArchitectureSpec, SchemaBundle, ValidationReport, ValidationIssue
from llm_client import call_structured, MODEL_STRONG, MODEL_CHEAP
from pipeline.schema_gen import (
    DBSchema, APISchema, UISchema, AuthSchema,
)

MAX_REPAIR_ATTEMPTS = 3


def _issues_by_layer(issues: list[ValidationIssue]) -> dict[str, list[ValidationIssue]]:
    grouped: dict[str, list[ValidationIssue]] = {}
    for issue in issues:
        grouped.setdefault(issue.layer, []).append(issue)
    return grouped


def _format_issues(issues: list[ValidationIssue]) -> str:
    lines = [f"- [{i.issue_type}] at {i.location}: {i.detail}" for i in issues]
    return "\n".join(lines)


def _repair_layer(
    layer: str,
    issues: list[ValidationIssue],
    architecture: ArchitectureSpec,
    bundle: SchemaBundle,
) -> tuple[object, dict]:
    """Regenerate exactly one layer, given the specific issues found in it,
    using the OTHER (already-valid) layers as fixed ground truth context."""

    issue_text = _format_issues(issues)
    arch_json = architecture.model_dump_json(indent=2)

    if layer == "db":
        prompt = (
            f"ArchitectureSpec (ground truth):\n{arch_json}\n\n"
            f"Current (BROKEN) DBSchema:\n{bundle.db_schema.model_dump_json(indent=2)}\n\n"
            f"Specific issues found — fix ONLY these, keep everything else that is correct:\n{issue_text}"
        )
        system = ("You are repairing the DB schema layer of an app compiler. "
                  "Fix exactly the listed issues. Do not change anything that wasn't flagged.")
        return call_structured(system, prompt, DBSchema, model=MODEL_STRONG, temperature=0.1)

    if layer == "api":
        prompt = (
            f"ArchitectureSpec (ground truth):\n{arch_json}\n\n"
            f"Finalized DBSchema (ground truth, must match):\n{bundle.db_schema.model_dump_json(indent=2)}\n\n"
            f"Current (BROKEN) APISchema:\n{bundle.api_schema.model_dump_json(indent=2)}\n\n"
            f"Specific issues found — fix ONLY these:\n{issue_text}"
        )
        system = ("You are repairing the API schema layer of an app compiler. "
                  "Fix exactly the listed issues against the ground-truth DB schema. "
                  "Do not change anything that wasn't flagged.")
        return call_structured(system, prompt, APISchema, model=MODEL_STRONG, temperature=0.1)

    if layer == "ui":
        prompt = (
            f"ArchitectureSpec (ground truth):\n{arch_json}\n\n"
            f"Finalized APISchema (ground truth, must match):\n{bundle.api_schema.model_dump_json(indent=2)}\n\n"
            f"Current (BROKEN) UISchema:\n{bundle.ui_schema.model_dump_json(indent=2)}\n\n"
            f"Specific issues found — fix ONLY these:\n{issue_text}"
        )
        system = ("You are repairing the UI schema layer of an app compiler. "
                  "Fix exactly the listed issues against the ground-truth API schema. "
                  "Do not change anything that wasn't flagged.")
        return call_structured(system, prompt, UISchema, model=MODEL_CHEAP, temperature=0.2)

    if layer == "auth":
        prompt = (
            f"ArchitectureSpec (ground truth):\n{arch_json}\n\n"
            f"UISchema (ground truth):\n{bundle.ui_schema.model_dump_json(indent=2)}\n\n"
            f"APISchema (ground truth):\n{bundle.api_schema.model_dump_json(indent=2)}\n\n"
            f"Current (BROKEN) AuthSchema:\n{bundle.auth_schema.model_dump_json(indent=2)}\n\n"
            f"Specific issues found — fix ONLY these:\n{issue_text}"
        )
        system = ("You are repairing the Auth schema layer of an app compiler. "
                  "Fix exactly the listed issues. Do not grant access beyond what UI/API already specify.")
        return call_structured(system, prompt, AuthSchema, model=MODEL_CHEAP, temperature=0.1)

    raise ValueError(f"Unknown layer for repair: {layer}")


def repair_bundle(
    architecture: ArchitectureSpec,
    bundle: SchemaBundle,
    report: ValidationReport,
) -> tuple[SchemaBundle, int, list[dict]]:
    """
    Iteratively repairs only the layers with issues, re-validating after each
    pass, up to MAX_REPAIR_ATTEMPTS. Returns the (possibly improved) bundle,
    the number of attempts used, and telemetry for each repair call made.
    """
    from pipeline.validator import validate_bundle  # local import avoids circular import

    telemetry_log = []
    attempts = 0
    current_bundle = bundle
    current_report = report

    while not current_report.is_valid and attempts < MAX_REPAIR_ATTEMPTS:
        attempts += 1
        grouped = _issues_by_layer(current_report.issues)

        # architecture-layer logical issues can't be "schema-repaired" here;
        # they're surfaced to the caller as remaining issues instead.
        grouped.pop("architecture", None)
        grouped.pop("intent", None)

        bundle_dict = current_bundle.model_dump()

        for layer, issues in grouped.items():
            try:
                fixed_obj, t = _repair_layer(layer, issues, architecture, current_bundle)
                telemetry_log.append({"stage": f"repair.{layer}", "attempt": attempts, **t})
                bundle_dict[f"{layer}_schema"] = fixed_obj.model_dump()
            except Exception as e:
                telemetry_log.append({
                    "stage": f"repair.{layer}", "attempt": attempts,
                    "error": str(e)
                })

        current_bundle = SchemaBundle.model_validate(bundle_dict)
        current_report = validate_bundle(architecture, current_bundle)

    return current_bundle, attempts, telemetry_log
