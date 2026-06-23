"""
pipeline/validator.py — STAGE 4a: Validation Engine

This is the core detection half of "Validation + Repair Engine" — the part
the task description explicitly calls the most important piece of the system.

Two layers of checking happen here:
  1. Pydantic validation already happened implicitly when each stage parsed
     its structured output (that catches invalid_json / missing_field / type_mismatch
     automatically — we don't need to re-check that here, it would have raised
     already). This module assumes each individual object IS Pydantic-valid.
  2. CROSS-LAYER CONSISTENCY — the part Pydantic cannot check on its own, because
     it requires comparing fields ACROSS separate objects (UI references API,
     API references DB, Auth references both). This is where real bugs hide,
     and where hand-written rules earn their keep.

Every issue found is a structured ValidationIssue (typed, located, explained) —
never a vague "something is wrong" — because the repair engine in repair.py
needs an exact location to patch only the broken fragment.
"""

from models import (
    ArchitectureSpec, SchemaBundle, ValidationReport, ValidationIssue, IssueType
)


def validate_bundle(architecture: ArchitectureSpec, bundle: SchemaBundle) -> ValidationReport:
    issues: list[ValidationIssue] = []

    entity_names = {e.name for e in architecture.entities}
    table_names = {t.name for t in bundle.db_schema.tables}
    role_names = {r.name for r in architecture.roles}

    # --- DB layer: every architecture entity must have a table -------------
    for entity in architecture.entities:
        if entity.name not in table_names:
            issues.append(ValidationIssue(
                issue_type=IssueType.CROSS_LAYER_MISMATCH,
                layer="db",
                location=f"db_schema.tables[?name=='{entity.name}']",
                detail=f"Entity '{entity.name}' defined in architecture has no corresponding DB table."
            ))

    # --- DB layer: foreign keys must point at real tables -------------------
    for table in bundle.db_schema.tables:
        for col in table.columns:
            if col.foreign_key_table and col.foreign_key_table not in table_names:
                issues.append(ValidationIssue(
                    issue_type=IssueType.HALLUCINATED_FIELD,
                    layer="db",
                    location=f"db_schema.tables[{table.name}].columns[{col.name}].foreign_key_table",
                    detail=f"Column '{col.name}' references foreign_key_table "
                           f"'{col.foreign_key_table}' which does not exist in db_schema."
                ))

    # --- API layer: every endpoint.entity must exist as a DB table ----------
    for ep in bundle.api_schema.endpoints:
        if ep.entity not in table_names:
            issues.append(ValidationIssue(
                issue_type=IssueType.CROSS_LAYER_MISMATCH,
                layer="api",
                location=f"api_schema.endpoints[{ep.method} {ep.path}].entity",
                detail=f"Endpoint '{ep.method} {ep.path}' targets entity '{ep.entity}' "
                       f"which has no matching table in db_schema."
            ))
        for role in ep.allowed_roles:
            if role not in role_names:
                issues.append(ValidationIssue(
                    issue_type=IssueType.HALLUCINATED_FIELD,
                    layer="api",
                    location=f"api_schema.endpoints[{ep.method} {ep.path}].allowed_roles",
                    detail=f"Endpoint allows role '{role}' which is not defined in architecture.roles."
                ))

    # --- UI layer: bound_entity must exist + have a matching API endpoint ---
    api_entities = {ep.entity for ep in bundle.api_schema.endpoints}
    for page in bundle.ui_schema.pages:
        for role in page.visible_to_roles:
            if role not in role_names:
                issues.append(ValidationIssue(
                    issue_type=IssueType.HALLUCINATED_FIELD,
                    layer="ui",
                    location=f"ui_schema.pages[{page.route}].visible_to_roles",
                    detail=f"Page '{page.route}' is visible to role '{role}' which is not defined in architecture.roles."
                ))
        for comp in page.components:
            if comp.bound_entity and comp.bound_entity not in entity_names:
                issues.append(ValidationIssue(
                    issue_type=IssueType.HALLUCINATED_FIELD,
                    layer="ui",
                    location=f"ui_schema.pages[{page.route}].components[{comp.label}].bound_entity",
                    detail=f"Component '{comp.label}' is bound to entity '{comp.bound_entity}' "
                           f"which does not exist in architecture.entities."
                ))
            if comp.bound_entity and comp.bound_entity not in api_entities:
                issues.append(ValidationIssue(
                    issue_type=IssueType.CROSS_LAYER_MISMATCH,
                    layer="ui",
                    location=f"ui_schema.pages[{page.route}].components[{comp.label}].bound_entity",
                    detail=f"Component '{comp.label}' is bound to entity '{comp.bound_entity}' "
                           f"which has no corresponding API endpoint — UI cannot fetch/save this data."
                ))

    # --- Auth layer: roles must match architecture; rules must not exceed ---
    #     what UI/API already grant.
    valid_routes = {p.route for p in bundle.ui_schema.pages}
    valid_endpoints = {f"{ep.method} {ep.path}" for ep in bundle.api_schema.endpoints}

    for role in bundle.auth_schema.roles:
        if role not in role_names:
            issues.append(ValidationIssue(
                issue_type=IssueType.HALLUCINATED_FIELD,
                layer="auth",
                location="auth_schema.roles",
                detail=f"Auth schema lists role '{role}' which is not defined in architecture.roles."
            ))

    for rule in bundle.auth_schema.rules:
        for route in rule.can_access_routes:
            if route not in valid_routes:
                issues.append(ValidationIssue(
                    issue_type=IssueType.CROSS_LAYER_MISMATCH,
                    layer="auth",
                    location=f"auth_schema.rules[{rule.role}].can_access_routes",
                    detail=f"Auth rule grants role '{rule.role}' access to route '{route}' "
                           f"which does not exist in ui_schema.pages."
                ))

    # --- Logical inconsistency: a role with zero permissions anywhere -------
    for role in architecture.roles:
        has_any_permission = any(len(p.actions) > 0 for p in role.permissions)
        if not has_any_permission:
            issues.append(ValidationIssue(
                issue_type=IssueType.LOGICAL_INCONSISTENCY,
                layer="architecture",
                location=f"architecture.roles[{role.name}].permissions",
                detail=f"Role '{role.name}' exists but has no permissions on any entity — "
                       f"likely an incomplete/unusable role definition."
            ))

    return ValidationReport(is_valid=(len(issues) == 0), issues=issues)
