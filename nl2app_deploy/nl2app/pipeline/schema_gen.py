"""
pipeline/schema_gen.py — STAGE 3: Schema Generation

Converts ArchitectureSpec into the four concrete, executable schema
artifacts: UI schema, API schema, DB schema, Auth schema.

DESIGN DECISION: We generate DB schema FIRST, then feed the already-generated
DB schema into the API/UI/Auth prompts. This ordering is deliberate — it is
the single biggest lever for cross-layer consistency. If API and DB are
generated independently in parallel, they drift (different field names,
different entity casing). By generating DB first and passing it forward as
ground truth, API/UI/Auth are instructed to MATCH it rather than invent
their own version. This reduces (but does not eliminate) the repair burden
on Stage 4 — which is exactly the "cost vs quality / latency" tradeoff this
project asks you to make explicit.
"""

from models import (
    ArchitectureSpec, DBSchema, APISchema, UISchema, AuthSchema, SchemaBundle
)
from llm_client import call_structured, MODEL_STRONG, MODEL_CHEAP


DB_PROMPT = """You generate the DATABASE SCHEMA layer of a natural-language-to-application compiler.

Given an ArchitectureSpec, produce a DBSchema: one DBTable per entity, with columns matching
the entity's fields exactly (same names, mapped types). Add a primary key column 'id' (integer)
to every table if not already implied. For 'relation' type fields, create a foreign_key_table
column pointing at the related entity's table name.

Rules:
- Table names and column names MUST exactly match (snake_case of) the entity/field names
  in the input ArchitectureSpec. Do not rename or invent fields.
- Every entity in the input must produce exactly one table.
"""

API_PROMPT = """You generate the API SCHEMA layer of a natural-language-to-application compiler.

You receive an ArchitectureSpec AND the already-finalized DBSchema (ground truth — do not
contradict it). Produce an APISchema: REST endpoints for each entity supporting the actions
permitted by ArchitectureSpec roles/permissions.

Rules:
- Endpoint `entity` field MUST exactly match a table name in the provided DBSchema.
- Only create endpoints for actions actually granted to at least one role in ArchitectureSpec.
- `allowed_roles` on each endpoint must be the list of roles whose permissions include that action
  on that entity.
- Use REST conventions: GET /entity (list), GET /entity/{id}, POST /entity, PUT /entity/{id}, DELETE /entity/{id}.
"""

UI_PROMPT = """You generate the UI SCHEMA layer of a natural-language-to-application compiler.

You receive an ArchitectureSpec AND the finalized APISchema (ground truth). Produce a UISchema:
pages with components, where every component's bound_entity/bound_fields correspond to real
entities/fields, and every page that performs an action implies a matching APISchema endpoint exists.

Rules:
- bound_entity on any component MUST match an entity name that has a corresponding endpoint
  in the provided APISchema.
- visible_to_roles on each page must match roles that have read/access permission per ArchitectureSpec.
- Always include a login/auth page if any role exists.
- Include a dashboard page for roles with analytics-style permissions, if applicable.
"""

AUTH_PROMPT = """You generate the AUTH SCHEMA layer of a natural-language-to-application compiler.

You receive an ArchitectureSpec, the finalized APISchema, and the finalized UISchema (ground truth).
Produce an AuthSchema: for every role, the exact list of UI routes and API endpoint paths that role
may access, derived strictly from what UISchema.visible_to_roles and APISchema.allowed_roles already say.

Rules:
- Do not grant access broader than what UISchema/APISchema already specify per role.
- `roles` must list every role name found in ArchitectureSpec.
"""


def generate_db_schema(architecture: ArchitectureSpec) -> tuple[DBSchema, dict]:
    return call_structured(
        system_prompt=DB_PROMPT,
        user_prompt=f"ArchitectureSpec:\n{architecture.model_dump_json(indent=2)}",
        response_model=DBSchema,
        model=MODEL_STRONG,
        temperature=0.1,
    )


def generate_api_schema(architecture: ArchitectureSpec, db_schema: DBSchema) -> tuple[APISchema, dict]:
    prompt = (
        f"ArchitectureSpec:\n{architecture.model_dump_json(indent=2)}\n\n"
        f"Finalized DBSchema (ground truth, must match):\n{db_schema.model_dump_json(indent=2)}"
    )
    return call_structured(
        system_prompt=API_PROMPT,
        user_prompt=prompt,
        response_model=APISchema,
        model=MODEL_STRONG,
        temperature=0.1,
    )


def generate_ui_schema(architecture: ArchitectureSpec, api_schema: APISchema) -> tuple[UISchema, dict]:
    prompt = (
        f"ArchitectureSpec:\n{architecture.model_dump_json(indent=2)}\n\n"
        f"Finalized APISchema (ground truth, must match):\n{api_schema.model_dump_json(indent=2)}"
    )
    return call_structured(
        system_prompt=UI_PROMPT,
        user_prompt=prompt,
        response_model=UISchema,
        model=MODEL_CHEAP,   # layout generation is lower-stakes -> cheap model
        temperature=0.3,
    )


def generate_auth_schema(
    architecture: ArchitectureSpec, api_schema: APISchema, ui_schema: UISchema
) -> tuple[AuthSchema, dict]:
    prompt = (
        f"ArchitectureSpec:\n{architecture.model_dump_json(indent=2)}\n\n"
        f"APISchema:\n{api_schema.model_dump_json(indent=2)}\n\n"
        f"UISchema:\n{ui_schema.model_dump_json(indent=2)}"
    )
    return call_structured(
        system_prompt=AUTH_PROMPT,
        user_prompt=prompt,
        response_model=AuthSchema,
        model=MODEL_CHEAP,
        temperature=0.1,
    )


def generate_all_schemas(architecture: ArchitectureSpec) -> tuple[SchemaBundle, list[dict]]:
    """
    Sequential generation in dependency order: DB -> API -> UI -> Auth.
    Returns the bundle plus a list of telemetry dicts (one per sub-call).
    """
    telemetry_log = []

    db_schema, t1 = generate_db_schema(architecture)
    telemetry_log.append({"stage": "schema_gen.db", **t1})

    api_schema, t2 = generate_api_schema(architecture, db_schema)
    telemetry_log.append({"stage": "schema_gen.api", **t2})

    ui_schema, t3 = generate_ui_schema(architecture, api_schema)
    telemetry_log.append({"stage": "schema_gen.ui", **t3})

    auth_schema, t4 = generate_auth_schema(architecture, api_schema, ui_schema)
    telemetry_log.append({"stage": "schema_gen.auth", **t4})

    bundle = SchemaBundle(
        ui_schema=ui_schema, api_schema=api_schema, db_schema=db_schema, auth_schema=auth_schema
    )
    return bundle, telemetry_log
