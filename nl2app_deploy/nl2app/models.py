"""
models.py — The strict contract between every pipeline stage.

This is the most important file in the system. Every stage MUST produce
output that validates against these Pydantic models. Pydantic gives us,
for free:
  - required field enforcement
  - type safety
  - automatic rejection of malformed JSON (ValidationError with exact field paths)

We use these same models to build OpenAI "strict" structured-output schemas,
so the LLM is constrained at decoding time, not just checked after the fact.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# STAGE 1 — INTENT EXTRACTION OUTPUT
# ---------------------------------------------------------------------------

class AppFeature(str, Enum):
    AUTH = "auth"
    PAYMENTS = "payments"
    ANALYTICS = "analytics"
    NOTIFICATIONS = "notifications"
    FILE_UPLOAD = "file_upload"
    SEARCH = "search"
    REAL_TIME = "real_time"


class IntentEntity(BaseModel):
    name: str = Field(..., description="Entity/object name, e.g. 'Contact'")
    description: str = Field(..., description="What this entity represents")


class IntentRole(BaseModel):
    name: str = Field(..., description="Role name, e.g. 'admin', 'user'")
    description: str = Field(..., description="What this role can generally do")


class Ambiguity(BaseModel):
    field: str = Field(..., description="What part of the request is ambiguous/underspecified")
    assumption_made: str = Field(..., description="The reasonable default assumption the system chose")


class IntentSpec(BaseModel):
    """Structured intermediate form parsed from raw natural language."""
    app_name: str
    app_type: str = Field(..., description="e.g. 'CRM', 'e-commerce', 'booking system'")
    summary: str
    entities: List[IntentEntity]
    roles: List[IntentRole]
    features: List[AppFeature]
    ambiguities: List[Ambiguity] = Field(default_factory=list)
    requires_clarification: bool = Field(
        default=False,
        description="True only if the request is too vague/conflicting to proceed even with assumptions"
    )
    clarification_question: Optional[str] = None

    @field_validator("features", mode="before")
    @classmethod
    def drop_invalid_features(cls, v):
        """Drop any feature values the model invented outside the AppFeature enum."""
        valid = {e.value for e in AppFeature}
        if isinstance(v, list):
            return [item for item in v if item in valid]
        return v


# ---------------------------------------------------------------------------
# STAGE 2 — SYSTEM DESIGN / ARCHITECTURE OUTPUT
# ---------------------------------------------------------------------------

class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    ENUM = "enum"
    RELATION = "relation"


class EntityField(BaseModel):
    name: str
    type: FieldType
    required: bool = True
    enum_values: Optional[List[str]] = None
    relation_target: Optional[str] = Field(
        None, description="If type=relation, the entity name this points to"
    )


class ArchitectureEntity(BaseModel):
    name: str
    fields: List[EntityField]


class Permission(BaseModel):
    role: str
    entity: str
    actions: List[Literal["create", "read", "update", "delete"]]


class ArchitectureRole(BaseModel):
    name: str
    permissions: List[Permission]


class Flow(BaseModel):
    name: str = Field(..., description="e.g. 'premium_gating', 'role_based_dashboard_access'")
    description: str
    trigger: str
    steps: List[str]


class ArchitectureSpec(BaseModel):
    app_name: str
    entities: List[ArchitectureEntity]
    roles: List[ArchitectureRole]
    flows: List[Flow]


# ---------------------------------------------------------------------------
# STAGE 3 — SCHEMA GENERATION OUTPUT (UI / API / DB / AUTH)
# ---------------------------------------------------------------------------

class UIComponent(BaseModel):
    type: Literal["table", "form", "card", "chart", "button", "nav", "text"]
    label: str
    bound_entity: Optional[str] = Field(None, description="Entity this component displays/edits")
    bound_fields: List[str] = Field(default_factory=list)


class UIPage(BaseModel):
    name: str
    route: str = Field(..., description="e.g. '/dashboard'")
    components: List[UIComponent]
    visible_to_roles: List[str]


class UISchema(BaseModel):
    pages: List[UIPage]


class APIParam(BaseModel):
    name: str
    type: FieldType
    required: bool = True


class APIEndpoint(BaseModel):
    path: str = Field(..., description="e.g. '/api/contacts/{id}'")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    entity: str = Field(..., description="Entity this endpoint operates on")
    params: List[APIParam] = Field(default_factory=list)
    allowed_roles: List[str]
    description: str


class APISchema(BaseModel):
    base_path: str = "/api"
    endpoints: List[APIEndpoint]


class DBColumn(BaseModel):
    name: str
    type: FieldType
    primary_key: bool = False
    foreign_key_table: Optional[str] = None
    nullable: bool = True


class DBTable(BaseModel):
    name: str
    columns: List[DBColumn]


class DBSchema(BaseModel):
    tables: List[DBTable]


class AuthRule(BaseModel):
    role: str
    can_access_routes: List[str]
    can_access_endpoints: List[str]


class AuthSchema(BaseModel):
    strategy: Literal["session", "jwt"] = "jwt"
    roles: List[str]
    rules: List[AuthRule]


class SchemaBundle(BaseModel):
    """All four schema artifacts produced by Stage 3, validated together in Stage 4."""
    ui_schema: UISchema
    api_schema: APISchema
    db_schema: DBSchema
    auth_schema: AuthSchema


# ---------------------------------------------------------------------------
# STAGE 4 — VALIDATION / REPAIR
# ---------------------------------------------------------------------------

class IssueType(str, Enum):
    INVALID_JSON = "invalid_json"
    MISSING_FIELD = "missing_field"
    TYPE_MISMATCH = "type_mismatch"
    HALLUCINATED_FIELD = "hallucinated_field"
    CROSS_LAYER_MISMATCH = "cross_layer_mismatch"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"


class ValidationIssue(BaseModel):
    issue_type: IssueType
    layer: Literal["ui", "api", "db", "auth", "architecture", "intent"]
    location: str = Field(..., description="Dotted path to the offending field, e.g. 'api_schema.endpoints[2].entity'")
    detail: str


class ValidationReport(BaseModel):
    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FINAL COMPILED OUTPUT
# ---------------------------------------------------------------------------

class RuntimeComponent(BaseModel):
    id: str
    type: str
    label: str
    props: dict = Field(default_factory=dict)
    data_binding: Optional[str] = None

class RuntimePage(BaseModel):
    id: str
    name: str
    route: str
    components: List[RuntimeComponent]
    layout: str = "default"

class RuntimeRoute(BaseModel):
    path: str
    method: str
    handler: str
    auth_required: bool
    roles_allowed: List[str]

class RuntimeModel(BaseModel):
    name: str
    fields: List[dict]

class RuntimeApp(BaseModel):
    pages: List[RuntimePage]
    routes: List[RuntimeRoute]
    models: List[RuntimeModel]
    global_state: dict = Field(default_factory=dict)

class CompiledApp(BaseModel):
    intent: IntentSpec
    architecture: ArchitectureSpec
    schema_bundle: SchemaBundle
    repair_attempts: int
    final_validation: ValidationReport
    assumptions_made: List[str] = Field(default_factory=list)
    runtime_app: Optional[RuntimeApp] = None
