from models import SchemaBundle, RuntimeApp, RuntimeModel
from .page_renderer import render_pages
from .route_builder import build_routes
from .runtime_validator import validate_runtime

def build_runtime_app(schema_bundle: SchemaBundle) -> RuntimeApp:
    # Build models from DBSchema
    models = []
    for table in schema_bundle.db_schema.tables:
        models.append(RuntimeModel(
            name=table.name,
            fields=[{"name": c.name, "type": c.type.value} for c in table.columns]
        ))
    
    # Build routes from APISchema & AuthSchema
    routes = build_routes(schema_bundle.api_schema, schema_bundle.auth_schema)
    
    # Build UI Pages from UISchema
    pages = render_pages(schema_bundle.ui_schema)
    
    app = RuntimeApp(
        pages=pages,
        routes=routes,
        models=models,
        global_state={}
    )
    
    # Validate the generated runtime representation
    validate_runtime(app)
    
    return app
