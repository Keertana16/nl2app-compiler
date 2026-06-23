from typing import List
from models import APISchema, AuthSchema, RuntimeRoute

def build_routes(api_schema: APISchema, auth_schema: AuthSchema) -> List[RuntimeRoute]:
    routes = []
    for endpoint in api_schema.endpoints:
        # Determine if auth is required based on rules
        auth_req = False
        allowed = endpoint.allowed_roles
        
        # simple check if any role needs auth
        if allowed and "public" not in [r.lower() for r in allowed]:
            auth_req = True
            
        full_path = f"{api_schema.base_path}{endpoint.path}"
        # Remove trailing slash
        if full_path.endswith("/") and len(full_path) > 1:
            full_path = full_path[:-1]
            
        routes.append(RuntimeRoute(
            path=full_path,
            method=endpoint.method,
            handler=f"handle_{endpoint.method.lower()}_{endpoint.entity}",
            auth_required=auth_req,
            roles_allowed=allowed
        ))
    return routes
