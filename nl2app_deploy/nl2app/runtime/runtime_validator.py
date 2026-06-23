from models import RuntimeApp

def validate_runtime(app: RuntimeApp):
    # Basic sanity checks for the runtime representation
    if not app.pages:
        pass # Not necessarily an error, but good to know
    
    # In a full implementation, we might check if all bound_entities in components
    # actually match models in the app.models
    model_names = {m.name for m in app.models}
    
    for page in app.pages:
        for comp in page.components:
            if comp.data_binding and comp.data_binding not in model_names:
                # Log a warning or error
                pass
                
    return True
