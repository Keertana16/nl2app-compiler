from typing import List
from models import UIComponent, RuntimeComponent
import uuid

def render_components(ui_components: List[UIComponent]) -> List[RuntimeComponent]:
    components = []
    for comp in ui_components:
        components.append(RuntimeComponent(
            id=str(uuid.uuid4()),
            type=comp.type,
            label=comp.label,
            props={"bound_fields": comp.bound_fields},
            data_binding=comp.bound_entity
        ))
    return components
