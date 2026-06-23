from typing import List
from models import UISchema, RuntimePage
from .component_renderer import render_components
import uuid

def render_pages(ui_schema: UISchema) -> List[RuntimePage]:
    pages = []
    for page in ui_schema.pages:
        components = render_components(page.components)
        pages.append(RuntimePage(
            id=str(uuid.uuid4()),
            name=page.name,
            route=page.route,
            components=components,
            layout="default"
        ))
    return pages
