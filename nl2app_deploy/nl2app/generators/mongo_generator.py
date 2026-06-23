from models import RuntimeApp

def generate_mongo_files(app: RuntimeApp) -> dict:
    files = {}
    
    type_map = {
        "string": "String",
        "integer": "Number",
        "float": "Number",
        "boolean": "Boolean",
        "date": "Date",
        "datetime": "Date",
        "email": "String",
        "enum": "String",
        "relation": "mongoose.Schema.Types.ObjectId"
    }

    for model in app.models:
        name = model.name.capitalize()
        schema_fields = []
        
        for field in model.fields:
            if field["name"].lower() == "id": continue
            ftype = type_map.get(field["type"], "String")
            schema_fields.append(f"  {field['name']}: {{ type: {ftype} }}")
            
        fields_str = ",\\n".join(schema_fields)
        
        model_code = f"""const mongoose = require('mongoose');

const {name}Schema = new mongoose.Schema({{
{fields_str}
}}, {{ timestamps: true }});

module.exports = mongoose.model('{name}', {name}Schema);
"""
        files[f"backend/models/{name}.js"] = model_code
        
    return files
