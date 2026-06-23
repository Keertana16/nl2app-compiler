from models import RuntimeApp

def generate_express_files(app: RuntimeApp) -> dict:
    files = {}
    
    files["backend/package.json"] = """{
  "name": "generated-backend",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "mongoose": "^7.0.0",
    "cors": "^2.8.5",
    "dotenv": "^16.0.3"
  }
}"""

    routes_setup = []
    
    for route in app.routes:
        method = route.method.lower()
        path = route.path
        # convert OpenAPI path param {id} to express :id
        express_path = path.replace("{", ":").replace("}", "")
        
        handler_code = f"""
app.{method}('{express_path}', async (req, res) => {{
  try {{
    // TODO: implement {route.handler}
    res.json({{ message: "{route.handler} successful" }});
  }} catch (error) {{
    res.status(500).json({{ error: error.message }});
  }}
}});
"""
        routes_setup.append(handler_code)

    files["backend/server.js"] = "\\n".join([
        "const express = require('express');",
        "const cors = require('cors');",
        "const mongoose = require('mongoose');",
        "require('dotenv').config();",
        "",
        "const app = express();",
        "app.use(cors());",
        "app.use(express.json());",
        "",
        "// Routes",
        "\\n".join(routes_setup),
        "",
        "const PORT = process.env.PORT || 5000;",
        "app.listen(PORT, () => console.log(`Server running on port ${PORT}`));"
    ])
    
    return files
