import json
from models import RuntimeApp
from llm_client import call_structured_raw_json, MODEL_STRONG

def generate_react_files(app: RuntimeApp) -> dict:
    files = {}
    
    # Base layout and App (Vite + Tailwind + Lucide)
    files["frontend/package.json"] = """{
  "name": "generated-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.10.0",
    "lucide-react": "^0.260.0",
    "framer-motion": "^10.12.0",
    "clsx": "^1.2.1",
    "tailwind-merge": "^1.12.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "autoprefixer": "^10.4.14",
    "postcss": "^8.4.23",
    "tailwindcss": "^3.3.2",
    "vite": "^4.3.2"
  }
}"""

    files["frontend/vite.config.js"] = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})"""

    files["frontend/tailwind.config.js"] = """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}"""

    files["frontend/postcss.config.js"] = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}"""

    files["frontend/index.html"] = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Generated App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>"""

    files["frontend/src/index.css"] = """@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-slate-50 text-slate-900;
  }
}"""

    files["frontend/src/main.jsx"] = """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)"""

    app_jsx_routes = []
    app_jsx_imports = []
    
    system_prompt = """You are an expert Frontend Engineer. You write highly-interactive, beautiful, production-ready React.js components using Tailwind CSS and Lucide-react icons. 
Given a UI page spec, return ONLY a valid JSON object containing the React source code.
Format:
{
  "code": "import React from 'react';\\nimport { Home } from 'lucide-react';\\n\\nexport default function Page() { return <div className=\\"p-8\\">...</div>; }"
}
"""

    for page in app.pages:
        comp_name = page.name.replace(" ", "") + "Page"
        app_jsx_imports.append(f"import {comp_name} from './pages/{comp_name}';")
        app_jsx_routes.append(f'            <Route path="{page.route}" element={{<{comp_name} />}} />')
        
        user_prompt = f"Write the React component for the page '{page.name}'.\n"
        user_prompt += f"It contains these components:\n"
        for comp in page.components:
            user_prompt += f"- {comp.label} (type: {comp.type}, bound_fields: {comp.props.get('bound_fields', [])})\n"
        user_prompt += "\nMake it look incredible, like a modern SaaS dashboard. Implement mock interactions, empty states, nice borders, subtle shadows, and lucide icons. Ensure the default export is named " + comp_name + "."

        try:
            parsed, tel = call_structured_raw_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=MODEL_STRONG,
                temperature=0.3
            )
            page_code = parsed.get("code", f"export default function {comp_name}() {{ return <div>Failed to generate</div>; }}")
        except Exception as e:
            print(f"Error generating page {comp_name}: {e}")
            page_code = f"export default function {comp_name}() {{ return <div>Error generating page.</div>; }}"
            
        files[f"frontend/src/pages/{comp_name}.jsx"] = page_code
        
    nav_links = "\n".join([f'            <Link to="{p.route}" className="flex items-center gap-2 p-2 px-3 hover:bg-slate-100 rounded-lg text-slate-600 transition-colors font-medium">{p.name}</Link>' for p in app.pages])

    files["frontend/src/App.jsx"] = f"""import React from 'react';
import {{ BrowserRouter as Router, Routes, Route, Link }} from 'react-router-dom';
{"\\n".join(app_jsx_imports)}

export default function App() {{
  return (
    <Router>
      <div className="flex h-screen bg-slate-50 font-sans">
        <nav className="w-64 bg-white border-r border-slate-200 flex flex-col p-4 shadow-sm z-10">
          <div className="font-bold text-2xl mb-8 mt-2 text-indigo-600 tracking-tight px-2 flex items-center gap-2">
            <div className="w-6 h-6 bg-indigo-600 rounded-md"></div>
            AppLogo
          </div>
          <div className="flex flex-col gap-1">
{nav_links}
          </div>
        </nav>
        <main className="flex-1 overflow-auto bg-slate-50/50">
          <Routes>
{"\\n".join(app_jsx_routes)}
          </Routes>
        </main>
      </div>
    </Router>
  );
}}"""
    
    return files
