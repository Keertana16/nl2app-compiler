"""
app.py — FastAPI web server for the nl2app AI App Compiler.

POST /generate  — runs run_pipeline + run_smoke_test, returns full JSON
GET  /          — serves the HTML/JS UI (no build step, vanilla JS + CSS)
GET  /health    — health check for Render/Replit deployment probes
GET  /docs      — FastAPI auto-generated API documentation
"""

import json
import io
import zipfile
import os
import sys

# Ensure local imports work on Vercel
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from orchestrator import run_pipeline
from runtime_sim import run_smoke_test

app = FastAPI(
    title="nl2app AI App Compiler",
    description="Converts natural language app descriptions into validated, executable application configurations.",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Groq llama-3.3-70b pricing (per million tokens)
# ---------------------------------------------------------------------------
_PRICE_IN  = 0.59 / 1_000_000
_PRICE_OUT = 0.79 / 1_000_000


def _compute_cost(telemetry: list[dict]) -> float:
    return round(sum(
        (t.get("prompt_tokens") or 0) * _PRICE_IN +
        (t.get("completion_tokens") or 0) * _PRICE_OUT
        for t in telemetry
    ), 6)


def _compute_quality(data: dict) -> int:
    score = 100
    final_issues = (data.get("stage_outputs") or {}).get("final_validation", {}).get("issues", [])
    score -= len(final_issues) * 8
    score -= (data.get("compiled_app") or {}).get("repair_attempts", 0) * 5
    smoke    = data.get("runtime_proof") or {}
    pass_rate = smoke.get("pass_rate")
    if pass_rate is not None:
        score = int(score * pass_rate)
    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# HTML / CSS / JS — single-file SPA, no build step
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>nl2app — AI App Compiler</title>
<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
/* ─── Reset & Base ──────────────────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0a0b;--surface:rgba(20,20,22,0.7);--surface2:rgba(30,30,34,0.7);--border:rgba(255,255,255,0.08);
  --border2:rgba(255,255,255,0.12);--accent:#7c3aed;--accent2:#a78bfa;--accent-dim:rgba(124,58,237,0.15);
  --green:#10b981;--green-dim:rgba(16,185,129,0.1);--green-border:rgba(16,185,129,0.3);
  --red:#ef4444;--red-dim:rgba(239,68,68,0.1);--red-border:rgba(239,68,68,0.3);
  --yellow:#f59e0b;--yellow-dim:rgba(245,158,11,0.1);--yellow-border:rgba(245,158,11,0.3);
  --text:#f8fafc;--text2:#94a3b8;--text3:#64748b;
  --radius:12px;--radius-sm:8px;--radius-xs:6px;
  --font-mono:'Fira Code','Cascadia Code','JetBrains Mono',monospace;
}
html{scroll-behavior:smooth}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--bg);color:var(--text);min-height:100vh;line-height:1.5;
  background-image: radial-gradient(circle at 50% -20%, rgba(124,58,237,0.15) 0%, rgba(10,10,11,1) 70%);}
::selection{background:#6366f130;color:var(--accent2)}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:#2d3348;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#3d4560}

/* ─── Header ────────────────────────────────────────────────────────────── */
header{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:14px 28px;display:flex;align-items:center;gap:14px;
  position:sticky;top:0;z-index:100;
  backdrop-filter:blur(12px);
}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:32px;height:32px;background:linear-gradient(135deg,#6366f1,#818cf8);
  border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0}
.logo-text h1{font-size:1.05rem;font-weight:700;color:var(--accent2);letter-spacing:-.01em}
.logo-text p{font-size:0.72rem;color:var(--text3);margin-top:1px}
.header-meta{margin-left:auto;display:flex;align-items:center;gap:10px}
.ver-badge{font-size:0.7rem;color:var(--text3);background:var(--surface2);
  padding:3px 10px;border-radius:12px;border:1px solid var(--border);letter-spacing:.04em}
.docs-link{font-size:0.72rem;color:var(--accent2);text-decoration:none;
  padding:4px 10px;border-radius:var(--radius-sm);border:1px solid var(--border2);
  transition:all .15s}
.docs-link:hover{background:var(--accent-dim);border-color:var(--accent)}

/* ─── Layout ────────────────────────────────────────────────────────────── */
.main{max-width:980px;margin:28px auto;padding:0 16px}

/* ─── Card ──────────────────────────────────────────────────────────────── */
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:22px;margin-bottom:18px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  box-shadow: 0 4px 24px rgba(0,0,0,0.2); transition: transform 0.2s, box-shadow 0.2s;}
.card:hover{box-shadow: 0 8px 32px rgba(0,0,0,0.3);}
.card-title{font-size:0.72rem;font-weight:700;color:var(--text3);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px;
  display:flex;align-items:center;gap:8px}
.card-title::before{content:'';width:3px;height:12px;background:var(--accent);
  border-radius:2px;display:inline-block}

/* ─── Input area ────────────────────────────────────────────────────────── */
.prompt-wrap{position:relative}
textarea{
  width:100%;background:var(--bg);border:1px solid var(--border2);
  border-radius:var(--radius-sm);color:var(--text);padding:14px 16px;
  font-size:0.93rem;resize:vertical;min-height:96px;outline:none;
  transition:border-color .2s,box-shadow .2s;font-family:inherit;
  line-height:1.6
}
textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px #6366f118}
.char-count{position:absolute;bottom:10px;right:14px;font-size:0.68rem;
  color:var(--text3);pointer-events:none}

.examples{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.ex-label{font-size:0.8rem;color:var(--text2);font-weight:600;align-self:center;margin-right:4px;}
.ex-btn{
  background:var(--surface2);border:1px solid var(--border2);color:var(--text2);
  padding:5px 12px;border-radius:20px;font-size:0.72rem;cursor:pointer;
  transition:all .15s;white-space:nowrap
}
.ex-btn:hover{border-color:var(--accent);color:var(--accent2);background:var(--accent-dim)}

/* ─── Run button ────────────────────────────────────────────────────────── */
#run{
  background:linear-gradient(135deg,var(--accent),#9333ea);color:white;
  border:none;padding:13px 0;border-radius:var(--radius-sm);font-size:0.95rem;
  font-weight:600;cursor:pointer;width:100%;margin-top:14px;
  transition:all .2s;letter-spacing:.02em;
  display:flex;align-items:center;justify-content:center;gap:8px;
  box-shadow: 0 4px 12px rgba(124,58,237,0.3);
}
#run:hover{opacity:.9; transform: translateY(-1px); box-shadow: 0 6px 16px rgba(124,58,237,0.4);}
#run:active{transform:scale(.99)}
#run:disabled{background:var(--surface2);color:var(--text3);cursor:not-allowed;opacity:1;transform:none}

/* ─── Spinner ───────────────────────────────────────────────────────────── */
.spinner{
  width:15px;height:15px;border:2px solid #ffffff22;border-top-color:#fff;
  border-radius:50%;animation:spin .55s linear infinite;flex-shrink:0
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ─── Pipeline progress ─────────────────────────────────────────────────── */
#pipeline-progress{display:none;margin-bottom:16px}
.progress-label{font-size:0.75rem;color:var(--text3);margin-bottom:8px;display:flex;justify-content:space-between}
.progress-bar-track{background:var(--surface2);border-radius:4px;height:4px;overflow:hidden}
.progress-bar-fill{
  height:100%;border-radius:4px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  transition:width .4s ease;width:0%
}
.stages{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
.stage-dot{
  display:flex;align-items:center;gap:5px;
  font-size:0.68rem;color:var(--text3);
  padding:3px 9px;border-radius:12px;border:1px solid var(--border);
  background:var(--surface2);transition:all .3s
}
.stage-dot.active{color:var(--accent2);border-color:var(--accent);background:var(--accent-dim)}
.stage-dot.done{color:var(--green);border-color:var(--green-border);background:var(--green-dim)}
.stage-dot .dot{width:5px;height:5px;border-radius:50%;background:currentColor;flex-shrink:0}
.stage-dot.active .dot{animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ─── Badges ────────────────────────────────────────────────────────────── */
.status-row{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:16px}
.badge{
  padding:4px 12px;border-radius:16px;font-size:0.72rem;font-weight:600;
  display:inline-flex;align-items:center;gap:5px
}
.badge.ok{background:var(--green-dim);color:var(--green);border:1px solid var(--green-border)}
.badge.err{background:var(--red-dim);color:var(--red);border:1px solid var(--red-border)}
.badge.info{background:var(--accent-dim);color:var(--accent2);border:1px solid #312e81}
.badge.warn{background:var(--yellow-dim);color:var(--yellow);border:1px solid var(--yellow-border)}

/* ─── Error / Clarification boxes ───────────────────────────────────────── */
.msg-box{border-radius:var(--radius-sm);padding:16px;margin-bottom:14px;font-size:0.88rem}
.msg-box.err{background:#1c050510;border:1px solid var(--red-border);color:var(--red)}
.msg-box.warn{background:#1c100310;border:1px solid var(--yellow-border);color:var(--yellow)}
.msg-box.info{background:#0f172a10;border:1px solid #312e81;color:var(--accent2)}
.msg-box strong{font-weight:700;display:block;margin-bottom:6px}
.err-detail{font-size:0.78rem;color:var(--text3);margin-top:8px;font-family:var(--font-mono)}
.retry-btn{
  margin-top:12px;padding:7px 16px;background:transparent;
  border:1px solid var(--red-border);color:var(--red);border-radius:var(--radius-sm);
  font-size:0.8rem;cursor:pointer;transition:all .15s
}
.retry-btn:hover{background:var(--red-dim)}

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
.tab-nav-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-bottom:16px}
.tabs{
  display:flex;gap:2px;padding-bottom:0;
  border-bottom:1px solid var(--border);
  min-width:max-content
}
.tab{
  padding:8px 14px;border-radius:var(--radius-sm) var(--radius-sm) 0 0;
  font-size:0.78rem;cursor:pointer;border:1px solid transparent;
  border-bottom:none;background:transparent;color:var(--text3);
  transition:all .15s;white-space:nowrap;display:flex;align-items:center;gap:5px;
  position:relative;bottom:-1px
}
.tab.active{background:var(--surface2);color:var(--accent2);
  border-color:var(--border);border-bottom-color:var(--surface2)}
.tab:hover:not(.active){color:var(--text2);background:var(--surface2)30}
.tab-badge{
  background:var(--accent-dim);color:var(--accent2);
  border-radius:8px;padding:1px 6px;font-size:0.65rem;font-weight:700
}
.tab-badge.err{background:var(--red-dim);color:var(--red)}
.tab-badge.ok{background:var(--green-dim);color:var(--green)}
.tab-pane{display:none;animation:fadeIn .18s ease}
.tab-pane.active{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}

/* ─── JSON Viewer ────────────────────────────────────────────────────────── */
.json-viewer-wrap{position:relative}
.json-toolbar{
  display:flex;align-items:center;gap:6px;
  background:var(--surface2);border:1px solid var(--border);
  border-bottom:none;border-radius:var(--radius-sm) var(--radius-sm) 0 0;
  padding:7px 12px
}
.json-toolbar-label{font-size:0.7rem;color:var(--text3);flex:1;font-family:var(--font-mono)}
.jbtn{
  background:transparent;border:1px solid var(--border2);color:var(--text3);
  padding:3px 10px;border-radius:var(--radius-xs);font-size:0.7rem;cursor:pointer;
  transition:all .15s;display:flex;align-items:center;gap:4px;white-space:nowrap
}
.jbtn:hover{border-color:var(--accent);color:var(--accent2)}
.jbtn.copied{border-color:var(--green-border);color:var(--green)}
pre.json-pre{
  background:var(--bg);border:1px solid var(--border);
  border-radius:0 0 var(--radius-sm) var(--radius-sm);
  padding:16px;overflow:auto;font-size:0.76rem;line-height:1.7;
  white-space:pre-wrap;word-break:break-word;max-height:460px;
  font-family:var(--font-mono);margin:0
}
/* Syntax highlight tokens */
.jk{color:#818cf8}          /* key */
.js{color:#86efac}          /* string value */
.jn{color:#fb923c}          /* number */
.jb{color:#f472b6}          /* boolean */
.jnull{color:#94a3b8}       /* null */
.jp{color:#64748b}          /* punctuation */

/* ─── Section titles ────────────────────────────────────────────────────── */
.section-title{
  font-size:0.72rem;font-weight:700;color:var(--text3);
  text-transform:uppercase;letter-spacing:.09em;margin:18px 0 10px;
  display:flex;align-items:center;gap:6px
}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}

/* ─── Assumption list ───────────────────────────────────────────────────── */
.assumption-list{background:var(--bg);border:1px solid var(--border);
  border-radius:var(--radius-sm);overflow:hidden}
.assumption-item{
  display:flex;gap:10px;padding:9px 14px;font-size:0.82rem;color:var(--text2);
  border-bottom:1px solid var(--border);align-items:flex-start
}
.assumption-item:last-child{border-bottom:none}
.assumption-arrow{color:var(--accent);flex-shrink:0;margin-top:1px}

/* ─── Issue cards ───────────────────────────────────────────────────────── */
.issues-list{display:flex;flex-direction:column;gap:8px}
.issue-card{
  background:var(--bg);border-radius:var(--radius-sm);
  padding:12px 14px;border-left:3px solid var(--red-border);
  border:1px solid var(--border);border-left-width:3px
}
.issue-card.resolved{border-left-color:var(--green-border)}
.issue-card.ok-msg{border-left-color:var(--green-border);background:var(--green-dim)20}
.issue-type{
  font-size:0.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.07em;color:var(--red);margin-bottom:4px;
  display:flex;align-items:center;gap:6px
}
.issue-type.resolved{color:var(--green)}
.issue-location{font-size:0.73rem;color:var(--accent2);
  font-family:var(--font-mono);margin-bottom:5px}
.issue-detail{font-size:0.82rem;color:var(--text2)}

/* ─── Repair ────────────────────────────────────────────────────────────── */
.repair-card{
  background:var(--bg);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:14px;margin-bottom:10px
}
.repair-card-header{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap}
.attempt-badge{
  background:var(--yellow-dim);color:var(--yellow);
  border:1px solid var(--yellow-border);padding:2px 9px;
  border-radius:12px;font-size:0.68rem;font-weight:700
}
.layer-badge{
  background:var(--accent-dim);color:var(--accent2);
  border:1px solid #312e81;padding:2px 9px;
  border-radius:12px;font-size:0.68rem
}
.repair-meta{font-size:0.76rem;color:var(--text3)}
.repair-meta span{color:var(--text2)}

/* ─── Runtime ───────────────────────────────────────────────────────────── */
.verdict-box{
  border-radius:var(--radius-sm);padding:14px 18px;
  font-weight:600;font-size:0.9rem;margin-bottom:16px;
  display:flex;align-items:center;gap:10px
}
.verdict-box.pass{background:var(--green-dim);border:1px solid var(--green-border);color:var(--green)}
.verdict-box.fail{background:var(--red-dim);border:1px solid var(--red-border);color:var(--red)}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.stat-box{
  background:var(--bg);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:14px;text-align:center
}
.stat-num{font-size:1.9rem;font-weight:800;color:var(--accent2)}
.stat-num.g{color:var(--green)}
.stat-num.r{color:var(--red)}
.stat-label{font-size:0.7rem;color:var(--text3);margin-top:3px}

/* ─── Table ─────────────────────────────────────────────────────────────── */
.table-wrap{overflow-x:auto;border-radius:var(--radius-sm);border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:0.78rem;min-width:500px}
th{
  text-align:left;padding:9px 13px;background:var(--surface2);
  color:var(--text3);font-weight:600;font-size:0.72rem;letter-spacing:.04em;
  text-transform:uppercase;border-bottom:1px solid var(--border)
}
td{padding:9px 13px;border-bottom:1px solid var(--border);color:var(--text2)}
tr:last-child td{border-bottom:none}
td code{font-family:var(--font-mono);font-size:0.73rem;color:var(--accent2)}
.cell-pass{color:var(--green);font-weight:700}
.cell-fail{color:var(--red);font-weight:700}

/* ─── Cost & Quality ────────────────────────────────────────────────────── */
.metrics-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:18px}
.metric-card{
  background:var(--bg);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:16px
}
.metric-label{font-size:0.7rem;color:var(--text3);text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:6px}
.metric-value{font-size:1.5rem;font-weight:800;color:var(--accent2)}
.metric-sub{font-size:0.72rem;color:var(--text3);margin-top:3px}
.quality-track{background:var(--surface2);border-radius:4px;height:6px;overflow:hidden;margin:8px 0}
.quality-fill{height:100%;border-radius:4px;transition:width .6s ease}

.tel-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:9px}
.tel-card{
  background:var(--bg);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:12px
}
.tel-stage{font-size:0.68rem;color:var(--accent);font-weight:700;margin-bottom:3px}
.tel-model{font-size:0.65rem;color:var(--text3);margin-bottom:5px}
.tel-val{font-size:0.82rem;color:var(--text)}
.tel-tokens{font-size:0.68rem;color:var(--text3);margin-top:2px}

.tradeoffs{background:var(--bg);border:1px solid var(--border);
  border-radius:var(--radius-sm);overflow:hidden}
.tradeoff-row{
  display:flex;gap:0;border-bottom:1px solid var(--border);
  font-size:0.82rem
}
.tradeoff-row:last-child{border-bottom:none}
.tradeoff-k{
  padding:9px 14px;color:var(--accent2);font-weight:600;
  background:var(--surface2);min-width:180px;border-right:1px solid var(--border)
}
.tradeoff-v{padding:9px 14px;color:var(--text2)}

/* ─── Utility ───────────────────────────────────────────────────────────── */
.hidden{display:none!important}
.ok-green{color:var(--green)}
.ok-card{background:var(--green-dim);border:1px solid var(--green-border);
  border-radius:var(--radius-sm);padding:12px 16px;color:var(--green);
  font-size:0.85rem;font-weight:600}

/* ─── Preview ───────────────────────────────────────────────────────────── */
.preview-container { display: flex; flex-direction: column; gap: 10px; }
.preview-device-toggles { display: flex; gap: 8px; justify-content: center; margin-bottom: 16px; }
.device-btn { background: var(--surface2); border: 1px solid var(--border); color: var(--text2); padding: 6px 12px; border-radius: var(--radius-sm); font-size: 0.75rem; cursor: pointer; }
.device-btn.active { background: var(--accent-dim); color: var(--accent2); border-color: var(--accent); }
.preview-frame { border: 1px solid var(--border); border-radius: var(--radius); background: #f8fafc; color: #0f172a; min-height: 500px; margin: 0 auto; transition: width 0.3s; overflow: hidden; display: flex; flex-direction: column; }
.preview-frame.desktop { width: 100%; }
.preview-frame.tablet { width: 768px; }
.preview-frame.mobile { width: 375px; height: 667px; }
.preview-nav { background: #fff; border-bottom: 1px solid #e2e8f0; padding: 12px 20px; display: flex; gap: 16px; font-size: 0.85rem; font-weight: 600; align-items: center; }
.preview-nav-item { cursor: pointer; color: #64748b; }
.preview-nav-item.active { color: #3b82f6; }
.preview-content { padding: 24px; flex: 1; overflow-y: auto; }
.preview-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); background: #fff; }
.preview-card h3 { margin-bottom: 16px; font-size: 1rem; color: #1e293b; }
.preview-form label { display: block; font-size: 0.75rem; color: #64748b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.preview-form input, .preview-form select { width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; margin-bottom: 16px; background: #fff; color: #0f172a; }
.preview-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-bottom: 16px; }
.preview-table th, .preview-table td { border-bottom: 1px solid #e2e8f0; padding: 12px 8px; text-align: left; }
.preview-table th { font-weight: 600; color: #64748b; text-transform: uppercase; font-size: 0.7rem; letter-spacing: .05em; }
.preview-btn { background: #3b82f6; color: #fff; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; }

/* ─── Files ─────────────────────────────────────────────────────────────── */
.files-container { display: flex; height: 600px; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; background: var(--bg); }
.files-sidebar { width: 250px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
.files-header { padding: 12px 16px; font-weight: 700; font-size: 0.8rem; color: var(--text3); border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: .05em; }
.files-tree { flex: 1; overflow-y: auto; padding: 10px 0; }
.file-item { padding: 6px 16px; cursor: pointer; display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--text2); transition: all .15s; }
.file-item:hover { background: var(--surface2); color: var(--text); }
.file-item.active { background: var(--accent-dim); color: var(--accent2); border-right: 3px solid var(--accent); }
.file-item-icon { color: var(--text3); }
.files-editor { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.files-editor-header { padding: 12px 16px; background: var(--surface2); border-bottom: 1px solid var(--border); font-family: var(--font-mono); font-size: 0.8rem; color: var(--text2); display: flex; justify-content: space-between; }
.files-editor-content { flex: 1; overflow: auto; padding: 16px; background: #0f111a; margin: 0; font-family: var(--font-mono); font-size: 0.8rem; color: #e2e8f0; white-space: pre; }
.hljs-keyword { color: #c678dd; }
.hljs-string { color: #98c379; }
.hljs-comment { color: #5c6370; font-style: italic; }
.hljs-function { color: #61afef; }
.hljs-tag { color: #e06c75; }

/* ─── Responsive ────────────────────────────────────────────────────────── */
@media(max-width:640px){
  header{padding:12px 16px}
  .logo-text p{display:none}
  .main{padding:0 10px;margin:16px auto}
  .card{padding:16px}
  .stat-grid{grid-template-columns:repeat(2,1fr)}
  .metrics-grid{grid-template-columns:1fr}
  .tradeoff-k{min-width:120px;font-size:0.75rem}
}

/* ─── Chat & Versioning ─────────────────────────────────────────────────── */
.refine-bar { display: flex; gap: 10px; margin-top: 16px; background: var(--surface2); padding: 12px; border-radius: var(--radius); border: 1px solid var(--border); }
.refine-bar input { flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 10px 16px; border-radius: 20px; font-size: 0.85rem; outline: none; }
.refine-bar input:focus { border-color: var(--accent); }
.refine-btn { background: var(--accent); color: #fff; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; font-weight: 600; font-size: 0.85rem; }
.versions-dropdown { background: var(--surface2); border: 1px solid var(--border); color: var(--text2); padding: 6px 12px; border-radius: var(--radius-sm); font-size: 0.75rem; cursor: pointer; outline: none; margin-left: auto; }
</style>
</head>
<body>

<!-- ─── Header ──────────────────────────────────────────────────────────── -->
<header>
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <div class="logo-text">
      <h1>nl2app</h1>
      <p>AI App Compiler · Natural language → validated executable schema</p>
    </div>
  </div>
  <div class="header-meta">
    <a href="/docs" target="_blank" class="docs-link">API Docs ↗</a>
    <span class="ver-badge">v1.0.0</span>
  </div>
</header>

<!-- ─── Main ─────────────────────────────────────────────────────────────── -->
<div class="main">

  <!-- Input card -->
  <div class="card">
    <div class="card-title">Describe the app you want to build</div>
    <div class="prompt-wrap">
      <textarea id="prompt"
        placeholder="e.g. Build a CRM with contacts, deals, and companies. Admin can do everything, sales reps can manage their own deals, managers view all deals and run reports."
        oninput="updateCharCount(this)"
        onkeydown="if(event.key==='Enter'&&(event.ctrlKey||event.metaKey))compile()"></textarea>
      <span class="char-count" id="char-count">0</span>
    </div>
    <div class="examples">
      <span class="ex-label">App Templates:</span>
      <button class="ex-btn" onclick="setPrompt(this)">Simple CRM with admin and sales roles</button>
      <button class="ex-btn" onclick="setPrompt(this)">E-commerce store with products, orders, customers</button>
      <button class="ex-btn" onclick="setPrompt(this)">Project management tool with tasks, sprints, developer and manager roles</button>
      <button class="ex-btn" onclick="setPrompt(this)">Hospital booking system with patients, doctors, appointments</button>
      <button class="ex-btn" onclick="setPrompt(this)">Online learning platform with courses, students, instructors</button>
      <button class="ex-btn" onclick="setPrompt(this)">Helpdesk with tickets, agents, customers, SLA rules</button>
    </div>
    <div style="display:flex; align-items:center; gap: 12px; margin-top: 14px;">
      <button id="run" onclick="compile()" style="flex:1; margin-top:0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        Compile App
      </button>
      <select id="version-select" class="versions-dropdown hidden" onchange="restoreVersion(this.value)">
        <option value="">Versions...</option>
      </select>
    </div>
  </div>

  <!-- Results area -->
  <div id="results" class="hidden">
    <div class="card">

      <!-- Pipeline progress (shown during loading) -->
      <div id="pipeline-progress">
        <div class="progress-label">
          <span id="stage-label">Initializing…</span>
          <span id="stage-pct">0%</span>
        </div>
        <div class="progress-bar-track"><div class="progress-bar-fill" id="prog-fill"></div></div>
        <div class="stages" id="stage-dots"></div>
      </div>

      <!-- Status badges -->
      <div class="status-row" id="status-row"></div>

      <!-- Error box -->
      <div id="err-box" class="msg-box err hidden">
        <strong>⚠ Compilation Failed</strong>
        <div id="err-text"></div>
        <div id="err-detail" class="err-detail hidden"></div>
        <button class="retry-btn" onclick="compile()">↺ Retry</button>
      </div>

      <!-- Clarification box -->
      <div id="clarify-box" class="msg-box warn hidden">
        <strong>❓ Clarification Needed</strong>
        <div id="clarify-text"></div>
      </div>

      <!-- Tabs -->
      <div id="main-tabs" class="hidden">
        <div class="tab-nav-wrap">
          <div class="tabs" id="tab-bar">
            <button class="tab active" data-tab="preview">📱 Preview</button>
            <button class="tab" data-tab="files">📂 Files</button>
            <button class="tab" data-tab="deploy">🚀 Deploy</button>
            <button class="tab" data-tab="export">📦 Export</button>
            <button class="tab" data-tab="runtime">🔥 Runtime</button>
            <button class="tab" data-tab="intent">Intent</button>
            <button class="tab" data-tab="architecture">Architecture</button>
            <button class="tab" data-tab="db">DB Schema</button>
            <button class="tab" data-tab="api">API Schema</button>
            <button class="tab" data-tab="ui">UI Schema</button>
            <button class="tab" data-tab="auth">Auth Rules</button>
            <button class="tab" data-tab="validation">Validation <span id="val-badge" class="tab-badge hidden"></span></button>
            <button class="tab" data-tab="repair">Repair <span id="rep-badge" class="tab-badge hidden"></span></button>
            <button class="tab" data-tab="cost">Cost &amp; Quality</button>
          </div>
        </div>

        <div id="pane-preview"      class="tab-pane active"></div>
        <div id="pane-files"        class="tab-pane"></div>
        <div id="pane-deploy"       class="tab-pane"></div>
        <div id="pane-export"       class="tab-pane"></div>
        <div id="pane-runtime"      class="tab-pane"></div>
        <div id="pane-intent"       class="tab-pane"></div>
        <div id="pane-architecture" class="tab-pane"></div>
        <div id="pane-db"           class="tab-pane"></div>
        <div id="pane-api"          class="tab-pane"></div>
        <div id="pane-ui"           class="tab-pane"></div>
        <div id="pane-auth"         class="tab-pane"></div>
        <div id="pane-validation"   class="tab-pane"></div>
        <div id="pane-repair"       class="tab-pane"></div>
        <div id="pane-cost"         class="tab-pane"></div>
      </div>

    </div>
  </div>
</div><!-- /.main -->

<script>
/* ══════════════════════════════════════════════════════
   Utility helpers
══════════════════════════════════════════════════════ */
const esc = s => String(s || '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const $ = id => document.getElementById(id);

let _appHistory = []; // stores compiled app objects
let _currentVersion = 0;
let _basePrompt = "";

function updateCharCount(el){
  $('char-count').textContent = el.value.length;
}

function setPrompt(btn){
  const ta = $('prompt');
  ta.value = btn.textContent.trim();
  ta.focus();
  updateCharCount(ta);
}

/* ── Tab system ─────────────────────────────────────── */
document.addEventListener('click', e => {
  const btn = e.target.closest('[data-tab]');
  if(!btn) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  $('pane-' + btn.dataset.tab).classList.add('active');
});

/* ── JSON syntax highlighter (no external deps) ─────── */
function highlightJSON(raw){
  // Tokenise with a single regex pass
  return raw.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\],:])/g,
    match => {
      let cls = 'jn'; // number
      if(/^"/.test(match)){
        cls = /:$/.test(match) ? 'jk' : 'js';
      } else if(/true|false/.test(match)){
        cls = 'jb';
      } else if(match === 'null'){
        cls = 'jnull';
      } else if(/[{}\[\],:]/.test(match)){
        cls = 'jp';
      }
      return `<span class="${cls}">${esc(match)}</span>`;
    }
  );
}

/* ── JSON viewer widget ─────────────────────────────── */
function jsonViewer(id, obj, label){
  const raw = JSON.stringify(obj, null, 2);
  const hl  = highlightJSON(raw);
  const bytes = new TextEncoder().encode(raw).length;
  const sizeLabel = bytes < 1024 ? `${bytes} B` : `${(bytes/1024).toFixed(1)} KB`;
  return `
<div class="json-viewer-wrap">
  <div class="json-toolbar">
    <span class="json-toolbar-label">${esc(label || '')} · ${sizeLabel}</span>
    <button class="jbtn" onclick="copyJSON('${id}','${esc(label||'')}')">⎘ Copy</button>
    <button class="jbtn" onclick="downloadJSON('${id}','${esc(label||'')}')">↓ Download</button>
  </div>
  <pre class="json-pre" id="${id}">${hl}</pre>
</div>`;
}

function copyJSON(id, label){
  const pre = $(id);
  const text = pre.innerText; // plain text (no HTML tags)
  navigator.clipboard.writeText(text).then(()=>{
    // find the copy button within the same toolbar
    const btn = pre.previousElementSibling.querySelector('.jbtn');
    const orig = btn.innerHTML;
    btn.innerHTML = '✓ Copied!';
    btn.classList.add('copied');
    setTimeout(()=>{ btn.innerHTML=orig; btn.classList.remove('copied'); }, 1800);
  });
}

function downloadJSON(id, label){
  const text = $(id).innerText;
  const blob = new Blob([text], {type:'application/json'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = (label||'data').replace(/\s+/g,'_').toLowerCase() + '.json';
  a.click();
  URL.revokeObjectURL(url);
}

/* ══════════════════════════════════════════════════════
   Pipeline progress animation
══════════════════════════════════════════════════════ */
const STAGES = [
  'Intent extraction',
  'Architecture design',
  'Schema generation',
  'Validation',
  'Repair',
  'Runtime smoke test'
];
let _progressTimer = null;
let _stageIdx = 0;

function startProgress(){
  _stageIdx = 0;
  const dots = $('stage-dots');
  dots.innerHTML = STAGES.map((s,i)=>
    `<div class="stage-dot${i===0?' active':''}" id="sdot-${i}"><span class="dot"></span>${s}</div>`
  ).join('');
  $('pipeline-progress').style.display = 'block';
  updateProgress(0);
  // Auto-advance every ~5 seconds (pipeline takes 25-45 s real time)
  _progressTimer = setInterval(()=>{
    if(_stageIdx < STAGES.length - 1) advanceStage();
  }, 5200);
}

function advanceStage(){
  const prev = $('sdot-' + _stageIdx);
  if(prev){ prev.classList.remove('active'); prev.classList.add('done'); }
  _stageIdx = Math.min(_stageIdx + 1, STAGES.length - 1);
  const cur = $('sdot-' + _stageIdx);
  if(cur){ cur.classList.remove('done'); cur.classList.add('active'); }
  const pct = Math.round((_stageIdx / (STAGES.length - 1)) * 90);
  updateProgress(pct);
}

function updateProgress(pct){
  $('prog-fill').style.width = pct + '%';
  $('stage-pct').textContent = pct + '%';
  $('stage-label').textContent = STAGES[_stageIdx] + '…';
}

function finishProgress(){
  clearInterval(_progressTimer);
  // Mark all done
  STAGES.forEach((_,i)=>{
    const d = $('sdot-' + i);
    if(d){ d.classList.remove('active'); d.classList.add('done'); }
  });
  updateProgress(100);
  setTimeout(()=>{ $('pipeline-progress').style.display='none'; }, 600);
}

function resetProgress(){
  clearInterval(_progressTimer);
  $('pipeline-progress').style.display = 'none';
}

/* ══════════════════════════════════════════════════════
   Core compile function
══════════════════════════════════════════════════════ */
function updateVersionDropdown() {
  const sel = $('version-select');
  sel.innerHTML = _appHistory.map((h, i) => `<option value="${i}">v${i+1}: ${h.prompt.substring(0,20)}...</option>`).join('');
  sel.value = _currentVersion;
  sel.classList.remove('hidden');
}

function restoreVersion(idx) {
  _currentVersion = parseInt(idx);
  renderAll(_appHistory[_currentVersion].data);
}

async function compile(){
  const prompt = $('prompt').value.trim();
  if(!prompt) return;

  // Reset UI
  const btn = $('run');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div>Compiling…';

  $('results').classList.remove('hidden');
  $('status-row').innerHTML = '';
  $('err-box').classList.add('hidden');
  $('clarify-box').classList.add('hidden');
  $('main-tabs').classList.add('hidden');
  startProgress();

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    
    if(!data.success) throw new Error(data.error || 'Unknown error');
    
    if(_appHistory.length === 0) _basePrompt = prompt;
    _appHistory.push({ prompt: prompt, data: data });
    _currentVersion = _appHistory.length - 1;
    updateVersionDropdown();
    
    renderAll(data);
  } catch(err) {
    finishProgress();
    $('status-row').innerHTML = '<span class="badge err">✗ Network Error</span>';
    $('err-text').textContent = err.message;
    $('err-box').classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Compile App';
  }
}

function renderAll(data) {
  finishProgress();
  renderResults(data);
}

/* ══════════════════════════════════════════════════════
   Result renderers
══════════════════════════════════════════════════════ */
function renderResults(data){
  if(!data.success){
    if(data.needs_clarification){
      $('status-row').innerHTML = '<span class="badge warn">❓ Needs Clarification</span>';
      $('clarify-text').textContent = data.clarification_question || '';
      $('clarify-box').classList.remove('hidden');
    } else {
      $('status-row').innerHTML =
        '<span class="badge err">✗ Failed</span>' +
        '<span class="badge info">⏱ ' + data.total_latency_seconds + 's</span>';
      $('err-text').textContent = data.error || 'An unknown error occurred.';
      const det = $('err-detail');
      if(data.traceback){ det.textContent = data.traceback; det.classList.remove('hidden'); }
      $('err-box').classList.remove('hidden');
    }
    return;
  }

  const smoke   = data.runtime_proof || {};
  const repairs = data.compiled_app.repair_attempts || 0;
  const initIssues  = data.stage_outputs?.initial_validation?.issues || [];
  const finalIssues = data.stage_outputs?.final_validation?.issues || [];
  const quality = data.quality_score || 0;
  const cost    = data.estimated_cost_usd;
  const passRate = smoke.pass_rate != null ? (smoke.pass_rate*100).toFixed(0)+'%' : 'N/A';

  $('status-row').innerHTML =
    '<span class="badge ok">✓ Compiled</span>' +
    '<span class="badge info">⏱ ' + data.total_latency_seconds + 's</span>' +
    '<span class="badge ' + (repairs > 0 ? 'warn' : 'info') + '">🔧 ' + repairs + ' repair' + (repairs !== 1 ? 's' : '') + '</span>' +
    '<span class="badge ' + (finalIssues.length === 0 ? 'ok' : 'warn') + '">' + (finalIssues.length === 0 ? '✓' : '⚠') + ' ' + finalIssues.length + ' issue' + (finalIssues.length !== 1 ? 's' : '') + '</span>' +
    '<span class="badge ' + (smoke.failed === 0 ? 'ok' : 'err') + '">🔥 ' + passRate + ' runtime</span>' +
    '<span class="badge info">💰 ' + (cost != null ? '$' + cost.toFixed(5) : 'N/A') + '</span>';

  // Update tab badges
  const vb = $('val-badge');
  if(initIssues.length > 0){
    vb.textContent = initIssues.length;
    vb.className   = 'tab-badge ' + (finalIssues.length === 0 ? 'ok' : 'err');
    vb.classList.remove('hidden');
  }
  const rb = $('rep-badge');
  if(repairs > 0){
    rb.textContent = repairs;
    rb.className   = 'tab-badge warn';
    rb.classList.remove('hidden');
  }

  $('main-tabs').classList.remove('hidden');

  renderPreview(data);
  renderFiles(data);
  renderDeploy(data);
  renderExport(data);
  renderRuntime(data);
  renderIntent(data);
  renderArchitecture(data);
  renderSchemaPane('db',  data.compiled_app.schema_bundle.db_schema,   'db_schema.json');
  renderSchemaPane('api', data.compiled_app.schema_bundle.api_schema,  'api_schema.json');
  renderSchemaPane('ui',  data.compiled_app.schema_bundle.ui_schema,   'ui_schema.json');
  renderSchemaPane('auth',data.compiled_app.schema_bundle.auth_schema, 'auth_schema.json');
  renderValidation(data);
  renderRepair(data);
  renderCost(data);
}

/* ── Runtime tab ──────────────────────────────────────*/
function renderRuntime(data){
  const smoke = data.runtime_proof;
  if(!smoke){
    $('pane-runtime').innerHTML = '<p style="color:var(--text3);padding:16px">No runtime data available.</p>';
    return;
  }
  const isPass = smoke.failed === 0;
  const rows = (smoke.details || []).map(d => `
    <tr>
      <td><code>${esc(d.endpoint)}</code></td>
      <td>${esc(d.role)}</td>
      <td>${esc(String(d.expected))}</td>
      <td>${esc(String(d.actual_status))}</td>
      <td class="${d.passed ? 'cell-pass' : 'cell-fail'}">${d.passed ? '✓ PASS' : '✗ FAIL'}</td>
    </tr>`).join('');

  $('pane-runtime').innerHTML = `
    <div class="verdict-box ${isPass ? 'pass' : 'fail'}">
      ${isPass ? '✓' : '✗'} ${esc(smoke.verdict || '')}
    </div>
    <div class="stat-grid">
      <div class="stat-box"><div class="stat-num">${smoke.total_checks}</div><div class="stat-label">Total Checks</div></div>
      <div class="stat-box"><div class="stat-num g">${smoke.passed}</div><div class="stat-label">Passed</div></div>
      <div class="stat-box"><div class="stat-num r">${smoke.failed}</div><div class="stat-label">Failed</div></div>
      <div class="stat-box"><div class="stat-num">${smoke.pass_rate != null ? (smoke.pass_rate*100).toFixed(0) + '%' : 'N/A'}</div><div class="stat-label">Pass Rate</div></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Endpoint</th><th>Role</th><th>Expected Status</th><th>Actual Status</th><th>Result</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ── Preview tab ──────────────────────────────────────*/
function renderPreview(data){
  const runtime = data.stage_outputs.runtime;
  if(!runtime || !runtime.pages || runtime.pages.length === 0){
    $('pane-preview').innerHTML = '<p style="color:var(--text3);padding:16px">No runtime preview available.</p>';
    return;
  }
  
  // Default to first page
  let html = `
    <div class="preview-container">
      <div class="preview-device-toggles">
        <button class="device-btn active" onclick="setDevice('desktop', this)">Desktop</button>
        <button class="device-btn" onclick="setDevice('tablet', this)">Tablet</button>
        <button class="device-btn" onclick="setDevice('mobile', this)">Mobile</button>
      </div>
      <div class="preview-frame desktop" id="preview-frame">
        <div class="preview-nav">
          <div style="font-weight:800; color:#0f172a; margin-right:16px;">AppLogo</div>
          ${runtime.pages.map((p, i) => `<div class="preview-nav-item ${i===0?'active':''}" onclick="navPreview('${p.id}', this)">${esc(p.name)}</div>`).join('')}
        </div>
        <div class="preview-content" id="preview-content">
          ${renderPreviewPage(runtime.pages[0])}
        </div>
      </div>
      <div class="refine-bar">
      <input type="text" id="refine-input" placeholder="Refine this app (e.g. 'Add a dark mode toggle')">
      <button class="refine-btn" onclick="refineApp()">Refine 🪄</button>
    </div>
  </div>`;
  $('pane-preview').innerHTML = html;
  
  // Store runtime for navigation
  window._runtimeApp = runtime;
}

function setDevice(type, btn) {
  document.querySelectorAll('.device-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const frame = $('preview-frame');
  frame.className = 'preview-frame ' + type;
}

function navPreview(pageId, navItem) {
  document.querySelectorAll('.preview-nav-item').forEach(n => n.classList.remove('active'));
  navItem.classList.add('active');
  const page = window._runtimeApp.pages.find(p => p.id === pageId);
  if(page) {
    $('preview-content').innerHTML = renderPreviewPage(page);
  }
}

function renderPreviewPage(page) {
  let html = `<div style="background:#eef2ff; color:#4f46e5; padding:12px; border-radius:6px; margin-bottom:20px; font-size:0.85rem; border:1px solid #c7d2fe;">
    <strong>✨ High-Fidelity UI Generated!</strong> This is just a structural wireframe. The actual exported React code contains beautiful Tailwind CSS components written by the LLM. Go to the <strong>Files</strong> or <strong>Export</strong> tab to view and download the real UI.
  </div>`;
  html += `<h2 style="margin-bottom:20px; color:#0f172a">${esc(page.name)}</h2>`;
  
  page.components.forEach(comp => {
    html += `<div class="preview-card"><h3>${esc(comp.label)}</h3>`;
    if(comp.type === 'form') {
      html += `<div class="preview-form">`;
      (comp.props.bound_fields || []).forEach(f => {
        html += `<label>${esc(f)}</label><input type="text" placeholder="Enter ${esc(f)}..." />`;
      });
      if(!comp.props.bound_fields || comp.props.bound_fields.length === 0) {
        html += `<label>Name</label><input type="text" placeholder="Name..." />`;
      }
      html += `<button class="preview-btn">Submit</button></div>`;
    } else if(comp.type === 'table') {
      html += `<table class="preview-table"><thead><tr>`;
      const cols = comp.props.bound_fields && comp.props.bound_fields.length > 0 ? comp.props.bound_fields : ['ID', 'Name', 'Date', 'Status'];
      cols.forEach(c => html += `<th>${esc(c)}</th>`);
      html += `</tr></thead><tbody>`;
      for(let i=0; i<3; i++) {
        html += `<tr>`;
        cols.forEach(c => html += `<td>Sample ${esc(c)}</td>`);
        html += `</tr>`;
      }
      html += `</tbody></table>`;
    } else if(comp.type === 'card' || comp.type === 'text') {
      html += `<p style="color:#64748b; font-size:0.9rem">Sample content for ${esc(comp.label)}...</p>`;
    } else {
      html += `<div style="padding:20px; border:1px dashed #cbd5e1; text-align:center; color:#94a3b8; border-radius:4px">[${esc(comp.type)} placeholder]</div>`;
    }
    html += `</div>`;
  });
  
  return html;
}

/* ── Files tab ───────────────────────────────────────*/
function renderFiles(data) {
  const files = data.stage_outputs.files;
  if(!files || Object.keys(files).length === 0){
    $('pane-files').innerHTML = '<p style="color:var(--text3);padding:16px">No generated files available.</p>';
    return;
  }
  
  window._generatedFiles = files;
  const fileKeys = Object.keys(files).sort();
  
  let html = `
    <div class="files-container">
      <div class="files-sidebar">
        <div class="files-header">Project Explorer</div>
        <div class="files-tree" id="files-tree">
          ${fileKeys.map(k => `
            <div class="file-item" onclick="viewFile('${esc(k)}', this)">
              <span class="file-item-icon">📄</span> ${esc(k)}
            </div>
          `).join('')}
        </div>
      </div>
      <div class="files-editor">
        <div class="files-editor-header">
          <span id="current-file-name">Select a file</span>
          <button class="jbtn" onclick="copyCurrentFile()">⎘ Copy</button>
        </div>
        <pre class="files-editor-content" id="file-editor-content">// No file selected</pre>
      </div>
    </div>
  `;
  $('pane-files').innerHTML = html;
  
  // Select first file by default
  if(fileKeys.length > 0) {
    const firstItem = $('files-tree').querySelector('.file-item');
    if(firstItem) viewFile(fileKeys[0], firstItem);
  }
}

function viewFile(path, el) {
  document.querySelectorAll('.file-item').forEach(n => n.classList.remove('active'));
  if(el) el.classList.add('active');
  
  $('current-file-name').textContent = path;
  
  let code = window._generatedFiles[path] || "";
  
  // Super simple syntax highlight for react/express JS
  let highlighted = esc(code)
    .replace(/\b(import|export|from|const|let|var|function|return|if|else|for|while|class|extends|require|module|exports)\b/g, '<span class="hljs-keyword">$1</span>')
    .replace(/("[^"]*"|'[^']*'|`[^`]*`)/g, '<span class="hljs-string">$1</span>')
    .replace(/(&lt;\/?)([a-zA-Z0-9]+)(\s|&gt;|\/&gt;)/g, '$1<span class="hljs-tag">$2</span>$3')
    .replace(/(\/\/.*$)/gm, '<span class="hljs-comment">$1</span>');
    
  $('file-editor-content').innerHTML = highlighted;
}

function copyCurrentFile() {
  const path = $('current-file-name').textContent;
  const code = window._generatedFiles[path] || "";
  navigator.clipboard.writeText(code).then(() => {
    // optional toast
  });
}

/* ── Deploy tab ───────────────────────────────────────*/
function renderDeploy(data) {
  const files = data.stage_outputs.files || {};
  let html = `<div style="padding: 16px;">`;
  html += `<h2 style="margin-bottom: 16px; color: var(--text);">🚀 Deployment Center</h2>`;
  html += `<p style="color: var(--text2); margin-bottom: 24px;">Generated configuration files for standard deployment platforms.</p>`;
  
  const platforms = [
    { name: "Vercel", file: "vercel.json" },
    { name: "Render", file: "render.yaml" },
    { name: "Netlify", file: "netlify.toml" }
  ];
  
  html += `<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px;">`;
  platforms.forEach(p => {
    if(files[p.file]) {
      html += `
        <div class="card" style="margin-bottom: 0;">
          <h3 style="margin-bottom: 8px; color: var(--accent2);">${p.name}</h3>
          <p style="font-size: 0.8rem; color: var(--text3); margin-bottom: 12px;">Configuration: <code>${p.file}</code></p>
          <pre style="background: #0f111a; padding: 12px; border-radius: 4px; font-size: 0.75rem; color: #98c379; overflow-x: auto;">${esc(files[p.file])}</pre>
        </div>
      `;
    }
  });
  html += `</div></div>`;
  $('pane-deploy').innerHTML = html;
}

/* ── Export tab ───────────────────────────────────────*/
function renderExport(data) {
  let html = `
    <div style="padding: 32px; text-align: center;">
      <h2 style="margin-bottom: 16px; color: var(--text);">📦 One-Click Download</h2>
      <p style="color: var(--text2); margin-bottom: 32px; max-width: 500px; margin-left: auto; margin-right: auto;">
        Download the complete generated source code (Frontend, Backend, and Database models) as a ready-to-run ZIP archive.
      </p>
      <button class="preview-btn" style="padding: 14px 28px; font-size: 1rem; cursor: pointer;" onclick="downloadProject()">
        Download Application ZIP
      </button>
    </div>
  `;
  $('pane-export').innerHTML = html;
}

function downloadProject() {
  if(!window._generatedFiles) return;
  fetch('/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ files: window._generatedFiles })
  })
  .then(res => res.blob())
  .then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated-app.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  })
  .catch(err => alert("Export failed: " + err));
}

/* ── Intent tab ───────────────────────────────────────*/
function renderIntent(data){
  const intent      = data.compiled_app.intent;
  const assumptions = data.compiled_app.assumptions_made || [];
  const assumed     = assumptions.length
    ? assumptions.map(a => `<div class="assumption-item"><span class="assumption-arrow">→</span>${esc(a)}</div>`).join('')
    : '<div class="assumption-item"><span class="assumption-arrow">→</span>No explicit assumptions recorded.</div>';

  $('pane-intent').innerHTML =
    jsonViewer('json-intent', intent, 'intent.json') +
    '<div class="section-title">Assumptions &amp; Ambiguities Resolved</div>' +
    '<div class="assumption-list">' + assumed + '</div>';
}

/* ── Architecture tab ─────────────────────────────────*/
function renderArchitecture(data){
  const arch = data.compiled_app.architecture;
  
  let html = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
    <h2 style="font-size: 1rem; color: var(--text);">Architecture Graph</h2>
    <span style="font-size: 0.75rem; color: var(--text3);">Drag to pan, scroll to zoom</span>
  </div>
  <div id="arch-network" style="width:100%; height:600px; border:1px solid var(--border); border-radius:var(--radius); background:var(--bg);"></div>
  <div style="margin-top:20px;">
    ${jsonViewer('json-arch', arch, 'architecture.json')}
  </div>`;
  
  $('pane-architecture').innerHTML = html;
  
  // Build nodes and edges for vis-network
  const nodes = [];
  const edges = [];
  
  // Entities
  if(arch.entities) {
    arch.entities.forEach((ent, i) => {
      nodes.push({ id: 'ent_' + ent.name, label: ent.name, group: 'entity', shape: 'box', color: {background: '#3b82f6', border: '#2563eb'}, font: {color: '#ffffff'} });
      
      // Relations
      if(ent.fields) {
        ent.fields.forEach(f => {
          if(f.type === 'relation' && f.relation_target) {
            edges.push({ from: 'ent_' + ent.name, to: 'ent_' + f.relation_target, label: f.name, arrows: 'to', font: {size: 10, color: 'var(--text2)'}, color: 'var(--border2)' });
          }
        });
      }
    });
  }
  
  // Roles
  if(arch.roles) {
    arch.roles.forEach(role => {
      nodes.push({ id: 'role_' + role.name, label: 'Role: ' + role.name, group: 'role', shape: 'ellipse', color: {background: '#8b5cf6', border: '#7c3aed'}, font: {color: '#ffffff'} });
      
      if(role.permissions) {
        role.permissions.forEach(perm => {
          edges.push({ from: 'role_' + role.name, to: 'ent_' + perm.entity, label: perm.actions.join(','), arrows: 'to', dashes: true, font: {size: 9, color: 'var(--text3)'}, color: 'var(--border2)' });
        });
      }
    });
  }
  
  // Initialize network immediately after inserting into DOM
  setTimeout(() => {
    const container = document.getElementById('arch-network');
    const graphData = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
    const options = {
      physics: { stabilization: true, barnesHut: { gravitationalConstant: -3000, springLength: 150 } },
      interaction: { hover: true, zoomView: true, dragView: true }
    };
    new vis.Network(container, graphData, options);
  }, 100);
}

/* ── Generic schema pane ──────────────────────────────*/
function renderSchemaPane(name, obj, filename){
  $('pane-' + name).innerHTML = jsonViewer('json-' + name, obj, filename);
}

/* ── Validation tab ───────────────────────────────────*/
function renderValidation(data){
  const init  = data.stage_outputs?.initial_validation || {};
  const final = data.stage_outputs?.final_validation   || {};
  const initIssues  = init.issues  || [];
  const finalIssues = final.issues || [];
  const resolved    = initIssues.length - finalIssues.length;

  function cards(issues, showResolved){
    if(!issues.length)
      return '<div class="ok-card">✓ No issues found</div>';
    return '<div class="issues-list">' +
      issues.map(i => {
        const isRes = showResolved && !finalIssues.find(f => f.location===i.location && f.issue_type===i.issue_type);
        return `<div class="issue-card ${isRes ? 'resolved' : ''}">
          <div class="issue-type ${isRes ? 'resolved' : ''}">
            ${isRes ? '✓ RESOLVED' : '✗ OPEN'} · ${esc(i.issue_type)} · ${esc(i.layer)}
          </div>
          <div class="issue-location">${esc(i.location)}</div>
          <div class="issue-detail">${esc(i.detail)}</div>
        </div>`;
      }).join('') +
    '</div>';
  }

  $('pane-validation').innerHTML =
    '<div class="section-title">Initial Validation — ' + initIssues.length + ' issue(s) detected</div>' +
    cards(initIssues, true) +
    '<div class="section-title" style="margin-top:20px">Final State — ' +
      finalIssues.length + ' open · ' + resolved + ' resolved by repair</div>' +
    (finalIssues.length === 0
      ? '<div class="ok-card">✓ All issues resolved — final schema is clean</div>'
      : cards(finalIssues, false));
}

/* ── Repair tab ───────────────────────────────────────*/
function renderRepair(data){
  const tel     = (data.telemetry || []).filter(t => t.stage && t.stage.startsWith('repair'));
  const attempts = data.compiled_app.repair_attempts || 0;

  if(attempts === 0 || !tel.length){
    $('pane-repair').innerHTML =
      '<div class="ok-card" style="margin-top:4px">✓ No repairs needed — all schemas were valid on first generation.</div>';
    return;
  }

  // Group by attempt number
  const grouped = {};
  tel.forEach(t => {
    const k = t.attempt || 1;
    (grouped[k] = grouped[k] || []).push(t);
  });

  const initIssues  = data.stage_outputs?.initial_validation?.issues  || [];
  const finalIssues = data.stage_outputs?.final_validation?.issues || [];

  let html = '<div class="section-title">' + attempts + ' repair cycle(s) performed</div>';
  Object.entries(grouped).forEach(([attempt, calls]) => {
    html += `<div class="repair-card">
      <div class="repair-card-header">
        <span class="attempt-badge">Cycle ${attempt}</span>
        ${calls.map(c => `<span class="layer-badge">${esc(c.stage.replace('repair.',''))} layer</span>`).join('')}
      </div>
      ${calls.map(c => `<div class="repair-meta">
        Model: <span>${esc(c.model || '—')}</span> &nbsp;·&nbsp;
        Latency: <span>${c.latency_seconds}s</span> &nbsp;·&nbsp;
        Tokens: <span>${c.prompt_tokens || 0} in / ${c.completion_tokens || 0} out</span>
        ${c.error ? `&nbsp;·&nbsp;<span style="color:var(--red)">Error: ${esc(c.error)}</span>` : ''}
      </div>`).join('')}
    </div>`;
  });

  if(initIssues.length){
    html += '<div class="section-title">Issues targeted for repair</div><div class="issues-list">';
    initIssues.forEach(i => {
      const res = !finalIssues.find(f => f.location===i.location && f.issue_type===i.issue_type);
      html += `<div class="issue-card ${res ? 'resolved' : ''}">
        <div class="issue-type ${res ? 'resolved' : ''}">${res ? '✓ RESOLVED' : '✗ UNRESOLVED'} · ${esc(i.issue_type)} · ${esc(i.layer)}</div>
        <div class="issue-location">${esc(i.location)}</div>
        <div class="issue-detail">${esc(i.detail)}</div>
      </div>`;
    });
    html += '</div>';
  }
  $('pane-repair').innerHTML = html;
}

/* ── Cost & Quality tab ───────────────────────────────*/
function renderCost(data){
  const tel      = data.telemetry || [];
  const totalIn  = tel.reduce((s,t) => s + (t.prompt_tokens  || 0), 0);
  const totalOut = tel.reduce((s,t) => s + (t.completion_tokens || 0), 0);
  const cost     = data.estimated_cost_usd;
  const quality  = data.quality_score || 0;
  const qColor   = quality >= 80 ? '#22c55e' : quality >= 60 ? '#fbbf24' : '#f87171';

  const telCards = tel.map(t => `
    <div class="tel-card">
      <div class="tel-stage">${esc(t.stage)}</div>
      <div class="tel-model">${esc(t.model || '—')}</div>
      <div class="tel-val">⏱ ${t.latency_seconds}s</div>
      <div class="tel-tokens">${t.prompt_tokens || 0} in · ${t.completion_tokens || 0} out</div>
      ${t.cost_usd != null ? `<div class="tel-tokens" style="color:var(--accent2)">$${t.cost_usd.toFixed(6)}</div>` : ''}
    </div>`).join('');

  $('pane-cost').innerHTML = `
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Estimated LLM Cost</div>
        <div class="metric-value">${cost != null ? '$' + cost.toFixed(5) : 'N/A'}</div>
        <div class="metric-sub">llama-3.3-70b · $0.59/M in · $0.79/M out</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total Tokens Used</div>
        <div class="metric-value">${(totalIn + totalOut).toLocaleString()}</div>
        <div class="metric-sub">${totalIn.toLocaleString()} in · ${totalOut.toLocaleString()} out</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total Latency</div>
        <div class="metric-value">${data.total_latency_seconds}s</div>
        <div class="metric-sub">${tel.length} LLM calls across ${[...new Set(tel.map(t=>t.stage.split('.')[0]))].length} stages</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Output Quality Score</div>
        <div class="metric-value" style="color:${qColor}">${quality}<span style="font-size:1rem;font-weight:400">/100</span></div>
        <div class="quality-track"><div class="quality-fill" style="width:${quality}%;background:${qColor}"></div></div>
        <div class="metric-sub">Validation issues · repair cycles · runtime pass rate</div>
      </div>
    </div>
    
    <div class="section-title">Telemetry Visualizer</div>
    <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px; margin-bottom:20px; max-height:400px; display:flex; justify-content:center;">
      <canvas id="telemetryChart"></canvas>
    </div>

    <div class="section-title">Per-Stage Breakdown</div>
    <div class="tel-grid">${telCards}</div>
    <div class="section-title" style="margin-top:18px">Cost vs Quality Tradeoffs</div>
    <div class="tradeoffs">
      <div class="tradeoff-row"><div class="tradeoff-k">Model selection</div><div class="tradeoff-v">Strong model (llama-3.3-70b) used for all stages — higher quality, higher cost than 8B</div></div>
      <div class="tradeoff-row"><div class="tradeoff-k">Generation order</div><div class="tradeoff-v">Sequential DB→API→UI→Auth reduces cross-layer drift vs parallel; adds latency</div></div>
      <div class="tradeoff-row"><div class="tradeoff-k">Targeted repair</div><div class="tradeoff-v">Only broken layers are regenerated — cheaper and more stable than full retry</div></div>
      <div class="tradeoff-row"><div class="tradeoff-k">Temperature</div><div class="tradeoff-v">0.1–0.2 minimises hallucination at small creativity cost; better for schemas</div></div>
      <div class="tradeoff-row"><div class="tradeoff-k">Infrastructure</div><div class="tradeoff-v">Groq free tier — ~10× faster than OpenAI at comparable quality for JSON tasks</div></div>
    </div>`;
    
  setTimeout(() => {
    const ctx = document.getElementById('telemetryChart');
    if(!ctx) return;
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: tel.map(t => t.stage.substring(0, 15)),
        datasets: [{
          label: 'Latency (s)',
          data: tel.map(t => t.latency_seconds),
          backgroundColor: '#7c3aed',
          yAxisID: 'y'
        }, {
          label: 'Total Tokens',
          data: tel.map(t => (t.prompt_tokens || 0) + (t.completion_tokens || 0)),
          backgroundColor: '#3b82f6',
          yAxisID: 'y1'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#f8fafc' } } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { type: 'linear', display: true, position: 'left', title: {display: true, text: 'Latency (s)', color: '#7c3aed'}, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y1: { type: 'linear', display: true, position: 'right', title: {display: true, text: 'Tokens', color: '#3b82f6'}, grid: { drawOnChartArea: false }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }, 100);
}

/* ── Chat & Versioning ────────────────────────────────*/
function refineApp() {
  const refineText = $('refine-input').value.trim();
  if(!refineText) return;
  
  // Combine base prompt with refinement
  const newPrompt = _basePrompt + "\\n\\nRefinement: " + refineText;
  $('prompt').value = newPrompt;
  compile();
}

function updateVersionDropdown() {
  const sel = $('version-select');
  if(_appHistory.length > 1) {
    sel.classList.remove('hidden');
    sel.innerHTML = _appHistory.map((h, i) => `<option value="${i}" ${i === _currentVersion ? 'selected' : ''}>v${i+1} (${h.prompt.substring(0, 15)}...)</option>`).join('');
  } else {
    sel.classList.add('hidden');
  }
}

function restoreVersion(idx) {
  idx = parseInt(idx);
  if(isNaN(idx) || !_appHistory[idx]) return;
  _currentVersion = idx;
  _basePrompt = _appHistory[idx].prompt;
  $('prompt').value = _basePrompt;
  renderAll(_appHistory[idx].data);
}

/* ── Keyboard shortcut ────────────────────────────────*/
document.addEventListener('keydown', e => {
  if((e.ctrlKey || e.metaKey) && e.key === 'Enter') compile();
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"success": False, "error": "prompt is required"}, status_code=400)

    result = run_pipeline(prompt)

    response = {
        "success":               result.success,
        "error":                 result.error,
        "needs_clarification":   result.needs_clarification,
        "clarification_question": result.clarification_question,
        "total_latency_seconds": result.total_latency_seconds,
        "telemetry":             result.telemetry,
        "stage_outputs":         result.stage_outputs,
        "compiled_app":          result.compiled_app.model_dump() if result.compiled_app else None,
        "runtime_proof":         None,
        "estimated_cost_usd":    None,
        "quality_score":         None,
    }

    if result.success and result.compiled_app:
        try:
            response["runtime_proof"] = run_smoke_test(result.compiled_app)
        except Exception as e:
            response["runtime_proof"] = {
                "error": str(e), "passed": 0, "failed": 0,
                "total_checks": 0, "pass_rate": 0, "details": [],
                "verdict": f"Smoke test error: {e}"
            }
        response["estimated_cost_usd"] = _compute_cost(result.telemetry)
        response["quality_score"]      = _compute_quality(response)

    return JSONResponse(response)

@app.post("/export")
async def export_project(request: Request):
    data = await request.json()
    files = data.get("files", {})
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_path, content in files.items():
            zip_file.writestr(file_path, content)
            
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=generated-app.zip"}
    )

