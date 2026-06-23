"""
eval/run_eval.py — Professional benchmark runner for the nl2app AI App Compiler.

Outputs (all written to eval/):
  metrics.json          — machine-readable full results + chart data
  metrics.csv           — flat per-prompt rows, importable into Excel / pandas
  evaluation_report.md  — human-readable report with tables, charts, failure analysis

Usage:
  python3 eval/run_eval.py              # run all 20 prompts
  python3 eval/run_eval.py --limit 5   # run first N prompts (smoke test)
  python3 eval/run_eval.py --dry-run   # validate dataset + wiring only (no LLM calls)
  python3 eval/run_eval.py --resume    # skip prompts that already have rows in metrics.json
"""

import csv
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from collections import Counter, defaultdict

# ── path bootstrap so we can run from any cwd ─────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orchestrator import run_pipeline
from runtime_sim import run_smoke_test

# ── output paths ──────────────────────────────────────────────────────────────
DATASET_PATH  = os.path.join(EVAL, "dataset.json")
METRICS_JSON  = os.path.join(EVAL, "metrics.json")
METRICS_CSV   = os.path.join(EVAL, "metrics.csv")
REPORT_MD     = os.path.join(EVAL, "evaluation_report.md")

# ── pricing: Groq llama-3.3-70b-versatile ────────────────────────────────────
PRICE_IN  = 0.59 / 1_000_000   # $ per input token
PRICE_OUT = 0.79 / 1_000_000   # $ per output token
MODEL     = "llama-3.3-70b-versatile"

CSV_FIELDS = [
    "id", "category", "domain", "prompt_preview",
    "success", "needs_clarification",
    "validation_failures_initial", "validation_failures_final", "issues_resolved",
    "repair_count", "retry_count",
    "total_latency_seconds", "avg_latency_per_call", "llm_calls",
    "total_tokens_in", "total_tokens_out", "total_tokens",
    "estimated_cost_usd",
    "execution_success", "smoke_pass_rate", "smoke_total", "smoke_passed", "smoke_failed",
    "quality_score",
    "error_type", "error_message",
    "run_timestamp",
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost(telemetry: list) -> float:
    return round(
        sum((t.get("prompt_tokens") or 0) * PRICE_IN +
            (t.get("completion_tokens") or 0) * PRICE_OUT
            for t in telemetry),
        6,
    )


def compute_quality(result, smoke: dict | None) -> int:
    """
    Quality score 0–100:
      Start at 100
      -8 per unresolved validation issue (final)
      -5 per repair cycle
      × smoke pass_rate  (runtime execution quality)
    """
    score = 100
    final_issues = []
    if result.stage_outputs and "final_validation" in result.stage_outputs:
        final_issues = result.stage_outputs["final_validation"].get("issues", [])
    score -= len(final_issues) * 8
    if result.compiled_app:
        score -= result.compiled_app.repair_attempts * 5
    if smoke and smoke.get("pass_rate") is not None:
        score = int(score * smoke["pass_rate"])
    return max(0, min(100, score))


# ─────────────────────────────────────────────────────────────────────────────
# Single prompt runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(item: dict) -> dict:
    """Run a single dataset item through pipeline + smoke test; return a metrics row."""
    pid      = item["id"]
    category = item["category"]
    domain   = item.get("domain", "")
    prompt   = item["prompt"]
    ts       = datetime.now(timezone.utc).isoformat()

    result = run_pipeline(prompt)

    # ── smoke test ──────────────────────────────────────────────────────────
    smoke             = None
    execution_success = False
    smoke_pass_rate   = None
    smoke_total = smoke_passed = smoke_failed = 0

    if result.success and result.compiled_app:
        try:
            smoke           = run_smoke_test(result.compiled_app)
            execution_success = smoke.get("failed", 1) == 0
            smoke_pass_rate   = smoke.get("pass_rate")
            smoke_total       = smoke.get("total_checks", 0)
            smoke_passed      = smoke.get("passed", 0)
            smoke_failed      = smoke.get("failed", 0)
        except Exception as exc:
            smoke = {"error": str(exc)}

    # ── token / cost / latency ───────────────────────────────────────────────
    tel        = result.telemetry or []
    tokens_in  = sum(t.get("prompt_tokens")    or 0 for t in tel)
    tokens_out = sum(t.get("completion_tokens") or 0 for t in tel)
    llm_calls  = len(tel)
    cost       = compute_cost(tel)
    quality    = compute_quality(result, smoke) if result.success else 0
    avg_lat    = round(result.total_latency_seconds / llm_calls, 2) if llm_calls else 0

    repair_count = result.compiled_app.repair_attempts if result.compiled_app else 0
    retry_count  = sum(1 for t in tel if t.get("stage", "").startswith("repair"))

    init_issues = final_issues = 0
    if result.stage_outputs:
        init_issues  = len((result.stage_outputs.get("initial_validation") or {}).get("issues", []))
        final_issues = len((result.stage_outputs.get("final_validation")   or {}).get("issues", []))

    error_type = error_msg = ""
    if result.error:
        parts      = result.error.split(":", 1)
        error_type = parts[0].strip()
        error_msg  = (parts[1].strip()[:300] if len(parts) > 1 else result.error[:300])

    return {
        # identity
        "id": pid, "category": category, "domain": domain,
        "prompt_preview": prompt[:100],
        # outcome flags
        "success": result.success,
        "needs_clarification": result.needs_clarification,
        # validation
        "validation_failures_initial": init_issues,
        "validation_failures_final":   final_issues,
        "issues_resolved": max(0, init_issues - final_issues),
        # repair
        "repair_count": repair_count,
        "retry_count":  retry_count,
        # latency
        "total_latency_seconds": result.total_latency_seconds,
        "avg_latency_per_call":  avg_lat,
        "llm_calls":             llm_calls,
        # tokens
        "total_tokens_in":  tokens_in,
        "total_tokens_out": tokens_out,
        "total_tokens":     tokens_in + tokens_out,
        # cost
        "estimated_cost_usd": cost,
        # runtime
        "execution_success": execution_success,
        "smoke_pass_rate":   round(smoke_pass_rate, 3) if smoke_pass_rate is not None else None,
        "smoke_total":       smoke_total,
        "smoke_passed":      smoke_passed,
        "smoke_failed":      smoke_failed,
        # quality
        "quality_score": quality,
        # error
        "error_type":    error_type,
        "error_message": error_msg,
        # meta
        "run_timestamp": ts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats over a list of rows
# ─────────────────────────────────────────────────────────────────────────────

def _safe_avg(vals):
    filtered = [v for v in vals if v is not None]
    return round(sum(filtered) / len(filtered), 3) if filtered else None


def aggregate(rows: list) -> dict:
    n = len(rows)
    if not n:
        return {}

    successes      = [r for r in rows if r["success"]]
    clarifications = [r for r in rows if r["needs_clarification"]]
    hard_failures  = [r for r in rows if not r["success"] and not r["needs_clarification"]]
    exec_success   = [r for r in rows if r["execution_success"]]
    repaired       = [r for r in rows if r["repair_count"] > 0]

    return {
        "n": n,
        # rates (%)
        "success_rate":       round(len(successes) / n * 100, 1),
        "clarification_rate": round(len(clarifications) / n * 100, 1),
        "failure_rate":       round(len(hard_failures) / n * 100, 1),
        "exec_success_rate":  round(len(exec_success) / n * 100, 1),
        "repair_rate":        round(len(repaired) / n * 100, 1),
        # counts
        "total_successes":      len(successes),
        "total_clarifications": len(clarifications),
        "total_failures":       len(hard_failures),
        "total_exec_success":   len(exec_success),
        # latency
        "avg_latency":    round(sum(r["total_latency_seconds"] for r in rows) / n, 2),
        "min_latency":    min(r["total_latency_seconds"] for r in rows),
        "max_latency":    max(r["total_latency_seconds"] for r in rows),
        "total_latency":  round(sum(r["total_latency_seconds"] for r in rows), 2),
        # repair / validation
        "avg_repairs":       round(sum(r["repair_count"] for r in rows) / n, 2),
        "avg_retries":       round(sum(r["retry_count"] for r in rows) / n, 2),
        "avg_val_initial":   round(sum(r["validation_failures_initial"] for r in rows) / n, 2),
        "avg_val_final":     round(sum(r["validation_failures_final"] for r in rows) / n, 2),
        "avg_issues_resolved": round(sum(r["issues_resolved"] for r in rows) / n, 2),
        "validation_failure_rate": round(
            sum(1 for r in rows if r["validation_failures_initial"] > 0) / n * 100, 1
        ),
        # tokens / cost
        "total_tokens_in":  sum(r["total_tokens_in"] for r in rows),
        "total_tokens_out": sum(r["total_tokens_out"] for r in rows),
        "total_tokens":     sum(r["total_tokens"] for r in rows),
        "total_cost_usd":   round(sum(r["estimated_cost_usd"] for r in rows), 5),
        "avg_cost_usd":     round(sum(r["estimated_cost_usd"] for r in rows) / n, 6),
        "avg_llm_calls":    round(sum(r["llm_calls"] for r in rows) / n, 1),
        # quality
        "avg_quality": round(sum(r["quality_score"] for r in rows) / n, 1),
        "min_quality": min(r["quality_score"] for r in rows),
        "max_quality": max(r["quality_score"] for r in rows),
        # smoke
        "avg_smoke_pass_rate": _safe_avg([r["smoke_pass_rate"] for r in rows]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Failure analysis
# ─────────────────────────────────────────────────────────────────────────────

def failure_analysis(rows: list) -> dict:
    failed = [r for r in rows if not r["success"] and not r["needs_clarification"]]
    clarified = [r for r in rows if r["needs_clarification"]]

    error_counter   = Counter(r["error_type"] for r in failed if r["error_type"])
    domain_failures = Counter(r["domain"] for r in failed)
    cat_failures    = Counter(r["category"] for r in failed)

    # which layers had the most initial validation issues?
    # (we only have aggregate counts per row, not per-layer, so we note repair rate by domain)
    high_repair_domains = sorted(
        {r["domain"]: r["repair_count"] for r in rows}.items(),
        key=lambda x: -x[1],
    )

    return {
        "total_failures":        len(failed),
        "total_clarifications":  len(clarified),
        "error_type_counts":     dict(error_counter.most_common()),
        "failures_by_domain":    dict(domain_failures.most_common()),
        "failures_by_category":  dict(cat_failures),
        "top_failure_categories": [
            {"type": k, "count": v, "pct": round(v / max(len(failed), 1) * 100, 1)}
            for k, v in error_counter.most_common(5)
        ],
        "high_repair_domains": [
            {"domain": d, "repair_count": c}
            for d, c in high_repair_domains[:5]
            if c > 0
        ],
        "clarification_domains": [r["domain"] for r in clarified],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart data (for programmatic consumption — also used to render ASCII bars)
# ─────────────────────────────────────────────────────────────────────────────

def build_chart_data(rows: list, all_stats: dict, real_stats: dict, edge_stats: dict) -> dict:
    # quality distribution buckets
    def quality_buckets(subset):
        buckets = {"90-100": 0, "70-89": 0, "50-69": 0, "0-49": 0}
        for r in subset:
            q = r["quality_score"]
            if q >= 90:   buckets["90-100"] += 1
            elif q >= 70: buckets["70-89"]  += 1
            elif q >= 50: buckets["50-69"]  += 1
            else:         buckets["0-49"]   += 1
        return buckets

    # latency distribution buckets
    def latency_buckets(subset):
        buckets = {"<15s": 0, "15-25s": 0, "25-40s": 0, ">40s": 0}
        for r in subset:
            lat = r["total_latency_seconds"]
            if lat < 15:   buckets["<15s"]   += 1
            elif lat < 25: buckets["15-25s"] += 1
            elif lat < 40: buckets["25-40s"] += 1
            else:          buckets[">40s"]   += 1
        return buckets

    # per-domain summary
    domains = sorted({r["domain"] for r in rows})
    domain_chart = []
    for d in domains:
        subset = [r for r in rows if r["domain"] == d]
        domain_chart.append({
            "domain":       d,
            "n":            len(subset),
            "success_rate": round(sum(1 for r in subset if r["success"]) / len(subset) * 100, 1),
            "avg_quality":  round(sum(r["quality_score"] for r in subset) / len(subset), 1),
            "avg_latency":  round(sum(r["total_latency_seconds"] for r in subset) / len(subset), 2),
            "avg_cost":     round(sum(r["estimated_cost_usd"] for r in subset) / len(subset), 6),
            "avg_repairs":  round(sum(r["repair_count"] for r in subset) / len(subset), 2),
        })

    # category comparison
    real_rows = [r for r in rows if r["category"] == "real"]
    edge_rows = [r for r in rows if r["category"] == "edge"]

    return {
        "category_comparison": {
            "labels":          ["Real-World", "Edge Cases"],
            "success_rate":    [real_stats.get("success_rate", 0),    edge_stats.get("success_rate", 0)],
            "avg_quality":     [real_stats.get("avg_quality", 0),     edge_stats.get("avg_quality", 0)],
            "avg_latency":     [real_stats.get("avg_latency", 0),     edge_stats.get("avg_latency", 0)],
            "exec_success_rate":[real_stats.get("exec_success_rate",0),edge_stats.get("exec_success_rate",0)],
        },
        "quality_distribution": {
            "all":  quality_buckets(rows),
            "real": quality_buckets(real_rows),
            "edge": quality_buckets(edge_rows),
        },
        "latency_distribution": {
            "all":  latency_buckets(rows),
            "real": latency_buckets(real_rows),
            "edge": latency_buckets(edge_rows),
        },
        "domain_breakdown": domain_chart,
        "metrics_over_prompts": [
            {
                "id":      r["id"],
                "domain":  r["domain"],
                "quality": r["quality_score"],
                "latency": r["total_latency_seconds"],
                "cost":    r["estimated_cost_usd"],
                "repairs": r["repair_count"],
                "success": r["success"],
            }
            for r in rows
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASCII bar chart renderer (for Markdown)
# ─────────────────────────────────────────────────────────────────────────────

def ascii_bar(value: float, max_val: float, width: int = 28, fill: str = "█") -> str:
    if max_val == 0:
        return " " * width
    filled = round(value / max_val * width)
    return fill * filled + "░" * (width - filled)


def ascii_bar_chart(title: str, data: dict, unit: str = "", max_val: float | None = None) -> str:
    if not data:
        return ""
    mv = max_val if max_val is not None else max(data.values(), default=1)
    lines = [f"```", f"{title}"]
    for label, val in data.items():
        bar  = ascii_bar(val, mv)
        lines.append(f"  {label:<14} {bar}  {val}{unit}")
    lines.append("```")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Report generators
# ─────────────────────────────────────────────────────────────────────────────

def _pct_bar(pct: float) -> str:
    """Inline % bar for markdown tables."""
    filled = round(pct / 10)
    return "▓" * filled + "░" * (10 - filled) + f" {pct}%"


def build_report(
    rows:        list,
    all_s:       dict,
    real_s:      dict,
    edge_s:      dict,
    fail_a:      dict,
    chart_data:  dict,
    run_at:      str,
    elapsed_sec: float,
) -> str:
    nl = "\n"

    # ── helper: summary metric table ──────────────────────────────────────
    def metric_table(s: dict) -> str:
        if not s:
            return "_No data_"
        return (
            "| Metric | Value | Bar |\n"
            "|--------|-------|-----|\n"
            f"| **Success Rate** | {s['success_rate']}% | {_pct_bar(s['success_rate'])} |\n"
            f"| **Execution Success Rate** | {s['exec_success_rate']}% | {_pct_bar(s['exec_success_rate'])} |\n"
            f"| **Validation Failure Rate** | {s['validation_failure_rate']}% | {_pct_bar(s['validation_failure_rate'])} |\n"
            f"| **Repair Rate** | {s['repair_rate']}% | {_pct_bar(s['repair_rate'])} |\n"
            f"| **Avg Quality Score** | {s['avg_quality']}/100 | {_pct_bar(s['avg_quality'])} |\n"
            f"| Clarification Rate | {s['clarification_rate']}% | |\n"
            f"| Failure Rate | {s['failure_rate']}% | |\n"
            f"| Avg Latency | {s['avg_latency']}s | |\n"
            f"| Avg Repair Cycles | {s['avg_repairs']} | |\n"
            f"| Avg Retry Count | {s['avg_retries']} | |\n"
            f"| Avg Validation Issues (initial) | {s['avg_val_initial']} | |\n"
            f"| Avg Validation Issues (final) | {s['avg_val_final']} | |\n"
            f"| Avg Issues Resolved by Repair | {s['avg_issues_resolved']} | |\n"
            f"| Avg LLM Calls | {s['avg_llm_calls']} | |\n"
            f"| Avg Cost per Request | ${s['avg_cost_usd']:.6f} | |\n"
            f"| Total LLM Cost | ${s['total_cost_usd']:.5f} | |\n"
            f"| Avg Smoke Pass Rate | {(str(round(s['avg_smoke_pass_rate']*100,1))+'%') if s.get('avg_smoke_pass_rate') is not None else 'N/A'} | |"
        )

    # ── quality distribution chart ────────────────────────────────────────
    qd = chart_data["quality_distribution"]["all"]
    quality_chart = ascii_bar_chart(
        "Quality Score Distribution (all prompts)",
        qd, " prompts", max_val=max(qd.values(), default=1)
    )

    ld = chart_data["latency_distribution"]["all"]
    latency_chart = ascii_bar_chart(
        "Latency Distribution (all prompts)",
        ld, " prompts", max_val=max(ld.values(), default=1)
    )

    # ── domain breakdown table ────────────────────────────────────────────
    dom_rows = chart_data["domain_breakdown"]
    dom_header = "| Domain | N | Success | Quality | Latency | Cost | Repairs |\n|--------|---|---------|---------|---------|------|---------|"
    dom_body = nl.join(
        f"| {d['domain']} | {d['n']} | {d['success_rate']}% "
        f"| {d['avg_quality']}/100 | {d['avg_latency']}s "
        f"| ${d['avg_cost']:.6f} | {d['avg_repairs']} |"
        for d in dom_rows
    )

    # ── category comparison ───────────────────────────────────────────────
    cc = chart_data["category_comparison"]
    cat_chart = ascii_bar_chart(
        "Success Rate by Category (%)",
        dict(zip(cc["labels"], cc["success_rate"])),
        "%", 100
    )
    qual_cat_chart = ascii_bar_chart(
        "Avg Quality Score by Category",
        dict(zip(cc["labels"], cc["avg_quality"])),
        "/100", 100
    )

    # ── per-prompt results table ──────────────────────────────────────────
    def result_icon(r):
        if r["success"]:              return "✓"
        if r["needs_clarification"]:  return "?"
        return "✗"

    prompt_rows = nl.join(
        f"| `{r['id']}` | {r['domain']} | {r['category']} "
        f"| {result_icon(r)} "
        f"| {r['repair_count']} | {r['validation_failures_initial']}→{r['validation_failures_final']} "
        f"| {r['total_latency_seconds']}s "
        f"| {r['quality_score']}/100 "
        f"| ${r['estimated_cost_usd']:.5f} "
        f"| {r['error_type'] or '—'} |"
        for r in rows
    )

    # ── top failure categories ────────────────────────────────────────────
    top_fails = fail_a["top_failure_categories"]
    fail_rows = (
        nl.join(f"| `{f['type']}` | {f['count']} | {f['pct']}% |" for f in top_fails)
        if top_fails else "| — | 0 | 0% |"
    )

    high_repair = fail_a.get("high_repair_domains", [])
    repair_rows = (
        nl.join(f"| {d['domain']} | {d['repair_count']} |" for d in high_repair)
        if high_repair else "| — | 0 |"
    )

    # ── failure domain breakdown ──────────────────────────────────────────
    fail_domain_chart_data = fail_a["failures_by_domain"]
    if fail_domain_chart_data:
        fail_domain_chart = ascii_bar_chart(
            "Failures by Domain",
            fail_domain_chart_data, " failures",
            max_val=max(fail_domain_chart_data.values(), default=1)
        )
    else:
        fail_domain_chart = "_No failures recorded._"

    # ── executive summary badges ──────────────────────────────────────────
    overall_grade = (
        "🟢 EXCELLENT" if all_s["success_rate"] >= 85 else
        "🟡 GOOD"      if all_s["success_rate"] >= 70 else
        "🟠 FAIR"      if all_s["success_rate"] >= 50 else
        "🔴 POOR"
    )

    return f"""# nl2app Compiler — Benchmark Evaluation Report

> **Generated:** {run_at}  
> **Model:** {MODEL}  
> **Dataset:** {all_s['n']} prompts ({real_s.get('n',0)} real-world · {edge_s.get('n',0)} edge cases)  
> **Total Runtime:** {elapsed_sec:.1f}s &nbsp;|&nbsp; **Total Cost:** ${all_s['total_cost_usd']:.5f}  
> **Overall Grade:** {overall_grade}

---

## 1 · Executive Summary

| Metric | All | Real-World | Edge Cases |
|--------|-----|------------|------------|
| **Success Rate** | {all_s['success_rate']}% | {real_s.get('success_rate','—')}% | {edge_s.get('success_rate','—')}% |
| **Execution Success Rate** | {all_s['exec_success_rate']}% | {real_s.get('exec_success_rate','—')}% | {edge_s.get('exec_success_rate','—')}% |
| **Validation Failure Rate** | {all_s['validation_failure_rate']}% | {real_s.get('validation_failure_rate','—')}% | {edge_s.get('validation_failure_rate','—')}% |
| **Repair Rate** | {all_s['repair_rate']}% | {real_s.get('repair_rate','—')}% | {edge_s.get('repair_rate','—')}% |
| **Avg Quality Score** | {all_s['avg_quality']}/100 | {real_s.get('avg_quality','—')}/100 | {edge_s.get('avg_quality','—')}/100 |
| **Avg Latency** | {all_s['avg_latency']}s | {real_s.get('avg_latency','—')}s | {edge_s.get('avg_latency','—')}s |
| **Total LLM Cost** | ${all_s['total_cost_usd']:.5f} | ${real_s.get('total_cost_usd',0):.5f} | ${edge_s.get('total_cost_usd',0):.5f} |
| Clarification Rate | {all_s['clarification_rate']}% | {real_s.get('clarification_rate','—')}% | {edge_s.get('clarification_rate','—')}% |
| Failure Rate | {all_s['failure_rate']}% | {real_s.get('failure_rate','—')}% | {edge_s.get('failure_rate','—')}% |
| Avg Repair Cycles | {all_s['avg_repairs']} | {real_s.get('avg_repairs','—')} | {edge_s.get('avg_repairs','—')} |

---

## 2 · Detailed Metrics

### 2.1 All Prompts ({all_s['n']} total)

{metric_table(all_s)}

### 2.2 Real-World Prompts ({real_s.get('n',0)} prompts)

{metric_table(real_s)}

### 2.3 Edge Case Prompts ({edge_s.get('n',0)} prompts)

{metric_table(edge_s)}

---

## 3 · Charts & Distributions

### 3.1 Quality Score Distribution

{quality_chart}

### 3.2 Latency Distribution

{latency_chart}

### 3.3 Success Rate vs Quality by Category

{cat_chart}

{qual_cat_chart}

---

## 4 · Domain Breakdown

{dom_header}
{dom_body}

---

## 5 · Failure Analysis

> **{fail_a['total_failures']} hard failures** · **{fail_a['total_clarifications']} clarification requests** out of {all_s['n']} total prompts.

### 5.1 Top Failure Categories

| Error Type | Count | % of Failures |
|------------|-------|---------------|
{fail_rows}

### 5.2 Failures by Domain

{fail_domain_chart}

### 5.3 Highest Repair Frequency by Domain

| Domain | Avg Repair Cycles |
|--------|-------------------|
{repair_rows}

### 5.4 Clarification Requests

Prompts that triggered clarification (too vague or conflicting to proceed with assumptions):

{nl.join(f"- `{d}`" for d in fail_a['clarification_domains']) or "- _None_"}

---

## 6 · Cost Analysis

| Metric | Value |
|--------|-------|
| Model | {MODEL} |
| Input price | $0.59 / 1M tokens |
| Output price | $0.79 / 1M tokens |
| Total input tokens | {all_s['total_tokens_in']:,} |
| Total output tokens | {all_s['total_tokens_out']:,} |
| Total tokens | {all_s['total_tokens']:,} |
| Total cost | ${all_s['total_cost_usd']:.5f} |
| Avg cost per request | ${all_s['avg_cost_usd']:.6f} |
| Avg LLM calls per request | {all_s['avg_llm_calls']} |

**Cost vs Quality tradeoffs:**

| Choice | Impact |
|--------|--------|
| Strong model (70b) for all stages | Higher quality · higher cost than 8B model |
| Sequential DB→API→UI→Auth generation | Reduces cross-layer drift · adds latency |
| Targeted repair (broken layers only) | Cheaper & more stable than full retry |
| Temperature 0.1–0.2 | Reduces hallucination · slight creativity cost |
| Groq free tier | ~10× faster than OpenAI at comparable structured-output quality |

---

## 7 · Per-Prompt Results

| ID | Domain | Category | Result | Repairs | Issues (init→final) | Latency | Quality | Cost | Error |
|----|--------|----------|--------|---------|----------------------|---------|---------|------|-------|
{prompt_rows}

**Legend:** ✓ Success · ? Clarification needed · ✗ Hard failure

---

## 8 · Recommendations

{_build_recommendations(all_s, real_s, edge_s, fail_a)}

---

*Report generated by `eval/run_eval.py` · nl2app AI App Compiler v1.0.0*
"""


def _build_recommendations(all_s, real_s, edge_s, fail_a) -> str:
    recs = []

    if all_s["success_rate"] < 80:
        recs.append("- **Improve prompt handling**: Success rate is below 80%. Consider stricter intent extraction fallbacks.")
    if all_s["validation_failure_rate"] > 50:
        recs.append("- **Tighten schema prompts**: Validation failure rate is high. Add more explicit constraint instructions to schema-gen prompts.")
    if all_s["repair_rate"] > 40:
        recs.append("- **Reduce repair dependency**: More than 40% of requests needed repair. Improve initial generation prompts to reduce cross-layer drift.")
    if all_s["exec_success_rate"] < 70:
        recs.append("- **Runtime stability**: Execution success rate is below 70%. Review smoke test failures for systematic patterns.")
    if edge_s.get("success_rate", 100) < real_s.get("success_rate", 100) - 20:
        recs.append("- **Edge case handling**: Edge prompts fail significantly more than real-world prompts. Improve ambiguity detection and assumption logic.")
    if all_s["avg_latency"] > 40:
        recs.append("- **Latency optimisation**: Avg latency is above 40s. Consider parallelising independent schema generation stages.")
    if all_s["avg_cost_usd"] > 0.01:
        recs.append("- **Cost reduction**: Avg cost per request is above $0.01. Consider using a smaller model for non-critical stages (UI, auth).")
    if fail_a["total_clarifications"] > 3:
        recs.append("- **Clarification triggers**: Many prompts triggered clarification. Fine-tune the vagueness threshold or default assumptions.")

    if not recs:
        recs.append("- ✓ System is performing well across all key metrics. No critical improvements needed.")

    return "\n".join(recs)


# ─────────────────────────────────────────────────────────────────────────────
# Write outputs
# ─────────────────────────────────────────────────────────────────────────────

def write_metrics_json(rows, all_s, real_s, edge_s, fail_a, chart_data, run_at, elapsed):
    payload = {
        "meta": {
            "generated_at":   run_at,
            "total_runtime_seconds": elapsed,
            "model":          MODEL,
            "price_in_per_token":  PRICE_IN,
            "price_out_per_token": PRICE_OUT,
            "n_prompts":      len(rows),
            "dataset_path":   DATASET_PATH,
        },
        "summary": {
            "all":  all_s,
            "real": real_s,
            "edge": edge_s,
        },
        "failure_analysis": fail_a,
        "chart_data":  chart_data,
        "per_prompt":  rows,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def write_metrics_csv(rows):
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
AMBER = "\033[93m"
CYAN  = "\033[96m"
DIM   = "\033[2m"
RESET = "\033[0m"

def _icon(r):
    if r["success"]:             return f"{GREEN}✓{RESET}"
    if r["needs_clarification"]: return f"{AMBER}?{RESET}"
    return f"{RED}✗{RESET}"

def _progress_bar(done, total, width=20):
    filled = round(done / total * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(limit: int | None = None, dry_run: bool = False, resume: bool = False):
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    # ── resume: load already-completed rows ─────────────────────────────────
    existing_ids: set = set()
    rows: list = []
    if resume and os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON) as f:
                existing = json.load(f)
            rows = existing.get("per_prompt", [])
            existing_ids = {r["id"] for r in rows}
            print(f"{CYAN}Resume:{RESET} skipping {len(existing_ids)} already-completed prompt(s).")
        except Exception:
            rows = []

    total    = len(dataset)
    todo     = [item for item in dataset if item["id"] not in existing_ids]
    run_start = time.time()

    print(f"\n{BOLD}nl2app Benchmark Eval{RESET} · {total} prompts · {len(todo)} to run\n" + "─" * 65)

    if dry_run:
        print(f"{AMBER}DRY RUN:{RESET} dataset loaded ({total} items), pipeline wiring OK.")
        return

    for i, item in enumerate(todo, 1):
        done_so_far = len(existing_ids) + i
        bar  = _progress_bar(done_so_far, total)
        print(f"\n[{done_so_far:02d}/{total}] {bar}  {CYAN}{item['id']}{RESET} ({item.get('domain','')})")
        print(f"  {DIM}{item['prompt'][:72]}…{RESET}")

        t0  = time.time()
        row = run_one(item)
        elapsed_one = time.time() - t0

        icon = _icon(row)
        q    = row["quality_score"]
        qc   = GREEN if q >= 80 else AMBER if q >= 60 else RED
        print(
            f"  {icon}  latency={row['total_latency_seconds']}s  "
            f"repairs={row['repair_count']}  "
            f"quality={qc}{q}{RESET}/100  "
            f"cost=${row['estimated_cost_usd']:.5f}"
            + (f"  {RED}err={row['error_type']}{RESET}" if row["error_type"] else "")
        )
        rows.append(row)

        # ── write partial metrics.json after every prompt (resume support) ──
        partial_elapsed = time.time() - run_start
        partial_all  = aggregate(rows)
        partial_real = aggregate([r for r in rows if r["category"] == "real"])
        partial_edge = aggregate([r for r in rows if r["category"] == "edge"])
        partial_fail = failure_analysis(rows)
        partial_chart = build_chart_data(rows, partial_all, partial_real, partial_edge)
        run_at_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        write_metrics_json(rows, partial_all, partial_real, partial_edge,
                           partial_fail, partial_chart, run_at_str, partial_elapsed)

    # ── final aggregation ────────────────────────────────────────────────────
    total_elapsed = time.time() - run_start
    run_at_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    all_s  = aggregate(rows)
    real_s = aggregate([r for r in rows if r["category"] == "real"])
    edge_s = aggregate([r for r in rows if r["category"] == "edge"])
    fail_a = failure_analysis(rows)
    chart_data = build_chart_data(rows, all_s, real_s, edge_s)

    # ── write all three outputs ───────────────────────────────────────────────
    write_metrics_json(rows, all_s, real_s, edge_s, fail_a, chart_data, run_at_str, total_elapsed)
    write_metrics_csv(rows)

    report_md = build_report(rows, all_s, real_s, edge_s, fail_a, chart_data, run_at_str, total_elapsed)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{BOLD}Results written:{RESET}")
    print(f"  {GREEN}✓{RESET}  {METRICS_JSON}")
    print(f"  {GREEN}✓{RESET}  {METRICS_CSV}")
    print(f"  {GREEN}✓{RESET}  {REPORT_MD}")
    print()
    print(f"{BOLD}Overall summary:{RESET}")
    print(f"  Success rate       {GREEN if all_s['success_rate']>=80 else AMBER}{all_s['success_rate']}%{RESET}")
    print(f"  Exec success rate  {GREEN if all_s['exec_success_rate']>=80 else AMBER}{all_s['exec_success_rate']}%{RESET}")
    print(f"  Avg quality score  {GREEN if all_s['avg_quality']>=80 else AMBER}{all_s['avg_quality']}{RESET}/100")
    print(f"  Avg latency        {all_s['avg_latency']}s")
    print(f"  Total LLM cost     ${all_s['total_cost_usd']:.5f}")
    print(f"  Total runtime      {total_elapsed:.1f}s")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nl2app benchmark eval runner")
    parser.add_argument("--limit",   type=int,  default=None, help="Only run first N prompts")
    parser.add_argument("--dry-run", action="store_true",     help="Validate setup only, no LLM calls")
    parser.add_argument("--resume",  action="store_true",     help="Skip prompts already in metrics.json")
    args = parser.parse_args()

    run_eval(limit=args.limit, dry_run=args.dry_run, resume=args.resume)
