"""
orchestrator.py — Wires the 5-stage compiler pipeline together end-to-end.

Pipeline:
  1. intent.extract_intent          (raw text -> IntentSpec)
  2. architecture.design_architecture (IntentSpec -> ArchitectureSpec)
  3. schema_gen.generate_all_schemas  (ArchitectureSpec -> SchemaBundle)
  4. validator.validate_bundle + repair.repair_bundle  (detect + targeted fix)
  5. refine.refine_bundle             (deterministic polish + documented notes)

Every stage's telemetry (latency, tokens, model) is collected into a single
list, which the eval framework consumes to compute real metrics.

This file is intentionally the ONLY place that knows the stage ORDER —
each individual pipeline/*.py module has no knowledge of what comes before
or after it. That separation is what makes this a "compiler with stages"
rather than one big procedural script.
"""

import time
from models import CompiledApp, IntentSpec
from pipeline.intent import extract_intent
from pipeline.architecture import design_architecture
from pipeline.schema_gen import generate_all_schemas
from pipeline.validator import validate_bundle
from pipeline.repair import repair_bundle
from pipeline.refine import refine_bundle


class PipelineResult:
    def __init__(self):
        self.compiled_app: CompiledApp | None = None
        self.telemetry: list[dict] = []
        self.total_latency_seconds: float = 0.0
        self.success: bool = False
        self.error: str | None = None
        self.needs_clarification: bool = False
        self.clarification_question: str | None = None
        self.stage_outputs: dict = {}  # raw per-stage JSON, for the UI to display


def run_pipeline(user_prompt: str) -> PipelineResult:
    result = PipelineResult()
    pipeline_start = time.time()

    try:
        # ---- STAGE 1: Intent Extraction ------------------------------------
        intent, t1 = extract_intent(user_prompt)
        result.telemetry.append({"stage": "1_intent_extraction", **t1})
        result.stage_outputs["intent"] = intent.model_dump()

        if intent.requires_clarification:
            result.needs_clarification = True
            result.clarification_question = intent.clarification_question
            result.total_latency_seconds = round(time.time() - pipeline_start, 3)
            return result  # stop here — failure handling: ask for clarification

        # ---- STAGE 2: System Design Layer ----------------------------------
        architecture, t2 = design_architecture(intent)
        result.telemetry.append({"stage": "2_architecture", **t2})
        result.stage_outputs["architecture"] = architecture.model_dump()

        # ---- STAGE 3: Schema Generation -------------------------------------
        bundle, t3_log = generate_all_schemas(architecture)
        result.telemetry.extend(t3_log)
        result.stage_outputs["schema_bundle_raw"] = bundle.model_dump()

        # ---- STAGE 4: Validation + Targeted Repair ---------------------------
        initial_report = validate_bundle(architecture, bundle)
        result.stage_outputs["initial_validation"] = initial_report.model_dump()

        repaired_bundle, repair_attempts, repair_telemetry = repair_bundle(
            architecture, bundle, initial_report
        )
        result.telemetry.extend(repair_telemetry)

        final_report = validate_bundle(architecture, repaired_bundle)
        result.stage_outputs["final_validation"] = final_report.model_dump()

        # ---- STAGE 5: Refinement ----------------------------------------------
        refined_bundle, notes = refine_bundle(architecture, repaired_bundle, final_report)
        result.stage_outputs["refinement_notes"] = notes

        # ---- Assemble final compiled output ------------------------------------
        assumptions = [f"{a.field}: {a.assumption_made}" for a in intent.ambiguities] + notes
        
        from runtime import build_runtime_app
        runtime_app = build_runtime_app(refined_bundle)
        result.stage_outputs["runtime"] = runtime_app.model_dump()

        from generators import generate_project_files
        virtual_files = generate_project_files(runtime_app)
        result.stage_outputs["files"] = virtual_files

        compiled = CompiledApp(
            intent=intent,
            architecture=architecture,
            schema_bundle=refined_bundle,
            repair_attempts=repair_attempts,
            final_validation=final_report,
            assumptions_made=assumptions,
            runtime_app=runtime_app
        )

        result.compiled_app = compiled
        result.success = True

    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {e}"

    result.total_latency_seconds = round(time.time() - pipeline_start, 3)
    return result
