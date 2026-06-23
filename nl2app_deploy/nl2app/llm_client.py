"""
llm_client.py — Single point of contact with the Groq API.

Groq uses the OpenAI-compatible API with json_object mode.
We describe the expected output as a readable template (not raw JSON schema)
to avoid models echoing back schema definitions instead of instances.

Also tracks latency + token usage per call for the eval framework.
"""

import os
import time
import json
import copy
from enum import Enum
from typing import Literal, Type, TypeVar, get_args, get_origin, Union
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from openai import OpenAI

T = TypeVar("T", bound=BaseModel)

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL_CHEAP = "llama-3.1-8b-instant"
MODEL_STRONG = "llama-3.1-8b-instant"


class LLMCallResult(BaseModel):
    """Wraps a structured LLM call with telemetry — used by eval framework."""
    model_config = {"arbitrary_types_allowed": True}


def _model_to_template(model: Type[BaseModel], indent: int = 0) -> str:
    """
    Convert a Pydantic model into a human-readable JSON template string.
    Shows field names, types, and nested structure without $ref/$defs noise.
    """
    pad = "  " * indent
    lines = ["{"]
    fields = model.model_fields
    items = list(fields.items())
    for i, (name, field) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        annotation = field.annotation
        desc = field.description or ""
        hint = f"  // {desc}" if desc else ""
        type_hint = _type_hint(annotation, indent + 1)
        lines.append(f'{pad}  "{name}": {type_hint}{comma}{hint}')
    lines.append(pad + "}")
    return "\n".join(lines)


def _type_hint(annotation, indent: int) -> str:
    """Produce a concise type hint string for the template."""
    if annotation is None:
        return "null"
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] -> X or null
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _type_hint(non_none[0], indent) + " | null"
        return "null"

    # List[X]
    if origin is list:
        if args:
            inner = _type_hint(args[0], indent)
            return f"[{inner}, ...]"
        return "[]"

    # Nested Pydantic model
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _model_to_template(annotation, indent)

    # Enum — list the allowed values explicitly
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        values = [f'"{e.value}"' for e in annotation]
        return " | ".join(values)

    # Literal["a", "b"] — list values
    if origin is Literal:
        return " | ".join(f'"{v}"' for v in args)

    # Primitives
    name = getattr(annotation, "__name__", str(annotation))
    type_map = {
        "str": '"string"', "int": "0", "float": "0.0", "bool": "true",
    }
    return type_map.get(name, f'"{name}"')


def call_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    model: str = MODEL_CHEAP,
    temperature: float = 0.2,
) -> tuple[T, dict]:
    """
    Calls Groq with json_object mode, guided by a readable output template.

    Returns:
        (parsed_pydantic_object, telemetry_dict)
    """
    template = _model_to_template(response_model)
    augmented_system = (
        f"{system_prompt}\n\n"
        f"IMPORTANT: Respond with a single JSON object matching this structure exactly.\n"
        f"Do NOT return an array. Do NOT return a schema. Return a DATA INSTANCE:\n\n"
        f"{template}\n\n"
        f"Return ONLY the JSON object — no markdown fences, no explanation."
    )

    start = time.time()

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": user_prompt},
        ],
    )

    latency = time.time() - start
    content = completion.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Groq returned malformed JSON: {e}\nRaw: {content[:500]}")

    try:
        parsed: T = response_model.model_validate(data)
    except Exception as e:
        raise ValueError(f"Response didn't match schema: {e}\nRaw: {content[:500]}")

    telemetry = {
        "model": model,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
        "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
    }

    return parsed, telemetry


def call_structured_raw_json(
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL_CHEAP,
    temperature: float = 0.2,
) -> tuple[dict, dict]:
    """
    Fallback path: plain JSON — used by the repair engine when patching
    an arbitrary JSON fragment rather than a full Pydantic model.
    """
    start = time.time()

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    latency = time.time() - start
    content = completion.choices[0].message.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid_json: model returned malformed JSON: {e}")

    telemetry = {
        "model": model,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
        "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
    }
    return parsed, telemetry
