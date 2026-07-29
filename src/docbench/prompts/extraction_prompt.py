from __future__ import annotations

from pydantic import BaseModel

_BASE_INSTRUCTIONS = """You are extracting structured data from a scanned receipt or invoice image.
Read every field directly off the image. Do not guess or hallucinate a value: if a field is not
present or not legible, return null for it. Transcribe values exactly as printed (same digits,
punctuation and date format shown on the document) -- do not reformat or reorder them yourself,
normalization happens downstream."""


def build_prompt(schema_cls: type[BaseModel], strict_json_only: bool = False) -> str:
    """Shared extraction prompt used by every adapter. Providers with native
    structured-output/tool-forcing (Gemini response_schema, OpenAI response_format,
    Anthropic forced tool use) rely on that mechanism to guarantee shape; pass
    strict_json_only=True to additionally ask the model to emit nothing but the JSON
    object, for providers (e.g. Together/Qwen) without guaranteed schema adherence."""
    fields = schema_cls.model_json_schema().get("properties", {})
    field_list = "\n".join(f"- {name}" for name in fields)
    prompt = f"{_BASE_INSTRUCTIONS}\n\nFields to extract:\n{field_list}"
    if strict_json_only:
        prompt += (
            "\n\nRespond with ONLY a single JSON object matching these fields. "
            "No prose, no markdown code fences."
        )
    return prompt
