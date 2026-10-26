"""Calls to the local Ollama model.

Text calls return LangChain ``AIMessage`` objects. Structured calls pass
the Pydantic schema to Ollama, which constrains the model to valid JSON,
and return a validated Pydantic object.
"""

import json
import logging
import time

from langchain_core.messages import AIMessage
from ollama import Client

from researchops.config import (
    MODEL,
    MODEL_ATTEMPTS,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Tokens reserved for chat formatting around the messages.
TOKEN_OVERHEAD = 200


def estimate_tokens(text):
    """
    Conservative approximate token count.

    Not the model's real tokenizer; only used to keep requests
    inside the configured context window.
    """
    return max(1, len(str(text).encode("utf-8")) // 3 + 1)


def strict_schema(schema):
    """
    Return the JSON schema for a Pydantic model, checking that it is
    strict: no extra properties, and every property required.
    """
    definition = schema.model_json_schema()

    def validate(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                properties = node.get("properties", {})

                if node.get("additionalProperties") is not False:
                    raise ValueError(
                        "Strict schema requires additionalProperties=false."
                    )

                if set(node.get("required", [])) != set(properties):
                    raise ValueError("Strict schema requires every field.")

            for value in node.values():
                validate(value)

        elif isinstance(node, list):
            for value in node:
                validate(value)

    validate(definition)
    return definition


def _get_usage(response):
    prompt_tokens = getattr(response, "prompt_eval_count", None)
    completion_tokens = getattr(response, "eval_count", None)

    if prompt_tokens is None or completion_tokens is None:
        return None

    return int(prompt_tokens) + int(completion_tokens)


def invoke_model(
    system,
    user,
    schema=None,
    output_tokens=700,
    strict=False,
):
    """
    Call the local model, retrying once on unusable output.

    With ``schema``, the response is constrained to that JSON schema
    and returned as a validated Pydantic object.
    """
    output_tokens = int(output_tokens)

    if output_tokens < 1:
        raise ValueError("output_tokens must be positive.")

    if strict and schema is None:
        raise ValueError("Strict output requires a schema.")

    schema_definition = None
    effective_system = system

    if schema is not None:
        schema_definition = (
            strict_schema(schema) if strict else schema.model_json_schema()
        )

        effective_system = (
            system
            + "\n\nReturn exactly one JSON object matching the provided "
            "schema. Return all required fields. Do not include Markdown "
            "fences."
        )

    estimated = (
        estimate_tokens(effective_system)
        + estimate_tokens(user)
        + output_tokens
        + TOKEN_OVERHEAD
    )

    if schema_definition is not None:
        estimated += estimate_tokens(
            json.dumps(schema_definition, separators=(",", ":"))
        )

    if estimated > OLLAMA_NUM_CTX:
        raise ValueError(
            f"Estimated request size {estimated} exceeds the configured "
            f"Ollama context {OLLAMA_NUM_CTX}. Reduce the input or raise "
            "OLLAMA_NUM_CTX."
        )

    request = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": 0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": output_tokens,
        },
        # Qwen3 thinking mode is off: answers stay short and the JSON
        # is not preceded by long reasoning.
        "think": False,
        "stream": False,
    }

    if schema_definition is not None:
        request["format"] = schema_definition

    client = Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)
    last_error = None

    for attempt in range(1, MODEL_ATTEMPTS + 1):
        start = time.perf_counter()

        try:
            response = client.chat(**request)
            content = response.message.content or ""

            if not content.strip():
                raise ValueError("Ollama returned an empty response.")

            result = (
                AIMessage(content=content)
                if schema is None
                else schema.model_validate_json(content)
            )

            usage = _get_usage(response)

            print(
                f"Ollama model={MODEL} "
                f"duration={time.perf_counter() - start:.1f}s "
                f"tokens={usage if usage is not None else 'unknown'} "
                f"estimated_request={estimated}"
            )

            return result

        except Exception as exc:
            last_error = exc

            print(
                f"Ollama attempt {attempt}/{MODEL_ATTEMPTS} failed after "
                f"{time.perf_counter() - start:.1f}s: "
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )

    raise last_error