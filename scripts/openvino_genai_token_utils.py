"""Small helpers for OpenVINO GenAI token-accounting receipts."""

from __future__ import annotations

import hashlib
import time
from typing import Any


def _to_list(value: Any) -> Any:
    data = getattr(value, "data", value)
    if hasattr(data, "tolist"):
        return data.tolist()
    return data


def _single_batch_ids(value: Any) -> list[int]:
    data = _to_list(value)
    if isinstance(data, tuple):
        data = list(data)
    if data and isinstance(data, list) and isinstance(data[0], list):
        data = data[0]
    return [int(token_id) for token_id in (data or [])]


def _decode_generated_text(tokenizer: Any, token_ids: list[int]) -> str:
    decoded = tokenizer.decode(token_ids)
    if isinstance(decoded, list):
        return str(decoded[0]) if decoded else ""
    return str(decoded)


def _prompt_evidence(tokenizer: Any, question: str, *, collect_host_timing: bool) -> dict[str, Any]:
    messages = [{"role": "user", "content": question}]
    render_start = time.perf_counter()
    rendered = tokenizer.apply_chat_template(messages, True)
    prompt_render_wall_ms = (time.perf_counter() - render_start) * 1000.0
    tokenize_start = time.perf_counter()
    tokenized_inputs = tokenizer.encode(rendered, add_special_tokens=False)
    prompt_tokenize_wall_ms = (time.perf_counter() - tokenize_start) * 1000.0
    token_ids = _single_batch_ids(tokenized_inputs.input_ids)
    evidence = {
        "rendered_prompt": rendered,
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "_tokenized_inputs": tokenized_inputs,
    }
    if collect_host_timing:
        evidence["_host_phase_timing"] = {
            "prompt_render_wall_ms": prompt_render_wall_ms,
            "prompt_tokenize_wall_ms": prompt_tokenize_wall_ms,
        }
    return evidence


def prompt_evidence(tokenizer: Any, question: str) -> dict[str, Any]:
    return _prompt_evidence(tokenizer, question, collect_host_timing=False)


def public_prompt_evidence(prompt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in prompt.items() if not key.startswith("_")}


def generate_with_direct_token_ids(
    pipe: Any,
    tokenizer: Any,
    ov_genai: Any,
    question: str,
    max_new_tokens: int,
    streamer: Any | None = None,
    collect_host_timing: bool = False,
) -> dict[str, Any]:
    prompt = _prompt_evidence(tokenizer, question, collect_host_timing=collect_host_timing)
    result = pipe.generate(
        prompt["_tokenized_inputs"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        streamer=streamer,
    )
    generated_token_ids = _single_batch_ids(result.tokens)
    generated_text = _decode_generated_text(tokenizer, generated_token_ids)
    generation = {
        "result": result,
        "prompt": public_prompt_evidence(prompt),
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "generated_token_ids_available_from_pipeline": True,
        "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
        "generated_token_count": len(generated_token_ids),
    }
    if collect_host_timing:
        generation["host_phase_timing"] = prompt.get("_host_phase_timing", {})
    return generation
