#!/usr/bin/env python3
"""Emit an OpenVINO GenAI LLMPipeline phase receipt for bounded SLM prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openvino_genai_token_utils import generate_with_direct_token_ids


CASES = [
    {
        "id": "math_2_plus_2",
        "question": "What is 2+2? Answer briefly.",
        "max_new_tokens": 8,
        "contains_any": ["4", "four"],
    },
    {
        "id": "capital_france",
        "question": "Name the capital of France.",
        "max_new_tokens": 8,
        "contains_any": ["Paris", "paris"],
    },
    {
        "id": "rust_sentence",
        "question": "Write one short sentence about Rust.",
        "max_new_tokens": 16,
        "contains_any": ["Rust", "rust", "programming", "language", "safe", "efficient"],
    },
]


DEVICE_BACKENDS = {
    "CPU": ("openvino-cpu", "dense_slm_openvino_cpu", "openvino-genai-llmpipeline-cpu"),
    "GPU.0": ("openvino-gpu", "dense_slm_openvino_gpu_arc140v", "openvino-genai-llmpipeline-gpu0"),
    "NPU": ("openvino-npu", "dense_slm_openvino_npu", "openvino-genai-llmpipeline-npu"),
    "AUTO": ("openvino-auto", "dense_slm_openvino_auto", "openvino-genai-llmpipeline-auto"),
}

OPENVINO_RUNTIME_AUTO_SCOPE = "openvino_runtime_auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--devices", nargs="+", default=["CPU", "GPU.0", "NPU"])
    parser.add_argument(
        "--manifest",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-ir-qwen25-int4-sym-manifest.json",
    )
    parser.add_argument(
        "--gguf-cpu-answer",
        default="ci/hardware/intel-258v/2026-05-08/slm-answer-corpus-qwen25-cpu-clean-provenance.json",
    )
    parser.add_argument(
        "--gguf-cpu-phase",
        default="ci/hardware/intel-258v/2026-05-08/slm-phase-warm-session-qwen25-cpu.json",
    )
    parser.add_argument(
        "--prior-comparison",
        default="ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-phase-comparison.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path),
    }


def wall_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def phase_ms(
    value_ms: float | None,
    *,
    status: str,
    source: str,
    scope: str,
    owner: str = "host_harness",
    notes: str | None = None,
    raw_value_ms: float | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "value_ms": value_ms,
        "status": status,
        "source": source,
        "scope": scope,
        "owner": owner,
    }
    if notes is not None:
        entry["notes"] = notes
    if raw_value_ms is not None:
        entry["raw_value_ms"] = raw_value_ms
    return entry


def measured_phase(value_ms: float, *, source: str, scope: str, owner: str = "host_harness") -> dict[str, Any]:
    return phase_ms(value_ms, status="measured", source=source, scope=scope, owner=owner)


def unavailable_phase(
    *,
    status: str = "not_exposed",
    source: str,
    scope: str,
    owner: str = "host_harness",
    notes: str | None = None,
    raw_value_ms: float | None = None,
) -> dict[str, Any]:
    return phase_ms(
        None,
        status=status,
        source=source,
        scope=scope,
        owner=owner,
        notes=notes,
        raw_value_ms=raw_value_ms,
    )


def scalar_metric(value: Any, source: str) -> dict[str, Any]:
    raw = finite_float(value)
    if raw is None:
        return {
            "value_ms": None,
            "status": "not_exposed",
            "source": source,
            "raw_value_ms": None,
        }
    if raw < 0.0:
        return {
            "value_ms": None,
            "status": "not_exposed",
            "source": source,
            "raw_value_ms": raw,
            "notes": "OpenVINO GenAI sentinel timing filtered from measured summaries",
        }
    return {
        "value_ms": raw,
        "status": "measured",
        "source": source,
        "raw_value_ms": raw,
    }


def metric_phase(metric: dict[str, Any], *, source: str, scope: str, owner: str = "openvino_runtime") -> dict[str, Any]:
    value_ms = metric.get("value_ms")
    if metric.get("status") == "measured" and isinstance(value_ms, (int, float)) and value_ms >= 0.0:
        return measured_phase(float(value_ms), source=source, scope=scope, owner=owner)
    return unavailable_phase(
        source=source,
        scope=scope,
        owner=owner,
        notes=metric.get("notes", "OpenVINO GenAI metric was not exposed as a measured non-sentinel value"),
        raw_value_ms=metric.get("raw_value_ms") if isinstance(metric.get("raw_value_ms"), (int, float)) else None,
    )


def mean_metric_phase(metric: dict[str, Any], *, source: str, scope: str, owner: str = "openvino_runtime") -> dict[str, Any]:
    value_ms = metric.get("mean_ms")
    if metric.get("status") == "measured" and isinstance(value_ms, (int, float)) and value_ms >= 0.0:
        return measured_phase(float(value_ms), source=source, scope=scope, owner=owner)
    raw_mean_ms = metric.get("raw_mean_ms")
    return unavailable_phase(
        source=source,
        scope=scope,
        owner=owner,
        notes=metric.get("notes", "OpenVINO GenAI metric was not exposed as a measured non-sentinel value"),
        raw_value_ms=raw_mean_ms if isinstance(raw_mean_ms, (int, float)) else None,
    )


def host_timing_phase(host_timing: dict[str, Any], field: str, *, scope: str) -> dict[str, Any]:
    value_ms = host_timing.get(field)
    if isinstance(value_ms, (int, float)) and value_ms >= 0.0:
        return measured_phase(float(value_ms), source="harness_wall_clock", scope=scope)
    return unavailable_phase(
        source="harness_wall_clock",
        scope=scope,
        notes=f"{field} was not recorded by this runner invocation",
    )


def cache_hit_status_entry() -> dict[str, Any]:
    return {
        "value": "unknown",
        "status": "not_exposed",
        "source": "openvino_genai_llmpipeline_receipt_source",
        "scope": "direct runtime cache-hit visibility for the exact model/runtime/device/config tuple",
        "owner": "openvino_runtime",
        "notes": "LLMPipeline receipt source does not expose direct cache-hit truth",
    }


def mean_std(pair: Any) -> dict[str, Any]:
    if pair is None:
        return {
            "mean_ms": None,
            "std_ms": None,
            "status": "not_exposed",
            "source": "openvino_genai_perf_metrics",
        }
    raw_mean = finite_float(getattr(pair, "mean", None))
    raw_std = finite_float(getattr(pair, "std", None))
    if raw_mean is None or raw_std is None:
        return {
            "mean_ms": None,
            "std_ms": None,
            "status": "not_exposed",
            "source": "openvino_genai_perf_metrics",
            "raw_mean_ms": raw_mean,
            "raw_std_ms": raw_std,
        }
    if raw_mean < 0.0 or raw_std < 0.0:
        return {
            "mean_ms": None,
            "std_ms": None,
            "status": "not_exposed",
            "source": "openvino_genai_perf_metrics",
            "raw_mean_ms": raw_mean,
            "raw_std_ms": raw_std,
            "notes": "OpenVINO GenAI sentinel timing filtered from measured summaries",
        }
    return {
        "mean_ms": raw_mean,
        "std_ms": raw_std,
        "status": "measured",
        "source": "openvino_genai_perf_metrics",
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def safe_get_property(ov_core: Any, device: str, property_name: str) -> dict[str, Any]:
    try:
        value = ov_core.get_property(device, property_name)
    except Exception as exc:  # pragma: no cover - depends on installed runtime devices.
        return {
            "property": property_name,
            "available": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "property": property_name,
        "available": True,
        "value": json_safe(value),
        "error": None,
    }


def is_openvino_runtime_auto(device: str) -> bool:
    return device.upper() == "AUTO"


def runtime_auto_receipt_fields(ov_core: Any, device: str) -> dict[str, Any]:
    if not is_openvino_runtime_auto(device):
        return {}
    return {
        "auto_scope": OPENVINO_RUNTIME_AUTO_SCOPE,
        "requested_openvino_device": "AUTO",
        "openvino_requested_device": "AUTO",
        "requested_runtime_device": "AUTO",
        "runtime_requested_device": "AUTO",
        "selected_device_visibility_status": "not_exposed",
        "execution_devices_status": "not_exposed",
        "execution_device_evidence": {
            "status": "not_exposed",
            "source": "openvino_genai_llmpipeline_receipt_source",
            "phase_scope": [
                "pipeline_construction",
                "generation",
                "openvino_genai_perf_metrics",
            ],
            "property_probe": safe_get_property(ov_core, device, "EXECUTION_DEVICES"),
            "actual_selected_device_available": False,
            "actual_selected_device": None,
            "not_exposed_reason": (
                "OpenVINO GenAI LLMPipeline did not expose selected execution devices "
                "to this receipt source"
            ),
        },
        "selected_device_proof": False,
        "openvino_runtime_auto_selected_device_proof": False,
        "promotion_eligible_for_profile": False,
        "low_power_evidence": False,
        "power_advantage_claim": False,
        "acceleration_claim": False,
    }


def fallback_status_for_device(device: str) -> str:
    if is_openvino_runtime_auto(device):
        return "no_application_fallback_used_auto_requested_selected_device_not_exposed"
    normalized = device.lower().replace(".", "")
    return f"no_fallback_used_{normalized}_device_requested_and_llmpipeline_constructed"


def perf_metrics(result: Any) -> dict[str, Any]:
    perf = result.perf_metrics
    load_time = scalar_metric(perf.get_load_time(), "openvino_genai_perf_metrics.load_time")
    return {
        "load_time_ms": load_time["value_ms"],
        "load_time_status": load_time["status"],
        "load_time_source": load_time["source"],
        "load_time_raw_ms": load_time["raw_value_ms"],
        "tokenization": mean_std(perf.get_tokenization_duration()),
        "generate": mean_std(perf.get_generate_duration()),
        "inference": mean_std(perf.get_inference_duration()),
        "time_to_first_token": mean_std(perf.get_ttft()),
        "time_per_output_token": mean_std(perf.get_tpot()),
        "inter_token_latency": mean_std(perf.get_ipot()),
        "throughput": mean_std(perf.get_throughput()),
        "detokenization": mean_std(perf.get_detokenization_duration()),
        "num_input_tokens": int(perf.get_num_input_tokens()),
        "num_generated_tokens": int(perf.get_num_generated_tokens()),
    }


def run_device(device: str, model_dir: Path, ov_genai: Any, ov_core: Any) -> dict[str, Any]:
    selected_backend, backend_lane, runtime = DEVICE_BACKENDS.get(
        device,
        (f"openvino-{device.lower()}", f"dense_slm_openvino_{device.lower()}", f"openvino-genai-llmpipeline-{device.lower()}"),
    )
    auto_fields = runtime_auto_receipt_fields(ov_core, device)
    try:
        resolved_device = ov_core.get_property(device, "FULL_DEVICE_NAME")
    except Exception as exc:  # pragma: no cover - depends on installed runtime devices.
        resolved_device = f"unavailable: {type(exc).__name__}: {exc}"

    construct_start = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(model_dir), device)
    construct_wall_ms = wall_ms(construct_start)
    tokenizer_start = time.perf_counter()
    tokenizer = pipe.get_tokenizer()
    tokenizer_wall_ms = wall_ms(tokenizer_start)
    device_phase_timing = {
        "pipeline_construct_wall_ms": measured_phase(
            construct_wall_ms,
            source="harness_wall_clock",
            scope="LLMPipeline construction envelope for model read, cache lookup, compile/load, transfer, and runtime setup",
        ),
        "tokenizer_load_or_construct_wall_ms": measured_phase(
            tokenizer_wall_ms,
            source="harness_wall_clock",
            scope="pipe.get_tokenizer() after LLMPipeline construction",
        ),
        "warm_repeat_summary": unavailable_phase(
            status="not_applicable",
            source="phase_runner_single_cold_asks",
            scope="same-process or resident repeat timing after a setup ask",
            notes="This bounded phase runner does not execute a resident warm-repeat loop",
        ),
    }

    cases = []
    for case in CASES:
        chunks: list[dict[str, Any]] = []
        generation_start = time.perf_counter()
        first_chunk_at: list[float | None] = [None]

        def streamer(text: str) -> Any:
            now = time.perf_counter()
            if text and first_chunk_at[0] is None:
                first_chunk_at[0] = now
            chunks.append({"elapsed_ms": (now - generation_start) * 1000.0, "text": text})
            return ov_genai.StreamingStatus.RUNNING

        generation = generate_with_direct_token_ids(
            pipe,
            tokenizer,
            ov_genai,
            case["question"],
            case["max_new_tokens"],
            streamer=streamer,
            collect_host_timing=True,
        )
        generation_wall_ms = wall_ms(generation_start)
        result = generation["result"]
        prompt = generation["prompt"]
        generated_text = generation["generated_text"]
        matched = [needle for needle in case["contains_any"] if needle in generated_text]
        first_chunk_ms = None
        if first_chunk_at[0] is not None:
            first_chunk_ms = (first_chunk_at[0] - generation_start) * 1000.0
        metrics = perf_metrics(result)
        prompt_host_timing = generation.get("host_phase_timing", {})
        load_metric = {
            "value_ms": metrics["load_time_ms"],
            "status": metrics["load_time_status"],
            "raw_value_ms": metrics["load_time_raw_ms"],
        }
        if first_chunk_ms is None:
            first_token_phase = unavailable_phase(
                source="harness_streamer_callback",
                scope="first decoded text chunk from generate start",
                notes="No non-empty streamer chunk was observed",
            )
        else:
            first_token_phase = measured_phase(
                first_chunk_ms,
                source="harness_streamer_callback",
                scope="first decoded text chunk from generate start",
            )
        case_phase_timing = {
            "prompt_render_wall_ms": host_timing_phase(
                prompt_host_timing,
                "prompt_render_wall_ms",
                scope="chat-template rendering before prompt tokenization",
            ),
            "prompt_tokenize_wall_ms": host_timing_phase(
                prompt_host_timing,
                "prompt_tokenize_wall_ms",
                scope="tokenizer.encode on the rendered prompt before generation",
            ),
            "openvino_load_or_compile_wall_ms": metric_phase(
                load_metric,
                source="openvino_genai_perf_metrics.load_time",
                scope="OpenVINO GenAI reported load/compile timing when exposed as a non-sentinel metric",
            ),
            "cache_lookup_wall_ms": unavailable_phase(
                source="openvino_genai_llmpipeline_receipt_source",
                scope="direct OpenVINO cache lookup or cache-hit/miss overhead",
                owner="openvino_runtime",
                notes="LLMPipeline receipt source does not expose cache lookup timing",
            ),
            "cache_hit_status": cache_hit_status_entry(),
            "first_generate_wall_ms": measured_phase(
                generation_wall_ms,
                source="harness_wall_clock",
                scope="first generate call after pipeline construction, including prefill, first token, and bounded decode",
            ),
            "first_token_ms": first_token_phase,
            "ttft_ms": mean_metric_phase(
                metrics["time_to_first_token"],
                source="openvino_genai_perf_metrics.time_to_first_token",
                scope="OpenVINO GenAI reported time to first token when exposed as a non-sentinel metric",
            ),
            "decode_total_ms": unavailable_phase(
                source="openvino_genai_llmpipeline_receipt_source",
                scope="generation after first token through stop condition",
                notes="The runner records total generation and first text chunk but does not split decode-only wall time",
            ),
            "generation_wall_ms": measured_phase(
                generation_wall_ms,
                source="harness_wall_clock",
                scope="total generate call wall time from generate start to stop condition",
            ),
        }
        cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "prompt_template": "qwen2.5",
                **prompt,
                "max_new_tokens": case["max_new_tokens"],
                "greedy": True,
                "generation_config": {
                    "do_sample": False,
                    "num_beams": 1,
                    "apply_chat_template": True,
                    "max_new_tokens": case["max_new_tokens"],
                },
                "generated_text": generated_text,
                "generated_token_ids": generation["generated_token_ids"],
                "generated_token_ids_available_from_pipeline": generation[
                    "generated_token_ids_available_from_pipeline"
                ],
                "generated_token_ids_source": generation["generated_token_ids_source"],
                "generated_token_count": generation["generated_token_count"],
                "generation_wall_ms": generation_wall_ms,
                "first_streamed_text_chunk_ms": first_chunk_ms,
                "first_streamed_text_chunk": chunks[0]["text"] if chunks else None,
                "streamed_chunks_count": len(chunks),
                "streamed_text": "".join(chunk["text"] for chunk in chunks),
                "openvino_perf_metrics": metrics,
                "host_phase_timing": case_phase_timing,
                "answer_gate": {
                    "kind": "contains_any",
                    "contains_any": case["contains_any"],
                    "matched": matched,
                    "passed": bool(matched),
                },
                "fallback_used": False,
                "selected_backend": selected_backend,
                "runtime_api": "openvino_genai",
                "runtime_device": device,
                **auto_fields,
            }
        )

    passed = sum(1 for case in cases if case["answer_gate"]["passed"])
    return {
        "requested_backend": selected_backend,
        "selected_backend": selected_backend,
        "runtime_api": "openvino_genai",
        "runtime_device": device,
        "resolved_device": resolved_device,
        "backend_lane": backend_lane,
        "fallback_used": False,
        "fallback_status": fallback_status_for_device(device),
        "selected_kernel_or_runtime": runtime,
        "pipeline_construct_wall_ms": construct_wall_ms,
        "tokenizer_load_or_construct_wall_ms": tokenizer_wall_ms,
        "host_phase_timing": device_phase_timing,
        "cases_total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "cases": cases,
        "phase_coverage": {
            "pipeline_construct_wall_ms": "measured_by_runner",
            "tokenizer_load_or_construct_wall_ms": "measured_by_runner",
            "prompt_render_wall_ms": "measured_by_runner_per_case",
            "prompt_tokenize_wall_ms": "measured_by_runner_per_case",
            "openvino_load_or_compile_wall_ms": "openvino_genai_perf_metrics_when_non_sentinel_otherwise_not_exposed",
            "cache_lookup_wall_ms": "not_exposed_by_openvino_genai_llmpipeline_receipt_source",
            "cache_hit_status": "not_exposed_direct_runtime_cache_hit_truth",
            "tokenization_duration": "openvino_genai_perf_metrics",
            "time_to_first_token": "openvino_genai_perf_metrics",
            "first_streamed_text_chunk": "measured_by_streamer_callback",
            "generate_duration": "openvino_genai_perf_metrics",
            "inference_duration": "openvino_genai_perf_metrics",
            "time_per_output_token": "openvino_genai_perf_metrics",
            "detokenization_duration": "openvino_genai_perf_metrics",
            "decode_128": "not_measured_small_bounded_smoke_only",
            "prefill_512": "not_measured_small_bounded_smoke_only",
            "selected_device_visibility": (
                "not_exposed_by_openvino_genai_llmpipeline_receipt_source"
                if is_openvino_runtime_auto(device)
                else "explicit_device_requested"
            ),
        },
        **auto_fields,
    }


def main() -> int:
    args = parse_args()
    import openvino as ov
    import openvino_genai as ov_genai

    asset_resolution_start = time.perf_counter()
    model_dir = args.model_dir
    asset_resolution_wall_ms = wall_ms(asset_resolution_start)
    model_metadata_start = time.perf_counter()
    model_files = {
        name: file_record(model_dir / name)
        for name in [
            "openvino_model.xml",
            "openvino_model.bin",
            "openvino_tokenizer.xml",
            "openvino_tokenizer.bin",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
        ]
    }
    model_metadata_wall_ms = wall_ms(model_metadata_start)
    core = ov.Core()

    devices = []
    for device in args.devices:
        devices.append(run_device(device, model_dir, ov_genai, core))

    receipt_build_start = time.perf_counter()
    all_passed = all(device["failed"] == 0 for device in devices)
    fallback_used_any = any(device["fallback_used"] for device in devices)
    runtime_auto_requested_any = any(is_openvino_runtime_auto(device) for device in args.devices)
    may_claim = [
        "OpenVINO GenAI PerfMetrics are recorded for bounded Qwen2.5 LLMPipeline runs on the requested OpenVINO devices.",
        "The receipt records tokenization, TTFT, generate, inference, TPOT, throughput, detokenization, first streamed text chunk, and bounded answer-gate fields exposed by the current OpenVINO GenAI API.",
        "The bounded answer gates passed with fallback_used=false for all requested OpenVINO devices.",
    ]
    must_not_claim = [
        "OpenVINO CPU/GPU/NPU speedup or sustained phase performance is proven.",
        "prefill_512 or decode_128 phase profiles are measured.",
        "OpenVINO GPU evidence proves native OpenCL execution.",
        "OpenVINO NPU evidence proves native NPU inference outside OpenVINO GenAI.",
        "Dense SLM receipts prove BitNet QK256/I2_S behavior.",
        "Broad dense SLM quality is proven beyond bounded answer gates.",
    ]
    if runtime_auto_requested_any:
        may_claim.append(
            "Runtime-layer OpenVINO AUTO was requested and selected-device visibility was recorded as not_exposed by this receipt source."
        )
        must_not_claim.extend(
            [
                "OpenVINO runtime AUTO selected-device proof is available.",
                "OpenVINO runtime AUTO proves GPU, NPU, low_power, speedup, power advantage, route promotion, or acceleration evidence.",
            ]
        )
    out = {
        "schema_version": "1.0.0",
        "artifact_kind": "intel_258v_dense_slm_openvino_phase_runner",
        "campaign": "intel-258v-platform",
        "item": "SLM-OV258V-006",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "machine_id": args.machine_id,
        "proof_stage": "phase_measured",
        "comparison_scope": "same_machine_qwen2_5_openvino_genai_perf_metrics_and_streamer_timing",
        "requested_backend": "openvino-cpu-gpu-npu",
        "selected_backend": "openvino-cpu-gpu-npu",
        "runtime_api": "openvino_genai",
        "fallback_used": fallback_used_any,
        "backend_lane": "dense_slm_openvino_phase_runner",
        "model_family": "qwen",
        "model_architecture": "qwen2",
        "quantization": "INT4_SYM",
        "prompt_template": "qwen2.5",
        "tokenizer_source": "hf_tokenizer_export",
        "paths": {
            "openvino_ir_manifest": args.manifest,
            "gguf_cpu_answer": args.gguf_cpu_answer,
            "gguf_cpu_phase": args.gguf_cpu_phase,
            "prior_phase_comparison": args.prior_comparison,
        },
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_model_dir": model_dir.as_posix(),
            "model_binary_committed": False,
            "files": model_files,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "openvino": {
                "version": ov.get_version(),
                "available_devices": core.available_devices,
            },
            "openvino_genai": {
                "version": getattr(ov_genai, "__version__", None),
            },
            "transformers": {
                "tokenizer_loaded_from_export_dir": False,
                "openvino_tokenizer_loaded_from_export_dir": True,
                "generated_token_ids_available_from_pipeline": True,
            },
        },
        "generation": {
            "devices_total": len(devices),
            "all_answer_gates_passed": all_passed,
            "quality_gate_scope": "bounded_three_case_dense_slm_smoke_only",
            "devices": devices,
        },
        "host_phase_timing_schema": "openvino_host_phase_timing.v1",
        "host_phase_timing": {
            "asset_resolution_wall_ms": measured_phase(
                asset_resolution_wall_ms,
                source="harness_wall_clock",
                scope="argument and path reference resolution before OpenVINO Core construction",
            ),
            "model_metadata_or_hash_wall_ms": measured_phase(
                model_metadata_wall_ms,
                source="harness_wall_clock",
                scope="file stat and sha256 metadata collection for OpenVINO model and tokenizer assets",
            ),
            "receipt_build_wall_ms": unavailable_phase(
                source="harness_wall_clock",
                scope="receipt object assembly after route timing is complete",
                notes="filled after receipt object construction",
            ),
            "receipt_write_wall_ms": unavailable_phase(
                source="self_referential_receipt_write",
                scope="receipt serialization and file write",
                notes="receipt write timing is not persisted to avoid a self-referential second write",
            ),
            "telemetry_collect_wall_ms": unavailable_phase(
                source="phase_runner_no_power_thermal_probe",
                scope="power, thermal, memory, and device telemetry collection",
                notes="This runner does not collect telemetry probes",
            ),
        },
        "verification": {
            "llmpipeline_constructed_for_all_devices": len(devices) == len(args.devices),
            "generation_ran_for_all_devices": True,
            "all_answer_gates_passed": all_passed,
            "fallback_used": fallback_used_any,
            "openvino_perf_metrics_recorded": True,
            "streamer_first_text_chunk_recorded": True,
            "generated_token_ids_available_from_pipeline": True,
            "host_phase_timing_status_source_recorded": True,
        },
        "claim_boundary": {
            "may_claim": may_claim,
            "must_not_claim": must_not_claim,
        },
    }
    out["host_phase_timing"]["receipt_build_wall_ms"] = measured_phase(
        wall_ms(receipt_build_start),
        source="harness_wall_clock",
        scope="receipt object assembly after route timing is complete",
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0 if all_passed and not fallback_used_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
