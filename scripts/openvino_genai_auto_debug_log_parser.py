#!/usr/bin/env python3
"""Parse OpenVINO GenAI AUTO debug logs into selected-device evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_BLOCK_TITLE = "Model: Stateful LLM model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Captured stdout/stderr debug log.")
    parser.add_argument("--json-out", type=Path, help="Optional output path for the evidence block.")
    parser.add_argument("--block-title", default=DEFAULT_BLOCK_TITLE)
    parser.add_argument(
        "--fail-if-missing",
        action="store_true",
        help="Exit non-zero if the requested model block is absent or has no execution devices.",
    )
    return parser.parse_args()


def split_devices(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def line_entry(line_no: int, text: str) -> dict[str, Any]:
    return {"line": line_no, "text": text.rstrip()}


def find_model_block(lines: list[str], block_title: str) -> tuple[int, int] | None:
    start = next((idx for idx, line in enumerate(lines) if line.strip() == block_title), None)
    if start is None:
        return None

    end = start + 1
    while end < len(lines):
        line = lines[end]
        stripped = line.strip()
        if line.startswith(" ") or stripped == "EXECUTION_DEVICES:":
            end += 1
            continue
        break
    return start, end


def parse_block_properties(block_lines: list[str]) -> dict[str, Any]:
    properties: dict[str, str] = {}
    nested_device_names: dict[str, str] = {}
    summary_devices: list[dict[str, str]] = []
    in_summary = False

    for line in block_lines:
        if line.strip() == "EXECUTION_DEVICES:":
            in_summary = True
            continue

        if in_summary:
            summary = re.match(r"^ ([^:]+):\s*(.+)$", line)
            if summary:
                device = summary.group(1).strip().strip("()")
                full_name = summary.group(2).strip()
                summary_devices.append({"device": device, "full_name": full_name})
                continue
            if line and not line.startswith(" "):
                in_summary = False

        top_level = re.match(r"^  ([A-Z0-9_]+):\s*(.*)$", line)
        if top_level:
            properties[top_level.group(1)] = top_level.group(2).strip()
            continue

        nested_device = re.match(r"^  ([A-Z]+(?:\.\d+)?):\s*$", line)
        if nested_device:
            nested_device_names[nested_device.group(1)] = nested_device.group(1)

    execution_devices = split_devices(properties.get("EXECUTION_DEVICES", ""))
    priorities = split_devices(properties.get("MULTI_DEVICE_PRIORITIES", ""))

    if not summary_devices:
        summary_devices = [
            {"device": device, "full_name": nested_device_names.get(device, "")}
            for device in execution_devices
            if device in nested_device_names
        ]

    return {
        "execution_devices": execution_devices,
        "execution_device_full_names": summary_devices,
        "multi_device_priorities": priorities,
    }


def interesting_line_refs(start_index: int, block_lines: list[str]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for offset, line in enumerate(block_lines):
        stripped = line.strip()
        if offset == 0 or "EXECUTION_DEVICES" in stripped or "MULTI_DEVICE_PRIORITIES" in stripped:
            refs.append(line_entry(start_index + offset + 1, line))
            continue
        if re.match(r"^[A-Z]+(?:\.\d+)?:\s*", stripped):
            refs.append(line_entry(start_index + offset + 1, line))
    return refs


def parse_auto_lines(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    compile_lines: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        if "[AUTO] select device:" in line:
            selected.append(line_entry(idx, line))
        if "[AUTO] Device:" in line and "Compile model took" in line:
            compile_lines.append(line_entry(idx, line))
    return selected, compile_lines


def unavailable_evidence(block_title: str) -> dict[str, Any]:
    return {
        "visibility_status": "not_exposed_by_genai_debug_log",
        "selected_device_visibility_source": "genai_debug_log",
        "source": "OpenVINO GenAI debug log",
        "block_title": block_title,
        "model_block": "stateful_llm_model",
        "phase_or_model_block_applicability": [],
        "execution_devices": [],
        "execution_device_full_names": [],
        "multi_device_priorities": [],
        "auto_scheduler_selected_device_lines": [],
        "auto_compile_lines": [],
        "line_refs": [],
        "gaps": ["stateful_llm_model_block_not_found_or_no_execution_devices"],
        "scope_note": (
            "No accepted stateful LLM model block with execution devices was parsed. "
            "This is unavailable selected-device evidence, not an empty proof."
        ),
    }


def build_debug_log_evidence(text: str, block_title: str = DEFAULT_BLOCK_TITLE) -> dict[str, Any]:
    lines = text.splitlines()
    block_range = find_model_block(lines, block_title)
    selected_lines, compile_lines = parse_auto_lines(lines)
    if block_range is None:
        evidence = unavailable_evidence(block_title)
        evidence["auto_scheduler_selected_device_lines"] = selected_lines
        evidence["auto_compile_lines"] = compile_lines
        return evidence

    start, end = block_range
    block_lines = lines[start:end]
    parsed = parse_block_properties(block_lines)
    if not parsed["execution_devices"]:
        evidence = unavailable_evidence(block_title)
        evidence["auto_scheduler_selected_device_lines"] = selected_lines
        evidence["auto_compile_lines"] = compile_lines
        evidence["block_line_range"] = [start + 1, end]
        return evidence

    return {
        "visibility_status": "exposed_by_genai_debug_log",
        "selected_device_visibility_source": "genai_debug_log",
        "source": "OpenVINO GenAI stateful LLMPipeline compiled-model debug dump",
        "block_title": block_title,
        "model_block": "stateful_llm_model",
        "phase_or_model_block_applicability": ["stateful_llm_model_block"],
        "block_line_range": [start + 1, end],
        "execution_devices": parsed["execution_devices"],
        "execution_device_full_names": parsed["execution_device_full_names"],
        "multi_device_priorities": parsed["multi_device_priorities"],
        "auto_scheduler_selected_device_lines": selected_lines,
        "auto_compile_lines": compile_lines,
        "line_refs": interesting_line_refs(start, block_lines),
        "scope_note": (
            "This is GenAI-path diagnostic evidence for the stateful LLM model block. "
            "Tokenizer, detokenizer, probe, warm-repeat, power, and route-policy "
            "claims need separate evidence."
        ),
    }


def main() -> int:
    args = parse_args()
    evidence = build_debug_log_evidence(args.log.read_text(encoding="utf-8"), args.block_title)
    payload = json.dumps(evidence, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    if args.fail_if_missing and evidence.get("visibility_status") != "exposed_by_genai_debug_log":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
