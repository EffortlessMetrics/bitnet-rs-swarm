#!/usr/bin/env python3
"""Emit a static PR Plan for narrow no-Rust evidence diffs.

This helper intentionally covers only the stable schema-1 plan shape used by
empty-label docs, campaign/tracker metadata, ``ci/hardware/**`` evidence, and
small repository metadata PRs. Workflow, policy, manifest, Rust, labelled, and
unknown diffs fall back to the Rust planner in ``xtask ci plan``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SELECTED_LANES: list[dict[str, Any]] = [
    {
        "id": "pr-plan",
        "name": "PR Plan",
        "estimated_lem": 1,
        "reason": "plan artifact",
        "blocking": False,
    },
    {
        "id": "macos-arm64-route",
        "name": "Route macOS PR lane",
        "estimated_lem": 1,
        "reason": "cheap route job",
        "blocking": False,
    },
    {
        "id": "performance-tracking-route",
        "name": "Route Performance Tracking",
        "estimated_lem": 1,
        "reason": "cheap route job",
        "blocking": False,
    },
    {
        "id": "test-telemetry-route",
        "name": "Route Test Telemetry",
        "estimated_lem": 1,
        "reason": "cheap route job",
        "blocking": False,
    },
    {
        "id": "compatibility-msrv-route",
        "name": "Route MSRV Compatibility",
        "estimated_lem": 1,
        "reason": "cheap route job",
        "blocking": False,
    },
    {
        "id": "always-on-guards",
        "name": "Guards / PR Size Guard / Markdownlint / Link Check",
        "estimated_lem": 4,
        "reason": "always-on",
        "blocking": True,
    },
]


SKIPPED_LANES: list[dict[str, Any]] = [
    {
        "id": "ci-core-build-test",
        "name": "CI (Core) - build/test/clippy/docs",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "feature-matrix-full-cli",
        "name": "Feature Matrix (full-cli PR smoke)",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "feature-matrix-full",
        "name": "Feature Matrix (full)",
        "reason": "not selected for changed files or labels",
        "blocking": False,
    },
    {
        "id": "bdd-grid-check",
        "name": "BDD Grid Check",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "policy",
        "name": "Policy",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "macos-arm64-clippy",
        "name": "Clippy (macOS ARM64)",
        "reason": "not selected for changed files or labels",
        "blocking": False,
    },
    {
        "id": "compatibility-msrv",
        "name": "Compatibility (MSRV)",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "compatibility-ffi-abi",
        "name": "Compatibility (ABI/FFI)",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "compatibility-tokenizer",
        "name": "Compatibility (tokenizer)",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "gpu-native",
        "name": "GPU CI Matrix (native compile)",
        "reason": "not selected for changed files or labels",
        "blocking": True,
    },
    {
        "id": "gpu-docker",
        "name": "GPU CI Matrix (Docker)",
        "reason": "not selected for changed files or labels",
        "blocking": False,
    },
    {
        "id": "property-tests",
        "name": "Property Tests (smoke)",
        "reason": "not selected for changed files or labels",
        "blocking": False,
    },
    {
        "id": "ripr-advisory",
        "name": "ripr static exposure (advisory)",
        "reason": "not selected for changed files or labels",
        "blocking": False,
    },
]


def parse_labels(raw: str) -> list[str]:
    text = (raw or "[]").strip()
    if not text or text == "null":
        return []
    value = json.loads(text)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, dict):
        items = value.get("items", [])
        if isinstance(items, list) and all(isinstance(item, str) for item in items):
            return items
    raise ValueError(f"could not parse labels JSON: {raw}")


def is_tracker_path(path: str) -> bool:
    return path.startswith("docs/tracking/") or path.startswith(".codex/campaigns/")


def is_repo_metadata_path(path: str) -> bool:
    return path.startswith(".rails/") or path.startswith(".uselesskey/")


def is_hardware_receipt_path(path: str) -> bool:
    return path.startswith("ci/hardware/")


def is_policy_docs_path(path: str) -> bool:
    return path.startswith("policy/") or path.startswith("docs/ci/") or path == "codecov.yml"


def is_docs_path(path: str) -> bool:
    return (
        path.startswith("docs/")
        or path.endswith(".md")
        or path.startswith("README")
        or path.startswith("CHANGELOG")
        or path.startswith("CONTRIBUTING")
        or path.startswith("SECURITY")
        or path.startswith("COMPATIBILITY")
        or path.startswith("THIRD_PARTY")
        or path == "CLAUDE.md"
    )


def is_rust_input_path(path: str) -> bool:
    return (
        path.startswith("crates/")
        or path.startswith("crossval/")
        or path.startswith("tests/")
        or path.startswith("tests-new/")
        or path.startswith("tools/")
        or path.startswith("xtask/")
        or path.startswith("xtask-build-helper/")
        or path.startswith("fuzz/")
        or path.startswith("src/")
        or path.startswith("examples/")
        or path.startswith("benches/")
        or (path.startswith("scripts/") and path.endswith(".rs"))
        or path == "build.rs"
        or path == "Cargo.toml"
        or path == "Cargo.lock"
        or path == "rust-toolchain.toml"
        or path == "Makefile"
    )


def is_control_plane_path(path: str) -> bool:
    return (
        path.startswith(".github/")
        or path.startswith(".cargo/")
        or path.startswith("policy/")
        or path in {"clippy.toml", "codecov.yml"}
    )


def eligible(paths: list[str], labels: list[str]) -> tuple[bool, str]:
    if labels:
        return False, "labels present"
    if not paths:
        return False, "no changed files"

    for path in paths:
        if is_control_plane_path(path):
            return False, f"control-plane path requires Rust planner: {path}"
        if is_policy_docs_path(path):
            return False, f"policy/docs path requires Rust planner: {path}"
        if is_rust_input_path(path):
            return False, f"Rust-input path requires Rust planner: {path}"
        if not (
            is_docs_path(path)
            or is_tracker_path(path)
            or is_hardware_receipt_path(path)
            or is_repo_metadata_path(path)
        ):
            return False, f"unsupported fast-path path: {path}"

    return True, "eligible no-Rust docs/tracker/evidence metadata path set"


def build_plan(paths: list[str], labels: list[str]) -> dict[str, Any]:
    docs_only = all(is_docs_path(path) and not is_tracker_path(path) for path in paths)
    tracker_only = all(is_tracker_path(path) for path in paths)
    tracker_or_campaign_only = tracker_only
    hardware_receipt_only = all(is_hardware_receipt_path(path) for path in paths)
    model_validation_changed = any(is_hardware_receipt_path(path) for path in paths)
    risk_packs = (
        ["docs_tracking"]
        if any(is_docs_path(path) or is_tracker_path(path) for path in paths)
        else []
    )

    return {
        "schema_version": 1,
        "budget": {
            "preferred_default_lem": 25,
            "default_limit_lem": 35,
            "estimated_lem": 9,
            "posture": "pennies",
        },
        "classification": {
            "no_rust_inputs": True,
            "docs_only": docs_only,
            "tracker_only": tracker_only,
            "tracker_or_campaign_only": tracker_or_campaign_only,
            "hardware_receipt_only": hardware_receipt_only,
            "policy_docs_only": False,
            "rust_inputs_changed": False,
            "manifest_or_toolchain_changed": False,
            "public_api_changed": False,
            "gpu_changed": False,
            "macos_changed": False,
            "model_validation_changed": model_validation_changed,
            "coverage_requested": False,
            "full_ci_requested": False,
        },
        "selected_lanes": SELECTED_LANES,
        "skipped_lanes": SKIPPED_LANES,
        "packages": {
            "changed": [],
            "direct_dependents": [],
            "canaries": [],
            "selected": [],
            "broad_sweep_required": False,
            "selection_reason": "no Rust package selection for changed files",
        },
        "risk_packs": risk_packs,
        "labels": labels,
    }


def read_changed(path: Path) -> list[str]:
    return [
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_output(path: str | None, key: str, value: str) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def append_summary(path: str | None, plan: dict[str, Any]) -> None:
    if not path:
        return
    lines = [
        "## CI Plan",
        "",
        (
            "Fast path: empty-label no-Rust docs / tracker / hardware evidence PR; "
            "Rust planner compile skipped."
        ),
        "",
        "### Selected lanes",
        "",
    ]
    for lane in plan["selected_lanes"]:
        lines.append(
            f"- `{lane['id']}` - {lane['name']} "
            f"({lane['estimated_lem']} LEM, blocking={str(lane['blocking']).lower()})"
        )
    lines.append("")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run(args: argparse.Namespace) -> int:
    try:
        labels = parse_labels(args.labels_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"fast path unavailable: {exc}", file=sys.stderr)
        append_output(args.github_output, "fast_path", "false")
        return 0

    paths = read_changed(Path(args.changed_file))
    ok, reason = eligible(paths, labels)
    if not ok:
        print(f"fast path unavailable: {reason}")
        append_output(args.github_output, "fast_path", "false")
        return 0

    plan = build_plan(paths, labels)
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    append_summary(args.github_summary, plan)
    append_output(args.github_output, "fast_path", "true")
    if args.print:
        print(json.dumps(plan, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed-file", required=True)
    parser.add_argument("--labels-json", default="[]")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--github-summary")
    parser.add_argument("--github-output")
    parser.add_argument("--print", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
