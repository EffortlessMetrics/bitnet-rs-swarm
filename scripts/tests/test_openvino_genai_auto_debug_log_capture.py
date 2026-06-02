#!/usr/bin/env python3
"""Tests for runtime AUTO debug-log capture evidence wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = SCRIPTS_DIR / "tests" / "fixtures"


class OpenVinoGenAiAutoDebugLogCaptureTests(unittest.TestCase):
    def test_capture_wrapper_emits_validator_shaped_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = FIXTURES_DIR / "openvino-auto-debug-log-stateful.txt"
            capture_log_fixture = tmp_path / "captured-debug-log.txt"
            capture_log_fixture.write_text(
                "\n".join(
                    [
                        "[WARNING] Paged Attention backend initialization failed. Falling back to SDPA backend.",
                        (
                            "[12:00:00.0000]W[plugin.cpp:967][AUTO] Setting property "
                            "ov::intel_auto::enable_startup_fallback to false for stateful model."
                        ),
                        (
                            "[12:00:00.0001]W[plugin.cpp:971][AUTO] Setting property "
                            "ov::intel_auto::enable_running_fallback to false for stateful model."
                        ),
                        fixture.read_text(encoding="utf-8"),
                    ]
                ),
                encoding="utf-8",
            )
            fake_phase = tmp_path / "fake_phase_receipt.py"
            fake_phase.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    import os
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--model-dir")
                    parser.add_argument("--machine-id")
                    parser.add_argument("--devices", nargs="+")
                    parser.add_argument("--json-out", required=True)
                    args = parser.parse_args()

                    receipt = {
                        "schema_version": "1.0.0",
                        "artifact_kind": "intel_258v_dense_slm_openvino_phase_runner",
                        "campaign": "intel-258v-platform",
                        "item": "SLM-OV258V-006",
                        "machine_id": "intel-258v",
                        "runtime_api": "openvino_genai",
                        "model_family": "qwen",
                        "model_architecture": "qwen2",
                        "quantization": "INT4_SYM",
                        "prompt_template": "qwen2.5",
                        "tokenizer_source": "hf_tokenizer_export",
                        "model": {"local_model_dir": args.model_dir, "model_binary_committed": False},
                        "environment": {
                            "python": "3.12.10",
                            "platform": "Windows-test",
                            "openvino": {"version": "2026.2.0-test", "available_devices": ["CPU", "GPU", "NPU"]},
                            "openvino_genai": {"version": "2026.2.0.0-test"},
                        },
                        "generation": {
                            "devices": [
                                {
                                    "runtime_device": "AUTO",
                                    "runtime_api": "openvino_genai",
                                    "fallback_used": False,
                                    "fallback_status": "no_application_fallback_used_auto_requested_selected_device_not_exposed",
                                    "selected_device_visibility_status": "not_exposed",
                                    "openvino_runtime_auto_selected_device_proof": False,
                                    "cases": [
                                        {
                                            "id": "math_2_plus_2",
                                            "answer_gate": {"passed": True},
                                            "fallback_used": False,
                                            "selected_device_visibility_status": "not_exposed",
                                            "openvino_runtime_auto_selected_device_proof": False,
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.json_out).write_text(json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
                    print(Path(os.environ["FAKE_OPENVINO_LOG_FIXTURE"]).read_text(encoding="utf-8"))
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            phase_json = tmp_path / "phase.json"
            log_out = tmp_path / "debug-log.txt"
            evidence_json = tmp_path / "evidence.json"
            env = dict(**os.environ, FAKE_OPENVINO_LOG_FIXTURE=str(capture_log_fixture))
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "openvino_genai_auto_debug_log_capture.py"),
                    "--phase-script",
                    str(fake_phase),
                    "--model-dir",
                    "models/openvino/qwen2.5-0.5b-instruct-int4-sym",
                    "--phase-json-out",
                    str(phase_json),
                    "--debug-log-out",
                    str(log_out),
                    "--evidence-json-out",
                    str(evidence_json),
                    "--created-utc",
                    "2026-06-02T00:00:00Z",
                ],
                check=True,
                env=env,
            )

            evidence = json.loads(evidence_json.read_text(encoding="utf-8"))

        self.assertEqual(evidence["artifact_kind"], "lunar_lake_openvino_auto_genai_debug_log_evidence")
        self.assertEqual(evidence["source_phase_receipt"]["requested_devices"], ["AUTO"])
        self.assertEqual(
            evidence["source_phase_receipt"]["phase_receipt_selected_device_visibility_status"],
            "not_exposed",
        )
        self.assertEqual(evidence["debug_log"]["openvino_log_level_env"], "2")
        self.assertEqual(evidence["genai_debug_log_evidence"]["visibility_status"], "exposed_by_genai_debug_log")
        self.assertEqual(evidence["genai_debug_log_evidence"]["execution_devices"], ["GPU.0"])
        self.assertTrue(evidence["same_run_answer_and_fallback"]["all_answer_gates_passed"])
        self.assertTrue(evidence["same_run_answer_and_fallback"]["attention_backend_warning_observed"])
        self.assertTrue(evidence["same_run_answer_and_fallback"]["auto_startup_running_fallback_disabled_observed"])
        self.assertEqual(
            len(evidence["same_run_answer_and_fallback"]["auto_startup_running_fallback_disabled_lines"]),
            2,
        )
        self.assertIn(
            "not as application route/device fallback",
            evidence["same_run_answer_and_fallback"]["attention_backend_warning_boundary"],
        )
        self.assertIn("No route policy changed.", evidence["claim_boundary"]["must_not_claim"])


if __name__ == "__main__":
    unittest.main()
