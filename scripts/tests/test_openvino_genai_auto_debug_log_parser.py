#!/usr/bin/env python3
"""Tests for the OpenVINO GenAI AUTO debug-log parser."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = SCRIPTS_DIR / "tests" / "fixtures"

sys.path.insert(0, str(SCRIPTS_DIR))

from openvino_genai_auto_debug_log_parser import build_debug_log_evidence  # noqa: E402


class OpenVinoGenAiAutoDebugLogParserTests(unittest.TestCase):
    def test_parses_stateful_llm_execution_devices(self) -> None:
        text = (FIXTURES_DIR / "openvino-auto-debug-log-stateful.txt").read_text(encoding="utf-8")

        evidence = build_debug_log_evidence(text)

        self.assertEqual(evidence["visibility_status"], "exposed_by_genai_debug_log")
        self.assertEqual(evidence["selected_device_visibility_source"], "genai_debug_log")
        self.assertEqual(evidence["model_block"], "stateful_llm_model")
        self.assertEqual(evidence["execution_devices"], ["GPU.0"])
        self.assertEqual(evidence["multi_device_priorities"], ["GPU.0", "CPU"])
        self.assertEqual(
            evidence["execution_device_full_names"],
            [{"device": "GPU.0", "full_name": "Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)"}],
        )
        self.assertEqual(evidence["phase_or_model_block_applicability"], ["stateful_llm_model_block"])
        self.assertIn("stateful LLM model block", evidence["scope_note"])
        self.assertTrue(evidence["auto_scheduler_selected_device_lines"])
        self.assertTrue(evidence["auto_compile_lines"])

    def test_missing_stateful_block_is_unavailable_not_empty_proof(self) -> None:
        text = (FIXTURES_DIR / "openvino-auto-debug-log-tokenizer-only.txt").read_text(encoding="utf-8")

        evidence = build_debug_log_evidence(text)

        self.assertEqual(evidence["visibility_status"], "not_exposed_by_genai_debug_log")
        self.assertEqual(evidence["selected_device_visibility_source"], "genai_debug_log")
        self.assertEqual(evidence["execution_devices"], [])
        self.assertIn("stateful_llm_model_block_not_found_or_no_execution_devices", evidence["gaps"])
        self.assertIn("not an empty proof", evidence["scope_note"])

    def test_cli_emits_json_evidence_block(self) -> None:
        fixture = FIXTURES_DIR / "openvino-auto-debug-log-stateful.txt"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "openvino_genai_auto_debug_log_parser.py"),
                    "--log",
                    str(fixture),
                    "--json-out",
                    str(output),
                    "--fail-if-missing",
                ],
                check=True,
            )

            evidence = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(evidence["visibility_status"], "exposed_by_genai_debug_log")
        self.assertEqual(evidence["execution_devices"], ["GPU.0"])


if __name__ == "__main__":
    unittest.main()
