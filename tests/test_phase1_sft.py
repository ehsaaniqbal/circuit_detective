from __future__ import annotations

import json
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "phase1_sft.py"
SPEC = importlib.util.spec_from_file_location("phase1_sft", SCRIPT_PATH)
assert SPEC is not None
phase1_sft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase1_sft)


def test_tool_call_renders_trl_xml_format() -> None:
    rendered = phase1_sft.tool_call("submit_circuit", {"heads": ["L1H6"]})

    assert "<tool_call>" in rendered
    assert "<function=submit_circuit>" in rendered
    assert "<parameter=heads>" in rendered
    assert '["L1H6"]' in rendered


def test_synthetic_inspect_response_exposes_target_head_first() -> None:
    payload = json.loads(phase1_sft.synthetic_inspect_response("L1H6"))

    assert payload["result"]["scores"][0]["head_id"] == "L1H6"
    assert payload["done"] is False
