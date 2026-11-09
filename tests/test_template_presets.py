"""Selectable recipes remain equal to the actually accepted API graphs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bench.probe_server_session import graphs
from bench.run_templates import ACCEPTED, build_graph

ROOT = Path(__file__).resolve().parents[1]


def test_qwen_preset_matches_frozen_accepted_graph():
    actual, _ = build_graph(ACCEPTED[0], 'accepted')
    expected = json.loads((ROOT / 'tests/fixtures/qwen-api-initial-graph.json').read_text())['initial_graph']
    actual['461']['inputs']['filename_prefix'] = expected['461']['inputs']['filename_prefix']
    assert actual == expected


def test_wan_preset_matches_accepted_session_graph():
    actual, _ = build_graph(ACCEPTED[1], 'accepted')
    expected = graphs('preset')[0][1]
    actual['58']['inputs']['filename_prefix'] = expected['58']['inputs']['filename_prefix']
    assert actual == expected


def test_accepted_default_dry_run_has_only_qualified_models():
    result = subprocess.run([sys.executable, str(ROOT / 'bench/run_templates.py'), '--preset', 'accepted', '--dry-run'],
                            capture_output=True, text=True, check=True)
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert tuple(record['template'] for record in records) == ACCEPTED
    assert all(record['preset'] == 'accepted' for record in records)


@pytest.mark.parametrize('name', ['video_minimax_h3_t2v', 'video_wan2_2_14B_t2v'])
def test_pending_models_cannot_select_accepted_preset(name):
    with pytest.raises(ValueError, match='No accepted preset'):
        build_graph(name, 'accepted')
