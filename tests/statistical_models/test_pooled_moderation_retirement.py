# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The retired command must not fit a pool from current or stale output."""

import subprocess
import sys
from pathlib import Path


def test_retired_command_explains_the_reason_and_writes_no_output(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/pooled_moderation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--models-dir", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "analysis is retired" in result.stderr
    assert "20260912-pooled-moderation-retirement.md" in result.stderr
    assert list(tmp_path.iterdir()) == []
