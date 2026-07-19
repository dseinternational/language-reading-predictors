# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A tiny isolated real-sampler gate for the canonical ITT shape contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_fixed_seed_itt_posterior_and_predictive_shapes(tmp_path):
    """Compile and sample the real nutpie/PyMC path on bounded synthetic scores."""
    runner = Path(__file__).with_name("posterior_sampling_smoke_runner.py")
    cache = tmp_path / "cache"
    cache.mkdir()
    environment = {
        **os.environ,
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "PYTENSOR_FLAGS": f"base_compiledir={cache / 'pytensor'}",
        "XDG_CACHE_HOME": str(cache / "xdg"),
    }

    result = subprocess.run(
        [sys.executable, str(runner)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary == {
        "eta_shape": [2, 60, 18],
        "posterior_predictive_shape": [2, 60, 18],
        "tau_finite": True,
        "tau_shape": [2, 60],
        "y_in_bounds": True,
    }
