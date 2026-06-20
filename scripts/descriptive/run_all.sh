#!/usr/bin/env bash
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run every descriptive plot script in order (plot00 then plot01..plot22).
# Convenience only -- each script also runs standalone:
#   python scripts/descriptive/plot06_lettersounds_vs_wordreading_gain.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

for script in scripts/descriptive/plot*.py; do
    echo ">>> $script"
    python "$script"
done
