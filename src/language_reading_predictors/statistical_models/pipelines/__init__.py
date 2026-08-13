# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Family-owned fit orchestration modules (#394 and #521).

One module per model family, each exposing that family's ``fit_*`` entry point.
A module here reads top-to-bottom as one fit: declaration, preparation,
construction, inference, scientific summaries, finalisation — with the shared
mechanics delegated to :mod:`runtime`, :mod:`stages` and the artefact modules
rather than restated. Model modules, maintenance scripts and tests import the
owning family module directly; there is no aggregate compatibility facade.
"""
