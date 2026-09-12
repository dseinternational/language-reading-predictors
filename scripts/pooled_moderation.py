# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Explain why the historical pooled-moderation command has been retired."""

import argparse

RETIREMENT = (
    "The pooled-moderation analysis is retired. Current primary gain models do "
    "not estimate its target interaction, and the moderation variants do not "
    "all express it on a common baseline scale. See "
    "notes/20260912-pooled-moderation-retirement.md for the redesign requirements."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=RETIREMENT)
    parser.parse_known_args()
    parser.error(RETIREMENT)


if __name__ == "__main__":
    main()
