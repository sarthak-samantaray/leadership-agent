"""Suppress Python warnings project-wide (import and call `apply_runtime_bootstrap()` first)."""

from __future__ import annotations

import warnings


def apply_runtime_bootstrap() -> None:
    warnings.simplefilter("ignore")
    for cat in (
        DeprecationWarning,
        PendingDeprecationWarning,
        FutureWarning,
        UserWarning,
        ResourceWarning,
    ):
        warnings.filterwarnings("ignore", category=cat)
