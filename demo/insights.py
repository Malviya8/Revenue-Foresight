"""Back-compat shim — canonical module is repo-root `llm_layer.py`."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_layer import (  # noqa: E402,F401
    build_insight_context,
    generate_insights,
    heuristic_insights,
)
