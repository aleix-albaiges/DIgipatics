"""
Ensure ``src`` is on ``sys.path`` so scripts under ``scripts/`` can import
``paths`` and ``training_*`` modules. Import this module once at the top of a script:

    import sicap_imports  # noqa: F401
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Repo root: local modules at top level (e.g. sicap_mapping.py)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPO_ROOT = _ROOT
