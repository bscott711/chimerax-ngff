"""Make the bundle's pure modules importable outside ChimeraX.

src/__init__.py imports chimerax, so the package is registered as "cxngff" with its
path set and its __init__ never run; store.py and ngff.py then import normally
(ngff.py's relative import of store resolves to cxngff.store).
"""

import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

if "cxngff" not in sys.modules:
    pkg = types.ModuleType("cxngff")
    pkg.__path__ = [str(SRC)]
    sys.modules["cxngff"] = pkg
