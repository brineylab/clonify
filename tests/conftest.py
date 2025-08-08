import os
import sys

# Prefer installed package over local source when running tests from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
try:
    # Remove project root from the front of sys.path (pytest often inserts it)
    # so that site-packages takes precedence for `import clonify`.
    while sys.path and (sys.path[0] == PROJECT_ROOT or sys.path[0] in {"", "."}):
        sys.path.pop(0)
    # Ensure project root is still importable but with lowest precedence
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except Exception:
    pass
