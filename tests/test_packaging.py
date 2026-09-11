"""Guard the src-layout boundary.

History: this project used a flat layout with a hand-maintained py-modules
list in pyproject. Two modules were added without being listed, and because
tests and editable installs both import straight from the source tree, the
suite stayed green while the Docker image -- a non-editable `pip install .`
-- shipped worker containers that died on ModuleNotFoundError at job time.

The src layout removes that failure mode structurally: packages.find
discovers everything under src/, so nothing inside the package can go
unshipped. What it can't prevent is library code drifting back OUT of the
package -- a new module dropped at the repo root, imported by citedguard,
resolving fine locally (pytest puts rootdir on sys.path) and missing in the
wheel. That is the same bug in a new costume, so it's what these tests watch.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "citedguard"

# Launched by path (`streamlit run app.py`, and mcp_server.py over stdio),
# never imported by the library, so they stay at the root unpackaged.
ROOT_ENTRYPOINTS = {"app", "mcp_server"}

# Not shipped: a dev-only eval harness run from the source tree.
UNPACKAGED_DIRS = {"evals"}


def test_no_stray_library_modules_at_root():
    """A new root-level .py is either an entrypoint or it belongs in the package."""
    found = {p.stem for p in ROOT.glob("*.py")}
    stray = found - ROOT_ENTRYPOINTS
    assert not stray, (
        f"unexpected top-level modules: {sorted(stray)}. Library code belongs in "
        "src/citedguard/ so it gets packaged; if this really is an entrypoint, "
        "add it to ROOT_ENTRYPOINTS."
    )


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_package_never_imports_unpackaged_code():
    """Packaged code must not depend on anything left outside the wheel."""
    forbidden = ROOT_ENTRYPOINTS | UNPACKAGED_DIRS
    offenders = {
        str(p.relative_to(ROOT)): sorted(_imported_roots(p) & forbidden)
        for p in PKG.rglob("*.py")
        if _imported_roots(p) & forbidden
    }
    assert not offenders, (
        f"packaged modules import unpackaged code: {offenders}. This works in "
        "the source tree and fails in the installed wheel."
    )


def test_pyproject_discovers_the_package():
    """The src-layout discovery config is what makes the above guarantees hold."""
    cfg = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'where\s*=\s*\[\s*"src"\s*\]', cfg), "packages.find must point at src/"
    assert "py-modules" not in cfg, (
        "py-modules is back: that reintroduces the hand-maintained list this "
        "layout exists to eliminate."
    )
