"""Guard against pyproject's hand-maintained py-modules list going stale.

Nothing else catches this. Tests and a local `pip install -e .` both import
top-level modules straight from the source directory, so an unlisted module
resolves fine here and the suite stays green. The Dockerfile, though, does a
non-editable `pip install .` -- only what py-modules names gets copied into
site-packages -- and the worker service runs `rq worker`, a console script
whose sys.path[0] is the bin directory, not /app. So an unlisted module is
invisible to exactly the process that needs it, and the failure surfaces as a
ModuleNotFoundError inside a container, at job-execution time, rather than
anywhere near the commit that caused it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# app.py is the Streamlit entrypoint: it's launched by path (`streamlit run
# app.py`) and never imported as a module, so it is deliberately not packaged.
NOT_PACKAGED = {"app"}


def declared_py_modules() -> set[str]:
    """Read py-modules out of pyproject.toml.

    Parsed with a regex rather than tomllib because requires-python allows
    3.10, where tomllib doesn't exist yet, and a guard test that silently
    skips on the project's own minimum version is not a guard.
    """
    src = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^py-modules\s*=\s*\[(.*?)\]", src, re.S | re.M)
    assert match, "py-modules not found in pyproject.toml"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_every_top_level_module_is_packaged():
    on_disk = {p.stem for p in ROOT.glob("*.py")} - NOT_PACKAGED
    missing = on_disk - declared_py_modules()
    assert not missing, (
        f"top-level modules missing from pyproject py-modules: {sorted(missing)}. "
        "A non-editable `pip install .` would omit them and the worker "
        "containers would fail at import time."
    )


def test_declared_modules_all_exist():
    """A name left behind after a rename/delete breaks the build outright."""
    stale = {m for m in declared_py_modules() if not (ROOT / f"{m}.py").exists()}
    assert not stale, f"py-modules names files that no longer exist: {sorted(stale)}"
