"""
run_resolver_tests.py — run the cell-hierarchy resolver's own test suite.

``test_cell_hierarchy_resolver.py`` ships from the subtype-ref package written for
pytest, which is not installed in this environment (the rest of the suite uses
stdlib unittest). Rather than rewrite 39 upstream tests — and risk changing what
they assert — this runner supplies the three pytest features they actually use
(``fixture``, ``mark.parametrize``, ``raises``) and executes them directly.

No CLI, by package convention. Just:

    python tests/run_resolver_tests.py

Exit code is non-zero if any test fails.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
# `cell_hierarchy` lives with the module that consumes it, not at the repo root.
_CH_PARENT = (
    _PKG_ROOT
    / "cellcyrix"
    / "single_cell_pipeline_agent"
    / "singlecell_10x"
    / "celltype_consensus"
)
for p in (str(_CH_PARENT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
#  Minimal pytest shim — only what the upstream file uses.
# --------------------------------------------------------------------------- #
def _make_pytest_shim() -> types.ModuleType:
    mod = types.ModuleType("pytest")

    def fixture(*args, **kwargs):
        """Mark a function as a fixture; the runner calls it to build the value."""

        def wrap(fn):
            fn._is_fixture = True
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap

    class _Mark:
        @staticmethod
        def parametrize(argnames, argvalues):
            names = (
                [n.strip() for n in argnames.split(",")]
                if isinstance(argnames, str)
                else list(argnames)
            )

            def wrap(fn):
                fn._parametrize = (names, list(argvalues))
                return fn

            return wrap

        def __getattr__(self, _name):  # @pytest.mark.anything else -> no-op
            def wrap(fn=None, *a, **k):
                return fn if fn is not None else (lambda f: f)

            return wrap

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"DID NOT RAISE {self.exc}")
            return issubclass(et, self.exc)

    mod.fixture = fixture
    mod.mark = _Mark()
    mod.raises = _Raises
    mod.approx = lambda v, rel=1e-6, abs=1e-12: v
    return mod


sys.modules.setdefault("pytest", _make_pytest_shim())

import test_cell_hierarchy_resolver as suite  # noqa: E402


def main() -> None:
    fixtures = {
        n: f
        for n, f in vars(suite).items()
        if callable(f) and getattr(f, "_is_fixture", False)
    }
    # pytest built-in used by test_csv_round_trip; fresh dir per test, as upstream.
    fixtures.setdefault("tmp_path", lambda: Path(tempfile.mkdtemp(prefix="resolver_")))
    cache: dict = {}
    per_test = {"tmp_path"}  # not cached — each test gets its own

    def value_for(name: str):
        if name in per_test:
            return fixtures[name]()
        if name not in cache:
            cache[name] = fixtures[name]()
        return cache[name]

    tests = [
        (n, f)
        for n, f in sorted(vars(suite).items())
        if n.startswith("test_") and callable(f)
    ]

    passed, failures = 0, []
    for name, fn in tests:
        params = getattr(fn, "_parametrize", None)
        cases = (
            [
                dict(zip(params[0], v if isinstance(v, tuple) else (v,), strict=False))
                for v in params[1]
            ]
            if params
            else [{}]
        )
        for case in cases:
            kwargs = dict(case)
            for arg in inspect.signature(fn).parameters:
                if arg not in kwargs and arg in fixtures:
                    kwargs[arg] = value_for(arg)
            label = f"{name}{tuple(case.values()) if case else ''}"
            try:
                fn(**kwargs)
                passed += 1
                print(f"  ok    {label}")
            except Exception:
                failures.append((label, traceback.format_exc()))
                print(f"  FAIL  {label}")

    print("\n" + "=" * 64)
    print(
        f" resolver tests: {passed} passed | {len(failures)} failed "
        f"({len(tests)} test functions)"
    )
    print("=" * 64)
    for label, tb in failures:
        print(f"\n--- {label} ---\n{tb}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
