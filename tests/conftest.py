import ast

import pytest


DESELECTED_REASONS: list[tuple[str, str]] = []

_DATASET_BUILDER_MARK = "dataset_builder"


def _normalize_dataset_builder_expr(expr: str) -> str:
    """
    Replace the hyphenated alias with the valid pytest identifier.
    """
    return (expr or "").replace("dataset-builder", _DATASET_BUILDER_MARK)


def _dataset_builder_selected(config) -> bool:
    """
    Return True if the current -m expression explicitly references dataset_builder.
    """
    expr = _normalize_dataset_builder_expr(getattr(config.option, "markexpr", "") or "")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == _DATASET_BUILDER_MARK:
            return True
    return False


def _dataset_builder_enabled(config) -> bool:
    return config.getoption("--run-dataset-builder") or getattr(config, "_dataset_builder_selected", False)


def pytest_addoption(parser):
    parser.addoption(
        "--run-dataset-builder",
        action="store_true",
        default=False,
        help="Run DatasetBuilder integration tests that require external tools like GROMACS.",
    )


def pytest_configure(config):
    import dgl  # noqa: F401
    import torch  # noqa: F401
    import grappa  # noqa: F401

    current_expr = getattr(config.option, "markexpr", "")
    normalized_expr = _normalize_dataset_builder_expr(current_expr or "")
    if normalized_expr != current_expr:
        config.option.markexpr = normalized_expr

    config.addinivalue_line(
        "markers",
        "dataset_builder: mark DatasetBuilder tutorial/GROMACS integration tests (skipped unless --run-dataset-builder).",
    )

    config._dataset_builder_selected = _dataset_builder_selected(config)

    # Allow dataset-builder runs to opt back into slow tests without overriding every command line.
    default_markexpr = "not slow and not gpu"
    if config.getoption("--run-dataset-builder") and getattr(config.option, "markexpr", None) == default_markexpr:
        config.option.markexpr = "not gpu"

    DESELECTED_REASONS.clear()


def pytest_collection_modifyitems(config, items):
    if _dataset_builder_enabled(config):
        return

    skip_marker = pytest.mark.skip(
        reason="DatasetBuilder integration tests require --run-dataset-builder or -m dataset_builder."
    )
    for item in items:
        if "dataset_builder" in item.keywords:
            item.add_marker(skip_marker)


def pytest_deselected(items):
    for item in items:
        dataset_builder_enabled = _dataset_builder_enabled(item.config)
        reasons = []
        if item.get_closest_marker("slow"):
            reasons.append("marked slow (filtered by default -m expression)")
        if item.get_closest_marker("gpu"):
            reasons.append("requires GPU (filtered by default -m expression)")
        if item.get_closest_marker("dataset_builder") and not dataset_builder_enabled:
            reasons.append("needs --run-dataset-builder or -m dataset_builder")
        if not reasons:
            reasons.append("deselected by user expression")
        DESELECTED_REASONS.append((item.nodeid, "; ".join(reasons)))


def pytest_terminal_summary(terminalreporter):
    if not DESELECTED_REASONS:
        return

    terminalreporter.write_sep("-", "deselected tests detail")
    for nodeid, reason in DESELECTED_REASONS:
        terminalreporter.write_line(f"{nodeid}: {reason}")
