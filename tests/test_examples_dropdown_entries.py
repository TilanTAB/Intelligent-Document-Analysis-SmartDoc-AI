import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"


def _load_examples():
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXAMPLES":
                    return ast.literal_eval(node.value)
    raise AssertionError("EXAMPLES dictionary assignment was not found in main.py")


def test_examples_include_new_2024_nsf_entries_and_exclude_imf():
    examples = _load_examples()

    assert "IMF WEO April 2025" not in examples

    assert "NSF STEM Labor Force 2024 (LBR-1)" in examples
    assert "NSF R&D Trends 2024 (RD-1)" in examples
    assert "NSF KTI Industries 2024 (KTI-1)" in examples


def test_new_2024_nsf_questions_are_chart_focused():
    examples = _load_examples()

    assert "Figure LBR-1" in examples["NSF STEM Labor Force 2024 (LBR-1)"]["question"]
    assert "Figure RD-1" in examples["NSF R&D Trends 2024 (RD-1)"]["question"]
    assert "Figure KTI-1" in examples["NSF KTI Industries 2024 (KTI-1)"]["question"]

