import ast
from pathlib import Path

from configuration.example_license_registry import (
    ALLOWED_LICENSE_CLASSES,
    EXAMPLE_LICENSE_REGISTRY,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"
SAMPLES_DIR = REPO_ROOT / "samples"
IMF_SAMPLE_PATH = "samples/IMF_WEO_April_2025_text.pdf"


def _load_examples_from_main():
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXAMPLES":
                    return ast.literal_eval(node.value)
    raise AssertionError("EXAMPLES dictionary assignment was not found in main.py")


def _all_example_file_paths():
    examples = _load_examples_from_main()
    paths = []
    for payload in examples.values():
        paths.extend(payload.get("file_paths", []))
    return sorted(set(paths))


def test_every_example_file_is_registered():
    missing = [path for path in _all_example_file_paths() if path not in EXAMPLE_LICENSE_REGISTRY]
    assert not missing, f"Example file paths missing in license registry: {missing}"


def test_every_example_file_is_approved_and_allowed():
    invalid = []
    for path in _all_example_file_paths():
        record = EXAMPLE_LICENSE_REGISTRY[path]
        if record.get("status") != "approved":
            invalid.append(f"{path}: status={record.get('status')}")
        if record.get("license_class") not in ALLOWED_LICENSE_CLASSES:
            invalid.append(f"{path}: license_class={record.get('license_class')}")
    assert not invalid, f"Example file license violations: {invalid}"


def test_every_bundled_sample_pdf_is_approved():
    bundled = [
        f"samples/{pdf_path.name}"
        for pdf_path in sorted(SAMPLES_DIR.glob("*.pdf"))
    ]
    missing = [path for path in bundled if path not in EXAMPLE_LICENSE_REGISTRY]
    assert not missing, f"Bundled samples missing from registry: {missing}"

    non_approved = [
        path
        for path in bundled
        if EXAMPLE_LICENSE_REGISTRY[path].get("status") != "approved"
    ]
    assert not non_approved, f"Bundled samples must be approved: {non_approved}"


def test_imf_is_not_in_examples_and_marked_rejected():
    assert IMF_SAMPLE_PATH not in _all_example_file_paths()
    assert IMF_SAMPLE_PATH in EXAMPLE_LICENSE_REGISTRY
    assert EXAMPLE_LICENSE_REGISTRY[IMF_SAMPLE_PATH]["status"] == "rejected"

