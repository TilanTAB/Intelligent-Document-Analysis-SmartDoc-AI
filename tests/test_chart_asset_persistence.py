import os
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from configuration.parameters import parameters
from content_analyzer import document_parser as parser_module
from content_analyzer.document_parser import DocumentProcessor


def _install_fake_pdf_modules(monkeypatch):
    class FakePdfContext:
        def __init__(self, page_count: int):
            self.pages = [object() for _ in range(page_count)]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_convert_from_path(*_args, **kwargs):
        from PIL import Image

        output_folder = Path(kwargs["output_folder"])
        first_page = int(kwargs.get("first_page", 1))
        last_page = int(kwargs.get("last_page", first_page))
        paths = []
        for page_num in range(first_page, last_page + 1):
            img_path = output_folder / f"fake_page_{page_num:04d}.jpg"
            Image.new("RGB", (24, 24), color="white").save(img_path, "JPEG")
            paths.append(str(img_path))
        return paths

    monkeypatch.setitem(
        sys.modules,
        "pdf2image",
        SimpleNamespace(convert_from_path=fake_convert_from_path),
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfplumber",
        SimpleNamespace(open=lambda _path: FakePdfContext(page_count=1)),
    )


def _make_processor(monkeypatch, tmp_path: Path) -> DocumentProcessor:
    monkeypatch.setattr(parameters, "CACHE_DIR", str(tmp_path / "cache"), raising=False)
    monkeypatch.setattr(parameters, "ENABLE_CHART_EXTRACTION", True, raising=False)
    monkeypatch.setattr(parameters, "CHART_ENABLE_BATCH_ANALYSIS", False, raising=False)
    monkeypatch.setattr(parameters, "CHART_SKIP_GEMINI_DETECTION", True, raising=False)
    monkeypatch.setattr(DocumentProcessor, "_init_vision_client", lambda self: None)
    processor = DocumentProcessor()
    processor.vision_client = object()
    processor.vision_provider = "azure"
    processor.chart_vision_model = "dummy-model"
    return processor


def _write_fake_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    return pdf_path


def test_chart_asset_persistence_falls_back_to_copy_when_rename_fails(monkeypatch, tmp_path):
    _install_fake_pdf_modules(monkeypatch)
    processor = _make_processor(monkeypatch, tmp_path)
    pdf_path = _write_fake_pdf(tmp_path)

    monkeypatch.setattr(
        parser_module,
        "analyze_chart_images",
        lambda **_kwargs: "analysis text",
    )

    copy_calls = {"count": 0}
    real_copy2 = parser_module.shutil.copy2

    def copy_spy(src, dst, *args, **kwargs):
        copy_calls["count"] += 1
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(parser_module.shutil, "copy2", copy_spy)
    monkeypatch.setattr(os, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cross-device")))

    docs = processor._extract_charts_from_pdf(str(pdf_path), analyze_all_pages=True)

    assert docs
    assert copy_calls["count"] >= 1
    image_path = Path(docs[0].metadata["chart_image_path"])
    assert "chart_assets" in str(image_path)
    assert image_path.exists()


def test_chart_asset_paths_persist_under_cache_chart_assets(monkeypatch, tmp_path):
    _install_fake_pdf_modules(monkeypatch)
    processor = _make_processor(monkeypatch, tmp_path)
    pdf_path = _write_fake_pdf(tmp_path)

    monkeypatch.setattr(
        parser_module,
        "analyze_chart_images",
        lambda **_kwargs: "analysis text",
    )

    docs = processor._extract_charts_from_pdf(str(pdf_path), analyze_all_pages=True)

    assert docs
    image_path = Path(docs[0].metadata["chart_image_path"]).resolve()
    assert image_path.exists()
    assert processor.chart_assets_dir.resolve() in image_path.parents
    assert isinstance(docs[0], Document)
