import sys
from types import SimpleNamespace

from langchain_core.documents import Document

from configuration.parameters import parameters
from content_analyzer.document_parser import (
    DocumentProcessor,
    normalize_chart_detection_backend,
)


def _make_processor(monkeypatch: object) -> DocumentProcessor:
    monkeypatch.setattr(parameters, "ENABLE_CHART_EXTRACTION", False, raising=False)
    monkeypatch.setattr(parameters, "PRE_INGEST_MIN_CHUNK_CHARS", 1, raising=False)
    monkeypatch.setattr(parameters, "PRE_INGEST_MIN_TABLE_CHUNK_CHARS", 1, raising=False)
    monkeypatch.setattr(DocumentProcessor, "_init_vision_client", lambda self: None)
    return DocumentProcessor()


def test_fidelity_mode_uses_pdfplumber(monkeypatch):
    processor = _make_processor(monkeypatch)
    calls = {"fidelity": 0}

    def fake_pdfplumber(_file_path):
        calls["fidelity"] += 1
        return [Document(page_content="Fidelity text", metadata={"page": 1, "type": "text"})]

    monkeypatch.setattr(parameters, "PDF_PARSE_MODE", "fidelity", raising=False)
    monkeypatch.setattr(processor, "_load_pdf_with_pdfplumber", fake_pdfplumber)

    docs = processor._load_pdf_by_mode("dummy.pdf")

    assert calls["fidelity"] == 1
    assert docs[0].page_content == "Fidelity text"


def test_legacy_fast_mode_still_uses_pdfplumber(monkeypatch):
    processor = _make_processor(monkeypatch)
    calls = {"fidelity": 0}

    def fake_pdfplumber(_file_path):
        calls["fidelity"] += 1
        return [Document(page_content="Fidelity text", metadata={"page": 1, "type": "text"})]

    monkeypatch.setattr(parameters, "PDF_PARSE_MODE", "fast", raising=False)
    monkeypatch.setattr(processor, "_load_pdf_with_pdfplumber", fake_pdfplumber)

    docs = processor._load_pdf_by_mode("dummy.pdf")

    assert calls["fidelity"] == 1
    assert docs[0].page_content == "Fidelity text"


def test_legacy_auto_mode_still_uses_pdfplumber(monkeypatch):
    processor = _make_processor(monkeypatch)
    calls = {"fidelity": 0}

    def fake_pdfplumber(_file_path):
        calls["fidelity"] += 1
        return [Document(page_content="Fidelity text", metadata={"page": 1, "type": "text"})]

    monkeypatch.setattr(parameters, "PDF_PARSE_MODE", "auto", raising=False)
    monkeypatch.setattr(processor, "_load_pdf_with_pdfplumber", fake_pdfplumber)

    docs = processor._load_pdf_by_mode("dummy.pdf")

    assert calls["fidelity"] == 1
    assert docs[0].page_content == "Fidelity text"


def test_pdf_process_file_outputs_consistent_chunk_metadata(monkeypatch, tmp_path):
    processor = _make_processor(monkeypatch)
    monkeypatch.setattr(parameters, "PDF_PARSE_MODE", "auto", raising=False)

    source_doc = Document(
        page_content="This is a test page with enough text to chunk.",
        metadata={"page": 3, "type": "text"},
    )
    monkeypatch.setattr(processor, "_load_pdf_by_mode", lambda _file_path: [source_doc])

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%test\n")

    chunks = processor._process_file(SimpleNamespace(name=str(file_path)))

    assert chunks
    sample = chunks[0]
    assert sample.metadata["source"].startswith("sample.pdf::")
    assert sample.metadata["page"] == 3
    assert sample.metadata["type"] == "text"
    assert sample.metadata["chunk_id"].startswith("txt_")


def test_pdfplumber_text_fallback_uses_pypdf(monkeypatch, tmp_path):
    processor = _make_processor(monkeypatch)

    class FakePage:
        def extract_tables(self, *args, **kwargs):
            return []

        def find_tables(self, *args, **kwargs):
            return []

        def extract_text(self):
            raise Exception("unpack requires a buffer of 4 bytes")

    class FakePdf:
        def __init__(self):
            self.pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeReaderPage:
        def extract_text(self):
            return "fallback text from pypdf"

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [FakeReaderPage()]

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda _path: FakePdf()))
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%test\n")

    docs = processor._load_pdf_with_pdfplumber(str(file_path))

    assert docs
    assert "fallback text from pypdf" in docs[0].page_content


def test_pdf_only_mode_skips_chart_extraction(monkeypatch, tmp_path):
    processor = _make_processor(monkeypatch)
    source_doc = Document(
        page_content="PDF-only text content for testing.",
        metadata={"page": 1, "type": "text"},
    )
    calls = {"parse": 0, "charts": 0}

    def fake_parse(_file_path):
        calls["parse"] += 1
        return [source_doc]

    def fake_charts(_file_path, analyze_all_pages=False, detection_backend=None):
        calls["charts"] += 1
        return [Document(page_content="chart", metadata={"page": 1, "type": "chart"})]

    monkeypatch.setattr(processor, "_load_pdf_by_mode", fake_parse)
    monkeypatch.setattr(processor, "_extract_charts_from_pdf", fake_charts)

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%test\n")
    chunks = processor._process_file(SimpleNamespace(name=str(file_path)), pdf_analysis_mode="pdf_only")

    assert calls["parse"] == 1
    assert calls["charts"] == 0
    assert chunks
    assert all(chunk.metadata.get("type") != "chart" for chunk in chunks)


def test_vision_only_mode_skips_pdf_parse_and_forces_full_page_analysis(monkeypatch, tmp_path):
    processor = _make_processor(monkeypatch)
    processor.chart_extraction_enabled = True
    processor.vision_client = object()

    calls = {"parse": 0, "charts": 0, "analyze_all_pages": None}

    def fake_parse(_file_path):
        calls["parse"] += 1
        return [Document(page_content="text", metadata={"page": 1, "type": "text"})]

    def fake_charts(_file_path, analyze_all_pages=False, detection_backend=None):
        calls["charts"] += 1
        calls["analyze_all_pages"] = analyze_all_pages
        return [Document(page_content="vision page", metadata={"page": 1, "type": "text"})]

    monkeypatch.setattr(processor, "_load_pdf_by_mode", fake_parse)
    monkeypatch.setattr(processor, "_extract_charts_from_pdf", fake_charts)

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%test\n")
    chunks = processor._process_file(SimpleNamespace(name=str(file_path)), pdf_analysis_mode="vision_only")

    assert calls["parse"] == 0
    assert calls["charts"] == 1
    assert calls["analyze_all_pages"] is True
    assert chunks


def test_normalize_chart_detection_backend_aliases():
    assert normalize_chart_detection_backend("opencv") == "opencv_optimized"
    assert normalize_chart_detection_backend("pdf") == "pdfplumber"
    assert normalize_chart_detection_backend("unknown_backend") == "pdfplumber"


def test_both_mode_forwards_chart_detection_backend(monkeypatch, tmp_path):
    processor = _make_processor(monkeypatch)
    processor.chart_extraction_enabled = True
    processor.vision_client = object()

    calls = {"backend": None, "analyze_all_pages": None}

    source_doc = Document(
        page_content="Hybrid content",
        metadata={"page": 1, "type": "text"},
    )
    monkeypatch.setattr(processor, "_load_pdf_by_mode", lambda _file_path: [source_doc])

    def fake_charts(_file_path, analyze_all_pages=False, detection_backend=None):
        calls["backend"] = detection_backend
        calls["analyze_all_pages"] = analyze_all_pages
        return [Document(page_content="chart", metadata={"page": 1, "type": "chart"})]

    monkeypatch.setattr(processor, "_extract_charts_from_pdf", fake_charts)

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%test\n")
    chunks = processor._process_file(
        SimpleNamespace(name=str(file_path)),
        pdf_analysis_mode="both",
        chart_detection_backend="opencv_optimized",
    )

    assert chunks
    assert calls["backend"] == "opencv_optimized"
    assert calls["analyze_all_pages"] is False
