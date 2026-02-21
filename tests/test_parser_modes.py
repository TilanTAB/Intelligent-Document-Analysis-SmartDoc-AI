import sys
from types import SimpleNamespace

from langchain_core.documents import Document

from configuration.parameters import parameters
from content_analyzer.document_parser import DocumentProcessor


def _make_processor(monkeypatch: object) -> DocumentProcessor:
    monkeypatch.setattr(parameters, "ENABLE_CHART_EXTRACTION", False, raising=False)
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
