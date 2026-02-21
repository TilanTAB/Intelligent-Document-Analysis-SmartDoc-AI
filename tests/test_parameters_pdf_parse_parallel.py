from configuration.parameters import Settings


def test_settings_accepts_pdf_parse_parallel_knobs(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PDF_PARSE_PAGE_RANGE_WORKERS", "3")
    monkeypatch.setenv("PDF_PARSE_PAGE_RANGE_SIZE", "16")

    settings = Settings()
    assert settings.PDF_PARSE_PAGE_RANGE_WORKERS == 3
    assert settings.PDF_PARSE_PAGE_RANGE_SIZE == 16


def test_settings_rejects_non_positive_pdf_parse_parallel_workers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PDF_PARSE_PAGE_RANGE_WORKERS", "0")

    try:
        Settings()
    except Exception as exc:
        assert "PDF_PARSE_PAGE_RANGE_WORKERS must be >= 1" in str(exc)
    else:
        raise AssertionError("Expected Settings() to fail when PDF_PARSE_PAGE_RANGE_WORKERS=0")
