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


def test_settings_accepts_pdf_analysis_mode(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PDF_ANALYSIS_MODE", "vision_only")

    settings = Settings()
    assert settings.PDF_ANALYSIS_MODE == "vision_only"


def test_settings_rejects_invalid_pdf_analysis_mode(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PDF_ANALYSIS_MODE", "something_else")

    try:
        Settings()
    except Exception as exc:
        assert "PDF_ANALYSIS_MODE must be one of" in str(exc)
    else:
        raise AssertionError("Expected Settings() to fail for invalid PDF_ANALYSIS_MODE")


def test_settings_accepts_chart_detection_backend(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CHART_DETECTION_BACKEND", "opencv_optimized")

    settings = Settings()
    assert settings.CHART_DETECTION_BACKEND == "opencv_optimized"


def test_settings_rejects_invalid_chart_detection_backend(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CHART_DETECTION_BACKEND", "legacy_opencv")

    try:
        Settings()
    except Exception as exc:
        assert "CHART_DETECTION_BACKEND must be one of" in str(exc)
    else:
        raise AssertionError("Expected Settings() to fail for invalid CHART_DETECTION_BACKEND")
