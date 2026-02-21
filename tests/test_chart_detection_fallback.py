from content_analyzer.document_parser import detect_chart_on_page_path


def test_detect_chart_on_page_path_handles_missing_file():
    page_num, image_path, result = detect_chart_on_page_path((1, "/tmp/does-not-exist-chart.jpg"))

    assert page_num == 1
    assert image_path.endswith("does-not-exist-chart.jpg")
    assert result["has_chart"] is False
    assert result["confidence"] == 0.0
