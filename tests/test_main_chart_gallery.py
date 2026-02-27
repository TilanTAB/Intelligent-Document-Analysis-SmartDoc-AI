from pathlib import Path

from langchain_core.documents import Document

from main import build_chart_gallery_payload, extract_referenced_chart_images


def test_extract_referenced_chart_images_filters_missing_and_duplicates(tmp_path):
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"
    img4 = tmp_path / "img4.jpg"
    for p in (img1, img2, img3, img4):
        p.write_bytes(b"fake-image")

    missing = tmp_path / "missing.jpg"

    docs = [
        Document(page_content="a", metadata={"chart_image_path": str(img1), "page": 1, "type": "chart", "source": "a.pdf::hash"}),
        Document(page_content="b", metadata={"chart_image_path": str(img1), "page": 1, "type": "chart", "source": "a.pdf::hash"}),  # duplicate path
        Document(page_content="c", metadata={"chart_image_path": str(missing), "page": 2, "type": "chart", "source": "a.pdf::hash"}),  # missing file
        Document(page_content="d", metadata={"chart_image_path": str(img2), "page": 3, "type": "text", "source": "b.pdf::hash"}),
        Document(page_content="e", metadata={"chart_image_path": str(img3), "page": 4, "type": "chart", "source": "c.pdf::hash"}),
        Document(page_content="f", metadata={"chart_image_path": str(img4), "page": 5, "type": "chart", "source": "d.pdf::hash"}),  # should be trimmed by max_items
    ]

    items = extract_referenced_chart_images(docs, max_items=3)

    assert len(items) == 3
    returned_paths = [Path(p).resolve() for p, _caption in items]
    assert returned_paths == [img1.resolve(), img2.resolve(), img3.resolve()]
    assert all("• page " in caption for _path, caption in items)


def test_build_chart_gallery_payload_direct_mode(tmp_path):
    img1 = tmp_path / "direct.jpg"
    img1.write_bytes(b"direct")

    retrieved_docs = [
        Document(
            page_content="Figure 7 shows occupations by exposure",
            metadata={"chart_image_path": str(img1), "page": 7, "type": "chart", "source": "report.pdf::h1"},
        )
    ]
    chart_chunks = []

    items, note, mode = build_chart_gallery_payload(
        question="What does Figure 7 show?",
        retrieved_docs=retrieved_docs,
        chart_chunks=chart_chunks,
        max_items=3,
    )

    assert mode == "direct"
    assert note == ""
    assert len(items) == 1
    assert Path(items[0][0]).resolve() == img1.resolve()


def test_build_chart_gallery_payload_fallback_mode(tmp_path):
    img1 = tmp_path / "fallback.jpg"
    img1.write_bytes(b"fallback")

    retrieved_docs = [Document(page_content="Text-only evidence", metadata={"page": 25, "source": "report.pdf::h1"})]
    chart_chunks = [
        Document(
            page_content="Figure 18 shows automation exposure by occupation and clerical support trend.",
            metadata={"chart_image_path": str(img1), "page": 44, "type": "chart", "source": "report.pdf::h1"},
        )
    ]

    items, note, mode = build_chart_gallery_payload(
        question="Which occupations have high automation exposure?",
        retrieved_docs=retrieved_docs,
        chart_chunks=chart_chunks,
        max_items=3,
    )

    assert mode == "fallback"
    assert "No direct chart evidence" in note
    assert len(items) == 1
    assert Path(items[0][0]).resolve() == img1.resolve()


def test_build_chart_gallery_payload_none_mode(tmp_path):
    missing = tmp_path / "missing.jpg"

    retrieved_docs = [Document(page_content="Text-only evidence", metadata={"page": 25, "source": "report.pdf::h1"})]
    chart_chunks = [
        Document(
            page_content="Chart about sector output and inflation.",
            metadata={"chart_image_path": str(missing), "page": 44, "type": "chart", "source": "report.pdf::h1"},
        )
    ]

    items, note, mode = build_chart_gallery_payload(
        question="Which occupations have high automation exposure?",
        retrieved_docs=retrieved_docs,
        chart_chunks=chart_chunks,
        max_items=3,
    )

    assert mode == "none"
    assert items == []
    assert "No chart evidence retrieved" in note


def test_build_chart_gallery_payload_caps_to_three(tmp_path):
    retrieved_docs = [Document(page_content="Text-only evidence", metadata={"page": 25, "source": "report.pdf::h1"})]
    chart_chunks = []
    for idx in range(1, 6):
        p = tmp_path / f"chart_{idx}.jpg"
        p.write_bytes(b"x")
        chart_chunks.append(
            Document(
                page_content=f"Figure {idx} automation exposure occupation chart",
                metadata={"chart_image_path": str(p), "page": idx, "type": "chart", "source": "report.pdf::h1"},
            )
        )

    items, note, mode = build_chart_gallery_payload(
        question="Show occupation automation chart",
        retrieved_docs=retrieved_docs,
        chart_chunks=chart_chunks,
        max_items=3,
    )

    assert mode == "fallback"
    assert "No direct chart evidence" in note
    assert len(items) == 3
