#!/usr/bin/env python3
"""Capture README screenshots for chart-evidence gallery modes (direct/fallback/none)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
APP_URL = os.getenv("SMARTDOC_URL", "http://127.0.0.1:7860")
LOG_PATH = ROOT / os.getenv("SMARTDOC_LOG", "logs/app.log")
SAMPLE_PDF = ROOT / os.getenv("SMARTDOC_SAMPLE", "samples/OIT-NASK-IAGen_WP140_web.pdf")
OUT_DIR = ROOT / "docs/screenshots"
DEFAULT_APP_CMD = "env\\Scripts\\python.exe main.py" if os.name == "nt" else "./env/Scripts/python.exe main.py"
APP_CMD = os.getenv("SMARTDOC_APP_CMD", DEFAULT_APP_CMD)
APP_START_TIMEOUT_S = float(os.getenv("SMARTDOC_APP_START_TIMEOUT_S", "240"))
ANSWER_TIMEOUT_S = float(os.getenv("SMARTDOC_ANSWER_TIMEOUT_S", "420"))

TARGETS: List[Dict[str, object]] = [
    {
        "mode": "direct",
        "output": "chart-gallery-direct.png",
        "questions": [
            "What does Figure 18 show?",
            "According to Figure 18, which occupations are highlighted as having rising automation exposure?",
            "What does Figure 7 show?",
        ],
    },
    {
        "mode": "fallback",
        "output": "chart-gallery-fallback.png",
        "questions": [
            "Which occupations are most likely to be automated by AI?",
            "Which jobs are most exposed to GenAI according to the report?",
            "What occupations have the highest exposure to automation?",
        ],
    },
    {
        "mode": "none",
        "output": "chart-gallery-none.png",
        "questions": [
            "What is the Schwarzschild radius of Sagittarius A*?",
            "Explain quantum chromodynamics beta function coefficients.",
            "What is Europa moon radius in kilometers?",
        ],
    },
]

MODE_RE = re.compile(r"\[CHART_GALLERY\]\s+mode=(direct|fallback|none)")


def is_server_ready() -> bool:
    try:
        req = Request(APP_URL, method="GET")
        with urlopen(req, timeout=2) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def wait_for_server(timeout_s: float) -> bool:
    started = time.time()
    while time.time() - started < timeout_s:
        if is_server_ready():
            return True
        time.sleep(1.2)
    return False


def log_size() -> int:
    try:
        return LOG_PATH.stat().st_size
    except Exception:
        return 0


def modes_since(offset: int) -> List[str]:
    try:
        with LOG_PATH.open("rb") as fh:
            fh.seek(min(offset, LOG_PATH.stat().st_size))
            text = fh.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    return [m.group(1) for m in MODE_RE.finditer(text)]


def wait_for_mode(offset: int, timeout_s: float) -> str:
    started = time.time()
    while time.time() - started < timeout_s:
        modes = modes_since(offset)
        if modes:
            return modes[-1]
        time.sleep(1.2)
    raise TimeoutError(f"Timed out waiting for [CHART_GALLERY] mode in {LOG_PATH}")


def wait_for_enabled(locator, timeout_s: float = 120.0) -> None:
    started = time.time()
    while time.time() - started < timeout_s:
        try:
            if locator.is_visible() and not locator.is_disabled():
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("Submit button did not become enabled")


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app_proc: subprocess.Popen[str] | None = None

    try:
        if not is_server_ready():
            print(f"[screenshot] App not running at {APP_URL}. Starting with: {APP_CMD}", flush=True)
            app_out = (ROOT / "logs/readme_screenshots_app.out.log").open("a", encoding="utf-8")
            app_proc = subprocess.Popen(
                APP_CMD,
                cwd=ROOT,
                shell=True,
                stdout=app_out,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if not wait_for_server(APP_START_TIMEOUT_S):
                raise RuntimeError(f"App did not become ready within {APP_START_TIMEOUT_S:.0f}s")

        if not SAMPLE_PDF.exists():
            raise FileNotFoundError(f"Sample PDF not found: {SAMPLE_PDF}")

        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(
                "Missing Playwright Python dependency. Install with "
                "`pip install playwright` and then run "
                "`python -m playwright install chromium`."
            ) from exc

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1720, "height": 1100})

            page.goto(APP_URL, wait_until="domcontentloaded", timeout=120000)
            page.get_by_text("SmartDoc AI - Document Q&A").first.wait_for(timeout=120000)

            file_input = page.locator("input[type='file']").first
            file_input.set_input_files(str(SAMPLE_PDF))
            page.get_by_text(SAMPLE_PDF.name, exact=False).first.wait_for(timeout=120000)

            question_box = page.get_by_label("Ask a question").first
            submit = page.get_by_role("button", name=re.compile(r"Get Answer", re.I)).first

            for target in TARGETS:
                target_mode = str(target["mode"])
                questions = list(target["questions"])
                matched = False
                seen: List[str] = []

                for q in questions:
                    print(f"[screenshot] Asking for target={target_mode}: {q}", flush=True)
                    wait_for_enabled(submit, timeout_s=180)
                    question_box.fill(q)
                    offset = log_size()
                    submit.click()
                    mode = wait_for_mode(offset, ANSWER_TIMEOUT_S)
                    seen.append(f"{q} => {mode}")
                    wait_for_enabled(submit, timeout_s=ANSWER_TIMEOUT_S)
                    page.wait_for_timeout(1800)

                    if mode == target_mode:
                        out_file = OUT_DIR / str(target["output"])
                        answers_header = page.get_by_text("Answers").first
                        if answers_header.count() > 0:
                            answers_header.scroll_into_view_if_needed()
                        page.screenshot(path=str(out_file), full_page=True)
                        print(f"[screenshot] Captured {target_mode} => {out_file}", flush=True)
                        matched = True
                        break

                if not matched:
                    tried = "\n".join(f"  - {x}" for x in seen)
                    raise RuntimeError(
                        f"Could not produce mode={target_mode}. Tried:\n{tried}"
                    )

            browser.close()
            print("[screenshot] All screenshots captured.", flush=True)

    finally:
        if app_proc and app_proc.poll() is None:
            app_proc.terminate()
            try:
                app_proc.wait(timeout=10)
            except Exception:
                app_proc.kill()


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"[screenshot] ERROR: {exc}", flush=True)
        sys.exit(1)
