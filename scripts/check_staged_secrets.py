#!/usr/bin/env python3
"""
Fail-safe staged-content scanner for secrets and real endpoint URLs.

Usage:
  python scripts/check_staged_secrets.py

Exits non-zero when violations are detected.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List


KEY_VARS = {
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_EMBEDDING_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
}

URL_PATTERNS = [
    re.compile(r"https://[A-Za-z0-9.-]+\\.cognitiveservices\\.azure\\.com", re.IGNORECASE),
    re.compile(r"https://[A-Za-z0-9.-]+\\.openai\\.azure\\.com", re.IGNORECASE),
]

TOKEN_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
]

KEY_ASSIGNMENT = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$")


@dataclass
class Violation:
    path: str
    line_no: int
    reason: str
    detail: str


def run_git(args: List[str]) -> str:
    output = subprocess.check_output(["git", *args], stderr=subprocess.STDOUT)
    return output.decode("utf-8", errors="replace")


def staged_files() -> List[str]:
    out = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"]).strip()
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def staged_text(path: str) -> str:
    # Read from index, not working tree
    return run_git(["show", f":{path}"])


def is_placeholder(value: str) -> bool:
    value = value.strip().strip('"').strip("'")
    if not value:
        return True
    low = value.lower()

    placeholder_markers = [
        "your_",
        "your-",
        "<base64",
        "<redacted",
        "${",
        "example",
        "changeme",
        "your_api_key_here",
        "***",
    ]

    if any(marker in low for marker in placeholder_markers):
        return True

    # template-safe endpoint placeholders
    if low.startswith("https://") and ("your-resource" in low or "your-embedding-resource" in low):
        return True

    return False


def redact_value(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    if len(value) <= 8:
        return "<redacted>"
    return f"{value[:3]}...{value[-3:]}"


def scan_file(path: str, text: str) -> Iterable[Violation]:
    violations: List[Violation] = []
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith("#"):
            continue

        # Known key assignments
        m = KEY_ASSIGNMENT.match(line)
        if m:
            var, raw_val = m.groups()
            var = var.strip()
            val = raw_val.split("#", 1)[0].strip()
            val = val.strip('"').strip("'")

            if var in KEY_VARS and val and not is_placeholder(val):
                violations.append(
                    Violation(
                        path,
                        i,
                        "hardcoded_secret",
                        f"{var}={redact_value(val)}",
                    )
                )

        # Real endpoint URL scan
        for pat in URL_PATTERNS:
            for found in pat.findall(line):
                if is_placeholder(found):
                    continue
                violations.append(
                    Violation(
                        path,
                        i,
                        "real_endpoint_url",
                        found,
                    )
                )

        # Raw token patterns
        for pat in TOKEN_PATTERNS:
            for found in pat.findall(line):
                violations.append(
                    Violation(
                        path,
                        i,
                        "token_pattern_match",
                        redact_value(found),
                    )
                )

    return violations


def main() -> int:
    try:
        files = staged_files()
    except subprocess.CalledProcessError as exc:
        print("[secret-check] Failed to list staged files:")
        print(exc.output)
        return 2

    if not files:
        print("[secret-check] No staged files. Nothing to scan.")
        return 0

    violations: List[Violation] = []

    # explicit guard: .env must never be staged
    if any(path == ".env" for path in files):
        violations.append(Violation(".env", 1, "forbidden_file", ".env must not be committed"))

    for path in files:
        # Skip binary-ish known assets
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".pdf", ".bin", ".pickle")):
            continue

        try:
            text = staged_text(path)
        except subprocess.CalledProcessError:
            # Non-text or unavailable in index; ignore
            continue

        violations.extend(scan_file(path, text))

    if violations:
        print("[secret-check] Violations found in staged content:")
        for v in violations:
            print(f"  - {v.path}:{v.line_no} [{v.reason}] {v.detail}")
        print("[secret-check] Commit blocked. Replace with placeholders or unstage offending files.")
        return 1

    print("[secret-check] Staged content passed secret/endpoint checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
