"""
License registry for bundled sample PDFs used by built-in EXAMPLES.

Policy:
- Approved sample assets must be either permissive Creative Commons or
  U.S. federal public-domain materials.
- Rejected assets are intentionally tracked for auditability.
"""

from __future__ import annotations

from typing import Any, Dict

ALLOWED_LICENSE_CLASSES = {
    "public_domain_us_federal",
    "cc_by_4_0",
    "cc_by_3_0_igo",
    "cc_by_4_0_igo",
}

EXAMPLE_LICENSE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "samples/OIT-NASK-IAGen_WP140_web.pdf": {
        "sample_file": "samples/OIT-NASK-IAGen_WP140_web.pdf",
        "license_class": "cc_by_4_0",
        "status": "approved",
        "evidence_urls": [
            "https://www.ilo.org/rights-and-permissions",
        ],
        "verified_on": "2026-02-25",
        "notes": "ILO open-access knowledge products are licensed CC BY 4.0 unless stated otherwise.",
    },
    "samples/EnergyandAI.pdf": {
        "sample_file": "samples/EnergyandAI.pdf",
        "license_class": "cc_by_4_0",
        "status": "approved",
        "evidence_urls": [
            "https://www.iea.org/reports/energy-and-ai",
        ],
        "verified_on": "2026-02-25",
        "notes": "IEA report page lists CC BY 4.0 license.",
    },
    "samples/Digital Progress and Trends Report 2025, Strengthening AI Foundations.pdf": {
        "sample_file": "samples/Digital Progress and Trends Report 2025, Strengthening AI Foundations.pdf",
        "license_class": "cc_by_3_0_igo",
        "status": "approved",
        "evidence_urls": [
            "https://www.worldbank.org/en/about/legal/permissions",
        ],
        "verified_on": "2026-02-25",
        "notes": "PDF front matter indicates CC BY 3.0 IGO.",
    },
    "samples/NSF_Invention_Knowledge_Transfer_Innovation_2024.pdf": {
        "sample_file": "samples/NSF_Invention_Knowledge_Transfer_Innovation_2024.pdf",
        "license_class": "public_domain_us_federal",
        "status": "approved",
        "evidence_urls": [
            "https://ncses.nsf.gov/indicators/permissions",
            "https://ncses.nsf.gov/pubs/nsb20241",
        ],
        "verified_on": "2026-02-25",
        "notes": "U.S. federal government work; third-party components may have separate rights.",
    },
    "samples/NSF_Production_Trade_KTI_Industries_2022.pdf": {
        "sample_file": "samples/NSF_Production_Trade_KTI_Industries_2022.pdf",
        "license_class": "public_domain_us_federal",
        "status": "approved",
        "evidence_urls": [
            "https://ncses.nsf.gov/indicators/permissions",
            "https://ncses.nsf.gov/pubs/nsb20226",
        ],
        "verified_on": "2026-02-25",
        "notes": "U.S. federal government work; third-party components may have separate rights.",
    },
    "samples/NSF_RnD_Trends_International_Comparisons_2022.pdf": {
        "sample_file": "samples/NSF_RnD_Trends_International_Comparisons_2022.pdf",
        "license_class": "public_domain_us_federal",
        "status": "approved",
        "evidence_urls": [
            "https://ncses.nsf.gov/indicators/permissions",
            "https://ncses.nsf.gov/pubs/nsb20225",
        ],
        "verified_on": "2026-02-25",
        "notes": "U.S. federal government work; third-party components may have separate rights.",
    },
    "samples/NSF_STEM_Labor_Force_2024.pdf": {
        "sample_file": "samples/NSF_STEM_Labor_Force_2024.pdf",
        "license_class": "public_domain_us_federal",
        "status": "approved",
        "evidence_urls": [
            "https://ncses.nsf.gov/indicators/permissions",
            "https://ncses.nsf.gov/pubs/nsb20245",
        ],
        "verified_on": "2026-02-25",
        "notes": "U.S. federal government work; third-party components may have separate rights.",
    },
    "samples/NSF_RnD_Trends_International_Comparisons_2024.pdf": {
        "sample_file": "samples/NSF_RnD_Trends_International_Comparisons_2024.pdf",
        "license_class": "public_domain_us_federal",
        "status": "approved",
        "evidence_urls": [
            "https://ncses.nsf.gov/indicators/permissions",
            "https://ncses.nsf.gov/pubs/nsb20246",
        ],
        "verified_on": "2026-02-25",
        "notes": "U.S. federal government work; third-party components may have separate rights.",
    },
    "samples/NSF_Production_Trade_KTI_Industries_2024.pdf": {
        "sample_file": "samples/NSF_Production_Trade_KTI_Industries_2024.pdf",
        "license_class": "public_domain_us_federal",
        "status": "approved",
        "evidence_urls": [
            "https://ncses.nsf.gov/indicators/permissions",
            "https://ncses.nsf.gov/pubs/nsb20247",
        ],
        "verified_on": "2026-02-25",
        "notes": "U.S. federal government work; third-party components may have separate rights.",
    },
    "samples/IMF_WEO_April_2025_text.pdf": {
        "sample_file": "samples/IMF_WEO_April_2025_text.pdf",
        "license_class": "all_rights_reserved",
        "status": "rejected",
        "evidence_urls": [
            "https://www.imf.org/en/about/copyright-and-terms",
        ],
        "verified_on": "2026-02-25",
        "notes": "Removed from bundled examples and repository sample assets due to restrictive rights context.",
    },
}
