"""
Regression checks for the PAN OCR fallback (`ocr_pan_parser_v7`).

These are the three real cards that exposed the bugs, with the values read off
the card images by eye. They run the FULL fallback path (EasyOCR + Tesseract +
enhanced pass + image DOB recovery) with no Gemini involved, so they stay valid
while the Gemini spend cap is in force.

Run directly:  python tests/test_pan_extractor.py
Slow (~1 min): EasyOCR loads a torch model per card.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ocr_service import extract_text
from app.services.kyc.pan_extractor import extract_pan_card
from app.services.kyc.pan_image_ocr import run_pan_ocr

DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "uploaded_documents",
)

# (label, filename, expected fields as printed on the card)
CARDS = [
    (
        "MAHESH",
        "65c514ba-e086-4c43-8cf8-4233ef2af2f4_PAN CARD OF NC MAHESH.jpeg",
        {
            "pan_number": "AIRPM7370E",
            "name": "N C MAHESH",
            # Full-page OCR reads 28/10/1956 and "1958" appears in NEITHER
            # engine's text — recovered only by the digits-only image pass.
            "date_of_birth": "28/10/1958",
            # EasyOCR inserts a character (CHIKKIGAMOGE); the old
            # longest-candidate-wins rule preferred that over the correct read.
            "father_name": "CHIKKAMOGE GOWDA",
        },
    ),
    (
        "DEVARAJ",
        "f5bfb858-b131-4931-96f7-32e3a4eed787_PAN CARD OF M N DEVARAJ.jpeg",
        {
            "pan_number": "AUZPD1944P",
            # Tesseract reads the trailing J as S. Recovered by cross-engine
            # word merge, which needed the section splitter fix to see the
            # enhanced read at all.
            "name": "M N DEVARAJ",
            "date_of_birth": "01/06/1969",
            "father_name": "NAGARAJ GANGGAYYA MASANAGATTA",
        },
    ),
    (
        "GANESH",
        "404664b4-c396-4fe8-bfbd-f01958c044a9_PAN CARD OF GANESH RAO .G.jpeg",
        {
            "pan_number": "AJYPG2662F",
            "name": "GANESH RAO G",
            "date_of_birth": "06/10/1960",
            "father_name": "GOVINDAPPA SETTY",
        },
    ),
]


def run_card(path):
    text = extract_text(file_path=path, file_type="JPEG")

    try:
        text += "\n\n--- ENHANCED PAN OCR ---\n" + run_pan_ocr(path)
    except Exception as error:
        print("  (enhanced pass unavailable:", error, ")")

    return extract_pan_card(text, image_path=path)


def main():
    failures = []

    for label, filename, expected in CARDS:
        path = os.path.join(DOCS, filename)

        if not os.path.exists(path):
            print(f"SKIP  {label} — image not present at {path}")
            continue

        result = run_card(path)

        for field, want in expected.items():
            got = result.get(field)
            if got == want:
                print(f"PASS  {label}.{field} = {got!r}")
            else:
                print(f"FAIL  {label}.{field}: got {got!r}, want {want!r}")
                failures.append(f"{label}.{field}")

    print()
    print("ALL CHECKS PASSED" if not failures
          else "FAILURES: " + ", ".join(failures))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
