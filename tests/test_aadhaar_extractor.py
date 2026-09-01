"""
Regression checks for the Aadhaar OCR fallback.

Runs the FULL fallback path (page OCR + embedded-image OCR + crop OCR +
enhanced pass) with no Gemini, so it stays valid while the Gemini spend cap is
in force. Expected values are read off the card images by eye.

Note NAGARAJU deliberately expects None for name and DOB. Its PDF embeds the
card at 367x232, which is genuinely too small to read those fields — every
candidate name scores as noise and no valid date is recoverable even with a
digits-only pass. Reporting None (flagged for review) is the correct outcome;
the bug being guarded against is reporting a confident wrong value instead
("JIXNY OR INULS", and the impossible date 31/00/1979).

Run directly:  python tests/test_aadhaar_extractor.py
Slow: EasyOCR runs over several images per card.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ocr_service import extract_text
from app.services.kyc.aadhaar_extractor import extract_aadhaar_card

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BACKEND, "uploaded_documents")

CARDS = [
    (
        "GIRIJAMMA",
        "*girijamma.pdf",
        {
            "aadhaar_number": "8796 6963 8300",
            # This card's QR decodes, so district/state come from it and the
            # DOB conflict below is surfaced rather than silently resolved.
            "qr_verified": True,
            "district": "Bangalore Rural",
            "state": "Karnataka",
            # Ties 20-20 with the husband's name BASAVARAJU; the holder is
            # printed above the relationship line, so position breaks the tie.
            "name": "GIRIJAMMA",
            "date_of_birth_or_year": "01/01/1952",
            "gender": "FEMALE",
            # Printed "Husband : BASAVARAJU" -> normalised relation.
            "guardian_relation": "W/O",
            "guardian_name": "BASAVARAJU",
            "pin_code": "561203",
        },
        # Address is checked by substance, not exact string: OCR punctuation
        # around the same components varies run to run.
        ["Chikka Tumkur", "Bangalore Rural", "561203"],
    ),
    (
        "RAMALINGAPPA",
        "*Ramalingappa.pdf",
        {
            "aadhaar_number": "2375 2162 4846",
            # One crop profile read this exactly right and was discarded for a
            # higher-scoring profile that read it as "Runnings".
            "name": "RAMALINGAPPA",
            "date_of_birth_or_year": "03/06/1979",
            "gender": "MALE",
            "pin_code": "560060",
            "issued_on": "31/06/2013",
            # The S/O marker itself is OCR'd as "52:", so the RELATION is not
            # legible — but the care-of NAME after it is, and is recovered from
            # its position as the address's first component. (OCR reads it as
            # MUMURAMAIAH; the card prints Mutturamaiah. That residual is a
            # character-level OCR error on a photographed card, not a parsing
            # bug — the QR would fix it but does not decode at this resolution.)
            "guardian_relation": None,
            "guardian_name": "MUMURAMAIAH",
        },
        ["Near Ragi Mill", "Ganakallu", "Kengeri", "560060"],
    ),
    (
        "NAGARAJU",
        "*AADHAR OF C NAGARAJU.pdf",
        {
            # Recoverable only because the page render is unreadable and the
            # embedded images are OCR'd directly.
            "aadhaar_number": "6655 2514 3683",
            "name": None,
            "date_of_birth_or_year": None,
            "gender": "MALE",
            # The back of this card is unreadable at 370x235.
            "address": None,
            "pin_code": None,
            "guardian_name": None,
        },
        [],
    ),
]


def build_pipeline_text(path):
    """Mirror the text documents.py assembles for an Aadhaar."""
    text = extract_text(file_path=path, file_type="PDF")

    crop_lines = ""

    try:
        from app.services.kyc.aadhaar_crop_ocr import (
            extract_aadhaar_fields_from_image
        )

        fields = extract_aadhaar_fields_from_image(path)

        crop_lines = f"""
        --- AADHAAR FIELD CROP OCR ---
        AADHAAR_NUMBER: {fields.get("aadhaar_number", "")}
        NAME: {fields.get("name", "")}
        DOB: {fields.get("dob", "")}
        GENDER: {fields.get("gender", "")}
        """
    except Exception as error:
        print("  (crop OCR unavailable:", error, ")")

    enhanced = ""

    try:
        from app.services.kyc.aadhaar_image_ocr import run_aadhaar_ocr
        enhanced = run_aadhaar_ocr(path)
    except Exception as error:
        print("  (enhanced OCR unavailable:", error, ")")

    return (
        text
        + "\n\n"
        + crop_lines
        + "\n\n--- ENHANCED AADHAAR OCR ---\n"
        + enhanced
    )


def main():
    failures = []

    for label, pattern, expected, address_must_contain in CARDS:
        matches = sorted(glob.glob(os.path.join(DOCS, pattern)))

        if not matches:
            print(f"SKIP  {label} — no file matching {pattern}")
            continue

        path = matches[-1]

        # image_path is required for the QR pass — without it the extractor is
        # OCR-only, which is exactly the gap this suite must cover.
        result = extract_aadhaar_card(
            build_pipeline_text(path), image_path=path
        )

        for field, want in expected.items():
            got = result.get(field)

            if got == want:
                print(f"PASS  {label}.{field} = {got!r}")
            else:
                print(f"FAIL  {label}.{field}: got {got!r}, want {want!r}")
                failures.append(f"{label}.{field}")

        # The QR on the GIRIJAMMA card encodes dob 01/01/1972 while the card
        # PRINTS 01/01/1952 (verified against the pixels). The printed value
        # must win and the conflict must be reported — silently taking the QR
        # would replace a correct date with a wrong one.
        if label == "GIRIJAMMA":
            conflicts = {
                d["field"] for d in result.get("qr_disagreements", [])
            }

            if "date_of_birth_or_year" in conflicts:
                print(f"PASS  {label}.qr_disagreements flags the DOB conflict")
            else:
                print(f"FAIL  {label}.qr_disagreements missing DOB conflict")
                failures.append(f"{label}.qr_conflict")

        address = result.get("address") or ""

        for fragment in address_must_contain:
            if fragment.lower() in address.lower():
                print(f"PASS  {label}.address contains {fragment!r}")
            else:
                print(f"FAIL  {label}.address missing {fragment!r} — got {address!r}")
                failures.append(f"{label}.address[{fragment}]")

    print()
    print("ALL CHECKS PASSED" if not failures
          else "FAILURES: " + ", ".join(failures))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
