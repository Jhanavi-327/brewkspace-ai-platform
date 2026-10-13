import re
from typing import Dict


# OCR-tolerant stem of the word ಮ್ಯುಟೇಶನ್ ("mutation"). The vowel sign on ಟ is
# frequently dropped (-> ಮ್ಯುಟಶನ್) or misread as ಟೀ (-> ಮ್ಯುಟೀಶನ್) on noisy
# scans, so match the stable ಮ್ಯುಟ…ಶನ್ skeleton. This transliterated word is
# exclusive to a Mutation Register — a Patta / RTC / KYC document never spells it
# in Kannada — so it is a safe, high-signal anchor.
_MUTATION_STEM_RE = re.compile(r"ಮ್ಯುಟ.{0,2}ಶನ್")


DOCUMENT_KEYWORDS = {
    "RTC": [
        "previewrtc",
        "landrecords.karnataka.gov.in",
        "village account form no. 2",
        "r.t.c",
        "rtc",
        "valid from",
        "print page no",
        "mr ",
        "survey",
        "village",
        "taluk",
        "hobli",
        "gunta",
        "acre",
    ],

    "PAN_CARD": [
        "income tax department",
        "permanent account number",
        "permanent account number card",
        "account number card",
        "govt. of india",
        "govt of india",
        "pan",
    ],

    "AADHAAR_CARD": [
        "unique identification authority",
        "aadhaar",
        "aadhar",
        "government of india",
        "year of birth",
        "vid",
    ],

    "PATTA_BOOK": [
        "patta pusthaka",
        "mr_preview",
        "service64",
        "ಪಟ್ಟಾ ಪುಸ್ತಕ",
        "ಖಾತೆ ಪ್ರತಿ",
        "ಖಾತೆ ನಂಬರ್",
        "ಸ್ವಾದೀನದಾರರ",
        "ಅರೋಗ್ಯ ಕರ",
        "ವಿದ್ಯಾ ಕರ",
        "ಸ್ಥಳೀಯ ಕರ",
        "ಮೊಕ್ಕಾಂ",
    ],

    "EC": [
        "encumbrance certificate",
        "form no. 15",
        "form no. 16",
        "sub registrar",
        "registration district",
        "property schedule",
        "encumbrance",
    ],

    "Sale_Deed": [
        "sale deed",
        "absolute sale deed",
        "vendor",
        "purchaser",
        "sale consideration",
        "schedule property",
    ],

    "Mutation_Record": [
        "mutation",
        "mr number",
        "mutation register",
        "previous owner",
        "new owner",
    ],

    "Conversion_Order": [
        "conversion order",
        "non agricultural",
        "dc conversion",
        "land conversion",
    ],

    "Court_Order": [
        "court",
        "case no",
        "plaintiff",
        "defendant",
        "writ petition",
    ],
}

def looks_like_rtc(text: str) -> bool:
    text = text or ""
    lower = text.lower()
    compact = re.sub(r"[\s\u200b\u200c\u200d]+", "", text)

    # Clear RTC abbreviation, allowing punctuation and spaces.
    rtc_title = bool(re.search(r"\br\s*\.?\s*t\s*\.?\s*c\b", lower))

    # OCR can substitute G for C. This requires stronger supporting evidence.
    damaged_title = bool(re.search(r"\br\s*\.?\s*t\s*\.?\s*g\b", lower))

    kannada_title = (
        "ರೆಕಾರ್ಡ್" in compact
        and any(word in compact for word in ("ಗೇಣಿ", "ಗೀಣ"))
        and "ಪತ್ರಿ" in compact
    )

    signals = [
        bool(re.search(r"\bmr\s*[a-z]?\d+\s*/\s*\d{4}", lower)),
        bool(re.search(r"print\s*page\s*no", lower)),
        "ಖಾತೆ" in compact,
        "ಖರಾಬ್" in compact or "ಖರಾಬ" in compact,
        "ಬೆಳೆಯ" in compact or "ಬೆಳೆಗಳ" in compact,
        "ನೀರಾವರಿ" in compact,
        "ಸಾಗುವ" in compact or "ವ್ಯವಸಾಯಗಾರ" in compact,
    ]

    support = sum(signals)
    return (
        (rtc_title and support >= 2)
        or (kannada_title and support >= 3)
        or (damaged_title and support >= 4)
    )


def classify_document_text(text: str) -> Dict:
    text = text or ""

    text_lower = text.lower()
    text_upper = text.upper()

    # -------------------------------
    # 1. Strong PAN Card Detection
    # -------------------------------
    pan_match = re.search(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        text_upper
    )

    if (
        pan_match
        or "INCOME TAX DEPARTMENT" in text_upper
        or "PERMANENT ACCOUNT NUMBER" in text_upper
        or "ACCOUNT NUMBER CARD" in text_upper
        or "GOVT. OF INDIA" in text_upper
        or "GOVT OF INDIA" in text_upper
    ):
        return {
            "classification": "PAN_CARD",
            "confidence": 0.99,
            "candidates": [
                {
                    "type": "PAN_CARD",
                    "confidence": 0.99
                }
            ]
        }

    # Whitespace-insensitive copy so OCR spacing inside a Kannada phrase
    # (e.g. "ಖಾತೆ ಪ್ರತಿ" vs "ಖಾತೆಪ್ರತಿ") still matches.
    text_compact = re.sub(r"\s+", "", text)

    # OCR frequently injects a zero-width non-joiner/joiner (U+200C / U+200D) or
    # NBSP after a Kannada virama (e.g. "ಮ್ಯುಟೇಶನ್‌"), which would defeat an
    # exact substring match. Strip those along with whitespace for the Mutation
    # Register token tests below. Stripping is by codepoint (translate) so no
    # invisible characters appear in this source file.
    text_mr = re.sub(r"\s+", "", text).translate(
        {0x200B: None, 0x200C: None, 0x200D: None, 0x00A0: None}
    )

    # -------------------------------
    # 1b. Strong Karnataka Mutation Register Detection
    #
    # ನಮೂನೆ-12 / ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್ ಪ್ರತಿ (ನಿಯಮ 46). This MUST run before the
    # Patta and RTC blocks. The reliable anchor is the ಮ್ಯುಟೇಶನ್ ಆದೇಶದ (mutation
    # order) block, which prints in EVERY Mutation Register — both the
    # transferor/transferee layout AND the two-part survey+possessors layout
    # (ಈಗಿನ ಸರ್ವೆ ನಂಬರಿನ ವಿವರ) — and never in a Patta Book / RTC / KYC document.
    # It survives even when noisy OCR drops the ನಮೂನೆ-12 title or the ರಿಜಿಸ್ಟರ್
    # word, or renders "MR" in Latin instead of ಎಂ.ಆರ್. So those other documents
    # fall through untouched.
    # -------------------------------
    # Count OCR-tolerant hits of the mutation word. It titles every Mutation
    # Register (ಮ್ಯುಟೇಶನ್ ಪ್ರತಿ / ರಿಜಿಸ್ಟರ್) and repeats in the order block, and
    # is never spelled in Kannada on a Patta / RTC / KYC document — so even a
    # SINGLE surviving hit identifies the document when bad OCR mangles every
    # other token (a real failure mode on multi-page scans).
    mutation_stem_hits = len(_MUTATION_STEM_RE.findall(text_mr))

    strong_mr = (
        "ಮ್ಯುಟೇಶನ್ಆದೇಶದ" in text_mr
        or "ಮ್ಯುಟೇಶನ್ರಿಜಿಸ್ಟರ್" in text_mr
        or "ಹಕ್ಕುಬದಲಾವಣೆಮಾಡಿದವರು" in text_mr
        or "ಹಕ್ಕುಬದಲಾವಣೆಪಡೆದವರು" in text_mr
        or mutation_stem_hits >= 1
    )

    # MR-specific header/value tokens (spellings taken from real OCR). These are
    # deliberately NOT generic land-record words (ಸರ್ವೆ, ಭೂಕಂದಾಯ, ಖೇತುವಾರು) that
    # an RTC also carries — only fields unique to a mutation copy.
    mr_supporting_signals = sum(
        1
        for token in (
            "ನಮೂನೆ-12",
            "ನಮೂನೆ12",
            "ನಿಯಮ46",
            "ವಹಿವಾಟುಸಂಖ್ಯೆ",
            "ವಿಹವಾಟುಸಂಖ್ಯೆ",
            "ಬದಲಾವಣೆರೀತಿ",
            "ಸ್ವಾಧೀನತೆರೀತಿ",
            "ಪಹಣಿಬದಲಾವಣೆ",
            "ಅನುಮೋದಿಸಿದೆ",
            "ತಿರಸ್ಕರಿಸಿದೆ",
        )
        if token in text_mr
    )

    # Fall back to MR-specific header tokens for the rare scan where even the
    # mutation word is garbled below the stem's tolerance but several header
    # fields survive.
    if strong_mr or mr_supporting_signals >= 3:
        return {
            "classification": "MUTATION_REGISTER",
            "confidence": 0.95,
            "candidates": [
                {
                    "type": "MUTATION_REGISTER",
                    "confidence": 0.95
                }
            ]
        }

    # -------------------------------
    # 2. Strong Karnataka Patta Book / Khata Extract Detection
    #
    # This MUST run before RTC detection. A Patta Book is also served from
    # landrecords.karnataka.gov.in, so the RTC domain check below would
    # otherwise swallow it. Patta Book is distinguished from RTC by its title
    # (ಖಾತೆ ಪ್ರತಿ / ಪಟ್ಟಾ ಪುಸ್ತಕ), the MR_Preview URL, and the health /
    # education cess columns (ಅರೋಗ್ಯ ಕರ, ವಿದ್ಯಾ ಕರ) that an RTC never has.
    # -------------------------------

    strong_patta = (
        "mr_preview" in text_lower
        or "service64" in text_lower
        or "ಪಟ್ಟಾಪುಸ್ತಕ" in text_compact
        or "ಖಾತೆಪ್ರತಿ" in text_compact
    )

    patta_supporting_signals = sum(
        1
        for token in (
            "ಖಾತೆನಂಬರ್",
            "ಸ್ವಾದೀನದಾರರ",
            "ಅರೋಗ್ಯಕರ",
            "ವಿದ್ಯಾಕರ",
            "ಸ್ಥಳೀಯಕರ",
            "ಮೊಕ್ಕಾಂ",
        )
        if token in text_compact
    )

    if strong_patta or patta_supporting_signals >= 2:
        return {
            "classification": "PATTA_BOOK",
            "confidence": 0.95,
            "candidates": [
                {
                    "type": "PATTA_BOOK",
                    "confidence": 0.95
                }
            ]
        }

    # -------------------------------
    # 3. Strong Karnataka RTC Detection
    # -------------------------------
    # The first three anchors identify an RTC DOWNLOADED from Bhoomi — they are
    # the site's own URL, its preview path and the form title. A printed or
    # scanned RTC carries none of them: there is no URL on the page, and the
    # form title goes through OCR ("Village Account Form No. 2" came back
    # "AccULlil FDrq11 M, 2" on a Yalahanka RTC). That document reached the
    # Aadhaar block below and was classified AADHAAR_CARD at 0.97 off its
    # Bhoomi Land ID, which is printed in the same 4-4-4 shape as an Aadhaar
    # number. So match the tokens that survive a scan too: the digital-signature
    # footer, the validity line, the Latin abbreviation, and ಪಹಣಿ — the Kannada
    # name of this document, which a Patta / Mutation Register / KYC page never
    # carries (and both of those are decided above this block anyway).
    text_compact_lower = re.sub(r"\s+", "", text_lower)

    if (
        looks_like_rtc(text)
        or "previewrtc" in text_lower
        or "landrecords.karnataka.gov.in" in text_lower
        or "village account form no. 2" in text_lower
        or "rtcdigitallysigned" in text_compact_lower
        or "totilldate" in text_compact_lower
        or "r.t.c" in text_lower
        or "ಪಹಣಿ" in text_compact
    ):
        return {
            "classification": "RTC",
            "confidence": 0.95,
            "candidates": [
                {
                    "type": "RTC",
                    "confidence": 0.95
                }
            ]
        }

    # -------------------------------
    # 4. Strong Aadhaar Card Detection
    # -------------------------------
    # A bare 4-4-4 digit group USED TO BE sufficient on its own. It is not
    # evidence: a Karnataka RTC prints its Bhoomi Land ID in exactly that shape
    # ("ಭೂಮಿ ಲ್ಯಾಂಡ್ ಐಡಿ: 6128 8500 0486"), and on the Yalahanka RTC that number
    # was the ONLY Aadhaar signal in 13,000 characters of text — no UIDAI, no
    # AADHAAR, not even MALE or ADDRESS — yet it returned AADHAAR_CARD at 0.97.
    # The number now has to be corroborated by one of the named signals below,
    # every one of which is still sufficient by itself; the real cards keep
    # classifying on those (C NAGARAJU has no legible "AADHAAR" anywhere, but
    # does print MALE).
    if (
        "UNIQUE IDENTIFICATION AUTHORITY" in text_upper
        or "UNIQUE IDENTIFICATION" in text_upper
        or "UIDAI" in text_upper
        or "AADHAAR" in text_upper
        or "AADHAR" in text_upper
        or "GOVERNMENT OF INDIA" in text_upper
        or "YEAR OF BIRTH" in text_upper
        or "DATE OF BIRTH" in text_upper
        or "MALE" in text_upper
        or "FEMALE" in text_upper
        or "ADDRESS" in text_upper
    ):
        return {
            "classification": "AADHAAR_CARD",
            "confidence": 0.97,
            "candidates": [
                {
                    "type": "AADHAAR_CARD",
                    "confidence": 0.97
                }
            ]
        }

    # -------------------------------
    # 5. Keyword Score Classification
    # -------------------------------
    scores = {}

    for document_type, keywords in DOCUMENT_KEYWORDS.items():
        matched = 0

        for keyword in keywords:
            if keyword in text_lower:
                matched += 1

        confidence = matched / len(keywords)

        if matched > 0:
            scores[document_type] = round(confidence, 2)

    if not scores:
        return {
            "classification": "Unclassified",
            "confidence": 0.0,
            "candidates": []
        }

    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_type, top_confidence = sorted_scores[0]

    candidates = [
        {
            "type": document_type,
            "confidence": confidence
        }
        for document_type, confidence in sorted_scores[:3]
    ]

    classification = (
        top_type
        if top_confidence >= 0.30
        else "Unclassified"
    )

    return {
        "classification": classification,
        "confidence": top_confidence,
        "candidates": candidates
    }