"""
Offline checks for the RTC Gemini extractor's non-network logic.

Every case here is a real failure mode observed on landrecords.karnataka.gov.in
RTC PDFs with the previous Tesseract cell-OCR pipeline — printed form headings
emitted as values, "ಕೋಂ" recognised as "BOLO", and blank sections filled with
plausible-looking guesses. Run directly:  python tests/test_rtc_gemini_extractor.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import app.services.rtc.rtc_gemini_extractor as R

fails = []
def check(name, got, want):
    ok = got == want
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        print("   got :", repr(got))
        print("   want:", repr(want))
        fails.append(name)

# 1. Label-copying guard: the exact values the Tesseract mapper wrongly emitted.
d = R._normalize({
    "2_hissa": {"hissa_number": "\u0cb9\u0cbf\u0cb8\u0ccd\u0cb8\u0cbe"},
    "5_soil_type": {"soil_type": "\u0c92\u0c9f\u0ccd\u0c9f\u0cc1"},
    "7_tree_details": {"tree_count": "\u0cae\u0cb0\u0c97\u0cb3 \u0cb8\u0c82\u0c96\u0ccd\u0caf\u0cc6"},
})
stripped = R._strip_form_labels(d)
check("hissa form-label blanked", d["2_hissa"]["hissa_number"], None)
check("soil form-label blanked", d["5_soil_type"]["soil_type"], None)
check("tree form-label blanked", d["7_tree_details"]["tree_count"], None)
check("stripped list reported", sorted(stripped),
      ["2_hissa.hissa_number", "5_soil_type.soil_type", "7_tree_details.tree_count"])

# A genuine value must survive the same guard.
d2 = R._normalize({"2_hissa": {"hissa_number": "3"},
                   "5_soil_type": {"soil_type": "\u0c95\u0caa\u0ccd\u0caa\u0cc1"}})
R._strip_form_labels(d2)
check("real hissa survives", d2["2_hissa"]["hissa_number"], "3")
check("real soil value survives", d2["5_soil_type"]["soil_type"], "\u0c95\u0caa\u0ccd\u0caa\u0cc1")

# 1b. REGRESSION: the label guard must be FIELD-SCOPED. ಮುಂಗಾರು / ಹಿಂಗಾರು head
# block 8's sub-columns but are the real values of block 12's `season`. A global
# label set blanked all six seasons on an otherwise-correct live read.
d1b = R._normalize({
    "12_cultivation_details": [
        {"crop_year": "2025-2026", "season": "ಮುಂಗಾರು", "cultivator_name": "ಷಕೀಲಾಜಾನ್"},
        {"crop_year": "2025-2026", "season": "ಹಿಂಗಾರು", "cultivator_name": "ಷಕೀಲಾಜಾನ್"},
    ],
    "8_irrigation_details": {"kharif_extent": "ಮುಂಗಾರು"},
})
stripped_1b = R._strip_form_labels(d1b)
check("season kharif kept in block 12",
      d1b["12_cultivation_details"][0]["season"], "ಮುಂಗಾರು")
check("season rabi kept in block 12",
      d1b["12_cultivation_details"][1]["season"], "ಹಿಂಗಾರು")
check("same word blanked as a block 8 heading",
      d1b["8_irrigation_details"]["kharif_extent"], None)
check("only the block 8 field reported",
      stripped_1b, ["8_irrigation_details.kharif_extent"])

# Possession kinds are values, not headings, and must survive.
d1c = R._normalize({"10_possession_details": {
    "possession_type": ["ಪೌತಿ", "ಮೇಲಿನ ಜಂಟಿ", "ಸ್ವಂತ"]}})
check("possession kinds survive the guard",
      d1c["10_possession_details"]["possession_type"],
      ["ಪೌತಿ", "ಮೇಲಿನ ಜಂಟಿ", "ಸ್ವಂತ"])

# 2. Latin-in-Kannada corruption detector (the BOLO / BWOlO signature).
check("detects BOLO corruption",
      R._has_latin_in_kannada_field("\u0cb6\u0c95\u0cc0\u0cb2 \u0c9c\u0cbe\u0ca8\u0ccd BOLO \u0cb2\u0cc7"), True)
check("clean Kannada name passes",
      R._has_latin_in_kannada_field("\u0cb6\u0c95\u0cc0\u0cb2 \u0c9c\u0cbe\u0ca8\u0ccd \u0c95\u0ccb\u0c82 \u0cb2\u0cc7"), False)
check("pure Latin passes", R._has_latin_in_kannada_field("MR H17/2023-2024"), False)

# 3. Blank-section handling: null stand-ins and all-null template rows.
d3 = R._normalize({
    "header": {"taluk": "N/A", "village": "  "},
    "9_ownership_details": [{"owner_name": None, "extent": None}],
    "12_cultivation_details": [],
    "8_irrigation_details": {"source": "null"},
})
check("N/A -> None", d3["header"]["taluk"], None)
check("blank -> None", d3["header"]["village"], None)
check("'null' string -> None", d3["8_irrigation_details"]["source"], None)
check("all-null owner template dropped", d3["9_ownership_details"], [])

# 4. Missing blocks are filled to the pipeline contract.
d4 = R._normalize({})
check("all 13 blocks + header present",
      sorted(d4.keys()) == sorted(R._EMPTY_BLOCKS.keys()), True)
check("court_stay is a bool", d4["11_rights_liabilities"]["court_stay"], False)
check("tree_types is a list", d4["7_tree_details"]["tree_types"], [])
check("possession list default", d4["10_possession_details"]["owner_wise_possession"], [])

# 5. possession_type normalises to a de-duplicated list, labels excluded.
d5 = R._normalize({"10_possession_details": {
    "possession_type": ["\u0caa\u0ccc\u0ca4\u0cbf", "\u0caa\u0ccc\u0ca4\u0cbf", "\u0c92\u0c9f\u0ccd\u0c9f\u0cc1"]}})
check("possession deduped, label dropped",
      d5["10_possession_details"]["possession_type"], ["\u0caa\u0ccc\u0ca4\u0cbf"])
d5b = R._normalize({"10_possession_details": {"possession_type": "\u0cb8\u0ccd\u0cb5\u0c82\u0ca4"}})
check("scalar possession -> list",
      d5b["10_possession_details"]["possession_type"], ["\u0cb8\u0ccd\u0cb5\u0c82\u0ca4"])

# 5b. Comma-joined possession kinds are split into a real list.
d5c = R._normalize({"10_possession_details": {
    "possession_type": ["ಪೌತಿ, ಮೇಲಿನ ಜಂಟಿ"]}})
check("comma-joined kinds split",
      d5c["10_possession_details"]["possession_type"],
      ["ಪೌತಿ", "ಮೇಲಿನ ಜಂಟಿ"])

# 5c. REGRESSION: ಕಪ್ಪು / ಸರ್ಕಾರಿ are real black-ink VALUES on this form
# (verified against the 300 DPI render: the blank form is pre-printed blue,
# entries are black). They must never be treated as headings.
d5d = R._normalize({"5_soil_type": {"soil_type": "ಕಪ್ಪು"},
                    "6_land_classification": {"land_classification": "ಸರ್ಕಾರಿ"}})
stripped_5d = R._strip_form_labels(d5d)
check("soil value kept", d5d["5_soil_type"]["soil_type"], "ಕಪ್ಪು")
check("tenure value kept", d5d["6_land_classification"]["land_classification"], "ಸರ್ಕಾರಿ")
check("nothing stripped from 5/6", stripped_5d, [])

# 6. Cross-model diff catches the disagreements that matter.
a = R._normalize({"2_hissa": {"hissa_number": "3"}, "header": {"village": "\u0c95\u0cbe\u0ca1\u0cc1\u0c97\u0ccb\u0ca1\u0cbf"},
                  "9_ownership_details": [{"owner_name": "\u0cb6\u0c95\u0cc0\u0cb2 \u0c9c\u0cbe\u0ca8\u0ccd"}]})
b = R._normalize({"2_hissa": {"hissa_number": "5"}, "header": {"village": "\u0cb8\u0cbe\u0ca1\u0cc1\u0c97\u0ccb\u0ca1\u0cbf"},
                  "9_ownership_details": [{"owner_name": "\u0cb6\u0c95\u0cc0\u0cb2 \u0c9c\u0cbe\u0ca8\u0ccd"}]})
diff = R._diff_reads(a, b)
fields = sorted(x["field"] for x in diff["scalar_disagreements"])
check("diff flags hissa + village", fields, ["2_hissa.hissa_number", "header.village"])
check("identical owner not flagged", diff["owner_disagreements"], [])

# Folding ignores joiners/punctuation, not real characters.
c1 = R._normalize({"9_ownership_details": [{"owner_name": "\u0cb6\u0c95\u0cc0\u0cb2 \u0c9c\u0cbe\u0ca8\u0ccd\u200c"}]})
c2 = R._normalize({"9_ownership_details": [{"owner_name": "\u0cb6\u0c95\u0cc0\u0cb2\u0c9c\u0cbe\u0ca8\u0ccd"}]})
check("ZWNJ/space differences fold equal",
      R._diff_reads(c1, c2)["owner_disagreements"], [])

# 7. Scoring prefers the fuller read and penalises reintroduced corruption.
clean = R._normalize({"header": {"village": "\u0c95\u0cbe\u0ca1\u0cc1\u0c97\u0ccb\u0ca1\u0cbf"},
                      "9_ownership_details": [{"owner_name": "\u0cb6\u0c95\u0cc0\u0cb2 \u0c9c\u0cbe\u0ca8\u0ccd \u0c95\u0ccb\u0c82 \u0cb2\u0cc7"}]})
dirty = R._normalize({"header": {"village": "\u0c95\u0cbe\u0ca1\u0cc1\u0c97\u0ccb\u0ca1\u0cbf"},
                      "9_ownership_details": [{"owner_name": "\u0cb6\u0c95\u0cc0\u0cb2 \u0c9c\u0cbe\u0ca8\u0ccd BOLO \u0cb2\u0cc7"}]})
check("clean read outscores corrupted", R._score(clean) > R._score(dirty), True)
empty = R._normalize({})
check("fuller read outscores empty", R._score(clean) > R._score(empty), True)

# 7b. Error classification decides whether falling to the next model is worth
# anything. A project spend cap is shared by EVERY model on the key, so the
# chain must abort; a per-model daily quota must not abort it.
CAP = "429 RESOURCE_EXHAUSTED. Your project has exceeded its monthly spending cap."
DAILY = "429 RESOURCE_EXHAUSTED. Quota exceeded for quota metric 'Generate requests per day'."
GONE = "404 NOT_FOUND. This model models/gemini-2.5-flash-lite is no longer available."

check("spend cap detected", R._is_spend_cap_error(CAP), True)
check("per-model quota is NOT a spend cap", R._is_spend_cap_error(DAILY), False)
check("404 is not a spend cap", R._is_spend_cap_error(GONE), False)
check("spend cap still counts as quota", R._is_quota_error(CAP), True)
check("retired model detected as 404", R._is_missing_model_error(GONE), True)

# The shipped model list must contain no name known to be retired on this key.
check("no retired 2.5 models in chain",
      [m for m in R.RTC_MODELS if m.startswith("gemini-2.5")], [])
check("chain has fallback depth", len(R.RTC_MODELS) >= 3, True)

# A spend-cap failure still returns the full block contract, not a bare error.
cap_result = R._unavailable_result(Exception(CAP))
check("unavailable keeps all blocks",
      all(k in cap_result for k in R._EMPTY_BLOCKS), True)
check("unavailable is flagged for review",
      cap_result["extraction_status"], "requires_manual_review")
check("unavailable source tag",
      cap_result["extraction_source"], "gemini_vision_rtc_v1:unavailable")

# 8. JSON extraction tolerates fenced / prose-wrapped model output.
check("fenced JSON parsed", R._extract_json('```json\n{"a":1}\n```'), {"a": 1})
check("prose-wrapped JSON parsed", R._extract_json('Here you go: {"a":2} done'), {"a": 2})
check("garbage -> {}", R._extract_json("no json here"), {})

print()
print("ALL CHECKS PASSED" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
