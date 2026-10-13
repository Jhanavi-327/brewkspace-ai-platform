from email.mime import text
import re

from sqlalchemy import text
from typing import Any, Dict, List, Optional

def kannada_digits_to_english(text: str) -> str:
    if text is None:
        return ""

    return (
        str(text)
        .replace("೦", "0")
        .replace("೧", "1")
        .replace("೨", "2")
        .replace("೩", "3")
        .replace("೪", "4")
        .replace("೫", "5")
        .replace("೬", "6")
        .replace("೭", "7")
        .replace("೮", "8")
        .replace("೯", "9")
    )

def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def all_text(cells_with_text: List[Dict[str, Any]]) -> str:
    return "\n".join(
        clean_text(cell.get("text", ""))
        for cell in cells_with_text
        if clean_text(cell.get("text", ""))
    )


def find_first(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE | re.UNICODE | re.DOTALL)
    return clean_text(match.group(1)) if match else None




def find_extents(text: str) -> List[str]:
    return re.findall(r"\d+\.\d+\.\d+\.\d+", text)


def find_decimal_values(text: str) -> List[str]:
    return re.findall(r"\d+\.\d+", text)


def find_mr_number(text: str) -> Optional[str]:
    """
    First MR reference whose year range is a real one.

    A mutation year is an accounting year, so the two halves are always
    consecutive: 2004-2005, 2017-2018, 2019-2020. Taking the first match
    outright reported "MR 17/2015-2620" as the mutation number of the Yalahanka
    survey — 2620 is not a year, it is Tesseract reading 2020 through the
    "For Viewing Only" watermark, and the page carries eleven MR references for
    it to pick from. Requiring the years to run consecutively costs a genuine
    reference nothing and drops the impossible ones.
    """
    for match in re.finditer(
        r"MR\s*[A-Z]?\d+\/(\d{4})-(\d{4})",
        text or ""
    ):
        start_year = int(match.group(1))
        end_year = int(match.group(2))

        if end_year == start_year + 1:
            return clean_text(match.group(0))

    return None


def find_court_reference(text: str) -> Optional[str]:
    return (
        find_first(r"(noRA\/BET\/\d+\/\d+)", text)
        or find_first(r"(NORA\/BET\/\d+\/\d+)", text)
        or find_first(r"(RRT\/[A-Z0-9\/.\-]+)", text)
        or find_first(r"(NA\.CR\.[0-9\/\-]+)", text)
    )


def extract_header_from_fallback(
    fallback: Dict[str, Any],
    raw_text: Optional[str] = None
) -> Dict[str, Any]:

    header = fallback.get("header", {}) if fallback else {}

    if raw_text:

        # Prefer EasyOCR section only
        easy_text = raw_text

        if "--- SELECTED OCR ENGINE: EASYOCR ---" in raw_text:
            easy_text = raw_text.split(
                "--- SELECTED OCR ENGINE: EASYOCR ---",
                1
            )[1]

        if "--- TESSERACT OCR ---" in easy_text:
            easy_text = easy_text.split(
                "--- TESSERACT OCR ---",
                1
            )[0]

        lines = [
            clean_text(line)
            for line in easy_text.splitlines()
            if clean_text(line)
        ]

        for i, line in enumerate(lines):

            # Taluk may be same line or next line
            if "ತಾಲ್ಲೂ" in line:
                value = re.sub(
                    r".*ತಾಲ್ಲೂ\s*ಕು\s*[:：]?",
                    "",
                    line
                )
                value = clean_text(value)

                if not value and i + 1 < len(lines):
                    value = clean_text(lines[i + 1])

                if value and "ಮೊಹರು" not in value:
                    header["taluk"] = value

            # Hobli
            if "ಹೋಬಲಿ" in line or "ಹೋಬಳಿ" in line:
                value = re.sub(
                    r".*ಹೋಬ[ಲಳ]ಿ\s*[:：]?",
                    "",
                    line
                )
                value = clean_text(value)

                if not value and i + 1 < len(lines):
                    value = clean_text(lines[i + 1])

                if value:
                    header["hobli"] = value

            # Village
            if "ಗ್ರಾಮ" in line:
                value = re.sub(
                    r".*ಗ್ರಾಮ\s*[:：]?",
                    "",
                    line
                )
                value = clean_text(value)

                if not value and i + 1 < len(lines):
                    value = clean_text(lines[i + 1])

                if value and "ಪುಟ" not in value:
                    header["village"] = value

    return {
        "taluk": header.get("taluk"),
        "hobli": header.get("hobli"),
        "village": header.get("village"),
        "valid_from": header.get("valid_from"),
        "valid_to": header.get("valid_to"),
    }



def extract_header_from_cells(
    cells_with_text,
    fallback=None,
    raw_text=None,
    header_strip_text=None
):
    
    print("HEADER RAW TEXT TYPE:", type(raw_text))
    print("HEADER RAW TEXT LENGTH:", len(raw_text) if raw_text else 0)


    fallback_header = fallback.get("header", {}) if fallback else {}

    header = {
        "taluk": None,
        "hobli": None,
        "village": None,
        "valid_from": fallback_header.get("valid_from"),
        "valid_to": fallback_header.get("valid_to"),
    }

    source = raw_text or ""

    if "--- SELECTED OCR ENGINE: EASYOCR ---" in source:
        source = source.split(
            "--- SELECTED OCR ENGINE: EASYOCR ---",
            1
        )[1]

    if "--- TESSERACT OCR ---" in source:
        source = source.split(
            "--- TESSERACT OCR ---",
            1
        )[0]

    lines = [
        clean_text(line)
        for line in source.splitlines()
        if clean_text(line)
    ]

    for i, line in enumerate(lines):

        # Taluk: label may be previous line, value may contain Hobli after it
        if "ತಾಲ್ಲೂ" in line:
            value = re.sub(
                r".*ತಾಲ್ಲೂ\s*ಕು\s*[:：]?",
                "",
                line
            )
            value = clean_text(value)

            if not value and i + 1 < len(lines):
                value = clean_text(lines[i + 1])

            value = value.split("ಹೋಬಳಿ")[0]
            value = value.split("ಹೋಬಲಿ")[0]
            value = value.strip(" :：")
            value = clean_text(value)

            if value:
                header["taluk"] = value

        # Hobli: may be in same line as Taluk
        if "ಹೋಬಳಿ" in line or "ಹೋಬಲಿ" in line:
            value = re.sub(
                r".*ಹೋಬ[ಳಲ]ಿ\s*[:：]?",
                "",
                line
            )
            value = clean_text(value)

            if not value and i + 1 < len(lines):
                value = clean_text(lines[i + 1])

            value = value.split("ಗ್ರಾಮ")[0]
            value = value.split("ಪುಟ")[0]
            value = value.strip(" :：")
            value = clean_text(value)

            if value:
                header["hobli"] = value

        # Village: value may be on next line
        if "ಗ್ರಾಮ" in line:
            value = re.sub(
                r".*ಗ್ರಾಮ\s*[:：]?",
                "",
                line
            )
            value = clean_text(value)

            if not value and i + 1 < len(lines):
                value = clean_text(lines[i + 1])

            value = value.split("ಪುಟ")[0]
            value = value.strip(" :：")
            value = clean_text(value)

            bad_village_words = [
                # "ಗ್ರಾಮ ನಮೂನೆ 2" (Village Form No. 2) is printed in the corner
                # of every RTC, so the village label matches there first. OCR
                # drops the trailing vowel often enough that only spelling it
                # both ways catches it — "ನಮೂನ ೨" was reported as the village.
                "ನಮೂನ",
                "ನಮೂನೆ",
                "Valid",
                "from",
                "To",
                "ತಾಲ್ಲೂ",
                "ಹೋಬಳಿ",
                "ಹೋಬಲಿ",
                "ಫಾರಂ",
                "ಹಿಸ್ಸಾ",
                "ಪಟ್ಟಾ",
                "ಅಥವಾ",
                "ಸ್ವಾ",
            ]

            if value and not any(word in value for word in bad_village_words):
                header["village"] = value

    # The taluk/hobli/village line sits above the table border, so it is in no
    # cell and only whole-page OCR can see it — and on a watermarked scan that
    # pass loses the labels entirely. A crop of just that strip reads them when
    # the full page cannot. It is applied ONLY to fields still missing, so a
    # value the page text did produce is never replaced: the page read gives
    # RTC 59-P4 its taluk correctly, while the strip is what gives the
    # Yalahanka RTC a village at all.
    if header_strip_text and not header["village"]:
        # Only the VILLAGE is taken from the strip. Its label survives the crop
        # intact and is followed by a colon, so it can be cut out exactly. The
        # taluk and hobli labels do not survive on a watermarked scan — the
        # Yalahanka strip reads "ಹೆ.ಡೀ!2ಳಿ" for ಹೋಬಳಿ — which leaves nothing to
        # anchor to and yields the whole line as the value. A tangled taluk is
        # worse than the null the page pass already gives.
        #
        # The colon is required: "ಗ್ರಾಮ ನಮೂನೆ 2" (Village Form No. 2) is printed
        # in the corner and would otherwise match first.
        village_match = re.search(
            r"ಗ್ರಾಮ[^\s:：]*\s*[:：]\s*([^|\n]+)",
            header_strip_text
        )

        if village_match:
            value = village_match.group(1)
            value = re.split(r"ಪುಟ|ನಮೂನ", value)[0]
            value = clean_text(value).strip(" :：|.,-")

            if value and len(value) > 2:
                header["village"] = value

    return header


def extract_survey_number(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Optional[str]:

    normalized_text = kannada_digits_to_english(text)

    # The label is printed with a separator after it — "1. ಸರ್ವೆ ನಂಬರು: 59" —
    # and ":" is in neither \s* nor the value character class, so the pattern
    # could not step over it. The cell branch therefore NEVER matched on an RTC
    # that prints the colon (the common case), and every such document fell
    # through to the page-wide searches below, which scan the whole sheet and
    # can return a number belonging to another block entirely. Accept the
    # separator, and ಸಂಖ್ಯೆ as a label variant alongside ನಂಬರು.
    # ಸಂಬರು / ಸಂಬರ್ are ನಂಬರು with its ನ read as ಸ — a routine confusion between
    # two similar Kannada letters, and common enough to cost real values: the
    # Vajarahalli RTC prints its survey number 72 perfectly legibly, but the
    # label came back "ಸಂಬರು" and the pattern could not reach past it.
    label_value_re = re.compile(
        r"(?:ಸರ್ವೆ|ಸರ್ವ|ನಂಬರು|ನಂಬರ್|ಸಂಬರು|ಸಂಬರ್|ಸಂಖ್ಯೆ|ಸಂಖ್ಯ)\s*[:;.\-~]*\s*"
        r"([A-Za-z0-9\/\-\*]+)"
    )

    # 1. Cell-based: look for cell containing "1. ಸರ್ವೆ"
    for cell in cells_with_text:
        raw_cell_text = clean_text(cell.get("text", ""))

        # Digit translation is one character for one character, so a span in
        # the converted string covers the same span in the original — which is
        # what lets the script check below look at what was actually printed.
        cell_text = kannada_digits_to_english(raw_cell_text)

        if "ಸರ್ವೆ" in cell_text or "ಸರ್ವ" in cell_text:
            for match in label_value_re.finditer(cell_text):
                value = clean_text(match.group(1))

                printed = raw_cell_text[match.start(1):match.end(1)]

                # A survey number is printed in ONE script. A value mixing
                # Kannada and Latin digits was not read, it was guessed at:
                # RTC 59-P4's number 59 came back "೨9", a Kannada 2 beside a
                # Latin 9. Converting that yields a perfectly plausible "29"
                # for a parcel that is actually 59, so the mixture itself has
                # to be the signal — there is nothing else to catch it by.
                has_kannada_digit = any("\u0CE6" <= c <= "\u0CEF" for c in printed)
                has_latin_digit = any(c.isdigit() and c.isascii() for c in printed)

                if has_kannada_digit and has_latin_digit:
                    continue

                # A survey number always carries a digit. Without one the cell
                # was misread rather than read — Tesseract returned "og" for
                # the "59" on a Yalahanka RTC — and a letter blob must not be
                # reported as a parcel identifier.
                if any(character.isdigit() for character in value):
                    return value

            # This cell IS the survey-number box; it was simply not legible.
            # Stop here rather than falling through to the page-wide searches
            # below: they read the entire sheet and will return a number from
            # some other block, which is worse than reporting nothing. On the
            # Yalahanka RTC they produced "153" — the total extent 163.00.00.00
            # misread — for a parcel that is actually 59. A null is reviewable;
            # a plausible wrong parcel number is not. When no labelled cell is
            # found at all (grid not detected) the fallbacks still apply.
            return None

    # 2. OCR fallback. The page text this receives is the concatenation of the
    # same cells, so it re-reads the survey box among everything else and must
    # apply the SAME digit test — otherwise it hands back exactly the unreadable
    # value the cell loop just rejected ("og" for the 59 on a Yalahanka RTC).
    for match in re.finditer(
        r"1\s*[\.,]?\s*(?:ಸರ್ವೆ|ಸರ್ವ)\s*(?:ನಂಬರು|ನಂಬರ್|ಸಂಖ್ಯೆ)?"
        r"\s*[:;.\-~]*\s*([A-Za-z0-9\/\-\*]+)",
        normalized_text
    ):
        value = clean_text(match.group(1))

        if any(character.isdigit() for character in value):
            return value

    return fallback.get("1_survey_number", {}).get("survey_number")


def find_nearest_value_after_label(
    cells_with_text: List[Dict[str, Any]],
    label_keywords: List[str],
    reject_keywords: Optional[List[str]] = None
) -> Optional[str]:

    reject_keywords = reject_keywords or []

    label_cells = []

    for cell in cells_with_text:
        text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if not text:
            continue

        if all(keyword in text for keyword in label_keywords):
            label_cells.append(cell)

    if not label_cells:
        return None

    label_cell = sorted(
        label_cells,
        key=lambda c: (
            c.get("y", 0),
            c.get("x", 0)
        )
    )[0]

    lx = label_cell.get("x", 0)
    ly = label_cell.get("y", 0)
    lx2 = label_cell.get(
        "x2",
        lx + label_cell.get("width", 0)
    )

    candidates = []

    for cell in cells_with_text:
        if cell is label_cell:
            continue

        text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if not text:
            continue

        if any(word in text for word in reject_keywords):
            continue

        x = cell.get("x", 0)
        y = cell.get("y", 0)

        same_row_right = (
            abs(y - ly) < 120
            and x > lx2
        )

        below_same_area = (
            y > ly
            and y < ly + 450
            and abs(x - lx) < 500
        )

        if same_row_right or below_same_area:
            distance = abs(x - lx) + abs(y - ly)

            candidates.append({
                "text": text,
                "distance": distance
            })

    candidates.sort(
        key=lambda item: item["distance"]
    )

    for item in candidates:
        text = item["text"]

        match = re.search(
            r"\b([A-Za-z0-9\/\-]{1,10})\b",
            text
        )

        if match:
            return match.group(1)

    return None

def _plausible_parcel_token(value: str) -> bool:
    """
    Is this a survey / hissa value rather than an OCR smear?

    Both fields are numeric with optional suffixes ("59", "12/3-A", "2A") and
    are printed "*" or "-" when the parcel has no subdivision. What they are
    never made of is letters alone: "Bow" (for a blank hissa) and "og" (for the
    survey number 59) are the shapes Tesseract produces when it reads a box it
    cannot resolve, and reporting either as a parcel identifier is worse than
    reporting nothing, because nothing is reviewable and a plausible number is
    silently acted on.
    """
    if not value:
        return False

    if any(character.isdigit() for character in value):
        return True

    return value in {"*", "-"}


def extract_hissa(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Optional[str]:

    normalized_text = kannada_digits_to_english(text)

    # 1. Strong pattern for merged OCR:
    # captures value between "2. ಹಿಸ್ಸಾ / ಹಿಸಾ" and "5. ಮಣ್ಣಿನ"
    match = re.search(
        r"2\s*[\.,]?\s*(?:ಹಿಸ್ಸಾ|ಹಿಸಾ)?\s*[:;.\-~]*\s*[\n\s]*"
        r"([A-Za-z0-9\/\-\*]+)\s*[\n\s]*5\s*[\.,]?\s*ಮಣ್ಣ",
        normalized_text,
        re.UNICODE
    )

    if match:
        value = clean_text(match.group(1))
        if value and value not in ["5", "7", "8"] and _plausible_parcel_token(value):
            return value

    # 2. Cell-based: look for cell containing hissa
    for cell in cells_with_text:
        cell_text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if "ಹಿಸ್ಸಾ" not in cell_text and "ಹಿಸಾ" not in cell_text:
            continue

        # cut off next field labels if OCR merged multiple blocks
        cell_text = re.split(
            r"5\s*[\.,]?\s*ಮಣ್ಣ|7\s*[\.,]?\s*ಮರ|8\s*[\.,]?\s*ಖೇತು",
            cell_text
        )[0]

        # The label is printed "2. ಹಿಸ್ಸಾ: 1"; without the separator here the
        # pattern could not reach the value and every colon-printing RTC fell
        # through to the looser searches below.
        for match in re.finditer(
            r"(?:ಹಿಸ್ಸಾ|ಹಿಸಾ|ಹಿಸ್ಸ)\s*[:;.\-~]*\s*([A-Za-z0-9\/\-\*]+)",
            cell_text
        ):
            value = clean_text(match.group(1))
            if value and value not in ["5", "7", "8"] and _plausible_parcel_token(value):
                return value

        # Label found, value unreadable — do not guess from elsewhere.
        return None

    # 3. Layout helper fallback
    value = find_nearest_value_after_label(
        cells_with_text,
        label_keywords=["ಹಿಸ"],
        reject_keywords=[
            "ಮಣ್ಣಿನ",
            "ಮರಗಳ",
            "ಖೇತು",
            "ನೀರಾವರಿ",
            "ಕಂದಾಯ",
            "ವಿಸ್ತೀರ್ಣ",
        ]
    )

    if value and value not in ["5", "7", "8"]:
        return value

    # 4. Final fallback
    fallback_value = fallback.get("2_hissa", {}).get("hissa_number")

    if fallback_value in ["5", "7", "8"]:
        return None

    return fallback_value

def extract_khata_vara(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    extents = find_extents(text)
    old = fallback.get("3_khata_vara", {}) if fallback else {}

    total_extent = old.get("total_extent")
    phut_kharab_a = old.get("phut_kharab_a")
    phut_kharab_b = old.get("phut_kharab_b")
    remaining_extent = old.get("remaining_extent")

    if extents:
        non_zero = [e for e in extents if not e.startswith("0.00")]
        total_extent = non_zero[0] if non_zero else extents[0]

    phut_kharab_a = (
        find_first(r"ಪೂಟ್\s*ಖರಾಬ್\s*\(ಅ\)\s*(\d+\.\d+\.\d+\.\d+)", text)
        or find_first(r"ಖರಾಬ್\s*\(ಅ\).*?(\d+\.\d+\.\d+\.\d+)", text)
        or phut_kharab_a
    )

    phut_kharab_b = (
        find_first(r"ಪೂಟ್\s*ಖರಾಬ್\s*\(ಬ\)\s*(\d+\.\d+\.\d+\.\d+|\d+\.\d+)", text)
        or find_first(r"ಖರಾಬ್\s*\(ಬ\).*?(\d+\.\d+\.\d+\.\d+|\d+\.\d+)", text)
        or phut_kharab_b
    )

    remaining_extent = (
        find_first(r"ಉಳಿದದ್ದು\s*(\d+\.\d+\.\d+\.\d+)", text)
        or remaining_extent
    )

    # What is LEFT of a survey cannot exceed the whole of it. A stray mark read
    # as a digit turns 2.17.00.00 into 12.17.00.00 — a plausible-looking extent
    # that is ten times the parcel, on a sheet whose total is 3.04.00.00. The
    # comparison is on the printed acre.gunta groups, which order correctly as
    # integers without needing to know the units.
    def extent_key(value):
        try:
            return tuple(int(part) for part in value.split("."))
        except (AttributeError, ValueError):
            return None

    remaining_key = extent_key(remaining_extent)
    total_key = extent_key(total_extent)

    if remaining_key and total_key and remaining_key > total_key:
        remaining_extent = None

    return {
        "total_extent": total_extent,
        "phut_kharab_a": phut_kharab_a,
        "phut_kharab_b": phut_kharab_b,
        "remaining_extent": remaining_extent,
    }


def extract_land_revenue(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:
    old = fallback.get("4_land_revenue", {}) if fallback else {}

    # This used to look for the literal "5.46" — the land revenue of ONE
    # sample RTC — so the cell branch was dead for every other document and
    # block 4 always came from the page-level regex fallback, which on RTC
    # 59-P4 reported the extent 4.00 as the land revenue.
    #
    # Identify the cell STRUCTURALLY instead. Block 4's value column is the
    # only numbers-only cell on the sheet carrying five decimals — one per
    # printed row: land revenue, jodi, cess, water rate, total. The extent
    # cells nearby are not confusable: they hold dotted quads like
    # "4.00.00.00", which this pattern reads as only two decimals each, and
    # they carry their Kannada labels in the same cell.
    # Block 4 is printed in the header band at the top of the sheet, so only
    # the upper part of the page is searched. Without that bound the scan
    # reaches the cultivation table, whose extent column is also numbers-only
    # and also carries five decimals: on the Anekal RTC it matched a row of
    # zeros 64% down the page and reported a land revenue of 0.00 for a sheet
    # printing 4.54. The real block-4 cells sit at 18% and 27% on the other
    # documents, so the band is generous: block 4 was measured at 18% and 27%
    # down on the landscape sheets and 46% on the portrait A4 one, while the
    # cultivation table that produced the false match starts at 64%.
    page_height = max(
        (cell.get("y", 0) for cell in cells_with_text),
        default=0
    )
    header_band = page_height * 0.55 if page_height else None

    revenue_text = ""
    for cell in cells_with_text:
        cell_text = clean_text(cell.get("text", ""))

        if header_band and cell.get("y", 0) > header_band:
            continue

        if any("\u0C80" <= character <= "\u0CFF" for character in cell_text):
            continue

        # EXACTLY five: block 4 prints five rows (land revenue, jodi, cess,
        # water rate, total), so five is its signature. "At least five" is not
        # good enough — the cultivation table's extent column is also
        # numbers-only and produced fifteen decimals on the Yalahanka RTC,
        # winning the scan and reporting a crop extent as the cess.
        if len(find_decimal_values(cell_text)) == 5:
            revenue_text = cell_text
            break

    values = find_decimal_values(revenue_text)

    if len(values) >= 5:
        return {
            "land_revenue": values[0],
            "jodi": values[1],
            "cess": values[2],
            "water_rate": values[3],
            "total_revenue": values[4],
        }

    # Nothing matched by shape. That happens when the grid detector merges
    # block 4 into the extents column beside it — the Anekal sheet came back as
    # one cell reading "3.04.00.00 | (0) ಭೂಕಂದಾಯ | 4.54 0.27.00.00 (ಬ) ಜೋಡಿ
    # 0.00 ..." — where the printed row labels are the only thing separating
    # the two columns. Read the values off those labels instead.
    label_patterns = (
        ("land_revenue", r"ಭೂ\s*ಕಂದಾಯ"),
        ("jodi", r"ಜೋಡಿ"),
        ("cess", r"ಸೆಸ್ಸು|ಸಸ್ಸು|ಸೆಸ್ತು"),
        ("water_rate", r"ನೀರಿನ\s*ದರ|ನೀರಿಗೆ"),
        ("total_revenue", r"ಒಟ್ಟು"),
    )

    for cell in cells_with_text:
        cell_text = clean_text(cell.get("text", ""))

        if "ಕಂದಾಯ" not in cell_text:
            continue

        labelled = {}

        for field, label in label_patterns:
            # A short reach only, so a number belonging to the extents column
            # further along the merged line cannot be adopted as a revenue.
            match = re.search(
                rf"(?:{label})[^\d]{{0,12}}(\d+\.\d+)(?!\.)",
                cell_text
            )

            if match:
                labelled[field] = match.group(1)

        # The land revenue is the defining row: without it this is not block 4
        # and the reading is not worth preferring over what the fallback holds.
        if labelled.get("land_revenue"):
            return {
                field: labelled.get(field) or old.get(field)
                for field in (
                    "land_revenue", "jodi", "cess",
                    "water_rate", "total_revenue",
                )
            }

    return {
        "land_revenue": old.get("land_revenue"),
        "jodi": old.get("jodi"),
        "cess": old.get("cess"),
        "water_rate": old.get("water_rate"),
        "total_revenue": old.get("total_revenue"),
    }


def extract_soil_type(text: str) -> Dict[str, Any]:
    soil_keywords = ["ಕೆಂಪು", "ಕಪ್ಪು", "ಕೆಬ್ಬೆ", "ಮರಳು", "red", "black"]

    found = None
    for keyword in soil_keywords:
        if keyword.lower() in text.lower():
            found = keyword
            break

    return {
        "soil_type": found,
        "extraction_status": "extracted" if found else "requires_layout_or_manual_review"
    }


def extract_land_classification(
    text: str,
    raw_text: Optional[str] = None
) -> Dict[str, Any]:
    classification = None

    # "6. ಪಟ್ಟಾ" is block 6's printed HEADING and appears on every RTC, so a
    # bare `"ಪಟ್ಟಾ" in text` can never fail — whenever the real value was not
    # legible the heading was returned in its place. The Anekal RTC is
    # classified ಸರ್ಕಾರಿ and reported "ಪಟ್ಟಾ" for exactly that reason.
    # Removing the numbered heading first leaves only a genuine value behind.
    #
    # The cell text is consulted first and the whole-page text second: block 6
    # is a narrow box that the grid detector often swallows into a neighbour
    # (on the Anekal sheet its cell kept the heading and lost the value), while
    # the page pass read "6. ಪಟ್ಟಾ ಸರ್ಕಾರಿ" cleanly.
    for source in (text, raw_text):
        if not source:
            continue

        without_heading = re.sub(r"\d\s*[.,]?\s*ಪಟ್ಟಾ", " ", source)

        if "ಸರ್ಕಾರಿ" in without_heading:
            classification = "ಸರ್ಕಾರಿ"
            break

        if "ಪಟ್ಟಾ" in without_heading:
            classification = "ಪಟ್ಟಾ"
            break

    return {
        "land_classification": classification,
        "extraction_status": "extracted" if classification else "requires_layout_or_manual_review"
    }


def extract_tree_details(
    cells_with_text: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:

    tree_count = None

    for cell in cells_with_text:
        cell_text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        # Only inspect the actual Block 7 tree-count cell,
        # not nearby Block 8 heading.
        if "ಮರಗಳ" in cell_text and "ಸಂಖ್ಯೆ" in cell_text:
            match = re.search(
                r"ಮರಗಳ\s*ಸಂಖ್ಯೆ\s*[:\-]?\s*(\d+)",
                cell_text
            )

            if match:
                tree_count = match.group(1)

            break

    return {
        "tree_count": tree_count,
        "tree_types": [],
        "extraction_status": (
            "extracted"
            if tree_count
            else "requires_layout_or_manual_review"
        )
    }


def extract_irrigation_details(text: str) -> Dict[str, Any]:
    return {
        "source": None,
        "kharif_extent": None,
        "rabi_extent": None,
        "garden_extent": None,
        "total_extent": None,
        "extraction_status": "requires_layout_or_manual_review"
    }


def split_owner_name(raw_name: str) -> Dict[str, Any]:
    raw_text = clean_text(raw_name)
    cleaned = raw_text.replace("|", " ")

    cleaned = re.sub(r"\b[A-Za-z]{2,}\b", " ", cleaned)
    cleaned = re.sub(r"\d+\.\d+\.\d+\.\d+", " ", cleaned)
    cleaned = re.sub(r"\d+\.\d+", " ", cleaned)
    cleaned = normalize_repeated_words(cleaned)

    return {
        "owner_name": cleaned or None,
        "father_or_spouse_name": None,
        "address": None,
        "extent": None,
        "khata_no": None,
        "possession_type": (
            "ಮೇಲಿನ ಜಂಟಿ" if "ಮೇಲಿನ ಜಂಟಿ" in raw_text else None
        ),
        "raw_text": raw_text
    }



def has_kannada(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0C80-\u0CFF]",
            text
        )
    )


def extract_owner_details(
    cells_with_text: List[Dict[str, Any]],
    fallback: Dict[str, Any]
) -> List[Dict[str, Any]]:

    owners = []
    seen = set()

    ignore_words = [
        "Previous", "Next", "First", "Last",
        "ಸರ್ವೆ", "ಸಂಬರು", "ಖೇತವಾರು",
        "ವಿಸ್ತೀರ್ಣ", "ವಿನ್ನೀರ್ಣ", "ಖಾತೆ",
        "ಕಂದಾಯ", "ನೀರಾವರಿ", "ಮುಂಗಾರು",
        "ಹಿಂಗಾರು", "ಬಾಗಾಯ್ತು", "ಒಟ್ಟು",
        "ಹೆಸರು", "ಸಂಖ್ಯೆ", "ವರ್ಷ", "ಬೆಳೆ",
        "MR", "RRT", "ಖರಾಬ್", "ಗುತ್ತಿಗೆ",
        "ಅಮಿಶ್ರ", "ಮಿಶ್ರ", "ಹಕ್ಕುಗಳು",
        "ಋಣಗಳು", "ಖಯುಣ", "ಕಬ್ಬೆ",
        "ಸ್ವಾಥೀನ", "ಸ್ವಾಧೀನ", "ಕ್ರ ಸ",
        "ಮಣ್ಣಿನ", "ಮರಗಳ", "ರೂ", "ಪೈ",
        "ಎ ಗುಂ",
    ]

    def is_valid_owner_text(text: str) -> bool:
        text = clean_text(text)

        if not text or len(text) < 2:
            return False

        if not has_kannada(text):
            return False

        if any(word.lower() in text.lower() for word in ignore_words):
            return False

        # "12. ಸಾಗುವಳಿ...", "9. ಕಬ್ಜೆ...": every RTC block is titled with its
        # number, and those titles sit in the same column as the values. The
        # heading of block 12 was being reported as the sole owner of the
        # Yalahanka survey while the five real names sat in a cell below it.
        if re.match(r"^\s*\d{1,2}\s*[.,]", text):
            return False

        # Block 11 heads its two columns "ಹಕ್ಕುಗಳು:" (rights) and "ಋಣಗಳು:"
        # (liabilities), and they sit inside the ownership band. OCR mangles
        # them past the word list above — they came back "ಸಕ್ನುಗಳು;" and
        # "ಖುಣಗಳು:" and were reported as the two owners of the survey. The
        # punctuation is the giveaway that survives the mangling: a printed
        # label ends in a colon, a person's name does not.
        if text.rstrip().endswith((":", ";", "：")):
            return False

        if re.fullmatch(r"[\d\.\-/\s|\\_]+", text):
            return False

        if re.search(r"\d+\.\d+\.\d+", text):
            return False

        return True

    # 1. Try Block 9 header-based ownership column
    header_cell = None

    for cell in cells_with_text:
        text = clean_text(cell.get("text", ""))

        if (
            "ಕಬ್ಜೆ" in text
            or "ಸ್ವಾಧೀನದಾರ" in text
            or "ಸ್ವಾಧೀನ" in text
        ):
            header_cell = cell
            break

    candidate_cells = []

    if header_cell:
        hx = header_cell.get("x", 0)
        hx2 = header_cell.get(
            "x2",
            hx + header_cell.get("width", 0)
        )
        hy = header_cell.get("y", 0)

        for cell in cells_with_text:
            x = cell.get("x", 0)
            y = cell.get("y", 0)

            if (
                y > hy
                and y < 1250
                and x >= hx - 200
                and x <= hx2 + 300
            ):
                candidate_cells.append(cell)

    # 2. Try candidate cells
    for cell in candidate_cells:
        text = clean_text(cell.get("text", ""))
        text = text.replace("|", " ")
        text = clean_text(text)

        if not is_valid_owner_text(text):
            continue

        if text in seen:
            continue

        seen.add(text)

        owners.append({
            "owner_name": text,
            "father_or_spouse_name": None,
            "address": None,
            "extent": None,
            "khata_no": None,
            "possession_type": None,
            "raw_text": cell.get("text", "")
        })

    if owners:
        return owners

    # 3. Fallback: use cultivation/cultivator first column
    # This helps Government RTCs where owner and cultivator are same.
    # Take the FULLEST candidate rather than the first one down the page. The
    # names live in one tall cell spanning the cultivator column, and several
    # thinner cells — a stray mark, a clipped heading, an OCR smear — sit above
    # it in the same column. First-match-wins reported ". ಬಓಟ್ನ್ಟೀಣ್ಣ | af"
    # (18 characters of noise) over the 181-character cell holding
    # "ಸರ್ಕಾರಿ ಖರಾಬು ಗೋಮಾಳ ರಾಜಣ್ಣ ಬಿನ್ ಆಂಜನಪ್ಪ ...". Kannada characters, not
    # raw length, decide it, so a long Latin smear cannot win.
    best_cell = None
    best_text = ""
    best_score = 0

    for cell in cells_with_text:
        row = cell.get("row", -1)
        col = cell.get("column", -1)

        text = clean_text(cell.get("text", ""))

        if not text:
            continue

        if row >= 10 and col == 0:
            text = normalize_repeated_words(text)

            if not is_valid_owner_text(text):
                continue

            score = sum(1 for c in text if "\u0C80" <= c <= "\u0CFF")

            if score > best_score:
                best_cell, best_text, best_score = cell, text, score

    if best_cell:
        return [{
            "owner_name": best_text,
            "father_or_spouse_name": None,
            "address": None,
            "extent": None,
            "khata_no": None,
            "possession_type": None,
            "raw_text": best_cell.get("text", "")
        }]

    # 4. Final fallback
    return fallback.get(
        "9_ownership_details",
        []
    ) if fallback else []

def extract_possession_details(
    owners: List[Dict[str, Any]],
    text: str
) -> Dict[str, Any]:
    print("POSSESSION TEXT:")
    print(text[:5000])
    possession_types = []

    if "ಪೌತಿ" in text:
        possession_types.append("ಪೌತಿ")

    if "ಮೇಲಿನ ಜಂಟಿ" in text or "ಜಂಟಿ" in text:
        possession_types.append("ಮೇಲಿನ ಜಂಟಿ")

    possession_types = list(dict.fromkeys(possession_types))

    return {
        "possession_type": possession_types or None,
        "owner_wise_possession": [
            {
                "owner_name": owner.get("owner_name"),
                "possession_type": owner.get("possession_type") or (
                    possession_types[0] if possession_types else None
                )
            }
            for owner in owners
        ]
    }



def extract_rights_liabilities(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:
    print("INSIDE RIGHTS EXTRACTOR")
    old = fallback.get("11_rights_liabilities", {}) if fallback else {}

    rights_text = kannada_digits_to_english(text or "")

    for cell in cells_with_text:
        if not isinstance(cell, dict):
            continue

        cell_text = kannada_digits_to_english(
            clean_text(cell.get("text", ""))
        )

        if cell_text:
            rights_text += "\n" + cell_text

    mr_number = (
        find_mr_number(rights_text)
        or old.get("mr_number")
    )

    court_stay = (
        "court stay" in rights_text.lower()
        or "courtstay" in rights_text.lower()
        or "ಕೋರ್ಟ" in rights_text
        or "ಕೋರ್ಟ್" in rights_text
        or "ಆದೇಶ" in rights_text
        or "wp" in rights_text.lower()
        or bool(old.get("court_stay"))
    )

    court_reference = None

    wp_match = re.search(
    r"(?:nowp|no\s*wp|now\s*p|wp)\s*([0-9\/\-]+)",
    rights_text,
    re.IGNORECASE
    )

    if wp_match:
        court_reference = "WP" + wp_match.group(1)

    court_reference = (
        court_reference
        or find_first(r"(RRT\/[A-Z0-9\/.\-]+)", rights_text)
        or find_first(r"(NA\.CR\.[0-9\/\-]+)", rights_text)
        or old.get("court_reference")
    )

    recorded_on = None

    dated_match = re.search(
        r"dated\s*([A-Za-z]{3,9}\s+\d{1,2}\s+\d{4})",
        rights_text,
        re.IGNORECASE
    )

    if dated_match:
        recorded_on = dated_match.group(1)

    if not recorded_on:
        date_matches = re.findall(
            r"\d{2}\/\d{2}\/\d{4}",
            rights_text
        )

        if date_matches:
            recorded_on = date_matches[0]

    recorded_on = recorded_on or old.get("recorded_on")

    remarks = old.get("remarks")

    if "court stay as per" in rights_text.lower() or "courtstayas per" in rights_text.lower():
        remarks = "court stay as per order"
    elif "court stay" in rights_text.lower() or "courtstay" in rights_text.lower():
        remarks = "court stay"

    if remarks and dated_match:
        remarks = f"{remarks} dated {dated_match.group(1)}"

    print("\n========== RIGHTS OUTPUT ==========")
    print({
        "mr_number": mr_number,
        "court_stay": court_stay,
        "court_reference": court_reference,
        "recorded_on": recorded_on,
        "remarks": remarks
    })
    print("===================================\n")


    return {
        "mr_number": mr_number,
        "court_stay": court_stay,
        "court_reference": court_reference,
        "recorded_on": recorded_on,
        "rights": old.get("rights"),
        "liabilities": old.get("liabilities"),
        "remarks": remarks,
    }
import re
from typing import List, Dict, Any

print("######## SERVICES CULTIVATION FUNCTION ########")
def get_cultivation_extent_values(
    cells_with_text: List[Dict[str, Any]]
) -> List[str]:

    extent_values = []

    for cell in cells_with_text:
        row = cell.get("row", -1)
        col = cell.get("column", -1)

        text = clean_text(
            cell.get("text", "")
        )

        if row >= 10 and col in [7, 8, 10]:

            values = re.findall(
                r"\d+\.\d+\.\d+\.\d+|\d+\.\d+",
                kannada_digits_to_english(text)
            )

            extent_values.extend(values)

    print("EXTENT VALUES:", extent_values)

    return extent_values

def normalize_repeated_words(text: str) -> str:
    text = clean_text(text)
    text = text.replace("|", " ")

    words = text.split()
    result = []

    for word in words:
        if word not in result:
            result.append(word)

    return " ".join(result)

print("SERVICES RTC MAPPER RUNNING")

def extract_cultivation_details(
    cells_with_text: List[Dict[str, Any]],
    text: str,
    fallback_extraction: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:

    records = []


    normalized_text = kannada_digits_to_english(text)
    print("HAS BESIGE:", "ಬೇಸಿಗೆ" in normalized_text)
    print(normalized_text[-2000:])
    # -------------------------------
    # 1. Crop Year
    # -------------------------------

    years = re.findall(
        r"20\d{2}\s*[-–—]\s*20\d{2}",
        normalized_text
    )

    cleaned_years = []

    for year in years:

        year = re.sub(r"\s+", "", year)
        year = year.replace("–", "-").replace("—", "-")

        # Ignore MR years like 2013-2014 when newer crop year exists
        if year not in cleaned_years:
            cleaned_years.append(year)

    # Keep latest year only
    if cleaned_years:
        cleaned_years = [sorted(cleaned_years)[-1]]

    if not cleaned_years and fallback_extraction:

        fallback_years = []

        for item in fallback_extraction.get(
            "12_cultivation_details",
            []
        ):
            year = item.get("crop_year")

            if year:
                fallback_years.append(year)

        if fallback_years:
            cleaned_years = [
                sorted(fallback_years)[-1]
            ]

    print("CROP YEARS:", cleaned_years)

    # -------------------------------
    # 2. Season Detection
    # -------------------------------

    seasons = []

    season_text = normalized_text

    for cell in cells_with_text:
        if not isinstance(cell, dict):
            continue

        season_text += "\n" + clean_text(
            cell.get("text", "")
        )

    season_text = kannada_digits_to_english(
        season_text
    )

    season_text_compact = re.sub(
        r"\s+",
        "",
        season_text
    )

    print("HAS BESIGE:", "ಬೇಸಿಗೆ" in season_text)
    print("HAS BESIGA:", "ಬೇಸಿಗ" in season_text_compact)

    season_patterns = {
        "ಮುಂಗಾರು": [
            "ಮುಂಗಾರು",
            "ಮುಂಗಾರ"
        ],
        "ಹಿಂಗಾರು": [
            "ಹಿಂಗಾರು",
            "ಹಿಂಗಾರ"
        ],
        "ಬೇಸಿಗೆ": [
            "ಬೇಸಿಗೆ",
            "ಬೇಸಿಗ",
            "ಬೆಸಿಗೆ",
            "ಬೆಸಿಗ"
        ]
    }

    for season_name, patterns in season_patterns.items():

        for pattern in patterns:

            if pattern in season_text_compact:
                seasons.append(season_name)
                break

    seasons = list(dict.fromkeys(seasons))

    # ------------------------------------------------
    # Fallback from raw OCR rows
    # ------------------------------------------------

    if not seasons:

        if re.search(
            r"202\d-\d+\s+.*?ಮುಂಗಾರ",
            normalized_text,
            re.DOTALL
        ):
            seasons.append("ಮುಂಗಾರು")

        if re.search(
            r"202\d-\d+\s+.*?ಹಿಂಗಾರ",
            normalized_text,
            re.DOTALL
        ):
            seasons.append("ಹಿಂಗಾರು")

        if re.search(
            r"202\d-\d+\s+.*?ಬೇಸಿಗ",
            normalized_text,
            re.DOTALL
        ):
            seasons.append("ಬೇಸಿಗೆ")

    seasons = list(dict.fromkeys(seasons))

    if not seasons:
        seasons = [None]

    print("SEASONS FOUND:", seasons)

    # -------------------------------
    # 3. Cultivator Extraction
    # -------------------------------

    ignore_words = [
        "ವಿಸ್ತೀರ್ಣ",
        "ವಿನ್ನೀರ್ಣ",
        "ಗುತ್ತಿಗೆ",
        "ವರ್ಗ",
        "ಅಮಿಶ್ರ",
        "ಮಿಶ್ರ",
        "ಒಟ್ಟು",
        "ಹೆಸರು",
        "ಸಂಖ್ಯೆ",
        "ನೀರಾವರಿ",
        "ಬೆಳೆಯ",
        "ಮೂಲ",
        "ಉತ್ಪತ್ತಿ",
        "ವರ್ಷ",
        "ಕಾಲ",
        "ಎ ಗುಂ",
    ]

    candidates = []

    for cell in cells_with_text:

        row = cell.get("row", -1)
        col = cell.get("column", -1)
        y = cell.get("y", 0)

        cell_text = clean_text(
            cell.get("text", "")
        )

        if not cell_text:
            continue

        if y < 1200:
            continue

        if col not in [0, 1]:
            continue

        if len(cell_text) < 5:
            continue

        if any(
            word in cell_text
            for word in ignore_words
        ):
            continue

        if re.fullmatch(
            r"[\d\.\s/|()-]+",
            cell_text
        ):
            continue

        if re.search(
            r"\d+\.\d+\.\d+",
            cell_text
        ):
            continue

        candidates.append({
            "row": row,
            "col": col,
            "text": cell_text,
            "score": len(cell_text)
        })

    candidates.sort(
        key=lambda item: (
            item["row"],
            item["col"],
            -item["score"]
        )
    )

    cultivator_name = (
        candidates[0]["text"]
        if candidates
        else None
    )

    cultivator_name = (
        normalize_repeated_words(
            cultivator_name
        )
        if cultivator_name
        else None
    )

    print("CULTIVATOR:", cultivator_name)

    # -------------------------------
    # 4. Extents
    # -------------------------------

    extent_values = get_cultivation_extent_values(
        cells_with_text
    )

    crop_extent = (
        extent_values[0]
        if len(extent_values) > 0
        else None
    )

    irrigated_extent = (
        extent_values[1]
        if len(extent_values) > 1
        else None
    )

    # -------------------------------
    # 5. Build Records
    # -------------------------------

    for year in cleaned_years:

        for season in seasons:

            records.append({
                "crop_year": year,
                "season": season,
                "cultivator_name": cultivator_name,
                "cultivator_raw_text": cultivator_name,
                "cultivation_method": None,
                "land_use": None,
                "crop_name": None,
                "crop_extent": crop_extent,
                "irrigated_extent": irrigated_extent,
                "extraction_status": "partial"
            })

    print("CULTIVATION RECORDS:", records)

    return records

def map_rtc_cells_to_blocks(
    cells_with_text: List[Dict[str, Any]],
    table_layout: Optional[Dict[str, Any]] = None,
    fallback_extraction: Optional[Dict[str, Any]] = None,
    raw_text: Optional[str] = None,
    header_strip_text: Optional[str] = None
) -> Dict[str, Any]:
    print("INSIDE CELL MAPPER")
    fallback_extraction = fallback_extraction or {}

    text = all_text(cells_with_text)

    header = extract_header_from_cells(
        cells_with_text,
        fallback_extraction,
        raw_text,
        header_strip_text
    )

    survey_number = extract_survey_number(
        cells_with_text,
        text,
        fallback_extraction
    )

    hissa = extract_hissa(
        cells_with_text,
        text,
        fallback_extraction
    )

    khata_vara = extract_khata_vara(
        text,
        fallback_extraction
    )

    land_revenue = extract_land_revenue(
        cells_with_text,
        text,
        fallback_extraction
    )

    owners = extract_owner_details(
        cells_with_text,
        fallback_extraction
    )

    possession_details = extract_possession_details(
        owners,
        text
    )

    cultivation_details = extract_cultivation_details(
        cells_with_text,
        text,
        fallback_extraction
    )

    tree_details = extract_tree_details(
        cells_with_text,
        text
    )


    cultivation_source_text = raw_text if raw_text else text
    rights_source_text = raw_text if raw_text else text
    possession_source_text = raw_text if raw_text else text

    rights_liabilities = extract_rights_liabilities(
        cells_with_text,
        rights_source_text,
        fallback_extraction
    )

    cultivation_details = extract_cultivation_details(
        cells_with_text,
        cultivation_source_text,
        fallback_extraction
    )

    print("RAW TEXT HAS BESIGE:", bool(raw_text and "ಬೇಸಿಗೆ" in raw_text))
    print("CULTIVATION DETAILS BEFORE FORCE:", cultivation_details)


    # Force add Besige if present in full OCR but missed by cell OCR
    if "ಬೇಸಿಗೆ" in cultivation_source_text or "ಬೇಸಿಗ" in cultivation_source_text:

        has_besige = any(
            item.get("season") == "ಬೇಸಿಗೆ"
            for item in cultivation_details
        )

        if not has_besige and cultivation_details:
            base = cultivation_details[-1].copy()
            base["season"] = "ಬೇಸಿಗೆ"
            cultivation_details.append(base)

    return {
        "document_type": "RTC",

        "header": header,

        "1_survey_number": {
            "survey_number": survey_number
        },

        "2_hissa": {
            "hissa_number": hissa
        },

        "3_khata_vara": khata_vara,

        "4_land_revenue": land_revenue,

        "5_soil_type": extract_soil_type(
            text
        ),

        "6_land_classification": extract_land_classification(
            text,
            raw_text
        ),

        "7_tree_details": tree_details,

        "8_irrigation_details": extract_irrigation_details(
            text
        ),

        "9_ownership_details": owners,

        "10_possession_details": extract_possession_details(
            owners,
            possession_source_text
        ),

        "11_rights_liabilities": rights_liabilities,

        "12_cultivation_details": cultivation_details,

        "13_other_rights": {
            "other_rights": None,
            "restrictions": (
                "court stay"
                if "court stay" in rights_source_text.lower()
                else None
            ),
            "remarks": None
        },

        "extraction_source": "cell_ocr_primary_layout_mapper_v3"
    }


def map_rtc_cells(
    cells_with_text,
    table_layout=None,
    fallback_extraction=None,
    raw_text=None,
    header_strip_text=None
):
    return map_rtc_cells_to_blocks(
        cells_with_text=cells_with_text,
        table_layout=table_layout,
        fallback_extraction=fallback_extraction,
        raw_text=raw_text,
        header_strip_text=header_strip_text
    )