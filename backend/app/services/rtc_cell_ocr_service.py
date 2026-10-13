import cv2
import fitz
import numpy as np

from app.services.ocr_service import auto_rotate_for_ocr, TESSERACT_LANG
import pytesseract

from PIL import Image
from typing import Dict, Any, List


def pdf_page_to_image(
    file_path: str,
    page_number: int = 0,
    dpi: int = 300
):
    pdf = fitz.open(file_path)

    page = pdf[page_number]

    pix = page.get_pixmap(dpi=dpi)

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    pdf.close()

    # Same orientation correction as the page-level OCR. Both renderers must
    # apply it: the grid boxes are measured on one image and the crops taken
    # from the other, so a rotation applied to only one would misalign them.
    return auto_rotate_for_ocr(np.array(image))


def clean_cell_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text.strip()


def preprocess_crop(crop):
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return thresh


def run_tesseract(image):
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 11"
    ]

    results = []

    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(
                image,
                lang="kan+eng",
                config=cfg
            )

            txt = clean_cell_text(txt)

            if txt:
                results.append(txt)

        except Exception:
            pass

    if not results:
        return ""

    results.sort(key=len, reverse=True)

    return results[0]


def ocr_cell(
    image_np,
    cell: Dict[str, Any]
):
    x = max(0, cell["x"] + 2)
    y = max(0, cell["y"] + 2)

    x2 = cell["x2"] - 2
    y2 = cell["y2"] - 2

    crop = image_np[y:y2, x:x2]

    if crop.size == 0:
        return ""

    processed = preprocess_crop(crop)

    return run_tesseract(processed)


def assign_row_col_numbers(
    cells: List[Dict[str, Any]],
    tolerance: int = 20
):
    rows = []

    for cell in sorted(
        cells,
        key=lambda c: (c["y"], c["x"])
    ):
        placed = False

        for row in rows:
            if abs(
                row[0]["y"] - cell["y"]
            ) <= tolerance:
                row.append(cell)
                placed = True
                break

        if not placed:
            rows.append([cell])

    rows.sort(key=lambda r: r[0]["y"])

    final_cells = []

    for row_idx, row in enumerate(rows):

        row.sort(key=lambda c: c["x"])

        for col_idx, cell in enumerate(row):

            cell["row"] = row_idx

            cell["column"] = col_idx

            final_cells.append(cell)

    return final_cells


def extract_cells_with_text(
    file_path: str,
    table_layout: Dict[str, Any]
) -> List[Dict[str, Any]]:

    image_np = pdf_page_to_image(file_path)

    cells_with_text = []

    for cell in table_layout.get("cells", []):

        text = ocr_cell(
            image_np,
            cell
        )

        cells_with_text.append({
            **cell,
            "text": text
        })

    cells_with_text = assign_row_col_numbers(
        cells_with_text
    )

    return cells_with_text

# The taluk / hobli / village line is printed ABOVE the table's top border, so
# it is in no cell — the grid detector never sees it and the header can only
# come from page-level OCR. On a watermarked scan that whole-page pass is
# dominated by the table and destroys the header labels: the Yalahanka RTC
# returned null for taluk, hobli AND village, while the same strip cropped out
# and read on its own gives "ಗ್ರಾಮ: ಕುದುರೆಗೆರೆ" correctly.
#
# The crop is read at several heights because Tesseract's layout analysis is
# sharply sensitive to where the band is cut: 130..290 recovered three labels
# on that page and 126..286 — four pixels tighter — recovered none at all. Each
# candidate is scored by how many header labels survive, exactly as the Aadhaar
# crop pipeline scores its profiles, and the best reading wins.
_HEADER_STRIP_OFFSETS = (150, 160, 170, 190, 210)
_HEADER_STRIP_LABELS = ("ತಾಲ್ಲೂ", "ತಾಲೂಕು", "ಹೋಬಳಿ", "ಹೋಬಲಿ", "ಗ್ರಾಮ")


def ocr_header_strip(file_path: str, cells_with_text) -> str:
    try:
        image_np = pdf_page_to_image(file_path)

        table_top = min(
            (cell["y"] for cell in cells_with_text if cell.get("y", 0) > 250),
            default=0
        )

        if not table_top:
            return ""

        best_text = ""
        best_score = 0

        for offset in _HEADER_STRIP_OFFSETS:
            top = max(0, table_top - offset)
            strip = image_np[top:table_top + 4, :]

            if strip.size == 0:
                continue

            gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
            upscaled = cv2.resize(
                gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )

            text = pytesseract.image_to_string(
                upscaled,
                lang=TESSERACT_LANG,
                config="--psm 6"
            )

            score = sum(1 for label in _HEADER_STRIP_LABELS if label in text)

            if score > best_score:
                best_text, best_score = text, score

        return best_text

    except Exception as error:
        print("HEADER STRIP OCR FAILED:", error)
        return ""
