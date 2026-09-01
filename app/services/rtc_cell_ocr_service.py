import cv2
import fitz
import numpy as np
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

    return np.array(image)


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